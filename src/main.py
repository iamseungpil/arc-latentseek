"""
Main pipeline for ARC-LatentSeek integration
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import time
from datetime import datetime
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data import ARCDataLoader, ARCProblem
from .generators import BARCGenerator, BARCOutput
from .executors import CodeExecutor, ExecutionResult, GridRenderer
from .evaluators import GLMEvaluator, EvaluationResult, RewardModel
from .optimizers import LatentSeekOptimizer, OptimizationResult
from .alignment import BARCCodeAligner, AlignmentQualityAnalyzer


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    # Model paths
    barc_model: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B"
    glm_model: str = "THUDM/GLM-4.1V-9B-Thinking"
    
    # Generation settings
    num_candidates: int = 8
    temperature: float = 0.8
    max_new_tokens: int = 2048
    
    # Execution settings
    execution_timeout: int = 2
    
    # Optimization settings
    optimization_steps: int = 10
    optimization_threshold: float = -0.2
    use_description_based_optimization: bool = True
    
    # Alignment settings
    enable_code_alignment: bool = True
    alignment_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    alignment_temperature: float = 0.3
    alignment_max_tokens: int = 2048
    min_alignment_score: int = 20
    
    # Output settings
    output_dir: str = "results"
    save_visualizations: bool = True


@dataclass
class SolutionResult:
    """Result of solving a single ARC problem"""
    problem_id: str
    success: bool
    best_code: str
    best_description: str
    best_reward: float
    execution_accuracy: float
    evaluation_details: Dict
    visualization_path: Optional[str]
    time_taken: float
    
    # Before/After improvement tracking
    initial_success: bool = False  # Did any initial candidate succeed?
    improved_by_latentseek: bool = False  # Did LatentSeek find a solution?
    initial_best_accuracy: float = 0.0  # Best accuracy before optimization
    final_best_accuracy: float = 0.0   # Best accuracy after optimization
    
    def to_dict(self):
        return asdict(self)


class ARCLatentSeekPipeline:
    """Main pipeline for solving ARC problems"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize WandB
        self._init_wandb()
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        self.data_loader = ARCDataLoader()
        self.barc_generator = BARCGenerator(config.barc_model)
        self.code_executor = CodeExecutor(config.execution_timeout)
        self.glm_evaluator = GLMEvaluator(config.glm_model)
        self.reward_model = RewardModel()
        # Renderer will be initialized after creating date-stamped directory
        
        # Initialize alignment components
        if config.enable_code_alignment:
            logger.info("Initializing code alignment components...")
            self.code_aligner = BARCCodeAligner(
                model_path=config.alignment_model,
                temperature=config.alignment_temperature,
                max_new_tokens=config.alignment_max_tokens
            )
            self.quality_analyzer = AlignmentQualityAnalyzer()
        else:
            self.code_aligner = None
            self.quality_analyzer = None
        
        # Initialize LatentSeek optimizer
        self.latentseek_optimizer = LatentSeekOptimizer(
            barc_generator=self.barc_generator,
            code_executor=self.code_executor,
            glm_evaluator=self.glm_evaluator,
            max_steps=config.optimization_steps,
            reward_threshold=config.optimization_threshold
        )
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Create date-stamped visualization directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.visualizations_dir = os.path.join(config.output_dir, f"visualizations_{timestamp}")
        os.makedirs(self.visualizations_dir, exist_ok=True)
        
        # Update renderer with new visualization directory
        self.renderer = GridRenderer(self.visualizations_dir)
        
        logger.info("Pipeline initialized successfully")
    
    def _align_single_candidate(self, candidate_info):
        """Align a single candidate (for parallel processing)"""
        candidate, problem, candidate_idx = candidate_info
        
        if not self.config.enable_code_alignment or not self.code_aligner:
            return candidate_idx, candidate, None
            
        try:
            logger.info(f"Applying code alignment to candidate {candidate_idx+1}")
            
            # Align code
            aligned_candidate = self.code_aligner.align_code(candidate, problem)
            
            # Analyze alignment quality
            quality = None
            if self.quality_analyzer:
                quality = self.quality_analyzer.analyze_alignment_quality(
                    original_code=candidate.code,
                    aligned_code=aligned_candidate.code,
                    original_description=candidate.description,
                    aligned_description=aligned_candidate.description
                )
                
                # Use aligned candidate if quality is good enough
                if quality.improvement_score >= self.config.min_alignment_score:
                    logger.info(f"Using aligned candidate {candidate_idx+1} (score: {quality.improvement_score:.1f})")
                    return candidate_idx, aligned_candidate, quality
                else:
                    logger.warning(f"Alignment quality too low for candidate {candidate_idx+1} (score: {quality.improvement_score:.1f}), using original")
                    return candidate_idx, candidate, quality
            else:
                return candidate_idx, aligned_candidate, quality
                
        except Exception as e:
            logger.error(f"Alignment failed for candidate {candidate_idx+1}: {e}")
            return candidate_idx, candidate, None
    
    def _init_wandb(self):
        """Initialize WandB logging"""
        try:
            # Use same token as barc_post
            wandb.login(key="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
            
            wandb.init(
                project="arc-latentseek",
                config={
                    "barc_model": self.config.barc_model,
                    "glm_model": self.config.glm_model,
                    "num_candidates": self.config.num_candidates,
                    "optimization_steps": self.config.optimization_steps,
                    "optimization_threshold": self.config.optimization_threshold,
                    "execution_timeout": self.config.execution_timeout
                },
                tags=["arc", "latentseek", "barc", "glm"]
            )
            logger.info("WandB initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize WandB: {e}")
            wandb.init(mode="disabled")
        
    def solve_problem(self, problem: ARCProblem) -> SolutionResult:
        """
        Solve a single ARC problem
        
        Args:
            problem: ARC problem to solve
            
        Returns:
            SolutionResult with best solution found
        """
        start_time = time.time()
        logger.info(f"Solving problem {problem.uid}")
        
        best_solution = None
        best_reward = float('-inf')
        best_output = None
        best_execution = None
        candidate_details = []  # Track all candidates for visualization
        
        # Track before/after improvement
        initial_best_accuracy = 0.0
        initial_success = False
        candidates_tried_count = 0
        
        # Generate initial candidates
        logger.info(f"Generating {self.config.num_candidates} candidate solutions...")
        barc_start_time = time.time()
        candidates = self.barc_generator.generate(
            problem,
            temperature=self.config.temperature,
            max_new_tokens=self.config.max_new_tokens,
            num_candidates=self.config.num_candidates
        )
        barc_time = time.time() - barc_start_time
        logger.info(f"BARC generation completed in {barc_time:.2f}s")
        wandb.log({"barc_generation_time": barc_time})
        
        # Apply code alignment to candidates (sequential for memory safety)
        aligned_candidates = candidates.copy()
        if self.config.enable_code_alignment and self.code_aligner:
            logger.info(f"Applying code alignment to {len(candidates)} candidates...")
            alignment_start_time = time.time()
            
            for i, candidate in enumerate(candidates):
                try:
                    logger.info(f"Applying code alignment to candidate {i+1}")
                    
                    # Align code
                    aligned_candidate = self.code_aligner.align_code(candidate, problem)
                    
                    # Analyze alignment quality
                    if self.quality_analyzer:
                        quality = self.quality_analyzer.analyze_alignment_quality(
                            original_code=candidate.code,
                            aligned_code=aligned_candidate.code,
                            original_description=candidate.description,
                            aligned_description=aligned_candidate.description
                        )
                        
                        # Log alignment details
                        self._log_alignment_details(problem.uid, i+1, candidate, aligned_candidate, quality)
                        
                        # Use aligned candidate if quality is good enough
                        if quality.improvement_score >= self.config.min_alignment_score:
                            logger.info(f"Using aligned candidate {i+1} (score: {quality.improvement_score:.1f})")
                            aligned_candidates[i] = aligned_candidate
                        else:
                            logger.warning(f"Alignment quality too low for candidate {i+1} (score: {quality.improvement_score:.1f}), using original")
                    else:
                        aligned_candidates[i] = aligned_candidate
                        
                except Exception as e:
                    logger.error(f"Alignment failed for candidate {i+1}: {e}")
                    # Keep original candidate if alignment fails
            
            alignment_time = time.time() - alignment_start_time
            logger.info(f"Sequential alignment completed in {alignment_time:.2f}s")
            wandb.log({"alignment_time": alignment_time})
        
        # Evaluate each aligned candidate
        for i, candidate in enumerate(aligned_candidates):
            logger.info(f"Evaluating candidate {i+1}/{len(aligned_candidates)}")
            
            # Execute code
            execution_result = self.code_executor.execute(candidate.code, problem)
            
            if not execution_result.success:
                logger.warning(f"Candidate {i+1} execution failed: {execution_result.error_messages}")
                continue
                
            # Evaluate with GLM
            glm_start_time = time.time()
            evaluation_result = self.glm_evaluator.evaluate(
                problem,
                candidate,
                execution_result,
                f"temp_eval_{problem.uid}_candidate_{i}"
            )
            glm_time = time.time() - glm_start_time
            logger.info(f"GLM evaluation for candidate {i+1} completed in {glm_time:.2f}s")
            wandb.log({f"glm_evaluation_time_candidate_{i+1}": glm_time})
            
            logger.info(f"Candidate {i+1} - Accuracy: {execution_result.accuracy:.2%}, "
                       f"Reward: {evaluation_result.total_reward:.3f}")
            
            # Track initial performance (before optimization)
            candidates_tried_count += 1
            if execution_result.accuracy > initial_best_accuracy:
                initial_best_accuracy = execution_result.accuracy
            if evaluation_result.total_reward >= 1.0:  # Perfect solution
                initial_success = True
            
            # Log detailed information
            self._log_candidate_details(problem.uid, i+1, candidate, execution_result, evaluation_result, "initial")
            
            # Apply LatentSeek optimization if not all training pairs are correct
            if execution_result.accuracy < 1.0:
                if self.config.use_description_based_optimization:
                    logger.info(f"Applying description-based LatentSeek optimization to candidate {i+1}")
                    optimization_result = self.latentseek_optimizer.optimize_description_based(
                        problem,
                        candidate,
                        evaluation_result.total_reward
                    )
                else:
                    logger.info(f"Applying standard LatentSeek optimization to candidate {i+1}")
                    optimization_result = self.latentseek_optimizer.optimize(
                        problem,
                        candidate,
                        evaluation_result.total_reward
                    )
                
                if optimization_result.converged:
                    logger.info(f"LatentSeek converged in {optimization_result.optimization_steps} steps")
                    
                    # Use optimized output
                    optimized_candidate = optimization_result.final_output
                    optimized_execution = self.code_executor.execute(optimized_candidate.code, problem)
                    
                    if optimized_execution.success:
                        optimized_evaluation = self.glm_evaluator.evaluate(
                            problem,
                            optimized_candidate,
                            optimized_execution,
                            f"temp_eval_{problem.uid}_optimized_{i}"
                        )
                        
                        logger.info(f"Optimized candidate {i+1} - Accuracy: {optimized_execution.accuracy:.2%}, "
                                   f"Reward: {optimized_evaluation.total_reward:.3f}")
                        
                        # Log optimized details
                        self._log_candidate_details(problem.uid, i+1, optimized_candidate, 
                                                  optimized_execution, optimized_evaluation, "optimized")
                        
                        # Log optimization history
                        self._log_optimization_history(problem.uid, i+1, optimization_result)
                        
                        # Use optimized version if better
                        if optimized_evaluation.total_reward > evaluation_result.total_reward:
                            candidate = optimized_candidate
                            execution_result = optimized_execution
                            evaluation_result = optimized_evaluation
                            logger.info(f"Using optimized version for candidate {i+1}")
            
            # Track candidate for visualization
            candidate_details.append((candidate, execution_result))
            
            # Track best solution
            if evaluation_result.total_reward > best_reward:
                best_reward = evaluation_result.total_reward
                best_solution = candidate
                best_output = evaluation_result
                best_execution = execution_result
                
            # Check if we should stop
            if self.reward_model.should_stop_optimization(evaluation_result.total_reward):
                logger.info(f"Found good solution with reward {evaluation_result.total_reward:.3f}, stopping early")
                break
        
        # Prepare final result
        if best_solution is None:
            logger.error(f"No valid solution found for problem {problem.uid}")
            return SolutionResult(
                problem_id=problem.uid,
                success=False,
                best_code="",
                best_description="",
                best_reward=float('-inf'),
                execution_accuracy=0.0,
                evaluation_details={},
                visualization_path=None,
                time_taken=time.time() - start_time
            )
        
        # Save best visualization
        final_viz_path = None
        if self.config.save_visualizations:
            final_viz_path = os.path.join(
                self.visualizations_dir,
                f"{problem.uid}_best.png"
            )
            try:
                self.renderer.render_problem_with_output(
                    problem,
                    best_execution.output_grids,
                    final_viz_path
                )
                logger.info(f"Saved visualization to {final_viz_path}")
            except Exception as e:
                logger.error(f"Failed to save visualization: {e}")
                
        # Also save visualization for each candidate that was evaluated
        if self.config.save_visualizations and candidate_details:
            for i, (candidate, exec_result) in enumerate(candidate_details):
                if exec_result and exec_result.output_grids:
                    try:
                        candidate_viz_path = os.path.join(
                            self.visualizations_dir,
                            f"{problem.uid}_candidate_{i+1}.png"
                        )
                        self.renderer.render_problem_with_output(
                            problem,
                            exec_result.output_grids,
                            candidate_viz_path
                        )
                    except Exception as e:
                        logger.debug(f"Failed to save candidate {i+1} visualization: {e}")
        
        # Calculate final tracking metrics
        final_best_accuracy = best_execution.accuracy
        improved_by_latentseek = (final_best_accuracy > initial_best_accuracy) and not initial_success
        
        return SolutionResult(
            problem_id=problem.uid,
            success=best_execution.accuracy >= 1.0,  # Binary success
            best_code=best_solution.code,
            best_description=best_solution.description or "",
            best_reward=best_reward,
            execution_accuracy=best_execution.accuracy,
            evaluation_details={
                'component_scores': best_output.component_scores,
                'verifications': {k: v.passed for k, v in best_output.verifications.items()},
                'feedback': best_output.detailed_feedback
            },
            visualization_path=final_viz_path,
            time_taken=time.time() - start_time,
            # Before/after improvement tracking
            initial_success=initial_success,
            improved_by_latentseek=improved_by_latentseek,
            initial_best_accuracy=initial_best_accuracy,
            final_best_accuracy=final_best_accuracy
        )
    
    def solve_problems(self, 
                      split: str = "validation",
                      num_problems: Optional[int] = None,
                      problem_ids: Optional[List[str]] = None) -> List[SolutionResult]:
        """
        Solve multiple ARC problems
        
        Args:
            split: Dataset split ("train", "validation", or "all")
            num_problems: Number of problems to solve
            problem_ids: Specific problem IDs to solve
            
        Returns:
            List of SolutionResult objects
        """
        # Load problems
        problems = self.data_loader.get_problems(split, num_problems, problem_ids)
        logger.info(f"Loaded {len(problems)} problems from {split} split")
        
        results = []
        
        for i, problem in enumerate(problems):
            logger.info(f"\n{'='*50}")
            logger.info(f"Problem {i+1}/{len(problems)}: {problem.uid}")
            logger.info(f"{'='*50}")
            
            try:
                result = self.solve_problem(problem)
                results.append(result)
                
                # Log summary
                if result.success:
                    logger.info(f"✅ Successfully solved {problem.uid} "
                              f"(accuracy: {result.execution_accuracy:.2%}, "
                              f"reward: {result.best_reward:.3f})")
                else:
                    logger.info(f"❌ Failed to solve {problem.uid}")
                    
            except Exception as e:
                logger.error(f"Error solving problem {problem.uid}: {e}")
                results.append(SolutionResult(
                    problem_id=problem.uid,
                    success=False,
                    best_code="",
                    best_description="",
                    best_reward=float('-inf'),
                    execution_accuracy=0.0,
                    evaluation_details={'error': str(e)},
                    visualization_path=None,
                    time_taken=0.0
                ))
        
        # Save results
        self._save_results(results)
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        logger.info(f"\n{'='*50}")
        logger.info(f"SUMMARY: Solved {successful}/{len(results)} problems "
                   f"({successful/len(results)*100:.1f}%)")
        logger.info(f"{'='*50}")
        
        return results
    
    def _save_results(self, results: List[SolutionResult]):
        """Save results to JSON file with timestamp"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = os.path.join(self.config.output_dir, f"results_{timestamp}.json")
        
        with open(output_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
            
        logger.info(f"Results saved to {output_path}")
        
        # Also save summary with improvement statistics
        total_problems = len(results)
        successful = sum(1 for r in results if r.success)
        initial_successes = sum(1 for r in results if r.initial_success)
        improved_by_latentseek = sum(1 for r in results if r.improved_by_latentseek)
        
        summary = {
            'timestamp': timestamp,
            'total_problems': total_problems,
            'successful': successful,
            'success_rate': successful / total_problems if total_problems > 0 else 0,
            'initial_successes': initial_successes,
            'initial_success_rate': initial_successes / total_problems if total_problems > 0 else 0,
            'improved_by_latentseek': improved_by_latentseek,
            'latentseek_improvement_rate': improved_by_latentseek / total_problems if total_problems > 0 else 0,
            'average_accuracy': sum(r.execution_accuracy for r in results) / len(results) if results else 0,
            'average_initial_accuracy': sum(r.initial_best_accuracy for r in results) / len(results) if results else 0,
            'average_final_accuracy': sum(r.final_best_accuracy for r in results) / len(results) if results else 0,
            'average_reward': sum(r.best_reward for r in results) / len(results) if results else 0,
            'average_time': sum(r.time_taken for r in results) / len(results) if results else 0
        }
        
        summary_path = os.path.join(self.config.output_dir, f"summary_{timestamp}.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Summary saved to {summary_path}")
        
        # Keep latest links for convenience
        latest_results = os.path.join(self.config.output_dir, "results_latest.json")
        latest_summary = os.path.join(self.config.output_dir, "summary_latest.json")
        
        import shutil
        shutil.copy2(output_path, latest_results)
        shutil.copy2(summary_path, latest_summary)
        logger.info(f"Latest links created: {latest_results}, {latest_summary}")
    
    def _log_candidate_details(self, problem_id: str, candidate_num: int, 
                              barc_output: BARCOutput, execution_result: ExecutionResult,
                              evaluation_result: EvaluationResult, stage: str):
        """Log detailed candidate information to WandB and local files"""
        try:
            # Log BARC response details
            wandb.log({
                f"{stage}_candidate_{candidate_num}_problem_id": problem_id,
                f"{stage}_candidate_{candidate_num}_accuracy": execution_result.accuracy,
                f"{stage}_candidate_{candidate_num}_reward": evaluation_result.total_reward,
                f"{stage}_candidate_{candidate_num}_concepts": barc_output.concepts or "",
                f"{stage}_candidate_{candidate_num}_description": barc_output.description or "",
                f"{stage}_candidate_{candidate_num}_code_length": len(barc_output.code),
                f"{stage}_candidate_{candidate_num}_success": execution_result.success,
                f"{stage}_candidate_{candidate_num}_code": barc_output.code,  # Add full code to WandB
            })
            
            # Save detailed logs to local files for debugging
            os.makedirs(f"{self.config.output_dir}/logs", exist_ok=True)
            log_file = f"{self.config.output_dir}/logs/{problem_id}_{stage}_candidate_{candidate_num}.txt"
            
            with open(log_file, "w") as f:
                f.write(f"=== {stage.upper()} CANDIDATE {candidate_num} ===\n")
                f.write(f"Problem ID: {problem_id}\n")
                f.write(f"Success: {execution_result.success}\n")
                f.write(f"Accuracy: {execution_result.accuracy:.4f}\n")
                f.write(f"Reward: {evaluation_result.total_reward:.4f}\n")
                f.write(f"Code Length: {len(barc_output.code)}\n\n")
                
                f.write("=== CONCEPTS ===\n")
                f.write(f"{barc_output.concepts or 'None'}\n\n")
                
                f.write("=== DESCRIPTION ===\n")
                f.write(f"{barc_output.description or 'None'}\n\n")
                
                f.write("=== FULL CODE ===\n")
                f.write(f"{barc_output.code}\n\n")
                
                f.write("=== RAW RESPONSE ===\n")
                f.write(f"{barc_output.raw_response}\n\n")
                
                if execution_result.error_messages:
                    f.write("=== EXECUTION ERRORS ===\n")
                    for error in execution_result.error_messages:
                        f.write(f"{error}\n")
                    f.write("\n")
                
                f.write("=== EVALUATION FEEDBACK ===\n")
                for name, feedback in evaluation_result.detailed_feedback.items():
                    f.write(f"{name}: {feedback}\n")
            
            logger.info(f"Detailed logs saved to: {log_file}")
            
            # Log component scores
            for component, score in evaluation_result.component_scores.items():
                wandb.log({f"{stage}_candidate_{candidate_num}_{component}_score": score})
            
            # Log verifications
            for verifier, result in evaluation_result.verifications.items():
                wandb.log({
                    f"{stage}_candidate_{candidate_num}_{verifier}_passed": result.passed,
                    f"{stage}_candidate_{candidate_num}_{verifier}_confidence": result.confidence
                })
            
            # Log full responses as artifacts (for debugging)
            if wandb.run:
                # Create artifact for full responses
                artifact = wandb.Artifact(
                    f"{problem_id}_{stage}_candidate_{candidate_num}_responses", 
                    type="responses"
                )
                
                # Save BARC full response
                with open(f"temp_barc_{stage}_{candidate_num}.txt", "w") as f:
                    f.write(f"Raw Response:\n{barc_output.raw_response}\n\n")
                    f.write(f"Parsed Code:\n{barc_output.code}\n\n")
                    f.write(f"Concepts: {barc_output.concepts}\n\n")
                    f.write(f"Description: {barc_output.description}\n")
                
                artifact.add_file(f"temp_barc_{stage}_{candidate_num}.txt")
                
                # Save evaluation feedback
                with open(f"temp_eval_{stage}_{candidate_num}.txt", "w") as f:
                    f.write("Evaluation Feedback:\n")
                    for name, feedback in evaluation_result.detailed_feedback.items():
                        f.write(f"{name}: {feedback}\n")
                
                artifact.add_file(f"temp_eval_{stage}_{candidate_num}.txt")
                wandb.log_artifact(artifact)
                
                # Clean up temp files
                os.remove(f"temp_barc_{stage}_{candidate_num}.txt")
                os.remove(f"temp_eval_{stage}_{candidate_num}.txt")
                
        except Exception as e:
            logger.warning(f"Failed to log candidate details: {e}")
    
    def _log_alignment_details(self, problem_id: str, candidate_num: int,
                              original_output: BARCOutput, aligned_output: BARCOutput,
                              quality):
        """Log alignment process details to WandB"""
        try:
            # Import AlignmentQuality here to avoid circular imports
            from .alignment.quality_analyzer import AlignmentQuality
            
            # Log basic alignment metrics
            wandb.log({
                f"alignment_candidate_{candidate_num}_problem_id": problem_id,
                f"alignment_candidate_{candidate_num}_improvement_score": quality.improvement_score,
                f"alignment_candidate_{candidate_num}_structure_preserved": quality.structure_preserved,
                f"alignment_candidate_{candidate_num}_original_code_length": len(original_output.code),
                f"alignment_candidate_{candidate_num}_aligned_code_length": len(aligned_output.code),
                f"alignment_candidate_{candidate_num}_code_length_change": quality.code_length_change,
                f"alignment_candidate_{candidate_num}_description_changed": quality.description_changed,
            })
            
            # Log quality components
            wandb.log({
                f"alignment_candidate_{candidate_num}_has_concepts": quality.has_concepts,
                f"alignment_candidate_{candidate_num}_has_description": quality.has_description,
                f"alignment_candidate_{candidate_num}_has_transform_function": quality.has_transform_function,
                f"alignment_candidate_{candidate_num}_has_common_imports": quality.has_common_imports,
                f"alignment_candidate_{candidate_num}_has_color_constants": quality.has_color_constants,
                f"alignment_candidate_{candidate_num}_uses_common_functions": quality.uses_common_functions,
            })
            
            # Log full alignment responses as artifacts
            if wandb.run:
                artifact = wandb.Artifact(
                    f"{problem_id}_alignment_candidate_{candidate_num}",
                    type="alignment"
                )
                
                # Save original and aligned code
                with open(f"temp_alignment_original_{candidate_num}.py", "w") as f:
                    f.write(f"# Original BARC Code\n")
                    f.write(f"# Concepts: {original_output.concepts}\n")
                    f.write(f"# Description: {original_output.description}\n\n")
                    f.write(original_output.code)
                
                with open(f"temp_alignment_aligned_{candidate_num}.py", "w") as f:
                    f.write(f"# Aligned BARC Code\n")
                    f.write(f"# Concepts: {aligned_output.concepts}\n")
                    f.write(f"# Description: {aligned_output.description}\n\n")
                    f.write(aligned_output.code)
                    f.write(f"\n\n# Quality Analysis\n")
                    f.write(f"# Improvement Score: {quality.improvement_score}\n")
                    f.write(f"# Structure Preserved: {quality.structure_preserved}\n")
                
                artifact.add_file(f"temp_alignment_original_{candidate_num}.py")
                artifact.add_file(f"temp_alignment_aligned_{candidate_num}.py")
                wandb.log_artifact(artifact)
                
                # Clean up temp files
                os.remove(f"temp_alignment_original_{candidate_num}.py")
                os.remove(f"temp_alignment_aligned_{candidate_num}.py")
                
        except Exception as e:
            logger.warning(f"Failed to log alignment details: {e}")
    
    def _log_optimization_history(self, problem_id: str, candidate_num: int, 
                                 optimization_result: OptimizationResult):
        """Log LatentSeek optimization history"""
        try:
            # Log optimization summary
            wandb.log({
                f"optimization_candidate_{candidate_num}_steps": optimization_result.optimization_steps,
                f"optimization_candidate_{candidate_num}_converged": optimization_result.converged,
                f"optimization_candidate_{candidate_num}_initial_reward": optimization_result.reward_history[0],
                f"optimization_candidate_{candidate_num}_final_reward": optimization_result.reward_history[-1],
                f"optimization_candidate_{candidate_num}_improvement": 
                    optimization_result.reward_history[-1] - optimization_result.reward_history[0]
            })
            
            # Log reward history
            for step, reward in enumerate(optimization_result.reward_history):
                wandb.log({
                    f"optimization_candidate_{candidate_num}_step_{step}_reward": reward,
                    "optimization_step": step
                })
                
        except Exception as e:
            logger.warning(f"Failed to log optimization history: {e}")


def main():
    """Main entry point"""
    # Create default configuration
    config = PipelineConfig()
    
    # Create pipeline
    pipeline = ARCLatentSeekPipeline(config)
    
    # Solve some example problems
    results = pipeline.solve_problems(
        split="validation",
        num_problems=5
    )
    
    return results


if __name__ == "__main__":
    main()