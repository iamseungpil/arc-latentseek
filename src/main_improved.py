"""
ARC LatentSeek Pipeline - Improved version with parallelization and early stopping
"""

import os
import sys
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import argparse
from datetime import datetime
import wandb
from concurrent.futures import ThreadPoolExecutor, as_completed

from .data import ARCDataLoader, ARCProblem
from .generators import BARCGenerator, BARCOutput
from .executors import CodeExecutor, ExecutionResult
from .evaluators import GLMEvaluator, EvaluationResult
from .optimizers import LatentSeekOptimizer, OptimizationResult
# from .alignment import BARCCodeAligner  # Optional, disabled for now
from .utils import visualize_problem, save_visualization


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the pipeline"""
    mode: str = "solve"
    problems_split: str = "validation"
    num_problems: Optional[int] = None
    num_candidates: int = 4  # Reduced from 8
    output_dir: str = "results"
    use_code_alignment: bool = False
    use_description_optimization: bool = True
    save_visualizations: bool = True
    wandb_project: str = "arc-latentseek"
    device: str = "cuda:0"
    seed: int = 42
    alignment_method: str = "gpt4"
    alignment_batch_size: int = 4
    alignment_quality_threshold: float = 7.0
    save_barc_responses: bool = True
    save_glm_responses: bool = True
    
    @classmethod
    def from_args(cls, args):
        """Create config from command line arguments"""
        return cls(
            mode=args.mode,
            problems_split=args.problems,
            num_problems=args.num_problems,
            num_candidates=args.num_candidates,
            output_dir=args.output_dir,
            device=f"cuda:{args.gpu}" if args.gpu is not None else "cuda:0"
        )


@dataclass
class SolutionResult:
    """Result for a single problem"""
    problem_id: str
    success: bool
    best_accuracy: float
    best_candidate_idx: int
    candidates_tried: int
    time_taken: float
    error_messages: List[str] = None
    barc_raw_responses: List[str] = None
    glm_responses: List[Dict] = None
    best_concepts: str = None
    best_description: str = None
    best_code: str = None
    alignment_quality_scores: List[float] = None
    
    def to_dict(self):
        return {
            "problem_id": self.problem_id,
            "success": self.success,
            "best_accuracy": self.best_accuracy,
            "best_candidate_idx": self.best_candidate_idx,
            "candidates_tried": self.candidates_tried,
            "time_taken": self.time_taken,
            "error_messages": self.error_messages,
            "barc_raw_responses": self.barc_raw_responses,
            "glm_responses": self.glm_responses,
            "best_concepts": self.best_concepts,
            "best_description": self.best_description,
            "best_code": self.best_code,
            "alignment_quality_scores": self.alignment_quality_scores
        }


class ImprovedARCPipeline:
    """Improved ARC solving pipeline with parallelization and early stopping"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        self.data_loader = ARCDataLoader()
        self.barc_generator = BARCGenerator(device=config.device)
        self.code_executor = CodeExecutor()
        self.glm_evaluator = GLMEvaluator(device=config.device)
        self.latentseek_optimizer = LatentSeekOptimizer(
            barc_generator=self.barc_generator,
            executor=self.code_executor,
            glm_evaluator=self.glm_evaluator,
            device=config.device
        )
        
        # Code alignment disabled for now
        self.code_aligner = None
            
        # Create output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(config.output_dir, f"{config.problems_split}_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.logs_dir, exist_ok=True)
        
        if config.save_visualizations:
            self.viz_dir = os.path.join(self.output_dir, f"visualizations_{self.timestamp}")
            os.makedirs(self.viz_dir, exist_ok=True)
            
        self.latentseek_logs_dir = os.path.join(self.output_dir, "latentseek_logs")
        os.makedirs(self.latentseek_logs_dir, exist_ok=True)
            
        # Initialize wandb
        if config.wandb_project:
            wandb.init(
                project=config.wandb_project,
                name=f"{config.problems_split}_{self.timestamp}",
                config=config.__dict__
            )
    
    def _execute_candidate(self, candidate_info: Tuple[int, BARCOutput, ARCProblem]) -> Tuple[int, ExecutionResult]:
        """Execute a single candidate (for parallel processing)"""
        idx, candidate, problem = candidate_info
        
        try:
            if not hasattr(candidate, 'code'):
                logger.error(f"Candidate {idx+1} missing 'code' attribute!")
                return idx, None
                
            execution_result = self.code_executor.execute(candidate.code, problem)
            return idx, execution_result
            
        except Exception as e:
            logger.error(f"Error executing candidate {idx+1}: {e}")
            return idx, None
    
    def solve_problem(self, problem: ARCProblem) -> SolutionResult:
        """Solve a single ARC problem with early stopping"""
        start_time = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"Solving problem: {problem.uid}")
        logger.info(f"Train pairs: {len(problem.train_pairs)}, Test pairs: {len(problem.test_pairs)}")
        
        # Track results
        best_accuracy = 0.0
        best_candidate_idx = -1
        candidates_tried_count = 0
        failed_errors = []
        all_barc_raw_responses = []
        all_glm_responses = []
        candidate_details = []
        alignment_quality_scores = []
        
        # Track initial performance
        initial_best_accuracy = 0.0
        initial_success = False
        
        # Generate candidates
        logger.info(f"Generating {self.config.num_candidates} candidate solutions...")
        gen_start_time = time.time()
        candidates = self.barc_generator.generate(
            problem, 
            num_candidates=self.config.num_candidates
        )
        gen_time = time.time() - gen_start_time
        logger.info(f"Generation completed in {gen_time:.2f}s")
        wandb.log({"generation_time": gen_time})
        
        # Collect BARC raw responses
        for candidate in candidates:
            if hasattr(candidate, 'raw_response'):
                all_barc_raw_responses.append(candidate.raw_response)
        
        # Apply code alignment if enabled
        aligned_candidates = candidates
        if self.config.use_code_alignment and self.code_aligner:
            # ... alignment code (unchanged) ...
            pass
        
        # Parallel execution of candidates
        logger.info("Executing candidates in parallel...")
        execution_results = [None] * len(aligned_candidates)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all execution tasks
            future_to_idx = {
                executor.submit(self._execute_candidate, (i, candidate, problem)): i 
                for i, candidate in enumerate(aligned_candidates)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                idx, exec_result = future.result()
                execution_results[idx] = exec_result
                
                if exec_result and exec_result.success:
                    logger.info(f"Candidate {idx+1} execution completed - Accuracy: {exec_result.accuracy:.2%}")
                else:
                    logger.warning(f"Candidate {idx+1} execution failed")
        
        # Evaluate each candidate sequentially (GLM cannot be parallelized easily)
        for i, (candidate, execution_result) in enumerate(zip(aligned_candidates, execution_results)):
            if execution_result is None:
                continue
                
            logger.info(f"Evaluating candidate {i+1}/{len(aligned_candidates)}")
            candidates_tried_count += 1
            
            if not execution_result.success:
                # Still track failed candidates for visualization
                candidate_details.append((candidate, execution_result))
                continue
            
            # Early stopping check before GLM evaluation
            if execution_result.accuracy == 1.0:
                logger.info(f"🎯 Candidate {i+1} achieved 100% accuracy! Early stopping.")
                best_accuracy = 1.0
                best_candidate_idx = i
                initial_best_accuracy = 1.0
                initial_success = True
                
                # Log the perfect solution
                self._log_candidate_details(problem.uid, i+1, candidate, execution_result, None, "perfect")
                
                # Track for visualization
                candidate_details.append((candidate, execution_result))
                
                # Skip remaining candidates
                break
            
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
            
            # Collect GLM response
            if hasattr(evaluation_result, 'glm_raw_response'):
                all_glm_responses.append({
                    "candidate": i+1,
                    "response": evaluation_result.glm_raw_response
                })
            
            logger.info(f"Candidate {i+1} - Accuracy: {execution_result.accuracy:.2%}, "
                       f"Reward: {evaluation_result.total_reward:.3f}")
            
            # Track initial performance (before optimization)
            if execution_result.accuracy > initial_best_accuracy:
                initial_best_accuracy = execution_result.accuracy
            if evaluation_result.total_reward >= 1.0:
                initial_success = True
            
            # Log detailed information
            self._log_candidate_details(problem.uid, i+1, candidate, execution_result, evaluation_result, "initial")
            
            # Apply LatentSeek optimization if not all training pairs are correct
            if execution_result.accuracy < 1.0:
                # Always use description-based optimization
                logger.info(f"Applying description-based LatentSeek optimization to candidate {i+1}")
                optimization_result = self.latentseek_optimizer.optimize_description_based(
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
                        # Check for early stopping with optimized result
                        if optimized_execution.accuracy == 1.0:
                            logger.info(f"🎯 Optimized candidate {i+1} achieved 100% accuracy! Early stopping.")
                            best_accuracy = 1.0
                            best_candidate_idx = i
                            
                            # Log the perfect optimized solution
                            self._log_candidate_details(problem.uid, i+1, optimized_candidate, 
                                                      optimized_execution, None, "optimized_perfect")
                            
                            # Log optimization history
                            self._log_optimization_history(problem.uid, i+1, optimization_result)
                            
                            # Track for visualization
                            candidate_details.append((optimized_candidate, optimized_execution))
                            
                            # Skip remaining candidates
                            break
                        
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
            
            # Update best result
            if execution_result.accuracy > best_accuracy:
                best_accuracy = execution_result.accuracy
                best_candidate_idx = i
        
        # Collect all error messages
        if candidate_details:
            for candidate, exec_result in candidate_details:
                if exec_result and exec_result.error_messages:
                    failed_errors.extend(exec_result.error_messages)
        
        # Determine success
        success = best_accuracy >= 1.0
        
        # Log results summary
        logger.info(f"\n{'='*80}")
        logger.info(f"Problem {problem.uid} Results:")
        logger.info(f"  Initial best accuracy: {initial_best_accuracy:.2%}")
        logger.info(f"  Final best accuracy: {best_accuracy:.2%}")
        logger.info(f"  Improvement: {(best_accuracy - initial_best_accuracy):.2%}")
        logger.info(f"  Success: {success}")
        logger.info(f"  Candidates tried: {candidates_tried_count}/{self.config.num_candidates}")
        logger.info(f"  Time taken: {time.time() - start_time:.2f}s")
        
        # Save best code and details
        best_concepts = None
        best_description = None  
        best_code = None
        if best_candidate_idx >= 0 and candidate_details:
            best_candidate, _ = candidate_details[best_candidate_idx]
            if hasattr(best_candidate, 'concepts'):
                best_concepts = best_candidate.concepts
            if hasattr(best_candidate, 'description'):
                best_description = best_candidate.description
            if hasattr(best_candidate, 'code'):
                best_code = best_candidate.code
        
        # Save visualizations if enabled
        if self.config.save_visualizations and candidate_details:
            self._save_visualizations(problem, candidate_details, best_candidate_idx)
        
        # Log to wandb
        wandb.log({
            "problem_id": problem.uid,
            "initial_best_accuracy": initial_best_accuracy,
            "final_best_accuracy": best_accuracy,
            "improvement": best_accuracy - initial_best_accuracy,
            "success": success,
            "candidates_tried": candidates_tried_count,
            "time_taken": time.time() - start_time
        })
        
        return SolutionResult(
            problem_id=problem.uid,
            success=success,
            best_accuracy=best_accuracy,
            best_candidate_idx=best_candidate_idx,
            candidates_tried=candidates_tried_count,
            time_taken=time.time() - start_time,
            error_messages=failed_errors[:10] if failed_errors else None,
            barc_raw_responses=all_barc_raw_responses if self.config.save_barc_responses else None,
            glm_responses=all_glm_responses if self.config.save_glm_responses else None,
            best_concepts=best_concepts,
            best_description=best_description,
            best_code=best_code,
            alignment_quality_scores=alignment_quality_scores if alignment_quality_scores else None
        )
    
    def _log_candidate_details(self, problem_id: str, candidate_idx: int, candidate: BARCOutput, 
                              execution_result: ExecutionResult, evaluation_result: Optional[EvaluationResult],
                              prefix: str = "initial"):
        """Log detailed information about a candidate"""
        log_path = os.path.join(self.logs_dir, f"{problem_id}_{prefix}_candidate_{candidate_idx}.txt")
        
        with open(log_path, 'w') as f:
            f.write(f"=== {prefix.upper()} CANDIDATE {candidate_idx} ===\n")
            f.write(f"Problem ID: {problem_id}\n")
            f.write(f"Success: {execution_result.success}\n")
            f.write(f"Accuracy: {execution_result.accuracy:.4f}\n")
            
            if evaluation_result:
                f.write(f"Reward: {evaluation_result.total_reward:.4f}\n")
            
            if hasattr(candidate, 'code'):
                f.write(f"Code Length: {len(candidate.code)}\n")
            
            f.write(f"\n=== CONCEPTS ===\n")
            if hasattr(candidate, 'concepts'):
                f.write(f"{candidate.concepts}\n")
            
            f.write(f"\n=== DESCRIPTION ===\n")
            if hasattr(candidate, 'description'):
                f.write(f"{candidate.description}\n")
            
            f.write(f"\n=== FULL CODE ===\n")
            if hasattr(candidate, 'code'):
                f.write(candidate.code)
            
            if hasattr(candidate, 'raw_response'):
                f.write(f"\n\n=== RAW RESPONSE ===\n")
                f.write(candidate.raw_response)
            
            if execution_result.error_messages:
                f.write(f"\n\n=== ERRORS ===\n")
                for error in execution_result.error_messages:
                    f.write(f"{error}\n")
            
            if execution_result.comparison_results:
                f.write(f"\n\n=== COMPARISON RESULTS ===\n")
                for i, result in enumerate(execution_result.comparison_results):
                    f.write(f"Pair {i}: {result}\n")
            
            if evaluation_result and hasattr(evaluation_result, 'glm_raw_response'):
                f.write(f"\n\n=== GLM EVALUATION ===\n")
                f.write(f"{evaluation_result.glm_raw_response}\n")
    
    def _log_optimization_history(self, problem_id: str, candidate_idx: int, 
                                optimization_result: OptimizationResult):
        """Log LatentSeek optimization history"""
        log_path = os.path.join(self.latentseek_logs_dir, 
                              f"{problem_id}_candidate_{candidate_idx}_optimization.json")
        
        history_data = {
            "problem_id": problem_id,
            "candidate_num": candidate_idx,
            "optimization_steps": optimization_result.optimization_steps,
            "converged": optimization_result.converged,
            "reward_history": optimization_result.reward_history,
            "accuracy_history": optimization_result.accuracy_history,
            "initial_code": optimization_result.initial_code,
            "final_code": optimization_result.final_output.code if optimization_result.final_output else None,
            "final_description": optimization_result.final_output.description if optimization_result.final_output else None,
            "step_outputs": []
        }
        
        # Save main optimization log
        with open(log_path, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        # Save individual step files
        if optimization_result.step_outputs:
            step_dir = os.path.join(self.latentseek_logs_dir, 
                                  f"{problem_id}_candidate_{candidate_idx}_steps")
            os.makedirs(step_dir, exist_ok=True)
            
            for i, step_output in enumerate(optimization_result.step_outputs):
                step_file = os.path.join(step_dir, f"step_{i}.json")
                step_data = {
                    "step": i,
                    "description": step_output.description if hasattr(step_output, 'description') else None,
                    "code_length": len(step_output.code) if hasattr(step_output, 'code') else 0
                }
                with open(step_file, 'w') as f:
                    json.dump(step_data, f, indent=2)
    
    def _save_visualizations(self, problem: ARCProblem, candidate_details: List[Tuple[BARCOutput, ExecutionResult]], 
                           best_idx: int):
        """Save problem visualizations"""
        try:
            # Save best solution visualization
            if best_idx >= 0 and best_idx < len(candidate_details):
                best_candidate, best_result = candidate_details[best_idx]
                if best_result and best_result.output_grids:
                    save_path = os.path.join(self.viz_dir, f"{problem.uid}_best.png")
                    save_visualization(problem, best_result.output_grids, save_path)
                    logger.info(f"Saved best solution visualization to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
            
        # Also save visualization for each candidate that was evaluated
        if self.config.save_visualizations and candidate_details:
            for i, (candidate, exec_result) in enumerate(candidate_details):
                if exec_result and exec_result.output_grids:
                    try:
                        save_path = os.path.join(self.viz_dir, f"{problem.uid}_candidate_{i+1}.png")
                        save_visualization(problem, exec_result.output_grids, save_path)
                    except Exception as e:
                        logger.error(f"Failed to save visualization for candidate {i+1}: {e}")
    
    def solve_problems(self, problems: List[ARCProblem]) -> List[SolutionResult]:
        """Solve multiple ARC problems"""
        results = []
        
        for i, problem in enumerate(problems):
            logger.info(f"\nProcessing Problem {i+1}/{len(problems)}: {problem.uid}")
            
            try:
                result = self.solve_problem(problem)
                results.append(result)
                
                # Save incremental results after each problem
                self._save_incremental_results(results, i+1, len(problems))
                
                # Log progress
                success_count = sum(1 for r in results if r.success)
                logger.info(f"Progress: {i+1}/{len(problems)} problems, "
                          f"{success_count} solved ({success_count/(i+1)*100:.1f}%)")
                
            except Exception as e:
                logger.error(f"Failed to solve problem {problem.uid}: {e}")
                import traceback
                traceback.print_exc()
                
                # Save error result
                results.append(SolutionResult(
                    problem_id=problem.uid,
                    success=False,
                    best_accuracy=0.0,
                    best_candidate_idx=-1,
                    candidates_tried=0,
                    time_taken=0.0,
                    error_messages=[str(e)]
                ))
        
        # Save final results
        self._save_results(results)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _save_incremental_results(self, results: List[SolutionResult], current_idx: int, total: int):
        """Save results incrementally after each problem"""
        incremental_path = os.path.join(self.output_dir, "results_incremental.json")
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "progress": f"{current_idx}/{total}",
            "problems_attempted": current_idx,
            "problems_solved": sum(1 for r in results if r.success),
            "average_accuracy": sum(r.best_accuracy for r in results) / len(results) if results else 0,
            "results": [r.to_dict() for r in results]
        }
        
        with open(incremental_path, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _save_results(self, results: List[SolutionResult]):
        """Save results to JSON file with timestamp"""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Also save summary
        summary_file = os.path.join(self.output_dir, f"summary_{timestamp}.json")
        summary = {
            "timestamp": timestamp,
            "config": self.config.__dict__,
            "problems_attempted": len(results),
            "problems_solved": sum(1 for r in results if r.success),
            "average_accuracy": sum(r.best_accuracy for r in results) / len(results) if results else 0,
            "total_time": sum(r.time_taken for r in results),
            "problems_with_perfect_score": sum(1 for r in results if r.best_accuracy >= 1.0),
            "problems_with_partial_score": sum(1 for r in results if 0 < r.best_accuracy < 1.0)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
    
    def _print_summary(self, results: List[SolutionResult]):
        """Print results summary"""
        total = len(results)
        solved = sum(1 for r in results if r.success)
        perfect = sum(1 for r in results if r.best_accuracy >= 1.0)
        partial = sum(1 for r in results if 0 < r.best_accuracy < 1.0)
        avg_accuracy = sum(r.best_accuracy for r in results) / total if total > 0 else 0
        total_time = sum(r.time_taken for r in results)
        
        print(f"\n{'='*80}")
        print(f"FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total problems attempted: {total}")
        print(f"Problems solved (100%): {solved} ({solved/total*100:.1f}%)")
        print(f"Problems with perfect score: {perfect}")
        print(f"Problems with partial score: {partial}")
        print(f"Average accuracy: {avg_accuracy:.2%}")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
        print(f"Average time per problem: {total_time/total:.2f}s")
        print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="ARC LatentSeek Pipeline - Improved")
    parser.add_argument("--mode", default="solve", help="Mode: solve")
    parser.add_argument("--problems", default="validation", help="Dataset split: train, validation, or all")
    parser.add_argument("--num_problems", type=int, default=None, help="Number of problems to solve")
    parser.add_argument("--num_candidates", type=int, default=4, help="Number of candidates per problem (default: 4)")
    parser.add_argument("--output_dir", default="results", help="Output directory")
    parser.add_argument("--gpu", type=int, default=None, help="GPU device number")
    parser.add_argument("--no_visualization", action="store_true", help="Disable visualization")
    
    args = parser.parse_args()
    
    # Create config
    config = PipelineConfig.from_args(args)
    config.save_visualizations = not args.no_visualization
    
    # Initialize pipeline
    pipeline = ImprovedARCPipeline(config)
    
    # Load problems
    data_loader = ARCDataLoader()
    problems = data_loader.get_problems(
        split=args.problems,
        num_problems=args.num_problems
    )
    
    logger.info(f"Loaded {len(problems)} problems from {args.problems} split")
    
    # Solve problems
    if args.mode == "solve":
        results = pipeline.solve_problems(problems)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    # Cleanup
    if config.wandb_project:
        wandb.finish()
    
    return results


if __name__ == "__main__":
    main()