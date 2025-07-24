"""
Improved comparative experiment with 4 conditions
Based on original main.py structure
"""

import os
import sys
import json
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader, ARCProblem
from src.generators import BARCGenerator, BARCOutput
from src.executors import CodeExecutor, ExecutionResult, GridRenderer
from src.evaluators import GLMEvaluator, EvaluationResult, RewardModel, MultiTensorEvaluator, convert_multitensor_to_reward
from src.optimizers import LatentSeekOptimizer, OptimizationResult

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentCondition:
    """Single experimental condition"""
    name: str
    use_glm_description: bool
    use_multitensor_reward: bool
    gpu_id: int


@dataclass 
class StepResult:
    """Result from a single optimization step"""
    step: int
    reward: float
    accuracy: float
    description: str
    code: str
    execution_success: bool


@dataclass
class ExperimentResult:
    """Results from a single experiment"""
    condition: str
    problem_id: str
    candidate_id: int
    optimization_steps: List[StepResult]
    final_accuracy: float
    final_reward: float
    total_time: float
    best_step: int
    best_accuracy: float


class ComparativeExperimentV2:
    """Improved comparative experiments following main.py structure"""
    
    def __init__(self, output_dir: str = "comparative_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        os.makedirs(f"{output_dir}/visualizations", exist_ok=True)
        
        # Initialize components
        logger.info("Initializing components...")
        self.loader = ARCDataLoader()
        self.barc_generator = BARCGenerator()
        self.code_executor = CodeExecutor()
        self.glm_evaluator = GLMEvaluator()
        self.multitensor_evaluator = MultiTensorEvaluator()
        self.reward_model = RewardModel()
        self.renderer = GridRenderer()
        # LatentSeekOptimizer requires the other components
        self.latent_optimizer = LatentSeekOptimizer(
            barc_generator=self.barc_generator,
            code_executor=self.code_executor,
            glm_evaluator=self.glm_evaluator
        )
        
        # Use 5 problems from the previous experiments
        self.problems = ['2072aba6', 'bb52a14b', '136b0064', 'ea9794b1', 'f5aa3634']
        
        logger.info(f"Using 5 validation problems: {self.problems}")
        
        # Number of candidates and optimization steps
        self.num_candidates = 2
        self.optimization_steps = 5
        
        # GLM descriptions will be generated during experiments
        self.glm_descriptions = {}
    
    def _generate_glm_description(self, problem: ARCProblem) -> str:
        """Generate GLM description for a problem"""
        logger.info(f"Generating GLM description for problem {problem.uid}...")
        
        # Create visualization for GLM
        image_path = os.path.join(self.output_dir, f"temp_{problem.uid}_glm_input.png")
        self.renderer.render_arc_problem(problem, image_path)
        
        # Prompt for GLM to describe the pattern
        prompt = """Look at this ARC problem and provide a clear, concise description of the pattern or rule that transforms the input to the output.

Focus on:
1. What pattern appears in the input
2. How the input should be transformed to create the output
3. Any spatial relationships, colors, or geometric operations involved

Provide a description that could be used as guidance for code generation. Be specific but concise."""
        
        # Use GLM evaluator's internal method to get description
        try:
            response = self.glm_evaluator._run_glm_inference(image_path, prompt)
            
            # Extract meaningful part from GLM thinking
            if "<think>" in response:
                response = response.split("</think>")[0].replace("<think>", "").strip()
                # Take first few sentences
                sentences = response.split(".")[:3]
                response = ". ".join(sentences) + "."
            
            logger.info(f"Generated GLM description: {response[:100]}...")
            
            # Clean up temp image
            if os.path.exists(image_path):
                os.remove(image_path)
                
            return response
        except Exception as e:
            logger.error(f"Failed to generate GLM description: {e}")
            return ""
    
    def run_single_experiment(
        self, 
        condition: ExperimentCondition, 
        problem_id: str,
        candidate_idx: int
    ) -> ExperimentResult:
        """Run single experiment for one condition and problem"""
        logger.info(f"{'='*60}")
        logger.info(f"Running {condition.name} on {problem_id} candidate {candidate_idx}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Load problem
        problem = self.loader.get_problem_by_id(problem_id)
        
        # Generate candidates with/without GLM description
        candidates = self._generate_candidates(problem, condition)
        
        if candidate_idx >= len(candidates):
            logger.error(f"Candidate {candidate_idx} not found (only {len(candidates)} generated)")
            return None
            
        candidate = candidates[candidate_idx]
        
        # Track optimization steps
        step_results = []
        best_accuracy = 0.0
        best_step = 0
        best_candidate = candidate
        
        # Step 0: Initial evaluation
        logger.info("Step 0: Evaluating initial candidate...")
        initial_result = self._evaluate_candidate(candidate, problem, condition)
        
        step_result = StepResult(
            step=0,
            reward=initial_result['reward'],
            accuracy=initial_result['accuracy'],
            description=candidate.description if hasattr(candidate, 'description') else "",
            code=candidate.code,
            execution_success=initial_result['execution_success']
        )
        step_results.append(step_result)
        
        if step_result.accuracy > best_accuracy:
            best_accuracy = step_result.accuracy
            best_step = 0
            best_candidate = candidate
        
        # Save initial logs
        self._save_step_log(problem_id, condition.name, candidate_idx, 0, candidate, initial_result)
        
        # Optimization steps
        current_candidate = candidate
        for step in range(1, self.optimization_steps + 1):
            logger.info(f"\nStep {step}: Applying LatentSeek optimization...")
            
            try:
                # Apply LatentSeek optimization using description-based method
                logger.info(f"Calling optimize_description_based with params: problem={problem.uid}, initial_reward={step_results[-1].reward}")
                opt_result = self.latent_optimizer.optimize_description_based(
                    problem=problem,
                    initial_output=current_candidate,
                    initial_reward=step_results[-1].reward
                )
                
                if opt_result and opt_result.final_output:
                    optimized_candidate = opt_result.final_output
                else:
                    logger.warning(f"Optimization failed at step {step}, keeping previous candidate")
                    optimized_candidate = current_candidate
                
                # Evaluate optimized candidate
                eval_result = self._evaluate_candidate(optimized_candidate, problem, condition)
                
                step_result = StepResult(
                    step=step,
                    reward=eval_result['reward'],
                    accuracy=eval_result['accuracy'],
                    description=optimized_candidate.description if hasattr(optimized_candidate, 'description') else "",
                    code=optimized_candidate.code,
                    execution_success=eval_result['execution_success']
                )
                step_results.append(step_result)
                
                # Save step log
                self._save_step_log(problem_id, condition.name, candidate_idx, step, optimized_candidate, eval_result)
                
                # Update best
                if step_result.accuracy > best_accuracy:
                    best_accuracy = step_result.accuracy
                    best_step = step
                    best_candidate = optimized_candidate
                
                # Update current candidate for next iteration
                current_candidate = optimized_candidate
                
                # Early stopping if 100% accuracy
                if step_result.accuracy >= 100.0:
                    logger.info(f"🎯 Achieved 100% accuracy at step {step}! Early stopping.")
                    break
                    
            except Exception as e:
                logger.error(f"Error in optimization step {step}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # Continue with previous candidate
                step_result = StepResult(
                    step=step,
                    reward=step_results[-1].reward,
                    accuracy=step_results[-1].accuracy,
                    description=current_candidate.description if hasattr(current_candidate, 'description') else "",
                    code=current_candidate.code,
                    execution_success=False
                )
                step_results.append(step_result)
        
        total_time = time.time() - start_time
        
        # Create visualization
        self._create_visualization(problem, best_candidate, condition, candidate_idx)
        
        return ExperimentResult(
            condition=condition.name,
            problem_id=problem_id,
            candidate_id=candidate_idx,
            optimization_steps=step_results,
            final_accuracy=step_results[-1].accuracy,
            final_reward=step_results[-1].reward,
            total_time=total_time,
            best_step=best_step,
            best_accuracy=best_accuracy
        )
    
    def _generate_candidates(
        self, 
        problem: ARCProblem, 
        condition: ExperimentCondition
    ) -> List[BARCOutput]:
        """Generate candidates with/without GLM description"""
        
        if condition.use_glm_description:
            # Generate GLM description if not already cached
            if problem.uid not in self.glm_descriptions:
                self.glm_descriptions[problem.uid] = self._generate_glm_description(problem)
            
            glm_desc = self.glm_descriptions[problem.uid]
            
            if glm_desc:
                logger.info(f"Using GLM description: {glm_desc[:100]}...")
                additional_prompt = f"\nHint about the pattern: {glm_desc}\n"
            else:
                logger.warning(f"Failed to generate GLM description for {problem.uid}, proceeding without it")
                additional_prompt = ""
        else:
            additional_prompt = ""
        
        # Generate candidates
        logger.info(f"Generating {self.num_candidates} candidates...")
        
        candidates = self.barc_generator.generate(
            problem, 
            num_candidates=self.num_candidates,
            temperature=0.8,
            max_new_tokens=2048,
            additional_prompt=additional_prompt
        )
        
        logger.info(f"Generated {len(candidates)} candidates")
        return candidates
    
    def _evaluate_candidate(
        self, 
        candidate: BARCOutput, 
        problem: ARCProblem, 
        condition: ExperimentCondition
    ) -> Dict:
        """Evaluate candidate using appropriate reward function"""
        
        # Execute code on problem (not individual pairs)
        exec_result = self.code_executor.execute(candidate.code, problem)
        
        if condition.use_multitensor_reward:
            # Use multi-tensor evaluation
            if exec_result.success and exec_result.output_grids:
                generated_outputs = exec_result.output_grids
            else:
                # Use empty grids for failed executions
                generated_outputs = [np.zeros_like(pair.y) for pair in problem.train_pairs]
            
            multitensor_result = self.multitensor_evaluator.evaluate_multitensor(
                problem, generated_outputs
            )
            reward = convert_multitensor_to_reward(multitensor_result)
            
        else:
            # Use basic reward with GLM evaluation
            glm_result = self.glm_evaluator.evaluate(
                problem, candidate, exec_result
            )
            # Use GLM-based reward
            if hasattr(glm_result, 'total_reward'):
                reward = glm_result.total_reward
            else:
                # Fallback to accuracy if GLM evaluation fails
                reward = exec_result.accuracy if exec_result.success else -1.0
        
        return {
            'reward': reward,
            'accuracy': exec_result.accuracy * 100.0,  # Convert to percentage
            'execution_success': exec_result.success,
            'exec_result': exec_result
        }
    
    def _save_step_log(
        self, 
        problem_id: str, 
        condition: str, 
        candidate_idx: int,
        step: int,
        candidate: BARCOutput,
        eval_result: Dict
    ):
        """Save detailed log for each step"""
        log_data = {
            'problem_id': problem_id,
            'condition': condition,
            'candidate_idx': candidate_idx,
            'step': step,
            'code': candidate.code,
            'description': candidate.description if hasattr(candidate, 'description') else "",
            'reward': eval_result['reward'],
            'accuracy': eval_result['accuracy'],
            'execution_success': eval_result['execution_success'],
            'timestamp': datetime.now().isoformat()
        }
        
        filename = f"{problem_id}_{condition}_c{candidate_idx}_step{step}.json"
        filepath = os.path.join(self.output_dir, "logs", filename)
        
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def _create_visualization(
        self,
        problem: ARCProblem,
        candidate: BARCOutput,
        condition: ExperimentCondition,
        candidate_idx: int
    ):
        """Create visualization for best result"""
        try:
            # Execute code to get outputs
            exec_result = self.code_executor.execute(candidate.code, problem)
            
            if exec_result.success and exec_result.output_grids:
                # Create comparison image
                image_path = os.path.join(
                    self.output_dir, 
                    "visualizations",
                    f"{problem.uid}_{condition.name}_c{candidate_idx}_best.png"
                )
                
                self.renderer.render_problem_with_output(
                    problem,
                    exec_result.output_grids,
                    image_path
                )
                logger.info(f"Saved visualization to {image_path}")
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
    
    def run_experiments_for_gpu(self, gpu_id: int):
        """Run experiments assigned to specific GPU"""
        
        # Define conditions for each GPU
        if gpu_id == 5:
            conditions = [
                ExperimentCondition("basic_basic", False, False, 5),
                ExperimentCondition("glm_basic", True, False, 5)
            ]
        else:  # GPU 6
            conditions = [
                ExperimentCondition("basic_multitensor", False, True, 6),
                ExperimentCondition("glm_multitensor", True, True, 6)
            ]
        
        all_results = []
        
        for condition in conditions:
            logger.info(f"\n{'#'*60}")
            logger.info(f"Running condition: {condition.name} on GPU {gpu_id}")
            logger.info(f"{'#'*60}")
            
            for problem_id in self.problems:
                for candidate_idx in range(self.num_candidates):
                    try:
                        result = self.run_single_experiment(
                            condition, problem_id, candidate_idx
                        )
                        if result:
                            all_results.append(result)
                            self._save_result(result)
                    except Exception as e:
                        logger.error(f"Failed to run experiment: {e}")
                        import traceback
                        traceback.print_exc()
        
        # Save summary for this GPU
        self._save_gpu_summary(gpu_id, all_results)
        return all_results
    
    def _save_result(self, result: ExperimentResult):
        """Save single experiment result"""
        filename = f"{result.condition}_{result.problem_id}_c{result.candidate_id}_result.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # Convert StepResult objects to dicts
        result_dict = asdict(result)
        result_dict['optimization_steps'] = [asdict(step) for step in result.optimization_steps]
        
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
    
    def _save_gpu_summary(self, gpu_id: int, results: List[ExperimentResult]):
        """Save summary for GPU experiments"""
        summary = {
            'gpu_id': gpu_id,
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(results),
            'problems': self.problems,
            'glm_descriptions': self.glm_descriptions,
            'results_by_condition': {}
        }
        
        # Group by condition
        conditions = set(r.condition for r in results)
        for condition in conditions:
            condition_results = [r for r in results if r.condition == condition]
            summary['results_by_condition'][condition] = {
                'num_experiments': len(condition_results),
                'avg_final_accuracy': np.mean([r.final_accuracy for r in condition_results]),
                'avg_best_accuracy': np.mean([r.best_accuracy for r in condition_results]),
                'avg_best_step': np.mean([r.best_step for r in condition_results]),
                'perfect_solutions': sum(1 for r in condition_results if r.best_accuracy >= 100.0),
                'avg_time': np.mean([r.total_time for r in condition_results])
            }
        
        with open(os.path.join(self.output_dir, f'gpu{gpu_id}_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"GPU {gpu_id} EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        for condition, stats in summary['results_by_condition'].items():
            print(f"\n{condition}:")
            print(f"  Final Accuracy: {stats['avg_final_accuracy']:.1f}%")
            print(f"  Best Accuracy: {stats['avg_best_accuracy']:.1f}%")
            print(f"  Perfect Solutions: {stats['perfect_solutions']}/{stats['num_experiments']}")
            print(f"  Avg Best Step: {stats['avg_best_step']:.1f}")


def main():
    """Main function"""
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    
    # Set CUDA_VISIBLE_DEVICES to use only the specified GPU
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"Starting comparative experiment on GPU {gpu_id}")
    
    experiment = ComparativeExperimentV2(
        output_dir=f"comparative_results_gpu{gpu_id}"
    )
    
    results = experiment.run_experiments_for_gpu(gpu_id)
    
    print(f"\nExperiments completed on GPU {gpu_id}!")
    print(f"Results saved to: comparative_results_gpu{gpu_id}/")


if __name__ == "__main__":
    main()