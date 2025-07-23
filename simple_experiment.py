"""
Simplified experiment with only GLM reward and multitensor reward
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data import ARCDataLoader, ARCProblem
from src.generators import BARCOutput
from src.generators.barc_generator import BARCGenerator
from src.evaluators.arc_evaluator import ARCEvaluator
from src.evaluators.glm_evaluator import GLMEvaluator
from src.evaluators.multitensor_evaluator import MultiTensorEvaluator, convert_multitensor_to_reward
from src.optimizers import LatentSeekOptimizer, OptimizationResult
from src.executors.code_executor import CodeExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentCondition:
    """Experiment condition configuration"""
    name: str
    use_multitensor: bool  # True for multitensor, False for GLM reward


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
    """Result from a single experiment run"""
    condition: str
    problem_id: str
    candidate_id: int
    optimization_steps: List[StepResult]
    final_accuracy: float
    final_reward: float
    total_time: float
    best_step: int
    best_accuracy: float


class SimpleLatentSeekExperiment:
    """Run simplified experiments with GLM and multitensor rewards"""
    
    def __init__(self, gpu_id: int, num_candidates: int = 2):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.num_candidates = num_candidates
        
        # Create results directory
        self.results_dir = Path(f"simple_results_gpu{gpu_id}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        logger.info("Initializing components...")
        self.barc_generator = BARCGenerator()  # It auto-detects GPU
        self.arc_evaluator = ARCEvaluator()
        self.glm_evaluator = GLMEvaluator()
        self.multitensor_evaluator = MultiTensorEvaluator()
        self.code_executor = CodeExecutor()
        
        # Initialize optimizer with 20% optimization
        self.latent_optimizer = LatentSeekOptimizer(
            barc_generator=self.barc_generator,
            code_executor=self.code_executor,
            glm_evaluator=self.glm_evaluator,
            lr=0.03,
            max_steps=10,
            k=0.2,  # 20% optimization instead of 10%
            reward_threshold=0.5
        )
        
        # Load dataset
        loader = ARCDataLoader()
        self.train_problems = loader.train_problems
        self.val_problems = loader.validation_problems
        logger.info(f"Loaded {len(self.train_problems)} train, {len(self.val_problems)} val problems")
    
    def run_experiment(
        self,
        problems: List[ARCProblem],
        conditions: List[ExperimentCondition]
    ):
        """Run experiments on given problems with specified conditions"""
        
        for condition in conditions:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running condition: {condition.name} on GPU {self.gpu_id}")
            logger.info(f"{'='*60}")
            
            for problem in problems:
                for candidate_idx in range(self.num_candidates):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Running {condition.name} on {problem.uid} candidate {candidate_idx}")
                    logger.info(f"{'='*60}")
                    
                    try:
                        result = self._run_single_experiment(
                            problem, condition, candidate_idx
                        )
                        
                        # Save result
                        result_path = self.results_dir / f"{condition.name}_{problem.uid}_c{candidate_idx}_result.json"
                        with open(result_path, 'w') as f:
                            json.dump(asdict(result), f, indent=2)
                        
                        logger.info(
                            f"Completed {condition.name} on {problem.uid} candidate {candidate_idx}: "
                            f"accuracy={result.final_accuracy:.1f}%, reward={result.final_reward:.3f}"
                        )
                        
                    except Exception as e:
                        logger.error(f"Error in {condition.name} {problem.uid} candidate {candidate_idx}: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
    
    def _run_single_experiment(
        self,
        problem: ARCProblem,
        condition: ExperimentCondition,
        candidate_idx: int
    ) -> ExperimentResult:
        """Run single experiment with LatentSeek optimization"""
        
        start_time = time.time()
        problem_id = problem.uid
        
        # Generate initial candidates
        logger.info(f"Generating {self.num_candidates} candidates...")
        # Generate candidates one by one
        candidates = []
        for i in range(self.num_candidates):
            candidate = self.barc_generator.generate(problem, num_candidates=1)[0]
            candidates.append(candidate)
        
        if candidate_idx >= len(candidates):
            logger.error(f"Not enough candidates generated ({len(candidates)} < {candidate_idx + 1})")
            candidate_idx = len(candidates) - 1
        
        initial_candidate = candidates[candidate_idx]
        current_candidate = initial_candidate
        
        # Evaluate initial candidate
        logger.info(f"Step 0: Evaluating initial candidate...")
        initial_eval = self._evaluate_candidate(initial_candidate, problem, condition)
        
        step_results = [StepResult(
            step=0,
            reward=initial_eval['reward'],
            accuracy=initial_eval['accuracy'],
            description=initial_candidate.description or "",
            code=initial_candidate.code,
            execution_success=initial_eval['execution_success']
        )]
        
        best_accuracy = initial_eval['accuracy']
        best_step = 0
        best_candidate = initial_candidate
        
        # Optimization loop
        logger.info(f"\nStep 1: Applying LatentSeek optimization...")
        
        try:
            # Apply LatentSeek optimization with 20% token optimization
            opt_result = self.latent_optimizer.optimize(
                problem=problem,
                initial_output=current_candidate,
                initial_reward=initial_eval['reward']
            )
            
            if opt_result and opt_result.final_output:
                optimized_candidate = opt_result.final_output
            else:
                logger.warning(f"Optimization failed, keeping initial candidate")
                optimized_candidate = current_candidate
            
            # Evaluate optimized candidate
            eval_result = self._evaluate_candidate(optimized_candidate, problem, condition)
            
            step_result = StepResult(
                step=1,
                reward=eval_result['reward'],
                accuracy=eval_result['accuracy'],
                description=optimized_candidate.description if hasattr(optimized_candidate, 'description') else "",
                code=optimized_candidate.code,
                execution_success=eval_result['execution_success']
            )
            step_results.append(step_result)
            
            # Update best
            if step_result.accuracy > best_accuracy:
                best_accuracy = step_result.accuracy
                best_step = 1
                best_candidate = optimized_candidate
                
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            # Continue with initial candidate
            step_result = StepResult(
                step=1,
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
    
    def _evaluate_candidate(
        self,
        candidate: BARCOutput,
        problem: ARCProblem,
        condition: ExperimentCondition
    ) -> Dict:
        """Evaluate candidate and calculate reward"""
        
        # Execute code on all training examples
        exec_result = self.code_executor.execute(candidate.code, problem)
        
        execution_success = exec_result.success
        train_outputs = exec_result.output_grids if exec_result.success else []
        
        if not execution_success or len(train_outputs) != len(problem.train_pairs):
            return {
                'reward': -1.0,
                'accuracy': 0.0,
                'execution_success': False
            }
        
        # Compute base accuracy
        accuracy = self.arc_evaluator.evaluate_outputs(
            problem.train_pairs, train_outputs
        )
        
        # Compute reward based on condition
        if condition.use_multitensor:
            # Use multitensor evaluation
            multitensor_result = self.multitensor_evaluator.evaluate_multitensor(
                problem, train_outputs
            )
            reward = convert_multitensor_to_reward(multitensor_result)
            logger.info(f"Multitensor scores: {multitensor_result.to_dict()}")
        else:
            # Use GLM evaluation
            glm_result = self.glm_evaluator.evaluate_barc_output(
                problem, candidate, train_outputs
            )
            
            # Convert to reward
            passed_checks = sum([
                glm_result.understanding_check,
                glm_result.calculation_check,
                glm_result.answer_completeness,
                glm_result.answer_correct
            ])
            reward = -1.0 + (passed_checks * 0.2)  # Scale to [-1.0, -0.2]
            
            logger.info(
                f"GLM checks - Understanding: {glm_result.understanding_check}, "
                f"Calculation: {glm_result.calculation_check}, "
                f"Completeness: {glm_result.answer_completeness}, "
                f"Correct: {glm_result.answer_correct}"
            )
        
        return {
            'reward': reward,
            'accuracy': accuracy,
            'execution_success': execution_success
        }
    
    def _create_visualization(
        self,
        problem: ARCProblem,
        best_candidate: BARCOutput,
        condition: ExperimentCondition,
        candidate_idx: int
    ):
        """Create visualization of best results"""
        try:
            # Execute best candidate on all examples
            exec_result = self.code_executor.execute(best_candidate.code, problem)
            outputs = exec_result.output_grids if exec_result.success else [None] * len(problem.train_pairs)
            
            # Create visualization grid
            from src.evaluators.glm_evaluator import create_arc_grid_image
            
            vis_path = self.results_dir / f"{condition.name}_{problem.uid}_c{candidate_idx}_vis.png"
            
            # Create side-by-side visualization
            import numpy as np
            from PIL import Image, ImageDraw, ImageFont
            
            # Calculate dimensions
            num_examples = len(problem.train_pairs)
            grid_size = 300
            padding = 20
            
            # Create large image for all examples
            img_width = 3 * grid_size + 4 * padding  # input, expected, generated
            img_height = num_examples * (grid_size + padding) + padding + 50  # extra for header
            
            img = Image.new('RGB', (img_width, img_height), 'white')
            draw = ImageDraw.Draw(img)
            
            # Add headers
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((padding, 10), "Input", fill='black', font=font)
            draw.text((grid_size + 2*padding, 10), "Expected", fill='black', font=font)
            draw.text((2*grid_size + 3*padding, 10), "Generated", fill='red', font=font)
            
            # Add each example
            for i, (pair, output) in enumerate(zip(problem.train_pairs, outputs)):
                y_offset = i * (grid_size + padding) + 50 + padding
                
                # Input
                input_img = create_arc_grid_image(pair.x, size=grid_size)
                img.paste(input_img, (padding, y_offset))
                
                # Expected
                expected_img = create_arc_grid_image(pair.y, size=grid_size)
                img.paste(expected_img, (grid_size + 2*padding, y_offset))
                
                # Generated
                if output is not None:
                    gen_img = create_arc_grid_image(output, size=grid_size)
                    img.paste(gen_img, (2*grid_size + 3*padding, y_offset))
                else:
                    # Draw error box
                    error_box = Image.new('RGB', (grid_size, grid_size), 'lightgray')
                    error_draw = ImageDraw.Draw(error_box)
                    error_draw.text((10, grid_size//2), "EXECUTION ERROR", fill='red')
                    img.paste(error_box, (2*grid_size + 3*padding, y_offset))
            
            # Save
            img.save(vis_path)
            logger.info(f"Saved visualization to {vis_path}")
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Run simple LatentSeek experiments')
    parser.add_argument('gpu_id', type=int, help='GPU ID to use')
    parser.add_argument('--problems', type=int, default=5, help='Number of problems to test')
    args = parser.parse_args()
    
    # Define experiment conditions
    conditions = [
        ExperimentCondition(
            name="glm_reward",
            use_multitensor=False
        ),
        ExperimentCondition(
            name="multitensor_reward",
            use_multitensor=True
        ),
    ]
    
    # Create experiment runner
    experiment = SimpleLatentSeekExperiment(
        gpu_id=args.gpu_id,
        num_candidates=2
    )
    
    # Select subset of validation problems
    val_problems = experiment.val_problems[:args.problems]
    logger.info(f"Using {len(val_problems)} validation problems: {[p.uid for p in val_problems]}")
    
    # Run experiments
    experiment.run_experiment(val_problems, conditions)
    
    logger.info("\n" + "="*60)
    logger.info("Experiment completed!")
    logger.info("="*60)


if __name__ == "__main__":
    main()