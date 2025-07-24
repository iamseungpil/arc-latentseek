"""
Comparative experiment with loss-based optimization
Replaces policy gradient with direct loss optimization inspired by CompressARC
"""

import os
import sys
import json
import time
import torch
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data import load_arc_dataset, ARCProblem, BARCOutput
from src.generators.barc_generator import BARCGenerator
from src.evaluators.arc_evaluator import ARCEvaluator
from src.evaluators.glm_evaluator import GLMEvaluator
from src.optimizers.loss_based_optimizer import MultiTensorLossOptimizer
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
    use_glm_description: bool
    use_loss_based: bool  # New: use loss-based optimization
    use_multitensor_loss: bool  # New: use multi-tensor loss components


@dataclass 
class StepResult:
    """Result from a single optimization step"""
    step: int
    loss: float  # Changed from reward to loss
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
    final_loss: float  # Changed from reward
    total_time: float
    best_step: int
    best_accuracy: float


class LossBasedComparativeExperiment:
    """Run comparative experiments with loss-based optimization"""
    
    def __init__(self, gpu_id: int, num_candidates: int = 2):
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.num_candidates = num_candidates
        
        # Create results directory
        self.results_dir = Path(f"loss_based_results_gpu{gpu_id}")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize components
        logger.info("Initializing components...")
        self.barc_generator = BARCGenerator(device=self.device)
        self.arc_evaluator = ARCEvaluator()
        self.glm_evaluator = GLMEvaluator()
        self.code_executor = CodeExecutor()
        
        # Initialize loss-based optimizer
        self.loss_optimizer = MultiTensorLossOptimizer(
            model=self.barc_generator.model,
            tokenizer=self.barc_generator.tokenizer,
            lr=0.01,
            max_steps=20,
            kl_weight=0.1
        )
        
        # Load dataset
        self.train_problems, self.val_problems = load_arc_dataset()
        logger.info(f"Loaded {len(self.train_problems)} train, {len(self.val_problems)} val problems")
        
        # Cache for GLM descriptions
        self.glm_descriptions = {}
    
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
                            f"accuracy={result.final_accuracy:.1f}%, loss={result.final_loss:.4f}"
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
        """Run single experiment with loss-based optimization"""
        
        start_time = time.time()
        problem_id = problem.uid
        
        # Generate initial candidates
        logger.info(f"Generating {self.num_candidates} candidates...")
        candidates = self._generate_candidates(problem, condition)
        
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
            loss=initial_eval['loss'],
            accuracy=initial_eval['accuracy'],
            description=initial_candidate.description or "",
            code=initial_candidate.code,
            execution_success=initial_eval['execution_success']
        )]
        
        best_accuracy = initial_eval['accuracy']
        best_step = 0
        
        # Optimization loop
        max_steps = 1  # Just one optimization step for now
        
        for step in range(1, max_steps + 1):
            logger.info(f"\nStep {step}: Applying loss-based optimization...")
            
            try:
                if condition.use_loss_based:
                    # Get expected outputs for loss computation
                    expected_outputs = [pair.y for pair in problem.train_pairs]
                    
                    # Apply loss-based optimization
                    opt_result = self.loss_optimizer.optimize_with_loss(
                        problem=problem,
                        initial_output=current_candidate,
                        target_outputs=expected_outputs
                    )
                    
                    if opt_result.success and opt_result.final_output:
                        optimized_candidate = opt_result.final_output
                        optimization_loss = opt_result.final_loss
                    else:
                        logger.warning(f"Optimization failed at step {step}")
                        optimized_candidate = current_candidate
                        optimization_loss = float('inf')
                else:
                    # Fallback to no optimization
                    optimized_candidate = current_candidate
                    optimization_loss = initial_eval['loss']
                
                # Evaluate optimized candidate
                eval_result = self._evaluate_candidate(optimized_candidate, problem, condition)
                
                step_result = StepResult(
                    step=step,
                    loss=optimization_loss,
                    accuracy=eval_result['accuracy'],
                    description=optimized_candidate.description if hasattr(optimized_candidate, 'description') else "",
                    code=optimized_candidate.code,
                    execution_success=eval_result['execution_success']
                )
                step_results.append(step_result)
                
                # Update best
                if step_result.accuracy > best_accuracy:
                    best_accuracy = step_result.accuracy
                    best_step = step
                
                # Update current candidate
                current_candidate = optimized_candidate
                
                # Early stopping if 100% accuracy
                if step_result.accuracy >= 100.0:
                    logger.info(f"🎯 Achieved 100% accuracy at step {step}! Early stopping.")
                    break
                    
            except Exception as e:
                logger.error(f"Error in optimization step {step}: {e}")
                # Continue with previous candidate
                step_result = StepResult(
                    step=step,
                    loss=step_results[-1].loss,
                    accuracy=step_results[-1].accuracy,
                    description=current_candidate.description if hasattr(current_candidate, 'description') else "",
                    code=current_candidate.code,
                    execution_success=False
                )
                step_results.append(step_result)
        
        total_time = time.time() - start_time
        
        return ExperimentResult(
            condition=condition.name,
            problem_id=problem_id,
            candidate_id=candidate_idx,
            optimization_steps=step_results,
            final_accuracy=step_results[-1].accuracy,
            final_loss=step_results[-1].loss,
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
            # Generate GLM description if not cached
            if problem.uid not in self.glm_descriptions:
                self.glm_descriptions[problem.uid] = self._generate_glm_description(problem)
            
            glm_desc = self.glm_descriptions[problem.uid]
            
            if glm_desc:
                logger.info(f"Using GLM description: {glm_desc[:100]}...")
                additional_prompt = f"\nHint about the pattern: {glm_desc}\n"
            else:
                logger.warning(f"Failed to generate GLM description for {problem.uid}")
                additional_prompt = ""
        else:
            additional_prompt = ""
        
        # Generate candidates
        candidates = self.barc_generator.generate_batch(
            problem, 
            n=self.num_candidates,
            additional_prompt=additional_prompt
        )
        
        return candidates
    
    def _evaluate_candidate(
        self,
        candidate: BARCOutput,
        problem: ARCProblem,
        condition: ExperimentCondition
    ) -> Dict:
        """Evaluate candidate and compute loss"""
        
        # Execute code on training examples
        train_outputs = []
        execution_success = True
        
        for train_pair in problem.train_pairs:
            result = self.code_executor.execute(candidate.code, train_pair.x)
            if result['success'] and result['output'] is not None:
                train_outputs.append(result['output'])
            else:
                execution_success = False
                break
        
        if not execution_success or len(train_outputs) != len(problem.train_pairs):
            return {
                'loss': 1.0,  # Max loss
                'accuracy': 0.0,
                'execution_success': False
            }
        
        # Compute accuracy
        accuracy = self.arc_evaluator.evaluate_outputs(
            problem.train_pairs, train_outputs
        )
        
        # Compute loss
        if condition.use_multitensor_loss:
            # Use multi-tensor loss computation
            expected_outputs = [pair.y for pair in problem.train_pairs]
            loss_tensor, loss_components = self.loss_optimizer.compute_multitensor_loss(
                train_outputs, expected_outputs
            )
            loss = loss_tensor.item()
            logger.info(f"Multi-tensor loss components: {loss_components}")
        else:
            # Simple accuracy-based loss
            loss = 1.0 - accuracy / 100.0
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'execution_success': execution_success
        }
    
    def _generate_glm_description(self, problem: ARCProblem) -> Optional[str]:
        """Generate pattern description using GLM"""
        try:
            description = self.glm_evaluator.generate_pattern_description(problem)
            return description
        except Exception as e:
            logger.error(f"Failed to generate GLM description: {e}")
            return None


def main():
    """Main entry point"""
    import argparse
    parser = argparse.ArgumentParser(description='Run loss-based comparative experiments')
    parser.add_argument('gpu_id', type=int, help='GPU ID to use (5 or 6)')
    args = parser.parse_args()
    
    # Define experiment conditions based on GPU
    if args.gpu_id == 5:
        conditions = [
            ExperimentCondition(
                name="loss_basic",
                use_glm_description=False,
                use_loss_based=True,
                use_multitensor_loss=False
            ),
            ExperimentCondition(
                name="loss_multitensor", 
                use_glm_description=False,
                use_loss_based=True,
                use_multitensor_loss=True
            ),
        ]
    elif args.gpu_id == 6:
        conditions = [
            ExperimentCondition(
                name="glm_loss_basic",
                use_glm_description=True,
                use_loss_based=True,
                use_multitensor_loss=False
            ),
            ExperimentCondition(
                name="glm_loss_multitensor",
                use_glm_description=True,
                use_loss_based=True,
                use_multitensor_loss=True
            ),
        ]
    else:
        raise ValueError(f"Invalid GPU ID: {args.gpu_id}")
    
    # Create experiment runner
    experiment = LossBasedComparativeExperiment(
        gpu_id=args.gpu_id,
        num_candidates=2
    )
    
    # Select subset of validation problems
    val_problems = experiment.val_problems[:5]
    logger.info(f"Using {len(val_problems)} validation problems: {[p.uid for p in val_problems]}")
    
    # Run experiments
    experiment.run_experiment(val_problems, conditions)


if __name__ == "__main__":
    main()