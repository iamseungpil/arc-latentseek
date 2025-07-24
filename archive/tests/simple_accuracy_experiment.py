#!/usr/bin/env python3
"""
Simple accuracy-based experiment without GLM dependency
Uses only execution accuracy as reward
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader, ARCProblem
from src.generators import BARCGenerator, BARCOutput
from src.executors import CodeExecutor
from src.optimizers.latent_optimizer_fixed import FixedLatentSeekOptimizer

# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'accuracy_experiment_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def compute_accuracy(expected_outputs: List[np.ndarray], 
                    generated_outputs: List[np.ndarray]) -> float:
    """Compute accuracy between expected and generated outputs"""
    if len(expected_outputs) != len(generated_outputs):
        return 0.0
    
    correct = 0
    for expected, generated in zip(expected_outputs, generated_outputs):
        if generated is not None and np.array_equal(expected, generated):
            correct += 1
    
    return correct / len(expected_outputs) if expected_outputs else 0.0


def run_single_problem(problem_id: str, 
                      problem: ARCProblem,
                      generator: BARCGenerator,
                      executor: CodeExecutor,
                      optimizer: FixedLatentSeekOptimizer,
                      num_candidates: int = 2) -> Dict:
    """Run experiment on a single problem"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment on {problem_id}")
    logger.info(f"{'='*60}")
    
    results = []
    
    try:
        # Generate initial candidates
        logger.info(f"Generating {num_candidates} initial candidates...")
        candidates = generator.generate(problem, num_candidates=num_candidates)
        
        for i, candidate in enumerate(candidates):
            logger.info(f"\n--- Candidate {i} ---")
            
            # Execute code
            execution_result = executor.execute(candidate.code, problem)
            
            # Compute accuracy
            expected_outputs = [pair.y for pair in problem.train_pairs]
            generated_outputs = execution_result.output_grids if execution_result.success else []
            initial_accuracy = compute_accuracy(expected_outputs, generated_outputs)
            
            # Convert accuracy to reward (-1.0 to 0.0 scale)
            initial_reward = -1.0 + initial_accuracy
            
            logger.info(f"Initial: accuracy={initial_accuracy:.1%}, reward={initial_reward:.3f}")
            
            # Skip optimization if already perfect
            if initial_accuracy >= 1.0:
                logger.info("Already perfect, skipping optimization")
                results.append({
                    "candidate_idx": i,
                    "initial_reward": initial_reward,
                    "initial_accuracy": initial_accuracy,
                    "final_reward": initial_reward,
                    "final_accuracy": initial_accuracy,
                    "optimization_steps": 0,
                    "converged": True,
                    "reward_history": [initial_reward]
                })
                continue
            
            # Apply fixed LatentSeek optimization
            logger.info("Applying fixed LatentSeek optimization...")
            
            # Create a simple accuracy-based evaluator wrapper
            class AccuracyEvaluator:
                def __init__(self, executor, problem):
                    self.executor = executor
                    self.problem = problem
                    self.expected_outputs = [pair.y for pair in problem.train_pairs]
                
                def evaluate(self, problem, output, execution_result, prefix=""):
                    """Evaluate based on accuracy only"""
                    accuracy = compute_accuracy(
                        self.expected_outputs,
                        execution_result.output_grids if execution_result.success else []
                    )
                    reward = -1.0 + accuracy
                    
                    from dataclasses import dataclass
                    @dataclass
                    class SimpleResult:
                        total_reward: float
                        
                    return SimpleResult(total_reward=reward)
            
            # Temporarily replace evaluator in optimizer
            original_evaluator = optimizer.glm_evaluator
            optimizer.glm_evaluator = AccuracyEvaluator(executor, problem)
            
            try:
                opt_result = optimizer.optimize(problem, candidate, initial_reward)
                
                # Evaluate optimized result
                final_execution = executor.execute(opt_result.final_output.code, problem)
                final_accuracy = compute_accuracy(expected_outputs, 
                                                final_execution.output_grids if final_execution.success else [])
                final_reward = -1.0 + final_accuracy
                
                logger.info(f"Final: accuracy={final_accuracy:.1%}, reward={final_reward:.3f}")
                logger.info(f"Improvement: accuracy +{(final_accuracy - initial_accuracy)*100:.1f}%, reward +{final_reward - initial_reward:.3f}")
                
                results.append({
                    "candidate_idx": i,
                    "initial_reward": initial_reward,
                    "initial_accuracy": initial_accuracy,
                    "final_reward": final_reward,
                    "final_accuracy": final_accuracy,
                    "optimization_steps": opt_result.optimization_steps,
                    "converged": opt_result.converged,
                    "reward_history": opt_result.reward_history,
                    "reward_improvement": final_reward - initial_reward,
                    "accuracy_improvement": final_accuracy - initial_accuracy
                })
                
            finally:
                # Restore original evaluator
                optimizer.glm_evaluator = original_evaluator
            
    except Exception as e:
        logger.error(f"Failed on problem {problem_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "problem_id": problem_id,
            "error": str(e),
            "candidates": []
        }
    
    return {
        "problem_id": problem_id,
        "candidates": results
    }


def main():
    # Configuration
    gpu_id = 0  # When using CUDA_VISIBLE_DEVICES=5
    device = f"cuda:{gpu_id}"
    
    logger.info(f"Starting accuracy-based experiment on GPU {gpu_id}")
    logger.info("Using fixed LatentSeek optimizer with accuracy-based reward")
    
    # Initialize components
    logger.info("Initializing components...")
    
    # BARCGenerator
    generator = BARCGenerator("Qwen/Qwen2.5-0.5B-Instruct")
    generator.model = generator.model.to(device)
    
    executor = CodeExecutor()
    
    # Create dummy GLM evaluator for optimizer (won't be used)
    class DummyGLMEvaluator:
        pass
    
    optimizer = FixedLatentSeekOptimizer(
        barc_generator=generator,
        code_executor=executor,
        glm_evaluator=DummyGLMEvaluator(),  # Will be replaced
        lr=0.03,
        max_steps=10,
        k=0.2,  # Optimize 20% of generated tokens
        reward_threshold=0.5
    )
    
    # Load problems
    logger.info("Loading ARC problems...")
    data_loader = ARCDataLoader()
    all_problems_list = data_loader.get_problems(split="validation")
    
    # Convert to dict and select specific problems
    all_problems = {p.uid: p for p in all_problems_list}
    
    # Select problems (first 5 from validation set)
    problem_ids = list(all_problems.keys())[:5]
    logger.info(f"Selected problems: {problem_ids}")
    
    # Create output directory
    output_dir = f"accuracy_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # Run experiments
    for problem_id in problem_ids:
        if problem_id not in all_problems:
            logger.warning(f"Problem {problem_id} not found")
            continue
            
        problem = all_problems[problem_id]
        result = run_single_problem(
            problem_id, problem, generator, executor, optimizer
        )
        all_results.append(result)
        
        # Save intermediate results
        with open(f"{output_dir}/{problem_id}_result.json", "w") as f:
            json.dump(result, f, indent=2)
    
    # Save all results
    with open(f"{output_dir}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    total_candidates = 0
    total_improved = 0
    avg_initial_acc = 0
    avg_final_acc = 0
    
    for result in all_results:
        if "candidates" in result:
            for candidate in result["candidates"]:
                total_candidates += 1
                avg_initial_acc += candidate["initial_accuracy"]
                avg_final_acc += candidate["final_accuracy"]
                if candidate.get("accuracy_improvement", 0) > 0:
                    total_improved += 1
    
    if total_candidates > 0:
        avg_initial_acc /= total_candidates
        avg_final_acc /= total_candidates
        
        logger.info(f"Total problems: {len([r for r in all_results if 'candidates' in r])}")
        logger.info(f"Total candidates: {total_candidates}")
        logger.info(f"Average initial accuracy: {avg_initial_acc:.1%}")
        logger.info(f"Average final accuracy: {avg_final_acc:.1%}")
        logger.info(f"Average improvement: {(avg_final_acc - avg_initial_acc)*100:.1f}%")
        logger.info(f"Candidates improved: {total_improved}/{total_candidates} ({total_improved/total_candidates*100:.1f}%)")


if __name__ == "__main__":
    main()