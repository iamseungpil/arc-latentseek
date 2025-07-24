#!/usr/bin/env python3
"""
Run experiment with fixed LatentSeek optimizer
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import fixed optimizer
sys.path.insert(0, str(project_root / "src" / "optimizers"))
from latent_optimizer_fixed import FixedLatentSeekOptimizer

from src.data import ARCDataLoader, ARCProblem
from src.generators import BARCGenerator, BARCOutput
from src.executors import CodeExecutor
from src.evaluators import GLMEvaluator


# Configure logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f'gpu5_improved_{timestamp}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_experiment_on_problem(problem_id: str, problem: ARCProblem, 
                            generator: BARCGenerator, 
                            executor: CodeExecutor,
                            evaluator: GLMEvaluator,
                            optimizer: FixedLatentSeekOptimizer) -> Dict:
    """Run experiment on a single problem"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment on problem {problem_id}")
    logger.info(f"{'='*60}")
    
    results = {
        "problem_id": problem_id,
        "candidates": []
    }
    
    try:
        # Generate 2 candidates
        logger.info("Generating 2 initial candidates...")
        candidates = generator.generate(problem, num_candidates=2)
        
        for i, candidate in enumerate(candidates):
            logger.info(f"\n--- Candidate {i} ---")
            
            # Evaluate initial candidate
            initial_result = evaluator.evaluate_single(problem, candidate.code)
            logger.info(f"Initial: accuracy={initial_result.accuracy:.1%}, reward={initial_result.reward:.3f}")
            
            # Apply optimization
            logger.info("Applying fixed LatentSeek optimization...")
            opt_result = optimizer.optimize(problem, candidate, initial_result.reward)
            
            # Evaluate optimized result
            final_result = evaluator.evaluate_single(problem, opt_result.final_output.code)
            logger.info(f"Final: accuracy={final_result.accuracy:.1%}, reward={final_result.reward:.3f}")
            logger.info(f"Improvement: {final_result.reward - initial_result.reward:.3f}")
            
            results["candidates"].append({
                "candidate_idx": i,
                "initial_accuracy": initial_result.accuracy,
                "initial_reward": initial_result.reward,
                "final_accuracy": final_result.accuracy,
                "final_reward": final_result.reward,
                "improvement": final_result.reward - initial_result.reward,
                "optimization_steps": opt_result.optimization_steps,
                "converged": opt_result.converged,
                "reward_history": opt_result.reward_history
            })
            
    except Exception as e:
        logger.error(f"Error on problem {problem_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        results["error"] = str(e)
    
    return results


def main():
    # Configuration
    gpu_id = 0  # When using CUDA_VISIBLE_DEVICES=5, we need to use cuda:0
    device = f"cuda:{gpu_id}"
    
    logger.info(f"Starting experiment on GPU {gpu_id}")
    logger.info("Using fixed LatentSeek optimizer")
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Create config for BARCGenerator
    class Config:
        def __init__(self):
            self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"
            self.device = device
            self.temperature = 0.7
            self.max_tokens = 2048
    
    config = Config()
    
    # Initialize with config
    generator = BARCGenerator(config.model_name)
    generator.model = generator.model.to(device)
    
    executor = CodeExecutor()
    evaluator = GLMEvaluator()
    
    optimizer = FixedLatentSeekOptimizer(
        barc_generator=generator,
        code_executor=executor,
        glm_evaluator=evaluator,
        lr=0.03,
        max_steps=10,
        k=0.2,  # Optimize 20% of generated tokens
        reward_threshold=0.5
    )
    
    # Load problems
    logger.info("Loading ARC problems...")
    data_loader = ARCDataLoader()
    all_problems_list = data_loader.get_problems(split="validation")
    
    # Convert to dict format
    all_problems = {p.uid: p for p in all_problems_list}
    
    # Select problems
    problem_ids = ["2a5f8217", "bf89d739", "feca6190"]
    
    # Create output directory
    output_dir = f"gpu5_improved_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = []
    
    # Run experiments
    for problem_id in problem_ids:
        if problem_id not in all_problems:
            logger.warning(f"Problem {problem_id} not found")
            continue
            
        problem = all_problems[problem_id]
        result = run_experiment_on_problem(
            problem_id, problem, generator, executor, evaluator, optimizer
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
                if candidate["improvement"] > 0:
                    total_improved += 1
    
    if total_candidates > 0:
        avg_initial_acc /= total_candidates
        avg_final_acc /= total_candidates
        
        logger.info(f"Total problems: {len(problem_ids)}")
        logger.info(f"Total candidates: {total_candidates}")
        logger.info(f"Average initial accuracy: {avg_initial_acc:.1%}")
        logger.info(f"Average final accuracy: {avg_final_acc:.1%}")
        logger.info(f"Average improvement: {avg_final_acc - avg_initial_acc:.1%}")
        logger.info(f"Candidates improved: {total_improved}/{total_candidates}")


if __name__ == "__main__":
    main()