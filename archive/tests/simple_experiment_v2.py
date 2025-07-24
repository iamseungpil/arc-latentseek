#!/usr/bin/env python3
"""
Simple experiment runner with properly fixed LatentSeek implementation
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

from src.data import ARCDataLoader, ARCProblem
from src.generators import BARCGenerator, GeneratorConfig
from src.executors import CodeExecutor
from src.evaluators import GLMEvaluator
from src.optimizers.latent_optimizer_fixed import FixedLatentSeekOptimizer

# Configure logging
def setup_logging(gpu_id: int):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'simple_experiment_v2_gpu{gpu_id}_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def run_single_problem(problem_id: str, 
                      problem: ARCProblem,
                      generator: BARCGenerator,
                      executor: CodeExecutor,
                      evaluator: GLMEvaluator,
                      optimizer: FixedLatentSeekOptimizer,
                      num_candidates: int = 2) -> Dict:
    """Run experiment on a single problem"""
    
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"Running experiment on {problem_id}")
    logger.info(f"{'='*60}")
    
    results = []
    
    try:
        # Generate initial candidates
        logger.info(f"Generating {num_candidates} initial candidates...")
        candidates = generator.generate(problem, num_candidates=num_candidates)
        
        for i, candidate in enumerate(candidates):
            logger.info(f"\n--- Processing candidate {i} ---")
            
            # Evaluate initial candidate
            initial_result = evaluator.evaluate_single(problem, candidate.code)
            logger.info(f"Initial evaluation: accuracy={initial_result.accuracy:.1%}, reward={initial_result.reward:.3f}")
            
            # Skip optimization if already perfect
            if initial_result.accuracy >= 1.0:
                logger.info("Already perfect, skipping optimization")
                results.append({
                    "candidate_idx": i,
                    "initial_reward": initial_result.reward,
                    "initial_accuracy": initial_result.accuracy,
                    "final_reward": initial_result.reward,
                    "final_accuracy": initial_result.accuracy,
                    "optimization_steps": 0,
                    "converged": True,
                    "reward_history": [initial_result.reward]
                })
                continue
            
            # Apply fixed LatentSeek optimization
            logger.info("Applying fixed LatentSeek optimization...")
            opt_result = optimizer.optimize(problem, candidate, initial_result.reward)
            
            # Evaluate optimized result
            final_result = evaluator.evaluate_single(problem, opt_result.final_output.code)
            logger.info(f"Final evaluation: accuracy={final_result.accuracy:.1%}, reward={final_result.reward:.3f}")
            logger.info(f"Improvement: {final_result.reward - initial_result.reward:.3f}")
            
            # Store results
            results.append({
                "candidate_idx": i,
                "initial_reward": initial_result.reward,
                "initial_accuracy": initial_result.accuracy,
                "final_reward": final_result.reward,
                "final_accuracy": final_result.accuracy,
                "optimization_steps": opt_result.optimization_steps,
                "converged": opt_result.converged,
                "reward_history": opt_result.reward_history,
                "reward_improvement": final_result.reward - initial_result.reward,
                "accuracy_improvement": final_result.accuracy - initial_result.accuracy
            })
            
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


def run_experiment(problem_ids: List[str], gpu_id: int = 5):
    """Run fixed experiment on specified problems"""
    
    logger = setup_logging(gpu_id)
    
    # Configuration
    config = GeneratorConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        temperature=0.7,
        max_tokens=2048,
        device=f"cuda:{gpu_id}"
    )
    
    # Initialize components
    logger.info(f"Initializing components on GPU {gpu_id}")
    generator = BARCGenerator(config)
    executor = CodeExecutor()
    evaluator = GLMEvaluator(device=config.device)
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
    data_loader = ARCDataLoader()
    problems = data_loader.load_evaluation()
    selected_problems = {pid: prob for pid, prob in problems.items() if pid in problem_ids}
    
    if not selected_problems:
        logger.error(f"No valid problems found in {problem_ids}")
        return
    
    # Create output directory
    os.makedirs(f'fixed_results_v2_gpu{gpu_id}', exist_ok=True)
    
    all_results = {}
    
    # Run experiments
    for problem_id, problem in selected_problems.items():
        result = run_single_problem(
            problem_id, problem, generator, executor, evaluator, optimizer
        )
        all_results[problem_id] = result
        
        # Save intermediate results
        with open(f'fixed_results_v2_gpu{gpu_id}/{problem_id}_result.json', 'w') as f:
            json.dump(result, f, indent=2)
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f'fixed_results_v2_gpu{gpu_id}/all_results_{timestamp}.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    total_candidates = 0
    total_initial_accuracy = 0
    total_final_accuracy = 0
    total_improved = 0
    
    for problem_id, result in all_results.items():
        if 'candidates' in result:
            for candidate in result['candidates']:
                total_candidates += 1
                total_initial_accuracy += candidate['initial_accuracy']
                total_final_accuracy += candidate['final_accuracy']
                if candidate['accuracy_improvement'] > 0:
                    total_improved += 1
    
    if total_candidates > 0:
        avg_initial_accuracy = total_initial_accuracy / total_candidates
        avg_final_accuracy = total_final_accuracy / total_candidates
        
        logger.info(f"Total problems: {len(selected_problems)}")
        logger.info(f"Total candidates: {total_candidates}")
        logger.info(f"Average initial accuracy: {avg_initial_accuracy:.1%}")
        logger.info(f"Average final accuracy: {avg_final_accuracy:.1%}")
        logger.info(f"Average improvement: {avg_final_accuracy - avg_initial_accuracy:.1%}")
        logger.info(f"Candidates improved: {total_improved}/{total_candidates} ({total_improved/total_candidates:.1%})")


if __name__ == "__main__":
    import torch
    
    # Get GPU ID from command line
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    
    # Default problems
    problem_ids = ["2a5f8217", "bf89d739", "feca6190"]
    
    # Parse additional arguments
    if "--problems" in sys.argv:
        idx = sys.argv.index("--problems")
        if idx + 1 < len(sys.argv):
            num_problems = int(sys.argv[idx + 1])
            data_loader = ARCDataLoader()
            all_problems = data_loader.load_evaluation()
            problem_ids = list(all_problems.keys())[:num_problems]
    
    if "--all" in sys.argv:
        data_loader = ARCDataLoader()
        all_problems = data_loader.load_evaluation()
        problem_ids = list(all_problems.keys())[:10]  # First 10 problems
    
    if "--specific" in sys.argv:
        idx = sys.argv.index("--specific")
        if idx + 1 < len(sys.argv):
            problem_ids = sys.argv[idx + 1].split(",")
    
    logger = logging.getLogger(__name__)
    logger.info(f"Running experiment on problems: {problem_ids}")
    
    run_experiment(problem_ids, gpu_id)