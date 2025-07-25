#!/usr/bin/env python3
"""
GLM Reward-based LatentSeek Experiment
Uses GLM-4V visual evaluation for optimization
"""

import sys
from pathlib import Path
import torch
import json
from datetime import datetime
import logging

# Setup paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader
from src.generators.barc_generator_fixed import BARCGeneratorFixed
from src.executors import CodeExecutor
from src.executors.grid_renderer import GridRenderer
from src.evaluators.glm_evaluator import GLMEvaluator
from src.optimizers.latent_optimizer_fixed_v2 import LatentSeekOptimizerV2

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_glm_experiment(
    num_problems: int = 10,
    device: str = "cuda",
    model_name: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    output_dir: str = "results/glm"
):
    """Run GLM-based LatentSeek experiment"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    generator = BARCGeneratorFixed(model_name, device=device)
    executor = CodeExecutor()
    renderer = GridRenderer()
    
    # Initialize GLM evaluator
    evaluator = GLMEvaluator()
    
    # Initialize optimizer with GLM evaluation
    optimizer = LatentSeekOptimizerV2(
        barc_generator=generator,
        code_executor=executor,
        evaluator=evaluator,  # GLMEvaluator
        lr=0.03,
        max_steps=20,
        k=0.2,
        reward_threshold=0.95,  # Higher threshold to allow more optimization steps
        use_policy_gradient=True  # Use policy gradient for GLM
    )
    
    # Load problems
    data_loader = ARCDataLoader()
    problems = data_loader.get_problems(split="validation", num_problems=num_problems)
    
    # Results tracking
    results = {
        "experiment": "glm_latentseek",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "problems": []
    }
    
    total_initial_accuracy = 0
    total_final_accuracy = 0
    
    # Process each problem
    for i, problem in enumerate(problems):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing problem {i+1}/{num_problems}: {problem.uid}")
        logger.info(f"{'='*80}")
        
        problem_result = {
            "uid": problem.uid,
            "train_examples": len(problem.train_pairs),
        }
        
        try:
            # Generate initial solution
            logger.info("Generating initial solution...")
            candidates = generator.generate(problem, num_candidates=1, temperature=0.7)
            
            if not candidates or not candidates[0].code:
                logger.warning("Failed to generate initial solution")
                problem_result["status"] = "generation_failed"
                results["problems"].append(problem_result)
                continue
            
            initial_output = candidates[0]
            problem_result["initial_description"] = initial_output.description
            problem_result["initial_concepts"] = initial_output.concepts
            
            # Execute initial solution
            initial_result = executor.execute(initial_output.code, problem)
            initial_accuracy = initial_result.accuracy
            problem_result["initial_accuracy"] = initial_accuracy
            total_initial_accuracy += initial_accuracy
            
            # Evaluate initial solution with GLM
            initial_eval = evaluator.evaluate(
                problem, 
                initial_output, 
                initial_result,
                base_path=f"{problem.uid}_initial"
            )
            initial_reward = initial_eval.total_reward
            problem_result["initial_reward"] = initial_reward
            problem_result["initial_glm_analysis"] = {
                "component_scores": initial_eval.component_scores,
                "verifications": {k: v.passed for k, v in initial_eval.verifications.items()},
                "feedback": initial_eval.detailed_feedback
            }
            
            logger.info(f"Initial accuracy: {initial_accuracy:.1%}")
            logger.info(f"Initial GLM reward: {initial_reward:.3f}")
            
            # Run optimization
            logger.info("\nRunning LatentSeek optimization with GLM feedback...")
            opt_result = optimizer.optimize(problem, initial_output, initial_reward)
            
            problem_result["optimization_steps"] = opt_result.optimization_steps
            problem_result["converged"] = opt_result.converged
            problem_result["reward_history"] = opt_result.reward_history
            
            # Evaluate final solution
            final_output = opt_result.final_output
            final_result = executor.execute(final_output.code, problem)
            final_accuracy = final_result.accuracy
            problem_result["final_accuracy"] = final_accuracy
            total_final_accuracy += final_accuracy
            
            final_eval = evaluator.evaluate(
                problem, 
                final_output, 
                final_result,
                base_path=f"{problem.uid}_final"
            )
            problem_result["final_reward"] = final_eval.total_reward
            problem_result["final_glm_analysis"] = {
                "component_scores": final_eval.component_scores,
                "verifications": {k: v.passed for k, v in final_eval.verifications.items()},
                "feedback": final_eval.detailed_feedback
            }
            
            logger.info(f"Final accuracy: {final_accuracy:.1%}")
            logger.info(f"Final GLM reward: {final_eval.total_reward:.3f}")
            logger.info(f"Improvement: {(final_accuracy - initial_accuracy):.1%}")
            
            # Save comparison visualization
            viz_path = output_path / f"{problem.uid}_comparison.png"
            renderer.render_problem_with_output(
                problem,
                final_result.output_grids if final_result.success else [],
                str(viz_path)
            )
            problem_result["visualization"] = str(viz_path)
            
            # Save initial vs final code
            code_comparison_path = output_path / f"{problem.uid}_code_comparison.txt"
            with open(code_comparison_path, 'w') as f:
                f.write("=== INITIAL CODE ===\n")
                f.write(initial_output.code)
                f.write("\n\n=== FINAL CODE ===\n")
                f.write(final_output.code)
            problem_result["code_comparison_path"] = str(code_comparison_path)
            
            problem_result["status"] = "completed"
            
        except Exception as e:
            logger.error(f"Error processing problem {problem.uid}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            problem_result["status"] = "error"
            problem_result["error"] = str(e)
        
        results["problems"].append(problem_result)
        
        # Save intermediate results
        with open(output_path / "results.json", 'w') as f:
            json.dump(results, f, indent=2)
    
    # Calculate summary statistics
    avg_initial = total_initial_accuracy / len(problems) if problems else 0
    avg_final = total_final_accuracy / len(problems) if problems else 0
    
    results["summary"] = {
        "total_problems": len(problems),
        "average_initial_accuracy": avg_initial,
        "average_final_accuracy": avg_final,
        "average_improvement": avg_final - avg_initial
    }
    
    # Save final results
    with open(output_path / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*80}")
    logger.info("EXPERIMENT COMPLETE")
    logger.info(f"Average initial accuracy: {avg_initial:.1%}")
    logger.info(f"Average final accuracy: {avg_final:.1%}")
    logger.info(f"Average improvement: {(avg_final - avg_initial):.1%}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    # Set CUDA device
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    
    run_glm_experiment(
        num_problems=5,  # Start with 5 problems for testing
        device="cuda"
    )