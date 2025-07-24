#!/usr/bin/env python3
"""
Pre-Main Content V9 Experiment
Optimizes concepts + description (everything before 'def main')
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
from src.evaluators.multitensor_evaluator import MultiTensorEvaluator
from src.optimizers.latent_optimizer_premain_v9 import PreMainOptimizerV9

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_premain_v9_experiment(
    num_problems: int = 10,
    device: str = "cuda",
    model_name: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    output_dir: str = "results/premain_v9"
):
    """Run Pre-Main V9 experiment"""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    logger.info("Initializing components...")
    generator = BARCGeneratorFixed(model_name, device=device)
    executor = CodeExecutor()
    renderer = GridRenderer()
    evaluator = MultiTensorEvaluator()
    
    # Initialize V9 optimizer
    optimizer = PreMainOptimizerV9(
        barc_generator=generator,
        code_executor=executor,
        lr=0.005,  # Lower learning rate
        max_steps=30,
        # 5D weights
        accuracy_weight=0.3,
        color_weight=0.2,
        spatial_weight=0.2,
        pattern_weight=0.15,
        structure_weight=0.15,
        kl_weight=0.01,  # Lower KL weight
        convergence_threshold=0.01,
        update_ratio=0.8  # Update 80% of pre-main content
    )
    
    # Load problems
    data_loader = ARCDataLoader()
    problems = data_loader.get_problems(split="validation", num_problems=num_problems)
    
    # Results tracking
    results = {
        "experiment": "premain_v9",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "optimizer_config": {
            "lr": 0.005,
            "max_steps": 30,
            "update_ratio": 0.3,
            "weights": {
                "accuracy": 0.3,
                "color": 0.2,
                "spatial": 0.2,
                "pattern": 0.15,
                "structure": 0.15
            },
            "kl_weight": 0.01
        },
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
            problem_result["initial_concepts"] = initial_output.concepts
            problem_result["initial_description"] = initial_output.description
            
            logger.info(f"Initial concepts: {initial_output.concepts}")
            logger.info(f"Initial description: {initial_output.description[:200]}...")
            
            # Execute initial solution
            initial_result = executor.execute(initial_output.code, problem)
            initial_accuracy = initial_result.accuracy
            problem_result["initial_accuracy"] = initial_accuracy
            total_initial_accuracy += initial_accuracy
            
            # Evaluate initial solution
            initial_eval = evaluator.evaluate(problem, initial_output, initial_result)
            initial_reward = initial_eval.total_reward
            problem_result["initial_reward"] = initial_reward
            problem_result["initial_metrics"] = initial_eval.component_scores
            
            logger.info(f"Initial accuracy: {initial_accuracy:.1%}")
            logger.info(f"Initial reward: {initial_reward:.3f}")
            
            # Save initial code
            initial_code_path = output_path / f"{problem.uid}_initial_code.py"
            with open(initial_code_path, 'w') as f:
                f.write(initial_output.code)
            
            # Run V9 optimization
            logger.info("\nRunning Pre-Main V9 optimization...")
            opt_result = optimizer.optimize(problem, initial_output, initial_accuracy)
            
            problem_result["optimization_steps"] = opt_result.optimization_steps
            problem_result["converged"] = opt_result.converged
            problem_result["loss_history"] = opt_result.loss_history
            problem_result["accuracy_history"] = opt_result.accuracy_history
            problem_result["dimension_losses"] = {
                dim: [float(x) for x in losses] 
                for dim, losses in opt_result.dimension_losses.items()
            }
            
            # Evaluate final solution
            final_output = opt_result.final_output
            final_result = executor.execute(final_output.code, problem)
            final_accuracy = final_result.accuracy
            problem_result["final_accuracy"] = final_accuracy
            total_final_accuracy += final_accuracy
            
            problem_result["final_concepts"] = final_output.concepts
            problem_result["final_description"] = final_output.description
            logger.info(f"Final concepts: {final_output.concepts}")
            logger.info(f"Final description: {final_output.description[:200]}...")
            
            final_eval = evaluator.evaluate(problem, final_output, final_result)
            problem_result["final_reward"] = final_eval.total_reward
            problem_result["final_metrics"] = final_eval.component_scores
            
            logger.info(f"Final accuracy: {final_accuracy:.1%}")
            logger.info(f"Final reward: {final_eval.total_reward:.3f}")
            logger.info(f"Improvement: {(final_accuracy - initial_accuracy):.1%}")
            
            # Save visualization
            viz_path = output_path / f"{problem.uid}_comparison.png"
            renderer.render_problem_with_output(
                problem,
                final_result.output_grids if final_result.success else [],
                str(viz_path)
            )
            problem_result["visualization"] = str(viz_path)
            
            # Save final code
            code_path = output_path / f"{problem.uid}_final_code.py"
            with open(code_path, 'w') as f:
                f.write(final_output.code)
            problem_result["final_code_path"] = str(code_path)
            
            # Save pre-main evolution
            premain_path = output_path / f"{problem.uid}_premain_evolution.txt"
            with open(premain_path, 'w') as f:
                f.write("Pre-Main Content Evolution:\n")
                f.write("="*50 + "\n\n")
                for idx, premain in enumerate(opt_result.premain_history):
                    f.write(f"Step {idx}:\n")
                    f.write("-"*30 + "\n")
                    f.write(premain)
                    f.write("\n\n")
            
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
    logger.info("V9 EXPERIMENT COMPLETE")
    logger.info(f"Average initial accuracy: {avg_initial:.1%}")
    logger.info(f"Average final accuracy: {avg_final:.1%}")
    logger.info(f"Average improvement: {(avg_final - avg_initial):.1%}")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    # Set CUDA device
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    
    run_premain_v9_experiment(
        num_problems=5,  # Start with 5 problems
        device="cuda"
    )