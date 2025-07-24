#!/usr/bin/env python3
"""
Experiment with V14 - Context-aware optimization
"""

import os
import sys
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

# Import arc directly
import arc

# Now import our modules
from src.evaluators.simple_evaluator import SimpleEvaluator
from src.optimizers.latent_optimizer_v14_context_aware import LatentOptimizerV14ContextAware

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment_v14.log')
    ]
)
logger = logging.getLogger(__name__)

# Pre-generated real code samples
REAL_CODE_SAMPLES = {
    "2a5f8217": """# concepts:
# color mapping, object detection, color replacement

# description:
# In the input, you will see a grid containing several objects of different colors. 
# Each object is defined by a connected region of pixels of the same color. 
# To make the output, change the color of each object to match the color of the object directly below it. 
# If there is no object below, the color remains unchanged.

def main(input_grid):
    # Copy the input grid to the output grid
    output_grid = np.copy(input_grid)

    # Get the objects in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK)

    # For each object, change its color to the color of the object below it
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)

        # Check if there is an object directly below the bounding box
        if y + height < output_grid.shape[1]:  # Ensure we don't go out of bounds
            below_color = output_grid[x:x+width, y + height].max()  # Get the color of the pixels directly below

            # Change the color of the current object to the color of the object below
            output_grid[obj == output_grid[x, y]] = below_color

    return output_grid""",
}

def run_v14_experiment(
    model_name: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    num_problems: int = 1,
    device: str = "cuda:0"
):
    """Run V14 experiment with context-aware optimization."""
    
    # Initialize components
    logger.info("Initializing components...")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Initialize evaluator
    evaluator = SimpleEvaluator()
    
    # Initialize V14 optimizer
    optimizer = LatentOptimizerV14ContextAware(
        model=model,
        tokenizer=tokenizer,
        evaluator=evaluator,
        num_candidates=8,
        learning_rate=0.001,
        num_steps=20,
        warmup_steps=3,
        temperature=0.7
    )
    
    # Load validation problems directly from arc
    val_problems = {p.uid: p for p in arc.validation_problems}
    
    # Use only problems with real code
    problem_ids = list(REAL_CODE_SAMPLES.keys())[:num_problems]
    
    # Results storage
    results = {
        "experiment": "v14_context_aware",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_name,
            "num_problems": len(problem_ids),
            "num_candidates": 8,
            "num_steps": 20,
            "warmup_steps": 3,
            "learning_rate": 0.001,
            "temperature": 0.7
        },
        "problems": []
    }
    
    # Process each problem
    for i, problem_id in enumerate(problem_ids):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing problem {i+1}/{len(problem_ids)}: {problem_id}")
        logger.info(f"{'='*80}")
        
        problem = val_problems[problem_id]
        target_outputs = [test_pair.y for test_pair in problem.test_pairs]
        
        # Use real pre-generated code
        initial_code = REAL_CODE_SAMPLES[problem_id]
        
        # Log initial code structure
        lines = initial_code.split('\n')
        desc_lines = [line for line in lines if line.strip().startswith('#') and 'concepts:' not in line]
        logger.info(f"Initial code: {len(lines)} lines, {len(desc_lines)} description lines")
        
        # Evaluate initial solution
        logger.info("Evaluating initial real code...")
        eval_result = evaluator.evaluate_solution(problem_id, initial_code)
        initial_accuracy = eval_result.get("accuracy", 0.0)
        
        logger.info(f"Initial accuracy: {initial_accuracy:.1%}")
        
        # Run V14 optimization
        logger.info("\nRunning V14 context-aware optimization...")
        opt_result = optimizer.optimize(problem_id, initial_code, target_outputs)
        
        if opt_result["success"]:
            final_accuracy = opt_result["final_accuracy"]
            improvement = final_accuracy - initial_accuracy
            
            logger.info(f"Final accuracy: {final_accuracy:.1%}")
            logger.info(f"Improvement: {improvement:+.1%}")
            
            # Save results
            problem_result = {
                "uid": problem_id,
                "initial_accuracy": initial_accuracy,
                "final_accuracy": final_accuracy,
                "improvement": improvement,
                "num_steps": opt_result["num_steps"],
                "history": opt_result["history"],
                "status": "completed"
            }
            
            # Save final code
            code_path = Path(f"results/v14/{problem_id}_final.py")
            code_path.parent.mkdir(parents=True, exist_ok=True)
            with open(code_path, 'w') as f:
                f.write(opt_result["final_code"])
                
        else:
            logger.error(f"Optimization failed: {opt_result.get('error', 'Unknown error')}")
            problem_result = {
                "uid": problem_id,
                "initial_accuracy": initial_accuracy,
                "final_accuracy": initial_accuracy,
                "improvement": 0.0,
                "status": "failed",
                "error": opt_result.get("error", "Unknown error")
            }
            
        results["problems"].append(problem_result)
        
        # Save intermediate results
        with open("results/v14/results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
    # Calculate summary statistics
    completed = [p for p in results["problems"] if p["status"] == "completed"]
    if completed:
        avg_initial = sum(p["initial_accuracy"] for p in completed) / len(completed)
        avg_final = sum(p["final_accuracy"] for p in completed) / len(completed)
        avg_improvement = sum(p["improvement"] for p in completed) / len(completed)
        
        results["summary"] = {
            "total_problems": len(problem_ids),
            "completed_problems": len(completed),
            "average_initial_accuracy": avg_initial,
            "average_final_accuracy": avg_final,
            "average_improvement": avg_improvement
        }
        
        logger.info(f"\n{'='*80}")
        logger.info("V14 EXPERIMENT COMPLETE")
        logger.info(f"Average initial accuracy: {avg_initial:.1%}")
        logger.info(f"Average final accuracy: {avg_final:.1%}")
        logger.info(f"Average improvement: {avg_improvement:.1%}")
        logger.info(f"Results saved to: results/v14")
        logger.info(f"{'='*80}")
        
    # Save final results
    with open("results/v14/results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    # Run experiment on GPU5
    run_v14_experiment(device="cuda:0")