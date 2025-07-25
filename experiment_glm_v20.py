#!/usr/bin/env python3
"""
Experiment with V20 - GLM-based evaluation + description optimization
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
from src.evaluators.glm_evaluator import GLMEvaluator
from src.optimizers.latent_optimizer_glm_v20 import LatentOptimizerGLMV20

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment_glm_v20.log')
    ]
)
logger = logging.getLogger(__name__)

def run_glm_v20_experiment(
    model_name: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    num_problems: int = 5,
    device: str = "cuda:5"
):
    """Run V20 experiment with GLM-based evaluation."""
    
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
    
    # Initialize GLM evaluator
    evaluator = GLMEvaluator()
    
    # Initialize V20 optimizer
    optimizer = LatentOptimizerGLMV20(
        model=model,
        tokenizer=tokenizer,
        evaluator=evaluator,
        learning_rate=0.03,
        num_steps=20,
        temperature=1.0,
        max_new_tokens=1024,
        grad_clip=1.0,
        max_retries=10
    )
    
    # Load validation problems directly from arc
    val_problems = {p.uid: p for p in arc.validation_problems}
    problem_ids = list(val_problems.keys())[:num_problems]
    
    # Results storage
    results = {
        "experiment": "v20_glm_evaluation",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_name,
            "num_problems": num_problems,
            "num_steps": 20,
            "learning_rate": 0.03,
            "temperature": 1.0,
            "max_retries": 10,
            "evaluation": "GLM-4V visual evaluation"
        },
        "problems": []
    }
    
    # Process each problem
    for i, problem_id in enumerate(problem_ids):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing problem {i+1}/{num_problems}: {problem_id}")
        logger.info(f"{'='*80}")
        
        problem = val_problems[problem_id]
        target_outputs = [test_pair.y for test_pair in problem.test_pairs]
        
        # Run V20 optimization
        logger.info("Running V20 optimization with GLM evaluation...")
        opt_result = optimizer.optimize(problem_id, target_outputs)
        
        if opt_result["success"]:
            logger.info(f"Initial accuracy: {opt_result.get('initial_accuracy', 0.0):.1%}")
            logger.info(f"Final accuracy: {opt_result['final_accuracy']:.1%}")
            logger.info(f"Improvement: {opt_result.get('improvement', 0.0):+.1%}")
            logger.info(f"Description tokens optimized: {opt_result.get('description_tokens', 0)}")
            
            # Log GLM rewards
            logger.info("Initial GLM reward: %.3f", opt_result.get('initial_glm_reward', 0.0))
            logger.info("Final GLM reward: %.3f", opt_result.get('final_glm_reward', 0.0))
            
            # Save results
            problem_result = {
                "uid": problem_id,
                "initial_accuracy": opt_result.get("initial_accuracy", 0.0),
                "final_accuracy": opt_result["final_accuracy"],
                "improvement": opt_result.get("improvement", 0.0),
                "initial_glm_reward": opt_result.get("initial_glm_reward", 0.0),
                "final_glm_reward": opt_result.get("final_glm_reward", 0.0),
                "num_steps": opt_result["num_steps"],
                "description_tokens": opt_result.get("description_tokens", 0),
                "history": opt_result["history"],
                "status": "completed"
            }
            
            # Save final code
            if opt_result["final_code"]:
                code_path = Path(f"results/glm_v20/{problem_id}_final.py")
                code_path.parent.mkdir(parents=True, exist_ok=True)
                with open(code_path, 'w') as f:
                    f.write(opt_result["final_code"])
                    
                # Also save initial code for comparison
                if opt_result.get("initial_code"):
                    initial_code_path = Path(f"results/glm_v20/{problem_id}_initial.py")
                    with open(initial_code_path, 'w') as f:
                        f.write(opt_result.get("initial_code", ""))
                
        else:
            logger.error(f"Optimization failed: {opt_result.get('error', 'Unknown error')}")
            problem_result = {
                "uid": problem_id,
                "initial_accuracy": 0.0,
                "final_accuracy": 0.0,
                "improvement": 0.0,
                "status": "failed",
                "error": opt_result.get("error", "Unknown error")
            }
            
        results["problems"].append(problem_result)
        
        # Save intermediate results
        with open("results/glm_v20/results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
    # Calculate summary statistics
    completed = [p for p in results["problems"] if p["status"] == "completed"]
    if completed:
        avg_initial = sum(p["initial_accuracy"] for p in completed) / len(completed)
        avg_final = sum(p["final_accuracy"] for p in completed) / len(completed)
        avg_improvement = sum(p["improvement"] for p in completed) / len(completed)
        avg_initial_glm = sum(p["initial_glm_reward"] for p in completed) / len(completed)
        avg_final_glm = sum(p["final_glm_reward"] for p in completed) / len(completed)
        
        results["summary"] = {
            "total_problems": num_problems,
            "completed_problems": len(completed),
            "average_initial_accuracy": avg_initial,
            "average_final_accuracy": avg_final,
            "average_improvement": avg_improvement,
            "average_initial_glm_reward": avg_initial_glm,
            "average_final_glm_reward": avg_final_glm,
            "glm_reward_improvement": avg_final_glm - avg_initial_glm
        }
        
        logger.info(f"\n{'='*80}")
        logger.info("V20 GLM EVALUATION EXPERIMENT COMPLETE")
        logger.info(f"Average initial accuracy: {avg_initial:.1%}")
        logger.info(f"Average final accuracy: {avg_final:.1%}")
        logger.info(f"Average improvement: {avg_improvement:.1%}")
        logger.info(f"Average GLM reward improvement: {avg_final_glm - avg_initial_glm:+.3f}")
        logger.info(f"Results saved to: results/glm_v20")
        logger.info(f"{'='*80}")
        
    # Save final results
    with open("results/glm_v20/results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    # Run experiment on GPU 5
    run_glm_v20_experiment(device="cuda:5")