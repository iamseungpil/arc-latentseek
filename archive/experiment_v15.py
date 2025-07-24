#!/usr/bin/env python3
"""
Experiment with V15 - Proper BARC prompt format with LatentSeek optimization
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
from src.optimizers.latent_optimizer_v15_proper_prompt import LatentOptimizerV15ProperPrompt

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment_v15.log')
    ]
)
logger = logging.getLogger(__name__)

def run_v15_experiment(
    model_name: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    num_problems: int = 5,
    device: str = "cuda:5"
):
    """Run V15 experiment with proper BARC prompt format."""
    
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
    
    # Initialize V15 optimizer
    optimizer = LatentOptimizerV15ProperPrompt(
        model=model,
        tokenizer=tokenizer,
        evaluator=evaluator,
        k=0.2,  # Optimize 20% of tokens
        learning_rate=0.01,
        num_steps=50,
        temperature=1.0,
        max_new_tokens=1024
    )
    
    # Load validation problems directly from arc
    val_problems = {p.uid: p for p in arc.validation_problems}
    problem_ids = list(val_problems.keys())[:num_problems]
    
    # Results storage
    results = {
        "experiment": "v15_proper_barc_prompt",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_name,
            "num_problems": num_problems,
            "k": 0.2,
            "num_steps": 50,
            "learning_rate": 0.01,
            "temperature": 1.0
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
        
        # Run V15 optimization
        logger.info("Running V15 optimization with proper BARC prompt...")
        opt_result = optimizer.optimize(problem_id, target_outputs)
        
        if opt_result["success"]:
            logger.info(f"Initial accuracy: {opt_result.get('initial_accuracy', 0.0):.1%}")
            logger.info(f"Final accuracy: {opt_result['final_accuracy']:.1%}")
            logger.info(f"Improvement: {opt_result.get('improvement', 0.0):+.1%}")
            
            # Save results
            problem_result = {
                "uid": problem_id,
                "initial_accuracy": opt_result.get("initial_accuracy", 0.0),
                "final_accuracy": opt_result["final_accuracy"],
                "improvement": opt_result.get("improvement", 0.0),
                "num_steps": opt_result["num_steps"],
                "history": opt_result["history"],
                "status": "completed"
            }
            
            # Save final code
            if opt_result["final_code"]:
                code_path = Path(f"results/v15/{problem_id}_final.py")
                code_path.parent.mkdir(parents=True, exist_ok=True)
                with open(code_path, 'w') as f:
                    f.write(opt_result["final_code"])
                
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
        with open("results/v15/results.json", 'w') as f:
            json.dump(results, f, indent=2)
            
    # Calculate summary statistics
    completed = [p for p in results["problems"] if p["status"] == "completed"]
    if completed:
        avg_initial = sum(p["initial_accuracy"] for p in completed) / len(completed)
        avg_final = sum(p["final_accuracy"] for p in completed) / len(completed)
        avg_improvement = sum(p["improvement"] for p in completed) / len(completed)
        
        results["summary"] = {
            "total_problems": num_problems,
            "completed_problems": len(completed),
            "average_initial_accuracy": avg_initial,
            "average_final_accuracy": avg_final,
            "average_improvement": avg_improvement
        }
        
        logger.info(f"\n{'='*80}")
        logger.info("V15 EXPERIMENT COMPLETE")
        logger.info(f"Average initial accuracy: {avg_initial:.1%}")
        logger.info(f"Average final accuracy: {avg_final:.1%}")
        logger.info(f"Average improvement: {avg_improvement:.1%}")
        logger.info(f"Results saved to: results/v15")
        logger.info(f"{'='*80}")
        
    # Save final results
    with open("results/v15/results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    # Run experiment on GPU 5
    run_v15_experiment(device="cuda:5")