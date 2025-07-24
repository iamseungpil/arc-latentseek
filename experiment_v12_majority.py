#!/usr/bin/env python3
"""
Experiment with V12 Majority Voting Optimizer
- Uses proper policy gradient with sampling
- 8 candidate inference with majority voting
- 5D reward based on most frequent output
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
from src.generators.barc_generator_simple import BARCGeneratorSimple
from src.optimizers.latent_optimizer_v12_majority import LatentOptimizerV12Majority

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('experiment_v12_majority.log')
    ]
)
logger = logging.getLogger(__name__)

def run_v12_experiment(
    model_name: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
    num_problems: int = 5,
    device: str = "cuda:0"
):
    """Run V12 majority voting experiment."""
    
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
    
    # Initialize evaluator and generator
    evaluator = SimpleEvaluator()
    generator = BARCGeneratorSimple(model, tokenizer)
    
    # Initialize V12 optimizer with majority voting
    optimizer = LatentOptimizerV12Majority(
        model=model,
        tokenizer=tokenizer,
        evaluator=evaluator,
        num_candidates=8,
        learning_rate=0.01,
        num_steps=30,
        warmup_steps=5,
        temperature=1.0
    )
    
    # Load validation problems directly from arc
    val_problems = {p.uid: p for p in arc.validation_problems}
    problem_ids = list(val_problems.keys())[:num_problems]
    
    # Results storage
    results = {
        "experiment": "v12_majority_voting",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": model_name,
            "num_problems": num_problems,
            "num_candidates": 8,
            "num_steps": 30,
            "warmup_steps": 5
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
        
        # Generate initial solution
        logger.info("Generating initial solution...")
        initial_result = generator.generate(problem_id)
        initial_code = initial_result["code"]
        
        # Evaluate initial solution
        eval_result = evaluator.evaluate_solution(problem_id, initial_code)
        initial_accuracy = eval_result.get("accuracy", 0.0)
        initial_reward = eval_result.get("reward", 0.0)
        
        logger.info(f"Initial accuracy: {initial_accuracy:.1%}")
        logger.info(f"Initial 5D reward: {initial_reward:.3f}")
        
        # Run V12 optimization
        logger.info("\nRunning V12 Majority Voting optimization...")
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
            code_path = Path(f"results/v12_majority/{problem_id}_final.py")
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
        with open("results/v12_majority/results.json", 'w') as f:
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
        logger.info("V12 MAJORITY VOTING EXPERIMENT COMPLETE")
        logger.info(f"Average initial accuracy: {avg_initial:.1%}")
        logger.info(f"Average final accuracy: {avg_final:.1%}")
        logger.info(f"Average improvement: {avg_improvement:.1%}")
        logger.info(f"Results saved to: results/v12_majority")
        logger.info(f"{'='*80}")
        
    # Save final results
    with open("results/v12_majority/results.json", 'w') as f:
        json.dump(results, f, indent=2)
        
    return results

if __name__ == "__main__":
    # Run experiment on GPU5
    run_v12_experiment(device="cuda:0")