#!/usr/bin/env python3
"""
Integrated experiment: BARC Training-Time RL + ARC-LatentSeek Pipeline
This experiment combines BARC's training-time RL approach with the full ARC-LatentSeek pipeline.
"""

import os
import sys
import json
import time
import torch
import wandb
import logging
import traceback
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/home/ubuntu/barc_post')

# Import from arc-py
from arc import train_problems, validation_problems

# Import ARC-LatentSeek components
from src.main import ARCLatentSeekPipeline, PipelineConfig
from src.data_loader import ARCDataLoader
from src.generators.barc_generator import BARCGenerator
from src.alignment.code_aligner import BARCCodeAligner
from src.evaluators.glm_evaluator import GLMEvaluator
from src.evaluators.reward_model import RewardModel
from src.optimizers.latent_optimizer import LatentSeekOptimizer
from src.executors.code_executor import CodeExecutor
from src.executors.grid_renderer import GridRenderer

# Import BARC training-time RL components
from long_with_logit_reward2 import (
    ARCTrainingTimeRLTrainer,
    setup_model as barc_setup_model,
    init_wandb,
    CHECKPOINT_DIR,
    TRAINING_STEPS,
    MODEL_NAME
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("barc_integrated_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global configuration
EXPERIMENT_CONFIG = {
    "barc_training_steps": 50,  # Reduced for initial testing
    "barc_num_candidates": 8,
    "pipeline_num_candidates": 5,
    "enable_alignment": True,
    "enable_latent_seek": True,
    "max_problems": 400,  # All validation problems
    "batch_size": 10,  # Process problems in batches
    "use_wandb": True,
    "wandb_project": "arc-barc-integrated",
    "output_dir": "results/barc_integrated"
}

class IntegratedBARCPipeline:
    """Integrated pipeline combining BARC Training-Time RL with ARC-LatentSeek"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.barc_trainer = None
        self.pipeline = None
        self.results = []
        
        # Create output directory
        os.makedirs(config["output_dir"], exist_ok=True)
        
    def setup(self):
        """Initialize all components"""
        logger.info("🚀 Setting up integrated BARC-LatentSeek pipeline...")
        
        # 1. Set up BARC trainer with training-time RL
        logger.info("📦 Loading BARC model for training-time RL...")
        model, tokenizer = barc_setup_model()
        self.barc_trainer = ARCTrainingTimeRLTrainer(model, tokenizer)
        
        # 2. Set up ARC-LatentSeek pipeline
        logger.info("🔧 Configuring ARC-LatentSeek pipeline...")
        pipeline_config = PipelineConfig(
            barc_model_path=MODEL_NAME,
            glm_model_path="thudm/glm-4v-9b",
            alignment_model_path="meta-llama/Llama-3.1-8B-Instruct",
            num_candidates=self.config["pipeline_num_candidates"],
            enable_alignment=self.config["enable_alignment"],
            latent_seek_threshold=0.7,
            latent_seek_max_steps=10,
            enable_latent_seek=self.config["enable_latent_seek"],
            output_dir=self.config["output_dir"],
            save_visualizations=True
        )
        self.pipeline = ARCLatentSeekPipeline(pipeline_config)
        
        logger.info("✅ Setup completed successfully!")
        
    def train_barc_on_problems(self, problems: List):
        """Train BARC model using training-time RL on a subset of problems"""
        logger.info(f"🎯 Training BARC on {len(problems)} problems with training-time RL...")
        
        try:
            # Run training
            success = self.barc_trainer.train(
                problems, 
                total_steps=self.config["barc_training_steps"]
            )
            
            if success:
                logger.info("✅ BARC training completed successfully!")
                return True
            else:
                logger.error("❌ BARC training failed!")
                return False
                
        except Exception as e:
            logger.error(f"Error during BARC training: {e}")
            traceback.print_exc()
            return False
            
    def process_problem_with_pipeline(self, problem) -> Dict:
        """Process a single problem through the full pipeline"""
        try:
            logger.info(f"🔄 Processing problem {problem.uid} through pipeline...")
            
            # Convert arc-py problem to pipeline format
            problem_data = {
                "uid": problem.uid,
                "train_pairs": [
                    {"x": pair.x.tolist(), "y": pair.y.tolist()} 
                    for pair in problem.train_pairs
                ],
                "test_pairs": [
                    {"x": pair.x.tolist(), "y": pair.y.tolist()} 
                    for pair in problem.test_pairs
                ]
            }
            
            # Run through pipeline
            result = self.pipeline.run(problem_data)
            
            # Check success
            success = result.get("success", False)
            if success:
                logger.info(f"✅ Problem {problem.uid} solved successfully!")
            else:
                logger.info(f"❌ Problem {problem.uid} failed to solve")
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing problem {problem.uid}: {e}")
            traceback.print_exc()
            return {
                "uid": problem.uid,
                "success": False,
                "error": str(e)
            }
            
    def run_experiment(self, problems: List):
        """Run the full integrated experiment"""
        logger.info(f"🚀 Starting integrated experiment on {len(problems)} problems...")
        
        # Initialize WandB if enabled
        if self.config["use_wandb"]:
            wandb_run = init_wandb(
                project_name=self.config["wandb_project"],
                run_name=f"barc_integrated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                tags=["barc-ttrl", "arc-latentseek", "integrated", "full-pipeline"]
            )
        
        # Split problems for training and evaluation
        train_size = min(50, len(problems) // 2)  # Use 50 problems for training
        train_problems = problems[:train_size]
        eval_problems = problems[train_size:]
        
        logger.info(f"📊 Split: {len(train_problems)} for BARC training, {len(eval_problems)} for evaluation")
        
        # Phase 1: Train BARC with training-time RL
        logger.info("=" * 80)
        logger.info("🎯 PHASE 1: BARC Training-Time RL")
        logger.info("=" * 80)
        
        training_success = self.train_barc_on_problems(train_problems)
        if not training_success:
            logger.error("Training failed, aborting experiment")
            return
            
        # Phase 2: Evaluate with full pipeline
        logger.info("=" * 80)
        logger.info("🔍 PHASE 2: Full Pipeline Evaluation")
        logger.info("=" * 80)
        
        successful_count = 0
        total_time = 0
        
        # Process in batches
        batch_size = self.config["batch_size"]
        for i in range(0, len(eval_problems), batch_size):
            batch = eval_problems[i:i+batch_size]
            logger.info(f"📦 Processing batch {i//batch_size + 1}/{(len(eval_problems) + batch_size - 1)//batch_size}")
            
            for problem in batch:
                start_time = time.time()
                result = self.process_problem_with_pipeline(problem)
                elapsed_time = time.time() - start_time
                
                result["time"] = elapsed_time
                self.results.append(result)
                
                if result.get("success", False):
                    successful_count += 1
                    
                total_time += elapsed_time
                
                # Log progress
                current_accuracy = successful_count / len(self.results)
                logger.info(f"Progress: {len(self.results)}/{len(eval_problems)} | "
                          f"Accuracy: {current_accuracy:.2%} | "
                          f"Avg time: {total_time/len(self.results):.2f}s")
                
                # WandB logging
                if self.config["use_wandb"] and wandb.run:
                    wandb.log({
                        "problems_processed": len(self.results),
                        "current_accuracy": current_accuracy,
                        "successful_problems": successful_count,
                        "average_time_per_problem": total_time/len(self.results),
                        "last_problem_success": int(result.get("success", False))
                    })
                    
        # Final summary
        final_accuracy = successful_count / len(self.results) if self.results else 0
        logger.info("=" * 80)
        logger.info("📊 EXPERIMENT SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Total problems evaluated: {len(self.results)}")
        logger.info(f"Successful solutions: {successful_count}")
        logger.info(f"Final accuracy: {final_accuracy:.2%}")
        logger.info(f"Average time per problem: {total_time/len(self.results):.2f}s")
        
        # Save results
        self.save_results()
        
        # Final WandB logging
        if self.config["use_wandb"] and wandb.run:
            wandb.log({
                "final_accuracy": final_accuracy,
                "total_problems": len(self.results),
                "successful_problems": successful_count,
                "total_time": total_time,
                "experiment_completed": True
            })
            wandb.finish()
            
    def save_results(self):
        """Save experiment results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.config["output_dir"], f"results_{timestamp}.jsonl")
        with open(results_file, "w") as f:
            for result in self.results:
                f.write(json.dumps(result) + "\n")
                
        # Save summary
        summary = {
            "timestamp": timestamp,
            "config": self.config,
            "total_problems": len(self.results),
            "successful": sum(1 for r in self.results if r.get("success", False)),
            "accuracy": sum(1 for r in self.results if r.get("success", False)) / len(self.results) if self.results else 0,
            "average_time": sum(r.get("time", 0) for r in self.results) / len(self.results) if self.results else 0,
            "pipeline_components": {
                "barc_training": "training-time-rl",
                "barc_generation": "parallel",
                "alignment": "llama-3.1-8b",
                "evaluation": "glm-4v-9b",
                "optimization": "latent-seek"
            }
        }
        
        summary_file = os.path.join(self.config["output_dir"], f"summary_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"📁 Results saved to {results_file}")
        logger.info(f"📁 Summary saved to {summary_file}")

def main():
    """Main function to run the integrated experiment"""
    # Load all validation problems
    logger.info("📚 Loading ARC validation problems...")
    problems = validation_problems[:EXPERIMENT_CONFIG["max_problems"]]
    logger.info(f"📊 Loaded {len(problems)} problems")
    
    # Initialize and run pipeline
    pipeline = IntegratedBARCPipeline(EXPERIMENT_CONFIG)
    
    try:
        # Setup components
        pipeline.setup()
        
        # Run experiment
        pipeline.run_experiment(problems)
        
        logger.info("🎉 Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in experiment: {e}")
        traceback.print_exc()
        
        # Save any partial results
        if pipeline.results:
            pipeline.save_results()
            
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)