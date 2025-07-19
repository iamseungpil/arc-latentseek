#!/usr/bin/env python3
"""
Full ARC-LatentSeek pipeline experiment on all 400 validation problems.
This experiment runs the complete pipeline:
1. BARC parallel generation
2. Llama alignment
3. GLM rendering and reward
4. Latent seek optimization
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
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from arc-py
from arc import train_problems, validation_problems

# Import ARC-LatentSeek components
from src.main import ARCLatentSeekPipeline, PipelineConfig
from src.data import ARCDataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("full_pipeline_experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Experiment configuration
DEFAULT_CONFIG = {
    "num_candidates": 5,
    "enable_alignment": True,
    "enable_latent_seek": True,
    "latent_seek_threshold": 0.7,
    "latent_seek_max_steps": 10,
    "batch_size": 10,
    "use_wandb": True,
    "wandb_project": "arc-latentseek-full",
    "output_dir": "results/full_pipeline",
    "save_visualizations": True,
    "timeout": 30,
    "max_generation_tokens": 2048,
    "temperature": 0.8
}

class FullPipelineExperiment:
    """Full pipeline experiment runner"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pipeline = None
        self.results = []
        self.start_time = None
        
        # Create output directory
        os.makedirs(config["output_dir"], exist_ok=True)
        
    def setup(self):
        """Initialize pipeline"""
        logger.info("🚀 Setting up ARC-LatentSeek pipeline...")
        
        # Configure pipeline
        pipeline_config = PipelineConfig(
            barc_model="barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
            glm_model="THUDM/GLM-4.1V-9B-Thinking",
            alignment_model="meta-llama/Llama-3.1-8B-Instruct",
            num_candidates=self.config["num_candidates"],
            enable_code_alignment=self.config["enable_alignment"],
            optimization_threshold=self.config["latent_seek_threshold"],
            optimization_steps=self.config["latent_seek_max_steps"],
            use_description_based_optimization=self.config["enable_latent_seek"],
            output_dir=self.config["output_dir"],
            save_visualizations=self.config["save_visualizations"],
            execution_timeout=self.config["timeout"],
            max_new_tokens=self.config["max_generation_tokens"],
            temperature=self.config["temperature"]
        )
        
        # Initialize pipeline
        self.pipeline = ARCLatentSeekPipeline(pipeline_config)
        
        # Initialize WandB if enabled
        if self.config["use_wandb"]:
            wandb.init(
                project=self.config["wandb_project"],
                name=f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config,
                tags=["full-pipeline", "barc", "alignment", "glm", "latent-seek"]
            )
            
        logger.info("✅ Pipeline setup completed!")
        
    def process_problem(self, problem) -> Dict:
        """Process a single problem through the pipeline"""
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"🔄 Processing problem {problem.uid}")
            logger.info(f"{'='*80}")
            
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
            start_time = time.time()
            result = self.pipeline.solve_problem(problem_data)
            elapsed_time = time.time() - start_time
            
            # Add timing information
            result["time"] = elapsed_time
            
            # Log detailed results
            success = result.get("success", False)
            accuracy = result.get("best_accuracy", 0.0)
            reward = result.get("best_reward", 0.0)
            latent_improved = result.get("latent_seek_improved", False)
            
            logger.info(f"📊 Results for problem {problem.uid}:")
            logger.info(f"   - Success: {'✅' if success else '❌'}")
            logger.info(f"   - Best accuracy: {accuracy:.2%}")
            logger.info(f"   - Best reward: {reward:.3f}")
            logger.info(f"   - Latent seek improved: {'Yes' if latent_improved else 'No'}")
            logger.info(f"   - Time: {elapsed_time:.2f}s")
            
            # Log pipeline details
            if "pipeline_details" in result:
                details = result["pipeline_details"]
                logger.info(f"   - Candidates generated: {details.get('num_candidates', 0)}")
                logger.info(f"   - Aligned candidates: {details.get('num_aligned', 0)}")
                logger.info(f"   - Latent seek steps: {details.get('latent_steps', 0)}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error processing problem {problem.uid}: {e}")
            traceback.print_exc()
            return {
                "uid": problem.uid,
                "success": False,
                "error": str(e),
                "time": time.time() - start_time if 'start_time' in locals() else 0
            }
            
    def run_experiment(self, problems: List):
        """Run experiment on all problems"""
        self.start_time = time.time()
        logger.info(f"🚀 Starting experiment on {len(problems)} problems...")
        logger.info(f"📋 Pipeline configuration:")
        logger.info(f"   - BARC candidates: {self.config['num_candidates']}")
        logger.info(f"   - Alignment: {'Enabled' if self.config['enable_alignment'] else 'Disabled'}")
        logger.info(f"   - Latent seek: {'Enabled' if self.config['enable_latent_seek'] else 'Disabled'}")
        logger.info(f"   - Output directory: {self.config['output_dir']}")
        
        successful_count = 0
        total_time = 0
        
        # Process problems in batches
        batch_size = self.config["batch_size"]
        for i in range(0, len(problems), batch_size):
            batch = problems[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(problems) + batch_size - 1)//batch_size
            
            logger.info(f"\n{'='*80}")
            logger.info(f"📦 Processing batch {batch_num}/{total_batches}")
            logger.info(f"{'='*80}")
            
            for j, problem in enumerate(batch):
                problem_num = i + j + 1
                logger.info(f"\n[{problem_num}/{len(problems)}] Processing problem {problem.uid}")
                
                # Process problem
                result = self.process_problem(problem)
                self.results.append(result)
                
                # Update statistics
                if result.get("success", False):
                    successful_count += 1
                total_time += result.get("time", 0)
                
                # Calculate current metrics
                current_accuracy = successful_count / len(self.results)
                avg_time = total_time / len(self.results)
                
                # Log progress
                logger.info(f"\n📊 Progress Update:")
                logger.info(f"   - Problems processed: {len(self.results)}/{len(problems)}")
                logger.info(f"   - Current accuracy: {current_accuracy:.2%} ({successful_count}/{len(self.results)})")
                logger.info(f"   - Average time: {avg_time:.2f}s")
                logger.info(f"   - Estimated remaining: {(len(problems) - len(self.results)) * avg_time / 60:.1f} minutes")
                
                # WandB logging
                if self.config["use_wandb"] and wandb.run:
                    wandb.log({
                        "problems_processed": len(self.results),
                        "current_accuracy": current_accuracy,
                        "successful_problems": successful_count,
                        "average_time_per_problem": avg_time,
                        "last_problem_success": int(result.get("success", False)),
                        "last_problem_accuracy": result.get("best_accuracy", 0.0),
                        "last_problem_reward": result.get("best_reward", 0.0),
                        "last_problem_time": result.get("time", 0)
                    })
                    
                # Save intermediate results every 50 problems
                if len(self.results) % 50 == 0:
                    self.save_intermediate_results()
                    
        # Final summary
        self.print_summary()
        self.save_results()
        
        # Final WandB logging
        if self.config["use_wandb"] and wandb.run:
            wandb.log({
                "final_accuracy": successful_count / len(self.results) if self.results else 0,
                "total_problems": len(self.results),
                "successful_problems": successful_count,
                "total_time_hours": (time.time() - self.start_time) / 3600,
                "experiment_completed": True
            })
            wandb.finish()
            
    def print_summary(self):
        """Print experiment summary"""
        if not self.results:
            logger.warning("No results to summarize")
            return
            
        total_time = time.time() - self.start_time
        successful = sum(1 for r in self.results if r.get("success", False))
        accuracy = successful / len(self.results)
        avg_problem_time = sum(r.get("time", 0) for r in self.results) / len(self.results)
        
        # Calculate component statistics
        alignment_improved = sum(1 for r in self.results if r.get("alignment_improved", False))
        latent_improved = sum(1 for r in self.results if r.get("latent_seek_improved", False))
        
        logger.info("\n" + "="*80)
        logger.info("📊 EXPERIMENT SUMMARY")
        logger.info("="*80)
        logger.info(f"Total problems: {len(self.results)}")
        logger.info(f"Successful solutions: {successful}")
        logger.info(f"Final accuracy: {accuracy:.2%}")
        logger.info(f"Total time: {total_time/3600:.2f} hours")
        logger.info(f"Average time per problem: {avg_problem_time:.2f}s")
        logger.info(f"\nComponent effectiveness:")
        logger.info(f"   - Problems improved by alignment: {alignment_improved} ({alignment_improved/len(self.results):.1%})")
        logger.info(f"   - Problems improved by latent seek: {latent_improved} ({latent_improved/len(self.results):.1%})")
        logger.info("="*80)
        
    def save_intermediate_results(self):
        """Save intermediate results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = os.path.join(
            self.config["output_dir"], 
            f"intermediate_results_{len(self.results)}_problems_{timestamp}.jsonl"
        )
        
        with open(intermediate_file, "w") as f:
            for result in self.results:
                f.write(json.dumps(result) + "\n")
                
        logger.info(f"💾 Saved intermediate results to {intermediate_file}")
        
    def save_results(self):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = os.path.join(self.config["output_dir"], f"results_{timestamp}.jsonl")
        with open(results_file, "w") as f:
            for result in self.results:
                f.write(json.dumps(result) + "\n")
                
        # Calculate statistics
        successful = sum(1 for r in self.results if r.get("success", False))
        accuracy = successful / len(self.results) if self.results else 0
        total_time = time.time() - self.start_time
        
        # Save summary
        summary = {
            "timestamp": timestamp,
            "config": self.config,
            "total_problems": len(self.results),
            "successful": successful,
            "accuracy": accuracy,
            "total_time_hours": total_time / 3600,
            "average_time_per_problem": sum(r.get("time", 0) for r in self.results) / len(self.results) if self.results else 0,
            "component_stats": {
                "alignment_improved": sum(1 for r in self.results if r.get("alignment_improved", False)),
                "latent_seek_improved": sum(1 for r in self.results if r.get("latent_seek_improved", False)),
                "problems_with_errors": sum(1 for r in self.results if "error" in r)
            },
            "pipeline_flow": "BARC parallel → Llama alignment → GLM rendering → Latent seek"
        }
        
        summary_file = os.path.join(self.config["output_dir"], f"summary_{timestamp}.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"📁 Results saved to {results_file}")
        logger.info(f"📁 Summary saved to {summary_file}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run full ARC-LatentSeek pipeline experiment")
    parser.add_argument("--num_problems", type=int, default=400,
                        help="Number of problems to evaluate (default: 400)")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting index for problems (default: 0)")
    parser.add_argument("--num_candidates", type=int, default=5,
                        help="Number of BARC candidates to generate (default: 5)")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size for processing (default: 10)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable WandB logging")
    parser.add_argument("--no_alignment", action="store_true",
                        help="Disable Llama alignment")
    parser.add_argument("--no_latent_seek", action="store_true",
                        help="Disable latent seek optimization")
    parser.add_argument("--output_dir", type=str, default="results/full_pipeline",
                        help="Output directory for results")
    
    args = parser.parse_args()
    
    # Update configuration
    config = DEFAULT_CONFIG.copy()
    config["num_candidates"] = args.num_candidates
    config["batch_size"] = args.batch_size
    config["use_wandb"] = not args.no_wandb
    config["enable_alignment"] = not args.no_alignment
    config["enable_latent_seek"] = not args.no_latent_seek
    config["output_dir"] = args.output_dir
    
    # Load problems
    logger.info("📚 Loading ARC validation problems...")
    problems = validation_problems[args.start_idx:args.start_idx + args.num_problems]
    logger.info(f"📊 Loaded {len(problems)} problems (index {args.start_idx} to {args.start_idx + len(problems) - 1})")
    
    # Initialize and run experiment
    experiment = FullPipelineExperiment(config)
    
    try:
        # Setup
        experiment.setup()
        
        # Run experiment
        experiment.run_experiment(problems)
        
        logger.info("🎉 Experiment completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error in experiment: {e}")
        traceback.print_exc()
        
        # Save any partial results
        if experiment.results:
            experiment.save_results()
            
        return 1
        
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)