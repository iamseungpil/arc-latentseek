#!/usr/bin/env python3
"""
ARC-LatentSeek Experiment Runner
Run the full pipeline on 400 validation problems with all enhancements
"""

import os
import sys
import logging
from src.main import ARCLatentSeekPipeline, PipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run the experiment"""
    logger.info("="*60)
    logger.info("Starting ARC-LatentSeek Experiment with Unsloth Optimization")
    logger.info("="*60)
    
    # Set CUDA device to GPU 6
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    logger.info("Using GPU 6 for experiment")
    
    # Create enhanced configuration
    config = PipelineConfig(
        # Models (with unsloth optimization)
        barc_model="barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
        glm_model="THUDM/GLM-4.1V-9B-Thinking",
        alignment_model="meta-llama/Llama-3.1-8B-Instruct",
        
        # Generation settings
        num_candidates=8,
        temperature=0.8,
        max_new_tokens=2048,
        
        # Execution settings
        execution_timeout=2,
        
        # Optimization settings
        optimization_steps=10,
        optimization_threshold=1.0,  # Binary: only stop at perfect score
        use_description_based_optimization=True,
        
        # Alignment settings
        enable_code_alignment=True,
        alignment_temperature=0.3,
        alignment_max_tokens=2048,
        min_alignment_score=20,
        
        # Output settings
        output_dir="results",
        save_visualizations=True
    )
    
    logger.info("Configuration:")
    logger.info(f"- BARC Model: {config.barc_model}")
    logger.info(f"- GLM Model: {config.glm_model}")
    logger.info(f"- Alignment Model: {config.alignment_model}")
    logger.info(f"- Candidates: {config.num_candidates}")
    logger.info(f"- Optimization Steps: {config.optimization_steps}")
    logger.info(f"- Alignment Enabled: {config.enable_code_alignment}")
    logger.info(f"- Description-based Optimization: {config.use_description_based_optimization}")
    
    try:
        # Create pipeline
        logger.info("Initializing pipeline...")
        pipeline = ARCLatentSeekPipeline(config)
        
        # Run on all 400 validation problems
        logger.info("Starting experiment on validation set...")
        results = pipeline.solve_problems(
            split="validation",
            num_problems=400  # All validation problems
        )
        
        # Print final summary
        successful = sum(1 for r in results if r.success)
        initial_successes = sum(1 for r in results if r.initial_success)
        improved_by_latentseek = sum(1 for r in results if r.improved_by_latentseek)
        
        logger.info("="*60)
        logger.info("FINAL EXPERIMENT RESULTS")
        logger.info("="*60)
        logger.info(f"Total Problems: {len(results)}")
        logger.info(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
        logger.info(f"Initial Successes: {initial_successes} ({initial_successes/len(results)*100:.1f}%)")
        logger.info(f"Improved by LatentSeek: {improved_by_latentseek} ({improved_by_latentseek/len(results)*100:.1f}%)")
        
        avg_accuracy = sum(r.execution_accuracy for r in results) / len(results)
        avg_initial = sum(r.initial_best_accuracy for r in results) / len(results)
        avg_final = sum(r.final_best_accuracy for r in results) / len(results)
        
        logger.info(f"Average Accuracy: {avg_accuracy:.2%}")
        logger.info(f"Average Initial Accuracy: {avg_initial:.2%}")
        logger.info(f"Average Final Accuracy: {avg_final:.2%}")
        logger.info(f"Improvement: {avg_final - avg_initial:.2%}")
        logger.info("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()