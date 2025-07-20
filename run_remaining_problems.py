#!/usr/bin/env python3
"""
Run improved pipeline on remaining problems
"""

import sys
sys.path.append('/home/ubuntu/arc-latentseek')

from src.main_improved import ImprovedARCPipeline, PipelineConfig
from src.data import ARCDataLoader
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('improved_experiment.log'),
        logging.StreamHandler()
    ]
)

# Load remaining problems
loader = ARCDataLoader()
all_problems = loader.get_problems(split="validation")

# Problems already processed
processed_ids = {
    '136b0064', '2072aba6', '40f6cd08', '7039b2d7',
    '712bf12e', 'bb52a14b', 'ea9794b1', 'f5aa3634'
}

# Filter out processed problems
remaining_problems = [p for p in all_problems if p.uid not in processed_ids]

print(f"Starting experiment with {len(remaining_problems)} remaining problems")

# Create config
config = PipelineConfig(
    mode="solve",
    problems_split="validation_remaining",
    num_candidates=4,  # Reduced from 8
    output_dir="results",
    use_code_alignment=False,
    use_description_optimization=True,
    save_visualizations=True,
    device="cuda:0"  # Will use GPU 6 due to CUDA_VISIBLE_DEVICES
)

# Initialize pipeline
pipeline = ImprovedARCPipeline(config)

# Solve remaining problems
results = pipeline.solve_problems(remaining_problems)

print("Experiment completed!")