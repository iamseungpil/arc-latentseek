#!/usr/bin/env python3
"""Test pipeline flow to ensure GLM and LatentSeek are triggered with shape mismatch"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # Use GPU 6  

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.data import ARCDataLoader

# Test with just one problem
print("\n" + "="*80)
print("Testing Fixed Pipeline Flow")
print("="*80 + "\n")

# Load the first validation problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems('validation', num_problems=1)
problem = problems[0]

print(f"Problem ID: {problem.uid}")
print(f"Train pairs: {len(problem.train_pairs)}")
print(f"Input shape: {problem.train_pairs[0].x.shape}")
print(f"Output shape: {problem.train_pairs[0].y.shape}")

# Run the main pipeline with just 1 candidate
print("\nRunning pipeline with fixed executor...")
print("Expected flow:")
print("1. BARC generates code")
print("2. Code executes (may have shape mismatch)")
print("3. If no runtime errors → success=True")
print("4. GLM evaluation runs")
print("5. LatentSeek optimization attempts to fix")
print("\nStarting...\n")

# Import and run main with minimal config
import sys
sys.argv = [
    "test_pipeline_flow.py",
    "--mode", "solve",
    "--problems", "validation",
    "--num_problems", "1",
    "--num_candidates", "1",  # Just 1 candidate for testing
    "--max_latent_steps", "3",  # Reduce iterations
    "--output_dir", "results/test_flow"
]

from src.main import main
main()