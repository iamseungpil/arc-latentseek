#!/usr/bin/env python3
"""Test single problem execution with shape mismatch"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import sys
sys.argv = [
    "test_single_problem.py",
    "--mode", "solve",
    "--problems", "validation", 
    "--num_problems", "1",
    "--num_candidates", "1",
    "--output_dir", "results/test_single"
]

# Import after setting argv
from src.main import main

print("\n" + "="*80)
print("Testing Single Problem with Fixed Executor")
print("="*80)
print("\nThis test will:")
print("1. Generate BARC code for 1 validation problem")
print("2. Execute the code (may have shape mismatch)")
print("3. If execution succeeds (no runtime errors), proceed to GLM")
print("4. Apply LatentSeek optimization if accuracy < 1.0")
print("\nStarting...\n")

main()