#!/usr/bin/env python3
"""Run single problem test with detailed logging"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
from src.main import main

if __name__ == "__main__":
    # Override default config for single problem test
    import sys
    sys.argv = [
        "run_single_test.py",
        "--mode", "solve_single",
        "--problem_id", "2072aba6",  # The problem we've been testing
        "--num_candidates", "3",
        "--glm_model", "THUDM/GLM-4.1V-9B-Thinking",
        "--max_latent_steps", "5",  # Reduce for faster testing
        "--output_dir", "results/single_test"
    ]
    
    main()