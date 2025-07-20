#!/usr/bin/env python3
"""
Run remaining problems with improved settings
"""

import subprocess
import sys

# Get remaining problem IDs
processed_ids = {
    '136b0064', '2072aba6', '40f6cd08', '7039b2d7',
    '712bf12e', 'bb52a14b', 'ea9794b1', 'f5aa3634'
}

# Run with improved settings
cmd = [
    sys.executable, "-m", "src.main",
    "--problems", "validation",
    "--num_candidates", "4",  # Reduced from 8
    "--output_dir", "results_improved",
    "--num_problems", "392"  # Only remaining problems
]

print(f"Running command: {' '.join(cmd)}")
print(f"Skipping already processed problems: {processed_ids}")

# Note: We'll need to modify main.py to skip processed problems
# For now, let's run all 400 with reduced candidates
cmd = [
    sys.executable, "-m", "src.main",
    "--problems", "validation", 
    "--num_candidates", "4",
    "--output_dir", "results_improved",
    "--num_problems", "400"
]

print(f"\nRunning full validation with 4 candidates per problem...")
subprocess.run(cmd)