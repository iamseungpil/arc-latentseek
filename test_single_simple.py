#!/usr/bin/env python3
"""Simple test of single problem"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/data/.cache/huggingface"

import json
from arc import validation_problems
import numpy as np

# Get first validation problem directly
problem = validation_problems[0]
problem_id = problem.get('uid', 'unknown')

print(f"Problem ID: {problem_id}")
print(f"Number of train pairs: {len(problem['train'])}")
for i, pair in enumerate(problem['train']):
    input_grid = np.array(pair['input'])
    output_grid = np.array(pair['output'])
    print(f"Train pair {i}: input shape {input_grid.shape}, output shape {output_grid.shape}")
    
# Show grids
print("\nFirst training pair:")
print("Input:")
print(np.array(problem['train'][0]['input']))
print("\nOutput:")
print(np.array(problem['train'][0]['output']))