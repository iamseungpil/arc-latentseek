#!/usr/bin/env python3
"""Test shape mismatch handling"""

import numpy as np
from src.data import ARCProblem, ARCPair
from src.executors import CodeExecutor

# Create simple test case
train_pairs = [
    ARCPair(
        x=np.array([[0, 5], [5, 0]]),
        y=np.array([[1, 2, 1, 2], [2, 1, 2, 1], [1, 2, 1, 2], [2, 1, 2, 1]])  # 4x4 expected
    )
]

problem = ARCProblem(
    uid="test_shape",
    train_pairs=train_pairs,
    test_pairs=[]
)

# Code that returns wrong shape (2x2 instead of 4x4)
test_code = """
from common import *

def main(input_grid):
    # This returns wrong shape - should trigger shape mismatch
    return input_grid  # Returns 2x2 instead of 4x4
"""

executor = CodeExecutor()
result = executor.execute(test_code, problem)

print(f"Execution success: {result.success}")
print(f"Accuracy: {result.accuracy}")
print(f"Error messages: {result.error_messages}")
print(f"Comparison results: {result.comparison_results}")

if result.output_grids:
    print(f"\nOutput shape: {result.output_grids[0].shape}")
    print(f"Expected shape: {train_pairs[0].y.shape}")