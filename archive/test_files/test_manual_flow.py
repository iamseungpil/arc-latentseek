#!/usr/bin/env python3
"""Manually test the flow with shape mismatch"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from src.data import ARCProblem, ARCPair
from src.executors import CodeExecutor
from src.generators import BARCOutput

# Create test problem
train_pairs = [
    ARCPair(
        x=np.array([[0, 5, 0], [5, 5, 5], [0, 5, 0]]),
        y=np.array([
            [1, 2, 1, 1, 2, 1],
            [2, 2, 2, 2, 2, 2],
            [1, 2, 1, 1, 2, 1],
            [1, 2, 1, 1, 2, 1],
            [2, 2, 2, 2, 2, 2],
            [1, 2, 1, 1, 2, 1]
        ])
    )
]

problem = ARCProblem(uid="test", train_pairs=train_pairs, test_pairs=[])

# Create a BARC output with wrong shape code
barc_code = """
from common import *

def main(input_grid):
    # This code returns wrong shape (3x3 instead of 6x6)
    # But it runs without errors
    output = np.zeros_like(input_grid)
    output[input_grid == 5] = 2  # GRAY -> RED
    output[input_grid == 0] = 1  # BLACK -> BLUE
    return output
"""

barc_output = BARCOutput(
    code=barc_code,
    description="Transform colors but wrong shape",
    concepts="color transformation",
    plan="Transform colors",
    raw_response=barc_code
)

# Test execution
print("Testing shape mismatch handling...\n")
executor = CodeExecutor()
result = executor.execute(barc_output.code, problem)

print(f"Code execution:")
print(f"- Success: {result.success}")
print(f"- Accuracy: {result.accuracy}")
print(f"- Error messages: {result.error_messages}")
print(f"- Comparison results: {result.comparison_results}")

if result.output_grids:
    print(f"\nOutput shape: {result.output_grids[0].shape}")
    print(f"Expected shape: {train_pairs[0].y.shape}")
    print(f"\nOutput grid:\n{result.output_grids[0]}")

if result.success:
    print("\n✅ SUCCESS: Code executed without runtime errors")
    print("   Even though shape is wrong, this should proceed to GLM evaluation")
    print("   and then LatentSeek optimization can try to fix it")
else:
    print("\n❌ FAILED: This would skip GLM and LatentSeek")
    print("   This is wrong behavior - shape mismatch should not prevent optimization")