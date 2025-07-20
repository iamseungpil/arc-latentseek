#!/usr/bin/env python3
"""Simple test to verify shape mismatch handling"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import numpy as np
import logging
from src.data import ARCProblem, ARCPair
from src.executors import CodeExecutor
from src.generators import BARCOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create a simple problem
train_pairs = [
    ARCPair(
        x=np.array([[0, 1], [1, 0]]),
        y=np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
    )
]

problem = ARCProblem(uid="shape_test", train_pairs=train_pairs, test_pairs=[])

# Test 1: Code with shape mismatch (returns 2x2 instead of 4x4)
shape_mismatch_code = """
from common import *

def main(input_grid):
    # Returns input unchanged - wrong shape!
    return input_grid
"""

# Test 2: Code with runtime error
error_code = """
from common import *

def main(input_grid):
    # This will cause an error
    undefined_variable
    return input_grid
"""

# Test shape mismatch
print("=" * 80)
print("TEST 1: Shape Mismatch (should succeed)")
print("=" * 80)

executor = CodeExecutor()
result1 = executor.execute(shape_mismatch_code, problem)

print(f"Success: {result1.success}")
print(f"Accuracy: {result1.accuracy}")
print(f"Error messages: {result1.error_messages}")
print(f"Comparison results: {result1.comparison_results}")

if result1.success:
    print("✅ CORRECT: Shape mismatch allows GLM evaluation")
else:
    print("❌ WRONG: Shape mismatch should not prevent GLM evaluation")

# Test runtime error
print("\n" + "=" * 80)
print("TEST 2: Runtime Error (should fail)")
print("=" * 80)

result2 = executor.execute(error_code, problem)

print(f"Success: {result2.success}")
print(f"Accuracy: {result2.accuracy}")
print(f"Error messages: {result2.error_messages}")
print(f"Comparison results: {result2.comparison_results}")

if not result2.success:
    print("✅ CORRECT: Runtime errors prevent GLM evaluation")
else:
    print("❌ WRONG: Runtime errors should prevent GLM evaluation")