#!/usr/bin/env python3
"""Test BARC generation directly with hardcoded problem"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from src.data import ARCProblem, ARCPair
from src.generators import BARCGenerator
from src.executors import CodeExecutor

# Create a simple test problem (2072aba6 from logs)
train_pairs = [
    ARCPair(
        x=np.array([
            [0, 5, 0],
            [5, 5, 5],
            [0, 5, 0]
        ]),
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

test_pairs = [
    ARCPair(
        x=np.array([
            [5, 0, 5],
            [0, 0, 0],
            [5, 0, 5]
        ]),
        y=None  # Unknown for test
    )
]

problem = ARCProblem(
    uid="2072aba6_test",
    train_pairs=train_pairs,
    test_pairs=test_pairs
)

print("Problem created")
print(f"Train input shape: {train_pairs[0].x.shape}")
print(f"Train output shape: {train_pairs[0].y.shape}")

# Initialize BARC generator
print("\nInitializing BARC generator...")
barc_generator = BARCGenerator("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")

# Generate code
print("\nGenerating BARC code...")
barc_outputs = barc_generator.generate(problem, num_candidates=1)

if not barc_outputs:
    print("No outputs generated!")
else:
    barc_output = barc_outputs[0]
    print(f"\nDescription: {barc_output.description}")
    print(f"\nGenerated code:")
    print("=" * 80)
    print(barc_output.code)
    print("=" * 80)
    
    # Test execution
    print("\nTesting execution...")
    executor = CodeExecutor()
    
    # Test on training data
    result = executor.execute_single(barc_output.code, train_pairs[0].x)
    print(f"\nExecution result type: {type(result)}")
    
    if isinstance(result, np.ndarray):
        print(f"Result shape: {result.shape}")
        print(f"Expected shape: {train_pairs[0].y.shape}")
        print(f"\nResult:\n{result}")
        print(f"\nExpected:\n{train_pairs[0].y}")
        print(f"\nMatches: {np.array_equal(result, train_pairs[0].y)}")
    else:
        print(f"Error: {result}")