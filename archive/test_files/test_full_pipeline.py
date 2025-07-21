#!/usr/bin/env python3
"""Test full pipeline on single problem"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from src.data import ARCProblem, ARCPair
from src.generators import BARCGenerator
from src.executors import CodeExecutor
from src.evaluators import GLMEvaluator
from src.optimizers import LatentSeekOptimizer

# Create the 2072aba6 problem
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
    ),
    ARCPair(
        x=np.array([[5, 0, 5], [0, 0, 0], [5, 0, 5]]),
        y=np.array([
            [2, 1, 2, 2, 1, 2],
            [1, 1, 1, 1, 1, 1],
            [2, 1, 2, 2, 1, 2],
            [2, 1, 2, 2, 1, 2],
            [1, 1, 1, 1, 1, 1],
            [2, 1, 2, 2, 1, 2]
        ])
    ),
    ARCPair(
        x=np.array([[0, 0, 0], [5, 5, 5], [0, 0, 0]]),
        y=np.array([
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [1, 1, 1, 1, 1, 1]
        ])
    )
]

test_pairs = [ARCPair(x=np.array([[5, 5, 5], [0, 0, 0], [5, 5, 5]]), y=None)]

problem = ARCProblem(
    uid="2072aba6",
    train_pairs=train_pairs,
    test_pairs=test_pairs
)

print("Testing ARC-LatentSeek pipeline on problem 2072aba6\n")

# 1. Generate BARC code
print("1. Generating BARC code...")
barc_generator = BARCGenerator("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")
barc_outputs = barc_generator.generate(problem, num_candidates=1)

if not barc_outputs:
    print("No BARC outputs generated!")
    exit(1)

barc_output = barc_outputs[0]
print(f"Description: {barc_output.description}")
print(f"\nGenerated code (first 500 chars):")
print(barc_output.code[:500] + "...")

# 2. Execute code
print("\n2. Executing code...")
executor = CodeExecutor()
execution_result = executor.execute(barc_output.code, problem)

print(f"Execution success: {execution_result.success}")
print(f"Accuracy: {execution_result.accuracy}")
print(f"Error messages: {execution_result.error_messages}")
print(f"Comparison results: {execution_result.comparison_results}")

# Show detailed results for first pair
if execution_result.output_grids and isinstance(execution_result.output_grids[0], np.ndarray):
    print(f"\nFirst pair output shape: {execution_result.output_grids[0].shape}")
    print(f"Expected shape: {train_pairs[0].y.shape}")

# 3. If execution succeeded but accuracy < 1, try LatentSeek
if execution_result.success and execution_result.accuracy < 1.0:
    print(f"\n3. Trying LatentSeek optimization (accuracy {execution_result.accuracy} < 1.0)...")
    
    # Skip GLM evaluation and go straight to LatentSeek
    print("Initializing LatentSeek optimizer...")
    latent_optimizer = LatentSeekOptimizer()
    
    print("Running LatentSeek optimization...")
    opt_result = latent_optimizer.optimize(
        problem=problem,
        initial_output=barc_output,
        execution_result=execution_result,
        glm_evaluation=None  # Skip GLM for now
    )
    
    print(f"\nLatentSeek converged: {opt_result.converged}")
    print(f"Optimization steps: {opt_result.optimization_steps}")
    print(f"Final accuracy: {opt_result.final_accuracy}")
    
    if opt_result.converged:
        print("\nOptimized description:")
        print(opt_result.final_output.description)
else:
    print("\n3. Skipping LatentSeek (execution failed or already perfect)")