#!/usr/bin/env python3
"""Save BARC generated code for analysis"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from src.data import ARCProblem, ARCPair
from src.generators import BARCGenerator

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

# Generate multiple BARC codes
print("Generating 3 BARC solutions...")
barc_generator = BARCGenerator("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")
barc_outputs = barc_generator.generate(problem, num_candidates=3)

for i, output in enumerate(barc_outputs):
    print(f"\n{'='*80}")
    print(f"BARC Solution {i+1}")
    print(f"{'='*80}")
    print(f"Description: {output.description}\n")
    print("Code:")
    print(output.code)
    
    # Save to file
    with open(f"barc_solution_{i+1}.py", "w") as f:
        f.write(output.code)