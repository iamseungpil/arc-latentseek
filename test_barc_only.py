#!/usr/bin/env python3
"""Test BARC generation to see what's causing string indices error"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

from src.data import ARCDataLoader
from src.generators import BARCGenerator

# Load one problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems('validation', num_problems=1)
problem = problems[0]

print(f"Problem ID: {problem.uid}")

# Generate BARC code
barc_generator = BARCGenerator("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")
print("Generating BARC code...")

try:
    outputs = barc_generator.generate(problem, num_candidates=1)
    if outputs:
        print(f"\nGenerated {len(outputs)} outputs")
        for i, output in enumerate(outputs):
            print(f"\nOutput {i+1}:")
            print(f"Type: {type(output)}")
            print(f"Description: {output.description[:100]}...")
            print(f"Code length: {len(output.code)}")
            print(f"Has code: {'def main' in output.code or 'def transform' in output.code}")
    else:
        print("No outputs generated!")
except Exception as e:
    print(f"Error during generation: {e}")
    import traceback
    traceback.print_exc()