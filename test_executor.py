#!/usr/bin/env python3
"""Test the code executor directly"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.executors import CodeExecutor
from src.data import ARCProblem, ARCPair

# Test code from BARC
test_code = """from common import *

import numpy as np
from typing import *

# concepts:
# color inversion, pattern replication, grid expansion

# description:
# In the input you will see a 3x3 grid with a pattern. 
# To make the output, invert the colors of the pattern and replicate the pattern to fill the entire grid.

def main(input_grid):
    # Get the dimensions of the input grid
    height, width = input_grid.shape

    # Create an output grid with the same size as the input grid
    output_grid = np.zeros((height * 2, width * 2), dtype=int)

    # Invert colors of the input grid
    inverted_grid = np.where(input_grid == Color.BLACK, Color.GRAY, Color.BLACK)

    # Replicate the inverted pattern into the output grid
    for i in range(height):
        for j in range(width):
            output_grid[i*2:i*2+2, j*2:j*2+2] = inverted_grid[i, j]

    return output_grid"""

# Create a simple test problem
test_input = np.array([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0]
])

test_output = np.array([
    [5, 5, 0, 0, 5, 5],
    [5, 5, 0, 0, 5, 5],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [5, 5, 0, 0, 5, 5],
    [5, 5, 0, 0, 5, 5]
])

# Create test problem
test_problem = ARCProblem(
    uid="test_problem",
    train_pairs=[ARCPair(x=test_input, y=test_output)],
    test_pairs=[]
)

# Test executor
print("Testing code executor...")
executor = CodeExecutor(timeout=2)

# Test with single input
print("\n1. Testing single execution:")
result = executor.execute_single(test_code, test_input)
print(f"Result type: {type(result)}")
print(f"Result: {result}")

# Test with full problem
print("\n2. Testing full problem execution:")
exec_result = executor.execute(test_code, test_problem)
print(f"Success: {exec_result.success}")
print(f"Accuracy: {exec_result.accuracy}")
print(f"Error messages: {exec_result.error_messages}")
if exec_result.output_grids:
    print(f"Output shape: {exec_result.output_grids[0].shape}")
    print(f"Output grid:\n{exec_result.output_grids[0]}")

# Test Color imports
print("\n3. Testing Color imports in executor namespace:")
test_color_code = """
print("Testing Color access...")
print(f"Color.BLACK = {Color.BLACK}")
print(f"Color.GRAY = {Color.GRAY}")
print(f"Has PURPLE? {hasattr(Color, 'PURPLE')}")
print(f"Has BROWN? {hasattr(Color, 'BROWN')}")
if hasattr(Color, 'PURPLE'):
    print(f"Color.PURPLE = {Color.PURPLE}")
if hasattr(Color, 'BROWN'):
    print(f"Color.BROWN = {Color.BROWN}")
"""

print("\n4. Testing namespace directly:")
from src.executors.common import Color
print(f"Direct import - Color.BLACK = {Color.BLACK}")
print(f"Direct import - Color.GRAY = {Color.GRAY}")
print(f"Direct import - Has PURPLE? {hasattr(Color, 'PURPLE')}")
print(f"Direct import - Has BROWN? {hasattr(Color, 'BROWN')}")