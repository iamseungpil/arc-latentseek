#!/usr/bin/env python3

import numpy as np
from src.executors.code_executor import CodeExecutor
from src.data import ARCProblem, ARCPair

# Sample BARC code from the logs
barc_code = """
from common import *

import numpy as np
from typing import *

# concepts:
# repetition, coloring, grid transformation

# description:
# In the input you will see a 3x3 grid with a pattern of black and gray pixels.
# To make the output, repeat the pattern to fill a 6x6 grid, 
# and color all the gray pixels in the pattern red and all the black pixels in the pattern blue.

def main(input_grid):
    # Get the pattern from the input grid
    pattern = input_grid

    # Create an output grid of size 6x6
    output_grid = np.zeros((6, 6), dtype=int)

    # Fill the output grid with the repeated pattern
    for i in range(2):
        for j in range(2):
            output_grid[i*3:(i+1)*3, j*3:(j+1)*3] = pattern

    # Color the pixels according to the rules
    output_grid[output_grid == Color.BLACK] = Color.BLUE
    output_grid[output_grid == Color.GRAY] = Color.RED

    return output_grid
"""

# Create test data
input_grid = np.array([
    [0, 5, 0],
    [5, 0, 5],
    [0, 5, 0]
])

expected_output = np.array([
    [1, 2, 1, 1, 2, 1],
    [2, 1, 2, 2, 1, 2],
    [1, 2, 1, 1, 2, 1],
    [1, 2, 1, 1, 2, 1],
    [2, 1, 2, 2, 1, 2],
    [1, 2, 1, 1, 2, 1]
])

# Test the executor
executor = CodeExecutor()
result = executor.execute_single(barc_code, input_grid)

print(f"Result type: {type(result)}")
print(f"Result shape: {result.shape if isinstance(result, np.ndarray) else 'N/A'}")
print(f"Result:\n{result}")

if isinstance(result, np.ndarray):
    print(f"\nExpected:\n{expected_output}")
    print(f"\nMatches expected: {np.array_equal(result, expected_output)}")