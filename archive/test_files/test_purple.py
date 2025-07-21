#!/usr/bin/env python3

import numpy as np
from src.executors.code_executor import CodeExecutor

# Test code that uses Color.PURPLE
test_code = """
from common import *

def main(input_grid):
    output_grid = np.zeros_like(input_grid)
    output_grid[0, 0] = Color.PURPLE
    output_grid[0, 1] = Color.BROWN
    return output_grid
"""

# Create test data
input_grid = np.array([[0, 0], [0, 0]])

# Test the executor
executor = CodeExecutor()
result = executor.execute_single(test_code, input_grid)

print(f"Result type: {type(result)}")
print(f"Result:\n{result}")
print(f"Color.PURPLE value should be 8, got: {result[0, 0] if isinstance(result, np.ndarray) else 'ERROR'}")
print(f"Color.BROWN value should be 9, got: {result[0, 1] if isinstance(result, np.ndarray) else 'ERROR'}")