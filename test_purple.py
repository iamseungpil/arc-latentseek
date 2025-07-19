#!/usr/bin/env python3
"""Test Color.PURPLE specifically"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.executors import CodeExecutor
from src.data import ARCProblem, ARCPair

# Test code that uses PURPLE
test_code = """from common import *

def main(input_grid):
    output_grid = np.full_like(input_grid, Color.PURPLE)
    return output_grid"""

# Create a simple test problem
test_input = np.array([
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8]
])

# Create test problem
test_problem = ARCProblem(
    uid="test_purple",
    train_pairs=[ARCPair(x=test_input, y=test_input)],  # dummy output
    test_pairs=[]
)

# Test executor
print("Testing Color.PURPLE...")
executor = CodeExecutor(timeout=2)

# Test execution
exec_result = executor.execute(test_code, test_problem)
print(f"Success: {exec_result.success}")
print(f"Error messages: {exec_result.error_messages}")
if exec_result.output_grids and exec_result.success:
    print(f"Output grid:\n{exec_result.output_grids[0]}")
    print(f"All values are 8 (PURPLE)? {np.all(exec_result.output_grids[0] == 8)}")