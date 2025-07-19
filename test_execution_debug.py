#!/usr/bin/env python3
"""
Test script to debug code execution issues for problem 2a5f8217
"""

import sys
import os
import numpy as np

# Add paths
sys.path.append('/home/ubuntu/arc-latentseek')

# Import modules
from src.data.arc_loader import ARCDataLoader
from src.executors.code_executor_fixed import CodeExecutor

def print_grid(grid, title="Grid"):
    """Pretty print a grid with colors"""
    color_map = {
        0: "⬛", 1: "🟦", 2: "🟥", 3: "🟩", 4: "🟨",
        5: "⬜", 6: "🟪", 7: "🟧", 8: "🟫", 9: "🟫"
    }
    print(f"\n{title}:")
    for row in grid:
        print("".join(color_map.get(int(cell), "❓") for cell in row))

# Test code from the logs
test_codes = [
    {
        "name": "Color Mapping Code",
        "code": """from common import *

import numpy as np
from typing import *

# concepts:
# pattern extraction, color mapping, grid transformation

# description:
# In the input you will see a small pattern of pixels in the top left corner of the grid.
# To make the output grid, you should extract this pattern and change the color of each pixel according to the following mapping:
# blue -> red, yellow -> green, green -> blue, red -> yellow, and black remains black.

def color_mapping(color):
    mapping = {
        Color.BLUE: Color.RED,
        Color.YELLOW: Color.GREEN,
        Color.GREEN: Color.BLUE,
        Color.RED: Color.YELLOW,
    }
    return mapping.get(color, color)  # return the same color if not in the mapping

def main(input_grid):
    # Extract the pattern from the input grid
    pattern = crop(input_grid)

    # Create an output grid of the same size as the pattern
    output_grid = np.zeros_like(pattern)

    # Map colors according to the defined mapping
    for x in range(pattern.shape[0]):
        for y in range(pattern.shape[1]):
            output_grid[x, y] = color_mapping(pattern[x, y])

    return output_grid
"""
    },
    {
        "name": "BARC Color Test",
        "code": """
def transform(input_grid):
    # Test if BARC colors work
    output = input_grid.copy()
    
    # Try to use BARC color names
    try:
        # Check if we can use Color.PURPLE (8 in BARC)
        purple_mask = input_grid == 8
        if np.any(purple_mask):
            output[purple_mask] = 6  # Change to pink
            
        # Check if we can use Color.BROWN (9 in BARC)  
        brown_mask = input_grid == 9
        if np.any(brown_mask):
            output[brown_mask] = 7  # Change to orange
            
    except Exception as e:
        print(f"Error with BARC colors: {e}")
        
    return output
"""
    },
    {
        "name": "Direct Color Swap",
        "code": """
def transform(input_grid):
    # Direct color swapping based on the pattern
    output = input_grid.copy()
    
    # From the examples:
    # Blue (1) -> Purple/Teal (8) or Pink (6) or Brown/Maroon (9)
    # The pattern seems to be replacing blue with the color of another nearby object
    
    # Find all unique colors in the grid
    unique_colors = np.unique(input_grid)
    non_black_colors = [c for c in unique_colors if c != 0]
    
    # For each blue pixel, replace with the highest non-blue color
    blue_mask = input_grid == 1
    if np.any(blue_mask) and len(non_black_colors) > 1:
        # Find the replacement color (highest non-blue color)
        replacement = max([c for c in non_black_colors if c != 1])
        output[blue_mask] = replacement
    
    return output
"""
    }
]

def main():
    print("="*80)
    print("Code Execution Debug Test for Problem 2a5f8217")
    print("="*80)
    
    # Load the problem
    print("\n1. Loading problem 2a5f8217...")
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    if not problem:
        print("❌ Problem not found!")
        return
        
    print(f"✅ Loaded problem: {problem}")
    
    # Show first training example
    print("\n2. First Training Example:")
    print_grid(problem.train_pairs[0].x, "Input")
    print_grid(problem.train_pairs[0].y, "Expected Output")
    
    # Test executor
    executor = CodeExecutor(timeout=5)
    
    # Test each code
    for code_info in test_codes:
        print(f"\n{'='*60}")
        print(f"Testing: {code_info['name']}")
        print(f"{'='*60}")
        
        # Execute on all training pairs
        result = executor.execute(code_info['code'], problem)
        
        print(f"\nSuccess: {result.success}")
        print(f"Accuracy: {result.accuracy:.2%}")
        
        if result.error_messages:
            print("\nErrors:")
            for error in result.error_messages:
                print(f"  - {error}")
        
        # Show results for first pair
        if result.output_grids[0] is not None and not isinstance(result.output_grids[0], str):
            print("\nFirst Training Pair Result:")
            print_grid(problem.train_pairs[0].x, "Input")
            print_grid(result.output_grids[0], "Generated Output")
            print_grid(problem.train_pairs[0].y, "Expected Output")
            
            # Analyze differences
            if not np.array_equal(result.output_grids[0], problem.train_pairs[0].y):
                print("\nColor Analysis:")
                input_colors = np.unique(problem.train_pairs[0].x)
                output_colors = np.unique(result.output_grids[0])
                expected_colors = np.unique(problem.train_pairs[0].y)
                
                print(f"  Input colors: {input_colors}")
                print(f"  Generated colors: {output_colors}")
                print(f"  Expected colors: {expected_colors}")
                
                # Check specific transformations
                blue_in_input = np.any(problem.train_pairs[0].x == 1)
                blue_in_output = np.any(result.output_grids[0] == 1)
                blue_in_expected = np.any(problem.train_pairs[0].y == 1)
                
                print(f"\n  Blue (1) in input: {blue_in_input}")
                print(f"  Blue (1) in output: {blue_in_output}")
                print(f"  Blue (1) in expected: {blue_in_expected}")
                
                # Check for color 8 and 9
                has_8 = np.any(problem.train_pairs[0].y == 8)
                has_9 = np.any(problem.train_pairs[0].y == 9)
                print(f"\n  Expected has color 8: {has_8}")
                print(f"  Expected has color 9: {has_9}")

if __name__ == "__main__":
    main()