#!/usr/bin/env python3
"""
Test 1: BARC 모델 코드 실행 테스트
"""

import sys
import numpy as np
from src.data import ARCDataLoader
from src.executors import CodeExecutor
from common import *

def test_barc_code_execution():
    """Test the best BARC code from the experiment"""
    
    # The best code from the experiment
    best_code = """from common import *

import numpy as np
from typing import *

# concepts:
# color transformation, object detection, color matching

# description:
# In the input, you will see a grid with several colored objects. 
# For each object that is blue, change its color to the color of the nearest object that is not blue.
# If there is no other object, it remains unchanged.

def main(input_grid):
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)

    # Find all blue objects in the grid
    blue_objects = detect_objects(grid=input_grid, colors=[Color.BLUE], monochromatic=True, connectivity=4)

    for blue_object in blue_objects:
        # Get the position of the blue object
        x, y = object_position(blue_object, background=Color.BLACK, anchor='center')
        
        # Initialize the minimum distance and the target color
        min_distance = float('inf')
        target_color = None
        
        # Find the nearest object that is not blue
        for obj in detect_objects(grid=input_grid, colors=[c for c in Color.NOT_BLACK if c!= Color.BLUE], monochromatic=True, connectivity=4):
            # Get the position of the other object
            other_x, other_y = object_position(obj, background=Color.BLACK, anchor='center')
            
            # Calculate the Euclidean distance
            distance = np.sqrt((other_x - x) ** 2 + (other_y - y) ** 2)
            
            # Update the minimum distance and target color if a closer object is found
            if distance < min_distance:
                min_distance = distance
                target_color = object_colors(obj, background=Color.BLACK)[0]  # Get the color of the other object

        # If a target color was found, change the color of the blue object to the target color
        if target_color is not None:
            output_grid[blue_object == Color.BLUE] = target_color

    return output_grid"""
    
    # Load problem 2a5f8217
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    print("=== BARC CODE EXECUTION TEST ===")
    print(f"Problem: {problem.uid}")
    print(f"Training pairs: {len(problem.train_pairs)}")
    
    # Test code execution
    executor = CodeExecutor(timeout=5)
    result = executor.execute(best_code, problem)
    
    print(f"\n=== EXECUTION RESULTS ===")
    print(f"Success: {result.success}")
    print(f"Accuracy: {result.accuracy:.4f}")
    print(f"Output grids: {len(result.output_grids)}")
    
    if result.error_messages:
        print(f"\n=== ERRORS ===")
        for error in result.error_messages:
            print(f"- {error}")
    
    # Test each training pair individually
    print(f"\n=== DETAILED PAIR ANALYSIS ===")
    for i, pair in enumerate(problem.train_pairs):
        print(f"\nPair {i}:")
        print(f"  Input shape: {pair.x.shape}")
        print(f"  Expected shape: {pair.y.shape}")
        
        try:
            output = executor.execute_single(best_code, pair.x)
            if isinstance(output, np.ndarray):
                comparison, ratio = executor._compare_grids(output, pair.y)
                print(f"  Output shape: {output.shape}")
                print(f"  Comparison: {comparison}")
                print(f"  Match ratio: {ratio:.4f}")
                print(f"  Input unique colors: {np.unique(pair.x)}")
                print(f"  Expected unique colors: {np.unique(pair.y)}")
                print(f"  Output unique colors: {np.unique(output)}")
            else:
                print(f"  Error: {output}")
        except Exception as e:
            print(f"  Exception: {e}")

if __name__ == "__main__":
    test_barc_code_execution()