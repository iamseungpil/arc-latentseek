#!/usr/bin/env python3
"""
Simple shape analysis without using complex functions
"""

import sys
import numpy as np
from src.data import ARCDataLoader
from common import *
import time

def get_object_shape_simple(obj_mask):
    """Get shape of object using simple bounding box"""
    rows, cols = np.where(obj_mask)
    if len(rows) == 0:
        return (0, 0)
    
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    
    height = max_r - min_r + 1
    width = max_c - min_c + 1
    
    return (height, width)

def analyze_simple_shape_rule():
    """Simple analysis of shape-based rule"""
    
    print("=== SIMPLE SHAPE RULE ANALYSIS ===")
    
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    analysis_file = f"results/simple_shape_analysis_{timestamp}.txt"
    
    with open(analysis_file, "w") as f:
        f.write("=== SIMPLE SHAPE RULE ANALYSIS ===\n\n")
        
        for i, pair in enumerate(problem.train_pairs):
            print(f"\n=== PAIR {i} ===")
            f.write(f"=== PAIR {i} ===\n")
            
            # Manual object detection: find connected components by color
            unique_colors = np.unique(pair.x)
            
            objects_by_color = {}
            
            for color in unique_colors:
                if color == 0:  # Skip black background
                    continue
                    
                # Get all pixels of this color
                color_mask = (pair.x == color)
                color_positions = np.where(color_mask)
                
                if len(color_positions[0]) > 0:
                    shape = get_object_shape_simple(color_mask)
                    pixel_count = len(color_positions[0])
                    
                    objects_by_color[color] = {
                        'shape': shape,
                        'pixel_count': pixel_count,
                        'positions': color_positions
                    }
                    
                    print(f"  Color {color}: Shape {shape}, {pixel_count} pixels")
                    f.write(f"  Color {color}: Shape {shape}, {pixel_count} pixels\n")
            
            # Test shape-based rule
            if Color.BLUE in objects_by_color:
                blue_info = objects_by_color[Color.BLUE]
                blue_shape = blue_info['shape']
                
                print(f"\nBlue object shape: {blue_shape}")
                f.write(f"\nBlue object shape: {blue_shape}\n")
                
                # Find non-blue objects with same shape
                matching_objects = []
                for color, info in objects_by_color.items():
                    if color != Color.BLUE and info['shape'] == blue_shape:
                        matching_objects.append((color, info))
                
                if matching_objects:
                    print(f"Objects with same shape as blue:")
                    f.write(f"Objects with same shape as blue:\n")
                    
                    for color, info in matching_objects:
                        print(f"  Color {color}: Shape {info['shape']}")
                        f.write(f"  Color {color}: Shape {info['shape']}\n")
                    
                    # Check what happens to blue in output
                    blue_positions = blue_info['positions']
                    if len(blue_positions[0]) > 0:
                        # Check first blue pixel
                        r, c = blue_positions[0][0], blue_positions[1][0]
                        output_color = pair.y[r, c]
                        
                        expected_color = matching_objects[0][0]  # First match
                        
                        print(f"  Blue pixel at ({r},{c}) becomes color {output_color}")
                        print(f"  Expected (first match): {expected_color}")
                        f.write(f"  Blue pixel at ({r},{c}) becomes color {output_color}\n")
                        f.write(f"  Expected (first match): {expected_color}\n")
                        
                        if output_color == expected_color:
                            print(f"  ✅ SHAPE RULE MATCHES!")
                            f.write(f"  ✅ SHAPE RULE MATCHES!\n")
                        else:
                            print(f"  ❌ SHAPE RULE MISMATCH!")
                            f.write(f"  ❌ SHAPE RULE MISMATCH!\n")
                            
                        # Check all blue pixels for consistency
                        all_match = True
                        for j in range(len(blue_positions[0])):
                            r, c = blue_positions[0][j], blue_positions[1][j]
                            actual_color = pair.y[r, c]
                            if actual_color != expected_color:
                                all_match = False
                                break
                        
                        if all_match:
                            print(f"  ✅ ALL blue pixels consistently changed to {expected_color}")
                            f.write(f"  ✅ ALL blue pixels consistently changed to {expected_color}\n")
                        else:
                            print(f"  ❌ Blue pixels have inconsistent colors in output")
                            f.write(f"  ❌ Blue pixels have inconsistent colors in output\n")
                else:
                    print(f"  No objects with same shape as blue ({blue_shape})")
                    f.write(f"  No objects with same shape as blue ({blue_shape})\n")
                    
                    # Maybe check if rule is different - like changing to specific color
                    blue_r, blue_c = blue_positions[0][0], blue_positions[1][0]
                    output_color = pair.y[blue_r, blue_c]
                    
                    print(f"  Blue becomes color {output_color} (no shape match)")
                    f.write(f"  Blue becomes color {output_color} (no shape match)\n")
            
            f.write("\n" + "="*50 + "\n")
    
    print(f"\nSimple shape analysis saved to: {analysis_file}")

def test_glm_on_simple_cases():
    """Test GLM with very simple, clear cases"""
    
    print("\n=== TESTING GLM WITH SIMPLE CASES ===")
    
    # Create simple test cases that clearly show the pattern
    simple_test_code = """from common import *
import numpy as np

# Test case 1: Correct shape-based transformation
def transform(input_grid):
    output_grid = np.copy(input_grid)
    
    # This is the CORRECT rule based on analysis:
    # Blue should change to color of same-shaped object
    
    # Replace all blue (1) with teal (8) - this matches pair 0
    output_grid[output_grid == Color.BLUE] = Color.TEAL
    
    return output_grid"""
    
    # Test it
    from src.executors import CodeExecutor
    from src.generators import BARCOutput
    
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    executor = CodeExecutor()
    result = executor.execute(simple_test_code, problem)
    
    print(f"Simple test accuracy: {result.accuracy:.4f}")
    
    for i, pair in enumerate(problem.train_pairs):
        output = executor.execute_single(simple_test_code, pair.x)
        if isinstance(output, np.ndarray):
            comparison, ratio = executor._compare_grids(output, pair.y)
            print(f"Pair {i}: {ratio:.4f} match")

if __name__ == "__main__":
    analyze_simple_shape_rule()
    test_glm_on_simple_cases()