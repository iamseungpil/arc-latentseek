#!/usr/bin/env python3
"""
Test the correct rule: Blue objects -> Same shape object color
"""

import sys
import numpy as np
from src.data import ARCDataLoader
from src.executors import CodeExecutor
from common import *
import time

def analyze_correct_rule():
    """Analyze 2a5f8217 with the correct rule: same shape object color replacement"""
    
    print("=== CORRECT RULE ANALYSIS ===")
    print("Rule: Blue objects should be colored the same as objects with the same shape")
    
    # Load problem
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    analysis_file = f"results/correct_rule_analysis_{timestamp}.txt"
    
    with open(analysis_file, "w") as f:
        f.write("=== CORRECT RULE ANALYSIS ===\n")
        f.write("Rule: Blue objects should be colored the same as objects with the same shape\n\n")
        
        for i, pair in enumerate(problem.train_pairs):
            print(f"\n=== PAIR {i} ===")
            f.write(f"=== PAIR {i} ===\n")
            
            # Detect all objects
            all_objects = detect_objects(pair.x, monochromatic=True, connectivity=4)
            
            print(f"Total objects detected: {len(all_objects)}")
            f.write(f"Total objects detected: {len(all_objects)}\n")
            
            # Analyze each object
            object_info = []
            for j, obj in enumerate(all_objects):
                color = object_colors(obj, background=Color.BLACK)[0]
                bbox = bounding_box(obj)
                shape = (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)  # height, width
                
                object_info.append({
                    'id': j,
                    'color': color,
                    'shape': shape,
                    'bbox': bbox,
                    'positions': np.where(obj > 0)
                })
                
                print(f"  Object {j}: Color {color}, Shape {shape}, BBox {bbox}")
                f.write(f"  Object {j}: Color {color}, Shape {shape}, BBox {bbox}\n")
            
            # Find blue objects and their shape matches
            blue_objects = [obj for obj in object_info if obj['color'] == Color.BLUE]
            non_blue_objects = [obj for obj in object_info if obj['color'] != Color.BLUE and obj['color'] != Color.BLACK]
            
            print(f"\nBlue objects: {len(blue_objects)}")
            print(f"Non-blue objects: {len(non_blue_objects)}")
            f.write(f"\nBlue objects: {len(blue_objects)}\n")
            f.write(f"Non-blue objects: {len(non_blue_objects)}\n")
            
            # Test the rule
            print(f"\n--- RULE TESTING ---")
            f.write(f"\n--- RULE TESTING ---\n")
            
            for blue_obj in blue_objects:
                print(f"Blue object {blue_obj['id']} (shape {blue_obj['shape']}):")
                f.write(f"Blue object {blue_obj['id']} (shape {blue_obj['shape']}):\n")
                
                # Find objects with same shape
                same_shape_objects = [obj for obj in non_blue_objects if obj['shape'] == blue_obj['shape']]
                
                if same_shape_objects:
                    print(f"  Found {len(same_shape_objects)} objects with same shape:")
                    f.write(f"  Found {len(same_shape_objects)} objects with same shape:\n")
                    
                    for same_obj in same_shape_objects:
                        print(f"    Object {same_obj['id']}: Color {same_obj['color']}")
                        f.write(f"    Object {same_obj['id']}: Color {same_obj['color']}\n")
                    
                    # Check what color the blue object becomes in output
                    blue_positions = blue_obj['positions']
                    if len(blue_positions[0]) > 0:
                        r, c = blue_positions[0][0], blue_positions[1][0]
                        actual_output_color = pair.y[r, c]
                        expected_color = same_shape_objects[0]['color']  # Assume first match
                        
                        print(f"  Expected color: {expected_color}")
                        print(f"  Actual output color: {actual_output_color}")
                        f.write(f"  Expected color: {expected_color}\n")
                        f.write(f"  Actual output color: {actual_output_color}\n")
                        
                        if expected_color == actual_output_color:
                            print(f"  ✅ RULE MATCHES!")
                            f.write(f"  ✅ RULE MATCHES!\n")
                        else:
                            print(f"  ❌ RULE MISMATCH!")
                            f.write(f"  ❌ RULE MISMATCH!\n")
                else:
                    print(f"  No objects with same shape found")
                    f.write(f"  No objects with same shape found\n")
            
            f.write("\n" + "="*50 + "\n")
    
    print(f"\nCorrect rule analysis saved to: {analysis_file}")
    return analysis_file

def test_correct_implementation():
    """Test implementation with the correct rule"""
    
    correct_code = """from common import *
import numpy as np

# concepts: shape matching, object detection, color transformation
# description: Change blue objects to match the color of objects with the same shape

def transform(input_grid):
    output_grid = np.copy(input_grid)
    
    # Detect all objects
    all_objects = detect_objects(input_grid, monochromatic=True, connectivity=4)
    
    # Analyze each object
    object_info = []
    for obj in all_objects:
        if np.sum(obj) == 0:  # Skip empty objects
            continue
            
        color = object_colors(obj, background=Color.BLACK)[0]
        bbox = bounding_box(obj)
        shape = (bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1)  # height, width
        
        object_info.append({
            'obj': obj,
            'color': color,
            'shape': shape
        })
    
    # Find blue objects and match them with same-shape objects
    for obj_data in object_info:
        if obj_data['color'] == Color.BLUE:
            # Find non-blue objects with same shape
            same_shape_objects = [
                other for other in object_info 
                if (other['shape'] == obj_data['shape'] and 
                    other['color'] != Color.BLUE and 
                    other['color'] != Color.BLACK)
            ]
            
            if same_shape_objects:
                # Use the color of the first matching object
                target_color = same_shape_objects[0]['color']
                
                # Replace blue pixels with target color
                blue_mask = obj_data['obj'] > 0
                output_grid[blue_mask] = target_color
    
    return output_grid"""
    
    print("\n=== TESTING CORRECT IMPLEMENTATION ===")
    
    # Load problem and test
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    executor = CodeExecutor(timeout=10)
    result = executor.execute(correct_code, problem)
    
    print(f"Execution success: {result.success}")
    print(f"Accuracy: {result.accuracy:.4f}")
    
    if result.error_messages:
        print(f"Errors: {result.error_messages}")
    
    # Test each pair
    for i, pair in enumerate(problem.train_pairs):
        try:
            output = executor.execute_single(correct_code, pair.x)
            if isinstance(output, np.ndarray):
                comparison, ratio = executor._compare_grids(output, pair.y)
                print(f"Pair {i}: Match ratio {ratio:.4f}, Comparison: {comparison}")
            else:
                print(f"Pair {i}: Error - {output}")
        except Exception as e:
            print(f"Pair {i}: Exception - {e}")
    
    return result

if __name__ == "__main__":
    # First analyze the correct rule
    analysis_file = analyze_correct_rule()
    
    # Then test correct implementation
    result = test_correct_implementation()
    
    print(f"\n=== SUMMARY ===")
    print(f"Analysis saved to: {analysis_file}")
    print(f"Correct implementation accuracy: {result.accuracy:.4f}")