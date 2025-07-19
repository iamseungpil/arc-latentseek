#!/usr/bin/env python3
"""
Quick GLM test without loading heavy models - use existing loaded models
"""

import sys
import os
import numpy as np
from src.data import ARCDataLoader
from src.executors import CodeExecutor, ExecutionResult
from src.generators import BARCOutput
import time

def create_test_barc_output(code, description="Test description", concepts="Test concepts"):
    """Create a test BARCOutput for evaluation"""
    return BARCOutput(
        code=code,
        concepts=concepts,
        description=description,
        plan="Test plan",
        raw_response=f"Test response with code:\n{code}"
    )

def test_glm_pattern_recognition():
    """Test different rule hypotheses and see what GLM thinks"""
    
    print("=== GLM PATTERN RECOGNITION TEST ===")
    
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    # Different rule hypotheses to test
    test_scenarios = [
        {
            "name": "Correct Shape Rule",
            "description": "Change blue objects to match the color of objects with the same shape. If no same-shape object exists, use a fallback rule.",
            "code": """from common import *
import numpy as np

def transform(input_grid):
    output_grid = np.copy(input_grid)
    # Perfect solution for pair 0: blue->teal (same shape)
    # For other pairs, use nearest neighbor as fallback
    output_grid[output_grid == Color.BLUE] = Color.TEAL  # Works for pair 0
    return output_grid"""
        },
        {
            "name": "Wrong Rule - Remove Blue",
            "description": "Simply remove all blue pixels by changing them to black",
            "code": """from common import *
import numpy as np

def transform(input_grid):
    output_grid = np.copy(input_grid)
    output_grid[output_grid == Color.BLUE] = Color.BLACK
    return output_grid"""
        },
        {
            "name": "Wrong Rule - Nearest Color", 
            "description": "Change blue to the most common non-blue color",
            "code": """from common import *
import numpy as np

def transform(input_grid):
    output_grid = np.copy(input_grid)
    # Find most common non-blue, non-black color
    unique, counts = np.unique(input_grid, return_counts=True)
    for color, count in zip(unique, counts):
        if color not in [Color.BLACK, Color.BLUE]:
            output_grid[output_grid == Color.BLUE] = color
            break
    return output_grid"""
        }
    ]
    
    # Test execution accuracy of each scenario
    executor = CodeExecutor(timeout=5)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"results/glm_quick_test_{timestamp}.txt"
    
    with open(results_file, "w") as f:
        f.write("=== GLM PATTERN RECOGNITION TEST ===\n")
        f.write(f"Problem: {problem.uid}\n\n")
        
        for scenario in test_scenarios:
            print(f"\n=== {scenario['name']} ===")
            f.write(f"=== {scenario['name']} ===\n")
            
            # Test execution
            barc_output = create_test_barc_output(scenario['code'], scenario['description'])
            execution_result = executor.execute(barc_output.code, problem)
            
            print(f"Description: {scenario['description']}")
            print(f"Execution accuracy: {execution_result.accuracy:.4f}")
            f.write(f"Description: {scenario['description']}\n")
            f.write(f"Execution accuracy: {execution_result.accuracy:.4f}\n")
            
            # Test each pair
            for i, pair in enumerate(problem.train_pairs):
                try:
                    output = executor.execute_single(scenario['code'], pair.x)
                    if isinstance(output, np.ndarray):
                        comparison, ratio = executor._compare_grids(output, pair.y)
                        print(f"  Pair {i}: {ratio:.4f} match")
                        f.write(f"  Pair {i}: {ratio:.4f} match\n")
                    else:
                        print(f"  Pair {i}: Error - {output}")
                        f.write(f"  Pair {i}: Error - {output}\n")
                except Exception as e:
                    print(f"  Pair {i}: Exception - {e}")
                    f.write(f"  Pair {i}: Exception - {e}\n")
            
            f.write("\n")
    
    print(f"\nQuick GLM test results saved to: {results_file}")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print("The correct rule seems to be:")
    print("1. If blue object has same shape as another object -> use that color")
    print("2. If no same-shape match -> use fallback rule (unclear what)")
    print("3. This explains why BARC's 33% accuracy - it got Pair 0 perfect!")

def check_alignment_quality():
    """Check if alignment is working by comparing original vs aligned BARC output"""
    
    print(f"\n=== ALIGNMENT QUALITY CHECK ===")
    
    # Read the successful result from our experiment
    try:
        import json
        with open("/home/ubuntu/arc-latentseek/results/single_task/2a5f8217_detailed.json", "r") as f:
            result_data = json.load(f)
        
        best_code = result_data.get("best_code", "")
        best_description = result_data.get("best_description", "")
        
        print("Best code from experiment:")
        print(f"Description: {best_description}")
        print(f"Code length: {len(best_code)}")
        print(f"Uses 'from common import *': {'from common import *' in best_code}")
        print(f"Uses Color constants: {'Color.' in best_code}")
        print(f"Has proper structure: {'def main(' in best_code or 'def transform(' in best_code}")
        
        # Check if it matches the shape-based rule understanding
        shape_related_terms = ['shape', 'nearest', 'distance', 'object', 'detect_objects', 'bounding_box']
        found_terms = [term for term in shape_related_terms if term in best_code.lower()]
        
        print(f"Shape-related terms found: {found_terms}")
        print(f"Seems to understand object detection: {'detect_objects' in best_code}")
        
        return True
        
    except Exception as e:
        print(f"Could not read experiment result: {e}")
        return False

if __name__ == "__main__":
    test_glm_pattern_recognition()
    check_alignment_quality()