#!/usr/bin/env python3
"""Simple test for V12 components"""

import sys
import re

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

import arc
from src.evaluators.simple_evaluator import SimpleEvaluator

# Test with pre-generated code
test_code = '''
from common import *
import numpy as np

# concepts:
# color mapping, object detection, color replacement

# description:
# In the input, you will see a grid containing several objects of different colors. 
# Each object is defined by a connected region of pixels of the same color. 
# To make the output, change the color of each object to match the color of the object directly below it. 
# If there is no object below, the color remains unchanged.

def main(input_grid):
    # Copy the input grid to the output grid
    output_grid = np.copy(input_grid)
    
    # Get the objects in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK)
    
    # For each object, change its color to the color of the object below it
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        
        # Check if there is an object directly below the bounding box
        if y + height < output_grid.shape[1]:  # Ensure we don't go out of bounds
            below_color = output_grid[x:x+width, y + height].max()  # Get the color of the pixels directly below
            
            # Change the color of the current object to the color of the object below
            output_grid[obj == output_grid[x, y]] = below_color
    
    return output_grid
'''

def test_code_extraction():
    """Test code extraction from response"""
    print("=== TESTING CODE EXTRACTION ===")
    
    # Extract concepts
    concepts_match = re.search(r'# concepts:\s*\n#?\s*(.*?)(?=\n\n|\n#)', test_code, re.DOTALL)
    if concepts_match:
        concepts = concepts_match.group(1).strip()
        print(f"Concepts: {concepts}")
    
    # Extract description
    desc_match = re.search(r'# description:\s*\n((?:#[^\n]*\n)+)', test_code, re.DOTALL)
    if desc_match:
        desc_lines = re.findall(r'#\s*(.*?)$', desc_match.group(1), re.MULTILINE)
        description = ' '.join([line.strip() for line in desc_lines if line.strip()])
        print(f"\nDescription: {description}")
    
    # Extract main function
    func_match = re.search(r'def main\(.*?\):(.*?)(?=\n(?:def|$))', test_code, re.DOTALL)
    if func_match:
        print("\nMain function found: ✓")
    
    return True

def test_code_execution():
    """Test code execution"""
    print("\n=== TESTING CODE EXECUTION ===")
    
    evaluator = SimpleEvaluator()
    problem_id = "2a5f8217"
    
    # Test execution
    result = evaluator.evaluate_solution(problem_id, test_code)
    
    print(f"Execution Success: {result['execution_success']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    if result.get('error'):
        print(f"Error: {result['error']}")
    
    if result['generated_outputs']:
        print(f"Generated {len(result['generated_outputs'])} outputs")
        for i, output in enumerate(result['generated_outputs']):
            print(f"  Output {i+1} shape: {output.shape}")
    
    return result

def test_description_finding():
    """Test description token finding in V12"""
    print("\n=== TESTING DESCRIPTION TOKEN FINDING ===")
    
    # Simulate tokenized input
    text = """### System:
You are a world-class puzzle solver...

### Instruction:
Solve the following ARC puzzle...

### Response:
```python
# concepts:
# color mapping, object detection

# description:
# In the input, you will see a grid
# containing colored objects.

def main(input_grid):
    return output_grid
```"""
    
    # Find description pattern
    desc_pattern = r'# description:\s*\n((?:# .*\n)*)'
    match = re.search(desc_pattern, text)
    
    if match:
        desc_start = match.start(1)
        desc_end = match.end(1)
        description = text[desc_start:desc_end]
        print(f"Description found at positions {desc_start}-{desc_end}")
        print(f"Description content:\n{description}")
    
    return True

if __name__ == "__main__":
    print("Testing V12 components...\n")
    
    # Test 1: Code extraction
    test_code_extraction()
    
    # Test 2: Code execution
    result = test_code_execution()
    
    # Test 3: Description finding
    test_description_finding()
    
    print("\n✅ All tests completed!")