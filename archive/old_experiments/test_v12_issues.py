#!/usr/bin/env python3
"""Test V12 issues with description finding and execution"""

import re
import sys
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

from src.evaluators.simple_evaluator import SimpleEvaluator

# Sample code that V12 would process
test_code = '''from common import *
import numpy as np
from typing import *

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

def test_description_finding():
    """Test description finding logic"""
    print("=== TESTING DESCRIPTION FINDING ===")
    
    # Method 1: Current V12 pattern
    desc_pattern1 = r'# description:\s*\n((?:# .*\n)*)'
    match1 = re.search(desc_pattern1, test_code)
    
    if match1:
        desc1 = match1.group(1)
        print(f"V12 Current Pattern Found:")
        print(f"  Text: '{desc1}'")
        print(f"  Length: {len(desc1)} chars")
        print(f"  Lines: {desc1.count(chr(10))}")
    
    # Method 2: Better pattern
    desc_pattern2 = r'# description:\s*\n((?:# [^\n]+\n)+)'
    match2 = re.search(desc_pattern2, test_code)
    
    if match2:
        desc2 = match2.group(1)
        print(f"\nBetter Pattern Found:")
        print(f"  Text: '{desc2}'")
        print(f"  Length: {len(desc2)} chars")
        print(f"  Lines: {desc2.count(chr(10))}")
    
    # Method 3: Find all lines starting with # after description:
    lines = test_code.split('\n')
    desc_start = None
    desc_lines = []
    
    for i, line in enumerate(lines):
        if 'description:' in line:
            desc_start = i + 1
        elif desc_start is not None:
            if line.strip().startswith('#'):
                desc_lines.append(line)
            elif line.strip():  # Non-comment, non-empty line
                break
    
    print(f"\nLine-by-line Method Found:")
    print(f"  Lines: {len(desc_lines)}")
    print(f"  Full description:")
    for line in desc_lines:
        print(f"    {line}")

def test_modified_code_execution():
    """Test what happens when description is modified"""
    print("\n\n=== TESTING MODIFIED CODE EXECUTION ===")
    
    # Simulate what V12 does - modify description
    modified_codes = [
        # Case 1: Slightly corrupted description
        test_code.replace(
            "# In the input, you will see a grid containing several objects of different colors.",
            "# filled as a 2D array of the cell can either by a integer."
        ),
        
        # Case 2: Broken syntax in description
        test_code.replace(
            "# description:",
            "# description: contains filled as a"
        ),
        
        # Case 3: Normal code (control)
        test_code
    ]
    
    evaluator = SimpleEvaluator()
    
    for i, code in enumerate(modified_codes):
        print(f"\nCase {i+1}:")
        if i == 0:
            print("  Type: Corrupted description content")
        elif i == 1:
            print("  Type: Broken description syntax")
        else:
            print("  Type: Original (control)")
        
        # Try to execute
        result = evaluator.evaluate_solution("2a5f8217", code)
        
        print(f"  Execution Success: {result['execution_success']}")
        if result.get('error'):
            print(f"  Error: {result['error'][:100]}...")

def analyze_token_issue():
    """Analyze why only 7 tokens are found"""
    print("\n\n=== ANALYZING TOKEN ISSUE ===")
    
    # Simulate tokenization
    # In the actual code, tokens 15-22 would be somewhere in the middle
    # Let's count approximate tokens
    
    lines = test_code.split('\n')
    token_count = 0
    
    for i, line in enumerate(lines):
        # Rough estimate: each word/symbol is ~1 token
        tokens_in_line = len(line.split()) + line.count(':') + line.count(',') + line.count('.')
        token_count += tokens_in_line
        
        if 'description:' in line:
            print(f"'description:' found at line {i}, approximate token position: {token_count}")
            
            # Next line should be first description line
            if i+1 < len(lines):
                next_line = lines[i+1]
                next_tokens = len(next_line.split()) + 1  # +1 for newline
                print(f"First description line has ~{next_tokens} tokens")
                print(f"First description line: '{next_line}'")
                
                # If V12 only finds 7 tokens, it's probably just this first line!
                if next_tokens >= 7:
                    print(">>> This explains why V12 finds only 7 tokens!")

def propose_fix():
    """Propose fix for V12"""
    print("\n\n=== PROPOSED FIX ===")
    
    print("1. Fix description pattern to capture ALL description lines:")
    print("   Change: r'# description:\\s*\\n((?:# .*\\n)*)'")
    print("   To:     r'# description:\\s*\\n((?:# [^\\n]+\\n)+)'")
    print("   Or use line-by-line parsing")
    
    print("\n2. Add logging to see what's being generated:")
    print("   - Log generated code for each candidate")
    print("   - Log specific execution errors")
    
    print("\n3. Reduce modification strength:")
    print("   - Lower learning rate (0.01 -> 0.001)")
    print("   - Lower temperature (1.0 -> 0.5)")
    print("   - Smaller modifications to hidden states")
    
    print("\n4. Better error handling:")
    print("   - Catch and log specific Python errors")
    print("   - Try to fix common syntax errors")

if __name__ == "__main__":
    test_description_finding()
    test_modified_code_execution()
    analyze_token_issue()
    propose_fix()