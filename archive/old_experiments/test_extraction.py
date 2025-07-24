#!/usr/bin/env python3
"""
Test concepts + description extraction
"""

import re
from pathlib import Path

def extract_premain_content(code: str):
    """Extract concepts and description before 'def main'"""
    
    # Method 1: Simple regex
    print("\n=== Method 1: Simple Regex ===")
    concepts_match = re.search(r'# concepts:(.*?)(?=\n# description:)', code, re.DOTALL)
    description_match = re.search(r'# description:(.*?)(?=\ndef\s+main)', code, re.DOTALL)
    
    if concepts_match:
        concepts = concepts_match.group(1).strip()
        print(f"Concepts found:\n{concepts}")
    else:
        print("Concepts not found")
        
    if description_match:
        description = description_match.group(1).strip()
        print(f"\nDescription found:\n{description}")
    else:
        print("Description not found")
    
    # Method 2: Line by line parsing
    print("\n\n=== Method 2: Line by Line ===")
    lines = code.split('\n')
    
    # Find concepts section
    concepts_start = None
    concepts_end = None
    for i, line in enumerate(lines):
        if line.strip() == '# concepts:':
            concepts_start = i + 1
        elif concepts_start is not None and concepts_end is None:
            if line.strip() == '' or line.startswith('# description:'):
                concepts_end = i
                break
    
    if concepts_start is not None and concepts_end is not None:
        concepts_lines = lines[concepts_start:concepts_end]
        concepts_text = '\n'.join(line.lstrip('#').strip() for line in concepts_lines)
        print(f"Concepts (line by line):\n{concepts_text}")
    
    # Find description section
    desc_start = None
    desc_end = None
    for i, line in enumerate(lines):
        if line.strip() == '# description:':
            desc_start = i + 1
        elif desc_start is not None and desc_end is None:
            if line.strip() == '' or line.startswith('def main'):
                desc_end = i
                break
    
    if desc_start is not None and desc_end is not None:
        desc_lines = lines[desc_start:desc_end]
        desc_text = '\n'.join(line.lstrip('#').strip() for line in desc_lines)
        print(f"\nDescription (line by line):\n{desc_text}")
    
    # Method 3: Find all content between imports and def main
    print("\n\n=== Method 3: Everything between imports and def main ===")
    
    # Find where imports end
    import_end = 0
    for i, line in enumerate(lines):
        if line.startswith('from') or line.startswith('import'):
            import_end = i
        elif import_end > 0 and line.strip() == '':
            import_end = i + 1
            break
    
    # Find where def main starts
    def_main_start = len(lines)
    for i, line in enumerate(lines):
        if line.startswith('def main'):
            def_main_start = i
            break
    
    premain_content = '\n'.join(lines[import_end:def_main_start]).strip()
    print(f"Pre-main content:\n{premain_content}")
    print(f"\nTotal lines: {def_main_start - import_end}")
    

# Test with actual BARC output
test_code = """from common import *

import numpy as np
from typing import *

# concepts:
# color transformation, object detection, pattern matching

# description:
# In the input, you will see a grid with several colored objects. 
# Each object consists of a single color with a unique shape. 
# To make the output, you should change the color of each object to match the color of the largest object.

def main(input_grid):
    # Implementation here
    pass
"""

print("Testing with sample BARC output:")
extract_premain_content(test_code)

# Test with real file
print("\n\n" + "="*80)
print("Testing with real BARC output files:")
print("="*80)

# Check actual generated files
result_dirs = [
    "results/description_v8",
    "results/premain_v9", 
    "results/simple_v10"
]

for dir_path in result_dirs:
    path = Path(dir_path)
    if path.exists():
        # Find initial code files
        initial_files = list(path.glob("*_initial_code.py"))
        if initial_files:
            print(f"\n\nTesting file: {initial_files[0]}")
            with open(initial_files[0], 'r') as f:
                code = f.read()
            
            # Show first 30 lines
            lines = code.split('\n')
            print("First 30 lines of file:")
            for i, line in enumerate(lines[:30]):
                print(f"{i+1:3d}: {line}")
            
            print("\nExtracting content...")
            extract_premain_content(code)
            break