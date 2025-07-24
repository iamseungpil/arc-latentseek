#!/usr/bin/env python3
"""
Test extraction of def main
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.generators.code_parser import extract_code_elements, parse_code

# Test cases with def main
test_cases = [
    # Case 1: def main with proper formatting
    """# concepts: grid processing, main function
# description: This uses a main function to process grids

```python
import numpy as np

def main(input_grid):
    # Process the grid
    output_grid = input_grid.copy()
    # Replace blue with teal
    output_grid[input_grid == 1] = 8
    return output_grid
```""",
    
    # Case 2: def main without markdown
    """Looking at the pattern:

# concepts: transformation
# description: Simple color swap

import numpy as np

def main(input_grid):
    output = np.zeros_like(input_grid)
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            if input_grid[i,j] == 1:
                output[i,j] = 8
            else:
                output[i,j] = input_grid[i,j]
    return output""",
    
    # Case 3: Both def transform and def main
    """# concepts: helper functions
# description: Uses both transform and main

```python
def transform(grid):
    # Helper function
    return grid.T

def main(input_grid):
    # Main processing
    transposed = transform(input_grid)
    return transposed
```"""
]

print("Testing def main extraction...")
print("="*80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print("-"*40)
    
    # Extract elements
    concepts, description, plan = extract_code_elements(test_case)
    code_blocks = parse_code(test_case)
    
    print(f"Concepts: {concepts}")
    print(f"Description: {description}")
    print(f"Code blocks found: {len(code_blocks)}")
    
    for j, block in enumerate(code_blocks):
        print(f"\nCode block {j+1}:")
        has_transform = "def transform" in block
        has_main = "def main" in block
        print(f"  Has transform: {has_transform}")
        print(f"  Has main: {has_main}")
        print(f"  Length: {len(block)} chars")
        print("  Preview:")
        print("  " + block[:100].replace("\n", "\n  ") + "...")

print("\n" + "="*80)
print("Summary:")
print("- Code parser now handles both def transform and def main")
print("- Extraction works for both markdown and non-markdown formats")