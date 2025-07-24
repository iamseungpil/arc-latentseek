#!/usr/bin/env python3
"""
Test code extraction including description extraction
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.generators.code_parser import extract_code_elements, parse_code

# Test cases with different formats
test_cases = [
    # Case 1: Standard format with concepts and description
    """# concepts: grid transformation, color mapping, pattern recognition
# description: This problem involves identifying colored regions in the input grid
# and transforming them according to a specific pattern. The transformation
# appears to involve replacing certain colors with others based on their position.

```python
import numpy as np

def transform(input_grid):
    # Implementation here
    output_grid = input_grid.copy()
    return output_grid
```""",
    
    # Case 2: Format without proper markdown
    """Looking at the examples, I can see that:
# concepts: color swapping, mirror transformation
# description: The transformation swaps blue and teal colors while keeping black unchanged.

def transform(input_grid):
    output_grid = input_grid.copy()
    # Swap colors
    output_grid[input_grid == 1] = 8  # Blue to Teal
    output_grid[input_grid == 8] = 1  # Teal to Blue
    return output_grid""",
    
    # Case 3: Real output format from BARC
    """Let's analyze the pattern:

In Example 1:
- Blue (1) becomes Teal (8)
- Black (0) stays Black (0)

# concepts: color replacement
# description: Replace all blue cells with teal

```python
def transform(input_grid):
    output = input_grid.copy()
    output[input_grid == 1] = 8
    return output
```""",

    # Case 4: Multiple code blocks
    """First, let me analyze:

```python
# Helper function
def count_colors(grid):
    return np.unique(grid, return_counts=True)
```

# concepts: pattern detection, color analysis
# description: This solution counts colors and applies transformations based on frequency

```python
import numpy as np

def transform(input_grid):
    # Main transformation
    return input_grid
```"""
]

print("Testing code extraction...")
print("="*80)

for i, test_case in enumerate(test_cases, 1):
    print(f"\nTest Case {i}:")
    print("-"*40)
    
    # Extract elements
    concepts, description, plan = extract_code_elements(test_case)
    code_blocks = parse_code(test_case)
    
    print(f"Concepts: {concepts}")
    print(f"Description: {description[:100] + '...' if description and len(description) > 100 else description}")
    print(f"Plan: {plan[:50] + '...' if plan and len(plan) > 50 else plan}")
    print(f"Code blocks found: {len(code_blocks)}")
    
    if code_blocks:
        print(f"First code block length: {len(code_blocks[0])} chars")
        print("Code preview:")
        print(code_blocks[0][:200] + "..." if len(code_blocks[0]) > 200 else code_blocks[0])
    
    # Check if we can find def transform
    has_transform = any("def transform" in block for block in code_blocks)
    print(f"Has transform function: {has_transform}")

print("\n" + "="*80)
print("Testing with actual BARC output...")

# Test with the actual output we just generated
actual_output = """Let's start by analyzing the given data points.

### Step-by-Step Analysis

#### Example 1:
| Input | Output |
|-------|--------|
| Black | Black |
| Blue  | Teal |
| Black | Black |
| Teal | Teal |
| Black | Black |

# concepts: color transformation, pattern matching
# description: The transformation replaces Blue with Teal while keeping other colors unchanged

import numpy as np

def transform(input_grid):
    output_grid = input_grid.copy()
    # Replace Blue (1) with Teal (8)
    output_grid[input_grid == 1] = 8
    return output_grid
"""

concepts, description, plan = extract_code_elements(actual_output)
code_blocks = parse_code(actual_output)

print(f"\nConcepts: {concepts}")
print(f"Description: {description}")
print(f"Code blocks: {len(code_blocks)}")
if code_blocks:
    print(f"Code extracted successfully: {len(code_blocks[0])} chars")