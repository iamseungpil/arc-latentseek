#!/usr/bin/env python3
"""
Test BARC response structure
"""

import torch

# Example BARC response formats from the codebase
example_responses = [
    # Example 1: Standard format with concepts, description, then code
    """Let me analyze this puzzle step by step.

Looking at the training examples, I can see that:
- Example 1: The input is a 7x7 grid with colored cells, and the output transforms certain cells
- Example 2: Similar pattern where specific cells are modified based on their neighbors

# concepts: pattern detection, color transformation, grid manipulation
# description: The transformation identifies cells with specific neighbor patterns and changes their color based on the majority color of their neighbors. Cells that are isolated or have fewer than 3 neighbors of the same color remain unchanged.

```python
import numpy as np

def transform(input_grid):
    # Implementation of the transformation
    output_grid = np.copy(input_grid)
    # ... transformation logic ...
    return output_grid
```""",

    # Example 2: Code appears early in response
    """I'll solve this step by step.

# concepts: boundary detection, fill operation
# description: Fill enclosed regions with the color of their boundary

def transform(input_grid):
    output = np.copy(input_grid)
    # Find boundaries and fill
    return output

The algorithm works by detecting closed boundaries and filling them.""",

    # Example 3: Multiple code blocks
    """Let me understand the pattern first.

The transformation seems to involve:
1. Finding specific patterns
2. Applying rules

# concepts: pattern matching, rule application

Here's my solution:

```python
import numpy as np

def transform(input_grid):
    # First attempt at transformation
    return input_grid
```

Actually, I need to revise this:

```python
import numpy as np

def transform(input_grid):
    # Better implementation
    output = np.copy(input_grid)
    # Process the grid
    return output
```""",
]

# Analyze where code starts in each example
print("BARC Response Structure Analysis")
print("=" * 80)

for i, response in enumerate(example_responses):
    print(f"\nExample {i+1}:")
    print("-" * 40)
    
    # Find where key elements appear
    concepts_pos = response.find("# concepts:")
    desc_pos = response.find("# description:")
    def_transform_pos = response.find("def transform")
    code_block_pos = response.find("```python")
    
    print(f"Positions in response:")
    print(f"  # concepts: {concepts_pos}")
    print(f"  # description: {desc_pos}")
    print(f"  def transform: {def_transform_pos}")
    print(f"  ```python: {code_block_pos}")
    
    # Find what comes first - code or description
    code_start = min([pos for pos in [def_transform_pos, code_block_pos] if pos >= 0] or [999999])
    metadata_start = min([pos for pos in [concepts_pos, desc_pos] if pos >= 0] or [999999])
    
    print(f"\nAnalysis:")
    print(f"  Metadata (concepts/description) starts at: {metadata_start}")
    print(f"  Code starts at: {code_start}")
    print(f"  Code comes {'BEFORE' if code_start < metadata_start else 'AFTER'} metadata")
    
    # Calculate percentages
    total_length = len(response)
    if code_start < total_length:
        code_percentage = (code_start / total_length) * 100
        print(f"  Code starts at {code_percentage:.1f}% of response")

print("\n" + "=" * 80)
print("CONCLUSIONS:")
print("- BARC responses can have varied structures")
print("- Code can appear before or after concepts/description")
print("- The current implementation that looks for 'def transform' in the response")
print("  might be optimizing description text that comes AFTER the code")
print("- We need to ensure we're optimizing the actual code generation tokens")