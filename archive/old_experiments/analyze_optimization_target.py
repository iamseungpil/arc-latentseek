#!/usr/bin/env python3
"""
Detailed analysis of what LatentSeek should optimize in BARC outputs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import re
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")

# Multiple example BARC responses showing different structures
examples = [
    {
        "name": "Standard format",
        "response": """Let me analyze this puzzle step by step.

Looking at the training examples, I notice:
- The input grids contain colored objects
- The output grids show transformations of these objects

# concepts: object detection, color mapping, spatial transformation
# description: The task involves detecting colored objects in the input grid and applying a transformation rule based on their properties.

```python
import numpy as np

def transform(input_grid):
    output_grid = np.zeros_like(input_grid)
    # Detect objects and transform
    return output_grid
```"""
    },
    {
        "name": "Code-first format",
        "response": """# concepts: pattern matching, rule application
# description: Apply transformation rules to matching patterns

def transform(input_grid):
    output = np.copy(input_grid)
    # Find patterns and apply rules
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            # Check pattern
            pass
    return output

The algorithm works by scanning the grid for specific patterns."""
    },
    {
        "name": "Detailed explanation format",
        "response": """I'll solve this step-by-step.

First, let me understand what's happening in the examples:

Example 1 Analysis:
- Input: 7x7 grid with blue (1) and red (2) pixels
- Output: Some pixels change color based on neighbors
- Pattern: Pixels surrounded by 3+ same-color neighbors change to that color

Example 2 Analysis:
- Similar pattern but with different colors
- The rule seems consistent

Based on this analysis:

# concepts: neighbor counting, majority rule, color transformation
# description: Each pixel changes to the color of the majority of its neighbors if there are 3 or more neighbors of the same color. Otherwise, it stays the same.

Here's my solution:

```python
import numpy as np

def transform(input_grid):
    rows, cols = input_grid.shape
    output_grid = np.copy(input_grid)
    
    # Check each pixel
    for i in range(rows):
        for j in range(cols):
            # Count neighbors by color
            neighbor_counts = {}
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        color = input_grid[ni, nj]
                        neighbor_counts[color] = neighbor_counts.get(color, 0) + 1
            
            # Apply majority rule
            for color, count in neighbor_counts.items():
                if count >= 3:
                    output_grid[i, j] = color
                    break
    
    return output_grid
```"""
    }
]

print("BARC Output Structure Analysis")
print("=" * 80)

for example in examples:
    print(f"\n{example['name']}:")
    print("-" * 60)
    
    response = example['response']
    tokens = tokenizer.encode(response)
    decoded_tokens = [tokenizer.decode([token]) for token in tokens]
    
    # Find key positions
    code_start_pos = None
    concepts_pos = None
    description_pos = None
    
    # Find in original text
    text_code_start = float('inf')
    if "```python" in response:
        text_code_start = response.find("```python")
    if "def transform" in response:
        text_code_start = min(text_code_start, response.find("def transform"))
    
    text_concepts = response.find("# concepts:")
    text_description = response.find("# description:")
    
    # Find in tokens
    for i in range(len(decoded_tokens)):
        token = decoded_tokens[i]
        
        if "concepts" in token and i > 0 and "#" in decoded_tokens[i-1]:
            concepts_pos = i-1
        if "description" in token and i > 0 and "#" in decoded_tokens[i-1]:
            description_pos = i-1
        if "python" in token and i > 0 and "```" in decoded_tokens[i-1]:
            code_start_pos = i-1
        if "def" in token and i < len(decoded_tokens)-1 and "transform" in decoded_tokens[i+1]:
            if code_start_pos is None or i < code_start_pos:
                code_start_pos = i
    
    print(f"Total tokens: {len(tokens)}")
    print(f"\nText positions:")
    print(f"  concepts: {text_concepts}")
    print(f"  description: {text_description}")
    print(f"  code start: {text_code_start}")
    
    print(f"\nToken positions:")
    print(f"  concepts: {concepts_pos}")
    print(f"  description: {description_pos}")
    print(f"  code start: {code_start_pos}")
    
    # Analyze what gets optimized with k=0.2
    k = 0.2
    update_length = int(k * len(tokens))
    
    print(f"\nWith k={k} (optimizing {update_length} tokens):")
    
    if code_start_pos is not None:
        if code_start_pos >= update_length:
            print(f"  ❌ Code starts at token {code_start_pos}, but optimization only covers 0-{update_length-1}")
            print(f"  ⚠️  MISSING THE CODE ENTIRELY!")
        else:
            code_tokens_optimized = update_length - code_start_pos
            total_code_tokens = len(tokens) - code_start_pos
            percentage = (code_tokens_optimized / total_code_tokens) * 100
            print(f"  ✓ Optimizing {code_tokens_optimized}/{total_code_tokens} code tokens ({percentage:.1f}%)")
    
    # Show what actually gets optimized
    print(f"\nWhat gets optimized:")
    optimized_text = tokenizer.decode(tokens[:update_length])
    print(f"  '{optimized_text[:100]}...'")

print("\n" + "=" * 80)
print("RECOMMENDATIONS:")
print("1. Current implementation optimizes first 20% of tokens after prompt")
print("2. This often captures explanation/reasoning text, NOT the actual code")
print("3. Should identify where 'def transform' or '```python' starts")
print("4. Optimize tokens from code start position, not from beginning")
print("5. Could also focus on the actual function body (after 'def transform:')")
print("\nProposed fix:")
print("- Find where code generation actually begins")
print("- Start optimization from that point")
print("- This ensures we're optimizing code generation, not description")