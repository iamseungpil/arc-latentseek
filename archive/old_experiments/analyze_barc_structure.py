#!/usr/bin/env python3
"""Analyze BARC response structure in detail"""

import sys
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

from src.generators.barc_generator_simple import BARCGeneratorSimple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

# Pre-generate a response with simple setup
sample_response = """### System:
You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your goal is to analyze puzzles and provide solutions in well-documented code.

### Instruction:
Solve the following ARC (Abstraction and Reasoning Corpus) puzzle by examining input-output pairs and determining the transformation rule.

Example 1:
Input:
Black Black Black Red Red
Black Black Black Red Red
Blue Blue Blue Black Black
Blue Blue Blue Black Black
Blue Blue Blue Black Black

Output:
Red Red
Red Red

### Response:
Looking at this example, I can see that the transformation involves extracting a specific colored region from the input.

```python
# concepts:
# color extraction, region identification, pattern matching, size reduction

# description:
# The transformation identifies the red colored region in the input grid
# and extracts only that region as the output. The red 2x2 block in the 
# top-right corner of the input becomes the entire output grid.
# This is a form of color-based cropping or extraction.

def main(input_grid):
    # Find bounds of red region (color value 2)
    red_positions = []
    for i in range(len(input_grid)):
        for j in range(len(input_grid[0])):
            if input_grid[i][j] == 2:  # Red
                red_positions.append((i, j))
    
    if not red_positions:
        return [[]]
    
    # Get bounding box
    min_row = min(pos[0] for pos in red_positions)
    max_row = max(pos[0] for pos in red_positions)
    min_col = min(pos[1] for pos in red_positions)
    max_col = max(pos[1] for pos in red_positions)
    
    # Extract the region
    output_grid = []
    for i in range(min_row, max_row + 1):
        row = []
        for j in range(min_col, max_col + 1):
            row.append(input_grid[i][j])
        output_grid.append(row)
    
    return output_grid
```"""

# Extract code
import re
code_match = re.search(r'```python\n(.*?)```', sample_response, re.DOTALL)
if code_match:
    code = code_match.group(1).strip()
    
    print("Extracted code:")
    print("="*80)
    print(code)
    print("="*80)
    
    # Analyze line by line
    lines = code.split('\n')
    desc_start = None
    desc_end = None
    
    for i, line in enumerate(lines):
        if '# description:' in line:
            desc_start = i
            print(f"\nDescription starts at line {i}: {repr(line)}")
            
        if desc_start is not None and desc_end is None:
            if line.strip() and not line.strip().startswith('#'):
                desc_end = i
                print(f"Description ends at line {i-1}")
                print(f"Description lines: {desc_end - desc_start}")
                
                print("\nDescription content:")
                for j in range(desc_start, desc_end):
                    print(f"  {j}: {repr(lines[j])}")
                break
    
    # Now tokenize to understand token structure
    print("\n" + "="*80)
    print("TOKEN ANALYSIS:")
    print("="*80)
    
    # Use a dummy tokenizer for analysis
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Fast loading
    
    inputs = tokenizer(code, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    tokens = []
    for i in range(input_ids.shape[1]):
        token_id = input_ids[0, i].item()
        token_text = tokenizer.decode([token_id])
        tokens.append(token_text)
    
    print(f"Total tokens: {len(tokens)}")
    
    # Find description in tokens
    desc_token_start = None
    desc_token_end = None
    
    for i, token in enumerate(tokens):
        if 'description' in token:
            desc_token_start = i
            print(f"\nDescription starts around token {i}")
            
            # Look for 'def' to mark end
            for j in range(i, len(tokens)):
                if 'def' in tokens[j] and 'main' in tokens[j+1] if j+1 < len(tokens) else False:
                    desc_token_end = j
                    print(f"Description ends around token {j}")
                    print(f"Description spans approximately {j-i} tokens")
                    break
            break
    
    # Save structure analysis
    analysis = {
        "total_lines": len(lines),
        "description_line_start": desc_start,
        "description_line_end": desc_end - 1 if desc_end else None,
        "description_lines": desc_end - desc_start if desc_end and desc_start else None,
        "total_tokens": len(tokens),
        "description_token_range_approx": [desc_token_start, desc_token_end] if desc_token_start and desc_token_end else None
    }
    
    print("\nStructure Summary:")
    print(json.dumps(analysis, indent=2))