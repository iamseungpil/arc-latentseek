#!/usr/bin/env python
"""Debug description extraction"""

import sys
sys.path.append('/home/ubuntu/arc-latentseek')

from src.optimizers.latent_optimizer import LatentSeekOptimizer

# The actual generated code from the log
generated_code = """from common import *

import numpy as np
from typing import *

# concepts:
# pattern replication, color alternation

# description:
# In the input, you will see a 3x3 pattern made of black and gray pixels. 
# To make the output, replicate the pattern to fill the entire grid, 
# alternating between black and gray for each cell in the pattern.

def main(input_grid):
    # Get the dimensions of the input grid
    n, m = input_grid.shape

    # Create an output grid of the same size filled with black
    output_grid = np.full((n, m), Color.BLACK)

    # Iterate over the input grid and fill the output grid with alternating colors
    for i in range(n):
        for j in range(m):
            # Check the color in the input grid
            if input_grid[i, j] == Color.GRAY:
                # Fill the output grid with alternating colors (black and gray)
                output_grid[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = [[Color.GRAY, Color.BLACK], [Color.BLACK, Color.GRAY]]
            else:
                # Fill the output grid with alternating colors (black and gray)
                output_grid[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = [[Color.BLACK, Color.GRAY], [Color.GRAY, Color.BLACK]]
    
    return output_grid"""

# Test description extraction
class DummyOptimizer:
    def _find_description_token_positions(self, text):
        return LatentSeekOptimizer._find_description_token_positions(self, text)
    
    def _extract_description_from_text(self, text):
        return LatentSeekOptimizer._extract_description_from_text(self, text)

optimizer = DummyOptimizer()

print("Generated code:")
print("="*80)
print(generated_code)
print("="*80)

# Find description positions
desc_start, desc_end = optimizer._find_description_token_positions(generated_code)

print(f"\nFound description at character positions: {desc_start}-{desc_end}")
if desc_start is not None and desc_end is not None:
    print(f"Length: {desc_end - desc_start}")
    print(f"Extracted text: '{generated_code[desc_start:desc_end]}'")
    
    # Extract description
    description = optimizer._extract_description_from_text(generated_code)
    print(f"\nProcessed description: '{description}'")

# Now let's check what's at position 199-200
print(f"\nText at positions 199-200: '{generated_code[199:200]}'")
print(f"Text around positions 190-210: '{generated_code[190:210]}'")

# Check tokenization
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Tokenize the full text
tokens = tokenizer.tokenize(generated_code)
print(f"\nTotal tokens: {len(tokens)}")
print(f"Tokens 195-205: {tokens[195:205] if len(tokens) > 205 else tokens[195:]}")

# Find description in tokens
desc_text = "# description:"
for i, token in enumerate(tokens):
    if "description" in token.lower():
        print(f"\nFound 'description' at token {i}: '{token}'")