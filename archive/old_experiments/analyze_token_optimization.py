#!/usr/bin/env python3
"""
Analyze which tokens are being optimized in LatentSeek
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")

# Example BARC response
example_response = """Let me analyze this puzzle step by step.

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
    # Process each cell
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            # Check neighbors and apply rules
            pass
    return output_grid
```"""

# Tokenize the response
tokens = tokenizer.encode(example_response)
decoded_tokens = [tokenizer.decode([token]) for token in tokens]

print("Token Analysis of BARC Response")
print("=" * 80)
print(f"Total tokens in response: {len(tokens)}")
print()

# Find where key elements appear in token sequence
def find_token_positions(tokens, decoded_tokens, search_str):
    positions = []
    for i in range(len(decoded_tokens) - len(search_str.split())):
        window = ''.join(decoded_tokens[i:i+len(search_str.split())*2])
        if search_str.lower() in window.lower():
            positions.append(i)
    return positions

# Find approximate positions of key elements
print("Key element positions (approximate):")
for i, (token, decoded) in enumerate(zip(tokens, decoded_tokens)):
    if "concept" in decoded.lower():
        print(f"  Token {i}: '{decoded}' - CONCEPTS START")
    if "description" in decoded.lower():
        print(f"  Token {i}: '{decoded}' - DESCRIPTION START")
    if "```" in decoded:
        print(f"  Token {i}: '{decoded}' - CODE BLOCK START")
    if "def" in decoded and i < len(decoded_tokens)-1 and "transform" in decoded_tokens[i+1]:
        print(f"  Token {i}: '{decoded}' - DEF TRANSFORM START")

print()

# Calculate what 20% optimization would cover
k = 0.2
update_length = int(k * len(tokens))
print(f"With k={k}, LatentSeek would optimize {update_length} tokens")
print(f"This covers tokens 0 to {update_length-1}")
print()

print("What gets optimized (first 20% of tokens):")
print("-" * 40)
optimized_text = tokenizer.decode(tokens[:update_length])
print(optimized_text)
print("-" * 40)

print("\nWhat gets MISSED (remaining 80%):")
print("-" * 40)
missed_text = tokenizer.decode(tokens[update_length:])
print(missed_text)
print("-" * 40)

# Find where code actually starts
code_start_token = None
for i in range(len(decoded_tokens)-1):
    if "python" in decoded_tokens[i].lower() or ("def" in decoded_tokens[i] and "transform" in decoded_tokens[i+1]):
        code_start_token = i
        break

if code_start_token:
    code_percentage = (code_start_token / len(tokens)) * 100
    print(f"\nCode starts at token {code_start_token} ({code_percentage:.1f}% into response)")
    print(f"Current optimization covers tokens 0-{update_length-1}")
    print(f"Code tokens start at {code_start_token}")
    
    if code_start_token >= update_length:
        print("\n⚠️  WARNING: Current optimization MISSES the actual code!")
        print(f"   The code starts at token {code_start_token}, but optimization only covers up to token {update_length-1}")
    else:
        overlap = update_length - code_start_token
        print(f"\n✓ Optimization covers {overlap} tokens of actual code")

print("\nCONCLUSION:")
print("The current LatentSeek implementation optimizes the BEGINNING of the response,")
print("which often contains explanatory text rather than the actual code.")
print("To optimize code generation, we should identify where 'def transform' starts")
print("and optimize tokens from that point forward.")