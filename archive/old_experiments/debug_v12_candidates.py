#!/usr/bin/env python3
"""Debug why V12 candidates keep failing"""

import sys
import torch
import re

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

import arc
from transformers import AutoTokenizer, AutoModelForCausalLM

def debug_description_finding():
    """Debug description token finding"""
    
    # Test code
    test_code = '''### System:
You are a world-class puzzle solver...

### Response:
```python
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
    return output_grid
```'''
    
    # Load tokenizer
    model_name = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize
    inputs = tokenizer(test_code, return_tensors="pt")
    input_ids = inputs["input_ids"]
    
    print("=== TOKEN ANALYSIS ===")
    print(f"Total tokens: {input_ids.shape[1]}")
    
    # Decode tokens one by one
    tokens = []
    for i in range(input_ids.shape[1]):
        token_id = input_ids[0, i].item()
        token_text = tokenizer.decode([token_id])
        tokens.append((i, token_id, token_text))
    
    # Find description
    desc_start = None
    desc_end = None
    in_description = False
    
    print("\n=== SEARCHING FOR DESCRIPTION ===")
    for i, (idx, token_id, token_text) in enumerate(tokens):
        if "description" in token_text.lower() and ":" in tokens[i+1][2] if i+1 < len(tokens) else False:
            print(f"Found 'description:' at token {idx}")
            desc_start = idx + 2  # Skip "description:"
            in_description = True
        
        if in_description and token_text.strip() and not token_text.startswith('#'):
            # End of description
            desc_end = idx
            in_description = False
            
        if idx >= 30 and idx <= 70:  # Print relevant range
            marker = " <--" if idx == desc_start else (" <-- END" if idx == desc_end else "")
            print(f"Token {idx}: '{token_text}'{marker}")
    
    print(f"\nDescription tokens: [{desc_start}:{desc_end}] (length: {desc_end - desc_start if desc_end else 0})")
    
    # Test hidden state modification
    print("\n=== TESTING HIDDEN STATE MODIFICATION ===")
    
    # Simulate what happens when we modify description hidden states
    print("1. Original generation should work fine")
    print("2. Modified hidden states often break the syntax")
    print("3. Possible causes:")
    print("   - Hidden states encode not just content but also structure")
    print("   - Modifying description affects subsequent token generation")
    print("   - Temperature too high/low for meaningful modifications")
    
    return desc_start, desc_end

def test_temperature_effects():
    """Test how temperature affects generation"""
    print("\n=== TEMPERATURE EFFECTS ===")
    
    temperatures = [0.1, 0.5, 0.7, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\nTemperature {temp}:")
        print(f"  - Lower temp ({temp} < 0.7): More deterministic, less variation")
        print(f"  - Higher temp ({temp} > 1.0): More random, might break syntax")
    
    print("\nRecommendation: Try temperature 0.3-0.7 for description optimization")

def analyze_v12_issues():
    """Analyze why V12 keeps failing"""
    print("\n=== V12 FAILURE ANALYSIS ===")
    
    print("1. Description token range issue:")
    print("   - Only 7 tokens found (too short)")
    print("   - May be missing multi-line description")
    
    print("\n2. Hidden state modification issue:")
    print("   - Direct modification breaks token dependencies")
    print("   - Need more careful approach (e.g., smaller modifications)")
    
    print("\n3. Generation parameters:")
    print("   - Temperature might be too high")
    print("   - max_new_tokens might cut off code")
    
    print("\n4. Candidate evaluation:")
    print("   - All 8 candidates failing suggests systematic issue")
    print("   - Need to check what errors they produce")

if __name__ == "__main__":
    print("Debugging V12 candidate failures...\n")
    
    # Debug description finding
    desc_start, desc_end = debug_description_finding()
    
    # Test temperature effects
    test_temperature_effects()
    
    # Analyze issues
    analyze_v12_issues()
    
    print("\n✅ Debug complete!")