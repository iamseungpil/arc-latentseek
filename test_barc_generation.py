#!/usr/bin/env python3
"""Test BARC generation to understand structure"""

import sys
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

from src.generators.barc_generator_simple import BARCGeneratorSimple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model
model_name = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)

# Generate code
generator = BARCGeneratorSimple(model, tokenizer)
result = generator.generate("2a5f8217")

print("Generated code:")
print("="*80)
print(result["code"])
print("="*80)

# Tokenize and analyze
inputs = tokenizer(result["code"], return_tensors="pt")
input_ids = inputs["input_ids"]

print(f"\nTotal tokens: {input_ids.shape[1]}")

# Find description line by line
lines = result["code"].split('\n')
for i, line in enumerate(lines):
    if '# description:' in line:
        print(f"\nDescription starts at line {i}: {repr(line)}")
        # Show next lines until we hit 'def main'
        for j in range(i+1, len(lines)):
            if 'def main' in lines[j]:
                print(f"Description ends at line {j-1}")
                print(f"Total description lines: {j-i-1}")
                break
            print(f"  Line {j}: {repr(lines[j])}")
        break

# Find token positions
tokens = []
for i in range(input_ids.shape[1]):
    token_id = input_ids[0, i].item()
    token_text = tokenizer.decode([token_id])
    tokens.append(token_text)

# Find description in tokens
desc_start = None
for i, token in enumerate(tokens):
    if 'description' in token and i+1 < len(tokens) and ':' in tokens[i+1]:
        desc_start = i + 2  # Skip "description:" 
        print(f"\nDescription starts at token {desc_start}")
        break

if desc_start:
    # Find where description ends (look for "def")
    for i in range(desc_start, len(tokens)):
        if 'def' in tokens[i]:
            desc_end = i
            print(f"Description ends at token {desc_end}")
            print(f"Description token length: {desc_end - desc_start}")
            
            # Show description tokens
            print("\nDescription tokens:")
            for j in range(desc_start, min(desc_end, desc_start + 20)):
                print(f"  {j}: {repr(tokens[j])}")
            if desc_end - desc_start > 20:
                print(f"  ... ({desc_end - desc_start - 20} more tokens)")
            break