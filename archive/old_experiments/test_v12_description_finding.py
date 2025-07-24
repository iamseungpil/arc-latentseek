#!/usr/bin/env python3
"""Test V12 description finding with actual generated code"""

import sys
import torch
import re

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

import arc
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.generators.barc_generator_simple import BARCGeneratorSimple

def test_description_finding():
    """Test description finding with actual BARC generation"""
    
    print("Loading model and tokenizer...")
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
    
    # Generate for first problem
    generator = BARCGeneratorSimple(model, tokenizer)
    problem_id = "2a5f8217"
    
    print(f"\nGenerating solution for {problem_id}...")
    result = generator.generate(problem_id)
    
    print("\n=== GENERATED CODE ===")
    print(result["code"])
    
    # Tokenize the full response
    full_response = result["raw_response"]
    inputs = tokenizer(full_response, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    
    print(f"\n=== TOKEN ANALYSIS ===")
    print(f"Total tokens: {input_ids.shape[1]}")
    
    # Method 1: Simple regex on code
    desc_pattern = r'# description:\s*\n((?:# .*\n)*)'
    match = re.search(desc_pattern, result["code"])
    
    if match:
        desc_text = match.group(1)
        desc_lines = desc_text.count('\n')
        print(f"\nDescription found ({desc_lines} lines):")
        print(desc_text)
    
    # Method 2: Token-level analysis
    text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    
    # Find where code block starts
    code_start = text.find("```python")
    if code_start >= 0:
        code_text = text[code_start:]
        
        # Find description in code
        desc_match = re.search(r'# description:\s*\n((?:# [^\n]+\n)+)', code_text)
        if desc_match:
            desc_content = desc_match.group(1)
            print(f"\nDescription content:")
            print(desc_content)
            
            # Count tokens in description
            desc_only_tokens = tokenizer(desc_content, return_tensors="pt")
            print(f"Description token count: {desc_only_tokens['input_ids'].shape[1]}")
    
    # Method 3: Better token mapping
    print("\n=== BETTER TOKEN MAPPING ===")
    
    # Decode each token individually
    tokens = []
    for i in range(input_ids.shape[1]):
        token_id = input_ids[0, i].item()
        token_text = tokenizer.decode([token_id])
        tokens.append((i, token_text))
    
    # Find description start and end
    desc_start_idx = None
    desc_end_idx = None
    in_description = False
    
    for i, (idx, token) in enumerate(tokens):
        # Look for "description:"
        if "description" in token and i+1 < len(tokens) and ":" in tokens[i+1][1]:
            desc_start_idx = i + 2  # Skip "description:"
            in_description = True
            print(f"Found description start at token {desc_start_idx}")
        
        # Look for end of description (next non-comment line)
        if in_description and desc_start_idx is not None:
            # Check if we've hit a non-comment line
            if i >= desc_start_idx + 2 and token.strip() and not token.strip().startswith('#'):
                # Check if this is "def main" or similar
                if "def" in token or (i+1 < len(tokens) and "def" in tokens[i+1][1]):
                    desc_end_idx = i
                    in_description = False
                    print(f"Found description end at token {desc_end_idx}")
                    break
    
    if desc_start_idx and desc_end_idx:
        desc_length = desc_end_idx - desc_start_idx
        print(f"\nDescription spans tokens [{desc_start_idx}:{desc_end_idx}] ({desc_length} tokens)")
        
        # Show some description tokens
        print("\nFirst 10 description tokens:")
        for i in range(desc_start_idx, min(desc_start_idx + 10, desc_end_idx)):
            print(f"  Token {i}: '{tokens[i][1]}'")

if __name__ == "__main__":
    print("Testing V12 description finding...")
    test_description_finding()