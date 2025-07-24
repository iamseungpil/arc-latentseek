#!/usr/bin/env python3
"""
Debug BARC generation issue
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader
from src.generators import BARCGenerator

# Load a problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems(split="validation", num_problems=1)
problem = problems[0]

print(f"Testing with problem: {problem.uid}")

# Initialize generator
generator = BARCGenerator("Qwen/Qwen2.5-0.5B-Instruct")

# Get prompt
prompt = generator._create_prompt(problem)
text = generator.tokenizer.apply_chat_template(
    prompt, 
    tokenize=False, 
    add_generation_prompt=True
)

print("\nFULL PROMPT WITH CHAT TEMPLATE:")
print("="*80)
print(text[:1000] + "..." if len(text) > 1000 else text)
print("\nPrompt length:", len(text))

# Tokenize
inputs = generator.tokenizer(text, return_tensors="pt").to(generator.model.device)
print(f"\nInput IDs shape: {inputs.input_ids.shape}")
print(f"Input length: {inputs.input_ids.shape[1]} tokens")

# Try different generation parameters
print("\n" + "="*80)
print("TESTING GENERATION WITH DIFFERENT PARAMETERS:")
print("="*80)

# Test 1: Simple generation
print("\nTest 1: Simple generation")
with torch.no_grad():
    outputs = generator.model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False,  # Greedy
        pad_token_id=generator.tokenizer.pad_token_id,
        eos_token_id=generator.tokenizer.eos_token_id,
    )
    
input_len = inputs.input_ids.shape[1]
response = generator.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
print(f"Response length: {len(response)}")
print(f"Response: '{response[:200]}...'" if len(response) > 200 else f"Response: '{response}'")

# Test 2: With temperature
print("\n\nTest 2: With temperature")
with torch.no_grad():
    outputs = generator.model.generate(
        **inputs,
        max_new_tokens=500,
        temperature=0.7,
        do_sample=True,
        pad_token_id=generator.tokenizer.pad_token_id,
        eos_token_id=generator.tokenizer.eos_token_id,
    )
    
response = generator.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
print(f"Response length: {len(response)}")
print(f"Response preview: '{response[:500]}...'" if len(response) > 500 else f"Response: '{response}'")

# Check tokenizer settings
print("\n" + "="*80)
print("TOKENIZER SETTINGS:")
print("="*80)
print(f"EOS token: '{generator.tokenizer.eos_token}' (ID: {generator.tokenizer.eos_token_id})")
print(f"PAD token: '{generator.tokenizer.pad_token}' (ID: {generator.tokenizer.pad_token_id})")
print(f"Chat template: {generator.tokenizer.chat_template[:200]}..." if generator.tokenizer.chat_template else "No chat template")

# Test without skip_special_tokens
print("\n\nTest 3: Full output with special tokens")
with torch.no_grad():
    outputs = generator.model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=generator.tokenizer.pad_token_id,
        eos_token_id=generator.tokenizer.eos_token_id,
    )

full_response = generator.tokenizer.decode(outputs[0], skip_special_tokens=False)
print("Full output (first 500 chars):")
print(full_response[:500])