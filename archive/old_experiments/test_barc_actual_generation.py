#!/usr/bin/env python3
"""Test actual BARC generation to see what's happening"""

import sys
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

from src.generators.barc_generator_simple import BARCGeneratorSimple
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("Loading model...")
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

print("Generating code...")
generator = BARCGeneratorSimple(model, tokenizer)

# Test with first problem
result = generator.generate("2a5f8217")

print("\n" + "="*80)
print("RAW RESPONSE:")
print("="*80)
print(result["raw_response"][:1000] + "..." if len(result["raw_response"]) > 1000 else result["raw_response"])

print("\n" + "="*80)
print("EXTRACTED CODE:")
print("="*80)
print(result["code"])

print("\n" + "="*80)
print("METADATA:")
print("="*80)
print(f"Concepts: {result['concepts']}")
print(f"Description: {result['description']}")
print(f"Plan: {result['plan']}")

# Check if it's just template
if "[describe the transformation rule]" in result["code"]:
    print("\n⚠️ WARNING: Generator returned template code, not actual solution!")
else:
    print("\n✅ Generator produced actual code with description")