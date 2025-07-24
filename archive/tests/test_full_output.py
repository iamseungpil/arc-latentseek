#!/usr/bin/env python3
"""
Test to show full prompt, code, and description
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader
from src.generators.barc_generator_fixed import BARCGeneratorFixed

# Load test problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems(split="validation", num_problems=1)
problem = problems[0]

print("="*80)
print("FULL OUTPUT TEST")
print("="*80)

# Initialize generator
generator = BARCGeneratorFixed("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")

# Show the prompt
prompt = generator._create_prompt(problem)
print("\n1. FULL PROMPT:")
print("-"*80)
print("System prompt:")
print(prompt[0]["content"])
print("\nUser prompt (first 1000 chars):")
user_prompt = prompt[1]["content"]
print(user_prompt[:1000] + "..." if len(user_prompt) > 1000 else user_prompt)

# Generate response
candidates = generator.generate(problem, num_candidates=1, temperature=0.7)

if candidates:
    candidate = candidates[0]
    
    print("\n\n2. FULL RAW RESPONSE:")
    print("-"*80)
    print(candidate.raw_response)
    
    print("\n\n3. EXTRACTED CODE:")
    print("-"*80)
    print(candidate.code)
    
    print("\n\n4. EXTRACTED ELEMENTS:")
    print("-"*80)
    print(f"Concepts: {candidate.concepts}")
    print(f"Description: {candidate.description}")
    print(f"Plan: {candidate.plan}")
    
    # Check what function is in the code
    print("\n\n5. FUNCTION ANALYSIS:")
    print("-"*80)
    print(f"Has 'def transform': {'def transform' in candidate.code}")
    print(f"Has 'def main': {'def main' in candidate.code}")
    
    # Look for both patterns in raw response
    if 'def main' in candidate.raw_response and 'def transform' not in candidate.code:
        print("\nWARNING: 'def main' found in response but not extracted!")
        start = candidate.raw_response.find('def main')
        print(f"Position of 'def main': {start}")