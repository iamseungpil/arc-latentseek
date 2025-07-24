#!/usr/bin/env python3
"""
Simple test of BARC generation without unsloth
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader
from src.generators.barc_generator_fixed import BARCGeneratorFixed

print("Loading data...")
data_loader = ARCDataLoader()
problems = data_loader.get_problems(split="validation", num_problems=1)
problem = problems[0]

print(f"\nTesting with problem: {problem.uid}")
print(f"Problem has {len(problem.train_pairs)} training examples")

print("\nInitializing generator...")
generator = BARCGeneratorFixed("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")

print("\nGenerating response...")
candidates = generator.generate(problem, num_candidates=1, temperature=0.7)

if candidates:
    candidate = candidates[0]
    print("\nRaw response (first 500 chars):")
    print("-"*40)
    print(candidate.raw_response[:500] + "..." if len(candidate.raw_response) > 500 else candidate.raw_response)
    
    print(f"\n\nConcepts: {candidate.concepts}")
    print(f"Description: {candidate.description[:200] + '...' if candidate.description and len(candidate.description) > 200 else candidate.description}")
    print(f"\nCode extracted: {'YES' if candidate.code else 'NO'}")
    
    if candidate.code:
        print(f"Code length: {len(candidate.code)} chars")
        print("\nCode preview (first 300 chars):")
        print("-"*40)
        print(candidate.code[:300] + "..." if len(candidate.code) > 300 else candidate.code)
    else:
        print("\nTrying to extract code manually...")
        if "def transform" in candidate.raw_response:
            start = candidate.raw_response.find("def transform")
            print(f"Found 'def transform' at position {start}")
else:
    print("No candidates generated!")