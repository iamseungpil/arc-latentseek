#!/usr/bin/env python3
"""
Test BARC generation and code extraction
"""

import sys
from pathlib import Path

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
print(f"Problem has {len(problem.train_pairs)} training examples")

# Initialize generator
generator = BARCGenerator("Qwen/Qwen2.5-0.5B-Instruct")

# Test prompt creation
prompt = generator._create_prompt(problem)
print("\n" + "="*80)
print("PROMPT:")
print("="*80)
for msg in prompt:
    print(f"\n[{msg['role'].upper()}]:")
    print(msg['content'][:500] + "..." if len(msg['content']) > 500 else msg['content'])

# Generate response
print("\n" + "="*80)
print("GENERATING RESPONSE...")
print("="*80)

candidates = generator.generate(problem, num_candidates=1)
if candidates:
    candidate = candidates[0]
    print("\nRAW RESPONSE:")
    print("-"*80)
    print(candidate.raw_response[:1000] + "..." if len(candidate.raw_response) > 1000 else candidate.raw_response)
    
    print("\nEXTRACTED CODE:")
    print("-"*80)
    print(candidate.code if candidate.code else "NO CODE EXTRACTED!")
    
    print("\nEXTRACTED DESCRIPTION:")
    print("-"*80)
    print(candidate.description if candidate.description else "NO DESCRIPTION")
    
    # Test code extraction manually
    print("\n" + "="*80)
    print("MANUAL CODE EXTRACTION TEST:")
    print("="*80)
    
    import re
    
    # Pattern 1: ```python blocks
    python_pattern = r'```python\n(.*?)\n```'
    matches = re.findall(python_pattern, candidate.raw_response, re.DOTALL)
    if matches:
        print(f"Found {len(matches)} python code blocks")
        for i, match in enumerate(matches):
            print(f"\nCode block {i+1}:")
            print(match[:200] + "..." if len(match) > 200 else match)
    else:
        print("No ```python blocks found")
    
    # Pattern 2: def main or def transform
    if 'def main(' in candidate.raw_response:
        print("\nFound 'def main(' in response")
        start = candidate.raw_response.find('def main(')
        print(f"Position: {start}")
        print("Context:")
        print(candidate.raw_response[max(0, start-50):start+200])
    
    if 'def transform(' in candidate.raw_response:
        print("\nFound 'def transform(' in response")
        start = candidate.raw_response.find('def transform(')
        print(f"Position: {start}")
    
    # Check what format the response is in
    print("\n" + "="*80)
    print("RESPONSE FORMAT ANALYSIS:")
    print("="*80)
    print(f"Contains '```python': {'```python' in candidate.raw_response}")
    print(f"Contains 'def ': {'def ' in candidate.raw_response}")
    print(f"Contains 'import ': {'import ' in candidate.raw_response}")
    print(f"Contains '#': {'#' in candidate.raw_response}")
    
else:
    print("No candidates generated!")