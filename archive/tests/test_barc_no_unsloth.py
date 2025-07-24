#!/usr/bin/env python3
"""
Test BARC generation without unsloth optimization
"""

import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader
from src.generators import BARCGenerator

# Monkey patch to disable unsloth
class BARCGeneratorNoUnsloth(BARCGenerator):
    def _load_model(self):
        """Load BARC model without unsloth"""
        print(f"Loading BARC model without unsloth: {self.model_name}")
        
        # Standard transformers loading
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("BARC model loaded successfully (no unsloth)")

# Load a problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems(split="validation", num_problems=1)
problem = problems[0]

print(f"\nTesting with problem: {problem.uid}")
print(f"Problem has {len(problem.train_pairs)} training examples")

# Initialize generator without unsloth
generator = BARCGeneratorNoUnsloth("Qwen/Qwen2.5-0.5B-Instruct")

# Generate response
print("\n" + "="*80)
print("GENERATING RESPONSE...")
print("="*80)

candidates = generator.generate(problem, num_candidates=1, temperature=0.7)
if candidates:
    candidate = candidates[0]
    print("\nRAW RESPONSE (first 1000 chars):")
    print("-"*80)
    print(candidate.raw_response[:1000] + "..." if len(candidate.raw_response) > 1000 else candidate.raw_response)
    
    print("\n\nEXTRACTED CODE:")
    print("-"*80)
    if candidate.code:
        print(candidate.code)
    else:
        print("NO CODE EXTRACTED!")
        
        # Try to find code manually
        import re
        if '```python' in candidate.raw_response:
            matches = re.findall(r'```python\n(.*?)\n```', candidate.raw_response, re.DOTALL)
            if matches:
                print("\nFound code block manually:")
                print(matches[0])
else:
    print("No candidates generated!")