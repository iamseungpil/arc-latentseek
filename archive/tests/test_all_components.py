#!/usr/bin/env python3
"""
Comprehensive test of all components:
1. Code generation
2. Code extraction (including description)
3. LatentSeek optimization
"""

import sys
from pathlib import Path
import torch
import json

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader
from src.generators.barc_generator_fixed import BARCGeneratorFixed
from src.executors import CodeExecutor
from src.optimizers.latent_optimizer_fixed import FixedLatentSeekOptimizer

print("="*80)
print("COMPREHENSIVE COMPONENT TEST")
print("="*80)

# Load a problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems(split="validation", num_problems=1)
problem = problems[0]

print(f"\nTesting with problem: {problem.uid}")
print(f"Problem has {len(problem.train_pairs)} training examples")

# Initialize components
print("\n" + "="*80)
print("1. TESTING CODE GENERATION")
print("="*80)

generator = BARCGeneratorFixed("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")

# Generate multiple candidates to test consistency
candidates = generator.generate(problem, num_candidates=2, temperature=0.7)

for i, candidate in enumerate(candidates):
    print(f"\n--- Candidate {i+1} ---")
    print(f"Raw response length: {len(candidate.raw_response)} chars")
    print(f"\nConcepts: {candidate.concepts}")
    print(f"\nDescription: {candidate.description}")
    print(f"\nCode extracted: {'YES' if candidate.code else 'NO'}")
    
    if candidate.code:
        print(f"Code length: {len(candidate.code)} chars")
        print("\nCode preview (first 500 chars):")
        print("-"*40)
        print(candidate.code[:500] + "..." if len(candidate.code) > 500 else candidate.code)
    else:
        print("\nRAW RESPONSE (first 1000 chars):")
        print("-"*40)
        print(candidate.raw_response[:1000] + "..." if len(candidate.raw_response) > 1000 else candidate.raw_response)

# Test code execution
print("\n" + "="*80)
print("2. TESTING CODE EXECUTION")
print("="*80)

executor = CodeExecutor()
for i, candidate in enumerate(candidates):
    if candidate.code:
        print(f"\n--- Testing Candidate {i+1} ---")
        result = executor.execute(candidate.code, problem)
        print(f"Execution success: {result.success}")
        print(f"Errors: {result.error_messages if result.error_messages else 'None'}")
        print(f"Accuracy: {result.accuracy:.1%}")
        if result.output_grids:
            print(f"Output grids generated: {len(result.output_grids)}")

# Test description extraction in detail
print("\n" + "="*80)
print("3. TESTING DESCRIPTION EXTRACTION")
print("="*80)

# Test with a sample response that has description
test_response = """# concepts: grid transformation, color mapping, pattern recognition
# description: This problem involves identifying colored regions in the input grid
# and transforming them according to a specific pattern. The transformation
# appears to involve replacing certain colors with others based on their position.

```python
import numpy as np

def transform(input_grid):
    # Implementation here
    output_grid = input_grid.copy()
    return output_grid
```"""

from src.generators.code_parser import extract_code_elements, parse_code

concepts, description, plan = extract_code_elements(test_response)
print(f"Test extraction:")
print(f"Concepts: {concepts}")
print(f"Description: {description}")
print(f"Description length: {len(description) if description else 0} chars")

# Test LatentSeek optimization
print("\n" + "="*80)
print("4. TESTING LATENTSEEK OPTIMIZATION")
print("="*80)

if candidates and candidates[0].code:
    # Create dummy evaluator for testing
    class SimpleEvaluator:
        def __init__(self, executor, problem):
            self.executor = executor
            self.problem = problem
        
        def evaluate(self, problem, output, execution_result, prefix=""):
            # Simple accuracy-based reward
            reward = -1.0 + execution_result.accuracy
            
            from dataclasses import dataclass
            @dataclass
            class SimpleResult:
                total_reward: float
                
            return SimpleResult(total_reward=reward)
    
    # Initialize optimizer
    optimizer = FixedLatentSeekOptimizer(
        barc_generator=generator,
        code_executor=executor,
        glm_evaluator=SimpleEvaluator(executor, problem),
        lr=0.03,
        max_steps=5,  # Just a few steps for testing
        k=0.2,
        reward_threshold=0.5
    )
    
    # Test optimization
    candidate = candidates[0]
    initial_result = executor.execute(candidate.code, problem)
    initial_reward = -1.0 + initial_result.accuracy
    
    print(f"\nInitial accuracy: {initial_result.accuracy:.1%}")
    print(f"Initial reward: {initial_reward:.3f}")
    
    print("\nRunning LatentSeek optimization...")
    
    try:
        # Test hidden states extraction first
        hidden_states = generator.get_hidden_states(problem, candidate)
        print(f"Hidden states extracted: {len(hidden_states)} tokens")
        print(f"Hidden state shape: {hidden_states[0].shape if hidden_states else 'None'}")
        
        # Run optimization
        opt_result = optimizer.optimize(problem, candidate, initial_reward)
        
        print(f"\nOptimization completed!")
        print(f"Steps taken: {opt_result.optimization_steps}")
        print(f"Converged: {opt_result.converged}")
        print(f"Reward history: {[f'{r:.3f}' for r in opt_result.reward_history]}")
        
        # Test final result
        final_result = executor.execute(opt_result.final_output.code, problem)
        print(f"\nFinal accuracy: {final_result.accuracy:.1%}")
        print(f"Final reward: {-1.0 + final_result.accuracy:.3f}")
        print(f"Improvement: {final_result.accuracy - initial_result.accuracy:.1%}")
        
    except Exception as e:
        print(f"\nOptimization error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)