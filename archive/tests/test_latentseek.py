#!/usr/bin/env python3
"""
Test LatentSeek optimization
"""

import sys
from pathlib import Path
import torch
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader
from src.generators.barc_generator_fixed import BARCGeneratorFixed
from src.executors import CodeExecutor
from src.optimizers.latent_optimizer_fixed import FixedLatentSeekOptimizer

# Simple evaluator for testing
class SimpleAccuracyEvaluator:
    def __init__(self, executor, problem):
        self.executor = executor
        self.problem = problem
    
    def evaluate(self, problem, output, execution_result, prefix=""):
        # Simple accuracy-based reward
        reward = -1.0 + execution_result.accuracy
        
        @dataclass
        class SimpleResult:
            total_reward: float
            
        return SimpleResult(total_reward=reward)

print("="*80)
print("TESTING LATENTSEEK OPTIMIZATION")
print("="*80)

# Load a problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems(split="validation", num_problems=1)
problem = problems[0]

print(f"\nTesting with problem: {problem.uid}")
print(f"Problem has {len(problem.train_pairs)} training examples")

# Initialize components
print("\nInitializing components...")
generator = BARCGeneratorFixed("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
executor = CodeExecutor()

# Generate initial solution
print("\n1. Generating initial solution...")
candidates = generator.generate(problem, num_candidates=1, temperature=0.7)

if not candidates or not candidates[0].code:
    print("Failed to generate initial solution!")
    exit(1)

candidate = candidates[0]
print(f"Initial code generated: {len(candidate.code)} chars")
print(f"Concepts: {candidate.concepts}")
print(f"Description: {candidate.description[:100] + '...' if candidate.description and len(candidate.description) > 100 else candidate.description}")

# Execute initial solution
print("\n2. Testing initial solution...")
initial_result = executor.execute(candidate.code, problem)
initial_reward = -1.0 + initial_result.accuracy

print(f"Initial execution success: {initial_result.success}")
print(f"Initial accuracy: {initial_result.accuracy:.1%}")
print(f"Initial reward: {initial_reward:.3f}")

# Test hidden states extraction
print("\n3. Testing hidden states extraction...")
try:
    hidden_states = generator.get_hidden_states(problem, candidate)
    print(f"Hidden states extracted: {len(hidden_states)} tokens")
    if hidden_states:
        print(f"Hidden state shape: {hidden_states[0].shape}")
        print(f"Hidden state device: {hidden_states[0].device}")
except Exception as e:
    print(f"Error extracting hidden states: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Initialize optimizer
print("\n4. Initializing LatentSeek optimizer...")
optimizer = FixedLatentSeekOptimizer(
    barc_generator=generator,
    code_executor=executor,
    glm_evaluator=SimpleAccuracyEvaluator(executor, problem),
    lr=0.03,
    max_steps=10,
    k=0.2,
    reward_threshold=0.5
)

# Run optimization
print("\n5. Running LatentSeek optimization...")
print(f"Parameters: lr={optimizer.lr}, k={optimizer.k}, max_steps={optimizer.max_steps}")

try:
    opt_result = optimizer.optimize(problem, candidate, initial_reward)
    
    print(f"\n6. Optimization completed!")
    print(f"Steps taken: {opt_result.optimization_steps}")
    print(f"Converged: {opt_result.converged}")
    print(f"Reward history: {[f'{r:.3f}' for r in opt_result.reward_history]}")
    
    # Test final result
    if opt_result.final_output.code:
        final_result = executor.execute(opt_result.final_output.code, problem)
        final_accuracy = final_result.accuracy
        final_reward = -1.0 + final_accuracy
        
        print(f"\n7. Final results:")
        print(f"Final accuracy: {final_accuracy:.1%}")
        print(f"Final reward: {final_reward:.3f}")
        print(f"Improvement: {(final_accuracy - initial_result.accuracy):.1%}")
        
        # Show code changes
        if candidate.code != opt_result.final_output.code:
            print("\nCode was modified during optimization!")
            print(f"Original code length: {len(candidate.code)}")
            print(f"Optimized code length: {len(opt_result.final_output.code)}")
        else:
            print("\nCode remained the same after optimization.")
    else:
        print("\nNo code in final output!")
        
except Exception as e:
    print(f"\nOptimization error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)