#!/usr/bin/env python
"""Test the fixed optimizer implementation"""

import sys
sys.path.append('/home/ubuntu/arc-latentseek')

import torch
from src.data import ARCDataLoader
from src.generators import BARCGenerator, BARCOutput
from src.executors import CodeExecutor
from src.evaluators import GLMEvaluator
from src.optimizers import LatentSeekOptimizer

# Test with a simple problem
loader = ARCDataLoader()
problem = loader.get_problem_by_id('2072aba6')

print("Testing fixed LatentSeek optimizer...")
print("="*80)

# Initialize components
print("1. Initializing components...")
barc_generator = BARCGenerator()
code_executor = CodeExecutor()
glm_evaluator = GLMEvaluator()

# Create optimizer
optimizer = LatentSeekOptimizer(
    barc_generator=barc_generator,
    code_executor=code_executor,
    glm_evaluator=glm_evaluator
)

print("2. Generating initial solution...")
outputs = barc_generator.generate(problem, temperature=0.7, num_candidates=1)
if not outputs:
    print("Failed to generate initial solution")
    sys.exit(1)

initial_output = outputs[0]
print(f"Generated code ({len(initial_output.code)} chars):")
print("-"*40)
print(initial_output.code[:200] + "...")
print("-"*40)

# Test _generate_with_description_mapping
print("\n3. Testing _generate_with_description_mapping...")
try:
    result = optimizer._generate_with_description_mapping(problem, initial_output)
    if result:
        hidden_states_list, desc_start, desc_end = result
        print(f"✅ Success!")
        print(f"   - Hidden states: {len(hidden_states_list)}")
        print(f"   - Description tokens: {desc_start}-{desc_end}")
        print(f"   - Description length: {desc_end - desc_start if desc_start and desc_end else 0}")
    else:
        print("❌ Failed to generate description mapping")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n4. Testing optimize_description_based...")
try:
    # Execute to get initial reward
    exec_result = code_executor.execute(initial_output.code, problem)
    eval_result = glm_evaluator.evaluate(problem, initial_output, exec_result, "test")
    initial_reward = eval_result.total_reward
    
    print(f"Initial accuracy: {exec_result.accuracy:.2%}")
    print(f"Initial reward: {initial_reward:.3f}")
    
    # Run optimization
    opt_result = optimizer.optimize_description_based(
        problem=problem,
        initial_output=initial_output,
        initial_reward=initial_reward
    )
    
    print(f"\n✅ Optimization completed!")
    print(f"   - Steps: {opt_result.optimization_steps}")
    print(f"   - Final reward: {opt_result.reward_history[-1]:.3f}")
    print(f"   - Converged: {opt_result.converged}")
    
except Exception as e:
    print(f"❌ Error during optimization: {e}")
    import traceback
    traceback.print_exc()

print("\nTest completed!")