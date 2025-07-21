#!/usr/bin/env python3
"""Simplified comparative test without complex evaluators"""
import os
import sys
import json
from datetime import datetime

# Set GPU
gpu_id = sys.argv[1] if len(sys.argv) > 1 else "5"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader
from src.generators import BARCGenerator
from src.executors import CodeExecutor
from src.evaluators import GLMEvaluator

print(f"Running simplified test on GPU {gpu_id}")

# Load components
loader = ARCDataLoader()
barc_gen = BARCGenerator()
executor = CodeExecutor()
glm_eval = GLMEvaluator()

# Test one problem
problem_id = 'f3e62deb'
problem = loader.get_problem_by_id(problem_id)

print(f"\nTesting problem: {problem_id}")

# Test 1: Basic generation (no GLM)
print("\n1. Basic generation (no GLM):")
candidates = barc_gen.generate(problem, num_candidates=1)
if candidates:
    candidate = candidates[0]
    print(f"✓ Generated code ({len(candidate.code)} chars)")
    
    # Execute
    result = executor.execute(candidate.code, problem)
    print(f"✓ Execution: {'Success' if result.success else 'Failed'}")
    print(f"  Accuracy: {result.accuracy:.2%}")

# Test 2: With GLM description (manual)
print("\n2. With GLM description:")
# Generate GLM description
from src.executors import GridRenderer
renderer = GridRenderer()
image_path = f"/tmp/{problem_id}_test.png"
renderer.render_arc_problem(problem, image_path)

prompt = "Describe the pattern in this ARC problem briefly."
try:
    glm_desc = glm_eval._run_glm_inference(image_path, prompt)
    print(f"✓ GLM description: {glm_desc[:100]}...")
    
    # Manually modify the prompt
    original_create_prompt = barc_gen._create_prompt
    
    def custom_prompt(prob):
        msgs = original_create_prompt(prob)
        if msgs and msgs[0].get('role') == 'user':
            msgs[0]['content'] += f"\n\nHint: {glm_desc[:200]}"
        return msgs
    
    barc_gen._create_prompt = custom_prompt
    
    # Generate with hint
    candidates_with_glm = barc_gen.generate(problem, num_candidates=1)
    if candidates_with_glm:
        candidate_glm = candidates_with_glm[0]
        result_glm = executor.execute(candidate_glm.code, problem)
        print(f"✓ With GLM - Accuracy: {result_glm.accuracy:.2%}")
        
    # Restore
    barc_gen._create_prompt = original_create_prompt
    
except Exception as e:
    print(f"✗ GLM test failed: {e}")

print("\nTest completed!")