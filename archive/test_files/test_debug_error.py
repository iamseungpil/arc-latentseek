#!/usr/bin/env python3
"""Debug string indices error"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

import sys
import traceback
from src.data import ARCDataLoader
from src.generators import BARCGenerator
from src.executors import CodeExecutor
from src.evaluators import GLMEvaluator
from src.executors.grid_renderer import GridRenderer

# Load one problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems('validation', num_problems=1)
problem = problems[0]

print(f"Problem ID: {problem.uid}")

# Generate BARC code
barc_generator = BARCGenerator("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")
print("Generating BARC code...")

try:
    candidates = barc_generator.generate(problem, num_candidates=1)
    if candidates:
        candidate = candidates[0]
        print(f"\nCandidate type: {type(candidate)}")
        print(f"Has code: {hasattr(candidate, 'code')}")
        print(f"Has description: {hasattr(candidate, 'description')}")
        
        # Try to access attributes
        print(f"\nAccessing description attribute:")
        try:
            desc = candidate.description
            print(f"Description: {desc[:50] if desc else 'None'}...")
        except Exception as e:
            print(f"Error accessing description: {e}")
            traceback.print_exc()
            
        # Execute code
        print("\nExecuting code...")
        executor = CodeExecutor()
        execution_result = executor.execute(candidate.code, problem)
        print(f"Execution success: {execution_result.success}")
        print(f"Accuracy: {execution_result.accuracy}")
        
        # Debug execution result
        if not execution_result.success:
            print(f"Error messages: {execution_result.error_messages}")
        
        # Force GLM evaluation even with errors to debug
        if True:  # execution_result.success:
            print("\nInitializing GLM evaluator...")
            glm_evaluator = GLMEvaluator("THUDM/GLM-4.1V-9B-Thinking")
            
            print("Running GLM evaluation...")
            try:
                evaluation_result = glm_evaluator.evaluate(
                    problem,
                    candidate,
                    execution_result,
                    "test_debug"
                )
                print(f"Evaluation reward: {evaluation_result.total_reward}")
            except Exception as e:
                print(f"Error in GLM evaluation: {e}")
                traceback.print_exc()
                
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()