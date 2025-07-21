#!/usr/bin/env python3
"""Test the fixed pipeline with shape mismatch handling"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging
logging.basicConfig(level=logging.INFO)

from src.data import ARCDataLoader
from src.generators import BARCGenerator  
from src.executors import CodeExecutor
from src.evaluators import GLMEvaluator
from src.optimizers import LatentSeekOptimizer

# Load first validation problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems('validation', num_problems=1)
problem = problems[0]

print(f"\nProblem ID: {problem.uid}")
print(f"Train pairs: {len(problem.train_pairs)}")

# Initialize components
barc_generator = BARCGenerator("barc0/Llama-3.1-ARC-Potpourri-Induction-8B")
code_executor = CodeExecutor()

# Generate BARC code
print("\nGenerating BARC code...")
barc_outputs = barc_generator.generate(problem, num_candidates=1)

if not barc_outputs:
    print("No outputs generated!")
    exit(1)

barc_output = barc_outputs[0]
print(f"\nDescription: {barc_output.description}")

# Execute code
print("\nExecuting code...")
execution_result = code_executor.execute(barc_output.code, problem)

print(f"\nExecution success: {execution_result.success}")
print(f"Accuracy: {execution_result.accuracy}")
print(f"Error messages: {execution_result.error_messages}")
print(f"Comparison results: {execution_result.comparison_results}")

if execution_result.success:
    print("\n✅ Execution succeeded! Now can proceed to GLM evaluation and LatentSeek")
    
    if execution_result.accuracy < 1.0:
        print(f"\nAccuracy {execution_result.accuracy} < 1.0, would trigger LatentSeek optimization")
        
        # Initialize GLM evaluator (optional for testing)
        # glm_evaluator = GLMEvaluator("THUDM/GLM-4.1V-9B-Thinking") 
        # evaluation_result = glm_evaluator.evaluate(problem, barc_output, execution_result)
        # print(f"GLM reward: {evaluation_result.total_reward}")
else:
    print("\n❌ Execution failed, would skip GLM and LatentSeek")