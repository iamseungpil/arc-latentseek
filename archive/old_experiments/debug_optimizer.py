#!/usr/bin/env python3
"""
Debug script to test LatentSeekOptimizer API
"""
import os
import sys
sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader, ARCProblem
from src.generators import BARCGenerator, BARCOutput
from src.executors import CodeExecutor, ExecutionResult
from src.evaluators import GLMEvaluator, EvaluationResult, RewardModel
from src.optimizers import LatentSeekOptimizer, OptimizationResult

# Set environment for single GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

def main():
    print("Testing LatentSeekOptimizer API...")
    
    # Initialize components
    loader = ARCDataLoader()
    barc_generator = BARCGenerator()
    code_executor = CodeExecutor()
    glm_evaluator = GLMEvaluator()
    
    latent_optimizer = LatentSeekOptimizer(
        barc_generator=barc_generator,
        code_executor=code_executor,
        glm_evaluator=glm_evaluator
    )
    
    # Load a test problem
    problem = loader.get_problem_by_id('2072aba6')
    print(f"Loaded problem: {problem.uid}")
    
    # Generate a candidate
    print("Generating candidate...")
    candidates = barc_generator.generate(problem, num_candidates=1)
    if not candidates:
        print("Failed to generate candidates")
        return
    
    candidate = candidates[0]
    print(f"Generated candidate with {len(candidate.code)} chars")
    
    # Evaluate initial candidate
    print("Evaluating initial candidate...")
    exec_result = code_executor.execute(candidate.code, problem)
    glm_result = glm_evaluator.evaluate(problem, candidate, exec_result)
    initial_reward = glm_result.total_reward
    print(f"Initial reward: {initial_reward}")
    
    # Test optimization methods
    print("\n=== Testing optimize_description_based ===")
    try:
        opt_result = latent_optimizer.optimize_description_based(
            problem=problem,
            initial_output=candidate,
            initial_reward=initial_reward
        )
        print(f"optimize_description_based: SUCCESS")
        print(f"Final reward: {opt_result.reward_history[-1]}")
    except Exception as e:
        print(f"optimize_description_based: FAILED - {e}")
    
    print("\n=== Testing optimize ===")
    try:
        opt_result = latent_optimizer.optimize(
            problem=problem,
            initial_output=candidate,
            initial_reward=initial_reward
        )
        print(f"optimize: SUCCESS")
        print(f"Final reward: {opt_result.reward_history[-1]}")
    except Exception as e:
        print(f"optimize: FAILED - {e}")

if __name__ == "__main__":
    main()