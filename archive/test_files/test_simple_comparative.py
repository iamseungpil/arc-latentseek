#!/usr/bin/env python3
"""Simple test for comparative experiment"""
import os
import sys

# Set GPU
gpu_id = sys.argv[1] if len(sys.argv) > 1 else "5"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

from comparative_experiment_v2 import ComparativeExperimentV2, ExperimentCondition

print(f"Testing on GPU {gpu_id}")

# Create experiment
exp = ComparativeExperimentV2(output_dir=f"test_results_gpu{gpu_id}")

# Test just one condition and one problem
if gpu_id == "5":
    condition = ExperimentCondition("test_basic", False, False, 5)
else:
    condition = ExperimentCondition("test_multitensor", False, True, 6)

# Run single experiment
try:
    result = exp.run_single_experiment(
        condition,
        exp.problems[0],  # Just first problem
        0  # Just first candidate
    )
    
    if result:
        print(f"\nSuccess! Final accuracy: {result.final_accuracy}%")
        print(f"Optimization steps: {len(result.optimization_steps)}")
        
        # Check if LatentSeek was used
        if len(result.optimization_steps) > 1:
            step0 = result.optimization_steps[0]
            step_final = result.optimization_steps[-1]
            print(f"\nLatentSeek optimization:")
            print(f"Initial accuracy: {step0.accuracy}%")
            print(f"Final accuracy: {step_final.accuracy}%")
            print(f"Description changed: {step0.description != step_final.description}")
    else:
        print("Failed to run experiment")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()