"""
Run comparative experiments on GPU 4 and 5
"""
import os
import sys

# Set GPU before importing anything else
gpu_id = sys.argv[1] if len(sys.argv) > 1 else "4"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

from comparative_experiment_v2 import ComparativeExperimentV2, ExperimentCondition

def main():
    print(f"Running comparative experiment on GPU {gpu_id}")
    
    # Create experiment instance
    experiment = ComparativeExperimentV2(output_dir=f"comparative_results_gpu{gpu_id}")
    
    # Run experiments for this GPU
    results = experiment.run_experiments_for_gpu(int(gpu_id))
    
    print(f"\nGPU {gpu_id} experiments completed!")
    print(f"Results saved to: comparative_results_gpu{gpu_id}/")

if __name__ == "__main__":
    main()