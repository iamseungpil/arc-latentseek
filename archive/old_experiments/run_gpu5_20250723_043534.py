import os
import sys

# Set CUDA device BEFORE imports
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU5 experiment - output dir: comparative_results_gpu5_20250723_043534")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    experiment = ComparativeExperimentV2(output_dir="comparative_results_gpu5_20250723_043534")
    experiment.run_experiments_for_gpu(5)
