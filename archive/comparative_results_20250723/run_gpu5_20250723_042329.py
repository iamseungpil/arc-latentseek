import os
import sys

# IMPORTANT: Set CUDA device BEFORE any torch/transformers imports
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Now import after setting environment
sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU5 experiment - output dir: comparative_results_gpu5_20250723_042329")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    experiment = ComparativeExperimentV2(output_dir="comparative_results_gpu5_20250723_042329")
    experiment.run_experiments_for_gpu(5)
