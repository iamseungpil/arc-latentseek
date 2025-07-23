import sys
sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU6 experiment - output dir: comparative_results_gpu6_20250723_041753")
    experiment = ComparativeExperimentV2(output_dir="comparative_results_gpu6_20250723_041753")
    experiment.run_experiments_for_gpu(6)
