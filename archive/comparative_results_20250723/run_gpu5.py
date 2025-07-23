import sys
sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    experiment = ComparativeExperimentV2(output_dir="comparative_results_gpu5_20250723_040538")
    experiment.run_experiments_for_gpu(5)
