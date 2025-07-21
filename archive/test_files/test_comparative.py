import os
import sys

# Set GPU before importing anything else
gpu_id = sys.argv[1] if len(sys.argv) > 1 else "5"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

print(f"Testing comparative experiment on GPU {gpu_id}")
print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")

try:
    from comparative_experiment_v2 import ComparativeExperimentV2, ExperimentCondition
    print("✓ Imports successful")
    
    # Create experiment instance
    print("\nInitializing experiment...")
    experiment = ComparativeExperimentV2(output_dir=f"comparative_results_gpu{gpu_id}_test")
    print("✓ Experiment initialized")
    
    # Test a single problem
    print("\nTesting single experiment...")
    test_condition = ExperimentCondition("test_basic", False, False, int(gpu_id))
    
    # Run one experiment
    result = experiment.run_single_experiment(
        test_condition, 
        experiment.problems[0],  # First problem only
        0  # First candidate only
    )
    
    if result:
        print("✓ Test experiment completed successfully!")
        print(f"  Problem: {result.problem_id}")
        print(f"  Final accuracy: {result.final_accuracy}%")
    else:
        print("✗ Test experiment failed")
        
except Exception as e:
    print(f"\n✗ Error occurred: {e}")
    import traceback
    traceback.print_exc()