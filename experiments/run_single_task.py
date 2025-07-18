#!/usr/bin/env python3
"""
Run ARC-LatentSeek on a single task
"""

import sys
import os
import argparse
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import ARCLatentSeekPipeline, PipelineConfig
from src.data import ARCDataLoader


def main():
    parser = argparse.ArgumentParser(description="Run ARC-LatentSeek on a single problem")
    parser.add_argument("problem_id", type=str, help="Problem ID to solve")
    parser.add_argument("--num_candidates", type=int, default=8, 
                      help="Number of candidate solutions to generate")
    parser.add_argument("--output_dir", type=str, default="results/single_task",
                      help="Output directory")
    parser.add_argument("--no_viz", action="store_true",
                      help="Disable visualization saving")
    
    args = parser.parse_args()
    
    # Configure pipeline
    config = PipelineConfig(
        num_candidates=args.num_candidates,
        output_dir=args.output_dir,
        save_visualizations=not args.no_viz
    )
    
    # Create pipeline
    print(f"Initializing pipeline...")
    pipeline = ARCLatentSeekPipeline(config)
    
    # Load specific problem
    data_loader = ARCDataLoader()
    problem = data_loader.get_problem_by_id(args.problem_id)
    
    if problem is None:
        print(f"Error: Problem {args.problem_id} not found")
        return 1
    
    print(f"\nSolving problem: {problem.uid}")
    print(f"Training examples: {len(problem.train_pairs)}")
    print(f"Test examples: {len(problem.test_pairs)}")
    
    # Solve problem
    result = pipeline.solve_problem(problem)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"Success: {result.success}")
    print(f"Execution Accuracy: {result.execution_accuracy:.2%}")
    print(f"Best Reward: {result.best_reward:.3f}")
    print(f"Time Taken: {result.time_taken:.2f}s")
    
    if result.best_description:
        print(f"\nBest Description:")
        print(f"  {result.best_description}")
    
    print(f"\nVerifications:")
    for name, passed in result.evaluation_details.get('verifications', {}).items():
        status = "✅" if passed else "❌"
        print(f"  {status} {name}")
    
    if result.visualization_path:
        print(f"\nVisualization saved to: {result.visualization_path}")
    
    # Save detailed result
    detailed_path = os.path.join(config.output_dir, f"{problem.uid}_detailed.json")
    with open(detailed_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"\nDetailed results saved to: {detailed_path}")
    
    return 0 if result.success else 1


if __name__ == "__main__":
    sys.exit(main())