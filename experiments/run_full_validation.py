#!/usr/bin/env python3
"""
Run ARC-LatentSeek on all validation problems
"""

import sys
import os
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main import ARCLatentSeekPipeline, PipelineConfig


def main():
    parser = argparse.ArgumentParser(description="Run ARC-LatentSeek on validation set")
    parser.add_argument("--num_problems", type=int, default=400, 
                      help="Number of validation problems to solve")
    parser.add_argument("--num_candidates", type=int, default=8, 
                      help="Number of candidate solutions per problem")
    parser.add_argument("--output_dir", type=str, default="results/validation_full",
                      help="Output directory")
    parser.add_argument("--gpu", type=int, default=None,
                      help="GPU device to use")
    
    args = parser.parse_args()
    
    # Set GPU if specified
    if args.gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    
    # Configure pipeline
    config = PipelineConfig(
        num_candidates=args.num_candidates,
        output_dir=args.output_dir,
        save_visualizations=True,
        optimization_steps=10,
        optimization_threshold=-0.2,
        use_description_based_optimization=True,
        enable_code_alignment=True
    )
    
    # Create pipeline
    print(f"Initializing pipeline...")
    pipeline = ARCLatentSeekPipeline(config)
    
    # Run on validation problems
    print(f"\nRunning on {args.num_problems} validation problems...")
    results = []
    
    # Import needed modules
    import json
    from datetime import datetime
    from dataclasses import asdict
    
    # Get problems
    problems = pipeline.data_loader.get_problems("validation", args.num_problems)
    
    # Solve each problem and save results incrementally
    for i, problem in enumerate(problems):
        print(f"\n{'='*60}")
        print(f"Problem {i+1}/{len(problems)}: {problem.uid}")
        print(f"{'='*60}")
        
        # Solve the problem
        result = pipeline.solve_problem(problem)
        results.append(result)
        
        # Save intermediate results after each problem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        intermediate_file = os.path.join(args.output_dir, f"results_intermediate_{i+1}_problems.jsonl")
        
        with open(intermediate_file, "w") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")
        
        print(f"💾 Saved intermediate results ({i+1}/{len(problems)}) to {intermediate_file}")
    
    # Save final results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    results_file = os.path.join(args.output_dir, f"results_{timestamp}.jsonl")
    with open(results_file, "w") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")
    print(f"\nResults saved to {results_file}")
    
    # Save summary
    successful = sum(1 for r in results if r.success)
    improved = sum(1 for r in results if r.improved_by_latentseek)
    
    summary = {
        "timestamp": timestamp,
        "total_problems": len(results),
        "successful": successful,
        "success_rate": successful/len(results) if results else 0,
        "improved_by_latentseek": improved,
        "improvement_rate": improved/len(results) if results else 0,
        "average_time_per_problem": sum(r.time_taken for r in results)/len(results) if results else 0,
        "problem_ids": [r.problem_id for r in results],
        "success_ids": [r.problem_id for r in results if r.success]
    }
    
    summary_file = os.path.join(args.output_dir, f"summary_{timestamp}.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {summary_file}")
    
    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Total problems: {len(results)}")
    print(f"Successful: {successful} ({successful/len(results)*100:.1f}%)")
    print(f"Improved by LatentSeek: {improved} ({improved/len(results)*100:.1f}%)")
    print(f"Average time per problem: {sum(r.time_taken for r in results)/len(results):.1f}s")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())