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
    results = pipeline.solve_problems(
        split="validation",
        num_problems=args.num_problems
    )
    
    # Print final summary
    successful = sum(1 for r in results if r.success)
    improved = sum(1 for r in results if r.improved_by_latentseek)
    
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