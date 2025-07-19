#!/usr/bin/env python3
"""
Full ARC-LatentSeek experiment on 400 validation problems
With enhanced logging, binary rewards, and before/after tracking
"""

import os
import sys
import json
import time
from tqdm import tqdm
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import ARCLatentSeekPipeline, PipelineConfig


def run_full_experiment():
    """Run comprehensive experiment on all 400 validation problems"""
    
    print("🚀 Starting ARC-LatentSeek Full Validation Experiment")
    print("=" * 60)
    
    # Create enhanced configuration
    config = PipelineConfig(
        # Model settings
        barc_model="barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
        glm_model="THUDM/GLM-4.1V-9B-Thinking",
        
        # Generation settings (optimized for stability)
        num_candidates=3,  # Further reduced to 3 for memory stability
        temperature=0.8,
        max_new_tokens=1024,
        
        # Execution settings  
        execution_timeout=2,
        
        # Optimization settings (reduced for efficiency)
        optimization_steps=10,
        optimization_threshold=-0.2,
        use_description_based_optimization=True,
        
        # Alignment settings
        enable_code_alignment=True,
        alignment_model="meta-llama/Llama-3.1-8B-Instruct",
        alignment_temperature=0.3,
        alignment_max_tokens=2048,
        min_alignment_score=20,
        
        # Output settings
        output_dir="results/full_experiment",
        save_visualizations=True
    )
    
    print("Configuration:")
    print(f"  BARC Model: {config.barc_model}")
    print(f"  GLM Model: {config.glm_model}")
    print(f"  Candidates per problem: {config.num_candidates}")
    print(f"  LatentSeek steps: {config.optimization_steps}")
    print(f"  Code alignment: {config.enable_code_alignment}")
    print(f"  Output directory: {config.output_dir}")
    print()
    
    # Create pipeline
    print("🔧 Initializing pipeline...")
    pipeline = ARCLatentSeekPipeline(config)
    print("✅ Pipeline initialized successfully")
    print()
    
    # Test on single problem first
    print("🧪 Running initial test on single problem...")
    test_results = pipeline.solve_problems(
        split="validation",
        num_problems=1
    )
    
    if not test_results:
        print("❌ Initial test failed - aborting experiment")
        return
        
    test_result = test_results[0]
    print(f"✅ Test completed: {test_result.problem_id}")
    print(f"   Success: {test_result.success}")
    print(f"   Accuracy: {test_result.execution_accuracy:.2%}")
    print(f"   Reward: {test_result.best_reward:.3f}")
    print(f"   Initial accuracy: {test_result.initial_best_accuracy:.2%}")
    print(f"   Improved by LatentSeek: {test_result.improved_by_latentseek}")
    print()
    
    # Auto-continue for tmux (no interactive input)
    print("🤔 Test successful. Auto-continuing with full 400 problems...")
    print("⚠️  To cancel, press Ctrl+C within 5 seconds...")
    import time
    time.sleep(5)
    
    print("\n🎯 Starting full experiment on 400 validation problems...")
    print("⚠️  This will take several hours to complete")
    print()
    
    # Track timing
    experiment_start = time.time()
    
    # Run full experiment with progress tracking
    try:
        results = pipeline.solve_problems(
            split="validation",
            num_problems=400
        )
        
        experiment_time = time.time() - experiment_start
        
        # Print detailed results
        print("\n" + "=" * 60)
        print("📊 EXPERIMENT RESULTS")
        print("=" * 60)
        
        total_problems = len(results)
        successful = sum(1 for r in results if r.success)
        initial_successes = sum(1 for r in results if r.initial_success)
        improved_by_latentseek = sum(1 for r in results if r.improved_by_latentseek)
        
        print(f"Total problems attempted: {total_problems}")
        print(f"Final successes: {successful} ({successful/total_problems*100:.1f}%)")
        print(f"Initial successes: {initial_successes} ({initial_successes/total_problems*100:.1f}%)")
        print(f"Improved by LatentSeek: {improved_by_latentseek} ({improved_by_latentseek/total_problems*100:.1f}%)")
        print()
        
        # Accuracy statistics
        avg_final_accuracy = sum(r.final_best_accuracy for r in results) / total_problems
        avg_initial_accuracy = sum(r.initial_best_accuracy for r in results) / total_problems
        accuracy_improvement = avg_final_accuracy - avg_initial_accuracy
        
        print(f"Average initial accuracy: {avg_initial_accuracy:.2%}")
        print(f"Average final accuracy: {avg_final_accuracy:.2%}")
        print(f"Average accuracy improvement: {accuracy_improvement:.2%}")
        print()
        
        # Timing statistics
        avg_time_per_problem = sum(r.time_taken for r in results) / total_problems
        print(f"Total experiment time: {experiment_time/3600:.1f} hours")
        print(f"Average time per problem: {avg_time_per_problem:.1f} seconds")
        print()
        
        # Reward statistics
        avg_reward = sum(r.best_reward for r in results) / total_problems
        perfect_rewards = sum(1 for r in results if r.best_reward >= 1.0)
        print(f"Average reward: {avg_reward:.3f}")
        print(f"Perfect rewards (≥1.0): {perfect_rewards} ({perfect_rewards/total_problems*100:.1f}%)")
        print()
        
        # LatentSeek effectiveness analysis
        problems_needing_optimization = sum(1 for r in results if not r.initial_success)
        if problems_needing_optimization > 0:
            latentseek_success_rate = improved_by_latentseek / problems_needing_optimization * 100
            print(f"LatentSeek optimization effectiveness:")
            print(f"  Problems needing optimization: {problems_needing_optimization}")
            print(f"  Improved by LatentSeek: {improved_by_latentseek}")
            print(f"  LatentSeek success rate: {latentseek_success_rate:.1f}%")
        
        print("\n✅ Full experiment completed successfully!")
        print(f"📁 Results saved in: {config.output_dir}/")
        
    except KeyboardInterrupt:
        print("\n⚠️  Experiment interrupted by user")
        print("Partial results may be available in output directory")
    except Exception as e:
        print(f"\n❌ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_full_experiment()