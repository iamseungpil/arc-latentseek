#!/usr/bin/env python3
"""
Test the enhanced ARC-LatentSeek pipeline improvements on a single problem
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import ARCLatentSeekPipeline, PipelineConfig


def test_single_problem(problem_id="2a5f8217"):
    """Test pipeline improvements on a single problem"""
    
    print("🧪 Testing ARC-LatentSeek Pipeline Improvements")
    print("=" * 50)
    print(f"Testing problem: {problem_id}")
    print()
    
    # Create test configuration
    config = PipelineConfig(
        # Model settings
        barc_model="barc0/Llama-3.1-ARC-Potpourri-Induction-8B",
        glm_model="THUDM/GLM-4.1V-9B-Thinking",
        
        # Generation settings
        num_candidates=4,  # Fewer for quick test
        temperature=0.8,
        max_new_tokens=2048,
        
        # Execution settings  
        execution_timeout=2,
        
        # Optimization settings
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
        output_dir="results/test_improvements",
        save_visualizations=True
    )
    
    print("Configuration:")
    print(f"  BARC Model: {config.barc_model}")
    print(f"  GLM Model: {config.glm_model}")
    print(f"  Candidates: {config.num_candidates}")
    print(f"  LatentSeek steps: {config.optimization_steps}")
    print(f"  Binary rewards: Enabled")
    print(f"  Enhanced GLM prompts: Enabled")
    print(f"  Before/after tracking: Enabled")
    print()
    
    # Create pipeline
    print("🔧 Initializing pipeline...")
    pipeline = ARCLatentSeekPipeline(config)
    print("✅ Pipeline initialized")
    print()
    
    # Run test
    print(f"🚀 Running test on problem {problem_id}...")
    start_time = datetime.now()
    
    results = pipeline.solve_problems(
        split="validation",
        problem_ids=[problem_id]
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    if not results:
        print("❌ No results returned")
        return
    
    result = results[0]
    
    # Print results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS")
    print("=" * 50)
    
    print(f"Problem ID: {result.problem_id}")
    print(f"Success: {'✅ YES' if result.success else '❌ NO'}")
    print(f"Final accuracy: {result.execution_accuracy:.2%}")
    print(f"Best reward: {result.best_reward:.3f}")
    print(f"Time taken: {result.time_taken:.1f}s")
    print()
    
    # Before/after analysis
    print("📈 BEFORE/AFTER ANALYSIS:")
    print(f"Initial best accuracy: {result.initial_best_accuracy:.2%}")
    print(f"Final best accuracy: {result.final_best_accuracy:.2%}")
    print(f"Initial success: {'✅ YES' if result.initial_success else '❌ NO'}")
    print(f"Improved by LatentSeek: {'✅ YES' if result.improved_by_latentseek else '❌ NO'}")
    
    if result.final_best_accuracy > result.initial_best_accuracy:
        improvement = (result.final_best_accuracy - result.initial_best_accuracy) * 100
        print(f"Accuracy improvement: +{improvement:.1f}%")
    print()
    
    # Evaluation details
    print("🔍 EVALUATION DETAILS:")
    verifications = result.evaluation_details.get('verifications', {})
    for name, passed in verifications.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
    print()
    
    # GLM feedback
    feedback = result.evaluation_details.get('feedback', {})
    if feedback:
        print("💬 GLM FEEDBACK:")
        for verifier, text in feedback.items():
            print(f"  {verifier}: {text[:100]}...")
        print()
    
    # Files saved
    print("📁 OUTPUT FILES:")
    print(f"  Visualization: {result.visualization_path}")
    print(f"  Logs directory: {config.output_dir}/logs/")
    print(f"  Results: {config.output_dir}/results_*.json")
    print()
    
    print("✅ Test completed successfully!")
    
    return result


if __name__ == "__main__":
    # Test with our known problem
    test_single_problem("2a5f8217")