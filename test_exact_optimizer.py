"""Test the exact LatentSeek optimizer with model.model fix"""

import sys
import torch
from src.data import load_arc_problems
from src.generators.barc_generator_fixed import BARCGeneratorFixed
from src.executors import CodeExecutor
from src.evaluators.glm_evaluator import GLMEvaluator
from src.optimizers.latent_optimizer_exact import LatentSeekOptimizerExact

def test_model_interface():
    """Test that model.model works correctly"""
    print("Testing model interface...")
    
    # Load a small test
    generator = BARCGeneratorFixed(device="cuda:0")
    
    # Test model.model access
    assert hasattr(generator.model, 'model'), "Model should have .model attribute"
    assert hasattr(generator.model, 'lm_head'), "Model should have .lm_head attribute"
    
    # Test forward pass
    test_input = torch.tensor([[1, 2, 3]], device=generator.model.device)
    
    # Test model.model forward
    with torch.no_grad():
        outputs = generator.model.model(test_input, output_hidden_states=True)
        assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"
        print(f"✓ model.model outputs: {type(outputs)}, len={len(outputs)}")
        print(f"  outputs[0] shape: {outputs[0].shape}")
        print(f"  outputs[2] (hidden_states) len: {len(outputs[2])}")
    
    print("\nModel interface test passed!")

def test_optimization():
    """Test the exact optimizer on a simple problem"""
    print("\nTesting exact LatentSeek optimizer...")
    
    # Load components
    problems = load_arc_problems(split="train", max_problems=1)
    problem = problems[0]
    print(f"Testing on problem: {problem.uid}")
    
    # Initialize components
    generator = BARCGeneratorFixed(device="cuda:0")
    executor = CodeExecutor()
    evaluator = GLMEvaluator(generator.model, generator.tokenizer)
    
    # Generate initial solution
    print("Generating initial solution...")
    initial_outputs = generator.generate(problem, temperature=0.8, num_candidates=1)
    initial_output = initial_outputs[0]
    
    if not initial_output.code:
        print("No code generated, skipping test")
        return
    
    # Evaluate initial solution
    result = executor.execute(initial_output.code, problem)
    eval_result = evaluator.evaluate(problem, initial_output, result)
    initial_reward = eval_result.total_reward
    
    print(f"Initial accuracy: {result.accuracy:.1%}")
    print(f"Initial reward: {initial_reward:.3f}")
    
    # Create optimizer
    optimizer = LatentSeekOptimizerExact(
        barc_generator=generator,
        code_executor=executor,
        evaluator=evaluator,
        lr=0.03,
        max_steps=3,  # Just a few steps for testing
        k=0.1,
        reward_threshold=-0.2,
        grad_clip=1.0
    )
    
    # Run optimization
    print("\nRunning optimization...")
    opt_result = optimizer.optimize(problem, initial_output, initial_reward)
    
    print(f"\nOptimization complete:")
    print(f"  Steps: {opt_result.optimization_steps}")
    print(f"  Converged: {opt_result.converged}")
    print(f"  Original length: {opt_result.original_length}")
    print(f"  Optimized length: {opt_result.optimized_length}")
    print(f"  Update length: {opt_result.update_length}")
    print(f"  Reward history: {opt_result.reward_history}")
    
    if opt_result.final_output.code:
        final_result = executor.execute(opt_result.final_output.code, problem)
        print(f"  Final accuracy: {final_result.accuracy:.1%}")

if __name__ == "__main__":
    # Test model interface first
    test_model_interface()
    
    # Then test optimization
    test_optimization()