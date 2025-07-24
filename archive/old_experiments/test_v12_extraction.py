#!/usr/bin/env python3
"""Test V12 code extraction and execution"""

import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

import arc
from src.generators.barc_generator_simple import BARCGeneratorSimple
from src.evaluators.simple_evaluator import SimpleEvaluator

def visualize_grids(input_grid, output_grid, predicted_grid=None, title="ARC Problem"):
    """Visualize input, expected output, and predicted output grids"""
    # Color mapping
    colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    
    num_plots = 3 if predicted_grid is not None else 2
    fig, axes = plt.subplots(1, num_plots, figsize=(5*num_plots, 5))
    
    if num_plots == 2:
        axes = [axes[0], axes[1], None]
    
    # Plot input
    ax = axes[0]
    ax.set_title('Input', fontsize=14)
    ax.set_xticks(range(input_grid.shape[1]+1))
    ax.set_yticks(range(input_grid.shape[0]+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='black', linewidth=0.5)
    
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            color = colors[input_grid[i, j]]
            rect = patches.Rectangle((j, i), 1, 1, linewidth=0, facecolor=color)
            ax.add_patch(rect)
    
    ax.set_xlim(0, input_grid.shape[1])
    ax.set_ylim(input_grid.shape[0], 0)
    ax.set_aspect('equal')
    
    # Plot expected output
    ax = axes[1]
    ax.set_title('Expected Output', fontsize=14)
    ax.set_xticks(range(output_grid.shape[1]+1))
    ax.set_yticks(range(output_grid.shape[0]+1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True, color='black', linewidth=0.5)
    
    for i in range(output_grid.shape[0]):
        for j in range(output_grid.shape[1]):
            color = colors[output_grid[i, j]]
            rect = patches.Rectangle((j, i), 1, 1, linewidth=0, facecolor=color)
            ax.add_patch(rect)
    
    ax.set_xlim(0, output_grid.shape[1])
    ax.set_ylim(output_grid.shape[0], 0)
    ax.set_aspect('equal')
    
    # Plot predicted output if available
    if predicted_grid is not None:
        ax = axes[2]
        ax.set_title('Predicted Output', fontsize=14)
        ax.set_xticks(range(predicted_grid.shape[1]+1))
        ax.set_yticks(range(predicted_grid.shape[0]+1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color='black', linewidth=0.5)
        
        for i in range(predicted_grid.shape[0]):
            for j in range(predicted_grid.shape[1]):
                color = colors[predicted_grid[i, j]]
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0, facecolor=color)
                ax.add_patch(rect)
        
        ax.set_xlim(0, predicted_grid.shape[1])
        ax.set_ylim(predicted_grid.shape[0], 0)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    return fig

def test_extraction_and_execution():
    """Test code extraction and execution pipeline"""
    
    print("Loading model and tokenizer...")
    model_name = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Initialize generator and evaluator
    generator = BARCGeneratorSimple(model, tokenizer)
    evaluator = SimpleEvaluator()
    
    # Test on first validation problem
    problem_id = "2a5f8217"
    print(f"\nTesting on problem: {problem_id}")
    
    # Get problem
    problem = None
    for p in arc.validation_problems:
        if p.uid == problem_id:
            problem = p
            break
    
    # Generate solution
    print("Generating solution...")
    result = generator.generate(problem_id)
    
    print("\n=== EXTRACTED CODE ===")
    print(result["code"])
    
    print("\n=== CONCEPTS ===")
    print(result["concepts"])
    
    print("\n=== DESCRIPTION ===")
    print(result["description"])
    
    # Evaluate solution
    print("\n=== EXECUTION TEST ===")
    eval_result = evaluator.evaluate_solution(problem_id, result["code"])
    
    print(f"Execution Success: {eval_result['execution_success']}")
    print(f"Accuracy: {eval_result['accuracy']:.1%}")
    if eval_result.get('error'):
        print(f"Error: {eval_result['error']}")
    
    # Visualize results
    if eval_result['execution_success'] and eval_result['generated_outputs']:
        print("\n=== VISUALIZATION ===")
        # Show first test example
        test_input = problem.test_pairs[0].x
        test_output = problem.test_pairs[0].y
        predicted_output = eval_result['generated_outputs'][0]
        
        fig = visualize_grids(test_input, test_output, predicted_output, 
                             title=f"Problem {problem_id}")
        plt.savefig('test_v12_visualization.png', dpi=150, bbox_inches='tight')
        print("Saved visualization to test_v12_visualization.png")
        
        # Show grid dimensions
        print(f"\nGrid dimensions:")
        print(f"Input: {test_input.shape}")
        print(f"Expected: {test_output.shape}")
        print(f"Predicted: {predicted_output.shape}")
        
        # Show pixel accuracy
        if predicted_output.shape == test_output.shape:
            matches = np.sum(predicted_output == test_output)
            total = predicted_output.size
            print(f"Pixel accuracy: {matches}/{total} ({matches/total*100:.1f}%)")

if __name__ == "__main__":
    test_extraction_and_execution()