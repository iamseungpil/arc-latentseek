#!/usr/bin/env python3
"""Visualize V12 test results"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

import arc
from src.evaluators.simple_evaluator import SimpleEvaluator

# Test code
test_code = '''
from common import *
import numpy as np

# concepts:
# color mapping, object detection, color replacement

# description:
# In the input, you will see a grid containing several objects of different colors. 
# Each object is defined by a connected region of pixels of the same color. 
# To make the output, change the color of each object to match the color of the object directly below it. 
# If there is no object below, the color remains unchanged.

def main(input_grid):
    # Copy the input grid to the output grid
    output_grid = np.copy(input_grid)
    
    # Get the objects in the input grid
    objects = find_connected_components(input_grid, background=Color.BLACK)
    
    # For each object, change its color to the color of the object below it
    for obj in objects:
        # Get the bounding box of the object
        x, y, width, height = bounding_box(obj, background=Color.BLACK)
        
        # Check if there is an object directly below the bounding box
        if y + height < output_grid.shape[1]:  # Ensure we don't go out of bounds
            below_color = output_grid[x:x+width, y + height].max()  # Get the color of the pixels directly below
            
            # Change the color of the current object to the color of the object below
            output_grid[obj == output_grid[x, y]] = below_color
    
    return output_grid
'''

def visualize_results():
    """Visualize execution results"""
    # Get problem
    problem_id = "2a5f8217"
    problem = None
    for p in arc.validation_problems:
        if p.uid == problem_id:
            problem = p
            break
    
    # Execute code
    evaluator = SimpleEvaluator()
    result = evaluator.evaluate_solution(problem_id, test_code)
    
    if not result['execution_success']:
        print(f"Execution failed: {result['error']}")
        return
    
    # Get grids
    test_input = problem.test_pairs[0].x
    test_output = problem.test_pairs[0].y
    predicted = result['generated_outputs'][0]
    
    # Color mapping
    colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Helper function to plot grid
    def plot_grid(ax, grid, title):
        ax.set_title(title, fontsize=14)
        ax.set_xticks(range(grid.shape[1]+1))
        ax.set_yticks(range(grid.shape[0]+1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.grid(True, color='black', linewidth=0.5)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color = colors[grid[i, j]]
                rect = patches.Rectangle((j, i), 1, 1, linewidth=0, facecolor=color)
                ax.add_patch(rect)
        
        ax.set_xlim(0, grid.shape[1])
        ax.set_ylim(grid.shape[0], 0)
        ax.set_aspect('equal')
    
    # Plot all three grids
    plot_grid(axes[0], test_input, 'Input')
    plot_grid(axes[1], test_output, 'Expected Output')
    plot_grid(axes[2], predicted, 'Predicted Output')
    
    plt.suptitle(f'Problem {problem_id} - Accuracy: {result["accuracy"]:.1%}', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('v12_test_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to v12_test_visualization.png")
    
    # Print details
    print(f"\nExecution Success: {result['execution_success']}")
    print(f"Accuracy: {result['accuracy']:.1%}")
    print(f"Input shape: {test_input.shape}")
    print(f"Expected shape: {test_output.shape}")
    print(f"Predicted shape: {predicted.shape}")
    
    if predicted.shape == test_output.shape:
        matches = np.sum(predicted == test_output)
        total = predicted.size
        print(f"Pixel accuracy: {matches}/{total} ({matches/total*100:.1f}%)")

if __name__ == "__main__":
    print("✅ Successfully imported ARC datasets from arc-py: 400 train, 400 validation")
    visualize_results()