#!/usr/bin/env python3
"""Full visualization of V12 test with all examples"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

import arc
from src.evaluators.simple_evaluator import SimpleEvaluator
from src.executors.grid_renderer import GridRenderer

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

def test_all_examples():
    """Test and visualize all train and test examples"""
    
    # Get problem
    problem_id = "2a5f8217"
    problem = None
    for p in arc.validation_problems:
        if p.uid == problem_id:
            problem = p
            break
    
    if problem is None:
        print(f"Problem {problem_id} not found")
        return
    
    print(f"Problem {problem_id}:")
    print(f"- Train examples: {len(problem.train_pairs)}")
    print(f"- Test examples: {len(problem.test_pairs)}")
    
    # Initialize renderer
    renderer = GridRenderer()
    
    # Test on all train examples
    print("\n=== TRAIN EXAMPLES ===")
    train_results = []
    
    # Create exec environment once
    from typing import List, Tuple, Dict, Set, Optional, Union, Any
    exec_globals = {
        'np': np,
        'numpy': np,
        'List': List,
        'Tuple': Tuple,
        'Dict': Dict,
        'Set': set,
        'Optional': Optional,
        'Union': Union,
        'Any': Any,
    }
    
    # Add all common.py functions
    import common
    for name in dir(common):
        if not name.startswith('_'):
            exec_globals[name] = getattr(common, name)
    
    # Execute code to get main function
    exec(test_code, exec_globals)
    main_func = exec_globals['main']
    
    for i, pair in enumerate(problem.train_pairs):
        print(f"\nTrain example {i+1}:")
        try:
            # Execute on input
            output = main_func(pair.x)
            if not isinstance(output, np.ndarray):
                output = np.array(output)
            
            # Check accuracy
            correct = np.array_equal(output, pair.y)
            if output.shape == pair.y.shape:
                pixel_acc = np.sum(output == pair.y) / output.size * 100
            else:
                pixel_acc = 0.0
            
            print(f"  Shape: {pair.x.shape} -> {output.shape} (expected: {pair.y.shape})")
            print(f"  Correct: {correct}, Pixel accuracy: {pixel_acc:.1f}%")
            
            train_results.append({
                'input': pair.x,
                'expected': pair.y,
                'predicted': output,
                'correct': correct,
                'pixel_acc': pixel_acc
            })
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            train_results.append({
                'input': pair.x,
                'expected': pair.y,
                'predicted': None,
                'correct': False,
                'pixel_acc': 0.0,
                'error': str(e)
            })
    
    # Test on all test examples
    print("\n=== TEST EXAMPLES ===")
    test_results = []
    
    for i, pair in enumerate(problem.test_pairs):
        print(f"\nTest example {i+1}:")
        try:
            # Execute on input
            output = main_func(pair.x)
            if not isinstance(output, np.ndarray):
                output = np.array(output)
            
            # Check accuracy
            correct = np.array_equal(output, pair.y)
            if output.shape == pair.y.shape:
                pixel_acc = np.sum(output == pair.y) / output.size * 100
            else:
                pixel_acc = 0.0
            
            print(f"  Shape: {pair.x.shape} -> {output.shape} (expected: {pair.y.shape})")
            print(f"  Correct: {correct}, Pixel accuracy: {pixel_acc:.1f}%")
            
            test_results.append({
                'input': pair.x,
                'expected': pair.y,
                'predicted': output,
                'correct': correct,
                'pixel_acc': pixel_acc
            })
            
        except Exception as e:
            print(f"  Error: {str(e)}")
            test_results.append({
                'input': pair.x,
                'expected': pair.y,
                'predicted': None,
                'correct': False,
                'pixel_acc': 0.0,
                'error': str(e)
            })
    
    # Calculate overall accuracy
    all_results = train_results + test_results
    total_correct = sum(1 for r in all_results if r['correct'])
    total_examples = len(all_results)
    overall_accuracy = total_correct / total_examples * 100
    
    print(f"\n=== SUMMARY ===")
    print(f"Total examples: {total_examples}")
    print(f"Correct: {total_correct}")
    print(f"Overall accuracy: {overall_accuracy:.1f}%")
    
    # Visualize using renderer
    print("\n=== VISUALIZATION ===")
    
    # Create figure for all examples
    n_train = len(train_results)
    n_test = len(test_results)
    total = n_train + n_test
    
    fig, axes = plt.subplots(total, 3, figsize=(12, 4*total))
    if total == 1:
        axes = axes.reshape(1, -1)
    
    # Color mapping
    colors = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', 
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
    
    def plot_grid(ax, grid, title):
        if grid is None:
            ax.text(0.5, 0.5, 'Error', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            return
            
        ax.set_title(title, fontsize=12)
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
    
    # Plot all examples
    for idx, (result, is_train) in enumerate([(r, True) for r in train_results] + [(r, False) for r in test_results]):
        row = idx
        example_type = "Train" if is_train else "Test"
        example_num = idx + 1 if is_train else idx - n_train + 1
        
        plot_grid(axes[row, 0], result['input'], f'{example_type} {example_num} - Input')
        plot_grid(axes[row, 1], result['expected'], 'Expected')
        plot_grid(axes[row, 2], result['predicted'], 
                 f"Predicted ({'✓' if result['correct'] else '✗'} {result['pixel_acc']:.0f}%)")
    
    plt.suptitle(f'Problem {problem_id} - All Examples - Overall Accuracy: {overall_accuracy:.1f}%', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('v12_all_examples_viz.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization to v12_all_examples_viz.png")
    
    print("\nVisualization complete!")

if __name__ == "__main__":
    print("✅ Successfully imported ARC datasets from arc-py")
    test_all_examples()