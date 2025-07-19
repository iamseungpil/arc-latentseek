#!/usr/bin/env python3
"""
Analyze the actual pattern in problem 2a5f8217
"""

import sys
import os
import numpy as np

# Add paths
sys.path.append('/home/ubuntu/arc-latentseek')

# Import modules
from src.data.arc_loader import ARCDataLoader

def analyze_transformation(input_grid, output_grid):
    """Analyze what transformations happen between input and output"""
    print("\nTransformation Analysis:")
    
    # Get unique colors
    input_colors = set(input_grid.flatten())
    output_colors = set(output_grid.flatten())
    
    print(f"  Input colors: {sorted(input_colors)}")
    print(f"  Output colors: {sorted(output_colors)}")
    
    # Check what happens to each color
    color_mapping = {}
    for color in input_colors:
        # Find what this color becomes
        input_mask = input_grid == color
        if np.any(input_mask):
            output_values = output_grid[input_mask]
            unique_outputs = np.unique(output_values)
            if len(unique_outputs) == 1:
                color_mapping[color] = unique_outputs[0]
            else:
                color_mapping[color] = list(unique_outputs)
    
    print("\n  Color transformations:")
    for src, dst in sorted(color_mapping.items()):
        print(f"    {src} -> {dst}")
    
    return color_mapping

def main():
    print("="*80)
    print("Pattern Analysis for Problem 2a5f8217")
    print("="*80)
    
    # Load the problem
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    if not problem:
        print("❌ Problem not found!")
        return
    
    # Analyze each training example
    for i, pair in enumerate(problem.train_pairs):
        print(f"\n\nTraining Example {i+1}:")
        print(f"Input shape: {pair.x.shape}")
        print(f"Output shape: {pair.y.shape}")
        
        # Show the grids
        print("\nInput grid:")
        print(pair.x)
        print("\nOutput grid:")
        print(pair.y)
        
        # Analyze the transformation
        mapping = analyze_transformation(pair.x, pair.y)
        
        # Check for spatial patterns
        if pair.x.shape == pair.y.shape:
            diff_mask = pair.x != pair.y
            if np.any(diff_mask):
                print(f"\n  Pixels changed: {np.sum(diff_mask)} out of {pair.x.size}")
                changed_positions = np.argwhere(diff_mask)
                print(f"  First few changed positions: {changed_positions[:5].tolist()}")
        
    # Look for common pattern across all examples
    print("\n\n" + "="*80)
    print("PATTERN SUMMARY:")
    print("="*80)
    
    # Check if blue always transforms to the same color
    blue_transforms = []
    for pair in problem.train_pairs:
        blue_mask = pair.x == 1
        if np.any(blue_mask):
            output_values = pair.y[blue_mask]
            unique_outputs = np.unique(output_values)
            blue_transforms.append(unique_outputs)
    
    print(f"\nBlue (1) transformations across examples: {blue_transforms}")
    
    # Check what other colors are present
    all_input_colors = set()
    all_output_colors = set()
    for pair in problem.train_pairs:
        all_input_colors.update(pair.x.flatten())
        all_output_colors.update(pair.y.flatten())
    
    print(f"\nAll input colors across examples: {sorted(all_input_colors)}")
    print(f"All output colors across examples: {sorted(all_output_colors)}")
    
    # The key insight
    print("\n\nKEY INSIGHT:")
    print("The transformation seems to be: Replace blue (1) pixels with another color")
    print("The replacement color appears to be the non-black, non-blue color in the same example")

if __name__ == "__main__":
    main()