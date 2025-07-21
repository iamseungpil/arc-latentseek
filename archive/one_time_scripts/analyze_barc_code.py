#!/usr/bin/env python3
"""Analyze BARC code execution step by step"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from common import *

# Input grid from the problem
input_grid = np.array([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0]
])

print("Input grid:")
print(input_grid)

# Step 1: Find connected components
objects = find_connected_components(input_grid, monochromatic=False, background=Color.BLACK)
print(f"\nNumber of objects found: {len(objects)}")

if len(objects) > 0:
    original_shape = objects[0]
    print(f"\nOriginal shape:")
    print(original_shape)
    print(f"Shape dimensions: {original_shape.shape}")
    
    # Step 2: Scale the shape
    print("\nTrying to scale...")
    try:
        # First try scale_pattern (deprecated)
        scaled_shape = scale_pattern(original_shape, scale_factor=2)
        print(f"Scaled shape (scale_pattern):")
        print(scaled_shape)
        print(f"Scaled dimensions: {scaled_shape.shape}")
    except Exception as e:
        print(f"scale_pattern error: {e}")
        
    try:
        # Try scale_sprite instead
        scaled_shape = scale_sprite(original_shape, scale_factor=2)
        print(f"\nScaled shape (scale_sprite):")
        print(scaled_shape)
        print(f"Scaled dimensions: {scaled_shape.shape}")
    except Exception as e:
        print(f"scale_sprite error: {e}")
    
    # Step 3: Create output grid (BARC code creates 3x3, but expects 6x6)
    print(f"\nBig issue: BARC code creates output grid with input dimensions {input_grid.shape}")
    print(f"But the expected output should be (6, 6)!")
    
    # The correct code should be:
    print("\nCorrected version:")
    output_grid = np.full((6, 6), Color.BLACK)
    
    # The problem expects color transformation too
    print("\nAnother issue: BARC doesn't transform colors (5->2 for gray, 0->1 for black)")
    
    # What the problem actually wants:
    print("\nWhat the problem wants:")
    print("1. Repeat the 3x3 pattern to fill 6x6")
    print("2. Transform colors: BLACK (0) -> BLUE (1), GRAY (5) -> RED (2)")
    
    # Correct solution:
    correct_output = np.zeros((6, 6), dtype=int)
    for i in range(2):
        for j in range(2):
            correct_output[i*3:(i+1)*3, j*3:(j+1)*3] = input_grid
    
    # Color transformation
    correct_output[correct_output == Color.BLACK] = Color.BLUE
    correct_output[correct_output == Color.GRAY] = Color.RED
    
    print("\nCorrect output:")
    print(correct_output)