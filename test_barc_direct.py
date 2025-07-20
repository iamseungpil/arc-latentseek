#!/usr/bin/env python3
"""Test BARC code directly without model loading"""

import numpy as np
from common import *

# The BARC code that was generated (from logs)
def main(input_grid):
    # Find the shape in the input grid
    objects = find_connected_components(input_grid, monochromatic=False, background=Color.BLACK)
    assert len(objects) == 1  # There should only be one shape in the input grid

    original_shape = objects[0]
    
    # Transform the shape
    scaled_shape = scale_pattern(original_shape, scale_factor=2)
    
    # Create the output grid
    output_grid = np.full((input_grid.shape[0], input_grid.shape[1]), Color.BLACK)

    # Calculate the position to place the transformed shape in the center of the output grid
    shape_height, shape_width = scaled_shape.shape
    center_x = (output_grid.shape[0] - shape_height) // 2
    center_y = (output_grid.shape[1] - shape_width) // 2

    # Blit the transformed shape back into the output grid
    blit_sprite(output_grid, scaled_shape, x=center_x, y=center_y, background=Color.BLACK)

    return output_grid

# Test input
input_grid = np.array([
    [0, 5, 0],
    [5, 5, 5],
    [0, 5, 0]
])

print("Testing BARC generated code...")
print(f"Input shape: {input_grid.shape}")

try:
    output = main(input_grid)
    print(f"Output shape: {output.shape}")
    print(f"Output:\n{output}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# What the correct solution should be
print("\nCorrect solution:")
correct_output = np.zeros((6, 6), dtype=int)
for i in range(2):
    for j in range(2):
        correct_output[i*3:(i+1)*3, j*3:(j+1)*3] = input_grid

# Color transformation
correct_output[correct_output == 0] = 1  # BLACK -> BLUE
correct_output[correct_output == 5] = 2  # GRAY -> RED

print(f"Correct output shape: {correct_output.shape}")
print(f"Correct output:\n{correct_output}")