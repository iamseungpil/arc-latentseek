from common import *

import numpy as np
from typing import *

# concepts:
# color change, object detection, adjacency

# description:
# In the input you will see several colored objects scattered around a grid. 
# To make the output, change the color of each object to the color of the object that is adjacent to it (up, down, left, or right).
# If an object has no adjacent object, it remains unchanged.

def main(input_grid):
    # Create a copy of the input grid to avoid modifying the original
    output_grid = np.copy(input_grid)

    # Find all the objects in the grid
    background = Color.BLACK
    objects = find_connected_components(input_grid, monochromatic=True, connectivity=4, background=background)

    # Create a mapping of positions to colors
    position_colors = {}
    for obj in objects:
        x, y, w, h = bounding_box(obj, background=background)
        color = object_colors(obj, background=background)[0]
        for i in range(x, x + w):
            for j in range(y, y + h):
                if obj[i - x, j - y]!= background:  # Only consider non-background pixels
                    position_colors[(i, j)] = color

    # Change the color of each object based on adjacent objects
    for (i, j), color in position_colors.items():
        # Check adjacent positions
        adjacent_colors = set()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
            if (i + dx, j + dy) in position_colors:
                adjacent_colors.add(position_colors[(i + dx, j + dy)])

        # If there are adjacent colors, change to one of them (arbitrarily, since they are all the same)
        if adjacent_colors:
            output_grid[i, j] = adjacent_colors.pop()

    return output_grid