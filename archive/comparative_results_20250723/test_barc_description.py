#!/usr/bin/env python
"""Test description extraction with real BARC outputs"""

import sys
sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader
from src.generators import BARCGenerator

# Real BARC output examples from logs
real_barc_examples = [
    # Example 1: From logs
    """from common import *

import numpy as np
from typing import *

# concepts:
# repetition, pattern expansion, color alternation

# description:
# In the input you will see a small pattern of gray pixels. 
# To make the output, repeat the pattern horizontally and vertically to fill the entire grid. 
# Alternate the color of the pattern between blue and red in every repetition.

def main(input_grid):
    # Find the connected component which is the gray pattern
    objects = find_connected_components(input_grid, monochromatic=True)
    pattern = objects[0]  # Assuming there's only one pattern

    # Get the bounding box of the pattern
    x, y, width, height = bounding_box(pattern)

    # Create the output grid, which is larger than the input grid
    output_width = input_grid.shape[0] * width
    output_height = input_grid.shape[1] * height
    output_grid = np.full((output_width, output_height), Color.BLACK)

    # Fill the output grid with the pattern, alternating colors
    for i in range(input_grid.shape[0]):
        for j in range(input_grid.shape[1]):
            if input_grid[i, j] == Color.GRAY:
                # Determine the color based on position
                color = Color.BLUE if (i + j) % 2 == 0 else Color.RED
                # Place the pattern in the output grid
                for dx in range(width):
                    for dy in range(height):
                        output_grid[i * width + dx, j * height + dy] = color if pattern[dx, dy] == Color.GRAY else Color.BLACK

    return output_grid""",
    
    # Example 2: From logs
    """from common import *

import numpy as np
from typing import *

# concepts:
# color mapping, grid transformation

# description:
# In the input you will see a 3x3 grid with different colors. 
# To make the output grid, you should transform the input grid into a 6x6 grid, 
# where each 1x1 cell in the input grid corresponds to a 2x2 block in the output grid. 
# If the 1x1 cell is not black, fill the 2x2 block with the color of the 1x1 cell. 
# If the 1x1 cell is black, fill the 2x2 block with black.

def transform(input_grid: np.ndarray) -> np.ndarray:
    # Create an output grid of size 6x6
    output_grid = np.full((6, 6), Color.BLACK, dtype=int)

    # Iterate over the 1x1 cells in the input grid
    for x in range(input_grid.shape[0]):
        for y in range(input_grid.shape[1]):
            color = input_grid[x, y]
            # If the color is not black, fill the corresponding 2x2 block in the output grid
            if color!= Color.BLACK:
                output_grid[2*x:2*x+2, 2*y:2*y+2] = color

    return output_grid"""
]

def test_with_real_barc():
    """Test description extraction with real BARC outputs"""
    from src.optimizers.latent_optimizer import LatentSeekOptimizer
    
    class DummyOptimizer:
        def _find_description_token_positions(self, text):
            return LatentSeekOptimizer._find_description_token_positions(self, text)
        
        def _extract_description_from_text(self, text):
            return LatentSeekOptimizer._extract_description_from_text(self, text)
    
    optimizer = DummyOptimizer()
    
    print("Testing with Real BARC Outputs\n")
    print("="*80 + "\n")
    
    for i, code in enumerate(real_barc_examples):
        print(f"=== BARC Output Example {i+1} ===\n")
        
        # Show the full code
        print("Full Code:")
        print("-"*40)
        print(code)
        print("-"*40)
        
        # Find description positions
        desc_start, desc_end = optimizer._find_description_token_positions(code)
        
        if desc_start is not None and desc_end is not None:
            print(f"\n✅ Found description at character positions: {desc_start}-{desc_end}")
            print(f"\nRaw extracted text:")
            print(repr(code[desc_start:desc_end]))
            
            # Extract processed description
            description = optimizer._extract_description_from_text(code)
            print(f"\nProcessed description:")
            print(f"'{description}'")
            
            # Show which pattern was used
            if "# description:" in code:
                print("\n📌 Used Pattern 1: Explicit description tag")
            elif desc_start > code.find("# concepts:"):
                print("\n📌 Used Pattern 2: After concepts block")
            else:
                print("\n📌 Used Pattern 3: After function definition")
        else:
            print("\n❌ No description found!")
        
        print("\n" + "="*80 + "\n")

def test_actual_generation():
    """Test with actual BARC generation"""
    print("\nTesting with Actual BARC Generation\n")
    print("="*80 + "\n")
    
    try:
        # Load components
        loader = ARCDataLoader()
        generator = BARCGenerator()
        
        # Get a problem
        problem = loader.get_problem_by_id('2072aba6')
        
        print(f"Problem ID: {problem.uid}")
        print(f"Generating BARC solution...\n")
        
        # Generate solution
        outputs = generator.generate(problem, temperature=0.7, num_candidates=1)
        
        if outputs:
            output = outputs[0]
            print("Generated Code:")
            print("-"*40)
            print(output.code)
            print("-"*40)
            
            if hasattr(output, 'description') and output.description:
                print(f"\nExtracted Description: '{output.description}'")
            else:
                print("\nNo description attribute found")
                
    except Exception as e:
        print(f"Error in actual generation test: {e}")

if __name__ == "__main__":
    test_with_real_barc()
    test_actual_generation()