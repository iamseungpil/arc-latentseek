#!/usr/bin/env python3
"""
Final pattern analysis - understand the exact rule
"""

import sys
import os
import numpy as np
from scipy import ndimage

# Add paths
sys.path.append('/home/ubuntu/arc-latentseek')

# Import modules
from src.data.arc_loader import ARCDataLoader

def analyze_blue_replacements(input_grid, output_grid):
    """Analyze how each blue component gets replaced"""
    # Find connected blue components
    blue_mask = input_grid == 1
    labeled, num_components = ndimage.label(blue_mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
    
    print(f"\nFound {num_components} blue components")
    
    for comp_id in range(1, num_components + 1):
        comp_mask = labeled == comp_id
        positions = np.argwhere(comp_mask)
        
        if len(positions) == 0:
            continue
            
        # Get replacement color
        replacement_colors = output_grid[comp_mask]
        unique_replacements = np.unique(replacement_colors)
        
        print(f"\nBlue component {comp_id}:")
        print(f"  Positions: {positions[:3].tolist()}{'...' if len(positions) > 3 else ''}")
        print(f"  Size: {len(positions)} pixels")
        print(f"  Replaced with: {unique_replacements}")
        
        # Find what other objects are in the same row/column
        rows = positions[:, 0]
        cols = positions[:, 1]
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        print(f"  Bounding box: rows {min_row}-{max_row}, cols {min_col}-{max_col}")
        
        # Look for other colors in same rows/cols
        row_colors = set()
        col_colors = set()
        
        for r in range(min_row, max_row + 1):
            row_vals = input_grid[r, :]
            row_colors.update(row_vals[row_vals > 1])  # non-black, non-blue
            
        for c in range(min_col, max_col + 1):
            col_vals = input_grid[:, c]
            col_colors.update(col_vals[col_vals > 1])
            
        print(f"  Other colors in same rows: {sorted(row_colors)}")
        print(f"  Other colors in same cols: {sorted(col_colors)}")

def main():
    # Load the problem
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    if not problem:
        print("❌ Problem not found!")
        return
    
    print("="*80)
    print("DETAILED PATTERN ANALYSIS")
    print("="*80)
    
    # Analyze each example
    for i, pair in enumerate(problem.train_pairs):
        print(f"\n\nEXAMPLE {i+1}:")
        print("="*40)
        
        print("\nInput grid:")
        print(pair.x)
        
        print("\nOutput grid:")
        print(pair.y)
        
        analyze_blue_replacements(pair.x, pair.y)
    
    # Final insight
    print("\n\n" + "="*80)
    print("FINAL INSIGHT:")
    print("="*80)
    
    print("""
Looking at the patterns:

Example 1:
- Single blue object at top-left
- Gets replaced with color 8 (the other object in the grid)

Example 2:
- Multiple blue objects in different locations
- Top-left blue object (rows 1-2) -> becomes 6 (pink/purple)
- Bottom blue objects (rows 5-6) -> become 9 (brown/maroon)
- Right blue object -> becomes 7 (orange)

Example 3:
- Multiple blue objects
- Top-left blue (rows 0-2) -> becomes 7 (orange) 
- Top-right blue (row 0, col 5) -> becomes 6 (pink)
- Middle-right blue (rows 3-5) -> becomes 9 (brown)
- Bottom blue (rows 8-9) -> becomes 3 (green)

The pattern seems to be that each blue object takes the color of another 
non-blue object that is somehow "paired" with it - possibly based on
relative positions or some other spatial relationship.
""")

if __name__ == "__main__":
    main()