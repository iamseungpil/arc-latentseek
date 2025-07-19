#!/usr/bin/env python3
"""
Test the correct solution for problem 2a5f8217
"""

import sys
import os
import numpy as np

# Add paths
sys.path.append('/home/ubuntu/arc-latentseek')

# Import modules
from src.data.arc_loader import ARCDataLoader
from src.executors.code_executor_fixed import CodeExecutor

def print_grid(grid, title="Grid"):
    """Pretty print a grid with colors"""
    color_map = {
        0: "⬛", 1: "🟦", 2: "🟥", 3: "🟩", 4: "🟨",
        5: "⬜", 6: "🟪", 7: "🟧", 8: "🟫", 9: "🟫"
    }
    print(f"\n{title}:")
    for row in grid:
        print("".join(color_map.get(int(cell), "❓") for cell in row))

# The correct solution based on our analysis
correct_solution = """
def transform(input_grid):
    '''
    Pattern: Each blue (1) pixel gets replaced by the color of the nearest non-blue, non-black object.
    The replacement seems to be context-dependent - blue objects take on the color of nearby objects.
    '''
    import numpy as np
    
    output = input_grid.copy()
    
    # Find all blue positions
    blue_positions = list(zip(*np.where(input_grid == 1)))
    
    if not blue_positions:
        return output
    
    # For each blue position, find the nearest non-blue, non-black color
    for row, col in blue_positions:
        min_dist = float('inf')
        replacement_color = 1  # default to blue if nothing found
        
        # Search for nearest non-blue, non-black pixel
        for r in range(input_grid.shape[0]):
            for c in range(input_grid.shape[1]):
                color = input_grid[r, c]
                if color != 0 and color != 1:  # not black, not blue
                    dist = abs(r - row) + abs(c - col)  # Manhattan distance
                    if dist < min_dist:
                        min_dist = dist
                        replacement_color = color
        
        output[row, col] = replacement_color
    
    return output
"""

# Alternative solution that looks at connected components
alternative_solution = """
def transform(input_grid):
    '''
    Pattern: Blue objects get replaced by the color of other objects in the same "region" or nearby.
    Looking at the examples, it seems like blue shapes take on the color of other shapes
    that are in the same part of the grid.
    '''
    import numpy as np
    
    output = input_grid.copy()
    
    # Get all unique non-black, non-blue colors
    unique_colors = [c for c in np.unique(input_grid) if c != 0 and c != 1]
    
    if not unique_colors:
        return output
    
    # Find connected blue components
    from scipy import ndimage
    blue_mask = input_grid == 1
    labeled, num_features = ndimage.label(blue_mask)
    
    # For each blue component, find the best replacement color
    for component_id in range(1, num_features + 1):
        component_mask = labeled == component_id
        component_positions = list(zip(*np.where(component_mask)))
        
        if not component_positions:
            continue
            
        # Calculate center of this blue component
        center_r = sum(r for r, c in component_positions) / len(component_positions)
        center_c = sum(c for r, c in component_positions) / len(component_positions)
        
        # Find nearest non-blue, non-black object
        min_dist = float('inf')
        best_color = unique_colors[0] if unique_colors else 1
        
        for color in unique_colors:
            color_positions = list(zip(*np.where(input_grid == color)))
            if color_positions:
                # Find closest point of this color to the blue component center
                for r, c in color_positions:
                    dist = ((r - center_r) ** 2 + (c - center_c) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        best_color = color
        
        # Replace all pixels in this blue component
        output[component_mask] = best_color
    
    return output
"""

# Simpler heuristic that seems to work
simple_solution = """
def transform(input_grid):
    '''
    Looking at the examples more carefully:
    - Example 1: Blue becomes 8 (the only other non-black color)
    - Example 2: Different blue regions become different colors based on proximity
    - Example 3: Similar pattern
    
    The pattern seems to be that each separate blue object takes on the color
    of the nearest non-blue object.
    '''
    import numpy as np
    from scipy import ndimage
    
    output = input_grid.copy()
    
    # Find connected blue components
    blue_mask = input_grid == 1
    labeled, num_components = ndimage.label(blue_mask, structure=[[0,1,0],[1,1,1],[0,1,0]])
    
    # For each blue component
    for comp_id in range(1, num_components + 1):
        comp_mask = labeled == comp_id
        comp_positions = np.argwhere(comp_mask)
        
        if len(comp_positions) == 0:
            continue
            
        # Find center of component
        center = comp_positions.mean(axis=0)
        
        # Find all non-blue, non-black positions
        other_colors_mask = (input_grid != 0) & (input_grid != 1)
        other_positions = np.argwhere(other_colors_mask)
        
        if len(other_positions) == 0:
            continue
            
        # Find closest non-blue pixel
        distances = np.sum((other_positions - center) ** 2, axis=1)
        closest_idx = np.argmin(distances)
        closest_pos = other_positions[closest_idx]
        
        # Get the color at that position
        replacement_color = input_grid[closest_pos[0], closest_pos[1]]
        
        # Replace this blue component with that color
        output[comp_mask] = replacement_color
    
    return output
"""

def main():
    print("="*80)
    print("Testing Correct Solutions for Problem 2a5f8217")
    print("="*80)
    
    # Load the problem
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    if not problem:
        print("❌ Problem not found!")
        return
    
    # Test executor
    executor = CodeExecutor(timeout=5)
    
    # Test each solution
    solutions = [
        ("Nearest Color Solution", correct_solution),
        ("Component-Based Solution", alternative_solution),
        ("Simple Heuristic Solution", simple_solution)
    ]
    
    for name, code in solutions:
        print(f"\n\n{'='*60}")
        print(f"Testing: {name}")
        print(f"{'='*60}")
        
        # Execute on all training pairs
        result = executor.execute(code, problem)
        
        print(f"\nSuccess: {result.success}")
        print(f"Accuracy: {result.accuracy:.2%}")
        
        if result.error_messages:
            print("\nErrors:")
            for error in result.error_messages:
                print(f"  - {error}")
        
        # Show results for each training pair
        for i, (pair, output_grid, comparison) in enumerate(
            zip(problem.train_pairs, result.output_grids, result.comparison_results)
        ):
            print(f"\nTraining Pair {i+1}: {comparison.value}")
            
            if output_grid is not None and not isinstance(output_grid, str):
                if comparison.value != "equal":
                    print_grid(pair.x, "  Input")
                    print_grid(output_grid, "  Generated")
                    print_grid(pair.y, "  Expected")
        
        # If successful, test on test input
        if result.accuracy == 1.0 and problem.test_pairs:
            print("\n✅ Perfect accuracy! Testing on test input...")
            test_output = executor.execute_single(code, problem.test_pairs[0].x)
            
            if isinstance(test_output, np.ndarray):
                print_grid(problem.test_pairs[0].x, "Test Input")
                print_grid(test_output, "Test Output")
                
                # Show unique colors in output
                print(f"\nOutput colors: {sorted(np.unique(test_output))}")

if __name__ == "__main__":
    main()