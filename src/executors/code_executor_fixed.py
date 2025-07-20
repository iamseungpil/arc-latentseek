"""
Fixed Code Executor for ARC problems with common.py support
"""

import numpy as np
from typing import Union, List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
from func_timeout import func_timeout, FunctionTimedOut
import ast
import sys
import os

# Import from root common.py
root_dir = "/home/ubuntu/arc-latentseek"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from ..data import ARCProblem

# Import all utilities from common (BARC compatible)
from common import *


class GridComparisonResult(Enum):
    """Result of grid comparison"""
    EQUAL = "equal"
    SHAPE_MISMATCH = "shape_mismatch"
    CONTENT_MISMATCH = "content_mismatch"
    ERROR = "error"


@dataclass
class ExecutionResult:
    """Result of code execution on multiple training pairs"""
    success: bool
    output_grids: List[Union[np.ndarray, None]]
    error_messages: List[str]
    comparison_results: List[GridComparisonResult]
    accuracy: float


class CodeExecutor:
    """Execute generated code on ARC problems"""
    
    def __init__(self, timeout: int = 2):
        self.timeout = timeout
        
    def execute(self, code: str, problem: ARCProblem) -> ExecutionResult:
        """
        Execute code on all training pairs
        
        Args:
            code: Python code to execute
            problem: ARC problem with training pairs
            
        Returns:
            ExecutionResult with outputs and accuracy
        """
        output_grids = []
        error_messages = []
        comparison_results = []
        
        for i, pair in enumerate(problem.train_pairs):
            try:
                output = self._execute_single(code, pair.x)
                output_grids.append(output)
                
                # Compare with expected output
                comparison, _ = self._compare_grids(output, pair.y)
                comparison_results.append(comparison)
                
                if isinstance(output, str) and output.startswith("ERROR"):
                    error_messages.append(f"Pair {i}: {output}")
                    
            except Exception as e:
                error_msg = f"Pair {i}: {str(e)}"
                error_messages.append(error_msg)
                output_grids.append(None)
                comparison_results.append(GridComparisonResult.ERROR)
        
        # Calculate accuracy
        correct_count = sum(1 for c in comparison_results if c == GridComparisonResult.EQUAL)
        accuracy = correct_count / len(problem.train_pairs) if problem.train_pairs else 0.0
        
        # Consider execution successful if no runtime errors occur
        # Shape/content mismatches should be handled by GLM evaluation and LatentSeek
        # Only actual errors (syntax, runtime, etc.) should be considered failures
        success = len(error_messages) == 0
        
        return ExecutionResult(
            success=success,
            output_grids=output_grids,
            error_messages=error_messages,
            comparison_results=comparison_results,
            accuracy=accuracy
        )
    
    def execute_single(self, code: str, input_grid: np.ndarray) -> Union[np.ndarray, str]:
        """
        Execute code on a single input grid (convenience method)
        
        Args:
            code: Python code to execute
            input_grid: Single input grid
            
        Returns:
            Output grid or error string
        """
        return self._execute_single(code, input_grid)
    
    def _execute_single(self, code: str, input_grid: np.ndarray) -> Union[np.ndarray, str]:
        """Execute code on a single input grid with timeout"""
        def execute_code():
            try:
                # Create restricted namespace with necessary functions
                namespace = {
                    'np': np,
                    'numpy': np,
                    'enumerate': enumerate,
                    'range': range,
                    'len': len,
                    'list': list,
                    'set': set,
                    'dict': dict,
                    'sorted': sorted,
                    'min': min,
                    'max': max,
                    'sum': sum,
                    'zip': zip,
                    'abs': abs,
                    'all': all,
                    'any': any,
                    'input_grid': input_grid.copy(),
                    # Add Color class and constants
                    'Color': Color,
                    # Add BARC color aliases (8=Purple, 9=Brown instead of Teal/Maroon)
                    'BARC_Color': type('BARCColor', (), {
                        'BLACK': 0, 'BLUE': 1, 'RED': 2, 'GREEN': 3, 'YELLOW': 4,
                        'GRAY': 5, 'GREY': 5, 'PINK': 6, 'ORANGE': 7, 
                        'PURPLE': 8, 'BROWN': 9,  # BARC uses Purple/Brown
                        'TEAL': 8, 'MAROON': 9,   # Keep standard names too
                        'TRANSPARENT': 0, 'BACKGROUND': 0,
                        'ALL_COLORS': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        'NOT_BLACK': [1, 2, 3, 4, 5, 6, 7, 8, 9]
                    }),
                    # Add BARC common utilities
                    'bounding_box': bounding_box,
                    'blit': blit,
                    'flood_fill': flood_fill,
                    'crop': crop,
                    'find_connected_components': find_connected_components,
                    'object_position': object_position,
                    'object_colors': object_colors,
                    'collision': collision,
                    'contact': contact,
                    'detect_translational_symmetry': detect_translational_symmetry,
                    'detect_mirror_symmetry': detect_mirror_symmetry,
                    'detect_rotational_symmetry': detect_rotational_symmetry,
                    'orbit': orbit,
                    'random_sprite': random_sprite,
                    'random_free_location_for_sprite': random_free_location_for_sprite,
                    'object_interior': object_interior,
                    'object_boundary': object_boundary,
                    'object_neighbors': object_neighbors,
                    'detect_objects': detect_objects,
                    'is_contiguous': is_contiguous,
                    # Legacy names and common utilities for compatibility
                    'get_color_mapping': lambda grid: {color: i for i, color in enumerate(np.unique(grid))},
                    'replace_colors': lambda grid, color_map: np.vectorize(color_map.get)(grid, grid),
                    'rotate_grid': lambda grid, rotations=1: np.rot90(grid, k=-rotations),
                    'flip_grid': lambda grid, axis=0: np.flip(grid, axis=axis),
                    # Common rotation and mirror operations
                    'rot90': lambda grid: np.rot90(grid, k=-1),
                    'rot180': lambda grid: np.rot90(grid, k=-2),
                    'rot270': lambda grid: np.rot90(grid, k=-3),
                    'mirrorv': lambda grid: np.flipud(grid),  # vertical mirror
                    'mirrorh': lambda grid: np.fliplr(grid),  # horizontal mirror
                    'fill': flood_fill,  # alias for flood_fill
                    'get_neighbors': lambda pos, shape, connectivity=4: [
                        (r, c) for r, c in (
                            [(pos[0]-1, pos[1]), (pos[0]+1, pos[1]), (pos[0], pos[1]-1), (pos[0], pos[1]+1)] if connectivity == 4
                            else [(pos[0]+dr, pos[1]+dc) for dr in [-1,0,1] for dc in [-1,0,1] if dr != 0 or dc != 0]
                        ) if 0 <= r < shape[0] and 0 <= c < shape[1]
                    ],
                    'count_colors': lambda grid: dict(zip(*np.unique(grid, return_counts=True))),
                    'find_bounding_box': lambda positions: (
                        min(r for r, c in positions), min(c for r, c in positions),
                        max(r for r, c in positions), max(c for r, c in positions)
                    ) if positions else (0, 0, 0, 0),
                    'extract_subgrid': lambda grid, r1, c1, r2, c2: grid[r1:r2+1, c1:c2+1],
                    'paste_subgrid': lambda grid, subgrid, pos: blit(grid, subgrid, pos[0], pos[1]),
                }
                
                # Keep the code as is - don't modify imports
                modified_code = code
                
                # Execute code
                exec(modified_code, namespace)
                
                # Look for transform or main function
                if 'transform' in namespace:
                    return namespace['transform'](input_grid.copy())
                elif 'main' in namespace:
                    return namespace['main'](input_grid.copy())
                else:
                    return "ERROR: transform/main function not found"
                    
            except Exception as e:
                error_msg = f"ERROR: {str(e)}"
                return error_msg
        
        try:
            result = func_timeout(self.timeout, execute_code)
            return result
        except FunctionTimedOut:
            return "ERROR: Execution timed out"
        except Exception as e:
            return f"ERROR: {str(e)}"
    
    def _compare_grids(self, output_grid: Union[np.ndarray, str], 
                      expected_grid: np.ndarray) -> Tuple[GridComparisonResult, float]:
        """
        Compare output grid with expected grid
        
        Returns:
            (comparison_result, similarity_score)
        """
        # Handle error outputs
        if isinstance(output_grid, str):
            return GridComparisonResult.ERROR, 0.0
            
        if output_grid is None:
            return GridComparisonResult.ERROR, 0.0
            
        # Check shape
        if output_grid.shape != expected_grid.shape:
            return GridComparisonResult.SHAPE_MISMATCH, 0.0
            
        # Check content
        if np.array_equal(output_grid, expected_grid):
            return GridComparisonResult.EQUAL, 1.0
        else:
            # Calculate similarity
            matching_pixels = np.sum(output_grid == expected_grid)
            total_pixels = output_grid.size
            similarity = matching_pixels / total_pixels if total_pixels > 0 else 0.0
            return GridComparisonResult.CONTENT_MISMATCH, similarity