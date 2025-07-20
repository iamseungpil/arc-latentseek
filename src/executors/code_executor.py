"""
Safe code execution for ARC solutions
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import traceback
from func_timeout import func_timeout, FunctionTimedOut

from ..data import ARCProblem, ARCPair

# Import common from project root, not local executors
import sys
import os
# Hardcode the root path to ensure we get the right common.py
root_dir = "/home/ubuntu/arc-latentseek"
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
from common import *  # Import BARC common utilities including Color class from root


class GridComparisonResult(Enum):
    EQUAL = 0
    SHAPE_MISMATCH = 1
    CONTENT_MISMATCH = 2
    TYPE_MISMATCH = 3
    ERROR = 4
    NON_2D_ARRAY = 5


@dataclass
class ExecutionResult:
    """Result of code execution"""
    success: bool
    output_grids: List[np.ndarray]  # Output grids for each training example
    error_messages: List[str]
    comparison_results: List[GridComparisonResult]
    accuracy: float  # Overall accuracy across all training examples
    
    def __repr__(self):
        return f"ExecutionResult(success={self.success}, accuracy={self.accuracy:.2f})"


class CodeExecutor:
    """Execute generated code safely with timeout"""
    
    def __init__(self, timeout: int = 2):
        self.timeout = timeout
        
    def execute(self, code: str, problem: ARCProblem) -> ExecutionResult:
        """
        Execute code on all training pairs of a problem
        
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
                elif output is None:
                    error_messages.append(f"Pair {i}: Function returned None")
                elif not isinstance(output, np.ndarray):
                    error_messages.append(f"Pair {i}: Function returned {type(output).__name__} instead of numpy array")
                    
            except Exception as e:
                error_msg = f"Pair {i}: {str(e)}"
                error_messages.append(error_msg)
                output_grids.append(None)
                comparison_results.append(GridComparisonResult.ERROR)
        
        # Calculate accuracy
        correct_count = sum(1 for c in comparison_results if c == GridComparisonResult.EQUAL)
        accuracy = correct_count / len(problem.train_pairs) if problem.train_pairs else 0.0
        
        # Consider execution successful if no errors OR only shape/content mismatches
        # This allows LatentSeek to try fixing shape issues
        acceptable_mismatches = {GridComparisonResult.SHAPE_MISMATCH, GridComparisonResult.CONTENT_MISMATCH}
        has_only_acceptable_mismatches = all(
            c in acceptable_mismatches or c == GridComparisonResult.EQUAL 
            for c in comparison_results
        )
        
        # Success if no runtime errors, even if shapes don't match
        # Shape/content mismatches should be handled by LatentSeek optimization
        success = len(error_messages) == 0 and has_only_acceptable_mismatches
        
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
                    # Add BARC common utilities
                    'flood_fill': flood_fill,
                    'draw_line': draw_line,
                    'find_connected_components': find_connected_components,
                    'blit': blit,
                    'blit_sprite': blit_sprite,
                    'blit_object': blit_object,
                    'scale_sprite': scale_sprite,
                    'crop': crop,
                    'translate': translate,
                    'bounding_box': bounding_box,
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
                    # Legacy names for compatibility
                    'get_color_mapping': lambda grid: {color: i for i, color in enumerate(np.unique(grid))},
                    'replace_colors': lambda grid, color_map: np.vectorize(color_map.get)(grid, grid),
                    'rotate_grid': lambda grid, rotations=1: np.rot90(grid, k=-rotations),
                    'flip_grid': lambda grid, axis=0: np.flip(grid, axis=axis),
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
                
                # Import common module from root directory
                import sys
                import os
                # Force use of the project root directory
                # Hardcode the path since __file__ might not be reliable in exec context
                root_dir = "/home/ubuntu/arc-latentseek"
                
                # Add root to path if not already there
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)
                
                # Remove any other paths that might contain a different common module
                # Keep only standard paths and our root
                original_paths = sys.path.copy()
                sys.path = [p for p in sys.path if p == root_dir or not os.path.exists(os.path.join(p, 'common.py')) or 'site-packages' in p]
                
                # Now import the root common module
                import importlib
                if 'common' in sys.modules:
                    # Reload to ensure we get the right one
                    importlib.reload(sys.modules['common'])
                import common
                
                # Restore paths
                sys.path = original_paths
                
                # Add common module and all its symbols to namespace
                namespace['common'] = common
                for attr in dir(common):
                    if not attr.startswith('_'):
                        namespace[attr] = getattr(common, attr)
                
                # Execute code
                exec(code, namespace)
                
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
        """Compare output grid with expected grid"""
        if isinstance(output_grid, str) and output_grid.startswith("ERROR"):
            return GridComparisonResult.ERROR, 0.0
        
        if not isinstance(output_grid, np.ndarray):
            return GridComparisonResult.TYPE_MISMATCH, 0.0
        
        if len(output_grid.shape) != 2:
            return GridComparisonResult.NON_2D_ARRAY, 0.0
        
        if output_grid.shape != expected_grid.shape:
            return GridComparisonResult.SHAPE_MISMATCH, 0.0
        
        if np.array_equal(output_grid, expected_grid):
            return GridComparisonResult.EQUAL, 1.0
        
        # Calculate partial match ratio
        try:
            match_count = np.sum(output_grid == expected_grid)
            total_elements = np.prod(expected_grid.shape)
            ratio = match_count / total_elements
            return GridComparisonResult.CONTENT_MISMATCH, float(ratio)
        except Exception:
            return GridComparisonResult.ERROR, 0.0