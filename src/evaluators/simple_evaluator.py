"""
Simple evaluator wrapper for V12 experiment
"""
import numpy as np
from typing import Dict, List, Any
import re
import sys
import os

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

# Import common.py functionality
from common import *
from src.executors.code_executor_fixed import CodeExecutor


class SimpleEvaluator:
    """Simple evaluator for V12 experiment"""
    
    def __init__(self):
        self.executor = CodeExecutor(timeout=2)
    
    def evaluate_solution(self, problem_id: str, code: str) -> Dict[str, Any]:
        """
        Evaluate a solution code
        
        Returns dict with:
        - execution_success: bool
        - generated_outputs: List[np.ndarray] or None
        - accuracy: float (0.0 to 1.0)
        - error: str or None
        """
        # Import arc to get problem
        import arc
        
        # Get problem
        problem = None
        for p in arc.validation_problems:
            if p.uid == problem_id:
                problem = p
                break
                
        if problem is None:
            return {
                "execution_success": False,
                "generated_outputs": None,
                "accuracy": 0.0,
                "error": f"Problem {problem_id} not found"
            }
        
        # Extract main function
        func_match = re.search(r'def main\(.*?\):(.*?)(?=\n(?:def|$))', code, re.DOTALL)
        if not func_match:
            return {
                "execution_success": False,
                "generated_outputs": None,
                "accuracy": 0.0,
                "error": "No main function found"
            }
        
        # Try to execute on test pairs
        try:
            # Create exec environment
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
            exec_globals.update(globals())
            
            # Execute code
            exec(code, exec_globals)
            
            # Get main function
            if 'main' not in exec_globals:
                return {
                    "execution_success": False,
                    "generated_outputs": None,
                    "accuracy": 0.0,
                    "error": "main function not defined after execution"
                }
                
            main_func = exec_globals['main']
            
            # Execute on test inputs
            generated_outputs = []
            for test_pair in problem.test_pairs:
                try:
                    output = main_func(test_pair.x)
                    if not isinstance(output, np.ndarray):
                        output = np.array(output)
                    generated_outputs.append(output)
                except Exception as e:
                    return {
                        "execution_success": False,
                        "generated_outputs": None,
                        "accuracy": 0.0,
                        "error": f"Error executing main: {str(e)}"
                    }
            
            # Calculate accuracy
            correct = 0
            for gen_out, test_pair in zip(generated_outputs, problem.test_pairs):
                if gen_out.shape == test_pair.y.shape and np.array_equal(gen_out, test_pair.y):
                    correct += 1
                    
            accuracy = correct / len(problem.test_pairs) if len(problem.test_pairs) > 0 else 0.0
            
            return {
                "execution_success": True,
                "generated_outputs": generated_outputs,
                "accuracy": accuracy,
                "error": None
            }
            
        except Exception as e:
            return {
                "execution_success": False,
                "generated_outputs": None,
                "accuracy": 0.0,
                "error": str(e)
            }