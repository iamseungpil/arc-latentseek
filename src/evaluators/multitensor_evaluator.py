"""
Multi-tensor Evaluator based on CompressARC approach
Evaluates code outputs across multiple dimensions without GLM
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import ast
import re

from ..data import ARCProblem, ARCPair
from ..generators.barc_generator_fixed import BARCOutput
from ..executors import ExecutionResult


@dataclass
class MultiTensorResult:
    """Results from multi-tensor evaluation"""
    execution_reward: float       # Code runs without errors
    accuracy_reward: float        # Correct outputs
    code_quality_reward: float    # Code quality metrics
    structure_reward: float       # Structural similarity
    efficiency_reward: float      # Code efficiency
    total_reward: float          # Weighted sum
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'execution': self.execution_reward,
            'accuracy': self.accuracy_reward,
            'code_quality': self.code_quality_reward,
            'structure': self.structure_reward,
            'efficiency': self.efficiency_reward,
            'total': self.total_reward
        }


class MultiTensorEvaluator:
    """Multi-dimensional evaluator based on CompressARC"""
    
    def __init__(self, 
                 execution_weight: float = 0.2,
                 accuracy_weight: float = 0.4,
                 quality_weight: float = 0.15,
                 structure_weight: float = 0.15,
                 efficiency_weight: float = 0.1):
        """Initialize with weights for different components"""
        self.weights = {
            'execution': execution_weight,
            'accuracy': accuracy_weight,
            'quality': quality_weight,
            'structure': structure_weight,
            'efficiency': efficiency_weight
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
    
    def evaluate(self, 
                problem: ARCProblem, 
                barc_output: BARCOutput,
                execution_result: ExecutionResult,
                prefix: str = "") -> 'EvaluationResult':
        """
        Evaluate solution using multiple tensor components
        
        Args:
            problem: The ARC problem
            barc_output: Generated solution
            execution_result: Result from code execution
            prefix: Prefix for logging (unused but kept for compatibility)
            
        Returns:
            EvaluationResult with multi-tensor rewards
        """
        # 1. Execution reward (binary: 0 or 1)
        execution_reward = 1.0 if execution_result.success else 0.0
        
        # 2. Accuracy reward (percentage of correct outputs)
        accuracy_reward = execution_result.accuracy
        
        # 3. Code quality reward
        code_quality_reward = self._evaluate_code_quality(barc_output.code)
        
        # 4. Structure reward (output structure similarity)
        structure_reward = self._evaluate_structure_similarity(
            problem, execution_result
        )
        
        # 5. Efficiency reward (code length and complexity)
        efficiency_reward = self._evaluate_efficiency(barc_output.code)
        
        # Calculate total reward
        total_reward = (
            self.weights['execution'] * execution_reward +
            self.weights['accuracy'] * accuracy_reward +
            self.weights['quality'] * code_quality_reward +
            self.weights['structure'] * structure_reward +
            self.weights['efficiency'] * efficiency_reward
        )
        
        # Create result
        result = MultiTensorResult(
            execution_reward=execution_reward,
            accuracy_reward=accuracy_reward,
            code_quality_reward=code_quality_reward,
            structure_reward=structure_reward,
            efficiency_reward=efficiency_reward,
            total_reward=total_reward
        )
        
        # Return as EvaluationResult for compatibility
        from .glm_evaluator import EvaluationResult
        return EvaluationResult(
            total_reward=total_reward,
            component_scores=result.to_dict(),
            verifications={},  # Not used in multitensor
            detailed_feedback={
                'execution': f"Code execution: {'Success' if execution_reward > 0 else 'Failed'}",
                'accuracy': f"Accuracy: {accuracy_reward:.1%}",
                'quality': f"Code quality score: {code_quality_reward:.2f}",
                'structure': f"Structure similarity: {structure_reward:.2f}",
                'efficiency': f"Efficiency score: {efficiency_reward:.2f}"
            }
        )
    
    def _evaluate_code_quality(self, code: str) -> float:
        """
        Evaluate code quality based on:
        - Proper function definition
        - Use of numpy operations
        - Proper error handling
        - Code organization
        """
        score = 0.0
        
        # Check for proper function definition
        if 'def transform(' in code:
            score += 0.2
        
        # Check for numpy usage (efficient operations)
        numpy_ops = ['np.', 'numpy.', 'array', 'ndarray']
        if any(op in code for op in numpy_ops):
            score += 0.2
            
        # Check for proper imports
        if 'import numpy' in code or 'from numpy' in code:
            score += 0.1
            
        # Check for comments/documentation
        if code.count('#') > 0 or '"""' in code:
            score += 0.1
            
        # Check for proper structure (not too nested)
        try:
            tree = ast.parse(code)
            max_depth = self._get_max_depth(tree)
            if max_depth < 5:
                score += 0.2
            elif max_depth < 7:
                score += 0.1
        except:
            pass
            
        # Check for error handling
        if 'try:' in code or 'except' in code:
            score += 0.1
            
        # Check for type hints or docstrings
        if '->' in code or '"""' in code:
            score += 0.1
            
        return min(score, 1.0)
    
    def _get_max_depth(self, node, depth=0):
        """Get maximum nesting depth of AST"""
        max_d = depth
        for child in ast.iter_child_nodes(node):
            max_d = max(max_d, self._get_max_depth(child, depth + 1))
        return max_d
    
    def _evaluate_structure_similarity(self, 
                                     problem: ARCProblem,
                                     execution_result: ExecutionResult) -> float:
        """
        Evaluate how well the output structure matches expected:
        - Correct dimensions
        - Valid color values
        - Pattern consistency
        """
        if not execution_result.output_grids:
            return 0.0
            
        total_score = 0.0
        count = 0
        
        for i, (pair, output_grid) in enumerate(zip(problem.train_pairs, execution_result.output_grids)):
            if isinstance(output_grid, np.ndarray):
                score = 0.0
                expected = pair.y
                
                # Check dimensions match
                if output_grid.shape == expected.shape:
                    score += 0.4
                    
                # Check valid color range
                if output_grid.min() >= 0 and output_grid.max() <= 9:
                    score += 0.3
                else:
                    # Partial credit if mostly valid
                    valid_ratio = np.sum((output_grid >= 0) & (output_grid <= 9)) / output_grid.size
                    score += 0.3 * valid_ratio
                    
                # Check if output is not trivial (all same color)
                if len(np.unique(output_grid)) > 1:
                    score += 0.3
                elif len(np.unique(output_grid)) == len(np.unique(expected)):
                    score += 0.15  # Partial credit if at least same diversity
                    
                total_score += score
                count += 1
                
        return total_score / count if count > 0 else 0.0
    
    def _evaluate_efficiency(self, code: str) -> float:
        """
        Evaluate code efficiency:
        - Reasonable length
        - Not overly complex
        - Uses efficient operations
        """
        score = 0.0
        
        # Penalize very short or very long code
        lines = [l.strip() for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]
        num_lines = len(lines)
        
        if 5 <= num_lines <= 30:
            score += 0.4
        elif 3 <= num_lines <= 50:
            score += 0.2
            
        # Check for efficient numpy operations vs loops
        loop_count = code.count('for ') + code.count('while ')
        numpy_count = code.count('np.') + code.count('numpy.')
        
        if numpy_count > loop_count:
            score += 0.3
        elif numpy_count > 0:
            score += 0.15
            
        # Check for vectorized operations
        vectorized_ops = ['reshape', 'transpose', 'flatten', 'where', 'argmax', 'argmin']
        if any(op in code for op in vectorized_ops):
            score += 0.3
            
        return min(score, 1.0)