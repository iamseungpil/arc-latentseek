"""
Basic ARC problem evaluator
"""
import numpy as np
from typing import List, Tuple
import logging

from ..data import ARCPair

logger = logging.getLogger(__name__)


class ARCEvaluator:
    """Evaluates ARC problem solutions"""
    
    def evaluate_outputs(
        self, 
        expected_pairs: List[ARCPair], 
        generated_outputs: List[np.ndarray]
    ) -> float:
        """
        Evaluate generated outputs against expected outputs
        
        Args:
            expected_pairs: List of (input, expected_output) pairs
            generated_outputs: List of generated outputs
            
        Returns:
            Accuracy percentage (0-100)
        """
        if len(expected_pairs) != len(generated_outputs):
            logger.warning(
                f"Number of expected pairs ({len(expected_pairs)}) != "
                f"generated outputs ({len(generated_outputs)})"
            )
            return 0.0
        
        correct_count = 0
        
        for pair, generated in zip(expected_pairs, generated_outputs):
            expected = pair.y
            
            # Check if shapes match
            if generated.shape != expected.shape:
                logger.debug(
                    f"Shape mismatch: expected {expected.shape}, "
                    f"got {generated.shape}"
                )
                continue
            
            # Check if all elements match
            if np.array_equal(generated, expected):
                correct_count += 1
            else:
                # Log the difference for debugging
                diff_count = np.sum(generated != expected)
                total_elements = generated.size
                logger.debug(
                    f"Output mismatch: {diff_count}/{total_elements} "
                    f"elements differ ({diff_count/total_elements*100:.1f}%)"
                )
        
        accuracy = (correct_count / len(expected_pairs)) * 100.0
        logger.info(f"Evaluation result: {correct_count}/{len(expected_pairs)} correct ({accuracy:.1f}%)")
        
        return accuracy
    
    def evaluate_single(
        self,
        expected: np.ndarray,
        generated: np.ndarray
    ) -> bool:
        """
        Evaluate a single output
        
        Args:
            expected: Expected output grid
            generated: Generated output grid
            
        Returns:
            True if outputs match exactly
        """
        if generated.shape != expected.shape:
            return False
        
        return np.array_equal(generated, expected)