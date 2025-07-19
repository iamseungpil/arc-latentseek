"""
Reward model for ARC solutions
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from .verifiers import VerificationResult


@dataclass
class RewardConfig:
    """Configuration for reward calculation"""
    weights: Dict[str, float]
    threshold: float = -0.2  # Stop optimization if reward > threshold
    format_penalty: float = -2.0
    
    @classmethod
    def default(cls):
        """Default reward configuration from LatentSeek"""
        return cls(
            weights={
                'calculation_check': 2.0,
                'answer_correct': 1.0,
                'answer_completeness': 2.0,
                'understanding_check': 1.0
            }
        )


class RewardModel:
    """Calculate rewards based on verifications"""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig.default()
        
    def calculate_reward(self, 
                        verifications: Dict[str, VerificationResult],
                        has_format_error: bool = False) -> float:
        """
        Calculate binary reward: 1.0 if answer_correct is TRUE, else penalty
        
        Args:
            verifications: Dictionary of verification results
            has_format_error: Whether there's a format error
            
        Returns:
            1.0 if answer_correct=TRUE, else negative penalty
        """
        # Check if answer is correct (must solve ALL training examples)
        answer_correct = verifications.get('answer_correct')
        if answer_correct and answer_correct.passed:
            return 1.0  # Perfect score
        
        # If not correct, calculate penalty based on other verifications
        reward = 0.0
        total_weight = 0.0
        
        for name, verification in verifications.items():
            if name == 'answer_correct':
                continue  # Skip answer_correct for penalty calculation
                
            weight = self.config.weights.get(name, 1.0)
            total_weight += weight
            
            if not verification.passed:
                penalty = -weight * verification.confidence
                reward += penalty
                
        # Normalize to penalty range
        if total_weight > 0:
            reward = reward / total_weight
        else:
            reward = -1.0  # Default penalty if no other verifications
            
        # Apply format penalty if needed
        if has_format_error:
            reward += self.config.format_penalty
            
        return reward
    
    def should_stop_optimization(self, reward: float) -> bool:
        """Check if optimization should stop based on reward"""
        return reward >= 1.0  # Stop only when perfect solution found
    
    def get_reward_breakdown(self, 
                           verifications: Dict[str, VerificationResult]) -> Dict[str, float]:
        """Get detailed breakdown of reward components"""
        breakdown = {}
        
        for name, verification in verifications.items():
            weight = self.config.weights.get(name, 1.0)
            
            if verification.passed:
                breakdown[name] = 0.0
            else:
                breakdown[name] = -weight * verification.confidence
                
        return breakdown