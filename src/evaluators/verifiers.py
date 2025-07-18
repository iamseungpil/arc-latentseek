"""
Verification utilities
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class VerificationResult:
    """Result of a single verification"""
    passed: bool
    confidence: float  # 0.0 to 1.0
    feedback: str
    
    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({status}, conf={self.confidence:.2f})"


class Verifier:
    """Base verifier class"""
    
    def __init__(self):
        self.verification_types = [
            'calculation_check',
            'answer_correct',
            'answer_completeness',
            'understanding_check'
        ]
        
    def get_verification_prompt(self, verification_type: str, context: Dict) -> str:
        """Get prompt for specific verification type"""
        prompts = {
            'calculation_check': """
Check if the transformations and calculations are correct:
- Are mathematical operations performed correctly?
- Do the transformations follow consistent rules?
- Are there any logical errors in the pattern application?
""",
            'answer_correct': """
Verify if the generated outputs match the expected outputs:
- Do the generated grids have the correct values?
- Are the shapes and dimensions correct?
- Is the overall transformation accurate?
""",
            'answer_completeness': """
Check if the solution is complete:
- Does it provide a definitive output (not partial)?
- Are all necessary transformations applied?
- Is the output properly formatted?
""",
            'understanding_check': """
Verify if the description matches the implementation:
- Does the description accurately explain the transformation?
- Do the concepts align with what's implemented?
- Is there consistency between stated approach and actual output?
"""
        }
        
        return prompts.get(verification_type, "")
    
    def combine_verifications(self, verifications: Dict[str, VerificationResult]) -> Dict:
        """Combine multiple verifications into summary"""
        total_passed = sum(1 for v in verifications.values() if v.passed)
        total_checks = len(verifications)
        
        avg_confidence = sum(v.confidence for v in verifications.values()) / len(verifications) if verifications else 0
        
        return {
            'passed_checks': total_passed,
            'total_checks': total_checks,
            'success_rate': total_passed / total_checks if total_checks > 0 else 0,
            'average_confidence': avg_confidence,
            'all_passed': total_passed == total_checks
        }