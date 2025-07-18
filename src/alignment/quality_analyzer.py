"""
Alignment Quality Analyzer for BARC code improvements
"""

import re
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class AlignmentQuality:
    """Analysis of alignment quality"""
    has_concepts: bool
    has_description: bool
    has_transform_function: bool
    has_common_imports: bool
    has_color_constants: bool
    uses_common_functions: bool
    improvement_score: float
    structure_preserved: bool
    code_length_change: int
    description_changed: bool
    

class AlignmentQualityAnalyzer:
    """
    Analyzes the quality of code alignment based on barc_post criteria
    """
    
    def __init__(self):
        # Common.py function names to check for
        self.common_functions = [
            'find_connected_components', 'flood_fill', 'draw_line', 'blit_sprite',
            'blit_object', 'crop', 'translate', 'bounding_box', 'object_position',
            'object_colors', 'scale_sprite', 'collision', 'contact',
            'detect_translational_symmetry', 'detect_mirror_symmetry', 
            'detect_rotational_symmetry', 'orbit', 'random_sprite',
            'object_interior', 'object_boundary', 'object_neighbors',
            'detect_objects', 'is_contiguous'
        ]
        
        # Color constants to check for
        self.color_constants = [
            'Color.BLACK', 'Color.BLUE', 'Color.RED', 'Color.GREEN',
            'Color.YELLOW', 'Color.GREY', 'Color.GRAY', 'Color.PINK',
            'Color.ORANGE', 'Color.TEAL', 'Color.MAROON'
        ]
    
    def analyze_alignment_quality(self, 
                                original_code: str, 
                                aligned_code: str,
                                original_description: Optional[str] = None,
                                aligned_description: Optional[str] = None) -> AlignmentQuality:
        """
        Analyze the quality of alignment following barc_post criteria
        
        Args:
            original_code: Original BARC code
            aligned_code: Aligned code
            original_description: Original description
            aligned_description: Aligned description
            
        Returns:
            AlignmentQuality analysis
        """
        
        # Check basic BARC structure
        has_concepts = self._has_concepts_line(aligned_code)
        has_description = self._has_description_line(aligned_code)
        has_transform_function = self._has_transform_function(aligned_code)
        
        # Check common.py integration
        has_common_imports = self._has_common_imports(aligned_code)
        has_color_constants = self._has_color_constants(aligned_code)
        uses_common_functions = self._uses_common_functions(aligned_code)
        
        # Check structure preservation
        structure_preserved = has_concepts and has_description and has_transform_function
        
        # Calculate improvement score (based on barc_post criteria)
        improvement_score = self._calculate_improvement_score(
            original_code, aligned_code,
            has_concepts, has_description, has_transform_function,
            has_common_imports, has_color_constants, uses_common_functions
        )
        
        # Calculate other metrics
        code_length_change = len(aligned_code) - len(original_code)
        description_changed = (
            original_description is not None and 
            aligned_description is not None and
            original_description.strip() != aligned_description.strip()
        )
        
        return AlignmentQuality(
            has_concepts=has_concepts,
            has_description=has_description,
            has_transform_function=has_transform_function,
            has_common_imports=has_common_imports,
            has_color_constants=has_color_constants,
            uses_common_functions=uses_common_functions,
            improvement_score=improvement_score,
            structure_preserved=structure_preserved,
            code_length_change=code_length_change,
            description_changed=description_changed
        )
    
    def _has_concepts_line(self, code: str) -> bool:
        """Check if code has concepts line"""
        return bool(re.search(r'#\s*concepts?\s*:', code, re.IGNORECASE))
    
    def _has_description_line(self, code: str) -> bool:
        """Check if code has description line"""
        return bool(re.search(r'#\s*description\s*:', code, re.IGNORECASE))
    
    def _has_transform_function(self, code: str) -> bool:
        """Check if code has transform function"""
        return bool(re.search(r'def\s+transform\s*\(', code, re.IGNORECASE))
    
    def _has_common_imports(self, code: str) -> bool:
        """Check if code imports common.py"""
        patterns = [
            r'from\s+common\s+import',
            r'import\s+common'
        ]
        return any(re.search(pattern, code, re.IGNORECASE) for pattern in patterns)
    
    def _has_color_constants(self, code: str) -> bool:
        """Check if code uses Color constants"""
        return any(const in code for const in self.color_constants)
    
    def _uses_common_functions(self, code: str) -> bool:
        """Check if code uses common.py functions"""
        # Count how many common functions are used
        used_functions = sum(1 for func in self.common_functions if func in code)
        return used_functions > 0
    
    def _calculate_improvement_score(self, 
                                   original_code: str,
                                   aligned_code: str,
                                   has_concepts: bool,
                                   has_description: bool, 
                                   has_transform_function: bool,
                                   has_common_imports: bool,
                                   has_color_constants: bool,
                                   uses_common_functions: bool) -> float:
        """
        Calculate improvement score based on barc_post criteria
        
        Score ranges from 0 to 100, with higher scores indicating better alignment
        """
        score = 0.0
        
        # Base requirements (40 points total)
        if has_concepts:
            score += 10
        if has_description:
            score += 10  
        if has_transform_function:
            score += 20  # Most important
        
        # Common.py integration (30 points total)
        if has_common_imports:
            score += 10
        if has_color_constants:
            score += 10
        if uses_common_functions:
            score += 10
        
        # Code quality improvements (30 points total)
        # Check for better error handling
        if 'try:' in aligned_code or 'except:' in aligned_code:
            score += 5
        
        # Check for bounds checking
        if any(pattern in aligned_code for pattern in ['shape[0]', 'shape[1]', 'len(']):
            score += 5
        
        # Check for proper return statement
        if 'return ' in aligned_code:
            score += 5
        
        # Check for comments or documentation
        comment_lines = len([line for line in aligned_code.split('\n') 
                           if line.strip().startswith('#')])
        if comment_lines >= 3:  # concepts, description, plus others
            score += 5
        
        # Penalty for code that's too short (likely incomplete)
        if len(aligned_code.strip()) < 50:
            score -= 30
        
        # Bonus for using multiple common functions
        used_functions = sum(1 for func in self.common_functions if func in aligned_code)
        if used_functions >= 2:
            score += 5
        if used_functions >= 3:
            score += 5
        
        # Ensure score is in valid range
        return max(0.0, min(100.0, score))
    
    def get_alignment_success_criteria(self, quality: AlignmentQuality) -> bool:
        """
        Determine if alignment was successful based on barc_post criteria
        
        Args:
            quality: AlignmentQuality analysis
            
        Returns:
            True if alignment meets success criteria
        """
        # Based on barc_post success criteria
        return (
            quality.structure_preserved and
            quality.improvement_score >= 20 and  # Minimum threshold from barc_post
            quality.code_length_change >= -50  # Code shouldn't shrink too much
        )
    
    def get_quality_summary(self, quality: AlignmentQuality) -> Dict[str, Any]:
        """Get a summary of alignment quality metrics"""
        return {
            'overall_score': quality.improvement_score,
            'structure_preserved': quality.structure_preserved,
            'barc_requirements': {
                'has_concepts': quality.has_concepts,
                'has_description': quality.has_description,
                'has_transform_function': quality.has_transform_function
            },
            'common_integration': {
                'has_imports': quality.has_common_imports,
                'uses_color_constants': quality.has_color_constants,
                'uses_common_functions': quality.uses_common_functions
            },
            'code_changes': {
                'length_change': quality.code_length_change,
                'description_changed': quality.description_changed
            },
            'success': self.get_alignment_success_criteria(quality)
        }