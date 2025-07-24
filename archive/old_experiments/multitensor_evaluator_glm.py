"""
Multi-tensor GLM Evaluator based on CompressARC approach
Evaluates code outputs across 5 dimensions
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from PIL import Image, ImageDraw, ImageFont
import tempfile
import os

from .glm_evaluator import GLMEvaluator, EvaluationResult
from ..data import ARCProblem, ARCPair
from ..executors import GridRenderer

# ARC color palette
COLORS = {
    0: (0, 0, 0),        # Black
    1: (0, 116, 217),    # Blue  
    2: (255, 65, 54),    # Red
    3: (46, 204, 64),    # Green
    4: (255, 220, 0),    # Yellow
    5: (128, 128, 128),  # Gray
    6: (240, 18, 190),   # Magenta
    7: (255, 133, 27),   # Orange
    8: (0, 191, 255),    # Sky Blue
    9: (149, 0, 58),     # Maroon
}


@dataclass
class MultiTensorResult:
    """Results from multi-tensor evaluation"""
    example_accuracy: float  # Per-example accuracy
    color_transformation: float  # Color mapping accuracy
    spatial_transformation: float  # Spatial transformation accuracy
    pattern_recognition: float  # Pattern recognition accuracy
    structural_integrity: float  # Structure preservation accuracy
    overall_score: float  # Weighted average
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'example_accuracy': self.example_accuracy,
            'color_transformation': self.color_transformation,
            'spatial_transformation': self.spatial_transformation,
            'pattern_recognition': self.pattern_recognition,
            'structural_integrity': self.structural_integrity,
            'overall_score': self.overall_score
        }


class MultiTensorEvaluator(GLMEvaluator):
    """Multi-dimensional evaluator using GLM vision model"""
    
    def __init__(self, model_name: str = "THUDM/GLM-4.1V-9B-Thinking"):
        super().__init__(model_name)
        self.renderer = GridRenderer()
    
    def evaluate_multitensor(
        self, 
        problem: ARCProblem, 
        generated_outputs: List[np.ndarray]
    ) -> MultiTensorResult:
        """Evaluate using 5-dimensional multi-tensor approach"""
        
        # Create comparison image
        image_path = self._create_comparison_image(problem, generated_outputs)
        
        # Prompt for multi-tensor evaluation
        prompt = self._create_multitensor_prompt()
        
        # Get GLM evaluation
        response = self._run_glm_inference(image_path, prompt)
        
        # Parse scores from response
        # response is a string from _run_glm_inference
        scores = self._parse_multitensor_scores(response)
        
        # Clean up temp image
        if os.path.exists(image_path):
            os.unlink(image_path)
        
        return scores
    
    def _create_comparison_image(
        self, 
        problem: ARCProblem, 
        generated_outputs: List[np.ndarray]
    ) -> str:
        """Create comparison image showing input, expected, and generated outputs"""
        
        # Calculate dimensions
        num_examples = len(problem.train_pairs)
        grid_size = 100
        padding = 20
        
        # Create large image
        img_width = (3 * grid_size + 4 * padding)  # input, expected, generated + padding
        img_height = num_examples * (grid_size + padding) + padding
        
        img = Image.new('RGB', (img_width, img_height), 'white')
        draw = ImageDraw.Draw(img)
        
        # Try to load font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Add headers
        draw.text((padding, 5), "Input", fill='black', font=font)
        draw.text((grid_size + 2*padding, 5), "Expected", fill='black', font=font)
        draw.text((2*grid_size + 3*padding, 5), "Generated", fill='red', font=font)
        
        for i, (train_pair, gen_output) in enumerate(zip(problem.train_pairs, generated_outputs)):
            y_offset = i * (grid_size + padding) + padding + 25
            
            # Input grid
            input_img = self._create_grid_image(train_pair.x, grid_size)
            img.paste(input_img, (padding, y_offset))
            
            # Expected output
            expected_img = self._create_grid_image(train_pair.y, grid_size)
            img.paste(expected_img, (grid_size + 2*padding, y_offset))
            
            # Generated output
            gen_img = self._create_grid_image(gen_output, grid_size)
            img.paste(gen_img, (2*grid_size + 3*padding, y_offset))
        
        # Save to temp file
        temp_path = tempfile.mktemp(suffix='.png')
        img.save(temp_path)
        return temp_path
    
    def _create_grid_image(self, grid: np.ndarray, size: int) -> Image.Image:
        """Create PIL image from numpy grid"""
        cell_size = size // max(grid.shape)
        img = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(img)
        
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                color = COLORS.get(grid[i, j], (255, 255, 255))
                x1 = j * cell_size
                y1 = i * cell_size
                x2 = (j + 1) * cell_size
                y2 = (i + 1) * cell_size
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='black')
        
        return img
    
    def _create_multitensor_prompt(self) -> str:
        """Create prompt for multi-tensor evaluation"""
        return """Analyze this ARC problem across 5 dimensions and provide scores as percentages (0-100%):

1. EXAMPLE ACCURACY: How well does the generated output match the expected output for each training example?
   - Look at pixel-by-pixel correspondence
   - Consider overall pattern similarity

2. COLOR TRANSFORMATION: How accurately are colors transformed from input to output?
   - Color mapping correctness
   - Color preservation where needed
   - New color generation accuracy

3. SPATIAL TRANSFORMATION: How well are spatial relationships handled?
   - X-axis transformations (horizontal)
   - Y-axis transformations (vertical) 
   - Rotation/reflection accuracy
   - Size/scale handling

4. PATTERN RECOGNITION: How well are patterns identified and applied?
   - Input pattern detection
   - Rule application consistency
   - Edge case handling

5. STRUCTURAL INTEGRITY: How well is the overall structure preserved?
   - Grid dimensions correctness
   - Object boundary preservation
   - Connectivity maintenance

Provide your scores in this exact format:
Example Accuracy: X%
Color Transformation: Y%
Spatial Transformation: Z%
Pattern Recognition: W%
Structural Integrity: V%

Then provide a brief explanation of the main issues."""
    
    def _parse_multitensor_scores(self, explanation: str) -> MultiTensorResult:
        """Parse scores from GLM explanation"""
        scores = {
            'example_accuracy': 0.0,
            'color_transformation': 0.0,
            'spatial_transformation': 0.0,
            'pattern_recognition': 0.0,
            'structural_integrity': 0.0
        }
        
        # Parse percentage scores
        lines = explanation.split('\n')
        for line in lines:
            line = line.strip()
            if 'Example Accuracy:' in line:
                scores['example_accuracy'] = self._extract_percentage(line)
            elif 'Color Transformation:' in line:
                scores['color_transformation'] = self._extract_percentage(line)
            elif 'Spatial Transformation:' in line:
                scores['spatial_transformation'] = self._extract_percentage(line)
            elif 'Pattern Recognition:' in line:
                scores['pattern_recognition'] = self._extract_percentage(line)
            elif 'Structural Integrity:' in line:
                scores['structural_integrity'] = self._extract_percentage(line)
        
        # Calculate weighted average (structural integrity weighted higher)
        weights = {
            'example_accuracy': 0.3,
            'color_transformation': 0.2,
            'spatial_transformation': 0.2,
            'pattern_recognition': 0.2,
            'structural_integrity': 0.1
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores.keys())
        
        return MultiTensorResult(
            example_accuracy=scores['example_accuracy'],
            color_transformation=scores['color_transformation'],
            spatial_transformation=scores['spatial_transformation'],
            pattern_recognition=scores['pattern_recognition'],
            structural_integrity=scores['structural_integrity'],
            overall_score=overall_score
        )
    
    def _extract_percentage(self, text: str) -> float:
        """Extract percentage from text like 'Score: 75%'"""
        import re
        match = re.search(r'(\d+(?:\.\d+)?)%', text)
        if match:
            return float(match.group(1))
        return 0.0


def convert_multitensor_to_reward(result: MultiTensorResult) -> float:
    """Convert multi-tensor result to reward value (negative for minimization)"""
    # Convert percentage to negative reward (higher percentage = less negative reward)
    return -(100.0 - result.overall_score) / 100.0