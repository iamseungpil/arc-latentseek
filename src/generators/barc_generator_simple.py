"""
Simple BARC generator that accepts pre-loaded model
"""

import torch
import numpy as np
from typing import Dict, Optional
import re

from .code_parser import extract_code_elements


class BARCGeneratorSimple:
    """Simple generator that accepts pre-loaded model and tokenizer"""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def generate(self, problem_id: str) -> Dict:
        """Generate solution for a problem"""
        # Get problem from arc
        import arc
        
        problem = None
        for p in arc.validation_problems:
            if p.uid == problem_id:
                problem = p
                break
                
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found")
            
        # Create prompt
        prompt = self._create_prompt(problem)
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract code from response
        code = self._extract_code(response)
        
        # Extract elements
        concepts, description, plan = extract_code_elements(code)
        
        return {
            "code": code,
            "concepts": concepts,
            "description": description,
            "plan": plan,
            "raw_response": response
        }
        
    def _create_prompt(self, problem) -> str:
        """Create prompt for the problem"""
        # Color mapping
        COLOR_MAP = {
            0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
            5: "Gray", 6: "Pink", 7: "Orange", 8: "Teal", 9: "Maroon"
        }
        
        prompt = "### System:\nYou are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your goal is to analyze puzzles and provide solutions in well-documented code.\n\n"
        prompt += "### Instruction:\nSolve the following ARC (Abstraction and Reasoning Corpus) puzzle by examining input-output pairs and determining the transformation rule.\n\n"
        
        # Add examples
        for i, pair in enumerate(problem.train_pairs):
            prompt += f"Example {i+1}:\n"
            prompt += "Input:\n"
            prompt += self._grid_to_string(pair.x, COLOR_MAP)
            prompt += "\nOutput:\n"
            prompt += self._grid_to_string(pair.y, COLOR_MAP)
            prompt += "\n\n"
            
        # Add test input
        if problem.test_pairs:
            prompt += "Test Input:\n"
            prompt += self._grid_to_string(problem.test_pairs[0].x, COLOR_MAP)
            prompt += "\n\n"
            
        prompt += """Based on the examples, determine the transformation rule and provide a Python solution.

Your response should include:
1. A list of concepts used in the solution
2. A description of the transformation rule
3. A main(input_grid) function that implements the solution

Format your response as:
```python
# concepts:
# [list concepts separated by commas]

# description:
# [describe the transformation rule]

def main(input_grid):
    # implementation
    return output_grid
```

### Response:
"""
        
        return prompt
        
    def _grid_to_string(self, grid: np.ndarray, color_map: Dict[int, str]) -> str:
        """Convert grid to string representation"""
        lines = []
        for row in grid:
            row_str = " ".join([color_map.get(int(cell), str(cell)) for cell in row])
            lines.append(row_str)
        return "\n".join(lines)
        
    def _extract_code(self, response: str) -> str:
        """Extract code from response"""
        # Look for code block
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
            
        # Try to find code without markers
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if 'def main' in line:
                in_code = True
            if in_code:
                code_lines.append(line)
                
        if code_lines:
            return '\n'.join(code_lines)
            
        # Return entire response if no code found
        return response