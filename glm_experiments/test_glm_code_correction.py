#!/usr/bin/env python3
"""
GLM Experiment 4: Code Correction Test
Test GLM's ability to correct failing code
"""

import os
import sys
import json
import torch
from PIL import Image
import numpy as np
from typing import Dict, Tuple
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import ARCDataLoader
from src.executors import CodeExecutor, GridRenderer
from transformers import AutoProcessor, Glm4vForConditionalGeneration

class GLMCodeCorrectionTester:
    def __init__(self):
        """Initialize experiment runner"""
        self.data_loader = ARCDataLoader()
        self.code_executor = CodeExecutor()
        self.renderer = GridRenderer()
        
        # Load GLM model
        print("Loading GLM model...")
        self.processor = AutoProcessor.from_pretrained("THUDM/GLM-4.1V-9B-Thinking", use_fast=True)
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            "THUDM/GLM-4.1V-9B-Thinking",
            torch_dtype=torch.bfloat16,
            device_map={"": 0}  # Will use CUDA_VISIBLE_DEVICES
        )
        print("GLM model loaded")
        
    def get_test_problem(self) -> Tuple[str, str, str]:
        """Get problem with 0% accuracy and no errors"""
        problem_id = "2072aba6"
        
        # Extract code from log file
        log_path = "results_early_stop/logs/2072aba6_initial_candidate_1.txt"
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Extract code from FULL CODE section
        code_start = content.find("=== FULL CODE ===") + len("=== FULL CODE ===")
        code_end = content.find("=== RAW RESPONSE ===")
        code = content[code_start:code_end].strip()
        
        # Extract description
        desc_start = content.find("# description:") + len("# description:")
        desc_end = content.find("def main(input_grid):")
        desc_lines = content[desc_start:desc_end].strip().split('\n')
        description = ' '.join(line.strip('# ').strip() for line in desc_lines if line.strip())
        
        return problem_id, code, description
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from GLM's response"""
        # Try to find code block
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # If no code block, try to find code starting with 'from common import'
        if 'from common import' in response:
            start = response.find('from common import')
            # Find the end of the main function
            end = response.find('\n\n', start)
            if end == -1:
                end = len(response)
            return response[start:end].strip()
        
        return ""
    
    def run_experiment4_code_correction(self):
        """Experiment 4: Test GLM's ability to correct code"""
        print("\n=== Experiment 4: Code Correction ===")
        
        problem_id, original_code, description = self.get_test_problem()
        
        # Load problem
        problem = self.data_loader.get_problem_by_id(problem_id)
        
        # Execute original code
        print(f"Executing original code for {problem_id}...")
        original_result = self.code_executor.execute(original_code, problem)
        print(f"Original code - Success: {original_result.success}, Accuracy: {original_result.accuracy}")
        
        # Create comparison image
        os.makedirs("glm_experiments/exp4_code_correction", exist_ok=True)
        image_path = f"glm_experiments/exp4_code_correction/{problem_id}_comparison.png"
        self.renderer.render_problem_with_output(problem, original_result.output_grids, image_path)
        print(f"Created comparison image: {image_path}")
        
        # Prompt for code correction
        correction_prompt = f"""You are looking at an ARC problem where the code failed to produce correct outputs.

Current code description: "{description}"

Here is the failing code:
```python
{original_code}
```

Looking at the image:
- Column 1: Input grids (3x3)
- Column 2: Expected output grids (6x6) - this is the correct answer
- Column 3: Generated output grids - this is what the current code produces

The code is supposed to:
1. Take a 3x3 input grid with gray pixels
2. Repeat the pattern to fill a 6x6 grid
3. Change gray pixels to blue and red in a checkerboard pattern

Please analyze why the current code fails and provide a corrected version that will produce the expected outputs.

Important requirements:
- The corrected code must use the same imports and function signature
- The main function should take input_grid and return output_grid
- Use the Color constants (Color.BLACK, Color.GRAY, Color.BLUE, Color.RED)
- Make sure the 6x6 output matches the expected output exactly

Provide the complete corrected code."""

        print("\nAsking GLM to correct the code...")
        response = self._run_glm_inference(image_path, correction_prompt)
        
        # Extract corrected code from response
        corrected_code = self.extract_code_from_response(response)
        
        if not corrected_code:
            print("Failed to extract code from GLM's response")
            corrected_result = None
        else:
            print("\nExecuting GLM's corrected code...")
            try:
                corrected_result = self.code_executor.execute(corrected_code, problem)
                print(f"Corrected code - Success: {corrected_result.success}, Accuracy: {corrected_result.accuracy}")
                
                # Create comparison image for corrected code
                corrected_image_path = f"glm_experiments/exp4_code_correction/{problem_id}_corrected_comparison.png"
                self.renderer.render_problem_with_output(problem, corrected_result.output_grids, corrected_image_path)
                print(f"Created corrected comparison image: {corrected_image_path}")
                
            except Exception as e:
                print(f"Error executing corrected code: {e}")
                corrected_result = None
        
        # Save results
        results = {
            "problem_id": problem_id,
            "original_description": description,
            "original_code": original_code,
            "original_accuracy": original_result.accuracy,
            "glm_response": response,
            "corrected_code": corrected_code,
            "corrected_accuracy": corrected_result.accuracy if corrected_result else None,
            "corrected_success": corrected_result.success if corrected_result else False,
            "improvement": (corrected_result.accuracy - original_result.accuracy) if corrected_result else None
        }
        
        output_path = f"glm_experiments/exp4_code_correction/{problem_id}_results.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"\nResults saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EXPERIMENT 4 SUMMARY")
        print("="*60)
        print(f"Original accuracy: {original_result.accuracy * 100:.1f}%")
        if corrected_result:
            print(f"Corrected accuracy: {corrected_result.accuracy * 100:.1f}%")
            print(f"Improvement: {(corrected_result.accuracy - original_result.accuracy) * 100:.1f}%")
            if corrected_result.accuracy == 1.0:
                print("✅ GLM successfully corrected the code to achieve 100% accuracy!")
            elif corrected_result.accuracy > original_result.accuracy:
                print("✅ GLM improved the code accuracy!")
            else:
                print("❌ GLM's correction did not improve accuracy")
        else:
            print("❌ Failed to execute GLM's corrected code")
    
    def _run_glm_inference(self, image_path: str, prompt: str) -> str:
        """Run GLM inference on single image"""
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image", "image": image_path}
            ]
        }]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True
        ).to(self.model.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=2048,  # Increased for code generation
            temperature=0.1,
            do_sample=True
        )
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
            
        return response


if __name__ == "__main__":
    runner = GLMCodeCorrectionTester()
    runner.run_experiment4_code_correction()