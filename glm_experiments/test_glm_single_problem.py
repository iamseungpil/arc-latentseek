#!/usr/bin/env python3
"""
GLM Accuracy Test - Single Problem
Test GLM's evaluation accuracy with different approaches on one problem
"""

import os
import sys
import json
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import ARCDataLoader
from src.generators import BARCOutput
from src.executors import CodeExecutor, GridRenderer
from src.evaluators import GLMEvaluator
from transformers import AutoProcessor, Glm4vForConditionalGeneration

class GLMSingleProblemTester:
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
        # 2072aba6: Success=True, Accuracy=0.0
        problem_id = "2072aba6"
        
        # Extract code directly from log file
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
    
    def run_experiment1_image_understanding(self, problem_id: str, code: str, description: str):
        """Experiment 1: Test GLM's understanding of mismatched outputs"""
        print(f"\n=== Experiment 1: Image Understanding for {problem_id} ===")
        
        # Load problem
        problem = self.data_loader.get_problem_by_id(problem_id)
        
        # Execute code
        execution_result = self.code_executor.execute(code, problem)
        print(f"Code execution - Success: {execution_result.success}, Accuracy: {execution_result.accuracy}")
        
        # Create 3-column comparison image
        os.makedirs("glm_experiments/exp1_single", exist_ok=True)
        image_path = f"glm_experiments/exp1_single/{problem_id}_comparison.png"
        self.renderer.render_problem_with_output(problem, execution_result.output_grids, image_path)
        print(f"Created comparison image: {image_path}")
        
        # Test different prompts
        prompts = [
            # Prompt 1: Direct evaluation
            f"""This image shows an ARC problem with 3 columns:
- Column 1: Input grids
- Column 2: Expected output grids (correct answer)
- Column 3: Generated output grids (from code execution)

Code description: "{description}"

Question: Looking at the image, is the code correct or incorrect? Why do the outputs in column 3 differ from column 2?""",
            
            # Prompt 2: Detailed analysis
            f"""Analyze this ARC problem visualization:
- Column 1: Inputs
- Column 2: Expected outputs
- Column 3: Code-generated outputs

The code tried to: "{description}"

Please identify:
1. What pattern does the problem actually follow (based on columns 1->2)?
2. What pattern did the code implement (based on columns 1->3)?
3. How should the description be changed to solve this correctly?""",
            
            # Prompt 3: Step-by-step guidance
            f"""This shows an ARC puzzle where code failed to produce correct outputs.

Looking at each training example:
- Compare the expected output (column 2) with generated output (column 3)
- Identify the specific differences

Based on the pattern from input->expected, suggest a corrected description that would solve this problem."""
        ]
        
        results = []
        for i, prompt in enumerate(prompts):
            print(f"\nTesting prompt {i+1}...")
            response = self._run_glm_inference(image_path, prompt)
            results.append({
                "prompt_id": i+1,
                "prompt": prompt,
                "response": response
            })
            
        # Save results
        output_path = f"glm_experiments/exp1_single/{problem_id}_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                "problem_id": problem_id,
                "original_description": description,
                "execution_accuracy": execution_result.accuracy,
                "execution_success": execution_result.success,
                "code": code,
                "results": results
            }, f, indent=2)
            
        print(f"Results saved to {output_path}")
        
    def run_experiment2_row_separation(self, problem_id: str, code: str, description: str):
        """Experiment 2: Test with separate images for each training example"""
        print(f"\n=== Experiment 2: Row Separation for {problem_id} ===")
        
        # Load problem
        problem = self.data_loader.get_problem_by_id(problem_id)
        
        # Execute code
        execution_result = self.code_executor.execute(code, problem)
        
        # Create separate images for each training example
        os.makedirs("glm_experiments/exp2_single", exist_ok=True)
        example_images = []
        
        for i, (train_pair, output_grid) in enumerate(zip(problem.train_pairs, execution_result.output_grids)):
            image_path = f"glm_experiments/exp2_single/{problem_id}_example_{i}.png"
            
            # Create single row image for this example
            cell_size = 30
            padding = 10
            gap = 20
            
            h1, w1 = train_pair.x.shape
            h2, w2 = train_pair.y.shape
            
            if output_grid is not None and hasattr(output_grid, 'shape'):
                h3, w3 = output_grid.shape
            else:
                h3, w3 = h2, w2
            
            # Calculate dimensions
            grid_width = max(w1, w2, w3) * cell_size
            grid_height = max(h1, h2, h3) * cell_size
            total_width = 3 * grid_width + 2 * gap + 2 * padding
            total_height = grid_height + 50 + 2 * padding  # 50 for labels
            
            # Create image
            img = Image.new('RGB', (total_width, total_height), 'white')
            draw = ImageDraw.Draw(img)
            
            # Draw grids
            x_positions = [padding, padding + grid_width + gap, padding + 2 * (grid_width + gap)]
            labels = [f"Input {i+1}", f"Expected {i+1}", f"Generated {i+1}"]
            
            if output_grid is not None and hasattr(output_grid, 'shape'):
                output_display = output_grid
            else:
                # Create black grid if output is invalid
                output_display = np.zeros_like(train_pair.y)
            
            grids = [train_pair.x, train_pair.y, output_display]
            
            for idx, (x_pos, label, grid) in enumerate(zip(x_positions, labels, grids)):
                # Draw label
                draw.text((x_pos + grid_width//2 - 30, padding), label, fill='black')
                
                # Draw grid
                for y in range(grid.shape[0]):
                    for x in range(grid.shape[1]):
                        color_idx = int(grid[y, x])
                        if 0 <= color_idx < len(self.renderer.colors):
                            color = self.renderer.colors[color_idx]
                        else:
                            color = (0, 0, 0)  # Black for invalid
                        rect_x = x_pos + x * cell_size
                        rect_y = padding + 30 + y * cell_size
                        draw.rectangle([rect_x, rect_y, rect_x + cell_size - 1, rect_y + cell_size - 1], 
                                     fill=color, outline='gray')
            
            img.save(image_path)
            example_images.append(image_path)
            print(f"Created example image: {image_path}")
        
        # Test with different approaches
        approaches = [
            # Approach 1: All images at once
            {
                "name": "all_at_once",
                "prompt": f"""I'm showing you {len(example_images)} training examples from an ARC problem. 
Each image shows: Input | Expected Output | Generated Output

Code description: "{description}"

Looking at all examples together, why is the generated output wrong?""",
                "images": example_images
            },
            
            # Approach 2: Sequential analysis
            {
                "name": "sequential",
                "prompt": f"""Analyze these training examples one by one.
Code description: "{description}"

For each example, explain why the generated output differs from expected.""",
                "images": example_images
            }
        ]
        
        results = []
        for approach in approaches:
            print(f"\nTesting approach: {approach['name']}")
            
            if approach["name"] == "all_at_once":
                # Send all images together
                response = self._run_glm_inference_multiple(approach["images"], approach["prompt"])
            else:
                # Send images one by one
                responses = []
                for i, img in enumerate(approach["images"]):
                    example_prompt = f"{approach['prompt']}\n\nThis is example {i+1} of {len(approach['images'])}:"
                    resp = self._run_glm_inference(img, example_prompt)
                    responses.append(resp)
                response = "\n\n".join(responses)
                
            results.append({
                "approach": approach["name"],
                "response": response
            })
        
        # Save results
        output_path = f"glm_experiments/exp2_single/{problem_id}_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                "problem_id": problem_id,
                "original_description": description,
                "num_examples": len(example_images),
                "results": results
            }, f, indent=2)
            
        print(f"Results saved to {output_path}")
        
    def run_experiment3_multitensor(self, problem_id: str, code: str, description: str):
        """Experiment 3: Test multi-tensor evaluation approach"""
        print(f"\n=== Experiment 3: Multi-tensor Evaluation for {problem_id} ===")
        
        # Load problem
        problem = self.data_loader.get_problem_by_id(problem_id)
        
        # Execute code
        execution_result = self.code_executor.execute(code, problem)
        
        # Create comparison image
        os.makedirs("glm_experiments/exp3_single", exist_ok=True)
        image_path = f"glm_experiments/exp3_single/{problem_id}_comparison.png"
        self.renderer.render_problem_with_output(problem, execution_result.output_grids, image_path)
        print(f"Created comparison image: {image_path}")
        
        # Multi-tensor evaluation prompt
        multitensor_prompt = f"""Analyze this ARC problem using multi-dimensional evaluation:

Code description: "{description}"

Please evaluate along these dimensions:

1. **PER-EXAMPLE ACCURACY**: For each training example (row), rate:
   - Example 1: [0-100%] accuracy
   - Example 2: [0-100%] accuracy  
   - Example 3: [0-100%] accuracy

2. **COLOR TRANSFORMATION**: Analyze color handling:
   - Color mapping correctness: [0-100%]
   - Color preservation where needed: [0-100%]
   - New color generation accuracy: [0-100%]

3. **SPATIAL TRANSFORMATIONS**: Rate spatial operations:
   - X-axis transformations: [0-100%]
   - Y-axis transformations: [0-100%]
   - Rotation/reflection accuracy: [0-100%]
   - Size/scale handling: [0-100%]

4. **PATTERN RECOGNITION**: Evaluate pattern understanding:
   - Input pattern detection: [0-100%]
   - Rule application consistency: [0-100%]
   - Edge case handling: [0-100%]

5. **STRUCTURAL INTEGRITY**: Check output structure:
   - Grid dimensions correct: [YES/NO]
   - Object boundaries preserved: [0-100%]
   - Connectivity maintained: [0-100%]

Based on this multi-dimensional analysis, provide:
- Overall accuracy: [weighted average]
- Primary failure dimension: [which aspect failed most]
- Suggested fix: [specific improvement needed]"""
        
        # Run evaluation
        response = self._run_glm_inference(image_path, multitensor_prompt)
        
        # Also test with traditional prompt for comparison
        traditional_prompt = f"""This image shows an ARC problem. 
Code description: "{description}"

Is the code correct? Answer: TRUE or FALSE
Explanation: [brief reason]"""
        
        traditional_response = self._run_glm_inference(image_path, traditional_prompt)
        
        # Save results
        output_path = f"glm_experiments/exp3_single/{problem_id}_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                "problem_id": problem_id,
                "original_description": description,
                "execution_accuracy": execution_result.accuracy,
                "multitensor_evaluation": response,
                "traditional_evaluation": traditional_response
            }, f, indent=2)
            
        print(f"Results saved to {output_path}")
    
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
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True
        )
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
            
        return response
    
    def _run_glm_inference_multiple(self, image_paths: List[str], prompt: str) -> str:
        """Run GLM inference on multiple images"""
        content = [{"type": "text", "text": prompt}]
        for img_path in image_paths:
            content.append({"type": "image", "image": img_path})
            
        messages = [{
            "role": "user",
            "content": content
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
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True
        )
        
        response = self.processor.decode(output[0], skip_special_tokens=True)
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
            
        return response
    
    def run_all_experiments(self):
        """Run all experiments on single problem"""
        problem_id, code, description = self.get_test_problem()
        
        print(f"\n{'='*60}")
        print(f"Testing problem: {problem_id}")
        print(f"Description: {description[:100]}...")
        print(f"{'='*60}")
        
        # Run all three experiments
        self.run_experiment1_image_understanding(problem_id, code, description)
        self.run_experiment2_row_separation(problem_id, code, description)
        self.run_experiment3_multitensor(problem_id, code, description)
        
        print("\n" + "="*60)
        print("All experiments completed!")
        print("Results saved in glm_experiments/exp*_single/ subdirectories")


if __name__ == "__main__":
    runner = GLMSingleProblemTester()
    runner.run_all_experiments()