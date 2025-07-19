"""
GLM-based evaluation system for ARC solutions
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import os

try:
    from transformers import AutoProcessor, Glm4vForConditionalGeneration
except ImportError:
    # Fallback for older transformers versions
    from transformers import AutoProcessor, AutoModelForVision2Seq as Glm4vForConditionalGeneration

from ..data import ARCProblem
from ..generators import BARCOutput
from ..executors import ExecutionResult, GridRenderer
from .verifiers import Verifier, VerificationResult


@dataclass
class EvaluationResult:
    """Result of GLM evaluation"""
    total_reward: float
    component_scores: Dict[str, float]
    verifications: Dict[str, VerificationResult]
    detailed_feedback: Dict[str, str]
    
    def __repr__(self):
        return f"EvaluationResult(reward={self.total_reward:.3f})"


class GLMEvaluator:
    """Use GLM-4.1V to evaluate ARC solutions"""
    
    def __init__(self, model_path: str = "THUDM/GLM-4.1V-9B-Thinking"):
        self.model_path = model_path
        self.processor = None
        self.model = None
        self.renderer = GridRenderer()
        self.verifier = Verifier()
        self._load_model()
        
    def _load_model(self):
        """Load GLM model and processor"""
        print(f"Loading GLM model: {self.model_path}")
        self.processor = AutoProcessor.from_pretrained(self.model_path, use_fast=True)
        self.model = Glm4vForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("GLM model loaded successfully")
        
    def evaluate(self, 
                problem: ARCProblem,
                barc_output: BARCOutput,
                execution_result: ExecutionResult,
                base_path: str = "temp_eval") -> EvaluationResult:
        """
        Evaluate a BARC solution using GLM with dual images
        
        Args:
            problem: ARC problem
            barc_output: Generated code and description from BARC
            execution_result: Results of executing the code
            base_path: Base path for saving images
            
        Returns:
            EvaluationResult with rewards and feedback
        """
        # Render training comparison (input, expected, actual) in single image
        comparison_path = f"{base_path}_comparison.png"
        self.renderer.render_training_comparison(problem, execution_result, comparison_path)
        
        # Prepare verifications
        verifications = {}
        
        # 1. Answer Correctness (strict binary: must solve ALL training examples)
        verifications['answer_correct'] = VerificationResult(
            passed=execution_result.accuracy == 1.0,  # Must be perfect
            confidence=1.0,
            feedback=f"Code achieved {execution_result.accuracy:.2%} accuracy on training examples (must be 100% for TRUE)"
        )
        
        # 2. Calculate other verifications using GLM with comparison image
        glm_verifications = self._run_glm_verifications_single_image(
            comparison_path,
            barc_output
        )
        verifications.update(glm_verifications)
        
        # Calculate component scores
        component_scores = self._calculate_scores(verifications)
        
        # Calculate total reward
        total_reward = self._calculate_total_reward(component_scores)
        
        # Prepare detailed feedback
        detailed_feedback = {
            name: result.feedback 
            for name, result in verifications.items()
        }
        
        return EvaluationResult(
            total_reward=total_reward,
            component_scores=component_scores,
            verifications=verifications,
            detailed_feedback=detailed_feedback
        )
    
    def _run_glm_verifications(self, 
                              image_path: str,
                              barc_output: BARCOutput,
                              problem: ARCProblem) -> Dict[str, VerificationResult]:
        """Run GLM-based verifications"""
        verifications = {}
        
        # Prepare base context
        context = f"""
This image shows an ARC problem with:
- Training examples showing input-output transformations
- Generated outputs from the following code solution

Code Description: {barc_output.description or "No description provided"}
Code Concepts: {barc_output.concepts or "No concepts listed"}
"""
        
        # 1. Calculation/Transformation Logic Check
        calc_prompt = f"""{context}

Please analyze whether the generated outputs correctly implement the transformation pattern shown in the expected outputs.
Focus on:
- Are the transformations consistent across all examples?
- Do the generated outputs follow the same logic as expected outputs?
- Are there any calculation or logic errors?

Respond with:
VERIFICATION: [TRUE/FALSE]
CONFIDENCE: [0.0-1.0]
FEEDBACK: [Brief explanation]"""
        
        calc_result = self._run_glm_inference(image_path, calc_prompt)
        verifications['calculation_check'] = self._parse_verification_response(calc_result)
        
        # 2. Answer Completeness Check
        complete_prompt = f"""{context}

Please check if the generated outputs are complete solutions:
- Do they provide definitive answers (not partial solutions)?
- Are all necessary transformations applied?
- Is the output in the correct format?

Respond with:
VERIFICATION: [TRUE/FALSE]
CONFIDENCE: [0.0-1.0]
FEEDBACK: [Brief explanation]"""
        
        complete_result = self._run_glm_inference(image_path, complete_prompt)
        verifications['answer_completeness'] = self._parse_verification_response(complete_result)
        
        # 3. Understanding/Description Match Check
        understanding_prompt = f"""{context}

Please verify if the code description matches what's actually happening in the generated outputs:
- Does the description accurately describe the transformation?
- Do the stated concepts align with the visual patterns?
- Is there consistency between description and implementation?

Respond with:
VERIFICATION: [TRUE/FALSE]
CONFIDENCE: [0.0-1.0]
FEEDBACK: [Brief explanation]"""
        
        understanding_result = self._run_glm_inference(image_path, understanding_prompt)
        verifications['understanding_check'] = self._parse_verification_response(understanding_result)
        
        return verifications
    
    def _run_glm_verifications_single_image(self, 
                                           comparison_path: str,
                                           barc_output: BARCOutput) -> Dict[str, VerificationResult]:
        """Run GLM-based verifications using single comparison image"""
        
        # Enhanced prompt emphasizing consistency across ALL training examples
        prompt = f"""This image shows training examples for an ARC problem with three columns:
- Column 1 (INPUT): The input grids
- Column 2 (EXPECTED): The correct expected outputs  
- Column 3 (ACTUAL): The outputs generated by our code

Code Description: "{barc_output.description or 'No description provided'}"
Code Concepts: "{barc_output.concepts or 'No concepts listed'}"

<think>
CRITICAL REQUIREMENT: The rule/description must work for ALL training examples, not just some of them.

Analyze each training example row by row:

1. UNDERSTANDING CHECK: Does the description accurately explain the transformation shown in ALL training examples?
   - Check if the description matches the pattern in EVERY row
   - The rule must be consistent across all input-expected pairs
   - Look for any contradictions or exceptions

2. CALCULATION CHECK: Does the generated output follow the correct transformation rule for ALL examples?
   - Compare ACTUAL vs EXPECTED for each training example
   - Check if the same transformation logic is applied consistently
   - Look for any calculation or logic errors in any example

3. ANSWER COMPLETENESS: Are ALL generated outputs complete and properly formatted?
   - Check if every output provides a definitive solution
   - Verify all necessary transformations are applied in each case
   - Ensure output format is correct for all examples

4. ANSWER CORRECT: Is the overall approach correct for solving this ARC problem?
   - The rule must work for ALL training examples to be considered correct
   - Even if most examples are correct, if ANY example fails, the rule is incorrect
   - Assess if this approach would generalize to new test cases
</think>

<answer>
Evaluate based on the requirement that the rule must work for ALL training examples:

UNDERSTANDING_CHECK: [TRUE/FALSE]
CALCULATION_CHECK: [TRUE/FALSE] 
ANSWER_COMPLETENESS: [TRUE/FALSE]
ANSWER_CORRECT: [TRUE/FALSE]

FEEDBACK: [Detailed explanation focusing on consistency across all training examples. Specifically mention which examples pass/fail and why.]
</answer>"""

        # Run GLM inference with single comparison image
        response = self._run_glm_inference(comparison_path, prompt)
        
        # Parse the structured response
        verifications = self._parse_glm_structured_response(response)
        
        return verifications
    
    def _run_glm_verifications_dual_images(self,
                                          arc_problem_path: str,
                                          solution_result_path: str,
                                          barc_output: BARCOutput) -> Dict[str, VerificationResult]:
        """Run GLM-based verifications using dual images in think/answer format"""
        
        # Create the unified GLM prompt with dual images
        prompt = f"""Look at these two images:
1. The first image shows the original ARC training examples
2. The second image shows our generated solution applied to the test input

Code Description: "{barc_output.description or 'No description provided'}"
Code Concepts: "{barc_output.concepts or 'No concepts listed'}"

<think>
Analyze by comparing the two images:

1. UNDERSTANDING CHECK: Does the description match the pattern shown in the training examples?
   - Look at the training examples in the first image
   - Check if the description accurately explains the transformation pattern
   - Verify if the concepts align with what you observe

2. CALCULATION CHECK: Does the generated output follow the same transformation rule as the training examples?
   - Compare the transformation pattern from training examples
   - Check if the generated output in the second image follows the same logic
   - Look for any calculation or logic errors

3. ANSWER COMPLETENESS: Is the generated output complete and properly formatted?
   - Check if the output provides a definitive solution (not partial)
   - Verify if all necessary transformations are applied
   - Ensure the output format is correct

4. ANSWER CORRECT: Does the transformation logic appear correct based on the pattern?
   - Assess overall correctness of the approach
   - Check consistency with training examples
</think>

<answer>
Provide your evaluation in the following format:

UNDERSTANDING_CHECK: [TRUE/FALSE]
CALCULATION_CHECK: [TRUE/FALSE] 
ANSWER_COMPLETENESS: [TRUE/FALSE]
ANSWER_CORRECT: [TRUE/FALSE]

FEEDBACK: [Brief explanation of your assessment]
</answer>"""

        # Run GLM inference with dual images
        response = self._run_glm_inference_dual_images(
            arc_problem_path, solution_result_path, prompt
        )
        
        # Parse the structured response
        verifications = self._parse_glm_structured_response(response)
        
        return verifications
    
    def _run_glm_inference_dual_images(self, image1_path: str, image2_path: str, prompt: str) -> str:
        """Run GLM inference on two images with prompt"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image1_path},
                    {"type": "image", "url": image2_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_k=10
        )
        
        response = self.processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _parse_glm_structured_response(self, response: str) -> Dict[str, VerificationResult]:
        """Parse GLM structured response with LatentSeek verifier names"""
        verifications = {}
        
        # Map GLM response keys to LatentSeek verifier names
        key_mapping = {
            'UNDERSTANDING_CHECK': 'understanding_check',
            'CALCULATION_CHECK': 'calculation_check',
            'ANSWER_COMPLETENESS': 'answer_completeness',
            'ANSWER_CORRECT': 'answer_correct'
        }
        
        # Extract feedback
        feedback_match = re.search(r'FEEDBACK:\s*(.+)', response, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else response
        
        # Parse each verification
        for glm_key, latentseek_key in key_mapping.items():
            pattern = f'{glm_key}:\\s*(TRUE|FALSE)'
            match = re.search(pattern, response, re.IGNORECASE)
            
            if match:
                passed = match.group(1).upper() == 'TRUE'
                verifications[latentseek_key] = VerificationResult(
                    passed=passed,
                    confidence=0.8,  # High confidence for structured response
                    feedback=f"{glm_key}: {feedback[:100]}..."
                )
            else:
                # Default to failed if not found
                verifications[latentseek_key] = VerificationResult(
                    passed=False,
                    confidence=0.3,
                    feedback=f"Could not parse {glm_key} from response"
                )
        
        return verifications
    
    def _run_glm_inference(self, image_path: str, prompt: str) -> str:
        """Run GLM inference on image with prompt"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": image_path},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        
        output = self.model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            do_sample=True,
            top_k=10
        )
        
        response = self.processor.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response
    
    def _parse_verification_response(self, response: str) -> VerificationResult:
        """Parse GLM verification response"""
        # Default values
        passed = False
        confidence = 0.5
        feedback = response
        
        # Try to parse structured response
        verification_match = re.search(r'VERIFICATION:\s*(TRUE|FALSE)', response, re.IGNORECASE)
        if verification_match:
            passed = verification_match.group(1).upper() == 'TRUE'
            
        confidence_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        if confidence_match:
            try:
                confidence = float(confidence_match.group(1))
                confidence = max(0.0, min(1.0, confidence))
            except:
                pass
                
        feedback_match = re.search(r'FEEDBACK:\s*(.+)', response, re.DOTALL)
        if feedback_match:
            feedback = feedback_match.group(1).strip()
            
        return VerificationResult(
            passed=passed,
            confidence=confidence,
            feedback=feedback
        )
    
    def _calculate_scores(self, verifications: Dict[str, VerificationResult]) -> Dict[str, float]:
        """Calculate component scores from verifications"""
        # Weight configuration (same as LatentSeek)
        weights = {
            'calculation_check': 2.0,
            'answer_correct': 1.0,
            'answer_completeness': 2.0,
            'understanding_check': 1.0
        }
        
        scores = {}
        for name, verification in verifications.items():
            weight = weights.get(name, 1.0)
            # Score is 0 if passed, negative weight if failed (same as LatentSeek)
            score = 0.0 if verification.passed else -weight
            # Apply confidence scaling
            scores[name] = score * verification.confidence
            
        return scores
    
    def _calculate_total_reward(self, component_scores: Dict[str, float]) -> float:
        """Calculate total reward from component scores"""
        # Sum of weights for normalization
        weights = {
            'calculation_check': 2.0,
            'answer_correct': 1.0,
            'answer_completeness': 2.0,
            'understanding_check': 1.0
        }
        
        total_weight = sum(weights.values())
        total_penalty = sum(score for score in component_scores.values() if score < 0)
        
        # Normalize to 0.0 to -1.0 range
        normalized_reward = total_penalty / total_weight if total_weight > 0 else 0.0
        
        return normalized_reward