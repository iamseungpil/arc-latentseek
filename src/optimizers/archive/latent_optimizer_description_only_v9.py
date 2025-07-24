"""
Description-Only Latent Optimizer V9
Optimizes only the description section using policy gradient
Similar to original LatentSeek but targets description specifically
"""

import torch
import torch.nn.functional as F
import numpy as np
import re
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging

from ..data import ARCProblem
from ..generators.barc_generator_fixed import BARCGeneratorFixed, BARCOutput
from ..executors import CodeExecutor
from ..evaluators.multitensor_evaluator import MultiTensorEvaluator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    final_output: BARCOutput
    loss_history: List[float]
    accuracy_history: List[float]
    dimension_losses: Dict[str, List[float]]
    optimization_steps: int
    converged: bool
    description_history: List[str]  # Track description evolution
    

class DescriptionOnlyLatentOptimizerV9:
    """Optimize only description section using policy gradient"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.01,
                 max_steps: int = 30,
                 # 5D weights
                 accuracy_weight: float = 0.3,
                 color_weight: float = 0.2,
                 spatial_weight: float = 0.2,
                 pattern_weight: float = 0.15,
                 structure_weight: float = 0.15,
                 convergence_threshold: float = 0.01):
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = MultiTensorEvaluator()
        self.lr = lr
        self.max_steps = max_steps
        
        # 5D weights
        self.weights = {
            'accuracy': accuracy_weight,
            'color': color_weight,
            'spatial': spatial_weight,
            'pattern': pattern_weight,
            'structure': structure_weight
        }
        self.convergence_threshold = convergence_threshold
        
        # Direct access to model and tokenizer
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def _find_description_indices(self, tokens: List[int], token_strings: List[str]) -> Optional[Tuple[int, int]]:
        """
        Find token indices for description section only
        """
        # Find '# description:' start
        desc_start = None
        for i in range(len(token_strings) - 3):
            combined = ''.join(token_strings[i:i+3])
            if '# description:' in combined:
                # Move to after the colon and newline
                desc_start = i + 3
                while desc_start < len(token_strings) and token_strings[desc_start].strip() in ['', '#', ':']:
                    desc_start += 1
                break
        
        if desc_start is None:
            logger.warning("Could not find '# description:' in tokens")
            return None
        
        # Find where description ends (empty line or 'def main')
        desc_end = desc_start
        for i in range(desc_start, len(token_strings)):
            # Check for def main
            if i < len(token_strings) - 2:
                combined = ''.join(token_strings[i:i+2])
                if 'def main' in combined or '\ndef' in combined:
                    desc_end = i
                    break
            # Check for double newline (end of description)
            if i < len(token_strings) - 1:
                if token_strings[i] == '\n' and token_strings[i+1] == '\n':
                    desc_end = i
                    break
        
        if desc_end <= desc_start:
            desc_end = min(desc_start + 100, len(token_strings))  # Max 100 tokens for description
        
        return (desc_start, desc_end)
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_accuracy: float) -> OptimizationResult:
        """
        Optimize only the description section
        """
        logger.info(f"Starting Description-Only V9 optimization for problem {problem.uid}")
        logger.info(f"Initial accuracy: {initial_accuracy:.3f}")
        logger.info(f"Initial description: {initial_output.description[:200]}...")
        
        # Get hidden states from initial generation
        hidden_states_list = self.barc_generator.get_hidden_states(problem, initial_output)
        
        if not hidden_states_list:
            logger.warning("Failed to get hidden states")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False,
                description_history=[initial_output.description]
            )
        
        # Get tokens from the raw response
        response_tokens = self.tokenizer.encode(initial_output.raw_response, add_special_tokens=False)
        token_strings = [self.tokenizer.decode([t]) for t in response_tokens]
        
        # Find description indices
        desc_indices = self._find_description_indices(response_tokens, token_strings)
        if not desc_indices:
            logger.warning("Could not find description in response")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False,
                description_history=[initial_output.description]
            )
        
        start_idx, end_idx = desc_indices
        
        # Extract description text for logging
        desc_tokens = response_tokens[start_idx:end_idx]
        desc_text = self.tokenizer.decode(desc_tokens)
        logger.info(f"Description found at tokens [{start_idx}:{end_idx}] (length: {end_idx - start_idx})")
        logger.info(f"Description content: {desc_text}")
        
        # Get prompt info
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        # Extract hidden states for description only
        device = next(self.model.parameters()).device
        
        # Get the relevant hidden states for description
        desc_hidden_states = []
        for i in range(start_idx, end_idx):
            if i < len(hidden_states_list):
                h = hidden_states_list[i]
                if h.dim() == 2:
                    h = h.squeeze(0)
                h_new = h.clone().detach().to(device).requires_grad_(True)
                desc_hidden_states.append(h_new)
        
        if not desc_hidden_states:
            logger.warning("No hidden states found for description")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False,
                description_history=[initial_output.description]
            )
        
        desc_hidden_states = torch.stack(desc_hidden_states)
        desc_hidden_states = torch.nn.Parameter(desc_hidden_states)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([desc_hidden_states], lr=self.lr)
        
        # Get full tokens for generation
        full_text = prompt_text + initial_output.raw_response
        full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        
        # Tracking
        loss_history = []
        accuracy_history = [initial_accuracy]
        dimension_losses = {dim: [] for dim in self.weights.keys()}
        description_history = [initial_output.description]
        best_output = initial_output
        best_accuracy = initial_accuracy
        best_reward = self._calculate_reward(problem, initial_output)
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"\nOptimization step {step + 1}/{self.max_steps}")
            
            optimizer.zero_grad()
            
            # Generate tokens from optimized description hidden states
            with torch.no_grad():
                # Get logits for description tokens
                desc_logits = self.model.lm_head(desc_hidden_states)
                desc_tokens = torch.argmax(desc_logits, dim=-1)
                
                # Build input_ids up to the optimized description
                # prompt + original tokens before description + optimized description tokens
                optimized_input_ids = torch.cat([
                    full_tokens.input_ids[:, :prompt_length + start_idx],  # up to description
                    desc_tokens.unsqueeze(0)  # optimized description
                ], dim=1)
                
                # Generate the rest autoregressively from after description
                output = self.model.generate(
                    optimized_input_ids,
                    max_new_tokens=min(2048, 4096 - optimized_input_ids.shape[1]),
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
                
                # Decode full response
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Remove prompt part
                if prompt_text in generated_text:
                    generated_text = generated_text[len(prompt_text):]
                
                # Parse output
                from ..generators.code_parser import extract_code_elements, parse_code
                
                code_blocks = parse_code(generated_text)
                code = code_blocks[0] if code_blocks else ""
                
                # If no code found, try to extract it
                if not code:
                    # Find from imports to end
                    import_match = re.search(r'from common import \*', generated_text)
                    if import_match:
                        code = generated_text[import_match.start():]
                        # Clean up any trailing content
                        if '\n\n\n' in code:
                            code = code[:code.index('\n\n\n')]
                
                concepts, description, plan = extract_code_elements(generated_text)
                
                new_output = BARCOutput(
                    code=code,
                    concepts=concepts or initial_output.concepts,
                    description=description,
                    plan=plan,
                    raw_response=generated_text
                )
            
            # Log new description
            logger.info(f"New description: {description[:200]}...")
            description_history.append(description)
            
            # Calculate reward
            current_reward = self._calculate_reward(problem, new_output)
            
            if new_output.code:
                result = self.code_executor.execute(new_output.code, problem)
                accuracy = result.accuracy if result.success else 0.0
                
                logger.info(f"Code execution: {'Success' if result.success else 'Failed'}") 
                logger.info(f"New accuracy: {accuracy:.1%}")
                logger.info(f"New reward: {current_reward:.3f}")
                
                # Calculate 5D losses
                if result.success:
                    dim_errors = self._calculate_5d_reconstruction_error(problem, result, new_output)
                    for dim in self.weights.keys():
                        dimension_losses[dim].append(dim_errors[dim].item())
            else:
                accuracy = 0.0
                logger.warning("No code generated")
            
            # Policy gradient loss
            # Calculate log probabilities for the generated description tokens
            desc_logits = self.model.lm_head(desc_hidden_states)
            desc_probs = torch.softmax(desc_logits, dim=-1) + 1e-8
            desc_log_probs = torch.log(desc_probs[torch.arange(len(desc_tokens)), desc_tokens] + 1e-10)
            
            # Policy gradient loss (maximize reward)
            loss = -current_reward * desc_log_probs.sum()
            
            logger.info(f"Loss: {loss.item():.4f}")
            
            # Backprop
            loss.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([desc_hidden_states], max_norm=1.0)
            
            optimizer.step()
            
            # Track progress
            loss_history.append(loss.item())
            
            accuracy_history.append(accuracy)
            
            if accuracy > best_accuracy or (accuracy == best_accuracy and current_reward > best_reward):
                best_accuracy = accuracy
                best_reward = current_reward
                best_output = new_output
                logger.info(f"🎯 New best accuracy: {best_accuracy:.1%}, reward: {best_reward:.3f}")
            
            # Early stopping
            if accuracy >= 1.0:
                logger.info("🏆 Perfect accuracy achieved!")
                return OptimizationResult(
                    final_output=new_output,
                    loss_history=loss_history,
                    accuracy_history=accuracy_history,
                    dimension_losses=dimension_losses,
                    optimization_steps=step + 1,
                    converged=True,
                    description_history=description_history
                )
            
            # Convergence check
            if len(loss_history) > 5:
                recent_losses = loss_history[-5:]
                loss_variance = np.var(recent_losses)
                if loss_variance < self.convergence_threshold:
                    logger.info(f"✅ Converged with loss variance {loss_variance:.6f}")
                    break
        
        return OptimizationResult(
            final_output=best_output,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
            dimension_losses=dimension_losses,
            optimization_steps=step + 1,
            converged=False,
            description_history=description_history
        )
    
    def _calculate_reward(self, problem: ARCProblem, output: BARCOutput) -> float:
        """Calculate reward for the output"""
        if not output.code:
            return 0.0
        
        result = self.code_executor.execute(output.code, problem)
        if not result.success:
            return 0.0
        
        # Use evaluator to get comprehensive reward
        evaluation = self.evaluator.evaluate(problem, output, result)
        return evaluation.total_reward
    
    def _calculate_5d_reconstruction_error(self,
                                         problem: ARCProblem,
                                         execution_result,
                                         output: BARCOutput) -> Dict[str, torch.Tensor]:
        """
        Calculate 5D CompressARC reconstruction error
        """
        device = next(self.model.parameters()).device
        errors = {}
        
        if not execution_result.success or not execution_result.output_grids:
            return {dim: torch.tensor(10.0, device=device) for dim in self.weights.keys()}
        
        # Initialize error tensors
        accuracy_errors = []
        color_errors = []
        spatial_errors = []
        pattern_errors = []
        structure_errors = []
        
        for i, (pair, output_grid) in enumerate(zip(problem.train_pairs, execution_result.output_grids)):
            if i >= len(execution_result.output_grids) or not isinstance(output_grid, np.ndarray):
                # Invalid output
                accuracy_errors.append(torch.tensor(1.0, device=device))
                color_errors.append(torch.tensor(1.0, device=device))
                spatial_errors.append(torch.tensor(1.0, device=device))
                pattern_errors.append(torch.tensor(1.0, device=device))
                structure_errors.append(torch.tensor(1.0, device=device))
                continue
            
            expected = pair.y
            
            # 1. Example Accuracy
            if output_grid.shape == expected.shape:
                matches = (output_grid == expected).astype(float)
                accuracy = matches.mean()
                accuracy_errors.append(torch.tensor(1.0 - accuracy, device=device))
            else:
                accuracy_errors.append(torch.tensor(1.0, device=device))
            
            # 2. Color Transformation
            color_error = self._calculate_color_error(pair.x, expected, output_grid)
            color_errors.append(torch.tensor(color_error, device=device))
            
            # 3. Spatial Transformation
            spatial_error = self._calculate_spatial_error(expected, output_grid)
            spatial_errors.append(torch.tensor(spatial_error, device=device))
            
            # 4. Pattern Recognition
            pattern_error = self._calculate_pattern_error(expected, output_grid)
            pattern_errors.append(torch.tensor(pattern_error, device=device))
            
            # 5. Structural Integrity
            structure_error = self._calculate_structure_error(expected, output_grid)
            structure_errors.append(torch.tensor(structure_error, device=device))
        
        # Average errors
        errors['accuracy'] = torch.stack(accuracy_errors).mean()
        errors['color'] = torch.stack(color_errors).mean()
        errors['spatial'] = torch.stack(spatial_errors).mean()
        errors['pattern'] = torch.stack(pattern_errors).mean()
        errors['structure'] = torch.stack(structure_errors).mean()
        
        return errors
    
    def _calculate_color_error(self, input_grid, expected, output):
        """Calculate color transformation error"""
        if output.shape != expected.shape:
            return 1.0
        
        # Simple color accuracy
        color_matches = (output == expected).mean()
        return 1.0 - color_matches
    
    def _calculate_spatial_error(self, expected, output):
        """Calculate spatial transformation error"""
        if output.shape != expected.shape:
            return 1.0
        
        # Check if non-zero pixels are in similar positions
        expected_nonzero = expected > 0
        output_nonzero = output > 0
        
        spatial_overlap = (expected_nonzero == output_nonzero).mean()
        return 1.0 - spatial_overlap
    
    def _calculate_pattern_error(self, expected, output):
        """Calculate pattern recognition error"""
        if output.shape != expected.shape:
            return 1.0
        
        # Simple pattern check - unique color count
        expected_colors = len(np.unique(expected))
        output_colors = len(np.unique(output))
        
        if expected_colors > 0:
            color_diff = abs(expected_colors - output_colors) / expected_colors
            return min(color_diff, 1.0)
        return 0.0
    
    def _calculate_structure_error(self, expected, output):
        """Calculate structural integrity error"""
        error = 0.0
        
        # Shape check
        if output.shape != expected.shape:
            error += 0.5
        
        # Valid color range
        if output.min() < 0 or output.max() > 9:
            error += 0.5
        
        return min(error, 1.0)