"""
Description-Targeted Optimizer V8
Optimizes only the description part of the generated code
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import logging
import re

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
    description_history: List[str]  # Track description changes


class DescriptionTargetedOptimizerV8:
    """Optimizer that targets only the description part of generated code"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.01,
                 max_steps: int = 50,
                 # 5D weights
                 accuracy_weight: float = 0.3,
                 color_weight: float = 0.2,
                 spatial_weight: float = 0.2,
                 pattern_weight: float = 0.15,
                 structure_weight: float = 0.15,
                 kl_weight: float = 0.1,
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
        self.kl_weight = kl_weight
        self.convergence_threshold = convergence_threshold
        
        # Direct access to model and tokenizer
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def _find_description_indices(self, raw_response: str) -> Optional[Tuple[int, int]]:
        """
        Find the token indices for the description part
        Returns (start_idx, end_idx) or None if not found
        """
        # Tokenize the full response
        tokens = self.tokenizer.encode(raw_response, add_special_tokens=False)
        
        # Convert tokens back to strings for pattern matching
        token_strings = [self.tokenizer.decode([t]) for t in tokens]
        
        # Find "# description:" pattern
        desc_start = None
        for i in range(len(token_strings) - 2):
            # Check for "# description:" pattern
            if (token_strings[i].strip() == '#' and 
                'description' in token_strings[i+1].lower()):
                # Start after the colon
                for j in range(i+1, min(i+5, len(token_strings))):
                    if ':' in token_strings[j]:
                        desc_start = j + 1
                        break
                break
        
        if desc_start is None:
            logger.warning("Could not find '# description:' pattern")
            return None
        
        # Find end of description (empty line or "def")
        desc_end = len(tokens)  # Default to end
        for i in range(desc_start, len(token_strings)):
            # Check for double newline or def
            if i < len(token_strings) - 1:
                combined = token_strings[i] + token_strings[i+1]
                if '\n\n' in combined or 'def ' in combined:
                    desc_end = i
                    break
        
        logger.info(f"Description found at tokens [{desc_start}:{desc_end}]")
        logger.info(f"Description text: {self.tokenizer.decode(tokens[desc_start:desc_end])}")
        
        return desc_start, desc_end
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_accuracy: float) -> OptimizationResult:
        """
        Optimize only the description part of the generated code
        """
        logger.info(f"Starting Description-Targeted V8 optimization for problem {problem.uid}")
        logger.info(f"Initial accuracy: {initial_accuracy:.3f}")
        
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
                description_history=[]
            )
        
        # Find description indices
        desc_indices = self._find_description_indices(initial_output.raw_response)
        if not desc_indices:
            logger.warning("Could not find description in response")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False,
                description_history=[]
            )
        
        desc_start, desc_end = desc_indices
        
        # Get prompt tokens for context
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        # Adjust indices for prompt offset
        desc_start += prompt_length
        desc_end += prompt_length
        
        # Validate indices
        if desc_end <= desc_start or desc_start >= len(hidden_states_list):
            logger.warning(f"Invalid description indices: [{desc_start}:{desc_end}]")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False,
                description_history=[]
            )
        
        logger.info(f"Optimizing description tokens [{desc_start}:{desc_end}] out of {len(hidden_states_list)} total")
        
        # Extract hidden states for description
        device = next(self.model.parameters()).device
        
        # Fixed parts
        prefix_hidden = hidden_states_list[:desc_start]
        suffix_hidden = hidden_states_list[desc_end:]
        
        # Optimizable description part
        desc_hidden_states = []
        for i in range(desc_start, min(desc_end, len(hidden_states_list))):
            h = hidden_states_list[i]
            if h.dim() == 2:
                h = h.squeeze(0)
            h_new = h.clone().detach().to(device).requires_grad_(True)
            desc_hidden_states.append(h_new)
        
        desc_hidden_states = torch.stack(desc_hidden_states)
        original_desc_hidden = desc_hidden_states.clone().detach()
        desc_hidden_states = torch.nn.Parameter(desc_hidden_states)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([desc_hidden_states], lr=self.lr)
        
        # Get initial tokens up to description
        full_text = prompt_text + initial_output.raw_response
        full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prefix_tokens = full_tokens.input_ids[0, :desc_start]
        
        # Tracking
        loss_history = []
        accuracy_history = [initial_accuracy]
        dimension_losses = {dim: [] for dim in self.weights.keys()}
        description_history = [initial_output.description]
        best_output = initial_output
        best_accuracy = initial_accuracy
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"\nOptimization step {step + 1}/{self.max_steps}")
            
            optimizer.zero_grad()
            
            # Get logits for description tokens
            logits = self.model.lm_head(desc_hidden_states)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Generate tokens from description hidden states
            with torch.no_grad():
                desc_tokens = torch.argmax(logits, dim=-1)
            
            # Compute log probability of generated sequence
            token_log_probs = log_probs.gather(-1, desc_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Generate full output with modified description
            new_output = self._generate_with_modified_description(
                problem, prefix_tokens, desc_tokens, desc_hidden_states
            )
            
            if new_output:
                logger.info(f"New description: {new_output.description[:100]}...")
                description_history.append(new_output.description)
                
                # Debug: log the generated tokens
                desc_text = self.tokenizer.decode(desc_tokens)
                logger.debug(f"Generated desc tokens: {desc_text[:100]}...")
            else:
                logger.warning("Failed to generate output")
            
            # Calculate reward
            if new_output and new_output.code:
                result = self.code_executor.execute(new_output.code, problem)
                accuracy = result.accuracy if result.success else 0.0
                
                logger.info(f"Code execution: {'Success' if result.success else 'Failed'}")
                logger.info(f"Accuracy: {accuracy:.1%}")
                
                # Debug: save generated code for first few steps
                if step < 3:
                    debug_code_path = f"results/description_v8/debug_step{step}_code.py"
                    with open(debug_code_path, 'w') as f:
                        f.write(f"# Description: {new_output.description}\n\n")
                        f.write(new_output.code)
                    logger.info(f"Debug code saved to: {debug_code_path}")
                
                # Calculate 5D losses
                with torch.no_grad():
                    dim_errors = self._calculate_5d_reconstruction_error(
                        problem, result, new_output
                    )
                    
                    reconstruction_error = sum(
                        self.weights[dim] * dim_errors[dim].item()
                        for dim in self.weights.keys()
                    )
                    
                    reward = 1.0 - min(reconstruction_error, 1.0)
                
                # Policy gradient loss with baseline to ensure gradient flow
                # Add small baseline reward to encourage exploration
                baseline_reward = 0.1
                effective_reward = reward + baseline_reward
                pg_loss = -effective_reward * token_log_probs.sum()
                
                # KL regularization
                kl_loss = F.mse_loss(desc_hidden_states, original_desc_hidden)
                
                # Total loss
                loss = pg_loss + self.kl_weight * kl_loss
                
                logger.info(f"Reward: {reward:.3f}, Loss: {loss.item():.4f} (PG: {pg_loss.item():.4f}, KL: {kl_loss.item():.4f})")
                logger.debug(f"Token log probs sum: {token_log_probs.sum().item():.4f}")
                
                # Log dimension losses
                for dim in self.weights.keys():
                    dimension_losses[dim].append(dim_errors[dim].item())
                
            else:
                # Invalid generation
                logger.warning("Invalid generation - no code produced")
                accuracy = 0.0
                reward = 0.0
                
                # Use baseline to encourage exploration
                baseline_reward = 0.1
                pg_loss = -baseline_reward * token_log_probs.sum()
                kl_loss = F.mse_loss(desc_hidden_states, original_desc_hidden)
                loss = pg_loss + self.kl_weight * kl_loss
                
                logger.info(f"Invalid gen - Loss: {loss.item():.4f} (PG: {pg_loss.item():.4f}, KL: {kl_loss.item():.4f})")
                
                for dim in self.weights.keys():
                    dimension_losses[dim].append(1.0)
            
            # Backprop
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([desc_hidden_states], max_norm=1.0)
            
            # Log gradient norm
            grad_norm = desc_hidden_states.grad.norm().item()
            logger.info(f"Gradient norm: {grad_norm:.4f}")
            
            optimizer.step()
            
            # Track progress
            loss_history.append(loss.item())
            accuracy_history.append(accuracy)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_output = new_output
                logger.info(f"🎯 New best accuracy: {best_accuracy:.1%}")
            
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
                if loss_variance < self.convergence_threshold and best_accuracy > 0:
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
    
    def _generate_with_modified_description(self,
                                          problem: ARCProblem,
                                          prefix_tokens: torch.Tensor,
                                          desc_tokens: torch.Tensor,
                                          desc_hidden_states: torch.Tensor) -> Optional[BARCOutput]:
        """
        Generate complete output with modified description
        """
        try:
            # Combine prefix tokens with new description tokens
            input_ids = torch.cat([
                prefix_tokens.unsqueeze(0),
                desc_tokens.unsqueeze(0)
            ], dim=1)
            
            # Generate the rest (code part)
            max_new_tokens = min(1500, 4096 - input_ids.shape[1])
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode full response
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Extract only the generated part (remove prompt)
            prompt = self.barc_generator._create_prompt(problem)
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            
            if generated_text.startswith(prompt_text):
                generated_text = generated_text[len(prompt_text):]
            
            # Parse output
            from ..generators.code_parser import extract_code_elements, parse_code
            
            code_blocks = parse_code(generated_text)
            code = code_blocks[0] if code_blocks else ""
            
            if not code:
                # Try to find main function
                main_match = re.search(r'def main.*?(?=\n(?:def|$))', generated_text, re.DOTALL)
                if main_match:
                    code = main_match.group(0)
            
            concepts, description, plan = extract_code_elements(generated_text)
            
            return BARCOutput(
                code=code,
                concepts=concepts,
                description=description,
                plan=plan,
                raw_response=generated_text
            )
            
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
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