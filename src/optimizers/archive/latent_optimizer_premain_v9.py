"""
Pre-Main Content Optimizer V9
Optimizes everything before 'def main' (concepts + description)
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
    premain_history: List[str]  # Track pre-main content changes


class PreMainOptimizerV9:
    """Optimizer that targets everything before 'def main' (concepts + description)"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.005,  # Lower learning rate
                 max_steps: int = 30,
                 # 5D weights
                 accuracy_weight: float = 0.3,
                 color_weight: float = 0.2,
                 spatial_weight: float = 0.2,
                 pattern_weight: float = 0.15,
                 structure_weight: float = 0.15,
                 kl_weight: float = 0.01,  # Lower KL weight
                 convergence_threshold: float = 0.01,
                 update_ratio: float = 0.8):  # Update 80% of pre-main content
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = MultiTensorEvaluator()
        self.lr = lr
        self.max_steps = max_steps
        self.update_ratio = update_ratio
        
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
    
    def _find_premain_indices(self, raw_response: str) -> Optional[Tuple[int, int]]:
        """
        Find the token indices for everything before 'def main'
        Returns (start_idx, end_idx) or None if not found
        """
        # Simple approach: find '# concepts:' and 'def main' in the text
        concepts_match = re.search(r'# concepts:', raw_response)
        def_main_match = re.search(r'\ndef\s+main', raw_response)
        
        if not concepts_match or not def_main_match:
            logger.warning("Could not find '# concepts:' or 'def main' in response")
            return None
        
        # Get byte positions
        start_pos = concepts_match.start()
        end_pos = def_main_match.start()
        
        # Convert to token indices
        tokens = self.tokenizer.encode(raw_response, add_special_tokens=False)
        
        # Find corresponding token indices
        start_idx = None
        end_idx = None
        
        current_pos = 0
        for i, token_id in enumerate(tokens):
            token_text = self.tokenizer.decode([token_id])
            if start_idx is None and current_pos <= start_pos < current_pos + len(token_text.encode('utf-8')):
                start_idx = i
            if end_idx is None and current_pos <= end_pos < current_pos + len(token_text.encode('utf-8')):
                end_idx = i
                break
            current_pos += len(token_text.encode('utf-8'))
        
        if start_idx is None or end_idx is None or start_idx >= end_idx:
            logger.warning(f"Invalid token indices: start={start_idx}, end={end_idx}")
            return None
        
        pre_main_text = self.tokenizer.decode(tokens[start_idx:end_idx])
        logger.info(f"Pre-main content found at tokens [{start_idx}:{end_idx}] (length: {end_idx - start_idx})")
        logger.info(f"Pre-main content preview: {pre_main_text[:200]}...")
        
        return start_idx, end_idx
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_accuracy: float) -> OptimizationResult:
        """
        Optimize the pre-main content (concepts + description)
        """
        logger.info(f"Starting Pre-Main V9 optimization for problem {problem.uid}")
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
                premain_history=[]
            )
        
        # Find pre-main indices
        premain_indices = self._find_premain_indices(initial_output.raw_response)
        if not premain_indices:
            logger.warning("Could not find pre-main content in response")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False,
                premain_history=[]
            )
        
        start_idx, end_idx = premain_indices
        
        # Get prompt tokens for context
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        # Adjust indices for prompt offset
        start_idx += prompt_length
        end_idx += prompt_length
        
        # Validate indices
        if end_idx <= start_idx or start_idx >= len(hidden_states_list):
            logger.warning(f"Invalid pre-main indices after offset: [{start_idx}:{end_idx}]")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False,
                premain_history=[]
            )
        
        # Calculate update length (similar to LatentSeek)
        premain_length = end_idx - start_idx
        update_length = min(int(self.update_ratio * premain_length), premain_length)
        
        logger.info(f"Optimizing {update_length} tokens from pre-main content [{start_idx}:{start_idx + update_length}] out of {len(hidden_states_list)} total")
        
        # Extract hidden states for optimization
        device = next(self.model.parameters()).device
        
        # Fixed parts
        prefix_hidden = hidden_states_list[:start_idx]
        suffix_hidden = hidden_states_list[start_idx + update_length:]
        
        # Optimizable pre-main part
        opt_hidden_states = []
        for i in range(start_idx, min(start_idx + update_length, len(hidden_states_list))):
            h = hidden_states_list[i]
            if h.dim() == 2:
                h = h.squeeze(0)
            h_new = h.clone().detach().to(device).requires_grad_(True)
            opt_hidden_states.append(h_new)
        
        opt_hidden_states = torch.stack(opt_hidden_states)
        original_hidden = opt_hidden_states.clone().detach()
        opt_hidden_states = torch.nn.Parameter(opt_hidden_states)
        
        # Setup optimizer
        optimizer = torch.optim.Adam([opt_hidden_states], lr=self.lr)
        
        # Get initial tokens up to optimization point
        full_text = prompt_text + initial_output.raw_response
        full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prefix_tokens = full_tokens.input_ids[0, :start_idx]
        
        # Tracking
        loss_history = []
        accuracy_history = [initial_accuracy]
        dimension_losses = {dim: [] for dim in self.weights.keys()}
        premain_history = []
        best_output = initial_output
        best_accuracy = initial_accuracy
        
        # Extract initial pre-main content
        initial_premain = self._extract_premain_content(initial_output.raw_response)
        premain_history.append(initial_premain)
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"\nOptimization step {step + 1}/{self.max_steps}")
            
            optimizer.zero_grad()
            
            # Get logits for optimized tokens
            logits = self.model.lm_head(opt_hidden_states)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Generate tokens from hidden states
            with torch.no_grad():
                opt_tokens = torch.argmax(logits, dim=-1)
            
            # Compute log probability of generated sequence
            token_log_probs = log_probs.gather(-1, opt_tokens.unsqueeze(-1)).squeeze(-1)
            
            # Generate full output with modified pre-main content
            new_output = self._generate_with_modified_premain(
                problem, prefix_tokens, opt_tokens, opt_hidden_states
            )
            
            if new_output:
                new_premain = self._extract_premain_content(new_output.raw_response)
                logger.info(f"New pre-main content preview: {new_premain[:150]}...")
                premain_history.append(new_premain)
            else:
                logger.warning("Failed to generate output")
            
            # Calculate reward
            if new_output and new_output.code:
                result = self.code_executor.execute(new_output.code, problem)
                accuracy = result.accuracy if result.success else 0.0
                
                logger.info(f"Code execution: {'Success' if result.success else 'Failed'}")
                logger.info(f"Accuracy: {accuracy:.1%}")
                
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
                
                # Policy gradient loss (no baseline)
                pg_loss = -reward * token_log_probs.sum()
                
                # KL regularization
                kl_loss = F.mse_loss(opt_hidden_states, original_hidden)
                
                # Total loss
                loss = pg_loss + self.kl_weight * kl_loss
                
                logger.info(f"Reward: {reward:.3f}, Loss: {loss.item():.4f} (PG: {pg_loss.item():.4f}, KL: {kl_loss.item():.4f})")
                
                # Log dimension losses
                for dim in self.weights.keys():
                    dimension_losses[dim].append(dim_errors[dim].item())
                
            else:
                # Invalid generation - skip this step
                logger.warning("Invalid generation - skipping optimization step")
                continue
            
            # Only backprop if we have valid reward
            if reward > 0:
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([opt_hidden_states], max_norm=0.5)
                
                # Log gradient norm
                if opt_hidden_states.grad is not None:
                    grad_norm = opt_hidden_states.grad.norm().item()
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
                    premain_history=premain_history
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
            premain_history=premain_history
        )
    
    def _extract_premain_content(self, raw_response: str) -> str:
        """Extract everything before 'def main'"""
        # Find from '# concepts:' to before 'def main'
        concepts_match = re.search(r'(# concepts:.*?)(?=\ndef\s+main)', raw_response, re.DOTALL)
        if concepts_match:
            return concepts_match.group(1).strip()
        return ""
    
    def _generate_with_modified_premain(self,
                                       problem: ARCProblem,
                                       prefix_tokens: torch.Tensor,
                                       opt_tokens: torch.Tensor,
                                       opt_hidden_states: torch.Tensor) -> Optional[BARCOutput]:
        """
        Generate complete output with modified pre-main content
        """
        try:
            # Combine prefix tokens with optimized tokens
            input_ids = torch.cat([
                prefix_tokens.unsqueeze(0),
                opt_tokens.unsqueeze(0)
            ], dim=1)
            
            # Generate the rest
            max_new_tokens = min(2000, 4096 - input_ids.shape[1])
            
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
                # Try to extract main function
                main_match = re.search(r'def main.*?(?=\n(?:def|$))', generated_text, re.DOTALL)
                if main_match:
                    # Include imports and everything
                    import_match = re.search(r'^(from common import.*?)$', generated_text, re.MULTILINE)
                    if import_match:
                        code = generated_text[:main_match.end()]
            
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