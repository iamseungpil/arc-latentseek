"""
MultiTensor V5 - True CompressARC-style multi-dimensional reconstruction
Implements position-aware and dimension-wise loss calculation
"""

import torch
import torch.nn.functional as F
import numpy as np
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
    dimension_losses: Dict[str, List[float]]  # Track loss per dimension
    optimization_steps: int
    converged: bool


class MultiTensorOptimizerV5:
    """MultiTensor optimizer with true CompressARC-style multi-dimensional loss"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.01,
                 max_steps: int = 50,
                 k: float = 0.2,
                 kl_weight: float = 0.1,
                 position_weight: float = 3.0,
                 color_weight: float = 5.0,
                 size_weight: float = 2.0,
                 convergence_threshold: float = 0.01):
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = MultiTensorEvaluator()
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
        self.kl_weight = kl_weight
        self.position_weight = position_weight
        self.color_weight = color_weight
        self.size_weight = size_weight
        self.convergence_threshold = convergence_threshold
        
        # Direct access to model and tokenizer
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_accuracy: float) -> OptimizationResult:
        """
        Optimize using true multi-dimensional reconstruction error
        """
        logger.info(f"Starting MultiTensor V5 optimization for problem {problem.uid}")
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
                converged=False
            )
        
        # Get prompt length
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        # Calculate update window
        generated_length = len(hidden_states_list) - prompt_length
        if generated_length <= 0:
            logger.warning(f"Generated length too short: {generated_length}")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False
            )
        
        update_length = min(int(self.k * generated_length), 300)
        start_index = prompt_length
        
        if update_length <= 0:
            logger.warning("Update length is zero!")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
                dimension_losses={},
                optimization_steps=0,
                converged=False
            )
        
        logger.info(f"Optimizing {update_length} tokens from position {start_index}")
        logger.info(f"Total: {len(hidden_states_list)}, Prompt: {prompt_length}, Generated: {generated_length}")
        
        # Extract hidden states to optimize
        device = next(self.model.parameters()).device
        optimized_hidden_states = []
        for i in range(start_index, min(start_index + update_length, len(hidden_states_list))):
            h = hidden_states_list[i]
            if h.dim() == 2:  # [batch_size, hidden_dim]
                h = h.squeeze(0)  # Remove batch dimension
            optimized_hidden_states.append(h.clone().detach().to(device).requires_grad_(True))
        optimized_hidden_states = torch.nn.Parameter(torch.stack(optimized_hidden_states))
        
        # Store original for KL regularization
        original_hidden_states = optimized_hidden_states.clone().detach()
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Get initial sequence
        full_text = prompt_text + initial_output.raw_response
        full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        initial_input_ids = full_tokens.input_ids
        
        base_input_ids = prompt_tokens.input_ids
        original_seq = initial_input_ids[0][prompt_length:start_index].tolist()
        
        # Tracking
        loss_history = []
        accuracy_history = [initial_accuracy]
        dimension_losses = {
            'position': [],
            'color': [],
            'size': [],
            'kl': []
        }
        best_output = initial_output
        best_accuracy = initial_accuracy
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
            optimizer.zero_grad()
            
            # Generate from optimized hidden states
            new_output = self._generate_from_optimized(
                problem, optimized_hidden_states, original_seq, base_input_ids
            )
            
            if new_output and new_output.code:
                # Execute code
                result = self.code_executor.execute(new_output.code, problem)
                accuracy = result.accuracy if result.success else 0.0
                
                # Calculate multi-dimensional reconstruction error
                dim_errors = self._calculate_multidimensional_error(
                    problem, result, new_output
                )
                
                # KL divergence
                kl_loss = F.mse_loss(optimized_hidden_states, original_hidden_states)
                
                # Total loss with dimension-specific weights
                loss = (self.kl_weight * kl_loss + 
                       self.position_weight * dim_errors['position'] +
                       self.color_weight * dim_errors['color'] +
                       self.size_weight * dim_errors['size'])
                
                # Log details
                logger.info(f"Total Loss: {loss.item():.4f}")
                logger.info(f"  KL: {kl_loss.item():.4f}")
                logger.info(f"  Position: {dim_errors['position'].item():.4f}")
                logger.info(f"  Color: {dim_errors['color'].item():.4f}")
                logger.info(f"  Size: {dim_errors['size'].item():.4f}")
                logger.info(f"  Accuracy: {accuracy:.1%}")
                logger.info(f"  Description: {new_output.description}")
                
                # Backprop
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([optimized_hidden_states], max_norm=1.0)
                
                optimizer.step()
                
                # Track progress
                loss_history.append(loss.item())
                accuracy_history.append(accuracy)
                dimension_losses['kl'].append(kl_loss.item())
                dimension_losses['position'].append(dim_errors['position'].item())
                dimension_losses['color'].append(dim_errors['color'].item())
                dimension_losses['size'].append(dim_errors['size'].item())
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_output = new_output
                    logger.info(f"🎯 Updated best output with accuracy {best_accuracy:.1%}")
                
                # Early stopping
                if accuracy >= 1.0:
                    logger.info("🏆 Perfect accuracy achieved!")
                    return OptimizationResult(
                        final_output=new_output,
                        loss_history=loss_history,
                        accuracy_history=accuracy_history,
                        dimension_losses=dimension_losses,
                        optimization_steps=step + 1,
                        converged=True
                    )
                
                # Convergence check
                if len(loss_history) > 5:
                    recent_losses = loss_history[-5:]
                    loss_variance = np.var(recent_losses)
                    if loss_variance < self.convergence_threshold:
                        logger.info(f"✅ Converged with loss variance {loss_variance:.6f}")
                        return OptimizationResult(
                            final_output=best_output,
                            loss_history=loss_history,
                            accuracy_history=accuracy_history,
                            dimension_losses=dimension_losses,
                            optimization_steps=step + 1,
                            converged=True
                        )
            else:
                # High penalty for invalid generation
                loss = torch.tensor(100.0, device=device, requires_grad=True)
                loss.backward()
                optimizer.step()
                logger.warning("Failed to generate valid output")
                loss_history.append(100.0)
                accuracy_history.append(0.0)
                for key in dimension_losses:
                    dimension_losses[key].append(0.0)
        
        return OptimizationResult(
            final_output=best_output,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
            dimension_losses=dimension_losses,
            optimization_steps=self.max_steps,
            converged=False
        )
    
    def _calculate_multidimensional_error(self,
                                        problem: ARCProblem,
                                        execution_result,
                                        output: BARCOutput) -> Dict[str, torch.Tensor]:
        """
        Calculate CompressARC-style multi-dimensional reconstruction error
        """
        device = next(self.model.parameters()).device
        errors = {
            'position': torch.tensor(0.0, device=device, requires_grad=True),
            'color': torch.tensor(0.0, device=device, requires_grad=True),
            'size': torch.tensor(0.0, device=device, requires_grad=True)
        }
        
        if not execution_result.success or not execution_result.output_grids:
            # Maximum error for failed execution
            return {k: torch.tensor(10.0, device=device, requires_grad=True) for k in errors}
        
        total_examples = len(problem.train_pairs)
        
        for i, (pair, output_grid) in enumerate(zip(problem.train_pairs, execution_result.output_grids)):
            if i >= len(execution_result.output_grids):
                break
            
            expected = pair.y
            
            if not isinstance(output_grid, np.ndarray):
                # Invalid output
                for k in errors:
                    errors[k] = errors[k] + 5.0
                continue
            
            # 1. Size error (grid dimensions)
            expected_shape = expected.shape
            actual_shape = output_grid.shape
            size_error = abs(expected_shape[0] - actual_shape[0]) + abs(expected_shape[1] - actual_shape[1])
            errors['size'] = errors['size'] + size_error / (expected_shape[0] + expected_shape[1])
            
            if actual_shape != expected_shape:
                # Can't compute position/color errors if shapes don't match
                errors['position'] = errors['position'] + 1.0
                errors['color'] = errors['color'] + 1.0
                continue
            
            # 2. Position error (where objects are located)
            # Find non-black pixels and compare positions
            expected_positions = np.argwhere(expected > 0)
            actual_positions = np.argwhere(output_grid > 0)
            
            if len(expected_positions) > 0:
                # Calculate position mismatch
                position_error = 0.0
                for exp_pos in expected_positions:
                    # Find if this position has correct non-black pixel
                    if output_grid[exp_pos[0], exp_pos[1]] == 0:
                        position_error += 1.0
                position_error = position_error / len(expected_positions)
                errors['position'] = errors['position'] + position_error
            
            # 3. Color error (what colors are used)
            # Compare color distribution
            expected_colors = np.unique(expected)
            actual_colors = np.unique(output_grid)
            
            # Color accuracy at correct positions
            color_matches = 0
            total_pixels = expected.size
            for y in range(expected.shape[0]):
                for x in range(expected.shape[1]):
                    if expected[y, x] == output_grid[y, x]:
                        color_matches += 1
            
            color_error = 1.0 - (color_matches / total_pixels)
            errors['color'] = errors['color'] + color_error
        
        # Average over examples
        for k in errors:
            errors[k] = errors[k] / total_examples
        
        return errors
    
    def _mask_select_logprobs(self, mask: torch.Tensor, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CompressARC-style mask probability calculation
        Figure out the unnormalized log probability of taking each slice given the output mask.
        """
        logprobs = []
        for offset in range(mask.shape[0] - length + 1):
            logprob = -torch.sum(mask[:offset])
            logprob = logprob + torch.sum(mask[offset:offset+length])
            logprob = logprob - torch.sum(mask[offset+length:])
            logprobs.append(logprob)
        
        if not logprobs:
            # If no valid positions, return uniform
            return torch.tensor(0.0), torch.tensor([0.0])
        
        logprobs = torch.stack(logprobs, dim=0)
        log_partition = torch.logsumexp(logprobs, dim=0)
        return log_partition, logprobs
    
    def _generate_from_optimized(self,
                                problem: ARCProblem,
                                optimized_hidden: torch.Tensor,
                                original_seq: List[int],
                                base_input_ids: torch.Tensor) -> Optional[BARCOutput]:
        """
        Generate following original LatentSeek approach
        """
        try:
            # Start with base input_ids (prompt)
            input_ids = base_input_ids.clone()
            
            # Add original tokens before optimization window
            if original_seq:
                original_tokens = torch.tensor([original_seq], device=input_ids.device, dtype=torch.long)
                input_ids = torch.cat([input_ids, original_tokens], dim=-1)
            
            # Add optimized tokens
            with torch.no_grad():
                logits = self.model.lm_head(optimized_hidden)
                if len(logits.shape) == 3:
                    next_tokens = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    next_tokens = torch.argmax(logits, dim=-1)
                
                next_tokens = next_tokens.unsqueeze(0)
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Generate the rest autoregressively
            max_new_tokens = min(800, 4096 - input_ids.shape[1])
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Parse output
            from ..generators.code_parser import extract_code_elements, parse_code
            
            code_blocks = parse_code(generated_text)
            code = code_blocks[0] if code_blocks else ""
            
            if not code and ("def transform" in generated_text or "def main" in generated_text):
                for func_name in ["def transform", "def main"]:
                    if func_name in generated_text:
                        start = generated_text.find(func_name)
                        code = generated_text[start:]
                        break
            
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