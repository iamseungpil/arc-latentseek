"""
MultiTensor V7 - Full 5D CompressARC reconstruction error implementation
Implements all 5 dimensions of CompressARC evaluation as differentiable losses
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
    dimension_losses: Dict[str, List[float]]  # Track each dimension
    optimization_steps: int
    converged: bool


class MultiTensorOptimizerV7:
    """MultiTensor optimizer with full 5D CompressARC reconstruction error"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.01,
                 max_steps: int = 50,
                 k: float = 0.2,
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
        self.k = k
        
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
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_accuracy: float) -> OptimizationResult:
        """
        Optimize with 5D CompressARC reconstruction error
        """
        logger.info(f"Starting MultiTensor V7 (5D) optimization for problem {problem.uid}")
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
            # Create new tensor with gradient tracking
            # Since hidden states come from no_grad context, we need to create new tensors
            h_new = h.clone().detach().to(device).requires_grad_(True)
            optimized_hidden_states.append(h_new)
        
        # Stack into a single tensor
        optimized_hidden_states = torch.stack(optimized_hidden_states)
        
        # Store original for KL regularization
        original_hidden_states = optimized_hidden_states.clone().detach()
        
        # Make it a parameter for optimizer
        optimized_hidden_states = torch.nn.Parameter(optimized_hidden_states)
        
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
        dimension_losses = {dim: [] for dim in self.weights.keys()}
        best_output = initial_output
        best_accuracy = initial_accuracy
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
            optimizer.zero_grad()
            
            # Get logits and compute log probabilities for policy gradient
            logits = self.model.lm_head(optimized_hidden_states)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Debug shapes and values
            logger.debug(f"Optimized hidden states shape: {optimized_hidden_states.shape}")
            logger.debug(f"Logits shape: {logits.shape}, requires_grad: {logits.requires_grad}")
            logger.debug(f"Log probs shape: {log_probs.shape}, requires_grad: {log_probs.requires_grad}")
            
            # Generate tokens from optimized hidden states
            with torch.no_grad():
                if len(logits.shape) == 3:
                    generated_tokens = torch.argmax(logits, dim=-1).squeeze(-1)
                else:
                    generated_tokens = torch.argmax(logits, dim=-1)
            
            # Compute log probability of generated sequence
            token_log_probs = log_probs.gather(-1, generated_tokens.unsqueeze(-1)).squeeze(-1)
            logger.debug(f"Generated tokens shape: {generated_tokens.shape}")
            logger.debug(f"Token log probs: min={token_log_probs.min().item():.4f}, max={token_log_probs.max().item():.4f}, mean={token_log_probs.mean().item():.4f}")
            
            # Generate full output for evaluation
            new_output = self._generate_from_optimized(
                problem, optimized_hidden_states, original_seq, base_input_ids
            )
            
            # Debug: Log generated output
            if new_output:
                logger.info(f"Generated description: {new_output.description[:100]}...")
                logger.info(f"Generated code preview: {new_output.code[:200]}..." if new_output.code else "No code generated")
            else:
                logger.info("Generation failed - no output")
            
            # Calculate 5D reconstruction error as reward signal
            if new_output and new_output.code:
                result = self.code_executor.execute(new_output.code, problem)
                accuracy = result.accuracy if result.success else 0.0
                
                # Debug: Log execution result
                logger.info(f"Code execution: {'Success' if result.success else 'Failed'}")
                logger.info(f"Accuracy: {accuracy:.1%}")
                if result.output_grids:
                    logger.info(f"Output grids generated: {len(result.output_grids)}")
                    # Save visualization for debugging
                    if step == 0:  # Only save first step
                        from ..executors.grid_renderer import GridRenderer
                        renderer = GridRenderer()
                        debug_path = f"results/multitensor_v7/debug_{problem.uid}_step{step}.png"
                        renderer.render_problem_with_output(
                            problem, result.output_grids, debug_path
                        )
                        logger.info(f"Debug visualization saved to: {debug_path}")
                        # Also save the generated code
                        code_path = f"results/multitensor_v7/debug_{problem.uid}_step{step}_code.py"
                        with open(code_path, 'w') as f:
                            f.write(new_output.code)
                        logger.info(f"Debug code saved to: {code_path}")
                else:
                    logger.info("No output grids generated")
                
                # Calculate 5D losses (used as negative rewards)
                with torch.no_grad():
                    dim_errors = self._calculate_5d_reconstruction_error(
                        problem, result, new_output, optimized_hidden_states
                    )
                    
                    # Compute total reconstruction error
                    reconstruction_error = sum(
                        self.weights[dim] * dim_errors[dim].item()
                        for dim in self.weights.keys()
                    )
                    
                    # Convert error to reward (lower error = higher reward)
                    reward = 1.0 - min(reconstruction_error, 1.0)
                
                # Policy gradient loss: -reward * log_prob
                pg_loss = -reward * token_log_probs.sum()
                
                # KL regularization (differentiable)
                kl_loss = F.mse_loss(optimized_hidden_states, original_hidden_states)
                
                # Total loss
                loss = pg_loss + self.kl_weight * kl_loss
                
                # Debug reward and losses
                logger.debug(f"Reconstruction error: {reconstruction_error:.4f}")
                logger.debug(f"Reward: {reward:.4f}")
                logger.debug(f"PG component: reward={reward:.4f} * sum={token_log_probs.sum().item():.4f} = {(-reward * token_log_probs.sum()).item():.4f}")
                
                # Log dimension losses
                for dim in self.weights.keys():
                    dimension_losses[dim].append(dim_errors[dim].item())
                
            else:
                # Invalid generation: use penalty
                logger.debug("Invalid generation - applying penalty")
                accuracy = 0.0
                reward = 0.0
                
                # Small penalty through log_probs to maintain gradient
                pg_loss = 0.1 * token_log_probs.sum()  # Encourage different tokens
                kl_loss = F.mse_loss(optimized_hidden_states, original_hidden_states)
                loss = pg_loss + self.kl_weight * kl_loss
                
                logger.debug(f"Invalid gen - PG loss: {pg_loss.item():.4f}, KL loss: {kl_loss.item():.4f}")
                
                for dim in self.weights.keys():
                    dimension_losses[dim].append(1.0)
            
            # Log details
            logger.info(f"Loss: {loss.item():.4f} (PG: {pg_loss.item():.4f}, KL: {kl_loss.item():.4f})")
            logger.info(f"  Accuracy: {accuracy:.1%}")
            if new_output and new_output.code and result.success:
                logger.info(f"  Reward: {reward:.3f}")
                logger.info(f"  Dimension errors: {', '.join(f'{dim}={dim_errors[dim].item():.3f}' for dim in self.weights.keys())}")
            
            # Debug log probabilities
            logger.debug(f"  Token log probs sum: {token_log_probs.sum().item():.4f}")
            logger.debug(f"  Log probs shape: {log_probs.shape}, Token log probs shape: {token_log_probs.shape}")
            
            # Backprop
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], max_norm=1.0)
            
            # Log gradient norm
            grad_norm = optimized_hidden_states.grad.norm().item()
            logger.info(f"  Gradient norm: {grad_norm:.4f}")
            
            optimizer.step()
            
            # Track progress
            loss_history.append(loss.item())
            accuracy_history.append(accuracy)
            
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
                if loss_variance < self.convergence_threshold and best_accuracy > 0:
                    logger.info(f"✅ Converged with loss variance {loss_variance:.6f}")
                    return OptimizationResult(
                        final_output=best_output,
                        loss_history=loss_history,
                        accuracy_history=accuracy_history,
                        dimension_losses=dimension_losses,
                        optimization_steps=step + 1,
                        converged=True
                    )
        
        return OptimizationResult(
            final_output=best_output,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
            dimension_losses=dimension_losses,
            optimization_steps=self.max_steps,
            converged=False
        )
    
    def _calculate_5d_reconstruction_error(self,
                                         problem: ARCProblem,
                                         execution_result,
                                         output: BARCOutput,
                                         hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate 5D CompressARC reconstruction error:
        1. Example accuracy - pixel-wise accuracy
        2. Color transformation - color mapping accuracy
        3. Spatial transformation - position/rotation/scale accuracy
        4. Pattern recognition - pattern consistency
        5. Structural integrity - overall structure preservation
        """
        device = hidden_states.device
        errors = {}
        
        if not execution_result.success or not execution_result.output_grids:
            # Maximum error for failed execution
            return {dim: torch.tensor(10.0, device=device, requires_grad=True) for dim in self.weights.keys()}
        
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
            
            # 1. Example Accuracy (pixel-wise accuracy)
            if output_grid.shape == expected.shape:
                matches = (output_grid == expected).astype(float)
                accuracy = matches.mean()
                accuracy_errors.append(torch.tensor(1.0 - accuracy, device=device))
            else:
                accuracy_errors.append(torch.tensor(1.0, device=device))
            
            # 2. Color Transformation
            color_error = self._calculate_color_transformation_error(pair.x, expected, output_grid)
            color_errors.append(torch.tensor(color_error, device=device))
            
            # 3. Spatial Transformation
            spatial_error = self._calculate_spatial_transformation_error(pair.x, expected, output_grid)
            spatial_errors.append(torch.tensor(spatial_error, device=device))
            
            # 4. Pattern Recognition
            pattern_error = self._calculate_pattern_recognition_error(pair.x, expected, output_grid)
            pattern_errors.append(torch.tensor(pattern_error, device=device))
            
            # 5. Structural Integrity
            structure_error = self._calculate_structural_integrity_error(expected, output_grid)
            structure_errors.append(torch.tensor(structure_error, device=device))
        
        # Average errors across examples
        errors['accuracy'] = torch.stack(accuracy_errors).mean()
        errors['color'] = torch.stack(color_errors).mean()
        errors['spatial'] = torch.stack(spatial_errors).mean()
        errors['pattern'] = torch.stack(pattern_errors).mean()
        errors['structure'] = torch.stack(structure_errors).mean()
        
        # Note: These errors are used as rewards in policy gradient, not direct losses
        # So we don't need to make them differentiable
        
        return errors
    
    def _calculate_color_transformation_error(self, input_grid: np.ndarray, 
                                            expected: np.ndarray, 
                                            output: np.ndarray) -> float:
        """Calculate how well colors are transformed from input to output"""
        # Find color mappings in expected transformation
        input_colors = np.unique(input_grid)
        expected_colors = np.unique(expected)
        
        if output.shape != expected.shape:
            return 1.0
        
        # Calculate color mapping consistency
        error = 0.0
        for color in input_colors:
            input_mask = (input_grid == color)
            if input_mask.any():
                # Find what this color maps to in expected
                expected_mapped = expected[input_mask]
                if len(expected_mapped) > 0:
                    expected_color = np.bincount(expected_mapped.flatten()).argmax()
                    # Check if output maintains this mapping
                    output_mapped = output[input_mask]
                    if len(output_mapped) > 0:
                        matches = (output_mapped == expected_color).mean()
                        error += (1.0 - matches)
        
        return error / len(input_colors) if len(input_colors) > 0 else 1.0
    
    def _calculate_spatial_transformation_error(self, input_grid: np.ndarray,
                                              expected: np.ndarray,
                                              output: np.ndarray) -> float:
        """Calculate spatial transformation accuracy (position, rotation, scale)"""
        if output.shape != expected.shape:
            # Size mismatch is a spatial error
            size_error = abs(output.shape[0] - expected.shape[0]) + abs(output.shape[1] - expected.shape[1])
            return min(size_error / (expected.shape[0] + expected.shape[1]), 1.0)
        
        # Calculate center of mass difference
        error = 0.0
        for color in range(1, 10):  # Skip black (0)
            expected_mask = (expected == color)
            output_mask = (output == color)
            
            if expected_mask.any():
                # Expected center of mass
                y_exp, x_exp = np.where(expected_mask)
                if len(y_exp) > 0:
                    exp_center = (y_exp.mean(), x_exp.mean())
                    
                    if output_mask.any():
                        # Output center of mass
                        y_out, x_out = np.where(output_mask)
                        out_center = (y_out.mean(), x_out.mean())
                        
                        # Normalized distance
                        dist = np.sqrt((exp_center[0] - out_center[0])**2 + 
                                     (exp_center[1] - out_center[1])**2)
                        max_dist = np.sqrt(expected.shape[0]**2 + expected.shape[1]**2)
                        error += dist / max_dist
                    else:
                        error += 1.0  # Color missing entirely
        
        return error / 9.0  # Average over colors
    
    def _calculate_pattern_recognition_error(self, input_grid: np.ndarray,
                                           expected: np.ndarray,
                                           output: np.ndarray) -> float:
        """Calculate pattern recognition accuracy"""
        if output.shape != expected.shape:
            return 1.0
        
        # Check for repeating patterns
        # Simple approach: check if local patterns are preserved
        error = 0.0
        window_size = 3
        
        if expected.shape[0] >= window_size and expected.shape[1] >= window_size:
            count = 0
            for i in range(expected.shape[0] - window_size + 1):
                for j in range(expected.shape[1] - window_size + 1):
                    expected_window = expected[i:i+window_size, j:j+window_size]
                    output_window = output[i:i+window_size, j:j+window_size]
                    
                    # Check if pattern is preserved
                    if np.array_equal(expected_window, output_window):
                        error += 0
                    else:
                        error += 1
                    count += 1
            
            error = error / count if count > 0 else 0.0
        else:
            # Too small for pattern analysis, use pixel accuracy
            error = 1.0 - (output == expected).mean()
        
        return error
    
    def _calculate_structural_integrity_error(self, expected: np.ndarray,
                                            output: np.ndarray) -> float:
        """Calculate overall structural integrity"""
        error = 0.0
        
        # 1. Shape consistency
        if output.shape != expected.shape:
            error += 0.3
        
        # 2. Valid color range
        if output.min() < 0 or output.max() > 9:
            invalid_ratio = ((output < 0) | (output > 9)).mean()
            error += 0.3 * invalid_ratio
        
        # 3. Color diversity (not all same color)
        expected_colors = len(np.unique(expected))
        output_colors = len(np.unique(output))
        if output_colors == 1 and expected_colors > 1:
            error += 0.2
        elif abs(output_colors - expected_colors) > 0:
            error += 0.1 * min(abs(output_colors - expected_colors) / expected_colors, 1.0)
        
        # 4. Connected components (rough check)
        # Check if non-black regions are similarly distributed
        if output.shape == expected.shape:
            expected_nonblack = (expected > 0).mean()
            output_nonblack = (output > 0).mean()
            density_diff = abs(expected_nonblack - output_nonblack)
            error += 0.2 * min(density_diff * 2, 1.0)
        else:
            error += 0.2
        
        return min(error, 1.0)
    
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