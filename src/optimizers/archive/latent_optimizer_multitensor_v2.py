"""
MultiTensor Optimizer V2 with Reconstruction Error
Inspired by CompressARC's multi-dimensional evaluation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
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
    reward_history: List[float]
    optimization_steps: int
    converged: bool


class MultiTensorOptimizerV2:
    """MultiTensor optimizer with reconstruction error inspired by CompressARC"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.01,  # Lower learning rate for direct loss
                 max_steps: int = 50,
                 k: float = 0.2,
                 reward_threshold: float = 0.9,
                 reconstruction_weight: float = 10.0):
        """
        Initialize MultiTensor optimizer with reconstruction error
        
        Args:
            barc_generator: BARC generator
            code_executor: Code executor
            lr: Learning rate
            max_steps: Maximum optimization steps
            k: Fraction of GENERATED tokens to optimize
            reward_threshold: Threshold for early stopping
            reconstruction_weight: Weight for reconstruction error (like CompressARC's 10x)
        """
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = MultiTensorEvaluator()
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
        self.reward_threshold = reward_threshold
        self.reconstruction_weight = reconstruction_weight
        
        # Cache model and tokenizer
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_reward: float) -> OptimizationResult:
        """
        Optimize using reconstruction error approach
        """
        logger.info(f"Starting MultiTensor V2 optimization for problem {problem.uid}")
        logger.info(f"Initial reward: {initial_reward:.3f}")
        
        # Check if already good enough
        if initial_reward >= self.reward_threshold:
            logger.info("Initial solution already meets threshold")
            return OptimizationResult(
                final_output=initial_output,
                reward_history=[initial_reward],
                optimization_steps=0,
                converged=True
            )
        
        # Get hidden states
        hidden_states_list = self.barc_generator.get_hidden_states(problem, initial_output)
        
        if not hidden_states_list:
            logger.warning("Failed to get hidden states")
            return OptimizationResult(
                final_output=initial_output,
                reward_history=[initial_reward],
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
        
        # Calculate update length (based on generated length only)
        generated_length = len(hidden_states_list) - prompt_length
        update_length = min(int(self.k * generated_length), 300)
        start_index = prompt_length
        
        if update_length <= 0:
            logger.warning("Update length is zero!")
            return OptimizationResult(
                final_output=initial_output,
                reward_history=[initial_reward],
                optimization_steps=0,
                converged=False
            )
        
        logger.info(f"Optimizing {update_length} tokens from position {start_index}")
        logger.info(f"Total: {len(hidden_states_list)}, Prompt: {prompt_length}, Generated: {generated_length}")
        
        # Extract hidden states to optimize
        device = next(self.model.parameters()).device
        optimized_hidden_states = torch.nn.Parameter(torch.stack([
            hidden_states_list[i].clone().detach().to(device).requires_grad_(True)
            for i in range(start_index, min(start_index + update_length, len(hidden_states_list)))
        ]))
        
        # Store original hidden states for KL regularization
        original_hidden_states = optimized_hidden_states.clone().detach()
        
        # Setup optimizer
        optimizer = torch.optim.AdamW([optimized_hidden_states], lr=self.lr, weight_decay=0.01)
        
        # Get initial sequence
        full_text = prompt_text + initial_output.raw_response
        full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        initial_input_ids = full_tokens.input_ids
        
        base_input_ids = prompt_tokens.input_ids
        original_seq = initial_input_ids[0][prompt_length:start_index].tolist()
        
        # Tracking
        reward_history = [initial_reward]
        best_output = initial_output
        best_reward = initial_reward
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
            optimizer.zero_grad()
            
            # Get logits from optimized hidden states
            logits = self.model.lm_head(optimized_hidden_states)  # [update_length, vocab_size]
            probs = torch.softmax(logits, dim=-1) + 1e-8
            
            # Generate output from optimized hidden states
            new_output = self._generate_from_optimized(
                problem, optimized_hidden_states, original_seq, base_input_ids
            )
            
            if new_output and new_output.code:
                # Execute code
                result = self.code_executor.execute(new_output.code, problem)
                
                # Compute multi-dimensional reconstruction error
                reconstruction_error = self._compute_reconstruction_error(
                    problem, result, new_output
                )
                
                # For gradient computation, use policy gradient approach
                # Get the tokens that were actually selected
                with torch.no_grad():
                    selected_tokens = torch.argmax(logits, dim=-1)  # [update_length]
                
                # Compute log probabilities of selected tokens
                if len(probs.shape) == 3:  # [update_length, 1, vocab_size]
                    probs = probs.squeeze(1)  # [update_length, vocab_size]
                
                log_probs = torch.log(probs[torch.arange(probs.shape[0]), selected_tokens] + 1e-10)
                
                # Policy gradient loss (negative reward * log prob)
                pg_loss = reconstruction_error * log_probs.sum()
                
                # KL regularization (keep hidden states close to original)
                kl_loss = F.mse_loss(optimized_hidden_states, original_hidden_states)
                
                # Total loss
                total_loss = pg_loss + 0.1 * kl_loss
                
                # Log details
                logger.info(f"Loss: {total_loss.item():.4f} (KL: {kl_loss.item():.4f}, Recon: {reconstruction_error.item():.4f})")
                logger.info(f"  Accuracy: {result.accuracy:.1%}")
                logger.info(f"  Description: {new_output.description}")
                
                # Log detailed reconstruction errors
                if result.output_grids and problem.train_pairs:
                    for i, (pair, output_grid) in enumerate(zip(problem.train_pairs[:3], result.output_grids[:3])):
                        expected = pair.y
                        if isinstance(output_grid, np.ndarray):
                            logger.info(f"  Example {i}: expected shape={expected.shape}, actual shape={output_grid.shape}")
                            if output_grid.shape == expected.shape:
                                diff = np.sum(output_grid != expected)
                                logger.info(f"    Pixel differences: {diff}/{expected.size} ({diff/expected.size*100:.1f}%)")
                                # Color distribution
                                expected_colors = np.unique(expected, return_counts=True)
                                actual_colors = np.unique(output_grid, return_counts=True)
                                logger.info(f"    Expected colors: {dict(zip(expected_colors[0], expected_colors[1]))}")
                                logger.info(f"    Actual colors: {dict(zip(actual_colors[0], actual_colors[1]))}")
                
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([optimized_hidden_states], max_norm=1.0)
                
                optimizer.step()
                
                # Evaluate for tracking
                eval_result = self.evaluator.evaluate(problem, new_output, result)
                current_reward = eval_result.total_reward
                reward_history.append(current_reward)
                
                if current_reward > best_reward:
                    best_reward = current_reward
                    best_output = new_output
                    logger.info(f"Updated best output with reward {best_reward:.3f}")
                
                # Early stopping
                if result.accuracy >= 1.0:
                    logger.info("🎯 Perfect accuracy achieved!")
                    return OptimizationResult(
                        final_output=new_output,
                        reward_history=reward_history,
                        optimization_steps=step + 1,
                        converged=True
                    )
                
                if current_reward >= self.reward_threshold:
                    logger.info(f"Reward threshold reached: {current_reward:.3f}")
                    return OptimizationResult(
                        final_output=best_output,
                        reward_history=reward_history,
                        optimization_steps=step + 1,
                        converged=True
                    )
            else:
                # High penalty for invalid generation
                loss = torch.tensor(100.0, device=device, requires_grad=True)
                loss.backward()
                optimizer.step()
                logger.warning("Failed to generate valid output")
        
        return OptimizationResult(
            final_output=best_output,
            reward_history=reward_history,
            optimization_steps=self.max_steps,
            converged=False
        )
    
    def _compute_reconstruction_error(self, 
                                    problem: ARCProblem,
                                    execution_result,
                                    output: BARCOutput) -> torch.Tensor:
        """
        Compute reconstruction error across multiple dimensions
        Inspired by CompressARC's approach
        """
        device = next(self.model.parameters()).device
        total_error = 0.0
        
        # 1. Grid Size Error (if applicable)
        size_errors = []
        if execution_result.output_grids:
            for i, (pair, output_grid) in enumerate(zip(problem.train_pairs, execution_result.output_grids)):
                if isinstance(output_grid, np.ndarray):
                    expected_shape = pair.y.shape
                    actual_shape = output_grid.shape
                    
                    # Size mismatch penalty
                    if actual_shape != expected_shape:
                        size_error = abs(actual_shape[0] - expected_shape[0]) + abs(actual_shape[1] - expected_shape[1])
                        size_errors.append(size_error / (expected_shape[0] + expected_shape[1]))
                    else:
                        size_errors.append(0.0)
                else:
                    size_errors.append(1.0)  # Maximum penalty for no output
        
        if size_errors:
            avg_size_error = sum(size_errors) / len(size_errors)
            total_error += avg_size_error
        
        # 2. Pixel-wise Reconstruction Error (main component)
        pixel_errors = []
        if execution_result.output_grids:
            for i, (pair, output_grid) in enumerate(zip(problem.train_pairs, execution_result.output_grids)):
                if isinstance(output_grid, np.ndarray):
                    expected = pair.y
                    
                    if output_grid.shape == expected.shape:
                        # Direct pixel comparison
                        pixel_diff = np.sum(output_grid != expected) / expected.size
                        pixel_errors.append(pixel_diff)
                    else:
                        # Shape mismatch - use normalized error
                        # Resize to compare (simple approach)
                        pixel_errors.append(1.0)  # Maximum error for shape mismatch
                else:
                    pixel_errors.append(1.0)
        
        if pixel_errors:
            avg_pixel_error = sum(pixel_errors) / len(pixel_errors)
            total_error += avg_pixel_error
        
        # 3. Color Distribution Error
        color_errors = []
        if execution_result.output_grids:
            for i, (pair, output_grid) in enumerate(zip(problem.train_pairs, execution_result.output_grids)):
                if isinstance(output_grid, np.ndarray):
                    expected = pair.y
                    
                    # Compare color histograms
                    expected_colors = np.bincount(expected.flatten(), minlength=10)[:10]
                    actual_colors = np.bincount(output_grid.flatten(), minlength=10)[:10]
                    
                    # Normalize
                    expected_colors = expected_colors / (expected.size + 1e-8)
                    actual_colors = actual_colors / (output_grid.size + 1e-8)
                    
                    # KL divergence between color distributions
                    kl_div = np.sum(expected_colors * np.log((expected_colors + 1e-8) / (actual_colors + 1e-8)))
                    color_errors.append(min(kl_div, 1.0))  # Cap at 1.0
                else:
                    color_errors.append(1.0)
        
        if color_errors:
            avg_color_error = sum(color_errors) / len(color_errors)
            total_error += avg_color_error * 0.5  # Lower weight for color distribution
        
        # 4. Execution Error
        if not execution_result.success:
            total_error += 1.0  # Add penalty for execution failure
        
        # 5. Structure Error (from evaluator)
        eval_result = self.evaluator.evaluate(problem, output, execution_result)
        structure_score = eval_result.component_scores.get('structure', 0.0)
        structure_error = 1.0 - structure_score
        total_error += structure_error * 0.3  # Lower weight
        
        # Return as scalar tensor (total_error is already a float)
        # Don't use requires_grad=True on a leaf tensor created from numpy
        # Instead, ensure the computation graph is maintained
        return torch.tensor(total_error, device=device)
    
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