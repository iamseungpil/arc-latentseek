"""
MultiTensor Loss-based LatentSeek Optimizer
Uses multi-dimensional evaluation as direct loss for gradient descent
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
    loss_history: List[float]
    optimization_steps: int
    converged: bool


class MultiTensorOptimizer:
    """Multi-dimensional loss-based optimizer"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.005,
                 max_steps: int = 100,
                 k: float = 0.2,
                 accuracy_weight: float = 0.5,
                 quality_weight: float = 0.2,
                 structure_weight: float = 0.2,
                 efficiency_weight: float = 0.1):
        """
        Initialize MultiTensor optimizer
        
        Args:
            barc_generator: BARC generator for code generation
            code_executor: Code executor for running generated code
            lr: Learning rate for optimization
            max_steps: Maximum optimization steps
            k: Fraction of tokens to optimize (0.1 = 10%)
            accuracy_weight: Weight for accuracy loss
            quality_weight: Weight for code quality loss
            structure_weight: Weight for structure loss
            efficiency_weight: Weight for efficiency loss
        """
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = MultiTensorEvaluator()
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
        
        # Loss weights
        self.weights = {
            'accuracy': accuracy_weight,
            'quality': quality_weight,
            'structure': structure_weight,
            'efficiency': efficiency_weight
        }
        
        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        
        # Cache model and tokenizer references
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_reward: float) -> OptimizationResult:
        """
        Optimize solution using multi-dimensional loss
        
        Args:
            problem: The ARC problem to solve
            initial_output: Initial generated solution
            initial_reward: Initial reward/accuracy
            
        Returns:
            OptimizationResult with optimized solution
        """
        logger.info(f"Starting MultiTensor optimization for problem {problem.uid}")
        logger.info(f"Initial reward: {initial_reward:.3f}")
        
        # Get hidden states for initial output
        hidden_states_list = self.barc_generator.get_hidden_states(problem, initial_output)
        
        if not hidden_states_list:
            logger.warning("Failed to get hidden states, returning initial output")
            return OptimizationResult(
                final_output=initial_output,
                reward_history=[initial_reward],
                loss_history=[],
                optimization_steps=0,
                converged=False
            )
        
        # Calculate tokens to optimize
        prompt_length = self._get_prompt_length(problem)
        generated_length = len(hidden_states_list) - prompt_length
        update_length = min(int(self.k * generated_length), 300)
        start_index = prompt_length
        
        logger.info(f"Optimizing {update_length} tokens from position {start_index}")
        
        # Extract hidden states to optimize
        device = next(self.model.parameters()).device
        hidden_to_optimize = torch.stack([
            hidden_states_list[i].clone().detach().to(device).requires_grad_(True)
            for i in range(start_index, min(start_index + update_length, len(hidden_states_list)))
        ])
        
        # Make it a parameter for optimization
        hidden_to_optimize = torch.nn.Parameter(hidden_to_optimize)
        optimizer = torch.optim.AdamW([hidden_to_optimize], lr=self.lr, weight_decay=0.01)
        
        # Initialize tracking
        best_output = initial_output
        best_loss = float('inf')
        reward_history = [initial_reward]
        loss_history = []
        
        for step in range(self.max_steps):
            optimizer.zero_grad()
            
            # Generate new output using autoregressive generation
            new_output = self._generate_from_optimized_hidden(
                problem, hidden_to_optimize, hidden_states_list, 
                prompt_length, start_index
            )
            
            if new_output and new_output.code:
                # Execute and evaluate
                result = self.code_executor.execute(new_output.code, problem)
                eval_result = self.evaluator.evaluate(problem, new_output, result)
                
                # Compute multi-dimensional loss
                losses = self._compute_multidimensional_loss(
                    problem, new_output, result, eval_result
                )
                
                total_loss = sum(self.weights[k] * v for k, v in losses.items())
                
                # Add L2 regularization on hidden state changes
                original_hidden = torch.stack([
                    hidden_states_list[i].to(device)
                    for i in range(start_index, min(start_index + update_length, len(hidden_states_list)))
                ])
                l2_loss = 0.1 * F.mse_loss(hidden_to_optimize, original_hidden)
                total_loss = total_loss + l2_loss
                
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_([hidden_to_optimize], max_norm=1.0)
                
                optimizer.step()
                
                # Log progress
                if step % 5 == 0 or step == self.max_steps - 1:
                    accuracy = result.accuracy
                    reward = eval_result.total_reward
                    reward_history.append(reward)
                    
                    logger.info(f"Step {step}: loss={total_loss.item():.4f}, accuracy={accuracy:.1%}, reward={reward:.3f}")
                    logger.info(f"  Losses: accuracy={losses['accuracy'].item():.4f}, "
                              f"quality={losses['quality'].item():.4f}, "
                              f"structure={losses['structure'].item():.4f}, "
                              f"efficiency={losses['efficiency'].item():.4f}")
                    logger.info(f"  Description: {new_output.description[:100]}...")
                    
                    # Log output comparison
                    if result.output_grids and problem.train_pairs:
                        for i, (pair, output_grid) in enumerate(zip(problem.train_pairs[:3], result.output_grids[:3])):
                            expected = pair.y
                            if isinstance(output_grid, np.ndarray):
                                logger.info(f"  Example {i}: expected shape={expected.shape}, actual shape={output_grid.shape}")
                                if output_grid.shape == expected.shape:
                                    diff = np.sum(output_grid != expected)
                                    logger.info(f"    Pixel differences: {diff}/{expected.size} ({diff/expected.size*100:.1f}%)")
                    
                    # Update best if improved
                    if total_loss.item() < best_loss and result.success:
                        best_output = new_output
                        best_loss = total_loss.item()
                        logger.info(f"Updated best output with loss {best_loss:.4f}")
                    
                    # Early stopping on perfect accuracy
                    if accuracy >= 1.0:
                        logger.info(f"🎯 Perfect accuracy achieved at step {step}!")
                        return OptimizationResult(
                            final_output=new_output,
                            reward_history=reward_history,
                            loss_history=loss_history,
                            optimization_steps=step + 1,
                            converged=True
                        )
            else:
                logger.warning(f"Failed to generate valid output at step {step}")
                # Use a high penalty loss
                total_loss = torch.tensor(10.0, device=device, requires_grad=True)
                total_loss.backward()
                optimizer.step()
            
            loss_history.append(total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss)
        
        return OptimizationResult(
            final_output=best_output,
            reward_history=reward_history,
            loss_history=loss_history,
            optimization_steps=self.max_steps,
            converged=False
        )
    
    def _get_prompt_length(self, problem: ARCProblem) -> int:
        """Get the length of the prompt tokens"""
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt")
        return prompt_tokens.input_ids.shape[1]
    
    def _compute_multidimensional_loss(self, 
                                     problem: ARCProblem,
                                     output: BARCOutput,
                                     execution_result,
                                     eval_result) -> Dict[str, torch.Tensor]:
        """
        Compute multi-dimensional losses based on evaluation
        
        Returns dict of losses (higher is worse)
        """
        device = next(self.model.parameters()).device
        losses = {}
        
        # 1. Accuracy loss (most important)
        # Convert accuracy to loss: 0 accuracy = 1 loss, 1 accuracy = 0 loss
        accuracy_loss = 1.0 - execution_result.accuracy
        losses['accuracy'] = torch.tensor(accuracy_loss, device=device, requires_grad=True)
        
        # 2. Code quality loss
        # Based on evaluator's code quality score (0-1, higher is better)
        quality_score = eval_result.component_scores.get('code_quality', 0.0)
        quality_loss = 1.0 - quality_score
        losses['quality'] = torch.tensor(quality_loss, device=device, requires_grad=True)
        
        # 3. Structure loss
        # Based on output structure similarity
        structure_score = eval_result.component_scores.get('structure', 0.0)
        structure_loss = 1.0 - structure_score
        losses['structure'] = torch.tensor(structure_loss, device=device, requires_grad=True)
        
        # 4. Efficiency loss
        efficiency_score = eval_result.component_scores.get('efficiency', 0.0)
        efficiency_loss = 1.0 - efficiency_score
        losses['efficiency'] = torch.tensor(efficiency_loss, device=device, requires_grad=True)
        
        return losses
    
    def _generate_from_optimized_hidden(self, 
                                      problem: ARCProblem,
                                      optimized_hidden: torch.Tensor,
                                      original_hidden_list: List[torch.Tensor],
                                      prompt_length: int,
                                      start_index: int) -> Optional[BARCOutput]:
        """Generate new output from optimized hidden states using autoregressive generation"""
        try:
            # Create prompt
            prompt = self.barc_generator._create_prompt(problem)
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            
            # Start with prompt tokens
            input_ids = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device).input_ids
            
            # Generate tokens up to optimization point from original hidden states
            with torch.no_grad():
                for i in range(prompt_length, start_index):
                    hidden = original_hidden_list[i].unsqueeze(0)
                    logits = self.model.lm_head(hidden)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Add optimized tokens
            with torch.no_grad():
                for i in range(optimized_hidden.shape[0]):
                    hidden = optimized_hidden[i].unsqueeze(0)
                    logits = self.model.lm_head(hidden)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Check token limit
            if input_ids.shape[1] >= 4000:
                logger.warning(f"Sequence too long ({input_ids.shape[1]}), truncating generation")
                generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            else:
                # Autoregressively generate the rest
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
                
                generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Parse output
            from ..generators.code_parser import extract_code_elements, parse_code
            
            code_blocks = parse_code(generated_text)
            code = code_blocks[0] if code_blocks else ""
            
            if not code and ("def transform" in generated_text or "def main" in generated_text):
                # Extract function definition
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
            logger.error(f"Error generating from optimized hidden: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
        return None