"""
Fixed LatentSeek Optimizer V2 - Exactly follows original LatentSeek approach
Only difference is reward/loss calculation
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging
import re

from ..data import ARCProblem
from ..generators.barc_generator_fixed import BARCGeneratorFixed, BARCOutput
from ..executors import CodeExecutor
from ..evaluators.glm_evaluator import GLMEvaluator
from ..evaluators.multitensor_evaluator import MultiTensorEvaluator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    final_output: BARCOutput
    reward_history: List[float]
    optimization_steps: int
    converged: bool


class LatentSeekOptimizerV2:
    """Fixed LatentSeek optimizer following original implementation exactly"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 evaluator=None,  # Can be GLMEvaluator or MultiTensorEvaluator
                 lr: float = 0.03,
                 max_steps: int = 20,
                 k: float = 0.2,
                 reward_threshold: float = 0.8,
                 use_policy_gradient: bool = True):
        """
        Initialize optimizer
        
        Args:
            barc_generator: BARC generator
            code_executor: Code executor
            evaluator: GLMEvaluator or MultiTensorEvaluator
            lr: Learning rate
            max_steps: Maximum optimization steps
            k: Fraction of GENERATED tokens to optimize (not including prompt)
            reward_threshold: Threshold for early stopping
            use_policy_gradient: Use policy gradient (True) or direct loss (False)
        """
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = evaluator
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
        self.reward_threshold = reward_threshold
        self.use_policy_gradient = use_policy_gradient
        
        # Cache model and tokenizer
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_reward: float) -> OptimizationResult:
        """
        Optimize following original LatentSeek approach exactly
        """
        logger.info(f"Starting LatentSeek V2 optimization for problem {problem.uid}")
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
        
        # Get hidden states for the initial output
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
        
        # Get initial full sequence tokens
        full_text = prompt_text + initial_output.raw_response
        full_tokens = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        initial_input_ids = full_tokens.input_ids
        
        # Calculate generated length first
        generated_length = len(hidden_states_list) - prompt_length
        
        # Find description range in the generated response
        generated_text = initial_output.raw_response
        desc_pattern = r'#\s*description:\s*\n((?:#[^\n]*\n)*)'
        match = re.search(desc_pattern, generated_text)
        
        if match:
            # Found description, calculate token positions
            desc_start_char = match.start()
            desc_end_char = match.end()
            
            # Convert character positions to token positions
            current_pos = 0
            desc_start_tok = None
            desc_end_tok = None
            
            generated_token_ids = initial_input_ids[0][prompt_length:].tolist()
            for i, token_id in enumerate(generated_token_ids):
                token_text = self.tokenizer.decode([token_id])
                if desc_start_tok is None and current_pos + len(token_text) > desc_start_char:
                    desc_start_tok = i
                if desc_end_tok is None and current_pos >= desc_end_char:
                    desc_end_tok = i
                    break
                current_pos += len(token_text)
            
            if desc_start_tok is not None and desc_end_tok is not None:
                # Use description tokens
                start_index = prompt_length + desc_start_tok
                update_length = desc_end_tok - desc_start_tok
                logger.info(f"Found description at tokens [{desc_start_tok}:{desc_end_tok}] ({update_length} tokens)")
            else:
                # Fallback to original method
                logger.warning("Could not determine description token range, using default 20%")
                update_length = min(int(self.k * generated_length), 300)
                start_index = prompt_length
        else:
            # No description found, use original method
            logger.warning("No description found in generated code, using default 20%")
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
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Store tokens before optimization window
        base_input_ids = prompt_tokens.input_ids
        original_seq = initial_input_ids[0][prompt_length:start_index].tolist()
        
        # Optimization tracking
        reward_history = [initial_reward]
        current_reward = initial_reward
        best_output = initial_output
        best_reward = initial_reward
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
            if current_reward >= self.reward_threshold:
                logger.info(f"Reward threshold reached: {current_reward:.3f}")
                return OptimizationResult(
                    final_output=best_output,
                    reward_history=reward_history,
                    optimization_steps=step,
                    converged=True
                )
            
            optimizer.zero_grad()
            
            # Get logits from optimized hidden states
            logits = self.model.lm_head(optimized_hidden_states)  # [update_length, hidden_dim] -> [update_length, vocab_size]
            
            if self.use_policy_gradient:
                # Policy gradient loss (like original LatentSeek)
                probs = torch.softmax(logits, dim=-1) + 1e-8
                next_token_ids = torch.argmax(probs, dim=-1)
                
                # Handle different tensor shapes
                if len(probs.shape) == 3:  # [update_length, 1, vocab_size]
                    log_probs = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)
                else:  # [update_length, vocab_size]
                    log_probs = torch.log(probs[torch.arange(update_length), next_token_ids] + 1e-10)
                
                loss = -current_reward * log_probs.sum()
            else:
                # Direct loss optimization (for MultiTensor)
                # Generate and evaluate to get loss
                new_output = self._generate_from_optimized(
                    problem, optimized_hidden_states, original_seq, base_input_ids
                )
                
                if new_output and new_output.code:
                    result = self.code_executor.execute(new_output.code, problem)
                    
                    if isinstance(self.evaluator, MultiTensorEvaluator):
                        eval_result = self.evaluator.evaluate(problem, new_output, result)
                        # Convert multi-dimensional scores to loss
                        accuracy_loss = 1.0 - result.accuracy
                        quality_loss = 1.0 - eval_result.component_scores.get('code_quality', 0.0)
                        structure_loss = 1.0 - eval_result.component_scores.get('structure', 0.0)
                        
                        loss = (0.6 * accuracy_loss + 0.2 * quality_loss + 0.2 * structure_loss)
                        loss = torch.tensor(loss, device=device, requires_grad=True)
                    else:
                        # GLM or accuracy-based loss
                        loss = torch.tensor(1.0 - result.accuracy, device=device, requires_grad=True)
                else:
                    loss = torch.tensor(10.0, device=device, requires_grad=True)
            
            logger.info(f"Loss: {loss.item():.4f}")
            loss.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], max_norm=1.0)
            optimizer.step()
            
            # Generate new answer (following original approach exactly)
            with torch.no_grad():
                new_output = self._generate_from_optimized(
                    problem, optimized_hidden_states, original_seq, base_input_ids
                )
                
                if new_output and new_output.code:
                    # Evaluate
                    result = self.code_executor.execute(new_output.code, problem)
                    
                    if self.evaluator:
                        if isinstance(self.evaluator, GLMEvaluator):
                            eval_result = self.evaluator.evaluate(
                                problem, new_output, result,
                                base_path=f"temp_{problem.uid}_step{step}"
                            )
                            current_reward = -eval_result.total_reward  # GLM returns negative rewards
                        else:
                            eval_result = self.evaluator.evaluate(problem, new_output, result)
                            current_reward = eval_result.total_reward
                    else:
                        current_reward = result.accuracy
                    
                    logger.info(f"Step {step + 1}: reward = {current_reward:.3f}, accuracy = {result.accuracy:.1%}")
                    logger.info(f"  Description: {new_output.description}")
                    
                    # Log output comparison
                    if result.output_grids and problem.train_pairs:
                        expected = problem.train_pairs[0].y
                        actual = result.output_grids[0] if result.output_grids else None
                        if isinstance(actual, np.ndarray):
                            logger.info(f"  Output shape: expected={expected.shape}, actual={actual.shape}")
                            if actual.shape == expected.shape:
                                diff = np.sum(actual != expected)
                                logger.info(f"  Pixel differences: {diff}/{expected.size}")
                    
                    reward_history.append(current_reward)
                    
                    if current_reward > best_reward:
                        best_reward = current_reward
                        best_output = new_output
                        logger.info(f"Updated best output with reward {best_reward:.3f}")
                    
                    # Early stopping on perfect accuracy
                    if result.accuracy >= 1.0:
                        logger.info("🎯 Perfect accuracy achieved!")
                        return OptimizationResult(
                            final_output=new_output,
                            reward_history=reward_history,
                            optimization_steps=step + 1,
                            converged=True
                        )
        
        return OptimizationResult(
            final_output=best_output,
            reward_history=reward_history,
            optimization_steps=self.max_steps,
            converged=False
        )
    
    def _generate_from_optimized(self,
                                problem: ARCProblem,
                                optimized_hidden: torch.Tensor,
                                original_seq: List[int],
                                base_input_ids: torch.Tensor) -> Optional[BARCOutput]:
        """
        Generate following original LatentSeek approach exactly
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
                # Get tokens from optimized hidden states
                logits = self.model.lm_head(optimized_hidden)
                
                # Handle different shapes
                if len(logits.shape) == 3:  # [update_length, 1, vocab_size]
                    next_tokens = torch.argmax(logits, dim=-1).squeeze(-1)  # [update_length]
                else:  # [update_length, vocab_size]
                    next_tokens = torch.argmax(logits, dim=-1)  # [update_length]
                
                next_tokens = next_tokens.unsqueeze(0)  # [1, update_length]
                input_ids = torch.cat([input_ids, next_tokens], dim=-1)
            
            # Generate the rest autoregressively
            max_new_tokens = min(800, 4096 - input_ids.shape[1])
            
            with torch.no_grad():
                cnt = 0
                while cnt < max_new_tokens:
                    # Get model output
                    outputs = self.model.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs[0][:, -1]  # Last hidden state
                    logits = self.model.lm_head(hidden_states)
                    next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    input_ids = torch.cat([input_ids, next_token_id], dim=-1)
                    
                    # Check for EOS
                    if next_token_id.item() == self.tokenizer.eos_token_id:
                        break
                    
                    cnt += 1
                    
                    # Check sequence length
                    if input_ids.shape[1] >= 4000:
                        logger.warning(f"Sequence too long ({input_ids.shape[1]}), stopping generation")
                        break
            
            # Decode
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
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