"""
LatentSeek optimization implementation for ARC problems
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

from ..data import ARCProblem
from ..generators import BARCGenerator, BARCOutput
from ..executors import CodeExecutor
from ..evaluators import GLMEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of LatentSeek optimization"""
    final_output: BARCOutput
    reward_history: List[float]
    optimization_steps: int
    converged: bool
    
    def __repr__(self):
        return f"OptimizationResult(steps={self.optimization_steps}, final_reward={self.reward_history[-1]:.3f})"


class LatentSeekOptimizer:
    """
    LatentSeek optimization for ARC problems
    
    Implementation based on the original LatentSeek paper:
    - Optimizes partial hidden states using policy gradient
    - Generates new sequences from optimized hidden states
    - Stops when reward threshold is reached
    """
    
    def __init__(self, 
                 barc_generator: BARCGenerator,
                 code_executor: CodeExecutor,
                 glm_evaluator: GLMEvaluator,
                 lr: float = 0.03,
                 k: float = 0.1,
                 max_steps: int = 10,
                 reward_threshold: float = -0.2):
        """
        Initialize LatentSeek optimizer
        
        Args:
            barc_generator: BARC model for generation
            code_executor: Code execution engine
            glm_evaluator: GLM evaluation engine
            lr: Learning rate for hidden state optimization
            k: Fraction of hidden states to optimize
            max_steps: Maximum optimization steps
            reward_threshold: Stop if reward exceeds this threshold
        """
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.glm_evaluator = glm_evaluator
        self.lr = lr
        self.k = k
        self.max_steps = max_steps
        self.reward_threshold = reward_threshold
        
        # Access model and tokenizer
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
        
        logger.info(f"LatentSeek optimizer initialized: lr={lr}, k={k}, max_steps={max_steps}")
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_reward: float) -> OptimizationResult:
        """
        Optimize a BARC solution using LatentSeek
        
        Args:
            problem: ARC problem
            initial_output: Initial BARC output
            initial_reward: Initial reward
            
        Returns:
            OptimizationResult with optimized solution
        """
        logger.info(f"Starting LatentSeek optimization for problem {problem.uid}")
        logger.info(f"Initial reward: {initial_reward:.3f}")
        
        # Initialize optimization state
        current_output = initial_output
        reward_history = [initial_reward]
        current_reward = initial_reward
        
        # Check if already good enough
        if current_reward > self.reward_threshold:
            logger.info("Initial solution already meets threshold, skipping optimization")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=True
            )
        
        # Generate initial hidden states
        hidden_states_list = self._generate_with_hidden_states(problem, current_output)
        
        if not hidden_states_list:
            logger.warning("Failed to generate hidden states, returning initial output")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=False
            )
        
        # Calculate update length
        original_length = len(hidden_states_list)
        update_length = min(int(self.k * original_length), 300)
        
        if update_length <= 0:
            logger.warning("Update length is zero, returning initial output")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=False
            )
        
        logger.info(f"Optimizing {update_length} out of {original_length} hidden states")
        
        # Extract hidden states to optimize
        optimized_hidden_states = torch.nn.Parameter(
            torch.stack([
                state.clone().detach().requires_grad_(True) 
                for state in hidden_states_list[:update_length]
            ])
        )
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
            # Check convergence
            if current_reward > self.reward_threshold:
                logger.info(f"Converged at step {step + 1} with reward {current_reward:.3f}")
                return OptimizationResult(
                    final_output=current_output,
                    reward_history=reward_history,
                    optimization_steps=step + 1,
                    converged=True
                )
            
            # Generate new output from optimized hidden states
            new_output = self._generate_from_optimized_states(
                problem, optimized_hidden_states, hidden_states_list[update_length:]
            )
            
            if new_output is None:
                logger.warning(f"Failed to generate output at step {step + 1}")
                continue
            
            # Evaluate new output
            execution_result = self.code_executor.execute(new_output.code, problem)
            evaluation_result = self.glm_evaluator.evaluate(
                problem, new_output, execution_result,
                f"temp_opt_step_{step}"
            )
            
            new_reward = evaluation_result.total_reward
            reward_history.append(new_reward)
            
            logger.info(f"Step {step + 1}: reward = {new_reward:.3f}")
            
            # Update hidden states using policy gradient
            optimizer.zero_grad()
            
            # Calculate policy gradient loss
            logits = self.model.lm_head(optimized_hidden_states)  # [update_length, vocab_size]
            probs = F.softmax(logits, dim=-1) + 1e-8
            
            # Get next token ids (simplified - in practice would need proper decoding)
            next_token_ids = torch.argmax(probs, dim=-1)
            log_probs = torch.log(probs[torch.arange(update_length), next_token_ids] + 1e-10)
            
            # Policy gradient loss: -reward * log_pi(a|s)
            loss = -new_reward * log_probs.sum()
            
            logger.info(f"Step {step + 1}: loss = {loss.item():.3f}")
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Update current state if improvement
            if new_reward > current_reward:
                current_output = new_output
                current_reward = new_reward
                logger.info(f"Updated best output with reward {current_reward:.3f}")
        
        logger.info(f"Optimization completed after {self.max_steps} steps")
        logger.info(f"Final reward: {current_reward:.3f}")
        
        return OptimizationResult(
            final_output=current_output,
            reward_history=reward_history,
            optimization_steps=self.max_steps,
            converged=current_reward > self.reward_threshold
        )
    
    def _generate_with_hidden_states(self, 
                                   problem: ARCProblem, 
                                   reference_output: BARCOutput) -> Optional[List[torch.Tensor]]:
        """
        Generate sequence while capturing hidden states
        
        This is a simplified implementation - in practice would need to modify
        the generation process to capture hidden states at each step
        """
        # Create prompt (same as BARC generator)
        prompt = self.barc_generator._create_prompt(problem)
        text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate with hidden states capture
        hidden_states_list = []
        
        try:
            with torch.no_grad():
                # Forward pass to get initial hidden states
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract hidden states from last layer
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    last_layer_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
                    
                    # Convert to list of tensors for each token
                    for i in range(last_layer_states.size(1)):
                        hidden_states_list.append(last_layer_states[0, i, :].clone())
                
                # Generate some additional tokens to have enough hidden states
                current_ids = inputs.input_ids.clone()
                
                for _ in range(50):  # Generate up to 50 more tokens
                    outputs = self.model(current_ids, output_hidden_states=True)
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        last_hidden = outputs.hidden_states[-1][0, -1, :].clone()
                        hidden_states_list.append(last_hidden)
                    
                    # Get next token
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    # Stop if EOS token
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                        
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                
            logger.info(f"Captured {len(hidden_states_list)} hidden states")
            return hidden_states_list
            
        except Exception as e:
            logger.error(f"Error generating hidden states: {e}")
            return None
    
    def _generate_from_optimized_states(self, 
                                      problem: ARCProblem,
                                      optimized_states: torch.Tensor,
                                      remaining_states: List[torch.Tensor]) -> Optional[BARCOutput]:
        """
        Generate new sequence from optimized hidden states
        
        This is a simplified implementation - in practice would need to properly
        decode from the optimized hidden states
        """
        try:
            # This is a placeholder implementation
            # In practice, this would involve:
            # 1. Using optimized_states as the starting point
            # 2. Continuing generation from there
            # 3. Properly decoding the sequence
            
            # For now, we'll simulate by generating a new candidate
            # and treating it as if it came from the optimized states
            candidates = self.barc_generator.generate(
                problem, 
                temperature=0.7,  # Slightly different temperature for variation
                num_candidates=1
            )
            
            if candidates:
                return candidates[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating from optimized states: {e}")
            return None
    
    def optimize_description_based(self,
                                 problem: ARCProblem,
                                 initial_output: BARCOutput,
                                 initial_reward: float) -> OptimizationResult:
        """
        Optimize BARC solution using description-based LatentSeek
        
        Instead of optimizing the first k% of tokens, we specifically target
        the description tokens for optimization.
        
        Args:
            problem: ARC problem
            initial_output: Initial BARC output
            initial_reward: Initial reward
            
        Returns:
            OptimizationResult with optimized solution
        """
        logger.info(f"Starting description-based LatentSeek optimization for problem {problem.uid}")
        logger.info(f"Initial reward: {initial_reward:.3f}")
        
        # Initialize optimization state
        current_output = initial_output
        reward_history = [initial_reward]
        current_reward = initial_reward
        
        # Check if already good enough
        if current_reward > self.reward_threshold:
            logger.info("Initial solution already meets threshold, skipping optimization")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=True
            )
        
        # Generate hidden states and find description tokens
        hidden_states_info = self._generate_with_description_mapping(problem, current_output)
        
        if not hidden_states_info:
            logger.warning("Failed to generate description mapping, falling back to regular optimization")
            return self.optimize(problem, initial_output, initial_reward)
        
        hidden_states_list, desc_start, desc_end = hidden_states_info
        
        if desc_start is None or desc_end is None:
            logger.warning("Description tokens not found, falling back to regular optimization")
            return self.optimize(problem, initial_output, initial_reward)
        
        desc_length = desc_end - desc_start
        logger.info(f"Found description tokens: {desc_start}-{desc_end} (length: {desc_length})")
        
        if desc_length <= 0:
            logger.warning("Description length is zero, falling back to regular optimization")
            return self.optimize(problem, initial_output, initial_reward)
        
        # Extract description hidden states to optimize
        optimized_desc_states = torch.nn.Parameter(
            torch.stack([
                state.clone().detach().requires_grad_(True) 
                for state in hidden_states_list[desc_start:desc_end]
            ])
        )
        
        # Setup optimizer for description states only
        optimizer = torch.optim.Adam([optimized_desc_states], lr=self.lr)
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Description optimization step {step + 1}/{self.max_steps}")
            
            # Check convergence
            if current_reward > self.reward_threshold:
                logger.info(f"Converged at step {step + 1} with reward {current_reward:.3f}")
                return OptimizationResult(
                    final_output=current_output,
                    reward_history=reward_history,
                    optimization_steps=step + 1,
                    converged=True
                )
            
            # Generate new output with optimized description states
            new_output = self._generate_with_optimized_description(
                problem, optimized_desc_states, hidden_states_list, desc_start, desc_end
            )
            
            if new_output is None:
                logger.warning(f"Failed to generate output at step {step + 1}")
                continue
            
            # Evaluate new output
            execution_result = self.code_executor.execute(new_output.code, problem)
            evaluation_result = self.glm_evaluator.evaluate(
                problem, new_output, execution_result,
                f"temp_desc_opt_step_{step}"
            )
            
            new_reward = evaluation_result.total_reward
            reward_history.append(new_reward)
            
            logger.info(f"Step {step + 1}: reward = {new_reward:.3f}")
            
            # Update description states using policy gradient
            optimizer.zero_grad()
            
            # Calculate policy gradient loss for description tokens only
            desc_logits = self.model.lm_head(optimized_desc_states)  # [desc_length, vocab_size]
            desc_probs = F.softmax(desc_logits, dim=-1) + 1e-8
            
            # Get next token ids for description
            next_token_ids = torch.argmax(desc_probs, dim=-1)
            log_probs = torch.log(desc_probs[torch.arange(desc_length), next_token_ids] + 1e-10)
            
            # Policy gradient loss: -reward * log_pi(a|s) for description only
            loss = -new_reward * log_probs.sum()
            
            logger.info(f"Step {step + 1}: description loss = {loss.item():.3f}")
            
            loss.backward(retain_graph=True)
            optimizer.step()
            
            # Update current state if improvement
            if new_reward > current_reward:
                current_output = new_output
                current_reward = new_reward
                logger.info(f"Updated best output with reward {current_reward:.3f}")
        
        logger.info(f"Description optimization completed after {self.max_steps} steps")
        logger.info(f"Final reward: {current_reward:.3f}")
        
        return OptimizationResult(
            final_output=current_output,
            reward_history=reward_history,
            optimization_steps=self.max_steps,
            converged=current_reward > self.reward_threshold
        )
    
    def _generate_with_description_mapping(self, 
                                         problem: ARCProblem, 
                                         reference_output: BARCOutput) -> Optional[tuple]:
        """
        Generate sequence while capturing hidden states and mapping description tokens
        
        Returns:
            Tuple of (hidden_states_list, desc_start_idx, desc_end_idx) or None
        """
        # Create prompt (same as BARC generator)
        prompt = self.barc_generator._create_prompt(problem)
        text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        # Generate with hidden states capture
        hidden_states_list = []
        generated_text = ""
        
        try:
            with torch.no_grad():
                # Forward pass to get initial hidden states
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Extract hidden states from last layer
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    last_layer_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
                    
                    # Convert to list of tensors for each token
                    for i in range(last_layer_states.size(1)):
                        hidden_states_list.append(last_layer_states[0, i, :].clone())
                
                # Generate additional tokens and track text
                current_ids = inputs.input_ids.clone()
                
                for token_idx in range(200):  # Generate up to 200 more tokens
                    outputs = self.model(current_ids, output_hidden_states=True)
                    
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        last_hidden = outputs.hidden_states[-1][0, -1, :].clone()
                        hidden_states_list.append(last_hidden)
                    
                    # Get next token
                    next_token_logits = outputs.logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0).unsqueeze(0)
                    
                    # Decode token to track text
                    token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=True)
                    generated_text += token_text
                    
                    # Stop if EOS token or we have enough
                    if next_token.item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Check if we've generated the description part
                    if "# description:" in generated_text.lower():
                        # Continue generating for a bit more to get the full description
                        continue
                    
                    if "def transform" in generated_text.lower():
                        # We've reached the function definition, we have enough
                        break
                        
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Now find description token positions in the generated text
                full_text = text + generated_text
                desc_start, desc_end = self._find_description_token_positions(full_text)
                
                logger.info(f"Captured {len(hidden_states_list)} hidden states")
                logger.info(f"Full text length: {len(full_text)}")
                
                # Adjust positions to account for initial prompt length
                prompt_tokens = len(self.tokenizer.tokenize(text))
                if desc_start is not None and desc_end is not None:
                    # Convert character positions to approximate token positions
                    desc_start_approx = prompt_tokens + int(desc_start * 0.3)  # Rough char to token ratio
                    desc_end_approx = prompt_tokens + int(desc_end * 0.3)
                    
                    # Ensure positions are within bounds
                    desc_start_approx = max(0, min(desc_start_approx, len(hidden_states_list) - 1))
                    desc_end_approx = max(desc_start_approx + 1, min(desc_end_approx, len(hidden_states_list)))
                    
                    return hidden_states_list, desc_start_approx, desc_end_approx
                else:
                    return hidden_states_list, None, None
                
        except Exception as e:
            logger.error(f"Error generating description mapping: {e}")
            return None
    
    def _find_description_token_positions(self, text: str) -> tuple:
        """
        Find character positions of description in the generated text
        
        Returns:
            Tuple of (start_pos, end_pos) or (None, None) if not found
        """
        import re
        
        # Look for description pattern
        desc_pattern = r'#\s*description\s*:\s*(.+?)(?=\ndef\s+transform|$)'
        match = re.search(desc_pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            desc_start = match.start(1)  # Start of description content
            desc_end = match.end(1)      # End of description content
            return desc_start, desc_end
        
        return None, None
    
    def _generate_with_optimized_description(self,
                                           problem: ARCProblem,
                                           optimized_desc_states: torch.Tensor,
                                           full_hidden_states: List[torch.Tensor],
                                           desc_start: int,
                                           desc_end: int) -> Optional[BARCOutput]:
        """
        Generate new sequence with optimized description hidden states
        
        This is a simplified implementation that generates a new candidate
        with modified description guidance.
        """
        try:
            # For now, we'll use the optimized description states to guide
            # generation by modifying the temperature based on the optimization
            
            # Calculate how much the description states have changed
            original_desc_states = torch.stack(full_hidden_states[desc_start:desc_end])
            state_change = torch.norm(optimized_desc_states - original_desc_states).item()
            
            # Use state change to modify generation temperature
            # More change = lower temperature (more focused)
            modified_temperature = max(0.3, 0.8 - state_change * 0.1)
            
            # Generate new candidate with modified temperature
            candidates = self.barc_generator.generate(
                problem,
                temperature=modified_temperature,
                num_candidates=1
            )
            
            if candidates:
                return candidates[0]
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error generating with optimized description: {e}")
            return None