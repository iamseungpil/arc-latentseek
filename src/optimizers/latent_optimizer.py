"""
LatentSeek optimization implementation for ARC problems
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
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
    all_outputs: List[BARCOutput] = field(default_factory=list)  # Store all generated outputs
    accuracy_history: List[float] = field(default_factory=list)  # Store accuracy at each step
    
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
        all_outputs = [initial_output]
        accuracy_history = []
        
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
        # Ensure all tensors are on the same device as the model
        model_device = next(self.model.parameters()).device
        # Stack hidden states - they already have batch dimension [1, hidden_dim]
        optimized_hidden_states = torch.nn.Parameter(torch.stack([
            state.clone().detach().to(model_device).requires_grad_(True)
            for state in hidden_states_list[:update_length]
        ]))  # Shape: [update_length, 1, hidden_dim]
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
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
            
            logger.info(f"Step {step + 1}: reward = {new_reward:.3f}, accuracy = {execution_result.accuracy:.2%}")
            
            # Check for perfect accuracy - early stopping
            if execution_result.accuracy >= 1.0:
                logger.info(f"🎯 Perfect accuracy achieved at step {step + 1}! Early stopping.")
                return OptimizationResult(
                    final_output=new_output,
                    reward_history=reward_history,
                    optimization_steps=step + 1,
                    converged=True
                )
            
            # Update hidden states using policy gradient
            optimizer.zero_grad()
            
            # Calculate policy gradient loss
            # Ensure hidden states are on the correct device
            model_device = next(self.model.parameters()).device
            if optimized_hidden_states.device != model_device:
                optimized_hidden_states = optimized_hidden_states.to(model_device)
            logits = self.model.lm_head(optimized_hidden_states)  # [update_length, 1, vocab_size]
            probs = F.softmax(logits, dim=-1) + 1e-8
            
            # Get next token ids 
            next_token_ids = torch.argmax(probs, dim=-1)  # [update_length, 1]
            
            # Handle different possible shapes
            if next_token_ids.dim() == 2:
                # If shape is [update_length, 1], use it directly for indexing
                log_probs = torch.log(probs[torch.arange(update_length), 0, next_token_ids[:, 0]] + 1e-10)
            elif next_token_ids.dim() == 1:
                # If already 1D, use directly
                log_probs = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)
            else:
                # Unexpected shape, log and handle gracefully
                logger.warning(f"Unexpected next_token_ids shape: {next_token_ids.shape}")
                next_token_ids_flat = next_token_ids.view(-1)[:update_length]
                log_probs = torch.log(probs.view(update_length, -1)[torch.arange(update_length), next_token_ids_flat] + 1e-10)
            
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
        Extract hidden states from the already generated sequence
        
        This ensures consistency between the sequence we evaluate and optimize
        """
        # Create the full text including the generated code
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        
        # Combine prompt with the actual generated code
        full_text = prompt_text + reference_output.code
        
        # Log the generated code for debugging
        logger.info(f"Full text length: {len(full_text)}")
        logger.debug(f"Generated code preview:\n{reference_output.code[:500]}...")
        
        # Tokenize the complete sequence
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        
        # Extract hidden states from the complete sequence
        hidden_states_list = []
        
        try:
            with torch.no_grad():
                # Single forward pass to get all hidden states
                if hasattr(self.model, 'model'):
                    # Use base model for wrapped models
                    outputs = self.model.model(**inputs, output_hidden_states=True)
                    last_layer_states = outputs[0]  # [batch, seq_len, hidden_size]
                else:
                    # Standard transformers model
                    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                    # Extract hidden states from last layer
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        last_layer_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
                    else:
                        logger.error("No hidden states found in model output")
                        return None
                
                # Convert to list of tensors for each token
                # Keep batch dimension to match original LatentSeek
                for i in range(last_layer_states.size(1)):
                    hidden_states_list.append(last_layer_states[:, i, :].clone())  # Keep [1, hidden_dim]
                
                
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
        Generate new sequence from optimized hidden states using Llama's generation mechanism
        """
        try:
            # Create initial prompt up to the point where we have hidden states
            prompt = self.barc_generator._create_prompt(problem)
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            # Combine optimized states with remaining states
            if remaining_states:
                # Stack remaining states - they already have batch dimension [1, hidden_dim]
                remaining_stacked = torch.stack(remaining_states).to(self.model.device)
                all_hidden_states = torch.cat([optimized_states, remaining_stacked], dim=0)
            else:
                # Create empty tensor with matching dimensions
                empty_states = torch.empty(0, 1, optimized_states.shape[-1]).to(self.model.device)
                all_hidden_states = torch.cat([optimized_states, empty_states], dim=0)
            
            # Based on original LatentSeek: decode tokens from optimized hidden states
            generated_seq = []
            
            # Reconstruct the input up to the optimization point
            base_seq_len = inputs.input_ids.shape[1]
            current_input_ids = inputs.input_ids.clone()
            
            # Generate tokens from optimized hidden states
            with torch.no_grad():
                # Get tokens from optimized states (matching original LatentSeek)
                next_tokens = torch.argmax(self.model.lm_head(optimized_states), dim=-1)  # [update_length, 1]
                
                # Handle different tensor shapes properly
                if next_tokens.dim() == 2 and next_tokens.shape[1] == 1:
                    next_tokens = next_tokens.squeeze(-1)  # [update_length]
                elif next_tokens.dim() == 1:
                    pass  # Already correct shape
                else:
                    # Flatten to ensure 1D
                    next_tokens = next_tokens.view(-1)
                
                # Add optimized tokens to input
                if next_tokens.dim() == 0:  # Single scalar
                    generated_seq.append(next_tokens.item())
                    next_tokens_tensor = next_tokens.unsqueeze(0).unsqueeze(0)  # [1, 1]
                else:
                    generated_seq.extend(next_tokens.tolist())
                    next_tokens_tensor = next_tokens.unsqueeze(0)  # [1, update_length]
                current_input_ids = torch.cat([current_input_ids, next_tokens_tensor], dim=-1)
                
                # Continue generation from the optimized point
                generated_text = ""
                for _ in range(2048):  # max_new_tokens
                    # Check if model has a base model attribute (common in wrapped models)
                    if hasattr(self.model, 'model'):
                        # Use base model to get hidden states
                        outputs = self.model.model(current_input_ids, output_hidden_states=True)
                        # In this case, outputs[0] is typically the last hidden state
                        hidden_states = outputs[0][:, -1, :]  # [batch, hidden_dim]
                    else:
                        # Standard transformers model
                        outputs = self.model(current_input_ids, output_hidden_states=True, return_dict=True)
                        # Get hidden states from the last layer
                        hidden_states = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
                    
                    # Apply lm_head to get logits
                    logits = self.model.lm_head(hidden_states)
                    next_token_id = torch.argmax(logits, dim=-1)
                    
                    # Check for EOS
                    token_value = next_token_id.item() if len(next_token_id.shape) == 0 else next_token_id[0].item()
                    if token_value == self.tokenizer.eos_token_id:
                        break
                    
                    # Handle scalar vs tensor properly
                    if next_token_id.dim() == 0:
                        generated_seq.append(next_token_id.item())
                        next_token_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                    else:
                        generated_seq.append(next_token_id[0].item() if next_token_id.numel() > 0 else 0)
                        next_token_tensor = next_token_id.unsqueeze(0) if next_token_id.dim() == 1 else next_token_id
                    
                    current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=-1)
            
            # Decode the complete sequence
            full_ids = torch.cat([inputs.input_ids[0], torch.tensor(generated_seq).to(self.model.device)])
            generated_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
            
            # Extract code from generated text
            code = self._extract_code_from_text(generated_text)
            description = self._extract_description_from_text(generated_text)
            
            if code:
                return BARCOutput(
                    code=code,
                    description=description,
                    concepts=None,
                    plan=None,
                    raw_response=generated_text
                )
            else:
                # Fallback to regular generation
                logger.warning("Failed to extract code from optimized generation, using fallback")
                candidates = self.barc_generator.generate(
                    problem, 
                    temperature=0.7,
                    num_candidates=1
                )
                return candidates[0] if candidates else None
                
        except Exception as e:
            logger.error(f"Error generating from optimized states: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback to regular generation
            candidates = self.barc_generator.generate(
                problem, 
                temperature=0.7,
                num_candidates=1
            )
            return candidates[0] if candidates else None
    
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
        all_outputs = [initial_output]
        accuracy_history = []
        
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
        
        # Store original values before regeneration attempt
        hidden_states_info_original = (hidden_states_list, desc_start, desc_end)
        
        # Check if description is too short (less than 20 tokens for better quality)
        MIN_DESC_LENGTH = 20
        if desc_length < MIN_DESC_LENGTH:
            logger.warning(f"Description too short ({desc_length} tokens < {MIN_DESC_LENGTH}), regenerating with better prompt")
            
            # Generate new candidate with more explicit instruction for detailed description
            description_prompt = """
Please provide a DETAILED description of the pattern/transformation in the comments.
The description MUST be at least 5-6 lines explaining:
1. What specific patterns you observe in the input grid
2. What exact transformation rules are being applied
3. Step-by-step explanation of how the output is generated
4. Any color mappings or spatial transformations
Include this as comments after # description: tag or after # concepts: tag.
"""
            
            # Generate new candidates with better prompt
            new_candidates = self.barc_generator.generate(
                problem,
                temperature=0.7,
                num_candidates=1,
                additional_prompt=description_prompt
            )
            
            if new_candidates and new_candidates[0]:
                logger.info("Generated new candidate with better description")
                current_output = new_candidates[0]
                
                # Re-generate hidden states with new output
                hidden_states_info = self._generate_with_description_mapping(problem, current_output)
                if hidden_states_info:
                    hidden_states_list, desc_start, desc_end = hidden_states_info
                    desc_length = desc_end - desc_start
                    logger.info(f"New description tokens: {desc_start}-{desc_end} (length: {desc_length})")
                    
                    if desc_length < MIN_DESC_LENGTH:
                        logger.warning(f"Still too short ({desc_length} tokens), but continuing with description-based optimization")
                        # Continue with what we have instead of fallback
                else:
                    logger.warning("Failed to re-generate description mapping, continuing with original")
                    # Reset to original values
                    hidden_states_list, desc_start, desc_end = hidden_states_info_original
                    desc_length = desc_end - desc_start
            else:
                logger.warning("Failed to generate better candidate, continuing with original")
                # Continue with original instead of fallback
        
        if desc_length <= 0:
            logger.warning("Description length is zero, falling back to regular optimization")
            return self.optimize(problem, initial_output, initial_reward)
        
        # Extract hidden states from beginning to end of description
        # This includes concepts and description for better optimization
        model_device = next(self.model.parameters()).device
        # Optimize from start to description end (not just description part)
        optimize_start = 0  # Start from beginning
        optimize_end = min(desc_end, len(hidden_states_list))  # Up to description end
        optimize_length = optimize_end - optimize_start
        
        logger.info(f"Optimizing tokens from {optimize_start} to {optimize_end} (length: {optimize_length})")
        
        # Stack hidden states - they already have batch dimension [1, hidden_dim]
        optimized_desc_states = torch.nn.Parameter(torch.stack([
            state.clone().detach().to(model_device).requires_grad_(True)
            for state in hidden_states_list[optimize_start:optimize_end]
        ]))  # Shape: [optimize_length, 1, hidden_dim]
        
        # Setup optimizer for description states only
        optimizer = torch.optim.Adam([optimized_desc_states], lr=self.lr)
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Description optimization step {step + 1}/{self.max_steps}")
            
            # Generate new output with optimized description states
            new_output = self._generate_with_optimized_description(
                problem, optimized_desc_states, hidden_states_list, desc_start, desc_end
            )
            
            if new_output is None:
                logger.warning(f"Failed to generate output at step {step + 1}")
                continue
            
            # After generating new output, find new description positions for next iteration
            # This is crucial because the new code may have different description location
            if step < self.max_steps - 1:  # Don't update on last step
                try:
                    # Generate hidden states for the new output to get updated token positions
                    new_hidden_states_info = self._regenerate_hidden_states(problem, new_output)
                    if new_hidden_states_info:
                        new_hidden_states_list, new_desc_start, new_desc_end = new_hidden_states_info
                        if new_desc_start is not None and new_desc_end is not None:
                            # Update positions and states for next iteration
                            new_desc_length = new_desc_end - new_desc_start
                            if new_desc_length > 0:
                                logger.info(f"Updated description tokens for step {step+2}: {new_desc_start}-{new_desc_end} (length: {new_desc_length})")
                                
                                # Update the optimized states to match new positions
                                if new_desc_length == desc_length:
                                    # Same length, can reuse optimized states
                                    desc_start, desc_end = new_desc_start, new_desc_end
                                    hidden_states_list = new_hidden_states_list
                                else:
                                    # Different length, need to adjust optimized states
                                    logger.info(f"Description length changed from {desc_length} to {new_desc_length}, adjusting...")
                                    desc_start, desc_end = new_desc_start, new_desc_end
                                    desc_length = new_desc_length
                                    hidden_states_list = new_hidden_states_list
                                    
                                    # Reinitialize optimized states with new length
                                    if new_desc_length <= len(new_hidden_states_list):
                                        optimized_desc_states = torch.nn.Parameter(torch.stack([
                                            state.clone().detach().to(model_device).requires_grad_(True)
                                            for state in new_hidden_states_list[desc_start:desc_end]
                                        ]))
                                        
                                        # Update optimizer with new parameters
                                        optimizer = torch.optim.Adam([optimized_desc_states], lr=self.lr)
                except Exception as e:
                    logger.warning(f"Failed to update description positions for step {step+2}: {e}")
            
            # Evaluate new output
            execution_result = self.code_executor.execute(new_output.code, problem)
            evaluation_result = self.glm_evaluator.evaluate(
                problem, new_output, execution_result,
                f"temp_desc_opt_step_{step}"
            )
            
            new_reward = evaluation_result.total_reward
            reward_history.append(new_reward)
            
            logger.info(f"Step {step + 1}: reward = {new_reward:.3f}, accuracy = {execution_result.accuracy:.2%}")
            
            # Log the code changes
            logger.info(f"Step {step + 1} generated code:")
            logger.info(f"{'='*80}")
            logger.info(new_output.code)
            logger.info(f"{'='*80}")
            if new_output.description:
                logger.info(f"Step {step + 1} description: {new_output.description}")
            
            # Store output and accuracy
            all_outputs.append(new_output)
            accuracy_history.append(execution_result.accuracy)
            
            # Check if all training pairs are correct
            if execution_result.accuracy >= 1.0:
                logger.info(f"Perfect accuracy achieved at step {step + 1}!")
                return OptimizationResult(
                    final_output=new_output,
                    reward_history=reward_history,
                    optimization_steps=step + 1,
                    converged=True
                )
            
            # Update description states using policy gradient (based on original LatentSeek)
            if step < self.max_steps - 1:  # Skip gradient update on last step
                optimizer.zero_grad()
                
                # Calculate policy gradient loss for description tokens only
                # Ensure hidden states are on the correct device
                model_device = next(self.model.parameters()).device
                if optimized_desc_states.device != model_device:
                    optimized_desc_states = optimized_desc_states.to(model_device)
                
                desc_logits = self.model.lm_head(optimized_desc_states)  # [desc_length, 1, vocab_size]
                desc_probs = F.softmax(desc_logits, dim=-1) + 1e-8
                
                # Get next token ids for description
                next_token_ids = torch.argmax(desc_probs, dim=-1)  # [desc_length, 1]
                next_token_ids = next_token_ids.squeeze(-1)  # [desc_length]
                
                log_probs = torch.log(desc_probs[torch.arange(optimize_length), 0, next_token_ids] + 1e-10)
                
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
        Extract hidden states from already generated sequence and map description tokens
        
        Returns:
            Tuple of (hidden_states_list, desc_start_idx, desc_end_idx) or None
        """
        # Create the full text including the generated code
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        
        # Combine prompt with the actual generated code
        full_text = prompt_text + reference_output.code
        
        # Log the generated code for debugging
        logger.info(f"Full text length: {len(full_text)}")
        logger.debug(f"Generated code preview:\n{reference_output.code[:500]}...")
        
        # Tokenize the complete sequence
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        
        # Extract hidden states from the complete sequence
        hidden_states_list = []
        
        try:
            with torch.no_grad():
                # Single forward pass to get all hidden states
                if hasattr(self.model, 'model'):
                    # Use base model for wrapped models
                    outputs = self.model.model(**inputs, output_hidden_states=True)
                    last_layer_states = outputs[0]  # [batch, seq_len, hidden_size]
                else:
                    # Standard transformers model
                    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                    # Extract hidden states from last layer
                    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                        last_layer_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
                    else:
                        logger.error("No hidden states found in model output")
                        return None
                
                # Convert to list of tensors for each token
                # Keep batch dimension to match original LatentSeek
                for i in range(last_layer_states.size(1)):
                    hidden_states_list.append(last_layer_states[:, i, :].clone())  # Keep [1, hidden_dim]
                
                # Find description token positions in the CODE ONLY (not in prompt)
                desc_start_char, desc_end_char = self._find_description_token_positions(reference_output.code)
                
                logger.info(f"Captured {len(hidden_states_list)} hidden states")
                logger.info(f"Full text length: {len(full_text)}")
                logger.info(f"Code length: {len(reference_output.code)}")
                
                if desc_start_char is not None and desc_end_char is not None:
                    # Log what was found as description
                    desc_text = reference_output.code[desc_start_char:desc_end_char]
                    logger.info(f"Found description in code: '{desc_text[:100]}...' (char positions: {desc_start_char}-{desc_end_char})")
                    
                    # Adjust character positions to account for prompt
                    desc_start_in_full = desc_start_char + len(prompt_text)
                    desc_end_in_full = desc_end_char + len(prompt_text)
                    
                    # Convert character positions to token positions
                    text_to_desc_start = full_text[:desc_start_in_full]
                    text_to_desc_end = full_text[:desc_end_in_full]
                    
                    tokens_to_desc_start = len(self.tokenizer.tokenize(text_to_desc_start))
                    tokens_to_desc_end = len(self.tokenizer.tokenize(text_to_desc_end))
                    
                    # Ensure positions are within bounds
                    desc_start_token = max(0, min(tokens_to_desc_start, len(hidden_states_list) - 1))
                    desc_end_token = max(desc_start_token + 1, min(tokens_to_desc_end, len(hidden_states_list)))
                    
                    logger.info(f"Found description tokens: {desc_start_token}-{desc_end_token} (length: {desc_end_token - desc_start_token})")
                    return hidden_states_list, desc_start_token, desc_end_token
                else:
                    logger.warning("No description found in generated code")
                    return hidden_states_list, None, None
                
        except Exception as e:
            logger.error(f"Error generating description mapping: {e}")
            return None
    
    def _find_description_token_positions(self, text: str) -> tuple:
        """
        Find character positions of description in the generated text using comprehensive patterns
        
        Returns:
            Tuple of (start_pos, end_pos) or (None, None) if not found
        """
        import re
        
        # Pattern 1: 명시적 description 태그 패턴
        desc_tag_pattern = r'# description:\s*\n((?:#[^\n]*\n)+)'
        desc_tag_match = re.search(desc_tag_pattern, text, re.DOTALL)
        
        if desc_tag_match:
            desc_start = desc_tag_match.start(1)  # Start of description content
            desc_end = desc_tag_match.end(1)      # End of description content  
            return desc_start, desc_end
        else:
            # Pattern 2: concepts 다음 연속된 주석 블록
            after_concepts_pattern = r'# concepts:[^\n]*\n((?:#[^\n]*\n)*)'
            after_concepts_match = re.search(after_concepts_pattern, text, re.DOTALL)
            
            if after_concepts_match and after_concepts_match.group(1).strip():
                desc_start = after_concepts_match.start(1)
                desc_end = after_concepts_match.end(1)
                return desc_start, desc_end
            else:
                # Pattern 3: main/transform 함수 뒤 주석 블록
                func_patterns = [r'def\s+main\s*\(', r'def\s+transform\s*\(']
                
                for func_pattern in func_patterns:
                    func_match = re.search(func_pattern, text)
                    if func_match:
                        after_def = text[func_match.start():]
                        comment_pattern = r'def[^:]*:\s*\n\s*(#[^\n]*(?:\n\s*#[^\n]*)*)'
                        comment_match = re.search(comment_pattern, after_def, re.DOTALL)
                        
                        if comment_match:
                            # 전체 텍스트에서의 절대 위치 계산
                            abs_start = func_match.start() + comment_match.start(1)
                            abs_end = func_match.start() + comment_match.end(1)
                            return abs_start, abs_end
        
        return None, None
        
    def _regenerate_hidden_states(self, problem: ARCProblem, output: BARCOutput) -> Optional[Tuple[List[torch.Tensor], Optional[int], Optional[int]]]:
        """
        Regenerate hidden states for a given output to get updated token positions
        """
        try:
            # Create prompt from the output
            prompt = self.barc_generator._create_prompt(problem)
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            
            # Capture hidden states during generation with the new output's code
            if hasattr(output, 'code') and output.code:
                new_text = prompt_text + output.code
            else:
                return None
            
            # Get hidden states for this text
            inputs = self.barc_generator.tokenizer(new_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.barc_generator.model(**inputs, output_hidden_states=True)
                last_layer_states = outputs.hidden_states[-1][0]  # [seq_len, hidden_dim]
                
            hidden_states_list = [last_layer_states[i] for i in range(last_layer_states.shape[0])]
            
            # Find description tokens in the new text
            desc_start, desc_end = self._find_description_token_positions(new_text)
            if desc_start is not None and desc_end is not None:
                return hidden_states_list, desc_start, desc_end
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error regenerating hidden states: {e}")
            return None
    
    def _generate_with_optimized_description(self,
                                           problem: ARCProblem,
                                           optimized_desc_states: torch.Tensor,
                                           full_hidden_states: List[torch.Tensor],
                                           desc_start: int,
                                           desc_end: int) -> Optional[BARCOutput]:
        """
        Generate new sequence with optimized description hidden states
        """
        try:
            # Splice the optimized description states into the full sequence
            modified_states = full_hidden_states.copy()
            
            logger.info(f"Debug: optimized_desc_states.shape = {optimized_desc_states.shape}")
            if len(modified_states) > 0:
                logger.info(f"Debug: full_hidden_states[0].shape = {modified_states[0].shape}")
            logger.info(f"Debug: desc_start={desc_start}, desc_end={desc_end}")
            
            # Replace description states with optimized ones
            # Ensure dimension compatibility based on actual shapes
            target_shape = modified_states[0].shape if modified_states else None
            
            if target_shape is None:
                logger.error("No hidden states to match shape against")
                return None
                
            logger.info(f"Debug: target_shape = {target_shape}")
            
            # Reshape optimized_desc_states to match target shape
            if optimized_desc_states.dim() == 2 and len(target_shape) == 2:
                # Both 2D: [num_tokens, hidden_dim] and [batch_size, hidden_dim]
                # Make sure optimized states match the batch dimension
                if optimized_desc_states.size(1) == target_shape[1]:
                    # Dimensions match, no reshape needed
                    pass
                else:
                    logger.error(f"Dimension mismatch: optimized {optimized_desc_states.shape} vs target {target_shape}")
                    return None
            elif optimized_desc_states.dim() == 3 and len(target_shape) == 2:
                # optimized is 3D [num_tokens, batch_size, hidden_dim], target is 2D [batch_size, hidden_dim]
                # Squeeze the batch dimension if it's 1
                if optimized_desc_states.size(1) == 1:
                    optimized_desc_states = optimized_desc_states.squeeze(1)  # [num_tokens, hidden_dim]
                else:
                    logger.error(f"Cannot squeeze batch dimension: {optimized_desc_states.shape}")
                    return None
            elif optimized_desc_states.dim() == 2 and len(target_shape) == 3:
                # optimized is 2D, target is 3D - add batch dimension
                if target_shape[0] == 1:
                    optimized_desc_states = optimized_desc_states.unsqueeze(1)  # [num_tokens, 1, hidden_dim]
                else:
                    logger.error(f"Cannot add batch dimension: target shape {target_shape}")
                    return None
            
            logger.info(f"Debug: optimized_desc_states.shape after reshape = {optimized_desc_states.shape}")
            
            for i, idx in enumerate(range(desc_start, desc_end)):
                if idx < len(modified_states) and i < optimized_desc_states.size(0):
                    state_to_insert = optimized_desc_states[i]
                    logger.info(f"Debug: Replacing state {idx}, original shape {modified_states[idx].shape}, new shape {state_to_insert.shape}")
                    logger.info(f"Debug: state_to_insert.dim()={state_to_insert.dim()}, modified_states[idx].dim()={modified_states[idx].dim()}")
                    logger.info(f"Debug: state_to_insert dtype={state_to_insert.dtype}, modified_states[idx] dtype={modified_states[idx].dtype}")
                    modified_states[idx] = state_to_insert
            
            # Create initial prompt
            prompt = self.barc_generator._create_prompt(problem)
            text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            
            # Stack all hidden states - ensure all have same dimensions
            try:
                # Check if all states have same shape
                shapes = [state.shape for state in modified_states]
                if len(set(shapes)) > 1:
                    logger.warning(f"Mixed shapes in hidden states: {shapes[:5]}...")
                    
                    # Find the target shape (most common shape)
                    from collections import Counter
                    shape_counts = Counter(shapes)
                    target_shape = shape_counts.most_common(1)[0][0]
                    logger.info(f"Using target shape: {target_shape}")
                    
                    # Make all states match the target shape
                    for i, state in enumerate(modified_states):
                        current_shape = state.shape
                        if current_shape != target_shape:
                            if current_shape == (4096,) and target_shape == (1, 4096):
                                # 1D to 2D: add batch dimension
                                modified_states[i] = state.unsqueeze(0)
                            elif current_shape == (1, 4096) and target_shape == (4096,):
                                # 2D to 1D: remove batch dimension
                                modified_states[i] = state.squeeze(0)
                            elif len(current_shape) == 3 and len(target_shape) == 2:
                                # 3D to 2D: squeeze first dimension if it's 1
                                if current_shape[0] == 1:
                                    modified_states[i] = state.squeeze(0)
                                else:
                                    logger.error(f"Cannot convert shape {current_shape} to {target_shape}")
                                    return None
                            elif len(current_shape) == 2 and len(target_shape) == 3:
                                # 2D to 3D: add batch dimension
                                if target_shape[0] == 1:
                                    modified_states[i] = state.unsqueeze(0)
                                else:
                                    logger.error(f"Cannot convert shape {current_shape} to {target_shape}")
                                    return None
                            else:
                                logger.error(f"Cannot convert shape {current_shape} to {target_shape}")
                                return None
                
                # Force all states to have the exact same shape before stacking
                fixed_states = []
                logger.info(f"Target shape for stacking: {target_shape}")
                
                for i, state in enumerate(modified_states):
                    current_shape = state.shape
                    logger.info(f"State {i}: shape={current_shape}, dtype={state.dtype}, dim={state.dim()}")
                    if current_shape != target_shape:
                        logger.info(f"Converting state {i}: {current_shape} -> {target_shape}")
                        
                        try:
                            # Direct reshape to target shape - most reliable
                            fixed_state = state.reshape(target_shape)
                            fixed_states.append(fixed_state)
                        except RuntimeError as e:
                            logger.error(f"Cannot reshape {current_shape} to {target_shape}: {e}")
                            # If reshape fails, try padding or truncating
                            if len(current_shape) == 1 and len(target_shape) == 2:
                                fixed_states.append(state.unsqueeze(0))
                            elif len(current_shape) == 2 and len(target_shape) == 1:
                                fixed_states.append(state.squeeze(0))
                            else:
                                logger.error(f"Cannot fix state shape {current_shape} to {target_shape}")
                                return None
                    else:
                        fixed_states.append(state)
                
                # Final verification before stacking
                final_shapes = [s.shape for s in fixed_states]
                if len(set(final_shapes)) != 1:
                    logger.error(f"Failed to unify shapes: {final_shapes[:5]}...")
                    return None
                
                logger.info(f"Stacking {len(fixed_states)} states with shape {fixed_states[0].shape}")
                all_hidden_states = torch.stack(fixed_states).to(self.model.device)
            except Exception as e:
                logger.error(f"Error stacking hidden states: {e}")
                return None
            
            # Generate from optimized description states (based on original LatentSeek)
            generated_seq = []
            
            # Rebuild the sequence with optimized description
            # First part: up to description start
            base_seq_len = inputs.input_ids.shape[1]
            
            # Create new sequence with optimized description tokens
            with torch.no_grad():
                # Generate tokens from optimized description states
                # Ensure proper shape for lm_head
                logger.info(f"Debug LM_HEAD: optimized_desc_states.shape = {optimized_desc_states.shape}")
                logger.info(f"Debug LM_HEAD: optimized_desc_states.dim() = {optimized_desc_states.dim()}")
                
                if optimized_desc_states.dim() == 3:
                    # [num_tokens, 1, hidden_dim] -> [num_tokens, hidden_dim]
                    desc_states_for_lm = optimized_desc_states.squeeze(1)
                    logger.info(f"Debug LM_HEAD: Squeezed to shape {desc_states_for_lm.shape}")
                elif optimized_desc_states.dim() == 2:
                    desc_states_for_lm = optimized_desc_states
                    logger.info(f"Debug LM_HEAD: Using 2D shape {desc_states_for_lm.shape}")
                else:
                    logger.error(f"Debug LM_HEAD: Unexpected dim {optimized_desc_states.dim()}, shape {optimized_desc_states.shape}")
                    desc_states_for_lm = optimized_desc_states
                
                logger.info(f"Debug LM_HEAD: Final desc_states_for_lm.shape = {desc_states_for_lm.shape}")
                logger.info(f"Debug LM_HEAD: About to call lm_head with shape {desc_states_for_lm.shape}")
                
                desc_tokens = torch.argmax(self.model.lm_head(desc_states_for_lm), dim=-1)  # [desc_length]
                logger.info(f"Debug LM_HEAD: desc_tokens.shape = {desc_tokens.shape}")
                
                # FIXED: Build input sequence following original LatentSeek approach
                # Start with prompt only (no accumulation!)
                prompt_length = inputs.input_ids.shape[1]
                current_input_ids = inputs.input_ids.clone()
                
                # Calculate offsets for generation
                desc_start_offset = desc_start - prompt_length
                desc_end_offset = desc_end - prompt_length
                
                # Generate pre-description tokens if any
                if desc_start_offset > 0:
                    # Generate tokens before description one by one
                    with torch.no_grad():
                        for i in range(desc_start_offset):
                            if hasattr(self.model, 'model'):
                                outputs = self.model.model(current_input_ids, output_hidden_states=True)
                                hidden_states = outputs[0][:, -1, :]
                            else:
                                outputs = self.model(current_input_ids, output_hidden_states=True, return_dict=True)
                                hidden_states = outputs.hidden_states[-1][:, -1, :]
                            
                            logits = self.model.lm_head(hidden_states)
                            next_token_id = torch.argmax(logits, dim=-1)
                            
                            if next_token_id.dim() == 0:
                                next_token_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                            else:
                                next_token_tensor = next_token_id.unsqueeze(0)
                            
                            current_input_ids = torch.cat([current_input_ids, next_token_tensor], dim=-1)
                
                # Add optimized description tokens
                logger.info(f"Debug FIXED: current_input_ids.shape before desc = {current_input_ids.shape}")
                logger.info(f"Debug FIXED: desc_tokens.shape = {desc_tokens.shape}")
                
                current_input_ids = torch.cat([current_input_ids, desc_tokens.unsqueeze(0)], dim=-1)
                
                # Continue generation after description
                generated_text = ""
                cnt = 0
                max_new_tokens = 2048
                
                while cnt < max_new_tokens:
                    # Check if model has a base model attribute (common in wrapped models)
                    if hasattr(self.model, 'model'):
                        # Use base model to get hidden states
                        logger.info(f"Debug MODEL: current_input_ids.shape = {current_input_ids.shape}")
                        outputs = self.model.model(current_input_ids, output_hidden_states=True)
                        logger.info(f"Debug MODEL: outputs[0].shape = {outputs[0].shape}")
                        # In this case, outputs[0] is typically the last hidden state
                        hidden_states = outputs[0][:, -1, :]  # [batch, hidden_dim]
                        logger.info(f"Debug MODEL: hidden_states.shape = {hidden_states.shape}")
                    else:
                        # Standard transformers model
                        outputs = self.model(current_input_ids, output_hidden_states=True, return_dict=True)
                        # Get hidden states from the last layer
                        hidden_states = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
                    
                    # Apply lm_head to get logits
                    logits = self.model.lm_head(hidden_states)
                    
                    # Sample with temperature
                    temperature = 0.7
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                    logger.info(f"Debug TOKEN: next_token_id.shape = {next_token_id.shape}")
                    logger.info(f"Debug TOKEN: current_input_ids.shape = {current_input_ids.shape}")
                    
                    # Check for EOS
                    token_value = next_token_id.item() if len(next_token_id.shape) == 0 else next_token_id[0].item()
                    if token_value == self.tokenizer.eos_token_id:
                        break
                    
                    generated_seq.append(next_token_id.item())
                    
                    # Ensure proper shape for concatenation
                    # current_input_ids is [batch, seq_len], need next_token to be [batch, 1]
                    if next_token_id.dim() == 1:
                        # If next_token_id is [batch], reshape to [batch, 1]
                        next_token_to_cat = next_token_id.unsqueeze(-1)
                    elif next_token_id.dim() == 2 and next_token_id.shape[1] == 1:
                        # If next_token_id is already [batch, 1], use as is
                        next_token_to_cat = next_token_id
                    else:
                        # If next_token_id is [batch, num_samples], take first sample
                        next_token_to_cat = next_token_id[:, 0:1]
                    
                    logger.info(f"Debug TOKEN: next_token_to_cat.shape = {next_token_to_cat.shape}")
                    current_input_ids = torch.cat([current_input_ids, next_token_to_cat], dim=-1)
                    cnt += 1
            
            # Decode the complete sequence
            generated_text = self.tokenizer.decode(current_input_ids[0], skip_special_tokens=True)
            
            # Extract code and description
            code = self._extract_code_from_text(generated_text)
            description = self._extract_description_from_text(generated_text)
            
            if code:
                return BARCOutput(
                    code=code,
                    description=description,
                    concepts=None,
                    plan=None,
                    raw_response=generated_text
                )
            else:
                # Fallback
                logger.warning("Failed to extract code from optimized generation")
                # Calculate state change for temperature adjustment
                original_desc_states = torch.stack(full_hidden_states[desc_start:desc_end])
                state_change = torch.norm(optimized_desc_states - original_desc_states).item()
                temperature = max(0.3, 0.8 - state_change * 0.1)
                
                candidates = self.barc_generator.generate(
                    problem,
                    temperature=temperature,
                    num_candidates=1
                )
                return candidates[0] if candidates else None
                
        except Exception as e:
            logger.error(f"Error generating with optimized description: {e}")
            # Fallback to temperature-based generation
            candidates = self.barc_generator.generate(
                problem,
                temperature=0.7,
                num_candidates=1
            )
            return candidates[0] if candidates else None
    
    
    def _extract_code_from_text(self, text: str) -> Optional[str]:
        """Extract Python code from generated text"""
        import re
        
        # Remove the prompt part if it exists
        code_start_markers = ["```python", "from common import", "import numpy", "def transform", "def main"]
        code_start_idx = len(text)
        for marker in code_start_markers:
            idx = text.find(marker)
            if idx != -1:
                code_start_idx = min(code_start_idx, idx)
        
        if code_start_idx < len(text):
            text = text[code_start_idx:]
        
        # Try to find code block
        code_pattern = r'```python\n(.*?)(?:\n```|$)'
        matches = re.findall(code_pattern, text, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Try to find code starting with imports
        # Look for code that starts with imports and continues until end or double newline
        code_patterns = [
            r'(from common import.*?)(?=\n\n\n|\Z)',
            r'(import.*?def (?:transform|main).*?)(?=\n\n\n|\Z)',
            r'(from common.*?def (?:transform|main).*?)(?=\n\n\n|\Z)'
        ]
        
        for pattern in code_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # Last resort: find any def transform/main
        func_pattern = r'(def (?:main|transform)\(.*?)(?=\n\n|\Z)'
        matches = re.findall(func_pattern, text, re.DOTALL)
        
        if matches:
            # Add standard imports
            return "from common import *\nimport numpy as np\nfrom typing import *\n\n" + matches[0].strip()
        
        return None
    
    def _extract_description_from_text(self, text: str) -> Optional[str]:
        """Extract description from generated text using multiple comprehensive patterns"""
        import re
        
        description = None
        
        # Pattern 1: 명시적 description 태그 패턴
        desc_tag_pattern = r'# description:\s*\n((?:#[^\n]*\n)+)'
        desc_tag_match = re.search(desc_tag_pattern, text, re.DOTALL)
        
        if desc_tag_match:
            desc_lines = re.findall(r'#\s*(.*?)$', desc_tag_match.group(1), re.MULTILINE)
            description = ' '.join([line.strip() for line in desc_lines if line.strip()])
        else:
            # Pattern 2: concepts 다음 연속된 모든 주석 블록
            after_concepts_pattern = r'# concepts:[^\n]*\n((?:#[^\n]*\n)*)'
            after_concepts_match = re.search(after_concepts_pattern, text, re.DOTALL)
            
            if after_concepts_match:
                desc_lines = re.findall(r'#\s*(.*?)$', after_concepts_match.group(1), re.MULTILINE)
                description = ' '.join([line.strip() for line in desc_lines if line.strip() and not line.startswith('description:')])
            else:
                # Pattern 3: main/transform 함수 뒤 주석 블록
                func_patterns = [r'def\s+main\s*\(', r'def\s+transform\s*\(']
                
                for func_pattern in func_patterns:
                    func_match = re.search(func_pattern, text)
                    if func_match:
                        # 함수 정의 이후 첫 번째 주석 블록 찾기
                        after_def = text[func_match.start():]
                        comment_pattern = r'def[^:]*:\s*\n\s*(#[^\n]*(?:\n\s*#[^\n]*)*)'
                        comment_match = re.search(comment_pattern, after_def, re.DOTALL)
                        
                        if comment_match:
                            comment_block = comment_match.group(1)
                            desc_lines = re.findall(r'#\s*(.*?)$', comment_block, re.MULTILINE)
                            description = ' '.join([line.strip() for line in desc_lines if line.strip()])
                            break
        
        return description if description else ""