"""
Fixed LatentSeek optimization implementation for ARC problems
Following the original LatentSeek approach without token accumulation
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


class LatentSeekOptimizer:
    """Fixed LatentSeek optimizer that follows original implementation without token accumulation"""
    
    def __init__(self,
                 barc_generator: BARCGenerator,
                 code_executor: CodeExecutor,
                 glm_evaluator: GLMEvaluator,
                 lr: float = 0.03,
                 max_steps: int = 20,
                 k: float = 0.1,
                 reward_threshold: float = 0.5):
        """
        Initialize fixed LatentSeek optimizer
        
        Args:
            barc_generator: BARC code generator
            code_executor: Code executor for running generated code
            glm_evaluator: GLM evaluator for reward calculation
            lr: Learning rate for optimization
            max_steps: Maximum optimization steps
            k: Fraction of tokens to optimize (0.1 = 10%)
            reward_threshold: Reward threshold for early stopping
        """
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.glm_evaluator = glm_evaluator
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
        self.reward_threshold = reward_threshold
        
        # Cache model and tokenizer references
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def _generate_hidden_states_for_output(self, problem: ARCProblem, output: BARCOutput) -> Optional[List[torch.Tensor]]:
        """
        Generate hidden states for a given output by running forward pass
        """
        try:
            # Create prompt
            prompt = self.barc_generator._create_prompt(problem)
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            
            # Add the generated code to get full sequence
            full_text = prompt_text + output.code
            
            # Tokenize and get hidden states
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                if hasattr(self.model, 'model'):
                    # Wrapped model (e.g., PEFT)
                    outputs = self.model.model(**inputs, output_hidden_states=True)
                else:
                    outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
                
                # Extract hidden states from last layer
                if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                    last_layer_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
                else:
                    logger.error("No hidden states found in model output")
                    return None
            
            # Convert to list of tensors for each token
            # Keep batch dimension to match original LatentSeek
            hidden_states_list = []
            for i in range(last_layer_states.size(1)):
                hidden_states_list.append(last_layer_states[:, i, :].clone())  # Keep [1, hidden_dim]
            
            return hidden_states_list
            
        except Exception as e:
            logger.error(f"Error generating hidden states: {e}")
            return None
    
    def _generate_from_optimized_states(self, 
                                      problem: ARCProblem,
                                      optimized_states: torch.Tensor,
                                      base_input_ids: torch.Tensor,
                                      start_index: int) -> Optional[BARCOutput]:
        """
        Generate new sequence from optimized hidden states
        Following the original LatentSeek approach
        """
        try:
            # Start with base input (prompt only, no accumulation)
            input_ids = base_input_ids.clone()
            
            # Generate tokens from optimized hidden states
            with torch.no_grad():
                # Get tokens from optimized states
                next_tokens = torch.argmax(self.model.lm_head(optimized_states), dim=-1)  # [update_length, 1]
                
                if next_tokens.dim() == 2 and next_tokens.shape[1] == 1:
                    next_tokens = next_tokens.squeeze(-1)  # [update_length]
                
                # Add optimized tokens to input
                next_tokens_tensor = next_tokens.unsqueeze(0)  # [1, update_length]
                input_ids = torch.cat([input_ids, next_tokens_tensor], dim=-1)
                
                # Continue generation from the optimized point
                generated_seq = []
                for _ in range(2048):  # max_new_tokens
                    if hasattr(self.model, 'model'):
                        outputs = self.model.model(input_ids, output_hidden_states=True)
                        hidden_states = outputs[0][:, -1, :]  # [batch, hidden_dim]
                    else:
                        outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
                        hidden_states = outputs.hidden_states[-1][:, -1, :]  # [batch, hidden_dim]
                    
                    logits = self.model.lm_head(hidden_states)
                    next_token_id = torch.argmax(logits, dim=-1)
                    
                    # Check for EOS
                    token_value = next_token_id.item() if len(next_token_id.shape) == 0 else next_token_id[0].item()
                    if token_value == self.tokenizer.eos_token_id:
                        break
                    
                    generated_seq.append(token_value)
                    
                    # Add token to sequence
                    if next_token_id.dim() == 0:
                        next_token_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                    else:
                        next_token_tensor = next_token_id.unsqueeze(0) if next_token_id.dim() == 1 else next_token_id
                    
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
            
            # Decode the complete sequence
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
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
    
    def _extract_code_from_text(self, text: str) -> Optional[str]:
        """Extract code from generated text"""
        # Look for code between ```python and ```
        import re
        code_pattern = r'```python\n(.*?)\n```'
        match = re.search(code_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Fallback: look for def main or def transform
        if 'def main(' in text or 'def transform(' in text:
            # Extract from first def to end
            start = text.find('def ')
            if start != -1:
                return text[start:]
        
        return None
    
    def _extract_description_from_text(self, text: str) -> Optional[str]:
        """Extract description from generated text"""
        import re
        
        # Pattern 1: explicit description tag
        desc_pattern = r'# description:\s*\n((?:#[^\n]*\n)+)'
        match = re.search(desc_pattern, text, re.DOTALL)
        if match:
            return match.group(1)
        
        # Pattern 2: comments after concepts
        after_concepts_pattern = r'# concepts:[^\n]*\n((?:#[^\n]*\n)*)'
        match = re.search(after_concepts_pattern, text, re.DOTALL)
        if match and match.group(1).strip():
            return match.group(1)
        
        return None
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_reward: float) -> OptimizationResult:
        """
        Fixed version of optimize following original LatentSeek
        """
        logger.info(f"Starting fixed LatentSeek optimization for problem {problem.uid}")
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
        
        # Generate initial hidden states from the actual generated sequence
        hidden_states_list = self._generate_hidden_states_for_output(problem, current_output)
        
        if not hidden_states_list:
            logger.warning("Failed to generate hidden states, returning initial output")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=False
            )
        
        # Create prompt only (without generated code)
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_inputs.input_ids.shape[1]
        
        # Calculate update length
        generated_length = len(hidden_states_list) - prompt_length
        update_length = min(int(self.k * generated_length), 300)
        start_index = 0  # Start optimizing from beginning of generation
        
        if update_length <= 0:
            logger.warning("Update length is zero, returning initial output")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=False
            )
        
        logger.info(f"Optimizing {update_length} out of {generated_length} generated tokens")
        
        # Create base_input_ids (prompt only, no accumulation!)
        base_input_ids = prompt_inputs.input_ids  # Just the prompt
        
        # Extract hidden states to optimize (from the generated part only)
        model_device = next(self.model.parameters()).device
        optimized_hidden_states = torch.nn.Parameter(torch.stack([
            state.clone().detach().to(model_device).requires_grad_(True)
            for state in hidden_states_list[prompt_length + start_index:prompt_length + start_index + update_length]
        ]))  # Shape: [update_length, 1, hidden_dim]
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
            # Generate new output from optimized hidden states
            new_output = self._generate_from_optimized_states(
                problem, optimized_hidden_states, base_input_ids, start_index
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
            logits = self.model.lm_head(optimized_hidden_states)  # [update_length, 1, vocab_size]
            probs = F.softmax(logits, dim=-1) + 1e-8
            
            # Get next token ids 
            next_token_ids = torch.argmax(probs, dim=-1)  # [update_length, 1]
            
            # Handle different possible shapes
            if next_token_ids.dim() == 2:
                log_probs = torch.log(probs[torch.arange(update_length), 0, next_token_ids[:, 0]] + 1e-10)
            else:
                log_probs = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)
            
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
    
    def optimize_description_based(self,
                                 problem: ARCProblem,
                                 initial_output: BARCOutput,
                                 initial_reward: float) -> OptimizationResult:
        """
        Optimize specifically the description tokens without token accumulation
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
        
        # Generate initial hidden states
        hidden_states_list = self._generate_hidden_states_for_output(problem, current_output)
        
        if not hidden_states_list:
            logger.warning("Failed to generate hidden states, returning initial output")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=False
            )
        
        # Find description positions
        desc_start, desc_end = self._find_description_token_positions(current_output.code)
        
        if desc_start is None or desc_end is None:
            logger.warning("Could not find description in code, falling back to regular optimization")
            return self.optimize(problem, initial_output, initial_reward)
        
        # Create prompt
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_inputs.input_ids.shape[1]
        
        # Map character positions to token positions
        full_text = prompt_text + current_output.code
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        
        # Find token positions for description
        char_to_token = []
        for i in range(inputs.input_ids.shape[1]):
            decoded = self.tokenizer.decode(inputs.input_ids[0, :i+1])
            char_to_token.append(len(decoded))
        
        # Find tokens corresponding to description
        desc_start_token = None
        desc_end_token = None
        prompt_char_length = len(prompt_text)
        
        for i, char_pos in enumerate(char_to_token):
            if desc_start_token is None and char_pos >= prompt_char_length + desc_start:
                desc_start_token = i
            if desc_end_token is None and char_pos >= prompt_char_length + desc_end:
                desc_end_token = i
                break
        
        if desc_start_token is None or desc_end_token is None:
            logger.warning("Could not map description to tokens, falling back to regular optimization")
            return self.optimize(problem, initial_output, initial_reward)
        
        desc_length = desc_end_token - desc_start_token
        logger.info(f"Found description tokens: {desc_start_token}-{desc_end_token} (length: {desc_length})")
        
        # Extract description hidden states to optimize
        model_device = next(self.model.parameters()).device
        optimized_desc_states = torch.nn.Parameter(torch.stack([
            state.clone().detach().to(model_device).requires_grad_(True)
            for state in hidden_states_list[desc_start_token:desc_end_token]
        ]))
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_desc_states], lr=self.lr)
        
        # Base input is just the prompt
        base_input_ids = prompt_inputs.input_ids
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Description optimization step {step + 1}/{self.max_steps}")
            
            # Generate with optimized description
            new_output = self._generate_with_optimized_description(
                problem, optimized_desc_states, base_input_ids, 
                desc_start_token - prompt_length, desc_end_token - prompt_length
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
            logits = self.model.lm_head(optimized_desc_states)
            probs = F.softmax(logits, dim=-1) + 1e-8
            
            # Get next token ids
            next_token_ids = torch.argmax(probs, dim=-1)
            
            # Calculate log probabilities
            if next_token_ids.dim() == 2:
                log_probs = torch.log(probs[torch.arange(desc_length), 0, next_token_ids[:, 0]] + 1e-10)
            else:
                log_probs = torch.log(probs[torch.arange(desc_length), 0, next_token_ids] + 1e-10)
            
            # Policy gradient loss
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
    
    def _generate_with_optimized_description(self,
                                           problem: ARCProblem,
                                           optimized_desc_states: torch.Tensor,
                                           base_input_ids: torch.Tensor,
                                           desc_start_offset: int,
                                           desc_end_offset: int) -> Optional[BARCOutput]:
        """
        Generate with optimized description states without token accumulation
        """
        try:
            # Start with base prompt
            input_ids = base_input_ids.clone()
            
            with torch.no_grad():
                # Generate pre-description tokens if any
                if desc_start_offset > 0:
                    # Generate tokens before description
                    for i in range(desc_start_offset):
                        if hasattr(self.model, 'model'):
                            outputs = self.model.model(input_ids, output_hidden_states=True)
                            hidden_states = outputs[0][:, -1, :]
                        else:
                            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
                            hidden_states = outputs.hidden_states[-1][:, -1, :]
                        
                        logits = self.model.lm_head(hidden_states)
                        next_token_id = torch.argmax(logits, dim=-1)
                        
                        if next_token_id.dim() == 0:
                            next_token_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                        else:
                            next_token_tensor = next_token_id.unsqueeze(0)
                        
                        input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
                
                # Add optimized description tokens
                desc_tokens = torch.argmax(self.model.lm_head(optimized_desc_states), dim=-1)
                if desc_tokens.dim() == 2 and desc_tokens.shape[1] == 1:
                    desc_tokens = desc_tokens.squeeze(-1)
                desc_tokens_tensor = desc_tokens.unsqueeze(0)
                input_ids = torch.cat([input_ids, desc_tokens_tensor], dim=-1)
                
                # Continue generation after description
                for _ in range(2048):  # max_new_tokens
                    if hasattr(self.model, 'model'):
                        outputs = self.model.model(input_ids, output_hidden_states=True)
                        hidden_states = outputs[0][:, -1, :]
                    else:
                        outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
                        hidden_states = outputs.hidden_states[-1][:, -1, :]
                    
                    logits = self.model.lm_head(hidden_states)
                    next_token_id = torch.argmax(logits, dim=-1)
                    
                    # Check for EOS
                    token_value = next_token_id.item() if len(next_token_id.shape) == 0 else next_token_id[0].item()
                    if token_value == self.tokenizer.eos_token_id:
                        break
                    
                    if next_token_id.dim() == 0:
                        next_token_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                    else:
                        next_token_tensor = next_token_id.unsqueeze(0)
                    
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
            
            # Decode the complete sequence
            generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            
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
                logger.warning("Failed to extract code from optimized generation")
                return None
                
        except Exception as e:
            logger.error(f"Error in _generate_with_optimized_description: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def _find_description_token_positions(self, code: str) -> tuple:
        """
        Find character positions of description in the code
        """
        import re
        
        # Pattern 1: explicit description tag
        desc_tag_pattern = r'# description:\s*\n((?:#[^\n]*\n)+)'
        desc_tag_match = re.search(desc_tag_pattern, code, re.DOTALL)
        
        if desc_tag_match:
            return desc_tag_match.start(1), desc_tag_match.end(1)
        
        # Pattern 2: comments after concepts
        after_concepts_pattern = r'# concepts:[^\n]*\n((?:#[^\n]*\n)*)'
        after_concepts_match = re.search(after_concepts_pattern, code, re.DOTALL)
        
        if after_concepts_match and after_concepts_match.group(1).strip():
            return after_concepts_match.start(1), after_concepts_match.end(1)
        
        # Pattern 3: comments in functions
        func_patterns = [r'def\s+main\s*\(', r'def\s+transform\s*\(']
        
        for func_pattern in func_patterns:
            func_match = re.search(func_pattern, code)
            if func_match:
                after_def = code[func_match.start():]
                comment_pattern = r'def[^:]*:\s*\n\s*(#[^\n]*(?:\n\s*#[^\n]*)*)'
                comment_match = re.search(comment_pattern, after_def, re.DOTALL)
                
                if comment_match:
                    abs_start = func_match.start() + comment_match.start(1)
                    abs_end = func_match.start() + comment_match.end(1)
                    return abs_start, abs_end
        
        return None, None