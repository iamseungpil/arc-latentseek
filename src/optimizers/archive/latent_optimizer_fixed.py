"""
Fixed LatentSeek optimization implementation for ARC problems
Addressing issues from original implementation while maintaining core algorithm
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

from ..data import ARCProblem
from ..generators.barc_generator_fixed import BARCGeneratorFixed, BARCOutput
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


class FixedLatentSeekOptimizer:
    """Fixed LatentSeek optimizer with proper indexing and no token accumulation"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
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
    
    def _generate_hidden_states_for_output(self, problem: ARCProblem, output: BARCOutput) -> Optional[Tuple[List[torch.Tensor], int]]:
        """
        Generate hidden states for a given output by running forward pass
        Returns: (hidden_states_list, prompt_length)
        """
        try:
            # Create prompt
            prompt = self.barc_generator._create_prompt(problem)
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            
            # Calculate prompt length
            prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            prompt_length = prompt_tokens.input_ids.shape[1]
            
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
                    return None, 0
            
            # Convert to list of tensors for each token
            hidden_states_list = []
            for i in range(last_layer_states.size(1)):
                hidden_states_list.append(last_layer_states[:, i, :].clone())  # Keep [1, hidden_dim]
            
            return hidden_states_list, prompt_length
            
        except Exception as e:
            logger.error(f"Error generating hidden states: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, 0
    
    def _generate_from_optimized_states(self, 
                                      problem: ARCProblem,
                                      optimized_states: torch.Tensor,
                                      prompt_tokens: torch.Tensor,
                                      prompt_length: int,
                                      start_index: int) -> Optional[BARCOutput]:
        """
        Generate new sequence from optimized hidden states
        Fixed version without token accumulation issues
        """
        try:
            # Start with prompt tokens
            input_ids = prompt_tokens.clone()
            
            # Generate tokens from optimized hidden states
            with torch.no_grad():
                # Get tokens from optimized states
                logits = self.model.lm_head(optimized_states)
                next_tokens = torch.argmax(logits, dim=-1)  # [update_length, vocab_size] or [update_length, 1, vocab_size]
                
                # Handle dimensions
                if next_tokens.dim() == 3:
                    next_tokens = next_tokens.squeeze(1)  # Remove middle dimension
                elif next_tokens.dim() == 2 and next_tokens.shape[1] == 1:
                    next_tokens = next_tokens.squeeze(-1)  # [update_length]
                
                # If we're starting after the prompt, we need to add the skipped tokens
                if start_index > prompt_length:
                    # This means we're optimizing tokens in the middle of generation
                    # We need to include tokens from prompt_length to start_index
                    # For simplicity, we'll just use the optimized tokens
                    logger.info(f"Note: start_index {start_index} > prompt_length {prompt_length}")
                
                # Add optimized tokens to input
                next_tokens_tensor = next_tokens.unsqueeze(0) if next_tokens.dim() == 1 else next_tokens
                input_ids = torch.cat([input_ids, next_tokens_tensor], dim=-1)
                
                # Continue generation from the optimized point
                max_new_tokens = 2048
                for i in range(max_new_tokens):
                    # Check if we're approaching model's max length
                    if input_ids.shape[1] >= 4000:  # Leave buffer before 4096
                        logger.warning(f"Approaching max sequence length ({input_ids.shape[1]}), stopping generation")
                        break
                    
                    # Get next token
                    if hasattr(self.model, 'model'):
                        outputs = self.model.model(input_ids, output_hidden_states=True)
                        hidden_states = outputs[0][:, -1, :]
                    else:
                        outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
                        hidden_states = outputs.hidden_states[-1][:, -1, :]
                    
                    logits = self.model.lm_head(hidden_states)
                    next_token_id = torch.argmax(logits, dim=-1)
                    
                    # Check for EOS
                    token_value = next_token_id.item() if next_token_id.dim() == 0 else next_token_id[0].item()
                    if token_value == self.tokenizer.eos_token_id:
                        break
                    
                    # Add token to sequence
                    if next_token_id.dim() == 0:
                        next_token_id = next_token_id.unsqueeze(0)
                    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
            
            # Decode the complete sequence (skip prompt)
            generated_text = self.tokenizer.decode(input_ids[0][prompt_length:], skip_special_tokens=True)
            
            # Use the same parser as the generator
            from ..generators.code_parser import extract_code_elements, parse_code
            
            # Extract code using the improved parser
            code_blocks = parse_code(generated_text)
            code = code_blocks[0] if code_blocks else ""
            
            # If no code found in blocks, try to extract from the whole response
            if not code:
                if "def transform" in generated_text:
                    # Extract everything from "def transform" to the end
                    start = generated_text.find("def transform")
                    code = generated_text[start:] if start != -1 else ""
                elif "def main" in generated_text:
                    # Extract everything from "def main" to the end
                    start = generated_text.find("def main")
                    code = generated_text[start:] if start != -1 else ""
            
            # Extract concepts, description, plan
            concepts, description, plan = extract_code_elements(generated_text)
            
            if code:
                return BARCOutput(
                    code=code,
                    description=description,
                    concepts=concepts,
                    plan=plan,
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
        Fixed version of optimize with proper indexing
        """
        logger.info(f"Starting fixed LatentSeek optimization for problem {problem.uid}")
        logger.info(f"Initial reward: {initial_reward:.3f}")
        
        # Initialize optimization state
        current_output = initial_output
        reward_history = [initial_reward]
        current_reward = initial_reward
        best_output = current_output
        best_reward = current_reward
        
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
        hidden_states_list, prompt_length = self._generate_hidden_states_for_output(problem, current_output)
        
        if not hidden_states_list:
            logger.warning("Failed to generate hidden states, returning initial output")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=False
            )
        
        # Get prompt tokens
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = prompt_inputs.input_ids
        
        # Calculate update length (following original LatentSeek)
        total_length = len(hidden_states_list)
        generated_length = total_length - prompt_length
        
        # Two strategies:
        # 1. Original: k% of total length
        # 2. Better: k% of generated tokens only
        # Let's use strategy 2 for better focus on generated content
        update_length = min(int(self.k * generated_length), 300)
        start_index = prompt_length  # Start from first generated token
        
        if update_length <= 0:
            logger.warning("Update length is zero, returning initial output")
            return OptimizationResult(
                final_output=current_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=False
            )
        
        logger.info(f"Optimizing {update_length} tokens from position {start_index}")
        logger.info(f"Total tokens: {total_length}, Prompt: {prompt_length}, Generated: {generated_length}")
        
        # Extract hidden states to optimize
        model_device = next(self.model.parameters()).device
        actual_end = min(start_index + update_length, len(hidden_states_list))
        
        optimized_hidden_states = torch.nn.Parameter(torch.stack([
            state.clone().detach().to(model_device).requires_grad_(True)
            for state in hidden_states_list[start_index:actual_end]
        ]))  # Shape: [update_length, 1, hidden_dim]
        
        # Setup optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
            # Generate new output from optimized hidden states
            new_output = self._generate_from_optimized_states(
                problem, optimized_hidden_states, prompt_tokens, prompt_length, start_index
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
            
            # Update best if improved
            if new_reward > best_reward:
                best_output = new_output
                best_reward = new_reward
                logger.info(f"Updated best output with reward {best_reward:.3f}")
            
            # Check for perfect accuracy - early stopping
            if execution_result.accuracy >= 1.0:
                logger.info(f"🎯 Perfect accuracy achieved at step {step + 1}! Early stopping.")
                return OptimizationResult(
                    final_output=new_output,
                    reward_history=reward_history,
                    optimization_steps=step + 1,
                    converged=True
                )
            
            # Check reward threshold
            if new_reward > self.reward_threshold:
                logger.info(f"Reward threshold met at step {step + 1}")
                return OptimizationResult(
                    final_output=new_output,
                    reward_history=reward_history,
                    optimization_steps=step + 1,
                    converged=True
                )
            
            # Update hidden states using policy gradient
            optimizer.zero_grad()
            
            # Calculate policy gradient loss
            logits = self.model.lm_head(optimized_hidden_states)
            
            # Handle different shapes
            if logits.dim() == 2:
                logits = logits.unsqueeze(1)  # Add sequence dimension
            
            probs = F.softmax(logits, dim=-1) + 1e-8
            
            # Get next token ids 
            next_token_ids = torch.argmax(probs, dim=-1)
            
            # Calculate log probabilities
            if logits.dim() == 3:
                if next_token_ids.dim() == 2 and next_token_ids.shape[1] == 1:
                    next_token_ids = next_token_ids.squeeze(1)
                log_probs = torch.log(probs[torch.arange(update_length), 0, next_token_ids] + 1e-10)
            else:
                log_probs = torch.log(probs[torch.arange(update_length), next_token_ids] + 1e-10)
            
            # Policy gradient loss: -reward * log_pi(a|s)
            loss = -new_reward * log_probs.sum()
            
            logger.info(f"Step {step + 1}: loss = {loss.item():.3f}")
            
            loss.backward(retain_graph=True)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], max_norm=1.0)
            
            optimizer.step()
        
        logger.info(f"Optimization completed after {self.max_steps} steps")
        logger.info(f"Best reward achieved: {best_reward:.3f}")
        
        return OptimizationResult(
            final_output=best_output,
            reward_history=reward_history,
            optimization_steps=self.max_steps,
            converged=False
        )