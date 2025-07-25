"""
Exact LatentSeek Optimizer - Matches original implementation precisely
Key fix: Uses model.model interface correctly for LlamaForCausalLM
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass
import logging

from ..data import ARCProblem
from ..generators.barc_generator_fixed import BARCGeneratorFixed, BARCOutput
from ..executors import CodeExecutor
from ..evaluators.glm_evaluator import GLMEvaluator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    final_output: BARCOutput
    reward_history: List[float]
    optimization_steps: int
    converged: bool
    original_length: int
    optimized_length: int
    update_length: int


class LatentSeekOptimizerExact:
    """Exact LatentSeek optimizer matching original implementation"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 evaluator: GLMEvaluator,
                 lr: float = 0.03,
                 max_steps: int = 10,
                 k: float = 0.1,
                 reward_threshold: float = -0.2,
                 grad_clip: Optional[float] = None,
                 max_new_tokens: int = 1024):
        """
        Initialize optimizer to match original LatentSeek exactly
        
        Args:
            barc_generator: BARC generator
            code_executor: Code executor
            evaluator: GLM evaluator (returns negative rewards)
            lr: Learning rate (default 0.03)
            max_steps: Maximum optimization steps (default 10)
            k: Fraction of tokens to optimize (default 0.1)
            reward_threshold: Threshold for early stopping (default -0.2)
            grad_clip: Gradient clipping value (optional)
            max_new_tokens: Max tokens to generate (default 1024)
        """
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = evaluator
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
        self.reward_threshold = reward_threshold
        self.grad_clip = grad_clip
        self.max_new_tokens = max_new_tokens
        
        # Cache model and tokenizer
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
        
        # Set up stop words like original
        self.stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
        if self.tokenizer.eos_token:
            self.stop_words.append(self.tokenizer.eos_token)
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_reward: float,
                start_index: int = 0) -> OptimizationResult:
        """
        Optimize following original LatentSeek approach exactly
        """
        logger.info(f"Starting exact LatentSeek optimization for problem {problem.uid}")
        logger.info(f"-- Original Output: {initial_output.description} -- Initial Reward: {initial_reward}")
        
        reward_history = [initial_reward]
        current_reward = initial_reward
        
        # Check if already good enough
        if current_reward > self.reward_threshold:
            logger.info("Initial solution already meets threshold")
            original_length = len(self.tokenizer.encode(initial_output.raw_response))
            return OptimizationResult(
                final_output=initial_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=True,
                original_length=original_length,
                optimized_length=original_length,
                update_length=0
            )
        
        # Get hidden states and input_ids for the initial output
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        
        # Get full sequence
        full_text = prompt_text + initial_output.raw_response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        input_ids = inputs.input_ids
        
        # Get hidden states
        with torch.no_grad():
            # Use model.model like original LatentSeek
            outputs = self.model.model(input_ids, output_hidden_states=True)
            all_hidden_states = outputs[2]  # Tuple of hidden states
            last_hidden_states = all_hidden_states[-1]  # Last layer
            
            # Create list of hidden states for each token
            hidden_states_list = []
            for i in range(last_hidden_states.size(1)):
                hidden_states_list.append(last_hidden_states[:, i, :])
        
        # Calculate lengths
        original_length = len(hidden_states_list)
        
        # Create base prompt tokens
        base_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        base_input_ids = base_inputs.input_ids.clone()
        prompt_length = base_input_ids.shape[1]
        
        # Calculate update length (matching original)
        update_length = min(int(self.k * original_length), 300)
        if update_length <= 0:
            logger.warning("Update Length Zero!!!")
            return OptimizationResult(
                final_output=initial_output,
                reward_history=reward_history,
                optimization_steps=0,
                converged=False,
                original_length=original_length,
                optimized_length=original_length,
                update_length=0
            )
        
        # Create optimized hidden states
        optimized_hidden_states = torch.nn.Parameter(torch.stack([
            state.clone().detach().requires_grad_(True)
            for state in hidden_states_list[start_index:min(start_index + update_length, len(hidden_states_list))]
        ]))
        
        # Configure optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Get original sequence tokens
        original_seq = []
        original_seq.extend(input_ids[0][len(base_input_ids[-1]):len(base_input_ids[-1]) + start_index].tolist())
        
        # Adjust input_ids to optimization start point
        input_ids = input_ids[:, :len(base_input_ids[-1]) + start_index]
        base_input_ids = input_ids.clone()
        new_answer = None
        
        # Optimization loop
        for step in range(self.max_steps):
            input_ids = base_input_ids.clone()
            
            if current_reward > self.reward_threshold:
                final_answer = new_answer if new_answer is not None else initial_output
                optimized_length = len(self.tokenizer.encode(final_answer.raw_response if isinstance(final_answer, BARCOutput) else final_answer))
                logger.info(f"-- Final Answer: {final_answer}, -- Current Reward: {current_reward}")
                return OptimizationResult(
                    final_output=final_answer if isinstance(final_answer, BARCOutput) else initial_output,
                    reward_history=reward_history,
                    optimization_steps=step,
                    converged=True,
                    original_length=original_length,
                    optimized_length=optimized_length,
                    update_length=update_length
                )
            
            optimizer.zero_grad()
            
            # Get logits from optimized hidden states
            logits = self.model.lm_head(optimized_hidden_states)  # [update_length, vocab_size]
            probs = torch.softmax(logits, dim=-1) + 1e-8
            
            # Get next tokens
            next_token_ids = torch.argmax(probs, dim=-1)
            if len(next_token_ids.shape) > 1:
                next_token_ids = next_token_ids.squeeze(-1)
            
            # Calculate log probabilities for policy gradient
            log_pi_xz = torch.log(probs[torch.arange(update_length), next_token_ids] + 1e-10)
            
            # Total loss (negative reward for gradient ascent)
            loss = -current_reward * log_pi_xz.sum()
            logger.info(f"-- Loss: {loss.item()}")
            loss.backward(retain_graph=True)
            
            # Gradient clipping if specified
            if self.grad_clip:
                torch.nn.utils.clip_grad_norm_([optimized_hidden_states], self.grad_clip)
            
            optimizer.step()
            
            # Generate new sequence
            generated_seq = []
            generated_seq.extend(original_seq)
            
            with torch.no_grad():
                # Get tokens from optimized hidden states
                next_tokens = torch.argmax(self.model.lm_head(optimized_hidden_states), dim=-1)
                if len(next_tokens.shape) > 1:
                    next_tokens = next_tokens.squeeze(-1)
                generated_seq.extend(next_tokens.tolist())
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(0)], dim=-1)
                
                # Generate full answer
                cnt = 0
                while True:
                    # Use model.model like original
                    outputs = self.model.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs[0][:, -1]
                    logits = self.model.lm_head(hidden_states)
                    next_token_id = torch.argmax(logits, dim=-1)
                    new_token = self.tokenizer.decode(next_token_id.item())
                    generated_seq.append(next_token_id.item())
                    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=-1)
                    cnt += 1
                    
                    if new_token in self.stop_words or new_token == self.tokenizer.eos_token:
                        break
                    if cnt > self.max_new_tokens:
                        break
                
                # Clean up GPU memory
                del outputs, hidden_states, next_token_id, new_token
                del logits, next_tokens, input_ids
                torch.cuda.empty_cache()
            
            # Decode and evaluate new answer
            new_answer_text = self.tokenizer.decode(generated_seq, skip_special_tokens=True)
            
            # Parse the generated text into BARCOutput
            from ..generators.code_parser import extract_code_elements, parse_code
            
            code_blocks = parse_code(new_answer_text)
            code = code_blocks[0] if code_blocks else ""
            
            if not code and ("def transform" in new_answer_text or "def main" in new_answer_text):
                for func_name in ["def transform", "def main"]:
                    if func_name in new_answer_text:
                        start = new_answer_text.find(func_name)
                        code = new_answer_text[start:]
                        break
            
            concepts, description, plan = extract_code_elements(new_answer_text)
            
            new_answer = BARCOutput(
                code=code,
                concepts=concepts,
                description=description,
                plan=plan,
                raw_response=new_answer_text
            )
            
            # Evaluate
            if code:
                result = self.code_executor.execute(code, problem)
                eval_result = self.evaluator.evaluate(
                    problem, new_answer, result,
                    base_path=f"temp_{problem.uid}_step{step}"
                )
                current_reward = eval_result.total_reward  # GLM returns negative rewards
            else:
                current_reward = -10.0  # Penalty for no code
            
            logger.info(f"-- New Answer: {new_answer.description}, -- Current Reward: {current_reward}")
            reward_history.append(current_reward)
        
        # Return final result
        final_answer = new_answer if new_answer else initial_output
        optimized_length = len(self.tokenizer.encode(final_answer.raw_response))
        logger.info(f"-- Final answer: {final_answer}")
        
        return OptimizationResult(
            final_output=final_answer,
            reward_history=reward_history,
            optimization_steps=self.max_steps,
            converged=False,
            original_length=original_length,
            optimized_length=optimized_length,
            update_length=update_length
        )