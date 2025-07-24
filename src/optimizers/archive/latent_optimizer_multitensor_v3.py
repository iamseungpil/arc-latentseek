"""
MultiTensor V3 - Policy gradient approach with multi-dimensional rewards
Uses accuracy and multi-dimensional evaluation for reward signal
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
from ..evaluators.multitensor_evaluator import MultiTensorEvaluator

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    final_output: BARCOutput
    reward_history: List[float]
    optimization_steps: int
    converged: bool


class MultiTensorOptimizerV3:
    """MultiTensor optimizer using policy gradient with multi-dimensional rewards"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.01,
                 max_steps: int = 50,
                 k: float = 0.2,
                 reward_threshold: float = 0.9):
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = MultiTensorEvaluator()
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
        self.reward_threshold = reward_threshold
        
        # Direct access to model and tokenizer
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_reward: float) -> OptimizationResult:
        """
        Optimize using policy gradient with multi-dimensional rewards
        """
        logger.info(f"Starting MultiTensor V3 optimization for problem {problem.uid}")
        logger.info(f"Initial reward: {initial_reward:.3f}")
        
        # Get hidden states from initial generation
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        
        # Debug logging
        logger.info(f"Prompt text length: {len(prompt_text)}")
        logger.info(f"Raw response length: {len(initial_output.raw_response)}")
        logger.info(f"First 100 chars of response: {initial_output.raw_response[:100]}")
        
        full_text = prompt_text + initial_output.raw_response
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        
        logger.info(f"Full input_ids shape: {inputs.input_ids.shape}")
        
        with torch.no_grad():
            outputs = self.model(inputs.input_ids, output_hidden_states=True)
            # Get the last layer's hidden states for all tokens
            # outputs.hidden_states[-1] has shape [batch_size, seq_len, hidden_dim]
            last_hidden_states = outputs.hidden_states[-1]  # Last layer
            # Convert to list of hidden states per token
            hidden_states_list = [last_hidden_states[0, i] for i in range(last_hidden_states.shape[1])]
        
        logger.info(f"Number of hidden states (tokens): {len(hidden_states_list)}")
        
        # Get prompt length
        prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_tokens.input_ids.shape[1]
        
        logger.info(f"Prompt token length: {prompt_length}")
        
        # Calculate update length (based on generated length only)
        generated_length = len(hidden_states_list) - prompt_length
        logger.info(f"Generated length: {generated_length} = {len(hidden_states_list)} - {prompt_length}")
        
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
            logits = self.model.lm_head(optimized_hidden_states)
            
            # Generate output from optimized hidden states
            new_output = self._generate_from_optimized(
                problem, optimized_hidden_states, original_seq, base_input_ids
            )
            
            if new_output and new_output.code:
                # Execute code
                result = self.code_executor.execute(new_output.code, problem)
                
                # Compute accuracy-based reward
                accuracy = result.accuracy if result.success else 0.0
                
                # Get multi-dimensional evaluation
                eval_result = self.evaluator.evaluate(problem, new_output, result)
                
                # Combine rewards
                total_reward = accuracy + 0.2 * eval_result.total_reward
                
                # For policy gradient, we need the log probabilities
                probs = torch.softmax(logits, dim=-1) + 1e-8
                
                # Get the tokens that were actually selected
                with torch.no_grad():
                    selected_tokens = torch.argmax(logits, dim=-1)
                    if selected_tokens.dim() > 1:
                        selected_tokens = selected_tokens.squeeze(-1)
                
                # Ensure probs has the right shape
                if len(probs.shape) == 3:
                    probs = probs.squeeze(1)
                
                # Compute log probabilities
                log_probs = torch.log(probs[torch.arange(len(selected_tokens)), selected_tokens] + 1e-10)
                
                # Policy gradient loss (maximize reward = minimize negative reward)
                pg_loss = -total_reward * log_probs.sum()
                
                # KL regularization
                kl_loss = F.mse_loss(optimized_hidden_states, original_hidden_states)
                
                # Total loss
                loss = pg_loss + 0.01 * kl_loss
                
                # Log details
                logger.info(f"Loss: {loss.item():.4f} (PG: {pg_loss.item():.4f}, KL: {kl_loss.item():.4f})")
                logger.info(f"  Total reward: {total_reward:.3f}, Accuracy: {accuracy:.1%}")
                logger.info(f"  Multi-scores: {eval_result.component_scores}")
                logger.info(f"  Description: {new_output.description}")
                
                # Log execution details
                if result.output_grids and problem.train_pairs:
                    for i, (pair, output_grid) in enumerate(zip(problem.train_pairs[:1], result.output_grids[:1])):
                        if isinstance(output_grid, np.ndarray):
                            expected = pair.y
                            logger.info(f"  Example {i}: expected shape={expected.shape}, actual shape={output_grid.shape}")
                            if output_grid.shape == expected.shape:
                                diff = np.sum(output_grid != expected)
                                logger.info(f"    Pixel accuracy: {1 - diff/expected.size:.1%}")
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_([optimized_hidden_states], max_norm=1.0)
                
                optimizer.step()
                
                # Track progress
                reward_history.append(total_reward)
                
                if total_reward > best_reward:
                    best_reward = total_reward
                    best_output = new_output
                    logger.info(f"🎯 Updated best output with reward {best_reward:.3f}")
                
                # Early stopping
                if accuracy >= 1.0:
                    logger.info("🏆 Perfect accuracy achieved!")
                    return OptimizationResult(
                        final_output=new_output,
                        reward_history=reward_history,
                        optimization_steps=step + 1,
                        converged=True
                    )
                
                if total_reward > self.reward_threshold:
                    logger.info(f"✅ Reward threshold reached: {total_reward:.3f}")
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
                reward_history.append(0.0)
        
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