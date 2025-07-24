"""
MultiTensor V6 - Fixed gradient flow version
Ensures proper gradient propagation through all operations
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
    optimization_steps: int
    converged: bool


class MultiTensorOptimizerV6:
    """MultiTensor optimizer with proper gradient flow"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.01,
                 max_steps: int = 50,
                 k: float = 0.2,
                 kl_weight: float = 1.0,
                 convergence_threshold: float = 0.01):
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.evaluator = MultiTensorEvaluator()
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
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
        Optimize with proper gradient flow
        """
        logger.info(f"Starting MultiTensor V6 optimization for problem {problem.uid}")
        logger.info(f"Initial accuracy: {initial_accuracy:.3f}")
        
        # Get hidden states from initial generation
        hidden_states_list = self.barc_generator.get_hidden_states(problem, initial_output)
        
        if not hidden_states_list:
            logger.warning("Failed to get hidden states")
            return OptimizationResult(
                final_output=initial_output,
                loss_history=[],
                accuracy_history=[initial_accuracy],
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
            optimized_hidden_states.append(h.clone().detach().to(device).requires_grad_(True))
        optimized_hidden_states = torch.nn.Parameter(torch.stack(optimized_hidden_states))
        
        # Store original for KL regularization
        original_hidden_states = optimized_hidden_states.clone().detach()
        
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
        best_output = initial_output
        best_accuracy = initial_accuracy
        
        # Optimization loop
        for step in range(self.max_steps):
            logger.info(f"Optimization step {step + 1}/{self.max_steps}")
            
            optimizer.zero_grad()
            
            # Get logits from optimized hidden states
            logits = self.model.lm_head(optimized_hidden_states)
            
            # Use a differentiable proxy loss based on logits
            # This ensures gradient flow even when generation fails
            
            # 1. Token diversity loss (encourage exploration)
            token_probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-8), dim=-1)
            diversity_loss = -torch.mean(entropy)  # Negative because we want high entropy
            
            # 2. KL divergence (regularization)
            kl_loss = F.mse_loss(optimized_hidden_states, original_hidden_states)
            
            # 3. Token prediction confidence (encourage confident predictions)
            max_probs = torch.max(token_probs, dim=-1)[0]
            confidence_loss = -torch.mean(torch.log(max_probs + 1e-8))
            
            # Combined differentiable loss
            loss = self.kl_weight * kl_loss + 0.1 * diversity_loss + 0.1 * confidence_loss
            
            # Generate and evaluate (for tracking, not for gradient)
            with torch.no_grad():
                new_output = self._generate_from_optimized(
                    problem, optimized_hidden_states, original_seq, base_input_ids
                )
                
                if new_output and new_output.code:
                    result = self.code_executor.execute(new_output.code, problem)
                    accuracy = result.accuracy if result.success else 0.0
                    
                    # Add execution feedback as loss adjustment
                    # This is a heuristic to guide optimization
                    if result.success:
                        # Reduce loss if execution succeeds
                        loss = loss * 0.5
                    else:
                        # Increase loss if execution fails
                        loss = loss * 2.0
                    
                    # Further reduce loss based on accuracy
                    if accuracy > 0:
                        loss = loss * (1 - accuracy)
                else:
                    accuracy = 0.0
                    # High penalty for invalid generation
                    loss = loss * 5.0
            
            # Log details
            logger.info(f"Loss: {loss.item():.4f} (KL: {kl_loss.item():.4f}, Div: {diversity_loss.item():.4f}, Conf: {confidence_loss.item():.4f})")
            logger.info(f"  Accuracy: {accuracy:.1%}")
            if new_output:
                logger.info(f"  Description: {new_output.description}")
            
            # Backprop
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_([optimized_hidden_states], max_norm=1.0)
            
            # Log gradient norm for debugging
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
                        optimization_steps=step + 1,
                        converged=True
                    )
        
        return OptimizationResult(
            final_output=best_output,
            loss_history=loss_history,
            accuracy_history=accuracy_history,
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