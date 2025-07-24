"""
CompressARC-style Loss-based LatentSeek Optimizer
Direct loss optimization instead of policy gradient
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

from ..data import ARCProblem
from ..generators.barc_generator_fixed import BARCGeneratorFixed, BARCOutput
from ..executors import CodeExecutor
from ..evaluators import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result of optimization process"""
    final_output: BARCOutput
    reward_history: List[float]
    loss_history: List[float]
    optimization_steps: int
    converged: bool


class CompressARCStyleOptimizer:
    """Loss-based optimizer inspired by CompressARC"""
    
    def __init__(self,
                 barc_generator: BARCGeneratorFixed,
                 code_executor: CodeExecutor,
                 lr: float = 0.01,
                 max_steps: int = 50,
                 k: float = 0.2,
                 kl_weight: float = 0.1,
                 structure_weight: float = 0.3,
                 accuracy_weight: float = 0.6):
        """
        Initialize CompressARC-style optimizer
        
        Args:
            barc_generator: BARC generator for code generation
            code_executor: Code executor for running generated code
            lr: Learning rate for optimization
            max_steps: Maximum optimization steps
            k: Fraction of tokens to optimize (0.1 = 10%)
            kl_weight: Weight for KL regularization
            structure_weight: Weight for structural loss
            accuracy_weight: Weight for accuracy loss
        """
        self.barc_generator = barc_generator
        self.code_executor = code_executor
        self.lr = lr
        self.max_steps = max_steps
        self.k = k
        self.kl_weight = kl_weight
        self.structure_weight = structure_weight
        self.accuracy_weight = accuracy_weight
        
        # Cache model and tokenizer references
        self.model = barc_generator.model
        self.tokenizer = barc_generator.tokenizer
    
    def optimize(self, 
                problem: ARCProblem,
                initial_output: BARCOutput,
                initial_reward: float) -> OptimizationResult:
        """
        Optimize solution using direct loss minimization
        
        Args:
            problem: The ARC problem to solve
            initial_output: Initial generated solution
            initial_reward: Initial reward/accuracy
            
        Returns:
            OptimizationResult with optimized solution
        """
        logger.info(f"Starting CompressARC-style optimization for problem {problem.uid}")
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
        
        # Store original hidden states for KL regularization
        original_hidden = hidden_to_optimize.clone().detach()
        
        # Make it a parameter for optimization
        hidden_to_optimize = torch.nn.Parameter(hidden_to_optimize)
        optimizer = torch.optim.Adam([hidden_to_optimize], lr=self.lr, betas=(0.5, 0.9))
        
        # Initialize tracking
        best_output = initial_output
        best_loss = float('inf')
        reward_history = [initial_reward]
        loss_history = []
        
        # Get expected outputs for loss calculation
        expected_outputs = self._get_expected_outputs(problem)
        
        for step in range(self.max_steps):
            optimizer.zero_grad()
            
            # Generate tokens from optimized hidden states
            logits = self.model.lm_head(hidden_to_optimize)
            
            # 1. KL Regularization Loss (like CompressARC)
            # Keep hidden states close to original
            kl_loss = F.mse_loss(hidden_to_optimize, original_hidden)
            
            # 2. Structure Loss
            # Ensure generated tokens follow expected patterns
            structure_loss = self._compute_structure_loss(logits, problem)
            
            # 3. Token Coherence Loss  
            # Encourage meaningful token sequences
            coherence_loss = self._compute_coherence_loss(logits)
            
            # 4. Format Loss
            # Ensure proper code format
            format_loss = self._compute_format_loss(logits)
            
            # Total loss (like CompressARC: KL + reconstruction)
            total_loss = (
                self.kl_weight * kl_loss +
                self.structure_weight * structure_loss +
                (1.0 - self.kl_weight - self.structure_weight) * coherence_loss +
                0.1 * format_loss
            )
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([hidden_to_optimize], max_norm=1.0)
            
            optimizer.step()
            
            # Every few steps, decode and evaluate
            if step % 5 == 0 or step == self.max_steps - 1:
                # Generate new output
                new_output = self._generate_from_optimized_hidden(
                    problem, hidden_to_optimize, hidden_states_list, 
                    prompt_length, start_index
                )
                
                if new_output and new_output.code:
                    # Execute and evaluate
                    result = self.code_executor.execute(new_output.code, problem)
                    accuracy = result.accuracy
                    
                    # Use accuracy as reward
                    reward = accuracy
                    reward_history.append(reward)
                    
                    logger.info(f"Step {step}: loss={total_loss.item():.4f}, accuracy={accuracy:.1%}")
                    logger.info(f"  KL loss: {kl_loss.item():.4f}, Structure loss: {structure_loss.item():.4f}")
                    logger.info(f"  Description: {new_output.description[:100]}...")
                    
                    # Log output comparison for first training example
                    if result.output_grids and problem.train_pairs:
                        expected = problem.train_pairs[0].y
                        actual = result.output_grids[0] if result.output_grids else None
                        if isinstance(actual, np.ndarray):
                            logger.info(f"  Output shape: expected={expected.shape}, actual={actual.shape}")
                            if actual.shape == expected.shape:
                                diff = np.sum(actual != expected)
                                logger.info(f"  Pixel differences: {diff}/{expected.size} ({diff/expected.size*100:.1f}%)")
                            else:
                                logger.info(f"  Shape mismatch - cannot compare pixels")
                        else:
                            logger.info(f"  No valid output grid generated")
                    
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
            
            loss_history.append(total_loss.item())
        
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
    
    def _get_expected_outputs(self, problem: ARCProblem) -> List[np.ndarray]:
        """Get expected output grids for the problem"""
        return [pair.y for pair in problem.train_pairs]
    
    def _compute_structure_loss(self, logits: torch.Tensor, problem: ARCProblem) -> torch.Tensor:
        """
        Compute structural loss to encourage meaningful patterns
        Similar to CompressARC's reconstruction loss
        """
        # Encourage diversity in token predictions (avoid repetition)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        
        # Higher entropy = more diversity = lower loss
        diversity_loss = -torch.mean(entropy)
        
        # Encourage tokens that are likely to form valid code
        code_token_ids = self._get_code_related_token_ids()
        if code_token_ids:
            code_probs = probs[:, :, code_token_ids].sum(dim=-1)
            code_loss = -torch.mean(torch.log(code_probs + 1e-8))
        else:
            code_loss = 0.0
        
        return diversity_loss + 0.5 * code_loss
    
    def _compute_coherence_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute coherence loss to ensure smooth transitions between tokens
        """
        # Adjacent tokens should have similar hidden representations
        if logits.shape[0] > 1:
            # Compute cosine similarity between adjacent positions
            logits_norm = F.normalize(logits, p=2, dim=-1)
            similarities = torch.sum(logits_norm[:-1] * logits_norm[1:], dim=-1)
            
            # We want high similarity (close to 1)
            coherence_loss = torch.mean(1 - similarities)
        else:
            coherence_loss = torch.tensor(0.0, device=logits.device)
        
        return coherence_loss
    
    def _compute_format_loss(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute format loss to encourage proper code structure
        """
        # Get top predicted tokens
        predicted_tokens = torch.argmax(logits, dim=-1)
        
        # Decode to check structure
        try:
            text = self.tokenizer.decode(predicted_tokens.flatten(), skip_special_tokens=True)
            
            # Penalize if missing key elements
            format_score = 0.0
            if 'def' in text:
                format_score += 0.3
            if 'transform' in text or 'main' in text:
                format_score += 0.3
            if 'return' in text:
                format_score += 0.2
            if 'import' in text:
                format_score += 0.2
                
            format_loss = 1.0 - format_score
        except:
            format_loss = 1.0
            
        return torch.tensor(format_loss, device=logits.device, requires_grad=True)
    
    def _get_code_related_token_ids(self) -> List[int]:
        """Get token IDs related to code keywords"""
        keywords = ['def', 'import', 'return', 'if', 'for', 'while', 'class', 
                   'numpy', 'np', 'array', 'grid', 'transform']
        
        token_ids = []
        for keyword in keywords:
            try:
                ids = self.tokenizer.encode(keyword, add_special_tokens=False)
                token_ids.extend(ids)
            except:
                pass
                
        return list(set(token_ids))
    
    def _generate_from_optimized_hidden(self, 
                                      problem: ARCProblem,
                                      optimized_hidden: torch.Tensor,
                                      original_hidden_list: List[torch.Tensor],
                                      prompt_length: int,
                                      start_index: int) -> Optional[BARCOutput]:
        """Generate new output from optimized hidden states using autoregressive generation"""
        try:
            # Create prompt from problem
            prompt = self.barc_generator._create_prompt(problem)
            prompt_text = self.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize prompt
            prompt_tokens = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
            input_ids = prompt_tokens.input_ids
            
            # Generate tokens up to the optimization point using original hidden states
            pre_optimization_tokens = []
            for i in range(prompt_length, start_index):
                hidden = original_hidden_list[i]
                logits = self.model.lm_head(hidden.unsqueeze(0))
                token = torch.argmax(logits, dim=-1)
                pre_optimization_tokens.append(token)
            
            if pre_optimization_tokens:
                pre_opt_ids = torch.stack([t.squeeze() for t in pre_optimization_tokens]).unsqueeze(0)
                input_ids = torch.cat([input_ids, pre_opt_ids], dim=1)
            
            # Add optimized tokens
            optimized_tokens = []
            for i in range(optimized_hidden.shape[0]):
                hidden = optimized_hidden[i]
                logits = self.model.lm_head(hidden.unsqueeze(0))
                token = torch.argmax(logits, dim=-1)
                optimized_tokens.append(token)
            
            if optimized_tokens:
                opt_ids = torch.stack([t.squeeze() for t in optimized_tokens]).unsqueeze(0)
                input_ids = torch.cat([input_ids, opt_ids], dim=1)
            
            # Now autoregressively generate the rest
            max_new_tokens = 800 - input_ids.shape[1]  # Leave some buffer
            
            with torch.no_grad():
                output = self.model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode the full output
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                
                # Parse as BARCOutput
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
            
        return None