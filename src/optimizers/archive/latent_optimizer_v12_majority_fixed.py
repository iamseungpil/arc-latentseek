"""
V12 Fixed: Properly find and optimize entire description
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import re
from collections import Counter
import logging

from ..evaluators.simple_evaluator import SimpleEvaluator

logger = logging.getLogger(__name__)

class LatentOptimizerV12MajorityFixed:
    def __init__(
        self, 
        model, 
        tokenizer,
        evaluator: SimpleEvaluator,
        num_candidates: int = 8,
        learning_rate: float = 0.001,  # Lower LR
        num_steps: int = 30,
        warmup_steps: int = 5,
        temperature: float = 0.7,  # Lower temperature
        reward_weights: Dict[str, float] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.num_candidates = num_candidates
        self.lr = learning_rate
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        self.temperature = temperature
        
        # 5D reward weights
        self.reward_weights = reward_weights or {
            'execution_success': 0.25,
            'pixel_accuracy': 0.25,
            'transformation': 0.20,
            'pattern': 0.15,
            'semantic': 0.15
        }
        
        # Move model to eval mode but enable gradients
        self.model.eval()
        
    def find_description_tokens_improved(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Find the FULL description token range."""
        # Decode tokens individually
        tokens = []
        for i in range(input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            token_text = self.tokenizer.decode([token_id])
            tokens.append(token_text)
        
        # Find description start
        desc_start = None
        for i in range(len(tokens) - 1):
            if "description" in tokens[i] and ":" in tokens[i+1]:
                desc_start = i + 2  # Skip "description:"
                break
                
        if desc_start is None:
            logger.warning("Could not find description start")
            return None, None
            
        # Find description end - look for first non-comment line
        desc_end = None
        for i in range(desc_start, len(tokens)):
            # Check if we've hit "def" which marks end of description
            if "def" in tokens[i]:
                desc_end = i
                break
            # Or check for empty line followed by non-comment
            elif i > desc_start + 5 and tokens[i].strip() == "" and i+1 < len(tokens) and not tokens[i+1].strip().startswith('#'):
                desc_end = i
                break
                
        if desc_end is None:
            # Fallback: assume description is at most 100 tokens
            desc_end = min(desc_start + 100, len(tokens))
            
        # Log what we found
        desc_length = desc_end - desc_start
        logger.info(f"Description found at tokens [{desc_start}:{desc_end}] (length: {desc_length})")
        
        # Show sample of description tokens
        sample_tokens = tokens[desc_start:min(desc_start + 10, desc_end)]
        logger.debug(f"First 10 description tokens: {sample_tokens}")
        
        return desc_start, desc_end
        
    def sample_tokens(self, logits: torch.Tensor, temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample tokens from logits and return both tokens and their log probabilities."""
        # Apply temperature
        logits = logits / temperature
        
        # Get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample tokens
        sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).squeeze(-1)
        
        # Get log probabilities of sampled tokens
        log_probs = F.log_softmax(logits, dim=-1)
        sampled_log_probs = log_probs.view(-1, log_probs.size(-1))[
            torch.arange(sampled_tokens.size(0)), sampled_tokens
        ]
        
        return sampled_tokens.view(logits.shape[:-1]), sampled_log_probs.view(logits.shape[:-1])
        
    def calculate_5d_reward(self, generated_outputs: List[np.ndarray], target_outputs: List[np.ndarray]) -> float:
        """Calculate 5D reward for generated outputs."""
        if not generated_outputs or any(out is None for out in generated_outputs):
            return 0.0
            
        rewards = {
            'execution_success': 1.0,  # Already successful if we got here
            'pixel_accuracy': 0.0,
            'transformation': 0.0,
            'pattern': 0.0,
            'semantic': 0.0
        }
        
        # Calculate pixel accuracy
        correct_pixels = 0
        total_pixels = 0
        for gen, target in zip(generated_outputs, target_outputs):
            if gen.shape == target.shape:
                correct_pixels += np.sum(gen == target)
                total_pixels += gen.size
            else:
                # Shape mismatch penalty
                total_pixels += max(gen.size, target.size)
                
        rewards['pixel_accuracy'] = correct_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Other rewards...
        rewards['transformation'] = 0.7  # Simplified
        rewards['pattern'] = 0.5
        rewards['semantic'] = 0.5
        
        # Weighted sum
        total_reward = sum(
            rewards[key] * self.reward_weights[key] 
            for key in rewards
        )
        
        return total_reward
        
    def optimize(self, problem_id: str, initial_code: str, target_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize using majority voting over multiple candidates."""
        logger.info(f"Starting V12 Fixed optimization for problem {problem_id}")
        
        # Tokenize initial code
        inputs = self.tokenizer(initial_code, return_tensors="pt", padding=True).to(self.model.device)
        input_ids = inputs["input_ids"]
        
        # Find FULL description token range
        desc_start, desc_end = self.find_description_tokens_improved(input_ids)
        if desc_start is None:
            logger.error("Could not find description tokens")
            return {
                "success": False,
                "error": "Could not find description tokens",
                "final_code": initial_code
            }
            
        # Initialize
        best_accuracy = 0.0
        best_code = initial_code
        history = []
        
        for step in range(self.num_steps):
            # Warmup
            current_temp = self.temperature
            if step < self.warmup_steps:
                current_temp = self.temperature * (0.3 + 0.7 * step / self.warmup_steps)
                
            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=True
                )
                
            # Get hidden states for FULL description
            last_hidden = outputs.hidden_states[-1]
            desc_hidden = last_hidden[:, desc_start:desc_end, :].clone().detach().requires_grad_(True)
            
            # Create optimizer
            optimizer = torch.optim.Adam([desc_hidden], lr=self.lr)
            
            # Generate candidates
            candidates_data = []
            
            for cand_idx in range(self.num_candidates):
                # Get logits for description tokens
                desc_logits = self.model.lm_head(desc_hidden)
                
                # Sample tokens
                sampled_tokens, log_probs = self.sample_tokens(desc_logits[0], current_temp)
                
                # Create new token sequence
                current_tokens = input_ids.clone()
                current_tokens[0, desc_start:desc_end] = sampled_tokens
                
                # Generate from start of code (not from description end)
                # This ensures proper code generation
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=current_tokens,
                        max_new_tokens=1024,
                        temperature=current_temp,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                # Decode
                generated_code = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                
                # Log first candidate of first step
                if step == 0 and cand_idx == 0:
                    logger.info(f"First candidate code preview:")
                    logger.info(generated_code[:300] + "...")
                
                # Evaluate
                try:
                    exec_result = self.evaluator.evaluate_solution(problem_id, generated_code)
                    
                    candidates_data.append({
                        'tokens': sampled_tokens,
                        'log_probs': log_probs,
                        'code': generated_code,
                        'outputs': exec_result.get('generated_outputs'),
                        'accuracy': exec_result.get('accuracy', 0.0),
                        'success': exec_result.get('execution_success', False),
                        'error': exec_result.get('error')
                    })
                except Exception as e:
                    logger.debug(f"Candidate {cand_idx} evaluation error: {str(e)}")
                    candidates_data.append({
                        'tokens': sampled_tokens,
                        'log_probs': log_probs,
                        'code': generated_code,
                        'outputs': None,
                        'accuracy': 0.0,
                        'success': False,
                        'error': str(e)
                    })
                    
            # Log candidate results
            successful_count = sum(1 for c in candidates_data if c['success'])
            logger.info(f"Step {step+1}: {successful_count}/{self.num_candidates} successful candidates")
            
            if successful_count == 0:
                logger.warning(f"Step {step+1}: No successful candidates")
                # Log first error for debugging
                if candidates_data[0]['error']:
                    logger.debug(f"First candidate error: {candidates_data[0]['error'][:100]}")
                continue
                
            # Majority voting on outputs
            successful_outputs = [
                (str(c['outputs']), c['outputs']) 
                for c in candidates_data 
                if c['success'] and c['outputs'] is not None
            ]
            
            if successful_outputs:
                # Count output frequencies
                output_strings = [out[0] for out in successful_outputs]
                output_counter = Counter(output_strings)
                
                # Get most common output
                majority_output_str, count = output_counter.most_common(1)[0]
                confidence = count / len(candidates_data)
                
                # Find the actual output array
                majority_output = next(out[1] for out in successful_outputs if out[0] == majority_output_str)
                
                # Calculate 5D reward
                reward = self.calculate_5d_reward(majority_output, target_outputs)
                weighted_reward = reward * confidence
                
                logger.info(f"Step {step+1}: Majority voting - {count}/{self.num_candidates} agree, "
                          f"confidence: {confidence:.2f}, reward: {reward:.3f}")
                
                # Update gradients
                total_loss = 0
                update_count = 0
                
                for candidate in candidates_data:
                    if candidate['success'] and str(candidate['outputs']) == majority_output_str:
                        # Policy gradient loss
                        loss = -weighted_reward * candidate['log_probs'].sum()
                        total_loss += loss
                        update_count += 1
                        
                if update_count > 0:
                    avg_loss = total_loss / update_count
                    optimizer.zero_grad()
                    avg_loss.backward()
                    optimizer.step()
                    
                # Track best result
                best_candidate = max(
                    (c for c in candidates_data if c['success']),
                    key=lambda x: x['accuracy'],
                    default=None
                )
                
                if best_candidate and best_candidate['accuracy'] > best_accuracy:
                    best_accuracy = best_candidate['accuracy']
                    best_code = best_candidate['code']
                    logger.info(f"New best accuracy: {best_accuracy:.1%}")
                    
            # Record history
            history.append({
                'step': step + 1,
                'num_successful': successful_count,
                'best_accuracy': best_accuracy
            })
            
            # Early stopping
            if best_accuracy >= 1.0:
                logger.info("Perfect accuracy achieved!")
                break
                
        return {
            "success": True,
            "final_code": best_code,
            "final_accuracy": best_accuracy,
            "history": history,
            "num_steps": len(history)
        }