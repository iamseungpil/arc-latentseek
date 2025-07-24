"""
V12 Latent Optimizer with Majority Voting
- Uses proper policy gradient with sampling (not argmax)
- Implements 8 candidate inference with majority voting
- 5D reward calculation based on most frequent output
- Confidence-weighted rewards based on voting consensus
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

class LatentOptimizerV12Majority:
    def __init__(
        self, 
        model, 
        tokenizer,
        evaluator: SimpleEvaluator,
        num_candidates: int = 8,
        learning_rate: float = 0.01,
        num_steps: int = 50,
        warmup_steps: int = 5,
        update_percentage: float = 0.15,  # Target description portion
        temperature: float = 1.0,
        reward_weights: Dict[str, float] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.num_candidates = num_candidates
        self.lr = learning_rate
        self.num_steps = num_steps
        self.warmup_steps = warmup_steps
        self.update_percentage = update_percentage
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
        
    def find_description_tokens(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Find the token range for description in the generated code."""
        # Decode to find description boundaries
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Log the full text for debugging
        logger.debug(f"Full text length: {len(text)} characters")
        logger.debug(f"First 500 chars: {text[:500]}...")
        
        # Find description pattern - match multi-line descriptions
        # Pattern matches: # description:\n followed by multiple # lines
        desc_pattern = r'# description:\s*\n((?:# [^\n]+\n)+)'
        match = re.search(desc_pattern, text)
        
        if not match:
            logger.warning("Could not find description in generated text")
            logger.debug(f"Text sample: {text[200:400]}")
            return None, None
            
        # Get byte positions
        desc_start = match.start(1)
        desc_end = match.end(1)
        
        # Log what was found
        desc_text = match.group(1)
        logger.debug(f"Found description at byte positions {desc_start}-{desc_end}")
        logger.debug(f"Description text: '{desc_text}'")
        
        # Convert byte positions to token indices
        # Create cumulative byte position mapping
        tokens = []
        byte_positions = [0]
        current_pos = 0
        
        for i, token_id in enumerate(input_ids[0]):
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            tokens.append(token_text)
            current_pos += len(token_text.encode('utf-8'))
            byte_positions.append(current_pos)
        
        # Find token indices
        start_token = None
        end_token = None
        
        for i in range(len(tokens)):
            if start_token is None and byte_positions[i] >= desc_start:
                start_token = i
            if end_token is None and byte_positions[i] >= desc_end:
                end_token = i
                break
        
        # Log token mapping
        logger.debug(f"Token range: [{start_token}:{end_token}]")
        if start_token is not None and end_token is not None:
            desc_tokens = [tokens[i] for i in range(start_token, min(end_token, len(tokens)))]
            logger.debug(f"Description tokens: {desc_tokens[:10]}...")
                
        return start_token, end_token
        
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
        
        # Transformation understanding (simplified)
        # Check if outputs maintain consistent transformation
        if len(generated_outputs) > 1:
            shape_consistency = all(
                gen.shape == generated_outputs[0].shape 
                for gen in generated_outputs[1:]
            )
            rewards['transformation'] = 1.0 if shape_consistency else 0.5
        else:
            rewards['transformation'] = 0.7
            
        # Pattern recognition (check for color patterns)
        unique_colors_match = 0
        for gen, target in zip(generated_outputs, target_outputs):
            gen_colors = set(gen.flatten())
            target_colors = set(target.flatten())
            if gen_colors == target_colors:
                unique_colors_match += 1
        rewards['pattern'] = unique_colors_match / len(generated_outputs)
        
        # Semantic coherence (non-random output)
        # Check if output has structure (not all same color, not random)
        semantic_score = 0
        for gen in generated_outputs:
            unique_colors = len(set(gen.flatten()))
            if 1 < unique_colors < gen.size * 0.8:  # Not uniform, not too random
                semantic_score += 1
        rewards['semantic'] = semantic_score / len(generated_outputs)
        
        # Weighted sum
        total_reward = sum(
            rewards[key] * self.reward_weights[key] 
            for key in rewards
        )
        
        return total_reward
        
    def optimize(self, problem_id: str, initial_code: str, target_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize using majority voting over multiple candidates."""
        logger.info(f"Starting V12 Majority Voting optimization for problem {problem_id}")
        
        # Tokenize initial code
        inputs = self.tokenizer(initial_code, return_tensors="pt", padding=True).to(self.model.device)
        input_ids = inputs["input_ids"]
        
        # Find description token range
        desc_start, desc_end = self.find_description_tokens(input_ids)
        if desc_start is None:
            logger.error("Could not find description tokens")
            return {
                "success": False,
                "error": "Could not find description tokens",
                "final_code": initial_code
            }
            
        logger.info(f"Description found at tokens [{desc_start}:{desc_end}] (length: {desc_end - desc_start})")
        
        # Initialize optimizer for hidden states
        best_accuracy = 0.0
        best_code = initial_code
        history = []
        
        for step in range(self.num_steps):
            # Warmup: reduce temperature
            current_temp = self.temperature
            if step < self.warmup_steps:
                current_temp = self.temperature * (0.5 + 0.5 * step / self.warmup_steps)
                
            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=True
                )
                
            # Get hidden states for description tokens
            last_hidden = outputs.hidden_states[-1]
            desc_hidden = last_hidden[:, desc_start:desc_end, :].clone().detach().requires_grad_(True)
            
            # Create optimizer for this step
            optimizer = torch.optim.Adam([desc_hidden], lr=self.lr)
            
            # Generate multiple candidates
            candidates_data = []
            
            for _ in range(self.num_candidates):
                # Generate from description onwards with gradients
                position_ids = torch.arange(desc_start, desc_end).unsqueeze(0).to(self.model.device)
                
                # Get logits for description tokens
                desc_logits = self.model.lm_head(desc_hidden)
                
                # Sample tokens and get log probabilities
                sampled_tokens, log_probs = self.sample_tokens(desc_logits[0], current_temp)
                
                # Generate rest of code autoregressively
                current_tokens = input_ids.clone()
                current_tokens[0, desc_start:desc_end] = sampled_tokens
                
                # Continue generation from description end
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=current_tokens[:, :desc_end],
                        max_new_tokens=1024,
                        temperature=current_temp,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )
                    
                # Decode and parse
                generated_code = self.tokenizer.decode(generated[0], skip_special_tokens=True)
                
                # Log generated code for first candidate of first step
                if step == 0 and _ == 0:
                    logger.debug(f"First candidate generated code (first 500 chars):")
                    logger.debug(generated_code[:500])
                
                # Extract function and evaluate
                func_match = re.search(r'def main\(.*?\):(.*?)(?=\n(?:def|$))', generated_code, re.DOTALL)
                if func_match:
                    # Execute code
                    exec_result = self.evaluator.evaluate_solution(problem_id, generated_code)
                    
                    candidates_data.append({
                        'tokens': sampled_tokens,
                        'log_probs': log_probs,
                        'code': generated_code,
                        'outputs': exec_result.get('generated_outputs'),
                        'accuracy': exec_result.get('accuracy', 0.0),
                        'success': exec_result.get('execution_success', False)
                    })
                else:
                    candidates_data.append({
                        'tokens': sampled_tokens,
                        'log_probs': log_probs,
                        'code': generated_code,
                        'outputs': None,
                        'accuracy': 0.0,
                        'success': False
                    })
                    
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
                
                # Calculate 5D reward for majority output
                reward = self.calculate_5d_reward(majority_output, target_outputs)
                weighted_reward = reward * confidence
                
                logger.info(f"Step {step+1}: Majority voting - {count}/{self.num_candidates} agree, "
                          f"confidence: {confidence:.2f}, reward: {reward:.3f}")
                
                # Update gradients for candidates that produced majority output
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
                    
                    logger.info(f"Updated {update_count} candidates, loss: {avg_loss.item():.4f}")
                    
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
                    
            else:
                logger.warning(f"Step {step+1}: No successful candidates")
                
            # Record history
            history.append({
                'step': step + 1,
                'num_successful': len(successful_outputs),
                'confidence': confidence if successful_outputs else 0.0,
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