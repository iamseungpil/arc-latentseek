"""
V13: Proper description finding with multi-line support
- Correctly finds entire multi-line description (not just 7 tokens)
- Uses majority voting over 8 candidates
- Implements proper policy gradient with sampling
- 5D reward calculation
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

class LatentOptimizerV13ProperDesc:
    def __init__(
        self, 
        model, 
        tokenizer,
        evaluator: SimpleEvaluator,
        num_candidates: int = 8,
        learning_rate: float = 0.001,
        num_steps: int = 30,
        warmup_steps: int = 5,
        temperature: float = 0.7,
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
        
    def find_description_tokens_properly(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """Find the FULL multi-line description token range."""
        # Decode full text first
        text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Find description section in text
        lines = text.split('\n')
        desc_line_start = None
        desc_line_end = None
        
        for i, line in enumerate(lines):
            if '# description:' in line:
                desc_line_start = i
                # Find where description ends - look for first non-comment line or empty line before def
                for j in range(i + 1, len(lines)):
                    # If we hit 'def main', description ended
                    if 'def main' in lines[j]:
                        desc_line_end = j
                        break
                    # If we hit an empty line followed by non-comment, description ended
                    if j > i + 1 and lines[j].strip() == '' and j + 1 < len(lines):
                        if not lines[j + 1].strip().startswith('#'):
                            desc_line_end = j
                            break
                break
        
        if desc_line_start is None:
            logger.warning("Could not find description in text")
            return None, None
            
        if desc_line_end is None:
            # Fallback: assume description is at most 20 lines
            desc_line_end = min(desc_line_start + 20, len(lines))
            
        logger.info(f"Description found from line {desc_line_start} to {desc_line_end-1} ({desc_line_end - desc_line_start} lines)")
        
        # Now find token positions
        # Build character position mapping
        char_positions = [0]
        current_text = ""
        tokens = []
        
        for i in range(input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
            tokens.append(token_text)
            current_text += token_text
            char_positions.append(len(current_text))
            
        # Find character positions of description lines
        desc_text = '\n'.join(lines[desc_line_start:desc_line_end])
        desc_start_char = text.find(desc_text)
        desc_end_char = desc_start_char + len(desc_text)
        
        # Map to token positions
        desc_start_token = None
        desc_end_token = None
        
        for i in range(len(tokens)):
            if desc_start_token is None and char_positions[i] >= desc_start_char:
                desc_start_token = i
            if desc_end_token is None and char_positions[i] >= desc_end_char:
                desc_end_token = i
                break
                
        if desc_end_token is None:
            desc_end_token = len(tokens)
            
        # Log what we found
        logger.info(f"Description spans tokens [{desc_start_token}:{desc_end_token}] ({desc_end_token - desc_start_token} tokens)")
        
        # Show preview
        if desc_start_token is not None and desc_end_token is not None:
            preview_tokens = tokens[desc_start_token:min(desc_start_token + 10, desc_end_token)]
            logger.debug(f"First 10 description tokens: {preview_tokens}")
            
        return desc_start_token, desc_end_token
        
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
        
        # Transformation understanding
        if len(generated_outputs) > 1:
            shape_consistency = all(
                gen.shape == generated_outputs[0].shape 
                for gen in generated_outputs[1:]
            )
            rewards['transformation'] = 1.0 if shape_consistency else 0.5
        else:
            rewards['transformation'] = 0.7
            
        # Pattern recognition
        unique_colors_match = 0
        for gen, target in zip(generated_outputs, target_outputs):
            gen_colors = set(gen.flatten())
            target_colors = set(target.flatten())
            if gen_colors == target_colors:
                unique_colors_match += 1
        rewards['pattern'] = unique_colors_match / len(generated_outputs)
        
        # Semantic coherence
        semantic_score = 0
        for gen in generated_outputs:
            unique_colors = len(set(gen.flatten()))
            if 1 < unique_colors < gen.size * 0.8:
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
        logger.info(f"Starting V13 optimization for problem {problem_id}")
        
        # Tokenize initial code
        inputs = self.tokenizer(initial_code, return_tensors="pt", padding=True).to(self.model.device)
        input_ids = inputs["input_ids"]
        
        # Find FULL description token range
        desc_start, desc_end = self.find_description_tokens_properly(input_ids)
        if desc_start is None:
            logger.error("Could not find description tokens")
            return {
                "success": False,
                "error": "Could not find description tokens",
                "final_code": initial_code,
                "final_accuracy": 0.0
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
                new_tokens = input_ids.clone()
                new_tokens[0, desc_start:desc_end] = sampled_tokens
                
                # Generate complete code from beginning
                with torch.no_grad():
                    generated = self.model.generate(
                        input_ids=new_tokens,
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
                    preview = generated_code[:500]
                    logger.info(preview + "..." if len(generated_code) > 500 else preview)
                
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
                if candidates_data[0]['error']:
                    logger.debug(f"First candidate error: {candidates_data[0]['error'][:200]}")
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