"""
V18: LatentSeek with 5D CompressARC-style multi-dimensional rewards
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import re
import logging

from ..evaluators.simple_evaluator import SimpleEvaluator

logger = logging.getLogger(__name__)

class LatentOptimizerV18MultiReward:
    def __init__(
        self, 
        model, 
        tokenizer,
        evaluator: SimpleEvaluator,
        learning_rate: float = 0.03,
        num_steps: int = 10,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        grad_clip: float = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.lr = learning_rate
        self.num_steps = num_steps
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.grad_clip = grad_clip
        
        # Move model to eval mode
        self.model.eval()
        
    def create_barc_prompt(self, problem_id: str) -> str:
        """Create BARC prompt following long_with_logit_reward2.py format"""
        import arc
        
        problem = None
        for p in arc.validation_problems:
            if p.uid == problem_id:
                problem = p
                break
                
        if problem is None:
            raise ValueError(f"Problem {problem_id} not found")
            
        # Format training examples
        examples_text = ""
        for i, pair in enumerate(problem.train_pairs):
            examples_text += f"Example {i+1}:\n"
            examples_text += "Input:\n"
            examples_text += self._grid_to_string(pair.x)
            examples_text += "\n\nOutput:\n"
            examples_text += self._grid_to_string(pair.y)
            examples_text += "\n\n"
            
        # Format test input
        test_input_text = self._grid_to_string(problem.test_pairs[0].x)
        
        # Create chat messages
        messages = [
            {
                "role": "system", 
                "content": "You are an world-class puzzle solver who are extremely good at spotting patterns and solving puzzles. You are also an expert Python programmer who can write code to solve puzzles."
            },
            {
                "role": "user",
                "content": f"""The following is a puzzle from the ARC dataset. Given training examples of input and output grids, predict the output grid for the test inputs.
Each grid is represented as a 2D array where each cell is represented by an color. The grid input and output are written as a string where each cell is separated by a space and each row is separated by a newline.
Here are the input and output grids for the training examples:
{examples_text}
Here are the input grids for the test example:
Input:
{test_input_text}"""
            }
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return prompt
        
    def _grid_to_string(self, grid: np.ndarray) -> str:
        """Convert grid to string representation"""
        lines = []
        for row in grid:
            lines.append(" ".join(str(cell) for cell in row))
        return "\n".join(lines)
        
    def calculate_5d_reward(self, generated_outputs: List[np.ndarray], target_outputs: List[np.ndarray]) -> Dict[str, float]:
        """Calculate 5-dimensional CompressARC-style rewards"""
        if not generated_outputs or generated_outputs[0] is None:
            return {
                "example_accuracy": 0.0,
                "color_transform": 0.0,
                "spatial_transform": 0.0,
                "pattern_recognition": 0.0,
                "structural_integrity": 0.0,
                "total_reward": 0.0
            }
            
        rewards = {}
        
        # 1. Example accuracy (exact match)
        rewards["example_accuracy"] = 0.0
        for gen, target in zip(generated_outputs, target_outputs):
            if gen is not None and np.array_equal(gen, target):
                rewards["example_accuracy"] = 1.0
                
        # For partial rewards when not exact match
        if rewards["example_accuracy"] == 0.0 and generated_outputs[0] is not None:
            gen = generated_outputs[0]
            target = target_outputs[0]
            
            # 2. Color transformation accuracy
            # Check if the color distribution is similar
            gen_colors = np.unique(gen)
            target_colors = np.unique(target)
            color_overlap = len(set(gen_colors) & set(target_colors)) / max(len(target_colors), 1)
            rewards["color_transform"] = color_overlap
            
            # 3. Spatial transformation accuracy
            # Check if shapes are in roughly correct positions
            spatial_score = 0.0
            if gen.shape == target.shape:
                # Calculate normalized distance between non-zero elements
                gen_nonzero = gen > 0
                target_nonzero = target > 0
                overlap = np.sum(gen_nonzero & target_nonzero)
                union = np.sum(gen_nonzero | target_nonzero)
                spatial_score = overlap / max(union, 1)
            rewards["spatial_transform"] = spatial_score
            
            # 4. Pattern recognition
            # Check if key patterns are preserved
            pattern_score = 0.0
            if gen.shape == target.shape:
                # Simple pattern: check if similar structures exist
                gen_patterns = self._extract_local_patterns(gen)
                target_patterns = self._extract_local_patterns(target)
                pattern_overlap = len(gen_patterns & target_patterns) / max(len(target_patterns), 1)
                pattern_score = pattern_overlap
            rewards["pattern_recognition"] = pattern_score
            
            # 5. Structural integrity
            # Check if the output maintains valid structure (no isolated pixels, etc.)
            structural_score = 1.0
            # Penalize if output is all zeros or all same color
            if len(np.unique(gen)) <= 1:
                structural_score = 0.0
            # Penalize if output has wrong dimensions
            elif gen.shape != target.shape:
                structural_score = 0.5
            rewards["structural_integrity"] = structural_score
            
        # Calculate weighted total reward
        weights = {
            "example_accuracy": 0.4,  # Highest weight for exact match
            "color_transform": 0.15,
            "spatial_transform": 0.15,
            "pattern_recognition": 0.15,
            "structural_integrity": 0.15
        }
        
        rewards["total_reward"] = sum(rewards[k] * weights[k] for k in weights)
        
        return rewards
        
    def _extract_local_patterns(self, grid: np.ndarray) -> set:
        """Extract 2x2 local patterns from grid"""
        patterns = set()
        h, w = grid.shape
        for i in range(h-1):
            for j in range(w-1):
                pattern = (grid[i,j], grid[i,j+1], grid[i+1,j], grid[i+1,j+1])
                patterns.add(pattern)
        return patterns
        
    def extract_hidden_states_per_token(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Extract hidden states for each token individually"""
        hidden_states_list = []
        
        # Process each token to get its hidden state
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True
            )
            # Get the last layer hidden states
            last_hidden_states = outputs.hidden_states[-1]
            
            # Extract each token's hidden state
            for i in range(last_hidden_states.size(1)):
                hidden_states_list.append(last_hidden_states[0, i, :].clone())
                
        return hidden_states_list
        
    def find_description_range(self, generated_text: str, token_ids: List[int]) -> Tuple[int, int]:
        """Find the token range for description section"""
        # Look for description pattern
        desc_pattern = r'#\s*description:\s*\n((?:#[^\n]*\n)*)'
        match = re.search(desc_pattern, generated_text)
        
        if not match:
            logger.warning("No description found in generated code")
            return None, None
            
        # Get character positions
        desc_start_char = match.start()
        desc_end_char = match.end()
        
        # Convert character positions to token positions
        current_pos = 0
        desc_start_tok = None
        desc_end_tok = None
        
        for i, token_id in enumerate(token_ids):
            token_text = self.tokenizer.decode([token_id])
            if desc_start_tok is None and current_pos + len(token_text) > desc_start_char:
                desc_start_tok = i
            if desc_end_tok is None and current_pos >= desc_end_char:
                desc_end_tok = i
                break
            current_pos += len(token_text)
            
        return desc_start_tok, desc_end_tok
        
    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from model response"""
        # Look for code blocks
        code_pattern = r'```python(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
            
        # If no code blocks, look for def main pattern
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if 'def main' in line or 'def transform' in line:
                in_code = True
            if in_code:
                code_lines.append(line)
                
        if code_lines:
            return '\n'.join(code_lines)
            
        return ""
        
    def optimize(self, problem_id: str, target_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize using 5D multi-reward approach"""
        logger.info(f"Starting V18 optimization with 5D rewards for problem {problem_id}")
        
        # Create BARC prompt
        prompt = self.create_barc_prompt(problem_id)
        
        # Tokenize prompt
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_inputs.input_ids.shape[1]
        
        logger.info(f"Prompt length: {prompt_length} tokens")
        
        # Generate initial response
        with torch.no_grad():
            generated = self.model.generate(
                **prompt_inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            
        # Extract hidden states for each token
        hidden_states_list = self.extract_hidden_states_per_token(generated)
        
        # Decode response
        full_response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        generated_text = full_response[len(prompt):]
        
        # Get token list for generated part
        generated_token_ids = generated[0][prompt_length:].tolist()
        
        # Find description range
        desc_start, desc_end = self.find_description_range(generated_text, generated_token_ids)
        
        if desc_start is None or desc_end is None:
            logger.error("Could not find description tokens")
            return {
                "success": False,
                "error": "Could not find description tokens",
                "final_code": "",
                "final_accuracy": 0.0
            }
            
        # Adjust indices for full sequence
        desc_start += prompt_length
        desc_end += prompt_length
        
        logger.info(f"Description found at tokens [{desc_start}:{desc_end}] ({desc_end - desc_start} tokens)")
        
        # Extract initial code
        initial_code = self.extract_code_from_response(generated_text)
        
        if not initial_code:
            logger.error("No code found in initial response")
            return {
                "success": False,
                "error": "No code found in initial response",
                "final_code": "",
                "final_accuracy": 0.0
            }
            
        # Evaluate initial solution
        eval_result = self.evaluator.evaluate_solution(problem_id, initial_code)
        initial_rewards = self.calculate_5d_reward(
            eval_result.get("generated_outputs", []),
            target_outputs
        )
        initial_total_reward = initial_rewards["total_reward"]
        
        logger.info(f"Initial rewards: {initial_rewards}")
        
        # If already perfect, return
        if initial_rewards["example_accuracy"] >= 1.0:
            return {
                "success": True,
                "final_code": initial_code,
                "final_accuracy": initial_rewards["example_accuracy"],
                "final_rewards": initial_rewards,
                "history": [{"step": 0, "rewards": initial_rewards}],
                "num_steps": 0
            }
            
        # Create optimizable parameters from description hidden states
        update_length = desc_end - desc_start
        optimized_hidden_states = torch.nn.Parameter(torch.stack(
            [state.clone().detach().requires_grad_(True) 
             for state in hidden_states_list[desc_start:desc_end]]
        ))
        
        # Configure optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Get the sequence before description
        prefix_ids = generated[0][:desc_start].clone()
        
        # Initialize history
        best_total_reward = initial_total_reward
        best_code = initial_code
        best_rewards = initial_rewards
        history = [{"step": 0, "rewards": initial_rewards}]
        
        # Optimization loop
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # Get logits from optimized hidden states
            logits = self.model.lm_head(optimized_hidden_states)  # [update_length, vocab_size]
            probs = torch.softmax(logits / self.temperature, dim=-1) + 1e-8
            
            # Sample tokens for policy gradient
            next_token_ids = torch.multinomial(probs, 1).squeeze(-1)  # [update_length]
            log_pi = torch.log(probs[torch.arange(update_length), next_token_ids] + 1e-10)
            
            # Construct sequence up to description end
            updated_ids = torch.cat([
                prefix_ids,
                next_token_ids
            ])
            
            # Continue generation from description end point
            with torch.no_grad():
                # Start from the updated sequence
                current_ids = updated_ids.unsqueeze(0)
                
                # Generate remaining tokens
                remaining_generated = self.model.generate(
                    input_ids=current_ids,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
            # Decode complete sequence
            full_text = self.tokenizer.decode(remaining_generated[0], skip_special_tokens=True)
            
            # Extract code from the part after prompt
            generated_code = self.extract_code_from_response(full_text[len(prompt):])
            
            if not generated_code:
                continue
                
            # Evaluate generated code
            eval_result = self.evaluator.evaluate_solution(problem_id, generated_code)
            current_rewards = self.calculate_5d_reward(
                eval_result.get("generated_outputs", []),
                target_outputs
            )
            
            # Calculate advantage (improvement over baseline)
            reward_advantage = current_rewards["total_reward"] - initial_total_reward
            
            # Policy gradient update
            if step < self.num_steps - 1:  # Don't update on last step
                loss = -reward_advantage * log_pi.sum()
                loss.backward()
                
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_([optimized_hidden_states], self.grad_clip)
                    
                optimizer.step()
                
            # Track best result
            if current_rewards["total_reward"] > best_total_reward:
                best_total_reward = current_rewards["total_reward"]
                best_code = generated_code
                best_rewards = current_rewards
                logger.info(f"Step {step+1}: New best total reward: {best_total_reward:.3f}")
                logger.info(f"  Rewards breakdown: {current_rewards}")
                
            history.append({
                "step": step + 1,
                "rewards": current_rewards
            })
            
            # Early stopping if perfect
            if current_rewards["example_accuracy"] >= 1.0:
                logger.info("Perfect accuracy achieved!")
                break
                
            # Log sample of optimized description
            if step % 5 == 0:
                desc_tokens = next_token_ids.tolist()
                desc_text = self.tokenizer.decode(desc_tokens)
                logger.info(f"Step {step} description preview: {desc_text[:100]}...")
                
        return {
            "success": True,
            "final_code": best_code,
            "final_accuracy": best_rewards["example_accuracy"],
            "final_rewards": best_rewards,
            "initial_accuracy": initial_rewards["example_accuracy"],
            "initial_rewards": initial_rewards,
            "improvement": best_rewards["example_accuracy"] - initial_rewards["example_accuracy"],
            "history": history,
            "num_steps": len(history) - 1,
            "description_tokens": update_length
        }