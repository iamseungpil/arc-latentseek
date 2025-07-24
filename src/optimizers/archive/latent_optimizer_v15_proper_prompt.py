"""
V15: Proper BARC prompt format with LatentSeek optimization
- Uses correct chat template format like long_with_logit_reward2.py
- Maintains full conversation context (system + user + assistant)
- Optimizes only the generated code portion (after prompt)
- Following original LatentSeek approach with 20% token optimization
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import re
import logging

from ..evaluators.simple_evaluator import SimpleEvaluator

logger = logging.getLogger(__name__)

class LatentOptimizerV15ProperPrompt:
    def __init__(
        self, 
        model, 
        tokenizer,
        evaluator: SimpleEvaluator,
        k: float = 0.2,  # Optimize 20% of generated tokens
        learning_rate: float = 0.01,
        num_steps: int = 50,
        temperature: float = 1.0,
        max_new_tokens: int = 1024
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.evaluator = evaluator
        self.k = k
        self.lr = learning_rate
        self.num_steps = num_steps
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        # Move model to eval mode
        self.model.eval()
        
    def create_barc_prompt(self, problem_id: str) -> str:
        """Create BARC prompt following long_with_logit_reward2.py format"""
        # Get problem from arc
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
        
    def extract_code_from_response(self, response: str) -> str:
        """Extract Python code from model response"""
        # Look for code blocks
        code_pattern = r'```python(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[-1].strip()  # Return last code block
            
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
        
    def calculate_reward(self, generated_outputs: List[np.ndarray], target_outputs: List[np.ndarray]) -> float:
        """Calculate reward based on accuracy"""
        if not generated_outputs:
            return 0.0
            
        correct = 0
        total = len(target_outputs)
        
        for gen, target in zip(generated_outputs, target_outputs):
            if gen is not None and np.array_equal(gen, target):
                correct += 1
                
        return correct / total
        
    def optimize(self, problem_id: str, target_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize using proper BARC prompt format and LatentSeek approach"""
        logger.info(f"Starting V15 optimization for problem {problem_id}")
        
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
            
        # Decode full response
        full_response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Extract code from response
        initial_code = self.extract_code_from_response(full_response[len(prompt):])
        
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
        initial_accuracy = eval_result.get("accuracy", 0.0)
        
        logger.info(f"Initial accuracy: {initial_accuracy:.1%}")
        
        # If already perfect, return
        if initial_accuracy >= 1.0:
            return {
                "success": True,
                "final_code": initial_code,
                "final_accuracy": initial_accuracy,
                "history": [{"step": 0, "accuracy": initial_accuracy}],
                "num_steps": 0
            }
            
        # Calculate number of tokens to optimize
        generated_length = generated[0].shape[0] - prompt_length
        update_length = min(int(self.k * generated_length), 300)  # Cap at 300 tokens
        
        logger.info(f"Generated {generated_length} tokens, optimizing {update_length} tokens")
        
        # Initialize optimization
        best_accuracy = initial_accuracy
        best_code = initial_code
        history = [{"step": 0, "accuracy": initial_accuracy}]
        
        # Current sequence for optimization
        current_ids = generated[0].clone()
        
        for step in range(self.num_steps):
            # Forward pass to get hidden states
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_ids.unsqueeze(0),
                    output_hidden_states=True
                )
                
            # Get hidden states for tokens to optimize
            last_hidden = outputs.hidden_states[-1]
            # Optimize the last `update_length` tokens before EOS
            optimize_start = -update_length - 1  # -1 for EOS token
            optimize_hidden = last_hidden[:, optimize_start:-1, :].clone().detach().requires_grad_(True)
            
            # Create optimizer
            optimizer = torch.optim.Adam([optimize_hidden], lr=self.lr)
            
            # Forward through lm_head to get logits
            logits = self.model.lm_head(optimize_hidden)
            
            # Sample new tokens
            probs = F.softmax(logits / self.temperature, dim=-1)
            sampled_tokens = torch.multinomial(probs.view(-1, probs.size(-1)), 1).squeeze(-1)
            sampled_tokens = sampled_tokens.view(logits.shape[:-1])
            
            # Calculate log probabilities for policy gradient
            log_probs = torch.log(probs[0, torch.arange(update_length), sampled_tokens[0]] + 1e-10)
            
            # Update sequence with sampled tokens
            new_ids = current_ids.clone()
            new_ids[optimize_start:-1] = sampled_tokens[0]
            
            # Generate complete response with updated tokens
            with torch.no_grad():
                # Continue generation from the updated sequence
                continued = self.model.generate(
                    input_ids=new_ids.unsqueeze(0),
                    max_new_tokens=100,  # Generate a bit more if needed
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
            # Decode and extract code
            full_response = self.tokenizer.decode(continued[0], skip_special_tokens=True)
            generated_code = self.extract_code_from_response(full_response[len(prompt):])
            
            if not generated_code:
                continue
                
            # Evaluate
            eval_result = self.evaluator.evaluate_solution(problem_id, generated_code)
            accuracy = eval_result.get("accuracy", 0.0)
            
            # Calculate reward
            reward = self.calculate_reward(
                eval_result.get("generated_outputs", []),
                target_outputs
            )
            
            # Policy gradient update
            if reward > 0:
                loss = -reward * log_probs.sum()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            # Track best result
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_code = generated_code
                current_ids = continued[0].clone()  # Update current sequence
                logger.info(f"Step {step+1}: New best accuracy: {best_accuracy:.1%}")
                
            history.append({
                "step": step + 1,
                "accuracy": accuracy,
                "reward": reward
            })
            
            # Early stopping
            if best_accuracy >= 1.0:
                logger.info("Perfect accuracy achieved!")
                break
                
        return {
            "success": True,
            "final_code": best_code,
            "final_accuracy": best_accuracy,
            "initial_accuracy": initial_accuracy,
            "improvement": best_accuracy - initial_accuracy,
            "history": history,
            "num_steps": len(history) - 1
        }