"""
V20: GLM-based visual evaluation + description optimization
- Use GLM-4V to evaluate visual similarity
- Optimize description tokens based on GLM reward
- Keep retry until valid mechanism
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
import re
import logging

from ..evaluators.glm_evaluator import GLMEvaluator
from ..executors.grid_renderer import GridRenderer
from ..evaluators.simple_evaluator import SimpleEvaluator

logger = logging.getLogger(__name__)

class LatentOptimizerGLMV20:
    def __init__(
        self, 
        model, 
        tokenizer,
        evaluator: GLMEvaluator,
        learning_rate: float = 0.03,
        num_steps: int = 20,
        temperature: float = 1.0,
        max_new_tokens: int = 1024,
        grad_clip: float = 1.0,
        max_retries: int = 10
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.glm_evaluator = evaluator
        self.simple_evaluator = SimpleEvaluator()  # For code execution
        self.renderer = GridRenderer()
        self.lr = learning_rate
        self.num_steps = num_steps
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.grad_clip = grad_clip
        self.max_retries = max_retries
        
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
            examples_text += f"Example {i+1}:\\n"
            examples_text += "Input:\\n"
            examples_text += self._grid_to_string(pair.x)
            examples_text += "\\n\\nOutput:\\n"
            examples_text += self._grid_to_string(pair.y)
            examples_text += "\\n\\n"
            
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
        return "\\n".join(lines)
        
    def calculate_glm_reward(self, generated_outputs: List[np.ndarray], target_outputs: List[np.ndarray]) -> float:
        """Calculate GLM visual similarity reward"""
        if not generated_outputs or generated_outputs[0] is None:
            return 0.0
            
        # Render grids
        generated_images = []
        target_images = []
        
        for gen, target in zip(generated_outputs, target_outputs):
            if gen is not None:
                # Render grids to temporary files
                import tempfile
                import os
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as gen_tmp:
                    gen_img = self.renderer.render_simple_grid(gen, gen_tmp.name)
                    gen_tmp_path = gen_tmp.name
                    
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as target_tmp:
                    target_img = self.renderer.render_simple_grid(target, target_tmp.name)
                    target_tmp_path = target_tmp.name
                generated_images.append(gen_img)
                target_images.append(target_img)
                
                # Clean up temp files
                try:
                    os.unlink(gen_tmp_path)
                    os.unlink(target_tmp_path)
                except:
                    pass
                
        if not generated_images:
            return 0.0
            
        # Get GLM evaluation
        glm_result = self.glm_evaluator.evaluate(generated_images, target_images)
        
        # Extract similarity score
        similarity = glm_result.get('visual_similarity', 0.0)
        
        # Log GLM feedback
        logger.debug(f"GLM similarity: {similarity}")
        logger.debug(f"GLM reasoning: {glm_result.get('reasoning', 'N/A')}")
        
        return similarity
        
    def generate_until_valid(self, prompt: str, problem_id: str, target_outputs: List[np.ndarray]) -> Tuple[torch.Tensor, str, float]:
        """Keep generating until we get valid output"""
        logger.info("Generating until valid output...")
        
        for retry in range(self.max_retries):
            # Tokenize prompt
            prompt_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                generated = self.model.generate(
                    **prompt_inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                
            # Decode and extract code
            full_response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
            generated_text = full_response[len(prompt):]
            code = self.extract_code_from_response(generated_text)
            
            if not code:
                logger.warning(f"Retry {retry+1}: No code found")
                continue
                
            # Evaluate execution
            eval_result = self.simple_evaluator.evaluate_solution(problem_id, code)
            
            # If we got any valid output
            if eval_result.get("generated_outputs") and eval_result["generated_outputs"][0] is not None:
                # Calculate GLM reward
                glm_reward = self.calculate_glm_reward(
                    eval_result["generated_outputs"],
                    target_outputs
                )
                logger.info(f"Valid output found on retry {retry+1} with GLM reward: {glm_reward:.3f}")
                return generated[0], code, glm_reward
                
            # Log failure
            error_msg = eval_result.get('error', 'Unknown error')
            logger.warning(f"Retry {retry+1}: Code execution failed - {error_msg}")
            
        # If all retries failed, return the last attempt
        logger.warning(f"All {self.max_retries} retries failed, using last attempt")
        return generated[0], code, 0.0
        
    def extract_hidden_states_per_token(self, input_ids: torch.Tensor) -> List[torch.Tensor]:
        """Extract hidden states for each token individually"""
        hidden_states_list = []
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids,
                output_hidden_states=True
            )
            last_hidden_states = outputs.hidden_states[-1]
            
            for i in range(last_hidden_states.size(1)):
                hidden_states_list.append(last_hidden_states[0, i, :].clone())
                
        return hidden_states_list
        
    def find_description_range(self, generated_text: str, token_ids: List[int]) -> Tuple[int, int]:
        """Find the token range for description section"""
        desc_pattern = r'#\\s*description:\\s*\\n((?:#[^\\n]*\\n)*)'
        match = re.search(desc_pattern, generated_text)
        
        if not match:
            logger.warning("No description found in generated code")
            return None, None
            
        desc_start_char = match.start()
        desc_end_char = match.end()
        
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
        # First try to find code blocks
        code_pattern = r'```python(.*?)```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            # Get the last code block and ensure it's complete
            code = matches[-1].strip()
            # If code seems truncated (doesn't end properly), try to complete it
            if code and not code.rstrip().endswith(('\\n', '}', ')', 'return')):
                # Look for any additional code after the block
                remaining = response.split('```')[-1]
                if remaining:
                    code += '\\n' + remaining.strip()
            return code
            
        # If no code blocks, try to extract from plain text
        lines = response.split('\\n')
        code_lines = []
        in_code = False
        imports_found = False
        
        for line in lines:
            # Look for imports to start code extraction
            if ('import' in line or 'from' in line) and not in_code:
                imports_found = True
                in_code = True
                code_lines.append(line)
            elif 'def main' in line or 'def transform' in line:
                if not imports_found:
                    # Add common imports if missing
                    code_lines.insert(0, 'from common import *')
                    code_lines.insert(1, 'import numpy as np')
                    code_lines.insert(2, 'from typing import *')
                    code_lines.insert(3, '')
                in_code = True
                code_lines.append(line)
            elif in_code:
                code_lines.append(line)
                
        if code_lines:
            return '\\n'.join(code_lines)
            
        return ""
        
    def optimize(self, problem_id: str, target_outputs: List[np.ndarray]) -> Dict[str, Any]:
        """Optimize with GLM-based evaluation"""
        logger.info(f"Starting V20 GLM optimization for problem {problem_id}")
        
        # Create BARC prompt
        prompt = self.create_barc_prompt(problem_id)
        self.current_prompt = prompt  # Store for later use
        prompt_length = len(self.tokenizer.encode(prompt))
        
        logger.info(f"Prompt length: {prompt_length} tokens")
        
        # Generate until we get valid output
        initial_ids, initial_code, initial_glm_reward = self.generate_until_valid(
            prompt, problem_id, target_outputs
        )
        
        logger.info(f"Initial GLM reward: {initial_glm_reward:.3f}")
        
        # Calculate initial accuracy
        eval_result = self.simple_evaluator.evaluate_solution(problem_id, initial_code)
        initial_accuracy = eval_result.get("accuracy", 0.0)
        
        # If already perfect, return
        if initial_accuracy >= 1.0:
            return {
                "success": True,
                "final_code": initial_code,
                "initial_code": initial_code,
                "final_accuracy": initial_accuracy,
                "initial_accuracy": initial_accuracy,
                "final_glm_reward": initial_glm_reward,
                "initial_glm_reward": initial_glm_reward,
                "history": [{"step": 0, "glm_reward": initial_glm_reward, "accuracy": initial_accuracy}],
                "num_steps": 0
            }
            
        # Extract hidden states
        hidden_states_list = self.extract_hidden_states_per_token(initial_ids)
        
        # Find description range
        generated_text = self.tokenizer.decode(initial_ids[prompt_length:], skip_special_tokens=True)
        generated_token_ids = initial_ids[prompt_length:].tolist()
        desc_start, desc_end = self.find_description_range(generated_text, generated_token_ids)
        
        if desc_start is None or desc_end is None:
            logger.error("Could not find description tokens")
            return {
                "success": True,
                "final_code": initial_code,
                "initial_code": initial_code,
                "final_accuracy": initial_accuracy,
                "initial_accuracy": initial_accuracy,
                "final_glm_reward": initial_glm_reward,
                "initial_glm_reward": initial_glm_reward,
                "history": [{"step": 0, "glm_reward": initial_glm_reward, "accuracy": initial_accuracy}],
                "num_steps": 0
            }
            
        # Adjust indices for full sequence
        desc_start += prompt_length
        desc_end += prompt_length
        
        logger.info(f"Description found at tokens [{desc_start}:{desc_end}] ({desc_end - desc_start} tokens)")
        
        # Create optimizable parameters
        update_length = desc_end - desc_start
        optimized_hidden_states = torch.nn.Parameter(torch.stack(
            [state.clone().detach().requires_grad_(True) 
             for state in hidden_states_list[desc_start:desc_end]]
        ))
        
        # Configure optimizer
        optimizer = torch.optim.Adam([optimized_hidden_states], lr=self.lr)
        
        # Get the sequence before description
        prefix_ids = initial_ids[:desc_start].clone()
        
        # Initialize history
        best_glm_reward = initial_glm_reward
        best_code = initial_code
        best_accuracy = initial_accuracy
        history = [{"step": 0, "glm_reward": initial_glm_reward, "accuracy": initial_accuracy}]
        
        # Optimization loop
        for step in range(self.num_steps):
            optimizer.zero_grad()
            
            # Get logits from optimized hidden states
            logits = self.model.lm_head(optimized_hidden_states)
            probs = torch.softmax(logits / self.temperature, dim=-1) + 1e-8
            
            # Sample tokens for policy gradient
            next_token_ids = torch.multinomial(probs, 1).squeeze(-1)
            log_pi = torch.log(probs[torch.arange(update_length), next_token_ids] + 1e-10)
            
            # Construct sequence up to description end
            updated_ids = torch.cat([
                prefix_ids,
                next_token_ids
            ])
            
            # Continue generation from description end point
            with torch.no_grad():
                current_ids = updated_ids.unsqueeze(0)
                
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
            generated_code = self.extract_code_from_response(full_text[len(self.current_prompt):])
            
            if not generated_code:
                continue
                
            # Evaluate generated code
            eval_result = self.simple_evaluator.evaluate_solution(problem_id, generated_code)
            
            # Calculate GLM reward if execution succeeded
            current_glm_reward = 0.0
            current_accuracy = 0.0
            
            if eval_result.get("generated_outputs") and eval_result["generated_outputs"][0] is not None:
                current_glm_reward = self.calculate_glm_reward(
                    eval_result["generated_outputs"],
                    target_outputs
                )
                current_accuracy = eval_result.get("accuracy", 0.0)
                
            # Calculate advantage using GLM reward
            reward_advantage = current_glm_reward - initial_glm_reward
            
            # Policy gradient update
            if step < self.num_steps - 1:
                loss = -reward_advantage * log_pi.sum()
                loss.backward()
                
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_([optimized_hidden_states], self.grad_clip)
                    
                optimizer.step()
                
            # Track best result
            if current_glm_reward > best_glm_reward or (current_glm_reward == best_glm_reward and current_accuracy > best_accuracy):
                best_glm_reward = current_glm_reward
                best_code = generated_code
                best_accuracy = current_accuracy
                logger.info(f"Step {step+1}: New best GLM reward: {best_glm_reward:.3f}, accuracy: {best_accuracy:.1%}")
                
            history.append({
                "step": step + 1,
                "glm_reward": current_glm_reward,
                "accuracy": current_accuracy
            })
            
            # Early stopping if perfect
            if current_accuracy >= 1.0:
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
            "initial_code": initial_code,
            "final_accuracy": best_accuracy,
            "initial_accuracy": initial_accuracy,
            "final_glm_reward": best_glm_reward,
            "initial_glm_reward": initial_glm_reward,
            "improvement": best_accuracy - initial_accuracy,
            "history": history,
            "num_steps": len(history) - 1,
            "description_tokens": update_length
        }