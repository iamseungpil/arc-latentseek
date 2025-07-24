"""
Fixed BARC Model Wrapper without unsloth
Following ../LatentSeek and ../barc_post implementations
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re

from ..data import ARCProblem
from .code_parser import extract_code_elements, parse_code


# Color mapping from BARC
COLOR_MAP = {
    0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
    5: "Gray", 6: "Pink", 7: "Orange", 8: "Teal", 9: "Maroon"
}


@dataclass
class BARCOutput:
    """Output from BARC generator"""
    code: str
    concepts: Optional[str]
    description: Optional[str] 
    plan: Optional[str]
    raw_response: str
    
    def __repr__(self):
        desc_preview = self.description[:50] + "..." if self.description and len(self.description) > 50 else self.description
        return f"BARCOutput(description='{desc_preview}')"


class BARCGeneratorFixed:
    """Fixed BARC wrapper without unsloth, following ../LatentSeek implementation"""
    
    def __init__(self, model_name: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B", device: str = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load BARC model following ../LatentSeek style"""
        print(f"Loading BARC model: {self.model_name}")
        
        # Load model following ../LatentSeek/src/main.py
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device if self.device != "cuda" else "auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print("BARC model loaded successfully (fixed version)")
    
    def _grid_to_string(self, grid: np.ndarray) -> str:
        """Convert grid to string representation"""
        return '\n'.join([' '.join([COLOR_MAP.get(cell, str(cell)) for cell in row]) for row in grid])
    
    def _create_prompt(self, problem: ARCProblem) -> List[Dict[str, str]]:
        """Create prompt following ../barc_post/long_with_logit_reward2.py format"""
        
        # Format training examples
        examples_text = ""
        for i, pair in enumerate(problem.train_pairs, 1):
            examples_text += f"Example {i}:\nInput:\n{self._grid_to_string(pair.x)}\nOutput:\n{self._grid_to_string(pair.y)}\n\n"
        
        # System prompt from long_with_logit_reward2.py
        system_content = "You are an world-class puzzle solver who are extremely good at spotting patterns and solving puzzles. You are also an expert Python programmer who can write code to solve puzzles."
        
        # User prompt format (simplified like long_with_logit_reward2.py)
        user_content = f"""The following is a puzzle from the ARC dataset. Given training examples of input and output grids, predict the output grid for the test inputs.
Each grid is represented as a 2D array where each cell is represented by an color. The grid input and output are written as a string where each cell is separated by a space and each row is separated by a newline.
Here are the input and output grids for the training examples:
{examples_text.strip()}"""
            
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    def generate(self, 
                problem: ARCProblem, 
                temperature: float = 0.8,
                max_new_tokens: int = 2048,
                num_candidates: int = 1,
                additional_prompt: str = "") -> List[BARCOutput]:
        """
        Generate code solutions for an ARC problem
        """
        # Create prompt
        prompt = self._create_prompt(problem)
        
        # Add additional prompt if provided
        if additional_prompt:
            prompt[-1]["content"] += f"\n\n{additional_prompt}"
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        outputs = []
        
        # Generate candidates one by one for better quality
        for i in range(num_candidates):
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            response = self.tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )
            
            # Parse code and extract elements
            code_blocks = parse_code(response)
            code = code_blocks[0] if code_blocks else ""
            
            # If no code found, try to extract from the whole response
            if not code:
                if "def transform" in response:
                    # Extract everything from "def transform" to the end
                    start = response.find("def transform")
                    code = response[start:] if start != -1 else ""
                elif "def main" in response:
                    # Extract everything from "def main" to the end
                    start = response.find("def main")
                    code = response[start:] if start != -1 else ""
            
            concepts, description, plan = extract_code_elements(response)  # Extract from full response
            
            outputs.append(BARCOutput(
                code=code,
                concepts=concepts,
                description=description,
                plan=plan,
                raw_response=response
            ))
            
        return outputs
    
    def get_hidden_states(self, problem: ARCProblem, output: BARCOutput) -> List[torch.Tensor]:
        """
        Get hidden states for a generated output (for LatentSeek optimization)
        Returns a list of hidden states for each token
        """
        # Create the full sequence (prompt + generated code)
        prompt = self._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        
        # Add the generated response
        full_text = prompt_text + output.raw_response
        
        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        
        # Get hidden states
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
        
        # Extract hidden states from last layer
        last_hidden_states = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
        
        # Convert to list of tensors for each token
        hidden_states_list = []
        for i in range(last_hidden_states.size(1)):
            hidden_states_list.append(last_hidden_states[:, i, :])  # Keep batch dimension
        
        return hidden_states_list