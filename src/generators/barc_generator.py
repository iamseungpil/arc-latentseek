"""
BARC Model Wrapper for Code Generation
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


class BARCGenerator:
    """Wrapper for BARC model to generate code for ARC problems"""
    
    def __init__(self, model_name: str = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load BARC model and tokenizer"""
        print(f"Loading BARC model: {self.model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Set padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.eval()
        print("BARC model loaded successfully")
        
    def _grid_to_string(self, grid: np.ndarray) -> str:
        """Convert numeric grid to color string format"""
        rows = []
        for row in grid:
            color_row = [COLOR_MAP.get(int(cell), f"Unknown_{cell}") for cell in row]
            rows.append(" ".join(color_row))
        return "\n".join(rows) + "\n"
    
    def _generate_examples_text(self, train_pairs) -> str:
        """Generate examples text for prompt"""
        text = ""
        for i, pair in enumerate(train_pairs, 1):
            text += f"Example {i}:\n"
            text += f"Input:\n{self._grid_to_string(pair.x)}\n"
            text += f"Output:\n{self._grid_to_string(pair.y)}\n\n"
        return text.strip()
    
    def _create_prompt(self, problem: ARCProblem, test_idx: int = 0) -> List[Dict[str, str]]:
        """Create BARC-style prompt"""
        # System prompt
        system_content = "You are an world-class puzzle solver who are extremely good at spotting patterns and solving puzzles. You are also an expert Python programmer who can write code to solve puzzles."
        
        # User prompt
        test_input = problem.test_pairs[test_idx].x
        user_content = f"""The following is a puzzle from the ARC dataset. Given training examples of input and output grids, predict the output grid for the test inputs.
            Each grid is represented as a 2D array where each cell is represented by an color. The grid input and output are written as a string where each cell is separated by a space and each row is separated by a newline.
            Here are the input and output grids for the training examples:
            {self._generate_examples_text(problem.train_pairs)}

            Here are the input grids for the test example:
            Input:
            {self._grid_to_string(test_input)}"""
            
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    def generate(self, problem: ARCProblem, 
                temperature: float = 0.8,
                max_new_tokens: int = 2048,
                num_candidates: int = 1) -> List[BARCOutput]:
        """
        Generate code solutions for an ARC problem
        
        Args:
            problem: ARC problem to solve
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
            num_candidates: Number of candidate solutions to generate
            
        Returns:
            List of BARCOutput objects
        """
        # Create prompt
        prompt = self._create_prompt(problem)
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            prompt, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        outputs = []
        
        for _ in range(num_candidates):
            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    temperature=temperature,
                    top_p=0.95,
                    top_k=50,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            input_len = inputs.input_ids.shape[1]
            response = self.tokenizer.decode(
                output[0][input_len:], 
                skip_special_tokens=True
            )
            
            # Parse code and extract elements
            code_blocks = parse_code(response)
            code = code_blocks[0] if code_blocks else ""
            
            concepts, description, plan = extract_code_elements(code)
            
            outputs.append(BARCOutput(
                code=code,
                concepts=concepts,
                description=description,
                plan=plan,
                raw_response=response
            ))
            
        return outputs
    
    def get_hidden_states(self, problem: ARCProblem, output: BARCOutput) -> torch.Tensor:
        """
        Get hidden states for a generated output (for LatentSeek optimization)
        
        This is a placeholder - actual implementation would need to capture
        hidden states during generation
        """
        # TODO: Implement actual hidden state extraction
        raise NotImplementedError("Hidden state extraction not yet implemented")