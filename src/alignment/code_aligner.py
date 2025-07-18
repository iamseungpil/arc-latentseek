"""
BARC Code Aligner using Llama3.1-8B Instruct
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict, Any
import logging
import re
import time

from ..data import ARCProblem
from ..generators import BARCOutput

logger = logging.getLogger(__name__)


class BARCCodeAligner:
    """
    Aligns BARC-generated code using Llama3.1-8B Instruct model
    
    Improves code quality by:
    - Ensuring description matches implementation
    - Adding proper common.py imports and Color constants
    - Fixing bugs and improving robustness
    - Maintaining BARC structure (# concepts:, # description:, def transform)
    """
    
    def __init__(self, 
                 model_path: str = "meta-llama/Llama-3.1-8B-Instruct",
                 temperature: float = 0.3,
                 max_new_tokens: int = 2048):
        """
        Initialize the code aligner
        
        Args:
            model_path: Path to Llama3.1-8B Instruct model
            temperature: Generation temperature (lower = more deterministic)
            max_new_tokens: Maximum tokens to generate
        """
        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        
        self.tokenizer = None
        self.model = None
        
        self._load_model()
        
    def _load_model(self):
        """Load Llama3.1-8B Instruct model"""
        try:
            logger.info(f"Loading alignment model: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else None
            )
            
            # Set padding token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info("Alignment model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load alignment model: {e}")
            raise
    
    def align_code(self, barc_output: BARCOutput, problem: ARCProblem) -> BARCOutput:
        """
        Align BARC code to improve quality and consistency
        
        Args:
            barc_output: Original BARC output
            problem: ARC problem context
            
        Returns:
            Improved BARCOutput with aligned code
        """
        start_time = time.time()
        
        # Create alignment prompt
        prompt = self._create_alignment_prompt(barc_output, problem)
        
        # Generate aligned code
        aligned_response = self._generate_aligned_code(prompt)
        
        # Parse aligned response
        aligned_output = self._parse_aligned_response(aligned_response, barc_output)
        
        alignment_time = time.time() - start_time
        
        logger.info(f"Code alignment completed in {alignment_time:.2f}s")
        logger.info(f"Original code length: {len(barc_output.code)}")
        logger.info(f"Aligned code length: {len(aligned_output.code)}")
        
        return aligned_output
    
    def _create_alignment_prompt(self, barc_output: BARCOutput, problem: ARCProblem) -> str:
        """Create alignment prompt based on barc_post style"""
        
        # Extract ARC problem description for context
        problem_context = self._format_problem_context(problem)
        
        prompt = f"""<|start_header_id|>system<|end_header_id|>
You are an expert ARC puzzle code reviewer and optimizer. Your task is to improve BARC-generated code while preserving its core structure and intent.

**Critical Requirements:**

1. **Preserve BARC Structure**: Always maintain the exact format:
   - `# concepts: [list of concepts]`
   - `# description: [clear description]` 
   - `def transform(input_grid):`

2. **Common.py Integration**: Use common.py utilities where appropriate:
   - `from common import *` at the top
   - Use functions like `find_connected_components`, `blit_sprite`, `flood_fill`, etc.
   - Prefer common.py functions over manual implementations

3. **Color Constants**: Use Color class constants instead of numbers:
   - `Color.BLACK` instead of `0`
   - `Color.BLUE` instead of `1`
   - `Color.RED` instead of `2`, etc.

4. **Pattern-Code Alignment**: Ensure the description accurately matches the implementation
   - If description says "fill rectangles", code should actually fill rectangles
   - If description mentions "connected components", code should use find_connected_components

5. **Code Quality**: Fix bugs and improve robustness:
   - Add proper bounds checking
   - Handle edge cases
   - Ensure the function returns a valid grid
   - Add error handling where appropriate

6. **ARC-Specific Best Practices**:
   - Work with numpy arrays efficiently
   - Preserve grid dimensions unless explicitly changing them
   - Use appropriate connectivity (4-way vs 8-way)
   - Handle background colors properly

<|eot_id|><|start_header_id|>user<|end_header_id|>
**ARC Problem Context:**
{problem_context}

**Original BARC Code:**
```python
{barc_output.code}
```

**Task:** Improve this code following all the requirements above. The improved code should be more robust, use appropriate common.py utilities, and ensure the description accurately matches the implementation.

**Improved Code:**<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```python
from common import *

"""
        
        return prompt.strip()
    
    def _format_problem_context(self, problem: ARCProblem) -> str:
        """Format ARC problem context for the prompt"""
        context = f"Problem ID: {problem.uid}\n"
        context += f"Training examples: {len(problem.train_pairs)}\n"
        
        if problem.train_pairs:
            first_pair = problem.train_pairs[0]
            context += f"Example input shape: {first_pair.x.shape}\n"
            context += f"Example output shape: {first_pair.y.shape}\n"
            
            # Add a brief description of the transformation pattern
            input_colors = len(set(first_pair.x.flatten()))
            output_colors = len(set(first_pair.y.flatten()))
            context += f"Input colors: {input_colors}, Output colors: {output_colors}\n"
        
        return context
    
    def _generate_aligned_code(self, prompt: str) -> str:
        """Generate aligned code using Llama3.1-8B"""
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=4096
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only the generated part
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Failed to generate aligned code: {e}")
            # Return original code as fallback
            return f"# Error in alignment: {str(e)}\n{prompt}"
    
    def _parse_aligned_response(self, response: str, original_output: BARCOutput) -> BARCOutput:
        """Parse the aligned response and extract improved code"""
        
        # Extract code from response (look for python code blocks)
        code_match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
        if code_match:
            aligned_code = code_match.group(1).strip()
        else:
            # If no code block, try to extract everything after "```python"
            python_start = response.find('```python')
            if python_start != -1:
                aligned_code = response[python_start + 9:].strip()
                # Remove trailing ``` if present
                if aligned_code.endswith('```'):
                    aligned_code = aligned_code[:-3].strip()
            else:
                # Fallback: use the whole response
                aligned_code = response.strip()
        
        # Extract concepts and description from aligned code
        concepts = self._extract_concepts(aligned_code)
        description = self._extract_description(aligned_code)
        
        # Validate aligned code has required structure
        if not self._validate_barc_structure(aligned_code):
            logger.warning("Aligned code doesn't have proper BARC structure, using original")
            return original_output
        
        # Create new BARCOutput with aligned code
        return BARCOutput(
            code=aligned_code,
            concepts=concepts,
            description=description,
            raw_response=response
        )
    
    def _extract_concepts(self, code: str) -> Optional[str]:
        """Extract concepts from code"""
        match = re.search(r'#\s*concepts?\s*:\s*(.+)', code, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _extract_description(self, code: str) -> Optional[str]:
        """Extract description from code"""
        match = re.search(r'#\s*description\s*:\s*(.+)', code, re.IGNORECASE)
        return match.group(1).strip() if match else None
    
    def _validate_barc_structure(self, code: str) -> bool:
        """Validate that code has proper BARC structure"""
        required_patterns = [
            r'#\s*concepts?\s*:',  # concepts line
            r'#\s*description\s*:', # description line
            r'def\s+transform\s*\('  # transform function
        ]
        
        for pattern in required_patterns:
            if not re.search(pattern, code, re.IGNORECASE):
                return False
        
        return True