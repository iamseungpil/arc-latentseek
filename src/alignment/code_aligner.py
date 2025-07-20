"""
BARC Code Aligner using Llama3.1-8B Instruct
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Optional, Dict, Any
import logging
import re
import time

# Using unsloth for optimization with transformers fallback

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
        """Load Llama3.1-8B Instruct model with unsloth optimization"""
        try:
            logger.info(f"Loading alignment model: {self.model_path}")
            
            # Try unsloth first, fallback to standard transformers
            try:
                from unsloth import FastLanguageModel
                logger.info("Attempting to load alignment model with unsloth optimization...")
                
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_path,
                    max_seq_length=4096,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,  # Use bfloat16 for better quality
                    device_map={"":0}  # Use only cuda:0
                )
                
                # Enable fast inference mode
                FastLanguageModel.for_inference(self.model)
                logger.info("✅ Alignment model loaded with unsloth optimization")
                
            except Exception as e:
                logger.warning(f"⚠️ Unsloth loading failed: {e}")
                logger.info("Falling back to standard transformers...")
                
                # Fallback to standard transformers
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map={"":0},  # Use only cuda:0
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager",
                )
                
                # Enable optimizations for faster inference
                self.model.eval()
                
                # Compile model for faster inference (PyTorch 2.0+)
                if hasattr(torch, 'compile'):
                    try:
                        logger.info("Compiling alignment model for faster inference...")
                        self.model = torch.compile(self.model, mode="reduce-overhead")
                        logger.info("Alignment model compilation successful")
                    except Exception as e:
                        logger.warning(f"Alignment model compilation failed: {e}, continuing without compilation")
            
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
        
        # Add raw alignment response to output
        aligned_output.alignment_response = aligned_response
        
        alignment_time = time.time() - start_time
        
        logger.info(f"Code alignment completed in {alignment_time:.2f}s")
        logger.info(f"Original code length: {len(barc_output.code)}")
        logger.info(f"Aligned code length: {len(aligned_output.code)}")
        
        return aligned_output
    
    def _create_alignment_prompt(self, barc_output: BARCOutput, problem: ARCProblem) -> str:
        """Create alignment prompt matching barc_sft_v2 format"""
        
        # Generate examples text using color names
        examples_text = []
        for i, pair in enumerate(problem.train_pairs):
            examples_text.append(f"Example {i+1}:")
            examples_text.append(f"Input: {self._grid_to_string(pair.x)}")
            examples_text.append(f"Output: {self._grid_to_string(pair.y)}")
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert ARC puzzle code reviewer who specializes in improving BARC model-generated solutions. Your task is to analyze and fix the given code while preserving its logical structure and improving its correctness.

**Critical Requirements:**

1. **Preserve BARC Structure**: Maintain the exact format with:
   - `# concepts:` section (list key pattern concepts)
   - `# description:` section (explain the transformation clearly)
   - `def transform(input_grid):` function

2. **Common.py Integration**: 
   - Use appropriate common.py utilities: `find_connected_components`, `object_position`, `blit_sprite`, `crop`, `detect_objects`, etc.
   - Use Color constants: `Color.BLACK`, `Color.BLUE`, `Color.RED`, etc.
   - Prefer common.py functions over manual numpy operations when available

3. **Pattern-Code Alignment**:
   - Ensure the `# description:` accurately describes what the code does
   - Make sure the code implementation matches the described pattern exactly
   - Fix any logical inconsistencies between description and implementation

4. **Code Quality**:
   - Fix syntax errors and logical bugs
   - Improve error handling and edge cases
   - Ensure proper grid bounds checking
   - Make the code more robust and efficient

5. **ARC-Specific Best Practices**:
   - Handle empty grids and edge cases gracefully
   - Use proper connectivity (4-way vs 8-way) for object detection
   - Ensure output grid has correct dimensions
   - Validate transformations against the given examples

**Analysis Process:**
1. First, understand the pattern from the examples
2. Check if the current description matches the pattern
3. Verify if the code correctly implements the description
4. Fix any discrepancies and improve robustness

Return ONLY the corrected Python code maintaining the exact BARC format.

<|eot_id|><|start_header_id|>user<|end_header_id|>

**Pattern Examples:**
{chr(10).join(examples_text)}

**Code to Review and Align:**
```python
{barc_output.code}
```

**Task**: Review this BARC-generated code and fix any issues while preserving its structure. Ensure the pattern description accurately describes the transformation and the code correctly implements it. Use common.py utilities appropriately.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

```python"""
        
        return prompt.strip()
    
    def _grid_to_string(self, grid: np.ndarray) -> str:
        """Convert grid to color string format matching BARC"""
        COLOR_MAP = {
            0: "Black", 1: "Blue", 2: "Red", 3: "Green", 4: "Yellow",
            5: "Gray", 6: "Pink", 7: "Orange", 8: "Purple", 9: "Brown"
        }
        rows = []
        for row in grid:
            color_row = [COLOR_MAP.get(int(cell), f"Unknown_{cell}") for cell in row]
            rows.append(" ".join(color_row))
        return "\n".join(rows)
    
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
        
        # Validate Python syntax before proceeding
        try:
            compile(aligned_code, '<string>', 'exec')
        except SyntaxError as e:
            logger.warning(f"Aligned code has syntax error: {e}, using original")
            return original_output
        
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
            plan=original_output.plan,  # Preserve original plan
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