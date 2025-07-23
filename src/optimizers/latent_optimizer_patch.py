"""
Patch for the original latent_optimizer.py to use the fixed implementation
"""

import torch
import torch.nn.functional as F
from typing import Optional, List
import logging

from ..data import ARCProblem
from ..generators import BARCOutput
from .latent_optimizer import LatentSeekOptimizer

logger = logging.getLogger(__name__)


def _generate_with_optimized_description_fixed(self,
                                              problem: ARCProblem,
                                              optimized_desc_states: torch.Tensor,
                                              full_hidden_states: List[torch.Tensor],
                                              desc_start: int,
                                              desc_end: int) -> Optional[BARCOutput]:
    """
    Fixed version: Generate with optimized description states without token accumulation
    """
    try:
        # Create prompt
        prompt = self.barc_generator._create_prompt(problem)
        prompt_text = self.tokenizer.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)
        prompt_length = prompt_inputs.input_ids.shape[1]
        
        # Start with base prompt only (no accumulation!)
        input_ids = prompt_inputs.input_ids.clone()
        
        with torch.no_grad():
            # Generate pre-description tokens if any
            desc_start_offset = desc_start - prompt_length
            desc_end_offset = desc_end - prompt_length
            
            if desc_start_offset > 0:
                # Generate tokens before description
                for i in range(desc_start_offset):
                    if hasattr(self.model, 'model'):
                        outputs = self.model.model(input_ids, output_hidden_states=True)
                        hidden_states = outputs[0][:, -1, :]
                    else:
                        outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
                        hidden_states = outputs.hidden_states[-1][:, -1, :]
                    
                    logits = self.model.lm_head(hidden_states)
                    next_token_id = torch.argmax(logits, dim=-1)
                    
                    if next_token_id.dim() == 0:
                        next_token_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                    else:
                        next_token_tensor = next_token_id.unsqueeze(0)
                    
                    input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
            
            # Add optimized description tokens
            # Ensure optimized_desc_states has correct shape
            if optimized_desc_states.dim() == 3 and optimized_desc_states.size(1) == 1:
                optimized_desc_states_squeezed = optimized_desc_states.squeeze(1)
            else:
                optimized_desc_states_squeezed = optimized_desc_states
            
            desc_tokens = torch.argmax(self.model.lm_head(optimized_desc_states_squeezed), dim=-1)
            if desc_tokens.dim() == 2 and desc_tokens.shape[1] == 1:
                desc_tokens = desc_tokens.squeeze(-1)
            desc_tokens_tensor = desc_tokens.unsqueeze(0)
            input_ids = torch.cat([input_ids, desc_tokens_tensor], dim=-1)
            
            # Continue generation after description
            for _ in range(2048):  # max_new_tokens
                if hasattr(self.model, 'model'):
                    outputs = self.model.model(input_ids, output_hidden_states=True)
                    hidden_states = outputs[0][:, -1, :]
                else:
                    outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)
                    hidden_states = outputs.hidden_states[-1][:, -1, :]
                
                logits = self.model.lm_head(hidden_states)
                next_token_id = torch.argmax(logits, dim=-1)
                
                # Check for EOS
                token_value = next_token_id.item() if len(next_token_id.shape) == 0 else next_token_id[0].item()
                if token_value == self.tokenizer.eos_token_id:
                    break
                
                if next_token_id.dim() == 0:
                    next_token_tensor = next_token_id.unsqueeze(0).unsqueeze(0)
                else:
                    next_token_tensor = next_token_id.unsqueeze(0)
                
                input_ids = torch.cat([input_ids, next_token_tensor], dim=-1)
        
        # Decode the complete sequence
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        
        # Extract code from generated text
        code = self._extract_code_from_text(generated_text)
        description = self._extract_description_from_text(generated_text)
        
        if code:
            return BARCOutput(
                code=code,
                description=description,
                concepts=None,
                plan=None,
                raw_response=generated_text
            )
        else:
            logger.warning("Failed to extract code from optimized generation")
            return None
            
    except Exception as e:
        logger.error(f"Error in _generate_with_optimized_description_fixed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def apply_patch():
    """Apply the fixed implementation as a patch"""
    # Monkey patch the method
    LatentSeekOptimizer._generate_with_optimized_description = _generate_with_optimized_description_fixed
    logger.info("Applied fixed LatentSeek patch to avoid token accumulation")