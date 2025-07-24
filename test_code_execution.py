#!/usr/bin/env python3
"""
Test code execution to debug failures
"""

import os
import sys
import torch
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add paths
sys.path.insert(0, '/home/ubuntu')
sys.path.insert(0, '/home/ubuntu/arc-latentseek')

# Import arc directly
import arc

# Now import our modules
from src.evaluators.simple_evaluator import SimpleEvaluator
from src.optimizers.latent_optimizer_v19_retry_until_valid import LatentOptimizerV19RetryUntilValid

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_single_generation():
    """Test a single generation and execution"""
    
    # Load model and tokenizer
    model_name = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B"
    logger.info("Loading model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cuda:5",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Initialize components
    evaluator = SimpleEvaluator()
    optimizer = LatentOptimizerV19RetryUntilValid(
        model=model,
        tokenizer=tokenizer,
        evaluator=evaluator,
        temperature=1.0,
        max_new_tokens=1024
    )
    
    # Test on first problem
    problem_id = "2a5f8217"
    problem = None
    for p in arc.validation_problems:
        if p.uid == problem_id:
            problem = p
            break
    
    # Create prompt
    prompt = optimizer.create_barc_prompt(problem_id)
    logger.info(f"Prompt length: {len(tokenizer.encode(prompt))} tokens")
    
    # Generate code
    logger.info("Generating code...")
    prompt_inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        generated = model.generate(
            **prompt_inputs,
            max_new_tokens=1024,
            temperature=1.0,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode and extract code
    full_response = tokenizer.decode(generated[0], skip_special_tokens=True)
    generated_text = full_response[len(prompt):]
    
    logger.info("Generated response:")
    logger.info("-" * 80)
    logger.info(generated_text[:1000] + "..." if len(generated_text) > 1000 else generated_text)
    logger.info("-" * 80)
    
    # Extract code
    code = optimizer.extract_code_from_response(generated_text)
    
    if not code:
        logger.error("No code found in response!")
        return
        
    logger.info("Extracted code:")
    logger.info("-" * 80)
    logger.info(code[:1000] + "..." if len(code) > 1000 else code)
    logger.info("-" * 80)
    
    # Test execution
    logger.info("Testing execution...")
    eval_result = evaluator.evaluate_solution(problem_id, code)
    
    logger.info(f"Execution success: {eval_result['execution_success']}")
    logger.info(f"Accuracy: {eval_result['accuracy']}")
    if eval_result.get('error'):
        logger.error(f"Error: {eval_result['error']}")
    
    # Calculate 5D rewards
    target_outputs = [test_pair.y for test_pair in problem.test_pairs]
    rewards = optimizer.calculate_5d_reward(
        eval_result.get("generated_outputs", []),
        target_outputs
    )
    
    logger.info("5D Rewards:")
    for k, v in rewards.items():
        logger.info(f"  {k}: {v:.3f}")

if __name__ == "__main__":
    test_single_generation()