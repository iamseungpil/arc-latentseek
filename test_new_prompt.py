#!/usr/bin/env python3
"""Test new prompt generation"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import ARCDataLoader
from src.generators.barc_generator_fixed import BARCGeneratorFixed
from src.executors import CodeExecutor
from src.executors.grid_renderer import GridRenderer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_new_prompt():
    """Test code generation with new prompt"""
    
    # Initialize
    logger.info("Initializing components...")
    generator = BARCGeneratorFixed(
        "barc0/Llama-3.1-ARC-Potpourri-Induction-8B", 
        device="cuda"
    )
    executor = CodeExecutor()
    renderer = GridRenderer()
    
    # Load a test problem
    data_loader = ARCDataLoader()
    problems = data_loader.get_problems(split="validation", num_problems=1)
    problem = problems[0]
    
    logger.info(f"\nTesting with problem: {problem.uid}")
    
    # Generate solution
    logger.info("Generating solution with new prompt...")
    candidates = generator.generate(problem, num_candidates=1, temperature=0.7)
    
    if not candidates or not candidates[0].code:
        logger.error("Failed to generate solution")
        return
    
    output = candidates[0]
    
    # Log generated content
    logger.info(f"\nGenerated description: {output.description}")
    logger.info(f"\nGenerated concepts: {output.concepts}")
    logger.info(f"\nGenerated code:\n{output.code}")
    
    # Also log the full raw response
    logger.info(f"\n{'='*80}\nFull raw response:\n{'='*80}\n{output.raw_response}\n{'='*80}")
    
    # Execute code
    logger.info("\nExecuting code...")
    result = executor.execute(output.code, problem)
    
    if result.success:
        logger.info(f"Code execution: Success")
        logger.info(f"Accuracy: {result.accuracy:.1%}")
        logger.info(f"Output grids generated: {len(result.output_grids)}")
        
        # Save visualization
        output_path = "test_new_prompt_output.png"
        renderer.render_problem_with_output(
            problem, result.output_grids, output_path
        )
        logger.info(f"Visualization saved to: {output_path}")
    else:
        logger.error(f"Code execution: Failed")
        logger.error(f"Error: {getattr(result, 'error_message', 'Unknown error')}")
    
    # Save code
    with open("test_new_prompt_code.py", "w") as f:
        f.write(output.code)
    logger.info("Code saved to: test_new_prompt_code.py")

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "5"
    test_new_prompt()