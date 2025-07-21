#!/usr/bin/env python3
"""Debug single validation problem with detailed logging"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 0

import logging
import numpy as np
from src.data import ARCDataLoader
from src.generators import BARCGenerator
from src.executors import CodeExecutor
# Create simple config
class Config:
    barc_model = "barc0/Llama-3.1-ARC-Potpourri-Induction-8B"
    execution_timeout = 2
    num_candidates = 1

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Load validation problem 0
    data_loader = ARCDataLoader()
    problems = data_loader.get_problems('validation', num_problems=1)
    problem = problems[0]
    
    logger.info(f"Problem ID: {problem.uid}")
    logger.info(f"Number of train pairs: {len(problem.train_pairs)}")
    for i, pair in enumerate(problem.train_pairs):
        logger.info(f"Train pair {i}: input shape {pair.x.shape}, output shape {pair.y.shape}")
    
    # Initialize components
    config = Config()
    
    barc_generator = BARCGenerator(config.barc_model)
    code_executor = CodeExecutor(config.execution_timeout)
    
    # Generate BARC code
    logger.info("Generating BARC code...")
    barc_outputs = barc_generator.generate(problem, num_candidates=1)
    
    if not barc_outputs:
        logger.error("No BARC outputs generated")
        return
        
    barc_output = barc_outputs[0]
    logger.info(f"Generated description: {barc_output.description}")
    logger.info(f"Generated code length: {len(barc_output.code)} characters")
    logger.info("Generated code:")
    print("=" * 80)
    print(barc_output.code)
    print("=" * 80)
    
    # Execute code
    logger.info("Executing code...")
    execution_result = code_executor.execute(barc_output.code, problem)
    
    logger.info(f"Execution success: {execution_result.success}")
    logger.info(f"Execution accuracy: {execution_result.accuracy}")
    
    if execution_result.error_messages:
        logger.error(f"Error messages: {execution_result.error_messages}")
    
    # Detailed execution analysis
    for i, (pair, output, comparison) in enumerate(zip(
        problem.train_pairs, 
        execution_result.output_grids,
        execution_result.comparison_results
    )):
        logger.info(f"\nTrain pair {i}:")
        logger.info(f"  Input shape: {pair.x.shape}")
        logger.info(f"  Expected output shape: {pair.y.shape}")
        
        if isinstance(output, np.ndarray):
            logger.info(f"  Actual output shape: {output.shape}")
            logger.info(f"  Comparison result: {comparison}")
            
            # Show first few values if shape is reasonable
            if output.size < 100:
                logger.info(f"  Expected output:\n{pair.y}")
                logger.info(f"  Actual output:\n{output}")
        else:
            logger.info(f"  Output type: {type(output)}")
            logger.info(f"  Output value: {output}")
    
    # Test individual execution
    logger.info("\nTesting individual execution on first training pair...")
    first_input = problem.train_pairs[0].x
    try:
        single_result = code_executor.execute_single(barc_output.code, first_input)
        logger.info(f"Single execution result type: {type(single_result)}")
        if isinstance(single_result, np.ndarray):
            logger.info(f"Single execution result shape: {single_result.shape}")
            if single_result.size < 100:
                logger.info(f"Single execution result:\n{single_result}")
        else:
            logger.info(f"Single execution result: {single_result}")
    except Exception as e:
        logger.error(f"Error in single execution: {e}", exc_info=True)

if __name__ == "__main__":
    main()