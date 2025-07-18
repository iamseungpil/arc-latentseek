#!/usr/bin/env python3
"""
Test script for the ARC-LatentSeek pipeline with alignment
"""

import os
import sys
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.main import ARCLatentSeekPipeline, PipelineConfig
from src.data import ARCDataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_alignment_pipeline():
    """Test the pipeline with alignment enabled"""
    
    logger.info("Starting alignment pipeline test")
    
    # Create configuration with alignment enabled
    config = PipelineConfig(
        # Use smaller models/settings for testing
        num_candidates=2,  # Reduce candidates for faster testing
        optimization_steps=3,  # Reduce steps for faster testing
        enable_code_alignment=True,
        use_description_based_optimization=True,
        min_alignment_score=10,  # Lower threshold for testing
        output_dir="test_results"
    )
    
    try:
        # Initialize pipeline
        logger.info("Initializing pipeline...")
        pipeline = ARCLatentSeekPipeline(config)
        
        # Load a single problem for testing
        logger.info("Loading test problems...")
        data_loader = ARCDataLoader()
        problems = data_loader.get_problems("validation", num_problems=1)
        
        if not problems:
            logger.error("No problems found!")
            return False
        
        test_problem = problems[0]
        logger.info(f"Testing with problem: {test_problem.uid}")
        
        # Solve the problem
        logger.info("Solving problem...")
        result = pipeline.solve_problem(test_problem)
        
        # Check results
        logger.info("Testing completed!")
        logger.info(f"Success: {result.success}")
        logger.info(f"Accuracy: {result.execution_accuracy:.2%}")
        logger.info(f"Reward: {result.best_reward:.3f}")
        logger.info(f"Time taken: {result.time_taken:.2f}s")
        
        # Print code snippet if available
        if result.best_code:
            logger.info("Generated code snippet:")
            logger.info(result.best_code[:200] + "..." if len(result.best_code) > 200 else result.best_code)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alignment_only():
    """Test just the alignment components without full pipeline"""
    
    logger.info("Testing alignment components only...")
    
    try:
        from src.alignment import BARCCodeAligner, AlignmentQualityAnalyzer
        from src.generators import BARCOutput
        from src.data import ARCDataLoader
        
        # Load a test problem
        data_loader = ARCDataLoader()
        problems = data_loader.get_problems("validation", num_problems=1)
        
        if not problems:
            logger.error("No problems found!")
            return False
        
        test_problem = problems[0]
        
        # Create a mock BARC output for testing
        mock_barc_output = BARCOutput(
            code="""def transform(input_grid):
    # Simple test transformation
    output_grid = input_grid.copy()
    output_grid[output_grid == 1] = 2
    return output_grid""",
            concepts="color replacement",
            description="Replace blue pixels with red pixels",
            raw_response="Test response"
        )
        
        # Test alignment
        logger.info("Testing code alignment...")
        aligner = BARCCodeAligner(temperature=0.3, max_new_tokens=512)
        aligned_output = aligner.align_code(mock_barc_output, test_problem)
        
        # Test quality analysis
        logger.info("Testing quality analysis...")
        analyzer = AlignmentQualityAnalyzer()
        quality = analyzer.analyze_alignment_quality(
            original_code=mock_barc_output.code,
            aligned_code=aligned_output.code,
            original_description=mock_barc_output.description,
            aligned_description=aligned_output.description
        )
        
        logger.info(f"Alignment quality score: {quality.improvement_score:.1f}")
        logger.info(f"Structure preserved: {quality.structure_preserved}")
        logger.info(f"Has common imports: {quality.has_common_imports}")
        logger.info(f"Uses color constants: {quality.has_color_constants}")
        
        return True
        
    except Exception as e:
        logger.error(f"Alignment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("ARC-LatentSeek Alignment Pipeline Test")
    print("=" * 60)
    
    # Test alignment components first
    print("\n1. Testing alignment components...")
    alignment_test_passed = test_alignment_only()
    
    if alignment_test_passed:
        print("✅ Alignment components test passed!")
        
        # Test full pipeline
        print("\n2. Testing full pipeline...")
        pipeline_test_passed = test_alignment_pipeline()
        
        if pipeline_test_passed:
            print("✅ Full pipeline test passed!")
            print("\n🎉 All tests completed successfully!")
        else:
            print("❌ Full pipeline test failed!")
            sys.exit(1)
    else:
        print("❌ Alignment components test failed!")
        sys.exit(1)