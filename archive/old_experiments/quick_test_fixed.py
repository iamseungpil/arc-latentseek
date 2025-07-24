"""
Quick test to verify the fixed LatentSeek implementation
"""

import os
import sys
import json
import logging
import torch

sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader
from src.generators import BARCGenerator, BARCOutput
from src.optimizers.latent_optimizer import LatentSeekOptimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_hidden_state_generation():
    """Test if we can generate from hidden states"""
    
    logger.info("Testing hidden state generation...")
    
    # Initialize only what we need
    loader = ARCDataLoader()
    barc_generator = BARCGenerator()
    
    # Create a minimal optimizer instance just to test the method
    optimizer = LatentSeekOptimizer(
        barc_generator=barc_generator,
        code_executor=None,  # Not needed for this test
        glm_evaluator=None,  # Not needed for this test
        max_steps=1
    )
    
    # Test the new methods
    logger.info("Testing _extract_code_from_text...")
    
    test_text = """
Here is the solution:

```python
from common import *
import numpy as np

def transform(grid):
    return grid * 2
```

That should work!
"""
    
    code = optimizer._extract_code_from_text(test_text)
    logger.info(f"Extracted code: {code is not None}")
    
    # Test description extraction
    test_text2 = """
# description:
# This problem involves duplicating the pattern
# and changing colors from gray to red
def transform(grid):
    pass
"""
    
    desc = optimizer._extract_description_from_text(test_text2)
    logger.info(f"Extracted description: '{desc}'")
    
    # Test hidden state decoding (simplified)
    logger.info("\nTesting hidden state decoding mechanics...")
    
    # Create dummy hidden states and input_ids
    device = optimizer.model.device
    hidden_size = optimizer.model.config.hidden_size
    
    dummy_hidden_states = torch.randn(10, hidden_size).to(device)
    dummy_input_ids = torch.tensor([[1, 2, 3, 4, 5]]).to(device)
    
    try:
        # This should at least not crash
        result_ids = optimizer._decode_from_hidden_states(
            dummy_hidden_states,
            dummy_input_ids,
            max_new_tokens=10
        )
        logger.info(f"Decoding test passed! Generated shape: {result_ids.shape}")
    except Exception as e:
        logger.error(f"Decoding test failed: {e}")
    
    logger.info("\nQuick test completed!")


if __name__ == "__main__":
    test_hidden_state_generation()