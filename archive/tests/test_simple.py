#!/usr/bin/env python
"""Simple test of description extraction"""

import sys
sys.path.append('/home/ubuntu/arc-latentseek')

from src.optimizers.latent_optimizer import LatentSeekOptimizer

# Test code
test_code = """from common import *

# concepts:
# pattern scaling

# description:
# Scale the input pattern by 2x

def transform(grid):
    return grid"""

# Test _find_description_token_positions
class DummyOptimizer:
    def _find_description_token_positions(self, text):
        return LatentSeekOptimizer._find_description_token_positions(self, text)

optimizer = DummyOptimizer()
desc_start, desc_end = optimizer._find_description_token_positions(test_code)

print(f"Description found at: {desc_start}-{desc_end}")
if desc_start and desc_end:
    print(f"Text: '{test_code[desc_start:desc_end]}'")