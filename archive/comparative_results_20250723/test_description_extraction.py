#!/usr/bin/env python
"""Test description extraction from BARC model outputs"""

import sys
sys.path.append('/home/ubuntu/arc-latentseek')

from src.optimizers.latent_optimizer import LatentSeekOptimizer

# Test sample code snippets
test_cases = [
    # Case 1: Standard format with explicit description tag
    """from common import *

# concepts:
# pattern recognition, color transformation

# description:
# In the input grid, find all blue pixels.
# Replace them with red pixels in the output.
# Keep all other colors unchanged.

def transform(grid):
    return grid
""",
    
    # Case 2: Description after concepts without explicit tag
    """from common import *

# concepts:
# rotation, scaling

# The input contains a shape that needs to be rotated 90 degrees clockwise.
# After rotation, scale it by factor of 2.

def main(input_grid):
    pass
""",
    
    # Case 3: Description inside function
    """from common import *

def transform(grid):
    # Find the largest connected component
    # Color it blue and remove all other objects
    # Return the modified grid
    return grid
"""
]

def test_extraction():
    """Test description extraction on sample cases"""
    # Import the actual methods we need
    from src.optimizers.latent_optimizer import LatentSeekOptimizer
    
    # Create a dummy instance just to access methods
    class DummyOptimizer:
        def _find_description_token_positions(self, text):
            return LatentSeekOptimizer._find_description_token_positions(self, text)
        
        def _extract_description_from_text(self, text):
            return LatentSeekOptimizer._extract_description_from_text(self, text)
    
    optimizer = DummyOptimizer()
    
    print("Testing description extraction...\n")
    
    for i, code in enumerate(test_cases):
        print(f"=== Test Case {i+1} ===")
        print("Code snippet:")
        print(code[:100] + "...")
        
        # Test character position finding
        desc_start, desc_end = optimizer._find_description_token_positions(code)
        
        if desc_start is not None and desc_end is not None:
            print(f"\nFound description at positions: {desc_start}-{desc_end}")
            print(f"Extracted text: '{code[desc_start:desc_end]}'")
            
            # Test description extraction
            description = optimizer._extract_description_from_text(code)
            print(f"Processed description: '{description}'")
        else:
            print("\nNo description found!")
        
        print("\n" + "-"*60 + "\n")

def test_tokenization():
    """Test how description positions translate to token positions"""
    from transformers import AutoTokenizer
    
    print("\n=== Testing Tokenization ===")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    
    test_text = """# description:
# This is a test description
# spanning multiple lines"""
    
    # Tokenize
    tokens = tokenizer.tokenize(test_text)
    print(f"Text: {test_text}")
    print(f"Tokens ({len(tokens)}): {tokens}")
    
    # Find character positions
    desc_start = test_text.find("This is")
    desc_end = test_text.find("lines") + len("lines")
    
    print(f"\nCharacter positions: {desc_start}-{desc_end}")
    print(f"Text slice: '{test_text[desc_start:desc_end]}'")
    
    # Convert to token positions
    text_before = test_text[:desc_start]
    text_until_end = test_text[:desc_end]
    
    tokens_before = len(tokenizer.tokenize(text_before))
    tokens_until_end = len(tokenizer.tokenize(text_until_end))
    
    print(f"Token positions: {tokens_before}-{tokens_until_end}")
    print(f"Token slice: {tokens[tokens_before:tokens_until_end]}")

if __name__ == "__main__":
    test_extraction()
    test_tokenization()