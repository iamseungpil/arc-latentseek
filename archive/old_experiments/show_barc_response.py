#!/usr/bin/env python3
"""
Show BARC response with code, description, and rendering
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import ARCDataLoader
from src.generators.barc_generator_fixed import BARCGeneratorFixed
from src.executors import CodeExecutor
from src.executors.grid_renderer import GridRenderer

# Load problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems(split="validation", num_problems=1)
problem = problems[0]

print("="*80)
print(f"BARC Response for Problem: {problem.uid}")
print("="*80)

# Use actual BARC model
generator = BARCGeneratorFixed("barc0/Llama-3.1-ARC-Potpourri-Induction-8B", device="cuda")

# Generate
candidates = generator.generate(problem, num_candidates=1, temperature=0.7)

if candidates and candidates[0].code:
    candidate = candidates[0]
    
    # Show raw response
    print("\n1. RAW RESPONSE:")
    print("-"*80)
    print(candidate.raw_response)
    print("-"*80)
    
    # Show extracted metadata
    print("\n2. EXTRACTED METADATA:")
    print("-"*80)
    print(f"Concepts: {candidate.concepts}")
    print(f"\nDescription: {candidate.description}")
    print("-"*80)
    
    # Show extracted code
    print("\n3. EXTRACTED CODE:")
    print("-"*80)
    print(candidate.code)
    print("-"*80)
    
    # Execute code
    executor = CodeExecutor()
    result = executor.execute(candidate.code, problem)
    
    print("\n4. EXECUTION RESULTS:")
    print("-"*80)
    print(f"Success: {result.success}")
    print(f"Accuracy: {result.accuracy:.1%}")
    if result.error_messages:
        print("\nErrors:")
        for i, error in enumerate(result.error_messages):
            print(f"  Pair {i}: {error}")
    print("-"*80)
    
    # Render results
    renderer = GridRenderer()
    output_path = "barc_response_visualization.png"
    renderer.render_problem_with_output(
        problem, 
        result.output_grids if result.success else [],
        output_path
    )
    print(f"\n5. VISUALIZATION saved to: {output_path}")
    
else:
    print("Generation failed!")