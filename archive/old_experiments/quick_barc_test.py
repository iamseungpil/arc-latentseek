#!/usr/bin/env python3
"""
Quick test with actual BARC model
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

print(f"Testing with problem: {problem.uid}")

# Use actual BARC model
generator = BARCGeneratorFixed("barc0/Llama-3.1-ARC-Potpourri-Induction-8B", device="cuda")

# Generate
print("\nGenerating code...")
candidates = generator.generate(problem, num_candidates=1, temperature=0.7)

if candidates and candidates[0].code:
    candidate = candidates[0]
    print(f"\n✓ Code generated: {len(candidate.code)} chars")
    print(f"✓ Has transform: {'def transform' in candidate.code}")
    print(f"✓ Description: {candidate.description[:100] + '...' if candidate.description else 'None'}")
    
    # Execute
    executor = CodeExecutor()
    result = executor.execute(candidate.code, problem)
    print(f"\n✓ Execution: {'Success' if result.success else 'Failed'}")
    print(f"✓ Accuracy: {result.accuracy:.1%}")
    
    # Render if successful
    if result.success and result.output_grids:
        renderer = GridRenderer()
        output_path = f"quick_test_{problem.uid}.png"
        renderer.render_problem_with_output(
            problem, 
            result.output_grids,
            output_path
        )
        print(f"\n✓ Rendered to: {output_path}")
else:
    print("✗ Generation failed")