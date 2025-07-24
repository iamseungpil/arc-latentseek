#!/usr/bin/env python3
"""
Test code extraction and rendering
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data import ARCDataLoader
from src.generators.barc_generator_fixed import BARCGeneratorFixed
from src.executors import CodeExecutor
from src.executors.grid_renderer import GridRenderer

print("="*80)
print("TEST: Code Extraction and Rendering")
print("="*80)

# Load a problem
data_loader = ARCDataLoader()
problems = data_loader.get_problems(split="validation", num_problems=1)
problem = problems[0]

print(f"\nProblem: {problem.uid}")
print(f"Training examples: {len(problem.train_pairs)}")

# Initialize components
generator = BARCGeneratorFixed("Qwen/Qwen2.5-0.5B-Instruct", device="cuda")
executor = CodeExecutor()
renderer = GridRenderer()

# Generate code
print("\n1. Generating code...")
candidates = generator.generate(problem, num_candidates=1, temperature=0.7)

if not candidates or not candidates[0].code:
    print("Failed to generate code!")
    exit(1)

candidate = candidates[0]
print("✓ Code generated successfully")

# Show extracted code
print("\n2. Extracted Code:")
print("-"*80)
print(candidate.code)
print("-"*80)

# Show extracted metadata
print("\n3. Extracted Metadata:")
print(f"Concepts: {candidate.concepts}")
print(f"Description: {candidate.description}")

# Execute code
print("\n4. Executing code...")
result = executor.execute(candidate.code, problem)

print(f"Execution success: {result.success}")
print(f"Accuracy: {result.accuracy:.1%}")

if result.error_messages:
    print("Errors:")
    for error in result.error_messages:
        print(f"  - {error}")

# Render results
print("\n5. Rendering results...")

# Create output directory
output_dir = Path("test_output")
output_dir.mkdir(exist_ok=True)

# Render first training example
if problem.train_pairs:
    print(f"\nRendering training example 1...")
    train_pair = problem.train_pairs[0]
    
    # Render input
    input_path = output_dir / f"{problem.uid}_train1_input.png"
    renderer.render_simple_grid(train_pair.x, str(input_path))
    print(f"✓ Input saved to: {input_path}")
    
    # Render expected output
    expected_path = output_dir / f"{problem.uid}_train1_expected.png"
    renderer.render_simple_grid(train_pair.y, str(expected_path))
    print(f"✓ Expected output saved to: {expected_path}")
    
    # Render predicted output if available
    if result.output_grids and len(result.output_grids) > 0 and isinstance(result.output_grids[0], np.ndarray):
        predicted_path = output_dir / f"{problem.uid}_train1_predicted.png"
        renderer.render_simple_grid(result.output_grids[0], str(predicted_path))
        print(f"✓ Predicted output saved to: {predicted_path}")
    else:
        print("✗ No predicted output available (execution failed)")

# Show the images
print("\n6. Displaying rendered images...")
print(f"Check the {output_dir} directory for the generated images.")

# Also render the whole problem  
# Check if we have valid numpy arrays in output_grids
valid_outputs = [grid for grid in result.output_grids if isinstance(grid, np.ndarray)] if result.output_grids else []

if valid_outputs:
    problem_path = output_dir / f"{problem.uid}_problem.png"
    renderer.render_problem_with_output(
        problem, 
        valid_outputs,
        str(problem_path),
        title=f"Problem {problem.uid} - Accuracy: {result.accuracy:.1%}"
    )
    print(f"✓ Problem visualization saved to: {problem_path}")
else:
    # Just render the problem without predictions
    problem_path = output_dir / f"{problem.uid}_problem_only.png"
    renderer.render_arc_problem(problem, str(problem_path))
    print(f"✓ Problem (without predictions) saved to: {problem_path}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)