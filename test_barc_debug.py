#!/usr/bin/env python3
"""
Test script to debug BARC code generation and execution for problem 2a5f8217
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add paths
sys.path.append('/home/ubuntu/arc-latentseek')

# Import modules
from src.data.arc_loader import ARCDataLoader
from src.generators.barc_generator import BARCGenerator
from src.executors.code_executor_fixed import CodeExecutor

def print_grid(grid, title="Grid"):
    """Pretty print a grid with colors"""
    color_map = {
        0: "⬛", 1: "🟦", 2: "🟥", 3: "🟩", 4: "🟨",
        5: "⬜", 6: "🟪", 7: "🟧", 8: "🟫", 9: "🟫"
    }
    print(f"\n{title}:")
    for row in grid:
        print("".join(color_map.get(int(cell), "❓") for cell in row))

def main():
    print("="*80)
    print("BARC Debug Test for Problem 2a5f8217")
    print("="*80)
    
    # 1. Load the specific problem
    print("\n1. Loading problem 2a5f8217...")
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    if not problem:
        print("❌ Problem 2a5f8217 not found!")
        return
        
    print(f"✅ Loaded problem: {problem}")
    print(f"   Train pairs: {len(problem.train_pairs)}")
    print(f"   Test pairs: {len(problem.test_pairs)}")
    
    # Show training examples
    print("\n2. Training Examples:")
    for i, pair in enumerate(problem.train_pairs):
        print(f"\n   Example {i+1}:")
        print_grid(pair.x, f"   Input {i+1}")
        print_grid(pair.y, f"   Output {i+1}")
    
    # 2. Generate code using BARC
    print("\n3. Generating code with BARC...")
    try:
        generator = BARCGenerator()
        
        # Generate multiple candidates
        outputs = generator.generate(problem, temperature=0.8, num_candidates=3)
    except Exception as e:
        print(f"\n❌ Error during BARC generation: {e}")
        print("\nTrying with fewer candidates...")
        try:
            outputs = generator.generate(problem, temperature=0.8, num_candidates=1)
        except Exception as e2:
            print(f"❌ Still failed: {e2}")
            return
    
    print(f"\n✅ Generated {len(outputs)} candidate solutions")
    
    # 3. Test each generated code
    executor = CodeExecutor(timeout=5)
    
    for i, output in enumerate(outputs):
        print(f"\n{'='*80}")
        print(f"CANDIDATE {i+1}")
        print(f"{'='*80}")
        
        # Show generated description
        if output.description:
            print(f"\nDescription: {output.description[:200]}...")
        
        # Show the generated code
        print("\nGenerated Code:")
        print("-"*40)
        print(output.code)
        print("-"*40)
        
        # Check for Color.PURPLE or Color.BROWN in code
        if "Color.PURPLE" in output.code or "Color.BROWN" in output.code:
            print("\n⚠️ WARNING: Code contains Color.PURPLE or Color.BROWN!")
            print("   These are BARC-specific colors that don't exist in standard ARC")
        
        # Check for numeric color values 8 and 9
        if "= 8" in output.code or "= 9" in output.code:
            print("\n⚠️ WARNING: Code uses color values 8 or 9!")
            print("   In BARC: 8=Purple, 9=Brown")
            print("   In ARC standard: 8=Teal, 9=Maroon")
        
        # 4. Execute the code
        print("\n4. Executing code...")
        result = executor.execute(output.code, problem)
        
        print(f"\n   Success: {result.success}")
        print(f"   Accuracy: {result.accuracy:.2%}")
        
        if result.error_messages:
            print("\n   Errors:")
            for error in result.error_messages:
                print(f"   - {error}")
        
        # Show execution results for each training pair
        print("\n   Execution Results:")
        for j, (pair, output_grid, comparison) in enumerate(
            zip(problem.train_pairs, result.output_grids, result.comparison_results)
        ):
            print(f"\n   Training Pair {j+1}: {comparison.value}")
            
            if output_grid is not None and not isinstance(output_grid, str):
                print_grid(pair.x, "     Input")
                print_grid(output_grid, "     Generated Output")
                print_grid(pair.y, "     Expected Output")
                
                # Check for color mismatches
                if not np.array_equal(output_grid, pair.y):
                    diff_mask = output_grid != pair.y
                    if np.any(diff_mask):
                        diff_positions = np.argwhere(diff_mask)
                        print(f"\n     Differences at {len(diff_positions)} positions:")
                        for pos in diff_positions[:5]:  # Show first 5 differences
                            r, c = pos
                            print(f"       Position ({r},{c}): Generated={output_grid[r,c]}, Expected={pair.y[r,c]}")
        
        # Test on test input
        if result.success and problem.test_pairs:
            print("\n5. Testing on test input...")
            test_output = executor.execute_single(output.code, problem.test_pairs[0].x)
            
            if isinstance(test_output, np.ndarray):
                print_grid(problem.test_pairs[0].x, "   Test Input")
                print_grid(test_output, "   Test Output")
            else:
                print(f"   Test execution failed: {test_output}")

if __name__ == "__main__":
    main()