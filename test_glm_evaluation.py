#!/usr/bin/env python3
"""
Test 3: GLM evaluation on 2a5f8217 to understand reasoning
"""

import sys
import numpy as np
from src.data import ARCDataLoader
from src.executors import CodeExecutor, ExecutionResult
from src.evaluators import GLMEvaluator
from src.generators import BARCOutput
import time

def create_test_barc_output(code, description="Test description", concepts="Test concepts"):
    """Create a test BARCOutput for evaluation"""
    return BARCOutput(
        code=code,
        concepts=concepts,
        description=description,
        plan="Test plan",
        raw_response=f"Test response with code:\n{code}"
    )

def test_glm_evaluation():
    """Test GLM evaluation on 2a5f8217 problem"""
    
    print("=== GLM EVALUATION TEST ===")
    
    # Load problem 2a5f8217
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    print(f"Problem: {problem.uid}")
    print(f"Training pairs: {len(problem.train_pairs)}")
    
    # Analyze the actual problem pattern
    print(f"\n=== PROBLEM ANALYSIS ===")
    for i, pair in enumerate(problem.train_pairs):
        print(f"\nPair {i}:")
        print(f"  Input shape: {pair.x.shape}")
        print(f"  Output shape: {pair.y.shape}")
        print(f"  Input unique colors: {np.unique(pair.x)} -> Output unique colors: {np.unique(pair.y)}")
        
        # Check if blue (1) disappears
        has_blue_input = 1 in pair.x
        has_blue_output = 1 in pair.y
        print(f"  Blue in input: {has_blue_input}, Blue in output: {has_blue_output}")
        
        if has_blue_input and not has_blue_output:
            blue_positions = np.where(pair.x == 1)
            print(f"  Blue positions in input: {len(blue_positions[0])} pixels")
            
            # Find what colors replace blue
            for j, (r, c) in enumerate(zip(blue_positions[0], blue_positions[1])):
                replacement_color = pair.y[r, c]
                print(f"    Blue at ({r},{c}) -> Color {replacement_color}")
    
    # Test different code scenarios
    test_scenarios = [
        {
            "name": "Perfect Solution",
            "code": """from common import *
import numpy as np

# concepts: color transformation, object detection, nearest neighbor
# description: Change blue objects to the color of the nearest non-blue object

def transform(input_grid):
    output_grid = np.copy(input_grid)
    
    # Find blue pixels
    blue_mask = (input_grid == Color.BLUE)
    blue_positions = np.where(blue_mask)
    
    for i in range(len(blue_positions[0])):
        r, c = blue_positions[0][i], blue_positions[1][i]
        
        # Find nearest non-blue, non-black pixel
        min_dist = float('inf')
        replacement_color = Color.BLUE
        
        for rr in range(input_grid.shape[0]):
            for cc in range(input_grid.shape[1]):
                if input_grid[rr, cc] not in [Color.BLACK, Color.BLUE]:
                    dist = abs(rr - r) + abs(cc - c)  # Manhattan distance
                    if dist < min_dist:
                        min_dist = dist
                        replacement_color = input_grid[rr, cc]
        
        output_grid[r, c] = replacement_color
    
    return output_grid""",
            "description": "Change blue objects to the color of the nearest non-blue object"
        },
        {
            "name": "Wrong Solution",
            "code": """from common import *
import numpy as np

# concepts: color removal
# description: Remove all blue pixels

def transform(input_grid):
    output_grid = np.copy(input_grid)
    output_grid[output_grid == Color.BLUE] = Color.BLACK
    return output_grid""",
            "description": "Remove all blue pixels by changing them to black"
        },
        {
            "name": "Syntax Error",
            "code": """from common import *
import numpy as np

# concepts: invalid syntax
# description: This has syntax errors

def transform(input_grid:
    output_grid = np.copy(input_grid
    return output_grid""",
            "description": "Code with syntax errors"
        }
    ]
    
    # Initialize GLM evaluator
    print(f"\n=== INITIALIZING GLM EVALUATOR ===")
    glm_evaluator = GLMEvaluator()
    
    # Test each scenario
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = f"results/glm_evaluation_test_{timestamp}.txt"
    
    with open(results_file, "w") as f:
        f.write("=== GLM EVALUATION TEST RESULTS ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Problem: {problem.uid}\n\n")
        
        for scenario in test_scenarios:
            print(f"\n=== TESTING: {scenario['name']} ===")
            f.write(f"=== SCENARIO: {scenario['name']} ===\n")
            
            # Create BARC output
            barc_output = create_test_barc_output(
                scenario['code'], 
                scenario['description']
            )
            
            # Execute code
            executor = CodeExecutor(timeout=5)
            execution_result = executor.execute(barc_output.code, problem)
            
            print(f"Execution success: {execution_result.success}")
            print(f"Execution accuracy: {execution_result.accuracy:.4f}")
            f.write(f"Execution success: {execution_result.success}\n")
            f.write(f"Execution accuracy: {execution_result.accuracy:.4f}\n")
            
            if execution_result.error_messages:
                print(f"Errors: {execution_result.error_messages}")
                f.write(f"Errors: {execution_result.error_messages}\n")
            
            # GLM evaluation
            try:
                print("Running GLM evaluation...")
                evaluation_result = glm_evaluator.evaluate(
                    problem,
                    barc_output,
                    execution_result,
                    f"test_glm_{scenario['name'].lower().replace(' ', '_')}"
                )
                
                print(f"GLM total reward: {evaluation_result.total_reward:.4f}")
                print(f"Component scores: {evaluation_result.component_scores}")
                print(f"Verifications: {[k for k, v in evaluation_result.verifications.items() if v.passed]}")
                
                f.write(f"GLM total reward: {evaluation_result.total_reward:.4f}\n")
                f.write(f"Component scores: {evaluation_result.component_scores}\n")
                f.write(f"Verifications passed: {[k for k, v in evaluation_result.verifications.items() if v.passed]}\n")
                
                # Write detailed feedback
                f.write("\n--- DETAILED FEEDBACK ---\n")
                for name, feedback in evaluation_result.detailed_feedback.items():
                    f.write(f"{name}: {feedback[:200]}...\n")
                
            except Exception as e:
                print(f"GLM evaluation failed: {e}")
                f.write(f"GLM evaluation failed: {e}\n")
            
            f.write("\n" + "="*50 + "\n\n")
    
    print(f"\nDetailed GLM test results saved to: {results_file}")

if __name__ == "__main__":
    test_glm_evaluation()