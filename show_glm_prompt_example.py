#!/usr/bin/env python3
"""
Example script to show GLM evaluator prompt with real ARC problem 2a5f8217
"""

import sys
import os
sys.path.append('/home/ubuntu')

# Import ARC module directly
try:
    from arc import train_problems, validation_problems
except ImportError:
    print("Error: Could not import arc module. Showing mock example instead.")
    train_problems = []
    validation_problems = []

from dataclasses import dataclass
from typing import Optional

@dataclass
class BARCOutput:
    """Output from BARC generator"""
    code: str
    concepts: Optional[str]
    description: Optional[str] 
    plan: Optional[str]
    raw_response: str

def show_glm_prompt_example():
    """Show GLM prompt for ARC problem 2a5f8217"""
    
    # Find the ARC problem
    problem = None
    all_problems = train_problems + validation_problems
    
    for p in all_problems:
        if str(p.uid) == "2a5f8217":
            problem = p
            break
    
    if problem is None:
        print("Error: Could not find problem 2a5f8217")
        # Let's see what problems are available
        print("Available problems (first 5):")
        for i, p in enumerate(all_problems[:5]):
            print(f"  {p.uid}")
        # Use first problem as example instead
        if all_problems:
            problem = all_problems[0]
            print(f"\nUsing problem {problem.uid} as example instead")
        else:
            print("No problems available - showing mock example")
            show_mock_glm_prompt()
            return
    
    print(f"Found problem: {problem.uid}")
    print(f"Training pairs: {len(problem.train_pairs)}")
    print(f"Test pairs: {len(problem.test_pairs)}")
    
    # Show the problem structure
    print("\nProblem structure:")
    for i, pair in enumerate(problem.train_pairs):
        print(f"  Train {i+1}: input {pair.x.shape} -> output {pair.y.shape}")
    for i, pair in enumerate(problem.test_pairs):
        print(f"  Test {i+1}: input {pair.x.shape} -> output {pair.y.shape}")
    
    # Create a mock BARC output (simulating what BARC would generate)
    mock_barc_output = BARCOutput(
        code="""
def solve(grid):
    # Pattern: Each blue cell (color 1) gets surrounded by red cells (color 2)
    # Step 1: Find all blue cells
    result = grid.copy()
    height, width = grid.shape
    
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 1:  # Blue cell
                # Add red cells around it
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if result[ni][nj] == 0:  # Only if currently black
                                result[ni][nj] = 2  # Make it red
    return result
""",
        concepts="pattern_completion, color_surrounding, spatial_relationships",
        description="The transformation adds red cells (color 2) around each blue cell (color 1), forming a surrounding pattern while preserving the original blue cells.",
        plan="1. Identify all blue cells in the grid 2. For each blue cell, add red cells in all adjacent positions 3. Only change black cells to red, preserve existing colors",
        raw_response="Looking at this pattern, I need to... [full BARC response would be longer]"
    )
    
    # Show the GLM dual image prompt
    print("\n" + "="*80)
    print("GLM DUAL IMAGE PROMPT STRUCTURE")
    print("="*80)
    
    print("\n1. FIRST IMAGE: ARC Problem Training Examples")
    print("   - Shows the original training input-output pairs")
    print("   - Rendered as a visualization with input and output grids side by side")
    print("   - File: temp_eval_problem.png")
    
    print("\n2. SECOND IMAGE: Solution Result")
    print("   - Shows test input and generated output from BARC code")
    print("   - Rendered as input -> output visualization")
    print("   - File: temp_eval_result.png")
    
    print("\n3. PROMPT TEXT:")
    prompt = f"""Look at these two images:
1. The first image shows the original ARC training examples
2. The second image shows our generated solution applied to the test input

Code Description: "{mock_barc_output.description or 'No description provided'}"
Code Concepts: "{mock_barc_output.concepts or 'No concepts listed'}"

<think>
Analyze by comparing the two images:

1. UNDERSTANDING CHECK: Does the description match the pattern shown in the training examples?
   - Look at the training examples in the first image
   - Check if the description accurately explains the transformation pattern
   - Verify if the concepts align with what you observe

2. CALCULATION CHECK: Does the generated output follow the same transformation rule as the training examples?
   - Compare the transformation pattern from training examples
   - Check if the generated output in the second image follows the same logic
   - Look for any calculation or logic errors

3. ANSWER COMPLETENESS: Is the generated output complete and properly formatted?
   - Check if the output provides a definitive solution (not partial)
   - Verify if all necessary transformations are applied
   - Ensure the output format is correct

4. ANSWER CORRECT: Does the transformation logic appear correct based on the pattern?
   - Assess overall correctness of the approach
   - Check consistency with training examples
</think>

<answer>
Provide your evaluation in the following format:

UNDERSTANDING_CHECK: [TRUE/FALSE]
CALCULATION_CHECK: [TRUE/FALSE] 
ANSWER_COMPLETENESS: [TRUE/FALSE]
ANSWER_CORRECT: [TRUE/FALSE]

FEEDBACK: [Brief explanation of your assessment]
</answer>"""
    
    print(prompt)
    
    print("\n" + "="*80)
    print("GLM RESPONSE PARSING")
    print("="*80)
    
    print("\nGLM would respond with something like:")
    mock_glm_response = """<think>
Looking at the two images:

1. The training examples show that blue cells (color 1) remain in place, and red cells (color 2) appear around them in specific patterns. The description mentions "surrounding pattern" which matches what I see.

2. In the solution result, the test input has been transformed following the same logic - blue cells are preserved and red cells have been added around them.

3. The output appears complete and properly formatted as a grid.

4. The transformation logic seems consistent with the training examples.
</think>

<answer>
UNDERSTANDING_CHECK: TRUE
CALCULATION_CHECK: TRUE
ANSWER_COMPLETENESS: TRUE
ANSWER_CORRECT: TRUE

FEEDBACK: The solution correctly implements the pattern of surrounding blue cells with red cells. The description accurately describes the transformation and the implementation follows the observed pattern from training examples.
</answer>"""
    
    print(mock_glm_response)
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    print("""
The GLM response gets parsed into VerificationResult objects:

understanding_check: VerificationResult(passed=True, confidence=0.8, feedback="UNDERSTANDING_CHECK: The solution correctly implements...")
calculation_check: VerificationResult(passed=True, confidence=0.8, feedback="CALCULATION_CHECK: The solution correctly implements...")
answer_completeness: VerificationResult(passed=True, confidence=0.8, feedback="ANSWER_COMPLETENESS: The solution correctly implements...")
answer_correct: VerificationResult(passed=True, confidence=0.8, feedback="ANSWER_CORRECT: The solution correctly implements...")

These are then converted to scores:
- understanding_check: 0.0 (passed, so no penalty)
- calculation_check: 0.0 (passed, so no penalty)  
- answer_completeness: 0.0 (passed, so no penalty)
- answer_correct: 0.0 (passed, so no penalty)

Total reward: 0.0 (no penalties, perfect score)
""")

def show_mock_glm_prompt():
    """Show mock GLM prompt when ARC data is not available"""
    print("\n" + "="*80)
    print("MOCK GLM DUAL IMAGE PROMPT EXAMPLE")
    print("="*80)
    
    # Create a mock BARC output for demonstration
    mock_barc_output = BARCOutput(
        code="""
def solve(grid):
    # Pattern: Each blue cell (color 1) gets surrounded by red cells (color 2)
    result = grid.copy()
    height, width = grid.shape
    
    for i in range(height):
        for j in range(width):
            if grid[i][j] == 1:  # Blue cell
                # Add red cells around it
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < height and 0 <= nj < width:
                            if result[ni][nj] == 0:  # Only if currently black
                                result[ni][nj] = 2  # Make it red
    return result
""",
        concepts="pattern_completion, color_surrounding, spatial_relationships",
        description="The transformation adds red cells (color 2) around each blue cell (color 1), forming a surrounding pattern while preserving the original blue cells.",
        plan="1. Identify all blue cells in the grid 2. For each blue cell, add red cells in all adjacent positions 3. Only change black cells to red, preserve existing colors",
        raw_response="Looking at this pattern, I need to... [full BARC response would be longer]"
    )
    
    print("\n1. FIRST IMAGE: ARC Problem Training Examples")
    print("   - Shows the original training input-output pairs")
    print("   - Rendered as a visualization with input and output grids side by side")
    print("   - File: temp_eval_problem.png")
    
    print("\n2. SECOND IMAGE: Solution Result")
    print("   - Shows test input and generated output from BARC code")
    print("   - Rendered as input -> output visualization")
    print("   - File: temp_eval_result.png")
    
    print("\n3. PROMPT TEXT:")
    prompt = f"""Look at these two images:
1. The first image shows the original ARC training examples
2. The second image shows our generated solution applied to the test input

Code Description: "{mock_barc_output.description or 'No description provided'}"
Code Concepts: "{mock_barc_output.concepts or 'No concepts listed'}"

<think>
Analyze by comparing the two images:

1. UNDERSTANDING CHECK: Does the description match the pattern shown in the training examples?
   - Look at the training examples in the first image
   - Check if the description accurately explains the transformation pattern
   - Verify if the concepts align with what you observe

2. CALCULATION CHECK: Does the generated output follow the same transformation rule as the training examples?
   - Compare the transformation pattern from training examples
   - Check if the generated output in the second image follows the same logic
   - Look for any calculation or logic errors

3. ANSWER COMPLETENESS: Is the generated output complete and properly formatted?
   - Check if the output provides a definitive solution (not partial)
   - Verify if all necessary transformations are applied
   - Ensure the output format is correct

4. ANSWER CORRECT: Does the transformation logic appear correct based on the pattern?
   - Assess overall correctness of the approach
   - Check consistency with training examples
</think>

<answer>
Provide your evaluation in the following format:

UNDERSTANDING_CHECK: [TRUE/FALSE]
CALCULATION_CHECK: [TRUE/FALSE] 
ANSWER_COMPLETENESS: [TRUE/FALSE]
ANSWER_CORRECT: [TRUE/FALSE]

FEEDBACK: [Brief explanation of your assessment]
</answer>"""
    
    print(prompt)
    
    print("\n" + "="*80)
    print("GLM RESPONSE PARSING")
    print("="*80)
    
    print("\nGLM would respond with something like:")
    mock_glm_response = """<think>
Looking at the two images:

1. The training examples show that blue cells (color 1) remain in place, and red cells (color 2) appear around them in specific patterns. The description mentions "surrounding pattern" which matches what I see.

2. In the solution result, the test input has been transformed following the same logic - blue cells are preserved and red cells have been added around them.

3. The output appears complete and properly formatted as a grid.

4. The transformation logic seems consistent with the training examples.
</think>

<answer>
UNDERSTANDING_CHECK: TRUE
CALCULATION_CHECK: TRUE
ANSWER_COMPLETENESS: TRUE
ANSWER_CORRECT: TRUE

FEEDBACK: The solution correctly implements the pattern of surrounding blue cells with red cells. The description accurately describes the transformation and the implementation follows the observed pattern from training examples.
</answer>"""
    
    print(mock_glm_response)
    
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)
    
    print("""
The GLM response gets parsed into VerificationResult objects:

understanding_check: VerificationResult(passed=True, confidence=0.8, feedback="UNDERSTANDING_CHECK: The solution correctly implements...")
calculation_check: VerificationResult(passed=True, confidence=0.8, feedback="CALCULATION_CHECK: The solution correctly implements...")
answer_completeness: VerificationResult(passed=True, confidence=0.8, feedback="ANSWER_COMPLETENESS: The solution correctly implements...")
answer_correct: VerificationResult(passed=True, confidence=0.8, feedback="ANSWER_CORRECT: The solution correctly implements...")

These are then converted to scores:
- understanding_check: 0.0 (passed, so no penalty)
- calculation_check: 0.0 (passed, so no penalty)  
- answer_completeness: 0.0 (passed, so no penalty)
- answer_correct: 0.0 (passed, so no penalty)

Total reward: 0.0 (no penalties, perfect score)
""")

if __name__ == "__main__":
    show_glm_prompt_example()