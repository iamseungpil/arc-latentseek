"""
Generate GLM descriptions for selected problems
"""
import sys
import os
sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader
from src.evaluators import GLMEvaluator
from src.executors import GridRenderer
import json

def generate_description_for_problem(problem_id: str, evaluator: GLMEvaluator, loader: ARCDataLoader) -> str:
    """Generate GLM description for a single problem"""
    problem = loader.get_problem_by_id(problem_id)
    
    # Use existing GridRenderer to create visualization
    renderer = GridRenderer()
    image_path = f"/tmp/{problem_id}_description_input.png"
    renderer.render_arc_problem(problem, image_path)
    
    # Prompt GLM to describe the problem
    prompt = """Look at this ARC problem and provide a clear, concise description of the pattern or rule that transforms the input to the output.

Focus on:
1. What pattern appears in the input
2. How the input should be transformed to create the output
3. Any spatial relationships, colors, or geometric operations involved

Provide a description that could be used as guidance for code generation. Be specific but concise."""
    
    # Use internal GLM inference method
    response = evaluator._run_glm_inference(image_path, prompt)
    return response

def main():
    print("Generating GLM descriptions for selected problems...")
    
    # Initialize components
    loader = ARCDataLoader()
    evaluator = GLMEvaluator()
    
    # Selected arc-py validation problems
    problems = ['f3e62deb', '639f5a19', '319f2597']
    
    descriptions = {}
    
    for problem_id in problems:
        print(f"Generating description for {problem_id}...")
        description = generate_description_for_problem(problem_id, evaluator, loader)
        descriptions[problem_id] = description
        print(f"✅ Generated description for {problem_id}")
    
    # Save descriptions
    with open('/home/ubuntu/arc-latentseek/glm_problem_descriptions.json', 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    print("✅ All descriptions generated and saved!")
    print("Descriptions:")
    for pid, desc in descriptions.items():
        print(f"\n{pid}: {desc[:100]}...")

if __name__ == "__main__":
    main()