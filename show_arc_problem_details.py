#!/usr/bin/env python3
"""
Show actual ARC problem 2a5f8217 details
"""

import sys
import numpy as np
sys.path.append('/home/ubuntu')

from arc import train_problems, validation_problems

def show_problem_details():
    """Show details of ARC problem 2a5f8217"""
    
    # Find the problem
    problem = None
    all_problems = train_problems + validation_problems
    
    for p in all_problems:
        if str(p.uid) == "2a5f8217":
            problem = p
            break
    
    if problem is None:
        print("Problem 2a5f8217 not found")
        return
    
    print(f"ARC Problem: {problem.uid}")
    print("="*50)
    
    # Show training examples
    for i, pair in enumerate(problem.train_pairs):
        print(f"\nTraining Example {i+1}:")
        print("Input:")
        print(pair.x)
        print("Output:")
        print(pair.y)
        print(f"Transformation: {pair.x.shape} -> {pair.y.shape}")
    
    # Show test example
    if problem.test_pairs:
        print(f"\nTest Example:")
        print("Input:")
        print(problem.test_pairs[0].x)
        print("Expected Output:")
        print(problem.test_pairs[0].y)
        print(f"Transformation: {problem.test_pairs[0].x.shape} -> {problem.test_pairs[0].y.shape}")

if __name__ == "__main__":
    show_problem_details()