#!/usr/bin/env python3
"""
Test 2: Alignment 변화 로깅 테스트
"""

import sys
import numpy as np
from src.data import ARCDataLoader
from src.generators import BARCGenerator
from src.alignment import BARCCodeAligner, AlignmentQualityAnalyzer
import time

def test_alignment_changes():
    """Test what changes during the alignment process"""
    
    print("=== ALIGNMENT CHANGE ANALYSIS ===")
    
    # Load problem and generate a BARC candidate
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    print(f"Problem: {problem.uid}")
    
    # Generate one candidate from BARC
    print("\n=== GENERATING BARC CANDIDATE ===")
    barc_gen = BARCGenerator()
    candidates = barc_gen.generate(problem, num_candidates=1, temperature=0.8)
    original_candidate = candidates[0]
    
    print(f"Original code length: {len(original_candidate.code)}")
    print(f"Original concepts: {original_candidate.concepts}")
    print(f"Original description: {original_candidate.description}")
    
    # Apply alignment
    print("\n=== APPLYING ALIGNMENT ===")
    aligner = BARCCodeAligner()
    aligned_candidate = aligner.align_code(original_candidate, problem)
    
    print(f"Aligned code length: {len(aligned_candidate.code)}")
    print(f"Aligned concepts: {aligned_candidate.concepts}")
    print(f"Aligned description: {aligned_candidate.description}")
    
    # Analyze quality
    print("\n=== ALIGNMENT QUALITY ANALYSIS ===")
    quality_analyzer = AlignmentQualityAnalyzer()
    quality = quality_analyzer.analyze_alignment_quality(
        original_candidate.code,
        aligned_candidate.code,
        original_candidate.description,
        aligned_candidate.description
    )
    
    print(f"Improvement score: {quality.improvement_score}")
    print(f"Structure preserved: {quality.structure_preserved}")
    print(f"Has common imports: {quality.has_common_imports}")
    print(f"Has color constants: {quality.has_color_constants}")
    print(f"Uses common functions: {quality.uses_common_functions}")
    print(f"Code length change: {quality.code_length_change}")
    print(f"Description changed: {quality.description_changed}")
    
    # Save detailed comparison
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    comparison_file = f"results/alignment_comparison_{timestamp}.txt"
    
    with open(comparison_file, "w") as f:
        f.write("=== ALIGNMENT COMPARISON ANALYSIS ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Problem: {problem.uid}\n\n")
        
        f.write("=== ORIGINAL CANDIDATE ===\n")
        f.write(f"Code length: {len(original_candidate.code)}\n")
        f.write(f"Concepts: {original_candidate.concepts}\n")
        f.write(f"Description: {original_candidate.description}\n\n")
        f.write("--- ORIGINAL CODE ---\n")
        f.write(original_candidate.code)
        f.write("\n\n")
        
        f.write("=== ALIGNED CANDIDATE ===\n")
        f.write(f"Code length: {len(aligned_candidate.code)}\n")
        f.write(f"Concepts: {aligned_candidate.concepts}\n")
        f.write(f"Description: {aligned_candidate.description}\n\n")
        f.write("--- ALIGNED CODE ---\n")
        f.write(aligned_candidate.code)
        f.write("\n\n")
        
        f.write("=== QUALITY METRICS ===\n")
        f.write(f"Improvement score: {quality.improvement_score}\n")
        f.write(f"Structure preserved: {quality.structure_preserved}\n")
        f.write(f"Has common imports: {quality.has_common_imports}\n")
        f.write(f"Has color constants: {quality.has_color_constants}\n")
        f.write(f"Uses common functions: {quality.uses_common_functions}\n")
        f.write(f"Code length change: {quality.code_length_change}\n")
        f.write(f"Description changed: {quality.description_changed}\n\n")
        
        f.write("=== SPECIFIC CHANGES ===\n")
        if quality.has_common_imports:
            f.write("✅ Added common imports\n")
        if quality.has_color_constants:
            f.write("✅ Uses Color constants\n")
        if quality.uses_common_functions:
            f.write("✅ Uses common functions\n")
        if quality.description_changed:
            f.write("✅ Description was modified\n")
    
    print(f"\nDetailed comparison saved to: {comparison_file}")
    
    return original_candidate, aligned_candidate, quality

if __name__ == "__main__":
    test_alignment_changes()