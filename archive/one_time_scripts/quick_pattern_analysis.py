#!/usr/bin/env python3
"""
Quick analysis of 2a5f8217 pattern without loading heavy models
"""

import sys
import numpy as np
from src.data import ARCDataLoader
import time

def analyze_2a5f8217_pattern():
    """Analyze the pattern in problem 2a5f8217"""
    
    print("=== 2a5f8217 PATTERN ANALYSIS ===")
    
    # Load problem
    loader = ARCDataLoader()
    problem = loader.get_problem_by_id("2a5f8217")
    
    print(f"Problem: {problem.uid}")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    analysis_file = f"results/pattern_analysis_{timestamp}.txt"
    
    with open(analysis_file, "w") as f:
        f.write("=== 2a5f8217 PATTERN ANALYSIS ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        
        # Analyze each training pair
        for i, pair in enumerate(problem.train_pairs):
            print(f"\n=== PAIR {i} ANALYSIS ===")
            f.write(f"=== PAIR {i} ANALYSIS ===\n")
            
            print(f"Input shape: {pair.x.shape}")
            print(f"Output shape: {pair.y.shape}")
            f.write(f"Input shape: {pair.x.shape}\n")
            f.write(f"Output shape: {pair.y.shape}\n")
            
            # Color analysis
            input_colors = np.unique(pair.x)
            output_colors = np.unique(pair.y)
            print(f"Input colors: {input_colors}")
            print(f"Output colors: {output_colors}")
            f.write(f"Input colors: {input_colors}\n")
            f.write(f"Output colors: {output_colors}\n")
            
            # Blue transformation analysis
            if 1 in input_colors:  # Blue present in input
                blue_positions = np.where(pair.x == 1)
                print(f"Blue pixels in input: {len(blue_positions[0])}")
                f.write(f"Blue pixels in input: {len(blue_positions[0])}\n")
                
                if 1 not in output_colors:  # Blue disappears
                    print("RULE: Blue pixels are transformed to other colors")
                    f.write("RULE: Blue pixels are transformed to other colors\n")
                    
                    # Analyze transformation for each blue pixel
                    transformation_map = {}
                    for j in range(len(blue_positions[0])):
                        r, c = blue_positions[0][j], blue_positions[1][j]
                        replacement_color = pair.y[r, c]
                        
                        if replacement_color not in transformation_map:
                            transformation_map[replacement_color] = []
                        transformation_map[replacement_color].append((r, c))
                    
                    print("Blue transformation map:")
                    f.write("Blue transformation map:\n")
                    for color, positions in transformation_map.items():
                        print(f"  Color {color}: {len(positions)} pixels at {positions[:3]}{'...' if len(positions) > 3 else ''}")
                        f.write(f"  Color {color}: {len(positions)} pixels at {positions[:3]}{'...' if len(positions) > 3 else ''}\n")
                    
                    # Find nearest neighbor pattern
                    print("\nNearest neighbor analysis:")
                    f.write("\nNearest neighbor analysis:\n")
                    
                    for j in range(min(5, len(blue_positions[0]))):  # Analyze first 5 blue pixels
                        r, c = blue_positions[0][j], blue_positions[1][j]
                        replacement_color = pair.y[r, c]
                        
                        # Find all non-blue, non-black pixels and their distances
                        distances = []
                        for rr in range(pair.x.shape[0]):
                            for cc in range(pair.x.shape[1]):
                                if pair.x[rr, cc] not in [0, 1]:  # Not black, not blue
                                    manhattan_dist = abs(rr - r) + abs(cc - c)
                                    euclidean_dist = np.sqrt((rr - r)**2 + (cc - c)**2)
                                    distances.append((manhattan_dist, euclidean_dist, pair.x[rr, cc], rr, cc))
                        
                        distances.sort()  # Sort by Manhattan distance
                        
                        print(f"  Blue at ({r},{c}) -> Color {replacement_color}")
                        f.write(f"  Blue at ({r},{c}) -> Color {replacement_color}\n")
                        
                        if distances:
                            nearest = distances[0]
                            print(f"    Nearest non-blue: Color {nearest[2]} at ({nearest[3]},{nearest[4]}) (Manhattan: {nearest[0]}, Euclidean: {nearest[1]:.2f})")
                            f.write(f"    Nearest non-blue: Color {nearest[2]} at ({nearest[3]},{nearest[4]}) (Manhattan: {nearest[0]}, Euclidean: {nearest[1]:.2f})\n")
                            
                            if nearest[2] == replacement_color:
                                print(f"    ✅ MATCHES! Blue becomes nearest neighbor color")
                                f.write(f"    ✅ MATCHES! Blue becomes nearest neighbor color\n")
                            else:
                                print(f"    ❌ MISMATCH! Expected {nearest[2]}, got {replacement_color}")
                                f.write(f"    ❌ MISMATCH! Expected {nearest[2]}, got {replacement_color}\n")
                        
                        # Show top 3 nearest for context
                        if len(distances) > 1:
                            print(f"    Top 3 nearest: {[(d[2], d[0]) for d in distances[:3]]}")
                            f.write(f"    Top 3 nearest: {[(d[2], d[0]) for d in distances[:3]]}\n")
            
            print("-" * 50)
            f.write("-" * 50 + "\n")
        
        # Final rule hypothesis
        print(f"\n=== RULE HYPOTHESIS ===")
        f.write(f"\n=== RULE HYPOTHESIS ===\n")
        
        rule = """
DISCOVERED RULE:
For each blue pixel in the input:
1. Find the nearest non-blue, non-black pixel using Manhattan distance
2. Replace the blue pixel with the color of that nearest pixel
3. If there are ties in distance, the algorithm chooses consistently

This is a "nearest neighbor color replacement" transformation.
"""
        print(rule)
        f.write(rule + "\n")
    
    print(f"\nDetailed analysis saved to: {analysis_file}")

if __name__ == "__main__":
    analyze_2a5f8217_pattern()