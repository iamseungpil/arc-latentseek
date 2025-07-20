#!/usr/bin/env python3
"""
Run improved experiment starting from where we left off
"""

import sys
sys.path.append('/home/ubuntu/arc-latentseek')

from src.data import ARCDataLoader

# Get all validation problems
loader = ARCDataLoader()
all_problems = loader.get_problems(split="validation")

# Problems already processed
processed_ids = {
    '136b0064', '2072aba6', '40f6cd08', '7039b2d7',
    '712bf12e', 'bb52a14b', 'ea9794b1', 'f5aa3634'
}

# Filter out processed problems
remaining_problems = [p for p in all_problems if p.uid not in processed_ids]

print(f"Total validation problems: {len(all_problems)}")
print(f"Already processed: {len(processed_ids)}")
print(f"Remaining: {len(remaining_problems)}")

# Save remaining problem IDs to file
with open('remaining_problems.txt', 'w') as f:
    for p in remaining_problems:
        f.write(f"{p.uid}\n")

print(f"\nRemaining problem IDs saved to remaining_problems.txt")
print(f"First 10 remaining problems:")
for i, p in enumerate(remaining_problems[:10]):
    print(f"  {i+1}. {p.uid}")