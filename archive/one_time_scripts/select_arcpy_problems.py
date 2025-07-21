"""
Select 3 arc-py validation problems for experiments
"""
import arc
import random

# Get all validation problem IDs
validation_ids = [p.uid for p in arc.validation_problems]

# Select 3 random problems
random.seed(42)  # For reproducibility
selected = random.sample(validation_ids, 3)

print("Selected arc-py validation problems:")
for pid in selected:
    problem = next(p for p in arc.validation_problems if p.uid == pid)
    print(f"- {pid}: {len(problem.train_pairs)} train examples")

print(f"\nSelected problems: {selected}")