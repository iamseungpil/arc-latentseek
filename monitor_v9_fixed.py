#!/usr/bin/env python3
"""Monitor V9 Fixed experiment"""

import json
from pathlib import Path

def monitor_v9_fixed():
    results_file = Path("results/premain_v9_fixed/results.json")
    
    if not results_file.exists():
        print("No results file found yet")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("V9 Fixed Experiment Status")
    print("="*50)
    
    if "summary" in results:
        summary = results["summary"]
        print(f"Problems processed: {summary['total_problems']}")
        print(f"Average initial accuracy: {summary['average_initial_accuracy']:.1%}")
        print(f"Average final accuracy: {summary['average_final_accuracy']:.1%}")
        print(f"Average improvement: {summary['average_improvement']:.1%}")
        print()
    
    print("Individual Problems:")
    print("-"*50)
    
    for problem in results.get("problems", []):
        uid = problem["uid"]
        status = problem.get("status", "unknown")
        initial_acc = problem.get("initial_accuracy", 0)
        final_acc = problem.get("final_accuracy", 0)
        improvement = final_acc - initial_acc
        
        print(f"{uid}: {initial_acc:.1%} → {final_acc:.1%} ({improvement:+.1%}) [{status}]")
        
        # Check premain history
        if "premain_history" in problem:
            history = problem["premain_history"]
            if len(history) > 1:
                # Check if premain content is being preserved
                first_non_empty = None
                for i, content in enumerate(history):
                    if content.strip():
                        if first_non_empty is None:
                            first_non_empty = i
                        elif i > first_non_empty and not content.strip():
                            print(f"  ⚠️  Pre-main content lost after step {i}")
                            break

if __name__ == "__main__":
    monitor_v9_fixed()