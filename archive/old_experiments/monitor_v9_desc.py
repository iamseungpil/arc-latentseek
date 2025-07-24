#!/usr/bin/env python3
"""Monitor V9 Description-Only experiment"""

import json
from pathlib import Path

def monitor_v9_desc():
    results_file = Path("results/description_only_v9/results.json")
    
    if not results_file.exists():
        print("No results file found yet")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("V9 Description-Only Experiment Status")
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
        
        # Check description evolution
        if "description_history" in problem:
            history = problem["description_history"]
            print(f"  Description changes: {len(history)} steps")
            if len(history) > 1:
                # Check if description improved
                initial_desc = history[0][:100]
                final_desc = history[-1][:100]
                if initial_desc != final_desc:
                    print(f"  Initial: {initial_desc}...")
                    print(f"  Final: {final_desc}...")

if __name__ == "__main__":
    monitor_v9_desc()