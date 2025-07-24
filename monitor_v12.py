#!/usr/bin/env python3
"""Monitor V12 Majority Voting experiment progress"""

import json
from pathlib import Path
import time

def monitor_v12():
    results_file = Path("results/v12_majority/results.json")
    
    print("V12 Majority Voting Experiment Monitor")
    print("=" * 60)
    
    while True:
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            
            print(f"\nTimestamp: {results.get('timestamp', 'N/A')}")
            print(f"Config: {results['config']['num_candidates']} candidates, "
                  f"{results['config']['num_steps']} steps")
            
            problems = results.get("problems", [])
            print(f"\nProblems processed: {len(problems)}")
            
            if problems:
                print("\nProblem Results:")
                print("-" * 60)
                for p in problems:
                    status_icon = "✓" if p["status"] == "completed" else "✗"
                    print(f"{status_icon} {p['uid']}: "
                          f"{p['initial_accuracy']:.1%} → {p['final_accuracy']:.1%} "
                          f"({p['improvement']:+.1%})")
                    
                    # Show latest history entry if available
                    if "history" in p and p["history"]:
                        latest = p["history"][-1]
                        print(f"  Step {latest['step']}: "
                              f"{latest['num_successful']}/8 successful, "
                              f"confidence: {latest['confidence']:.2f}")
                
                # Summary if available
                if "summary" in results:
                    s = results["summary"]
                    print("\n" + "=" * 60)
                    print("SUMMARY:")
                    print(f"Completed: {s['completed_problems']}/{s['total_problems']}")
                    print(f"Avg Initial: {s['average_initial_accuracy']:.1%}")
                    print(f"Avg Final: {s['average_final_accuracy']:.1%}")
                    print(f"Avg Improvement: {s['average_improvement']:.1%}")
                    break
        else:
            print("Waiting for results file...")
            
        time.sleep(5)
        print("\033[H\033[J", end="")  # Clear screen

if __name__ == "__main__":
    try:
        monitor_v12()
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")