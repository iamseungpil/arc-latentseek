#!/usr/bin/env python3
"""Monitor ongoing experiments and track description changes and accuracy improvements"""

import json
import os
import time
from datetime import datetime
from collections import defaultdict

def monitor_results():
    """Monitor experiment results in real-time"""
    
    result_dirs = [
        "/home/ubuntu/arc-latentseek/comparative_results_gpu5",
        "/home/ubuntu/arc-latentseek/comparative_results_gpu6"
    ]
    
    # Track seen files and their content
    seen_files = set()
    problem_states = defaultdict(lambda: {"steps": [], "best_accuracy": 0})
    
    print(f"\n{'='*80}")
    print(f"Monitoring experiments started at {datetime.now()}")
    print(f"{'='*80}\n")
    
    while True:
        try:
            for result_dir in result_dirs:
                if not os.path.exists(result_dir):
                    continue
                    
                log_dir = os.path.join(result_dir, "logs")
                if not os.path.exists(log_dir):
                    continue
                
                # Check all log files
                for filename in os.listdir(log_dir):
                    if not filename.endswith(".json"):
                        continue
                    
                    filepath = os.path.join(log_dir, filename)
                    
                    # Skip if already processed
                    if filepath in seen_files:
                        continue
                    
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        seen_files.add(filepath)
                        
                        # Extract key info
                        problem_id = data.get('problem_id')
                        condition = data.get('condition')
                        step = data.get('step')
                        accuracy = data.get('accuracy', 0)
                        reward = data.get('reward', 0)
                        description = data.get('description', '')
                        
                        # Create unique key for this experiment
                        exp_key = f"{problem_id}_{condition}_c{data.get('candidate_idx')}"
                        
                        # Store step info
                        step_info = {
                            'step': step,
                            'accuracy': accuracy,
                            'reward': reward,
                            'description': description[:100] + "..." if len(description) > 100 else description
                        }
                        
                        problem_states[exp_key]["steps"].append(step_info)
                        
                        # Check for improvements
                        if accuracy > problem_states[exp_key]["best_accuracy"]:
                            problem_states[exp_key]["best_accuracy"] = accuracy
                            improvement = True
                        else:
                            improvement = False
                        
                        # Print update
                        gpu = "GPU5" if "gpu5" in result_dir else "GPU6"
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] {gpu} | {exp_key}")
                        print(f"  Step {step}: Accuracy={accuracy:.1%}, Reward={reward:.3f}")
                        
                        if improvement and accuracy > 0:
                            print(f"  🎯 ACCURACY IMPROVED! Best so far: {accuracy:.1%}")
                        
                        if accuracy >= 1.0:
                            print(f"  ✨ PERFECT ACCURACY ACHIEVED!")
                        
                        # Check description changes
                        if len(problem_states[exp_key]["steps"]) > 1:
                            prev_desc = problem_states[exp_key]["steps"][-2]['description']
                            if prev_desc != step_info['description']:
                                print(f"  📝 Description changed!")
                                print(f"     Old: {prev_desc}")
                                print(f"     New: {step_info['description']}")
                        
                        print()
                        
                    except Exception as e:
                        print(f"Error reading {filepath}: {e}")
            
            # Also show summary every 30 seconds
            if int(time.time()) % 30 == 0:
                print(f"\n{'='*60}")
                print(f"SUMMARY at {datetime.now().strftime('%H:%M:%S')}")
                print(f"{'='*60}")
                
                for exp_key, state in problem_states.items():
                    if state["steps"]:
                        latest = state["steps"][-1]
                        print(f"{exp_key}: Step {latest['step']}, Accuracy={latest['accuracy']:.1%}, Best={state['best_accuracy']:.1%}")
                
                print(f"{'='*60}\n")
            
            time.sleep(2)  # Check every 2 seconds
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor_results()