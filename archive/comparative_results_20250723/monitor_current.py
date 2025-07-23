#!/usr/bin/env python
"""Monitor current experiment with focus on description changes"""

import os
import time
import json
from datetime import datetime

# Current result directories
GPU5_DIR = "comparative_results_gpu5_20250723_041753"
GPU6_DIR = "comparative_results_gpu6_20250723_041753"

def monitor():
    print(f"\n{'='*80}")
    print(f"Experiment Monitor - Started at {datetime.now()}")
    print(f"Monitoring: {GPU5_DIR} and {GPU6_DIR}")
    print(f"{'='*80}\n")
    
    seen_files = set()
    problem_descriptions = {}  # Track description changes
    
    while True:
        try:
            for gpu_id, result_dir in [(5, GPU5_DIR), (6, GPU6_DIR)]:
                log_dir = os.path.join(result_dir, "logs")
                
                if not os.path.exists(log_dir):
                    continue
                
                # Check for new files
                for filename in sorted(os.listdir(log_dir)):
                    if not filename.endswith(".json"):
                        continue
                    
                    filepath = os.path.join(log_dir, filename)
                    
                    if filepath in seen_files:
                        continue
                    
                    seen_files.add(filepath)
                    
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                        
                        # Extract info
                        problem_id = data.get('problem_id', 'unknown')
                        condition = data.get('condition', 'unknown')
                        candidate = data.get('candidate_idx', 0)
                        step = data.get('step', 0)
                        accuracy = data.get('accuracy', 0) * 100
                        reward = data.get('reward', 0)
                        description = data.get('description', '')
                        
                        # Create unique key
                        exp_key = f"{problem_id}_{condition}_c{candidate}"
                        
                        # Print update
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU{gpu_id} | {condition} | {problem_id} C{candidate} Step {step}")
                        print(f"  Accuracy: {accuracy:.1f}% | Reward: {reward:.3f}")
                        
                        # Check description changes
                        if exp_key in problem_descriptions:
                            prev_desc = problem_descriptions[exp_key]
                            if prev_desc != description:
                                print(f"  📝 DESCRIPTION CHANGED!")
                                print(f"     Previous: {prev_desc[:60]}...")
                                print(f"     New:      {description[:60]}...")
                        else:
                            print(f"  Initial description: {description[:60]}...")
                        
                        problem_descriptions[exp_key] = description
                        
                        if accuracy >= 100:
                            print(f"  ✨ PERFECT ACCURACY! ✨")
                        elif accuracy > 0:
                            print(f"  📈 Progress: {accuracy:.1f}% accuracy")
                        
                        print()
                        
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
            
            # Also check experiment logs for optimizer activity
            for gpu_id, result_dir in [(5, GPU5_DIR), (6, GPU6_DIR)]:
                log_file = os.path.join(result_dir, "experiment.log")
                if os.path.exists(log_file):
                    try:
                        # Get file size to check if growing
                        size = os.path.getsize(log_file)
                        if size > 0:
                            # Check last few lines for description activity
                            with open(log_file, 'r') as f:
                                f.seek(max(0, size - 5000))  # Read last 5KB
                                recent = f.read()
                                
                                if "Description optimization" in recent or "description tokens:" in recent:
                                    lines = recent.split('\n')
                                    for line in lines[-10:]:
                                        if "description" in line.lower() and "INFO" in line:
                                            print(f"[LOG GPU{gpu_id}] {line.strip()}")
                    except:
                        pass
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor()