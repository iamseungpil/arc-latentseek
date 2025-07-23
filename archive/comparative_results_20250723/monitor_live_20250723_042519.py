#!/usr/bin/env python
"""Live monitoring of experiment progress"""

import os
import time
import json
from datetime import datetime

# Result directories
GPU5_DIR = "comparative_results_gpu5_20250723_042519"
GPU6_DIR = "comparative_results_gpu6_20250723_042519"

def monitor():
    print(f"\n{'='*80}")
    print(f"Experiment Monitor - {datetime.now()}")
    print(f"GPU5: {GPU5_DIR}")
    print(f"GPU6: {GPU6_DIR}")
    print(f"{'='*80}\n")
    
    seen_files = set()
    prev_descriptions = {}
    
    while True:
        try:
            for gpu_id, result_dir in [(5, GPU5_DIR), (6, GPU6_DIR)]:
                log_dir = os.path.join(result_dir, "logs")
                
                if not os.path.exists(log_dir):
                    continue
                
                # Check for new log files
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
                        problem_id = data.get('problem_id', '')
                        condition = data.get('condition', '')
                        step = data.get('step', 0)
                        accuracy = data.get('accuracy', 0) * 100
                        reward = data.get('reward', 0)
                        description = data.get('description', '')
                        
                        # Print update
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU{gpu_id} | {condition} | {problem_id} Step {step}")
                        print(f"  Accuracy: {accuracy:.1f}% | Reward: {reward:.3f}")
                        
                        # Track description changes
                        key = f"{gpu_id}_{problem_id}_{condition}"
                        if key in prev_descriptions and prev_descriptions[key] != description:
                            print(f"  📝 DESCRIPTION CHANGED!")
                            print(f"     From: {prev_descriptions[key][:60]}...")
                            print(f"     To:   {description[:60]}...")
                        
                        prev_descriptions[key] = description
                        
                        if accuracy >= 100:
                            print(f"  ✨ PERFECT ACCURACY ACHIEVED! ✨")
                        
                        print()
                        
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
            
            time.sleep(3)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break

if __name__ == "__main__":
    monitor()
