#!/usr/bin/env python3
"""Live monitoring of experiment progress with description tracking"""

import os
import time
import json
from datetime import datetime

# Result directories
GPU5_DIR = "comparative_results_gpu5_20250723_040538"
GPU6_DIR = "comparative_results_gpu6_20250723_040538"

def monitor():
    print(f"\n{'='*80}")
    print(f"Live Experiment Monitor - Started at {datetime.now()}")
    print(f"{'='*80}\n")
    
    seen_files = set()
    
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
                        accuracy = data.get('accuracy', 0) * 100  # Convert to percentage
                        reward = data.get('reward', 0)
                        description = data.get('description', '')
                        
                        # Print update
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] GPU{gpu_id} | {condition} | {problem_id} C{candidate}")
                        print(f"  Step {step}: Accuracy={accuracy:.1f}%, Reward={reward:.3f}")
                        
                        if description:
                            print(f"  Description: {description[:80]}...")
                        
                        if accuracy >= 100:
                            print(f"  ✨ PERFECT ACCURACY ACHIEVED! ✨")
                        elif step > 0 and accuracy > 0:
                            print(f"  📈 Non-zero accuracy: {accuracy:.1f}%")
                        
                        print()
                        
                    except Exception as e:
                        print(f"Error reading {filename}: {e}")
            
            # Check experiment logs
            for gpu_id, result_dir in [(5, GPU5_DIR), (6, GPU6_DIR)]:
                log_file = os.path.join(result_dir, "experiment.log")
                if os.path.exists(log_file):
                    # Get last 5 lines that contain description info
                    try:
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            desc_lines = [l for l in lines[-100:] if 'description' in l.lower() or 'Description optimization' in l]
                            if desc_lines:
                                print(f"\nGPU{gpu_id} Recent Description Activity:")
                                for line in desc_lines[-3:]:
                                    print(f"  {line.strip()}")
                    except:
                        pass
            
            time.sleep(3)
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped")
            break
        except Exception as e:
            print(f"Monitor error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    monitor()