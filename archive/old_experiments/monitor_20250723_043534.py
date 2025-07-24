#!/usr/bin/env python
import os, time, json
from datetime import datetime

dirs = ["comparative_results_gpu5_20250723_043534", "comparative_results_gpu6_20250723_043534"]
seen = set()
last_desc = {}

print(f"\\nMonitoring experiments at {datetime.now()}")
print("="*80)

while True:
    for i, result_dir in enumerate(dirs):
        gpu_id = 5 + i
        log_dir = os.path.join(result_dir, "logs")
        if os.path.exists(log_dir):
            for f in sorted(os.listdir(log_dir)):
                if f.endswith(".json"):
                    path = os.path.join(log_dir, f)
                    if path not in seen:
                        seen.add(path)
                        try:
                            with open(path) as fp:
                                data = json.load(fp)
                            prob = data.get('problem_id', '')
                            cond = data.get('condition', '')
                            step = data.get('step', 0)
                            acc = data.get('accuracy', 0) * 100
                            desc = data.get('description', '')[:80]
                            
                            print(f"\\n[{datetime.now().strftime('%H:%M:%S')}] GPU{gpu_id} | {cond} | {prob} Step {step}")
                            print(f"  Accuracy: {acc:.1f}%")
                            
                            key = f"{gpu_id}_{prob}_{cond}"
                            if key in last_desc and last_desc[key] != desc:
                                print(f"  📝 Description changed!")
                            last_desc[key] = desc
                            
                            if acc >= 100:
                                print(f"  ✨ PERFECT ACCURACY!")
                        except: pass
    time.sleep(3)
