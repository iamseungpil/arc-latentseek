#!/bin/bash

# Create timestamped directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU5_DIR="comparative_results_gpu5_${TIMESTAMP}"
GPU6_DIR="comparative_results_gpu6_${TIMESTAMP}"

echo "Creating result directories..."
mkdir -p "${GPU5_DIR}"/{logs,visualizations}
mkdir -p "${GPU6_DIR}"/{logs,visualizations}

# Create GPU5 run script
cat > run_gpu5_${TIMESTAMP}.py << EOF
import os
import sys

# Set CUDA device BEFORE imports
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU5 experiment - output dir: ${GPU5_DIR}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    experiment = ComparativeExperimentV2(output_dir="${GPU5_DIR}")
    experiment.run_experiments_for_gpu(5)
EOF

# Create GPU6 run script
cat > run_gpu6_${TIMESTAMP}.py << EOF
import os
import sys

# Set CUDA device BEFORE imports
os.environ['CUDA_VISIBLE_DEVICES'] = '6'

sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU6 experiment - output dir: ${GPU6_DIR}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    experiment = ComparativeExperimentV2(output_dir="${GPU6_DIR}")
    experiment.run_experiments_for_gpu(6)
EOF

# Start tmux sessions
echo "Starting GPU5 experiment in tmux..."
tmux new-session -d -s exp_gpu5 \
    "cd /home/ubuntu/arc-latentseek && \
     source /data/miniforge3/etc/profile.d/conda.sh && \
     conda activate unsloth_env && \
     python run_gpu5_${TIMESTAMP}.py 2>&1 | tee ${GPU5_DIR}/experiment.log"

echo "Starting GPU6 experiment in tmux..."
tmux new-session -d -s exp_gpu6 \
    "cd /home/ubuntu/arc-latentseek && \
     source /data/miniforge3/etc/profile.d/conda.sh && \
     conda activate unsloth_env && \
     python run_gpu6_${TIMESTAMP}.py 2>&1 | tee ${GPU6_DIR}/experiment.log"

echo ""
echo "✅ Experiments started!"
echo ""
echo "GPU5: ${GPU5_DIR}"
echo "GPU6: ${GPU6_DIR}"
echo ""
echo "Monitor: tmux attach -t exp_gpu5 or exp_gpu6"
echo "Logs: tail -f ${GPU5_DIR}/experiment.log"

# Create monitoring script
cat > monitor_${TIMESTAMP}.py << 'MONITOR'
#!/usr/bin/env python
import os, time, json
from datetime import datetime

dirs = ["${GPU5_DIR}", "${GPU6_DIR}"]
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
MONITOR

sed -i "s|\${GPU5_DIR}|${GPU5_DIR}|g" monitor_${TIMESTAMP}.py
sed -i "s|\${GPU6_DIR}|${GPU6_DIR}|g" monitor_${TIMESTAMP}.py
chmod +x monitor_${TIMESTAMP}.py

echo ""
echo "Monitor with: python monitor_${TIMESTAMP}.py"