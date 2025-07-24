#!/bin/bash

# Kill any existing sessions
tmux kill-session -t exp_gpu5 2>/dev/null
tmux kill-session -t exp_gpu6 2>/dev/null

# Create timestamped directories
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GPU5_DIR="comparative_results_gpu5_${TIMESTAMP}"
GPU6_DIR="comparative_results_gpu6_${TIMESTAMP}"

echo "Creating result directories..."
mkdir -p "${GPU5_DIR}"/{logs,visualizations}
mkdir -p "${GPU6_DIR}"/{logs,visualizations}

# Create GPU5 run script with proper device mapping
cat > run_gpu5_${TIMESTAMP}.py << EOF
import os
import sys

# IMPORTANT: Set CUDA device BEFORE any torch/transformers imports
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Now import after setting environment
sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU5 experiment - output dir: ${GPU5_DIR}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    experiment = ComparativeExperimentV2(output_dir="${GPU5_DIR}")
    experiment.run_experiments_for_gpu(5)
EOF

# Create GPU6 run script with proper device mapping
cat > run_gpu6_${TIMESTAMP}.py << EOF
import os
import sys

# IMPORTANT: Set CUDA device BEFORE any torch/transformers imports
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Now import after setting environment
sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU6 experiment - output dir: ${GPU6_DIR}")
    print(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    experiment = ComparativeExperimentV2(output_dir="${GPU6_DIR}")
    experiment.run_experiments_for_gpu(6)
EOF

# Start tmux sessions with explicit GPU assignment
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
echo "✅ Experiments started successfully!"
echo ""
echo "Result directories:"
echo "  GPU5: ${GPU5_DIR}"
echo "  GPU6: ${GPU6_DIR}"
echo ""
echo "Monitor experiments:"
echo "  tmux attach -t exp_gpu5"
echo "  tmux attach -t exp_gpu6"
echo ""
echo "View logs:"
echo "  tail -f ${GPU5_DIR}/experiment.log"
echo "  tail -f ${GPU6_DIR}/experiment.log"
echo ""
echo "Monitor with Python script:"
echo "  python monitor_live_${TIMESTAMP}.py"

# Create monitoring script
cat > monitor_live_${TIMESTAMP}.py << 'EOF'
#!/usr/bin/env python
"""Live monitoring of experiment progress"""

import os
import time
import json
from datetime import datetime

# Result directories
GPU5_DIR = "${GPU5_DIR}"
GPU6_DIR = "${GPU6_DIR}"

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
EOF

# Replace placeholders in monitor script
sed -i "s|\${GPU5_DIR}|${GPU5_DIR}|g" monitor_live_${TIMESTAMP}.py
sed -i "s|\${GPU6_DIR}|${GPU6_DIR}|g" monitor_live_${TIMESTAMP}.py