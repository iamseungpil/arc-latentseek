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
import sys
sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU5 experiment - output dir: ${GPU5_DIR}")
    experiment = ComparativeExperimentV2(output_dir="${GPU5_DIR}")
    experiment.run_experiments_for_gpu(5)
EOF

# Create GPU6 run script
cat > run_gpu6_${TIMESTAMP}.py << EOF
import sys
sys.path.insert(0, '/home/ubuntu/arc-latentseek')
from comparative_experiment_v2 import ComparativeExperimentV2

if __name__ == "__main__":
    print(f"Starting GPU6 experiment - output dir: ${GPU6_DIR}")
    experiment = ComparativeExperimentV2(output_dir="${GPU6_DIR}")
    experiment.run_experiments_for_gpu(6)
EOF

# Start tmux sessions
echo "Starting GPU5 experiment in tmux..."
tmux new-session -d -s exp_gpu5 "cd /home/ubuntu/arc-latentseek && source /data/miniforge3/etc/profile.d/conda.sh && conda activate unsloth_env && CUDA_VISIBLE_DEVICES=5 python run_gpu5_${TIMESTAMP}.py 2>&1 | tee ${GPU5_DIR}/experiment.log"

echo "Starting GPU6 experiment in tmux..."
tmux new-session -d -s exp_gpu6 "cd /home/ubuntu/arc-latentseek && source /data/miniforge3/etc/profile.d/conda.sh && conda activate unsloth_env && CUDA_VISIBLE_DEVICES=6 python run_gpu6_${TIMESTAMP}.py 2>&1 | tee ${GPU6_DIR}/experiment.log"

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