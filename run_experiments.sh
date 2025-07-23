#!/bin/bash
# Script to run comparative experiments on GPUs 5 and 6

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_comparative_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

echo "Starting experiments at $(date)"
echo "Results will be saved to: $RESULTS_DIR"

# Kill any existing tmux sessions for gpu5/gpu6
tmux kill-session -t gpu5_experiment 2>/dev/null
tmux kill-session -t gpu6_experiment 2>/dev/null

# Start experiment on GPU 5
echo "Starting experiment on GPU 5..."
tmux new-session -d -s gpu5_experiment -c /home/ubuntu/arc-latentseek \
    "source ~/.bashrc && \
     conda activate unsloth_env && \
     export CUDA_VISIBLE_DEVICES=5 && \
     python comparative_experiment_v2.py --results_dir $RESULTS_DIR --gpu_id 5 2>&1 | tee ${RESULTS_DIR}/gpu5_log.txt"

# Start experiment on GPU 6
echo "Starting experiment on GPU 6..."
tmux new-session -d -s gpu6_experiment -c /home/ubuntu/arc-latentseek \
    "source ~/.bashrc && \
     conda activate unsloth_env && \
     export CUDA_VISIBLE_DEVICES=6 && \
     python comparative_experiment_v2.py --results_dir $RESULTS_DIR --gpu_id 6 2>&1 | tee ${RESULTS_DIR}/gpu6_log.txt"

echo "Experiments started in tmux sessions:"
echo "  - gpu5_experiment"
echo "  - gpu6_experiment"
echo ""
echo "To monitor:"
echo "  tmux attach -t gpu5_experiment"
echo "  tmux attach -t gpu6_experiment"
echo ""
echo "Logs are being saved to:"
echo "  ${RESULTS_DIR}/gpu5_log.txt"
echo "  ${RESULTS_DIR}/gpu6_log.txt"