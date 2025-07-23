#!/bin/bash
cd /home/ubuntu/arc-latentseek

# Create timestamped results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_comparative_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

echo "Starting experiments at $(date)"
echo "Results will be saved to: $RESULTS_DIR"

# Kill any existing tmux sessions
tmux kill-session -t gpu5_experiment 2>/dev/null
tmux kill-session -t gpu6_experiment 2>/dev/null

# Start GPU 5 experiment in tmux
echo "Starting GPU 5 experiment..."
tmux new-session -d -s gpu5_experiment -c /home/ubuntu/arc-latentseek \
    "export PATH='/data/miniforge3/bin:$PATH' && \
     export CUDA_VISIBLE_DEVICES=5 && \
     /data/miniforge3/envs/unsloth_env/bin/python comparative_experiment_v2.py \
     --results_dir $RESULTS_DIR --gpu_id 5 2>&1 | tee ${RESULTS_DIR}/gpu5_log.txt; \
     echo 'Experiment finished. Press any key to exit...'; read"

# Start GPU 6 experiment in tmux
echo "Starting GPU 6 experiment..."
tmux new-session -d -s gpu6_experiment -c /home/ubuntu/arc-latentseek \
    "export PATH='/data/miniforge3/bin:$PATH' && \
     export CUDA_VISIBLE_DEVICES=6 && \
     /data/miniforge3/envs/unsloth_env/bin/python comparative_experiment_v2.py \
     --results_dir $RESULTS_DIR --gpu_id 6 2>&1 | tee ${RESULTS_DIR}/gpu6_log.txt; \
     echo 'Experiment finished. Press any key to exit...'; read"

echo ""
echo "Experiments started successfully!"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t gpu5_experiment"
echo "  tmux attach -t gpu6_experiment"
echo ""
echo "To check logs:"
echo "  tail -f ${RESULTS_DIR}/gpu5_log.txt"
echo "  tail -f ${RESULTS_DIR}/gpu6_log.txt"