#!/bin/bash
# Run V19 experiment on GPU 5

# Create results directory
mkdir -p results/v19

# Set GPU 5
export CUDA_VISIBLE_DEVICES=5

# Run experiment
echo "Starting V19 Retry Until Valid experiment on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v19_gpu5 "python experiment_v19_retry_valid.py 2>&1 | tee results/v19/experiment.log"

echo "Experiment started in tmux session 'v19_gpu5'"
echo "To attach: tmux attach -t v19_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v19/experiment.log"