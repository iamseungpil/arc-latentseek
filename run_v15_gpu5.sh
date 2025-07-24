#!/bin/bash
# Run V15 experiment on GPU 5

# Create results directory
mkdir -p results/v15

# Set GPU 5
export CUDA_VISIBLE_DEVICES=5

# Run experiment
echo "Starting V15 experiment on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v15_gpu5 "python experiment_v15.py 2>&1 | tee results/v15/experiment.log"

echo "Experiment started in tmux session 'v15_gpu5'"
echo "To attach: tmux attach -t v15_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v15/experiment.log"