#!/bin/bash
# Run V17 Fixed experiment on GPU 5

# Create results directory
mkdir -p results/v17_fixed

# Set GPU 5
export CUDA_VISIBLE_DEVICES=5

# Run experiment
echo "Starting V17 Fixed experiment on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v17_fixed_gpu5 "python experiment_v17_fixed.py 2>&1 | tee results/v17_fixed/experiment.log"

echo "Experiment started in tmux session 'v17_fixed_gpu5'"
echo "To attach: tmux attach -t v17_fixed_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v17_fixed/experiment.log"