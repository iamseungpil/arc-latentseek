#!/bin/bash
# Run V17 experiment on GPU 5

# Create results directory
mkdir -p results/v17

# Set GPU 5
export CUDA_VISIBLE_DEVICES=5

# Run experiment
echo "Starting V17 experiment on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v17_gpu5 "python experiment_v17.py 2>&1 | tee results/v17/experiment.log"

echo "Experiment started in tmux session 'v17_gpu5'"
echo "To attach: tmux attach -t v17_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v17/experiment.log"