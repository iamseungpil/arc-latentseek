#!/bin/bash
# Run V14 experiment on GPU 5

# Create results directory
mkdir -p results/v14_gpu5

# Set GPU 5
export CUDA_VISIBLE_DEVICES=5

# Run experiment
echo "Starting V14 experiment on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v14_gpu5 "python experiment_v14.py 2>&1 | tee results/v14_gpu5/experiment.log"

echo "Experiment started in tmux session 'v14_gpu5'"
echo "To attach: tmux attach -t v14_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v14_gpu5/experiment.log"