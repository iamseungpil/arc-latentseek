#!/bin/bash
# Run V16 experiment on GPU 5

# Create results directory
mkdir -p results/v16

# Set GPU 5
export CUDA_VISIBLE_DEVICES=5

# Run experiment
echo "Starting V16 experiment on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v16_gpu5 "python experiment_v16.py 2>&1 | tee results/v16/experiment.log"

echo "Experiment started in tmux session 'v16_gpu5'"
echo "To attach: tmux attach -t v16_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v16/experiment.log"