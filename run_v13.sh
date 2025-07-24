#!/bin/bash
# Run V13 experiment on GPU5

# Create results directory
mkdir -p results/v13

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run experiment
echo "Starting V13 experiment on GPU5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v13 "python experiment_v13.py 2>&1 | tee results/v13/experiment.log"

echo "Experiment started in tmux session 'v13'"
echo "To attach: tmux attach -t v13"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v13/experiment.log"