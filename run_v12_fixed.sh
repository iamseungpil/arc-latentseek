#!/bin/bash
# Run V12 Fixed experiment on GPU5

# Create results directory
mkdir -p results/v12_fixed

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run experiment
echo "Starting V12 Fixed experiment on GPU5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v12_fixed "python experiment_v12_fixed.py 2>&1 | tee results/v12_fixed/experiment.log"

echo "Experiment started in tmux session 'v12_fixed'"
echo "To attach: tmux attach -t v12_fixed"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v12_fixed/experiment.log"