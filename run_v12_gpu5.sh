#!/bin/bash
# Run V12 Majority Voting experiment on GPU5

# Create results directory
mkdir -p results/v12_majority

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run experiment
echo "Starting V12 Majority Voting experiment on GPU5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v12_experiment "python experiment_v12_majority.py 2>&1 | tee results/v12_majority/experiment.log"

echo "Experiment started in tmux session 'v12_experiment'"
echo "To attach: tmux attach -t v12_experiment"
echo "To check status: tmux ls"