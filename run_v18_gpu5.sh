#!/bin/bash
# Run V18 experiment on GPU 5

# Create results directory
mkdir -p results/v18

# Set GPU 5
export CUDA_VISIBLE_DEVICES=5

# Run experiment
echo "Starting V18 Multi-Reward experiment on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s v18_gpu5 "python experiment_v18_multi_reward.py 2>&1 | tee results/v18/experiment.log"

echo "Experiment started in tmux session 'v18_gpu5'"
echo "To attach: tmux attach -t v18_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/v18/experiment.log"