#!/bin/bash
# Run V20 GLM experiment on GPU 5

# Create results directory
mkdir -p results/glm_v20

# Set GPU 5
export CUDA_VISIBLE_DEVICES=5

# Run experiment
echo "Starting V20 GLM-based experiment on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s glm_v20_gpu5 "python experiment_glm_v20.py 2>&1 | tee results/glm_v20/experiment.log"

echo "Experiment started in tmux session 'glm_v20_gpu5'"
echo "To attach: tmux attach -t glm_v20_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/glm_v20/experiment.log"