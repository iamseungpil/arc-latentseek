#!/bin/bash
# Run GLM experiment with description-based optimization on GPU 5

# Create results directory
mkdir -p results/glm

# Run experiment
echo "Starting GLM experiment with description optimization on GPU 5..."
echo "Timestamp: $(date)"

# Use tmux for persistent session
tmux new-session -d -s glm_fixed_gpu5 "python experiment_glm.py 2>&1 | tee results/glm/experiment.log"

echo "Experiment started in tmux session 'glm_fixed_gpu5'"
echo "To attach: tmux attach -t glm_fixed_gpu5"
echo "To check status: tmux ls"
echo ""
echo "Monitor with: tail -f results/glm/experiment.log"