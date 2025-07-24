#!/bin/bash
# Run LatentSeek baseline experiment (V10) on GPU5

# Create results directory
mkdir -p results/latentseek_baseline

# Activate conda environment
source /data/miniforge3/etc/profile.d/conda.sh
conda activate base

# Set GPU
export CUDA_VISIBLE_DEVICES=0

# Run experiment
echo "Starting LatentSeek baseline experiment on GPU5..."
echo "Timestamp: $(date)"

# Create tmux session for persistent execution
tmux new-session -d -s latentseek_baseline "
cd /home/ubuntu/arc-latentseek
python run_fixed_experiment.py \
    --optimizer simple_v10 \
    --num_problems 5 \
    --device cuda:0 \
    2>&1 | tee results/latentseek_baseline/experiment.log
"

echo "Experiment started in tmux session 'latentseek_baseline'"
echo "To attach: tmux attach -t latentseek_baseline"
echo "To check status: tmux ls"
echo ""
echo "Monitor results with: tail -f results/latentseek_baseline/experiment.log"