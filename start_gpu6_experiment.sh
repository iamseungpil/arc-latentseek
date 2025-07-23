#!/bin/bash
cd /home/ubuntu/arc-latentseek

# Setup environment
export PATH="/home/ubuntu/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=6

# Create results directory - use same timestamp as GPU 5
RESULTS_DIR="$1"
if [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

echo "Starting GPU 6 experiment at $(date)"
echo "Results directory: $RESULTS_DIR"

# Run experiment with conda python directly
/home/ubuntu/miniconda3/envs/unsloth_env/bin/python comparative_experiment_v2.py \
    --results_dir $RESULTS_DIR \
    --gpu_id 6 \
    2>&1 | tee ${RESULTS_DIR}/gpu6_log.txt