#!/bin/bash
cd /home/ubuntu/arc-latentseek

# Setup environment
export PATH="/home/ubuntu/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=5

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results_comparative_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

echo "Starting GPU 5 experiment at $(date)"
echo "Results directory: $RESULTS_DIR"

# Run experiment with conda python directly
/home/ubuntu/miniconda3/envs/unsloth_env/bin/python comparative_experiment_v2.py \
    --results_dir $RESULTS_DIR \
    --gpu_id 5 \
    2>&1 | tee ${RESULTS_DIR}/gpu5_log.txt