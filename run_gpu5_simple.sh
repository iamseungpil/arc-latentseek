#!/bin/bash
export CUDA_VISIBLE_DEVICES=5
# When CUDA_VISIBLE_DEVICES=5, the device will be cuda:0 in the script
python simple_experiment.py 0 --problems 3 2>&1 | tee simple_experiment_gpu5.log