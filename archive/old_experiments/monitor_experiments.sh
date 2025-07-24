#!/bin/bash
# Monitor script for experiments

RESULTS_DIR="$1"
if [ -z "$RESULTS_DIR" ]; then
    echo "Usage: $0 <results_dir>"
    exit 1
fi

echo "Monitoring experiments in $RESULTS_DIR"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    echo "=== Experiment Monitor - $(date) ==="
    echo ""
    
    # Check if tmux sessions are running
    echo "=== Tmux Sessions ==="
    tmux ls 2>/dev/null | grep -E "gpu[56]_experiment" || echo "No GPU experiment sessions found"
    echo ""
    
    # Check GPU usage
    echo "=== GPU Usage ==="
    nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv | grep -E "(index|[56],)"
    echo ""
    
    # Check for errors in logs
    echo "=== Recent Errors ==="
    for gpu in 5 6; do
        LOG_FILE="${RESULTS_DIR}/gpu${gpu}_log.txt"
        if [ -f "$LOG_FILE" ]; then
            echo "GPU $gpu errors:"
            grep -E "ERROR|Error|error|FAILED|Failed|failed" "$LOG_FILE" | tail -3 || echo "  No errors"
        fi
    done
    echo ""
    
    # Check optimization progress
    echo "=== Optimization Progress ==="
    for gpu in 5 6; do
        LOG_FILE="${RESULTS_DIR}/gpu${gpu}_log.txt"
        if [ -f "$LOG_FILE" ]; then
            echo "GPU $gpu:"
            grep -E "Step [0-9]+:|Running|optimize_description_based|Found description tokens" "$LOG_FILE" | tail -5
            echo ""
        fi
    done
    
    # Check for perfect accuracy
    echo "=== Perfect Solutions ==="
    for gpu in 5 6; do
        LOG_FILE="${RESULTS_DIR}/gpu${gpu}_log.txt"
        if [ -f "$LOG_FILE" ]; then
            PERFECT=$(grep -c "Perfect accuracy achieved" "$LOG_FILE" 2>/dev/null || echo "0")
            echo "GPU $gpu: $PERFECT perfect solutions"
        fi
    done
    
    sleep 30
done