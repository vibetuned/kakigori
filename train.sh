#!/bin/bash

# Gelato Training Supervisor Script
# 
# This script runs the training process and automatically relaunches it if it crashes
# (e.g., due to a hardware error like a segmentation fault).
# The training script already automatically resumes from the latest checkpoint.
#
# Usage:
#   ./supervise_train.sh
# 
# Or you can pass your own custom command:
#   ./supervise_train.sh uv run gelato-train --custom_args...

if [ $# -eq 0 ]; then
    # Default command
    CMD=(
        uv run train-model --resume
    )
else
    # Use user-provided command
    CMD=("$@")
fi

echo "================================================================="
echo "Starting Supervised Training Loop"
echo "Command: ${CMD[*]}"
echo "================================================================="

while true; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting training process..."
    
    # Execute the command
    "${CMD[@]}"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training completed successfully (Exit Code: 0). Exiting supervisor."
        break
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Process crashed with Exit Code: $EXIT_CODE."
        echo "Restarting in 60 seconds to resume from the latest checkpoint..."
        sleep 60
    fi
done
