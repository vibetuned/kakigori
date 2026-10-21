#!/bin/bash

# Kakigori Training Supervisor Script
# 
# This script runs the training process and automatically relaunches it if it crashes
# (e.g., due to a hardware error like a segmentation fault).
# The training script already automatically resumes from the latest checkpoint.
#
# Usage:
#   ./supervise_train.sh
# 
# Or you can pass your own custom command:
#   ./supervise_train.sh uv run kakigori-train --custom_args...

if [ $# -eq 0 ]; then
    # Default command
    CMD=(
        uv run train-model --resume
    )
else
    # Use user-provided command
    CMD=("$@")
fi

# Progress decides between a bug and a hardware fault. A run that crashes
# again before saving a new checkpoint made no progress: after
# MAX_STALLED_RESTARTS such restarts in a row it is a bug, and the supervisor
# gives up. A run that keeps saving checkpoints between crashes (memory faults
# every few hours) is resumed for as long as it moves. Progress is a new
# checkpoint-* under PROGRESS_DIR (pipeline-lib.sh sets it); without one, a
# crash after MIN_PROGRESS_SECONDS of running counts as progress.
MAX_STALLED_RESTARTS=${MAX_STALLED_RESTARTS:-3}
MIN_PROGRESS_SECONDS=${MIN_PROGRESS_SECONDS:-600}
RESTART_DELAY=${RESTART_DELAY:-60}
PROGRESS_DIR=${PROGRESS_DIR:-}
stalled=0
stamp=$(mktemp)
trap 'rm -f "$stamp"' EXIT

echo "================================================================="
echo "Starting Supervised Training Loop"
echo "Command: ${CMD[*]}"
echo "================================================================="

while true; do
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Starting training process..."
    
    # Execute the command
    touch "$stamp"
    started=$(date +%s)
    "${CMD[@]}"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Training completed successfully (Exit Code: 0). Exiting supervisor."
        break
    else
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] Process crashed with Exit Code: $EXIT_CODE."
        if [ -n "$PROGRESS_DIR" ]; then
            progressed=$(find "$PROGRESS_DIR" -name 'checkpoint-*' -newer "$stamp" -print -quit 2>/dev/null)
        elif [ $(( $(date +%s) - started )) -ge "$MIN_PROGRESS_SECONDS" ]; then
            progressed=yes
        else
            progressed=
        fi
        if [ -n "$progressed" ]; then
            stalled=0
        else
            stalled=$((stalled + 1))
            echo "No new checkpoint since the last start ($stalled in a row)."
            if [ "$stalled" -gt "$MAX_STALLED_RESTARTS" ]; then
                echo "Giving up: $MAX_STALLED_RESTARTS restarts in a row made no progress (a bug, not a hardware fault)."
                exit "$EXIT_CODE"
            fi
        fi
        # Without --resume our trainers create a FRESH run dir on relaunch and
        # silently redo the whole run — append it so the restart actually resumes.
        if [[ " ${CMD[*]} " != *" --resume"* ]]; then
            CMD+=(--resume)
            echo "Command lacked --resume; restarting as: ${CMD[*]}"
        fi
        echo "Restarting in $RESTART_DELAY seconds to resume from the latest checkpoint..."
        sleep "$RESTART_DELAY"
    fi
done
