#!/bin/bash
# The LITE training queue on the clean, balanced lite2 sets (scripts/build-lite2.sh):
# the detector from scratch (base -> synthetic fine-tune -> consolidation, the
# 896s recipes on train-lite2 / synthetic-lite2), then the GNN phases 1-2 on
# train-lite2 (STOP_AFTER=phase2: joint and stage D did not beat phase 2),
# every stage scored on validation-lite2. Resumes after a reboot: rerun it.
# The decoder (scoregrid) is trained afterwards on this detector and GNN's
# detections (scripts/scoregrid/run_lite2.sh, DET/GNN set to these runs).
#
# Usage:
#   ./pipeline-lite2.sh [DETECTOR_OUT] [GNN_OUT]
#   tmux new-session -d -s lite2 './pipeline-lite2.sh'
#   tail -f pipeline-lite2.log
#
# Smoke test on this machine first (10 steps per stage on 30 + 6 scores):
#   uv run python scripts/smoke_sets.py /tmp/smoke2 data/train-lite2 data/validation-lite2
#   MAX_STALLED_RESTARTS=0 VAL=/tmp/smoke2/val QUEUE_LOG=/tmp/smoke2/queue.log \
#   DET_EXTRA_ARGS="--img-dir /tmp/smoke2/train/imgs --ann-dir /tmp/smoke2/train/annotations --max-steps 10 --save-steps 10" \
#   GNN_EXTRA_ARGS="--img-dir /tmp/smoke2/train/imgs --ann-dir /tmp/smoke2/train/annotations --graph-dir /tmp/smoke2/train/graphs --max-steps 10 --save-steps 10 --eval-steps 10" \
#   ./pipeline-lite2.sh /tmp/smoke2/det /tmp/smoke2/gnn
set -u
cd "$(dirname "$0")"

export BASE_CONFIG=${BASE_CONFIG:-conf-lite/train_640x896_lite2.yaml}
export FT_CONFIG=${FT_CONFIG:-conf-lite/fine_tune_lite2.yaml}
export CONS_CONFIG=${CONS_CONFIG:-conf-lite/consolidation_lite2.yaml}
export P1_CONFIG=${P1_CONFIG:-conf-lite/train_gnn_phase1_lite2.yaml}
export P2_CONFIG=${P2_CONFIG:-conf-lite/train_gnn_phase2_lite2.yaml}
export VAL=${VAL:-data/validation-lite2}
export MIX=${MIX:-data/train-lite2}
export TRAIN_IMGS=${TRAIN_IMGS:-data/train-lite2/imgs}
export STOP_AFTER=${STOP_AFTER:-phase2}
export QUEUE_LOG=${QUEUE_LOG:-pipeline-lite2.log}

exec ./pipeline-full.sh "${1:-checkpoints_lite2}" "${2:-checkpoints_gnn_lite2}"
