#!/bin/bash
# The whole LITE training queue, unattended: the detector (pipeline-lite.sh:
# base -> synthetic fine-tune -> consolidation), then the GNN on its
# consolidated run (pipeline-gnn.sh: phase 1 -> phase 2 -> joint -> stage D),
# every stage evaluated on the validation set. Both pipelines resume after a
# reboot: rerun the same command.
#
# Usage:
#   ./pipeline-full.sh [DETECTOR_OUT] [GNN_OUT]
#   tmux new-session -d -s full './pipeline-full.sh checkpoints_lite_896s checkpoints_gnn_lite_896s'
#   tail -f pipeline-full.log
#
# Env: BASE_CONFIG (default conf-lite/train_640x896.yaml); DET_EXTRA_ARGS /
# GNN_EXTRA_ARGS (each pipeline's EXTRA_ARGS); QUEUE_LOG (default
# pipeline-full.log); and what the pipelines read (VAL, TRAIN_IMGS, PRUNE, ...).
# Smoke-test the whole queue before pushing a change (10 steps per stage on
# 30 + 6 scores, ~6 minutes; a crash fails at once):
#   uv run python scripts/smoke_sets.py /tmp/smoke
#   MAX_STALLED_RESTARTS=0 VAL=/tmp/smoke/val TRAIN_IMGS=/tmp/smoke/train/imgs QUEUE_LOG=/tmp/smoke/queue.log \
#   DET_EXTRA_ARGS="--img-dir /tmp/smoke/train/imgs --ann-dir /tmp/smoke/train/annotations --max-steps 10 --save-steps 10" \
#   GNN_EXTRA_ARGS="--img-dir /tmp/smoke/train/imgs --ann-dir /tmp/smoke/train/annotations --graph-dir /tmp/smoke/train/graphs --max-steps 10 --save-steps 10 --eval-steps 10" \
#   ./pipeline-full.sh /tmp/smoke/det /tmp/smoke/gnn
# A 10-step detector finds next to nothing, so the detection-path kern is
# empty there; stage D on real detections needs a trained detector's cache.

set -u
cd "$(dirname "$0")"

DET_OUT=${1:-checkpoints_lite_896s}
GNN_OUT=${2:-checkpoints_gnn_lite_896s}
BASE_CONFIG=${BASE_CONFIG:-conf-lite/train_640x896.yaml}
QUEUE_LOG=${QUEUE_LOG:-pipeline-full.log}

exec > >(tee -a "$QUEUE_LOG") 2>&1
echo "[$(date +'%F %T')] queue: detector -> $DET_OUT, GNN -> $GNN_OUT"

EXTRA_ARGS=${DET_EXTRA_ARGS:-} ./pipeline-lite.sh "$BASE_CONFIG" "$DET_OUT" || exit 1
EXTRA_ARGS=${GNN_EXTRA_ARGS:-} ./pipeline-gnn.sh "$(cat "$DET_OUT/.done_consolidation")" "$GNN_OUT" || exit 1
echo "[$(date +'%F %T')] queue done"
