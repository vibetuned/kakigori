#!/bin/bash
# LITE detector pipeline: base -> synthetic fine-tune -> consolidation, each
# stage evaluated on validation-lite. Every stage takes the previous stage's
# run as --fine-tune; rerunning after a reboot skips finished stages and
# resumes an interrupted one (scripts/pipeline-lib.sh).
#
# Usage:
#   ./pipeline-lite.sh [BASE_CONFIG] [OUTPUT_DIR]
#   ./pipeline-lite.sh conf-lite/train_640x896.yaml checkpoints_lite_896
#
# In tmux, following one log:
#   tmux new-session -d -s lite896 './pipeline-lite.sh conf-lite/train_640x896.yaml checkpoints_lite_896'
#   tail -f checkpoints_lite_896/pipeline.log
#
# Env: FT_CONFIG, CONS_CONFIG (default conf-lite/{fine_tune,consolidation}.yaml),
# NUM_WORKERS (default 4), LOGGING_STEPS (default 200), VAL (evaluation set,
# default data/validation-lite), EXTRA_ARGS (appended to every training
# command, e.g. a smoke test: EXTRA_ARGS="--max-steps 20 --save-steps 20").

set -u
cd "$(dirname "$0")"

BASE_CONFIG=${1:-conf-lite/train_640x896.yaml}
OUT=${2:-checkpoints_lite_896}
FT_CONFIG=${FT_CONFIG:-conf-lite/fine_tune.yaml}
CONS_CONFIG=${CONS_CONFIG:-conf-lite/consolidation.yaml}
NUM_WORKERS=${NUM_WORKERS:-4}
LOGGING_STEPS=${LOGGING_STEPS:-200}
EXTRA_ARGS=${EXTRA_ARGS:-}

mkdir -p "$OUT"
exec > >(tee -a "$OUT/pipeline.log") 2>&1
source scripts/pipeline-lib.sh

train() {  # train <stage> <config> [extra args]: train-model into $OUT, then evaluate
    local name=$1 config=$2
    shift 2
    run_stage "$name" uv run train-model --train-config "$config" \
        --output-dir "$OUT" --logging-dir "$OUT/tb" \
        --num-workers "$NUM_WORKERS" --logging-steps "$LOGGING_STEPS" "$@" $EXTRA_ARGS
    [ -f "$OUT/eval_$name.log" ] || eval_detector "$name" "$(stage_run "$name")"
}

say "pipeline: base $BASE_CONFIG, fine-tune $FT_CONFIG, consolidation $CONS_CONFIG -> $OUT"
train base "$BASE_CONFIG"
train finetune "$FT_CONFIG" --fine-tune "$(stage_run base)"
train consolidation "$CONS_CONFIG" --fine-tune "$(stage_run finetune)"
say "pipeline done: base $(stage_run base), fine-tune $(stage_run finetune), consolidation $(stage_run consolidation)"
