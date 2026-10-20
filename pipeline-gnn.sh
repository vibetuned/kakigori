#!/bin/bash
# LITE GNN pipeline on a trained detector: phase 1 (detector frozen) ->
# phase 2 (PANet neck adapts) -> joint (full backbone, edge + detection loss,
# one checkpoint for detection and edges), each evaluated on validation-lite:
#   gt   validate-predictions: ground-truth boxes, predicted edges -> kern
#   det  validate-detections: detected boxes (+ bbox repair) -> kern, the
#        honest end-to-end number (after phase 2: the standalone detector;
#        after joint: the joint model's own detector)
# scored by compare-kern (pitch / rhythm, total and median file). Rerunning
# after a reboot skips finished stages and evaluations and resumes an
# interrupted stage (scripts/pipeline-lib.sh).
#
# Usage:
#   ./pipeline-gnn.sh [DETECTOR_CHECKPOINT] [OUTPUT_DIR]
#   ./pipeline-gnn.sh checkpoints_lite_896/run_003 checkpoints_gnn_lite_896
#
# In tmux, following one log:
#   tmux new-session -d -s gnn896 './pipeline-gnn.sh checkpoints_lite_896/run_003 checkpoints_gnn_lite_896'
#   tail -f checkpoints_gnn_lite_896/pipeline.log
#
# Env: P1_CONFIG, P2_CONFIG, JOINT_CONFIG (default conf-lite/train_gnn_phase1,
# train_gnn_phase2, train_joint_full .yaml), LOGGING_STEPS (default 100),
# VAL (validation set for the kern evaluations, default data/validation-lite),
# EXTRA_ARGS (appended to every training command, e.g. a smoke test:
# EXTRA_ARGS="--max-steps 30 --eval-steps 30 --save-steps 30").

set -u
cd "$(dirname "$0")"

DET=${1:-checkpoints_lite_896/run_003}
OUT=${2:-checkpoints_gnn_lite_896}
P1_CONFIG=${P1_CONFIG:-conf-lite/train_gnn_phase1.yaml}
P2_CONFIG=${P2_CONFIG:-conf-lite/train_gnn_phase2.yaml}
JOINT_CONFIG=${JOINT_CONFIG:-conf-lite/train_joint_full.yaml}
LOGGING_STEPS=${LOGGING_STEPS:-100}
VAL=${VAL:-data/validation-lite}
EXTRA_ARGS=${EXTRA_ARGS:-}
CONF=(--config conf-lite/config.json --roles_file conf-lite/structure.json)

mkdir -p "$OUT"
exec > >(tee -a "$OUT/pipeline.log") 2>&1
source scripts/pipeline-lib.sh

train() {  # train <stage> <train-gnn|train-joint> <config> [extra args]
    local name=$1 cmd=$2 config=$3
    shift 3
    run_stage "$name" uv run "$cmd" --train-config "$config" --detector-checkpoint "$DET" \
        --output-dir "$OUT" --logging-dir "$OUT/tb" --logging-steps "$LOGGING_STEPS" "$@" $EXTRA_ARGS
}

kern_eval() {  # kern_eval <stage> <gt|det> [detector checkpoint]
    local stage=$1 kind=$2 run
    run=$(stage_run "$stage")
    local krn="$OUT/krn-$kind-$stage"
    if grep -q "^TOTAL" "$krn.compare.txt" 2>/dev/null; then
        say "kern $kind $stage already scored"
    else
        say "kern $kind $stage: $run -> $krn"
        if [ "$kind" = gt ]; then
            uv run validate-predictions --graph_dir $VAL/graphs --json_dir $VAL/annotations \
                --img_dir $VAL/imgs --out_dir "$krn" --gnn_checkpoint "$run" "${CONF[@]}" > "$krn.log" 2>&1
        else
            uv run validate-detections --img_dir $VAL/imgs --mei_dir $VAL/mei --out_dir "$krn" \
                --detector-checkpoint "$3" --gnn-checkpoint "$run" "${CONF[@]}" > "$krn.log" 2>&1
        fi
        uv run compare-kern --mei_dir $VAL/mei --krn_dir "$krn" > "$krn.compare.txt" 2>&1
    fi
    grep "^TOTAL\|^MEDIAN\|no kern output" "$krn.compare.txt" | sed "s/^/    $kind $stage /"
}

say "GNN pipeline: detector $DET; $P1_CONFIG -> $P2_CONFIG -> $JOINT_CONFIG -> $OUT"
train phase1 train-gnn "$P1_CONFIG"
kern_eval phase1 gt
train phase2 train-gnn "$P2_CONFIG" --fine-tune "$(stage_run phase1)"
kern_eval phase2 gt
kern_eval phase2 det "$DET"
train joint train-joint "$JOINT_CONFIG" --fine-tune "$(stage_run phase2)"
[ -f "$OUT/eval_joint.log" ] || eval_detector joint "$(stage_run joint)"
kern_eval joint gt
kern_eval joint det "$(stage_run joint)"
say "GNN pipeline done: phase1 $(stage_run phase1), phase2 $(stage_run phase2), joint $(stage_run joint)"
