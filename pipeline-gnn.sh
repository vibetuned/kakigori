#!/bin/bash
# LITE GNN pipeline on a trained detector: phase 1 (detector frozen) ->
# phase 2 (PANet neck adapts) -> joint (full backbone, edge + detection loss,
# one checkpoint for detection and edges) -> stage D (the joint model on its
# own detections: its boxes cached on the training pages, matched to the
# ground truth, plus the keep head), each evaluated on validation-lite:
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
# Stage D is also scored with the keep head pruning (det-prune, PRUNE).
#
# The GNN trains on data/gnn-mix-lite: train-lite (5% grand-staff pages) plus
# the clean synthetic-lite grand-staff scores of conf-lite/gnn_mix_synthetic.txt
# (26% grand staff in all), built from symlinks on the first run
# (scripts/mix_sets.py). On ground-truth boxes the train-lite GNN scored F1
# 98.1 on single-staff scores and 78.5 on grand staves.
#
# Env: P1_CONFIG, P2_CONFIG, JOINT_CONFIG, D_CONFIG (default
# conf-lite/train_gnn_phase1, train_gnn_phase2, train_joint_full,
# train_stage_d .yaml), TRAIN_IMGS (pages to cache detections for, default
# data/gnn-mix-lite/imgs: the stage-D config's img_dir), MIX (the training
# mix to build if missing), PRUNE (keep
# threshold, default 0.5), LOGGING_STEPS (default 100),
# VAL (validation set for the kern evaluations, default data/validation-lite),
# STOP_AFTER (phase2: end after phase 2 and its evaluations — joint and stage
# D did not improve on it, detection-driven kern F1 90.3 / 89.1 / 88.6),
# EXTRA_ARGS (appended to every training command, e.g. a smoke test:
# EXTRA_ARGS="--max-steps 30 --eval-steps 30 --save-steps 30").

set -u
cd "$(dirname "$0")"

DET=${1:-checkpoints_lite_896/run_003}
OUT=${2:-checkpoints_gnn_lite_896}
P1_CONFIG=${P1_CONFIG:-conf-lite/train_gnn_phase1.yaml}
P2_CONFIG=${P2_CONFIG:-conf-lite/train_gnn_phase2.yaml}
JOINT_CONFIG=${JOINT_CONFIG:-conf-lite/train_joint_full.yaml}
D_CONFIG=${D_CONFIG:-conf-lite/train_stage_d.yaml}
TRAIN_IMGS=${TRAIN_IMGS:-data/gnn-mix-lite/imgs}
MIX=${MIX:-data/gnn-mix-lite}
PRUNE=${PRUNE:-0.5}
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

kern_eval() {  # kern_eval <stage> <gt|det> [detector checkpoint [tag [validate-detections args]]]
    local stage=$1 kind=$2 det=${3:-} tag=${4:-} run
    run=$(stage_run "$stage")
    local krn="$OUT/krn-$kind$tag-$stage"
    shift $(( $# < 4 ? $# : 4 ))
    if grep -q "^TOTAL" "$krn.compare.txt" 2>/dev/null; then
        say "kern $kind$tag $stage already scored"
    else
        say "kern $kind$tag $stage: $run -> $krn"
        if [ "$kind" = gt ]; then
            uv run validate-predictions --graph_dir $VAL/graphs --json_dir $VAL/annotations \
                --img_dir $VAL/imgs --out_dir "$krn" --gnn_checkpoint "$run" "${CONF[@]}" > "$krn.log" 2>&1
        else
            uv run validate-detections --img_dir $VAL/imgs --mei_dir $VAL/mei --out_dir "$krn" \
                --detector-checkpoint "$det" --gnn-checkpoint "$run" "${CONF[@]}" "$@" > "$krn.log" 2>&1
        fi
        uv run compare-kern --mei_dir $VAL/mei --krn_dir "$krn" > "$krn.compare.txt" 2>&1
    fi
    grep "^TOTAL\|^PITCH PRECISION\|^MEDIAN\|no kern output" "$krn.compare.txt" | sed "s/^/    $kind$tag $stage /"
}

cache_detections() {  # the joint detector's boxes on the training pages, for stage D
    if [ -f "$OUT/.done_cache" ]; then
        say "detections already cached: $OUT/detections-train"
        return 0
    fi
    say "===== caching detections of $(stage_run joint) on $TRAIN_IMGS ====="
    if ! uv run cache-detections --img-dir "$TRAIN_IMGS" --out-dir "$OUT/detections-train" \
            --detector-checkpoint "$(stage_run joint)" --config conf-lite/config.json \
            > "$OUT/cache.log" 2>&1; then
        say "caching detections FAILED, see $OUT/cache.log"
        exit 1
    fi
    touch "$OUT/.done_cache"
    say "detections cached: $(ls "$OUT/detections-train" | wc -l) pages"
}

if [ ! -d "$MIX/graphs" ]; then
    say "building the GNN training mix $MIX"
    uv run python scripts/mix_sets.py "$MIX" data/train-lite data/synthetic-lite conf-lite/gnn_mix_synthetic.txt \
        || { say "building $MIX FAILED"; exit 1; }
fi
say "GNN pipeline: detector $DET; $P1_CONFIG -> $P2_CONFIG -> $JOINT_CONFIG -> $OUT"
train phase1 train-gnn "$P1_CONFIG"
kern_eval phase1 gt
train phase2 train-gnn "$P2_CONFIG" --fine-tune "$(stage_run phase1)"
kern_eval phase2 gt
kern_eval phase2 det "$DET"
if [ "${STOP_AFTER:-}" = phase2 ]; then
    say "GNN pipeline done (STOP_AFTER=phase2): phase1 $(stage_run phase1), phase2 $(stage_run phase2)"
    exit 0
fi
train joint train-joint "$JOINT_CONFIG" --fine-tune "$(stage_run phase2)"
[ -f "$OUT/eval_joint.log" ] || eval_detector joint "$(stage_run joint)"
kern_eval joint gt
kern_eval joint det "$(stage_run joint)"
cache_detections
train staged train-joint "$D_CONFIG" --fine-tune "$(stage_run joint)" --det-dir "$OUT/detections-train"
[ -f "$OUT/eval_staged.log" ] || eval_detector staged "$(stage_run staged)"
kern_eval staged gt
kern_eval staged det "$(stage_run staged)"
kern_eval staged det "$(stage_run staged)" -prune --prune-threshold "$PRUNE"
say "GNN pipeline done: phase1 $(stage_run phase1), phase2 $(stage_run phase2), joint $(stage_run joint), stage D $(stage_run staged)"
