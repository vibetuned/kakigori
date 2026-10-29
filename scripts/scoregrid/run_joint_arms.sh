#!/bin/bash
# Does the scoregrid decoder's loss improve the detector? Three arms from the
# same checkpoints (GNN wrapper phase 2 + its detector, decoder lite2-phase2),
# steps and learning rates (scoregrid/joint.py), then for the start and each
# arm: detection mAP on validation-lite2, and the transcription with the arm's
# own detector detecting (build-scoregrid-data --detector-checkpoint ARM) and
# its decoder decoding, real scores only.
#   A  detection only (--flow-weight 0): what fine-tuning on lite2 labels does
#   B  joint (detection + decoder loss, flow weight 5 ~ the detection loss)
#   C  decoder only (--train-detector none)
# Every step skips what is done; one GPU job at a time.
#
# Usage: scripts/scoregrid/run_joint_arms.sh
set -o pipefail
cd "$(dirname "$0")/../.."

START=checkpoints_gnn_lite_896s/run_002
DECODER=checkpoints_scoregrid/lite2-phase2/best.pt
ROOT=checkpoints_scoregrid/joint
TRAIN=(data/scoregrid2/train-lite2=data/train-lite2 data/scoregrid2/synthetic-lite2=data/synthetic-lite2)
VAL=data/validation-lite2
COMMON=(--checkpoint "$START" --decoder "$DECODER" --train "${TRAIN[@]}"
        --val data/scoregrid2/validation-lite2="$VAL" --steps 3000 --flow-weight 5)
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
quiet() { grep -v "Warn\|warn\|self.encoder"; }
mkdir -p "$ROOT"

declare -A ARGS=([A]="--flow-weight 0" [B]="" [C]="--train-detector none")
for arm in A B C; do
    out=$ROOT/$arm
    if [ ! -f "$out/decoder.pt" ]; then
        log "training arm $arm: ${ARGS[$arm]:-joint}"
        # shellcheck disable=SC2086
        uv run train-scoregrid-joint "${COMMON[@]}" ${ARGS[$arm]} --out "$out" > "$out.log" 2>&1 \
            || { log "arm $arm FAILED (see $out.log)"; exit 1; }
    fi
done

evaluate() {  # evaluate NAME WRAPPER DECODER
    local name=$1 wrapper=$2 decoder=$3 dir=$ROOT/eval-$1
    mkdir -p "$dir"
    [ -f "$dir/map.log" ] && grep -q "mAP @" "$dir/map.log" || uv run eval-model --checkpoint "$wrapper" \
        --img-dir "$VAL/imgs" --ann-dir "$VAL/annotations" --config conf-lite/config.json \
        --hierarchy conf-lite/hierarchy.json > "$dir/map.log" 2>&1
    [ -f "$dir/scoregrid/index.json" ] || uv run build-scoregrid-data --data "$VAL" --out "$dir/scoregrid" \
        --checkpoint "$wrapper" --detector-checkpoint "$wrapper" > "$dir/scoregrid.log" 2>&1
    [ -d "$dir/krn" ] || uv run eval-scoregrid --checkpoint "$decoder" --data "$dir/scoregrid" \
        --mei-dir "$VAL/mei" --out "$dir/krn" --steps 1 > "$dir/krn.log" 2>&1
    echo "$name: $(grep -h "mAP @ IoU=0.50 \|mAP @ IoU=0.50:0.95" "$dir/map.log" | sed 's/^.* - INFO - //' | paste -sd' ')"
}
log "== detection mAP on validation-lite2"
evaluate start "$START" "$DECODER"
for arm in A B C; do
    evaluate "$arm" "$ROOT/$arm" "$ROOT/$arm/decoder.pt"
done
log "== transcription on validation-lite2 (real scores), each model's own detections"
runs=()
for name in start A B C; do runs+=("$name=$ROOT/eval-$name/krn"); done
args=(--data "$VAL" --index "$ROOT/eval-start/scoregrid/index.json" --real-only)
uv run python scripts/scoregrid/kern_split.py "${args[@]}" "${runs[@]}" 2>&1 | quiet
uv run python scripts/scoregrid/paired_f1.py "${args[@]}" "${runs[@]}" 2>&1 | quiet
log "== done"
