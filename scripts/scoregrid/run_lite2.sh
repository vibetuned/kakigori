#!/bin/bash
# The scoregrid decoder (masked discrete flow matching) on the 896s models
# against the GNN + serializer path, on the lite2 sets:
#   detector checkpoints_lite_896s/run_003 (consolidation), GNN phase 2
#   checkpoints_gnn_lite_896s/run_002 (the best detection-driven kern);
#   decoder data from their detections on train-lite2 + synthetic-lite2
#   (images the detector never trained on, but ~1.4k train-lite scores),
#   scored on validation-lite2 (balanced, real scores only) and validation-lite.
# Every step skips what is done; GPU jobs one after another.
#
# Usage: scripts/scoregrid/run_lite2.sh
set -o pipefail
cd "$(dirname "$0")/../.."

DET=checkpoints_lite_896s/run_003
GNN=checkpoints_gnn_lite_896s/run_002
OUT=data/scoregrid2
RUN=checkpoints_scoregrid/lite2-phase2
SPARK=checkpoints_gnn_lite_896s_spark
CONF=(--config conf-lite/config.json --roles_file conf-lite/structure.json)
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }
quiet() { grep -v "Warn\|warn\|self.encoder"; }
mkdir -p "$OUT" "$SPARK"

# 1. the GNN + serializer path on validation-lite2 (validation-lite: scored on the Spark)
krn=$SPARK/krn-det-phase2-val2
if [ ! -d "$krn" ]; then
    log "GNN path on validation-lite2 -> $krn"
    uv run validate-detections --img_dir data/validation-lite2/imgs --mei_dir data/validation-lite2/mei \
        --out_dir "$krn" --detector-checkpoint "$DET" --gnn-checkpoint "$GNN" "${CONF[@]}" > "$krn.log" 2>&1 \
        || { log "GNN path FAILED (see $krn.log)"; exit 1; }
fi

# 2. decoder data from the same models' detections
for set in validation-lite validation-lite2 train-lite2 synthetic-lite2; do
    if [ ! -f "$OUT/$set/index.json" ]; then
        log "decoder data: $set"
        uv run build-scoregrid-data --data "data/$set" --out "$OUT/$set" --checkpoint "$GNN" \
            --detector-checkpoint "$DET" > "$OUT/$set.log" 2>&1 || { log "$set FAILED (see $OUT/$set.log)"; exit 1; }
    fi
done

# 3. the decoder (defaults: candidate pitches with dropout 0.5)
if ! grep -q "done: best" "$RUN.log" 2>/dev/null; then
    log "training $RUN"
    uv run train-scoregrid --train "$OUT/train-lite2" "$OUT/synthetic-lite2" --val "$OUT/validation-lite2" \
        --out "$RUN" --epochs 30 > "$RUN.log" 2>&1 || { log "training FAILED (see $RUN.log)"; exit 1; }
fi

# 4. decode both validation sets, then the comparisons
for set in validation-lite validation-lite2; do
    [ -d "$RUN/krn-$set" ] || uv run eval-scoregrid --checkpoint "$RUN/best.pt" --data "$OUT/$set" \
        --mei-dir "data/$set/mei" --out "$RUN/krn-$set" --steps 1 2>&1 | quiet | tail -1
done
log "== validation-lite2 (real scores)"
args=(--data data/validation-lite2 --index "$OUT/validation-lite2/index.json" --real-only)
uv run python scripts/scoregrid/kern_split.py "${args[@]}" "gnn=$krn" "decoder=$RUN/krn-validation-lite2" 2>&1 | quiet
uv run python scripts/scoregrid/paired_f1.py "${args[@]}" "gnn=$krn" "decoder=$RUN/krn-validation-lite2" 2>&1 | quiet
log "== validation-lite"
args=(--data data/validation-lite --index "$OUT/validation-lite/index.json")
uv run python scripts/scoregrid/kern_split.py "${args[@]}" "gnn=$SPARK/krn-det-phase2" \
    "decoder=$RUN/krn-validation-lite" 2>&1 | quiet
uv run python scripts/scoregrid/paired_f1.py "${args[@]}" "gnn=$SPARK/krn-det-phase2" \
    "decoder=$RUN/krn-validation-lite" 2>&1 | quiet
uv run python scripts/scoregrid/staff_diag.py --data "$OUT/validation-lite2" --staves 2 "$RUN/best.pt" 2>&1 | quiet
log "== done"
