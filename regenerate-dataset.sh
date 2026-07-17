#!/bin/bash
# Kakigori Dataset Regeneration Script
#
# Wipes everything in a dataset directory EXCEPT filter/ (the curated .mxl
# sources) and rebuilds the derived data with the current symbol/annotation
# configuration:
#
#   filter/*.mxl -> injected/       (inject-mids: stable ids, needed by graphs)
#                -> svgs/ + imgs/   (render-dataset, from injected/)
#                -> annotations/    (extract-annotations)
# and, with WITH_GRAPHS=1, additionally:
#                -> mei/ + krn/     (export-dataset, from injected/)
#                -> graphs/         (generate-graphs)
#
# Usage:
#   ./regenerate-dataset.sh [DATA_DIR]                # default: data/train-small
#   WITH_GRAPHS=1 ./regenerate-dataset.sh data/validation-test

set -euo pipefail

DATA_DIR="${1:-data/train-small}"
WITH_GRAPHS="${WITH_GRAPHS:-0}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

if [ ! -d "$DATA_DIR/filter" ] || ! ls "$DATA_DIR/filter"/*.mxl >/dev/null 2>&1; then
    log "ERROR: $DATA_DIR/filter does not exist or contains no .mxl files."
    exit 1
fi

N_MXL=$(ls "$DATA_DIR/filter"/*.mxl | wc -l)
log "Regenerating $DATA_DIR from $N_MXL filtered MXL files."
log "================================================================="

log "Cleaning derived data (keeping filter/ and raw/)..."
for entry in "$DATA_DIR"/*; do
    name=$(basename "$entry")
    if [ "$name" != "filter" ] && [ "$name" != "raw" ]; then
        log "  removing $entry"
        rm -rf "$entry"
    fi
done

log "Stage 0: injecting stable element ids..."
uv run inject-mids "$DATA_DIR/filter" --out_dir "$DATA_DIR/injected"

log "Stage 1: rendering MXL -> SVG + PNG..."
uv run render-dataset "$DATA_DIR/injected" \
    --svg_dir "$DATA_DIR/svgs" --img_dir "$DATA_DIR/imgs"

log "Stage 2: extracting bbox annotations from SVGs..."
# The extractor skips already-written JSONs, so retrying only reprocesses
# failures (transient errors right after the render stage have been observed)
N_IMGS=$(ls "$DATA_DIR/imgs" | wc -l)
for attempt in 1 2 3; do
    uv run extract-annotations \
        --svg_dir "$DATA_DIR/svgs" --img_dir "$DATA_DIR/imgs" \
        --out_dir "$DATA_DIR/annotations"
    N_ANN=$(ls "$DATA_DIR/annotations" 2>/dev/null | wc -l)
    if [ "$N_ANN" -ge "$N_IMGS" ]; then
        break
    fi
    log "  attempt $attempt: $N_ANN/$N_IMGS annotations written, retrying missing pages..."
done

if [ "$WITH_GRAPHS" = "1" ]; then
    log "Stage 3: exporting MEI groundtruth..."
    uv run export-dataset "$DATA_DIR/injected" \
        --mei_dir "$DATA_DIR/mei" --krn_dir "$DATA_DIR/krn"

    log "Stage 4: generating ground-truth graphs..."
    uv run generate-graphs "$DATA_DIR/mei" "$DATA_DIR/annotations" \
        --out_dir "$DATA_DIR/graphs" --roles_file conf/structure.json
else
    log "Stages 3-4 skipped (set WITH_GRAPHS=1 to also export MEI + graphs)."
fi

log "================================================================="
log "Done. Summary for $DATA_DIR:"
log "  svgs:        $(ls "$DATA_DIR/svgs" 2>/dev/null | wc -l) files"
log "  imgs:        $(ls "$DATA_DIR/imgs" 2>/dev/null | wc -l) files"
log "  annotations: $(ls "$DATA_DIR/annotations" 2>/dev/null | wc -l) files"
if [ "$WITH_GRAPHS" = "1" ]; then
    log "  mei:         $(ls "$DATA_DIR/mei" 2>/dev/null | wc -l) files"
    log "  graphs:      $(ls "$DATA_DIR/graphs" 2>/dev/null | wc -l) files"
fi
