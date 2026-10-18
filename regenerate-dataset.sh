#!/bin/bash
# Build a dataset directory from its curated sources with musicset.
#
#   filter/*.mxl -> imgs/ annotations/ graphs/ mei/ kern/  (what training reads)
#                   pages/ manifest/ injected/ svgs/       (musicset's own records)
#
# Builds are resumable and never stale: a score is skipped only when its
# manifest matches the current settings, and a changed conf re-projects
# annotations/ and graphs/ from pages/ without rendering. Nothing is deleted,
# so evaluation outputs kept in the directory (krn-det/, ...) survive.
#
# Usage:
#   ./regenerate-dataset.sh [DATA_DIR]            # default: data/train-lite
#   SEED=1 ./regenerate-dataset.sh data/validation-lite
#   PROFILE=fixed CONF=conf ./regenerate-dataset.sh data/train-small
#
# Environment:
#   CONF      label set: a conf directory (default conf-lite)
#   PROFILE   varied (a font, staff size, line weights, spacing, page per
#             score; default) or fixed (Verovio's default look)
#   SEED      seed of the varied profile (default 0)
#   VARIANTS  engravings per score (default 1)
#   WORKERS   parallel scores (default: musicset's, half the spare cores)

set -euo pipefail

DATA_DIR="${1:-data/train-lite}"
CONF="${CONF:-conf-lite}"
PROFILE="${PROFILE:-varied}"
SEED="${SEED:-0}"
VARIANTS="${VARIANTS:-1}"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

if [ ! -d "$DATA_DIR/filter" ] || ! ls "$DATA_DIR/filter"/*.mxl >/dev/null 2>&1; then
    log "ERROR: $DATA_DIR/filter does not exist or contains no .mxl files."
    exit 1
fi
if [ -d "$DATA_DIR/graphs" ] && ls "$DATA_DIR/graphs"/*.pt >/dev/null 2>&1; then
    log "WARNING: $DATA_DIR/graphs holds .pt graphs from the old pipeline; the GNN"
    log "         dataset prefers JSON but delete them to avoid mixing the two."
fi

log "Building $DATA_DIR from $(ls "$DATA_DIR/filter"/*.mxl | wc -l) sources" \
    "(labels $CONF, profile $PROFILE, seed $SEED, variants $VARIANTS)"

uv run musicset build "$DATA_DIR/filter" "$DATA_DIR" \
    --labels "$CONF" --graph --kern \
    --profile "$PROFILE" --seed "$SEED" --variants "$VARIANTS" \
    ${WORKERS:+--workers "$WORKERS"}

log "Done. imgs: $(ls "$DATA_DIR/imgs" | wc -l), annotations: $(ls "$DATA_DIR/annotations" | wc -l)," \
    "graphs: $(ls "$DATA_DIR/graphs" | wc -l), mei: $(ls "$DATA_DIR/mei" | wc -l)" \
    "(rejections in $DATA_DIR/failures.jsonl)"
