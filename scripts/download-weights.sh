#!/bin/bash
# Download the pretrained kakigori weights from Hugging Face into the
# layout the infer scripts expect:
#
#   checkpoints/release/vision/model.safetensors   (detector, 109 classes)
#   checkpoints/release/omr/model.safetensors      (GNN wrapper)
#
# Usage:
#   ./scripts/download-weights.sh
#
# Override the source repos with VISION_REPO / OMR_REPO env vars.

set -euo pipefail

VISION_REPO="${VISION_REPO:-vibetuned/kakigori-vision}"
OMR_REPO="${OMR_REPO:-vibetuned/kakigori-omr}"

mkdir -p checkpoints/release

echo "Downloading detector weights ($VISION_REPO)..."
uv run hf download "$VISION_REPO" model.safetensors \
    --local-dir checkpoints/release/vision

echo "Downloading GNN wrapper weights ($OMR_REPO)..."
uv run hf download "$OMR_REPO" model.safetensors \
    --local-dir checkpoints/release/omr

echo "Done:"
ls -lh checkpoints/release/vision/model.safetensors checkpoints/release/omr/model.safetensors
