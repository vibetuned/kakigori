#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo "Starting training pipeline..."
echo "============================"
echo ""

echo "Training model..."
echo "----------------"
echo ""

uv run train-model --train-config conf/train.yaml

echo "Fine-tuning model..."
echo "----------------"
echo ""

uv run train-model --train-config conf/fine_tune.yaml

echo "Consolidating model..."
echo "----------------"
echo ""

uv run train-model --train-config conf/consolidation.yaml    