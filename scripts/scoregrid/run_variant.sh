#!/bin/bash
# Train one scoregrid decoder variant on the balanced set (train-lite + the
# clean synthetic grand-staff scores) and score it on validation-lite: K=1
# compare-kern metrics, single vs grand staff, per-staff accuracy on the
# grand-staff systems, kern cleanliness. Outputs in checkpoints_scoregrid/NAME.
#
# Usage: scripts/scoregrid/run_variant.sh NAME [train-scoregrid flags...]
#    e.g. scripts/scoregrid/run_variant.sh no-cand --no-cand
set -o pipefail
cd "$(dirname "$0")/../.."
name=$1; shift
out=checkpoints_scoregrid/$name
quiet() { grep -v "Warn\|warn\|self.encoder"; }
uv run train-scoregrid --train data/scoregrid/train-lite data/scoregrid/synthetic-grand-lite \
  --val data/scoregrid/validation-lite --out "$out" --epochs 30 "$@" > "$out.log" 2>&1 \
  || { echo "$name training FAILED (see $out.log)"; exit 1; }
uv run eval-scoregrid --checkpoint "$out/best.pt" --data data/scoregrid/validation-lite \
  --mei-dir data/validation-lite/mei --out "$out/krn-val-k1" --steps 1 2>&1 | quiet | tail -1 \
  | sed "s/^.* - INFO - /$name K=1: /"
uv run python scripts/scoregrid/kern_split.py "$name=$out/krn-val-k1" 2>&1 | quiet | grep -v "^scores"
uv run python scripts/scoregrid/staff_diag.py --staves 1 "$out/best.pt" 2>&1 | quiet
uv run python scripts/scoregrid/staff_diag.py --staves 2 "$out/best.pt" 2>&1 | quiet
uv run python scripts/scoregrid/kern_clean.py "$name=$out/krn-val-k1" 2>&1 | quiet
