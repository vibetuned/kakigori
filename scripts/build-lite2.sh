#!/bin/bash
# The clean, balanced LITE datasets (lite2), one stage at a time.
#
#   data/train-lite2       PDMX real scores: one part, single or grand staff,
#                          deduplicated by content, no holdout score (validation-
#                          lite, PDMX split val/test) nor a copy of one; every
#                          grand-staff candidate (PDMX has ~6.1k), the scores
#                          carrying rare classes (conf-lite/rare_train_lite2.txt)
#                          and as many plain single-staff ones as give the same
#                          number of pages
#   data/validation-lite2  balanced the same way (conf-lite/validation_lite2.txt and
#                          rare_validation_lite2.txt), gated like the training sets
#                          except on kern quality (it is scored on the MEI), plus
#                          synthetic scores for the classes no real score has
#   data/synthetic-lite2   musicset synthesis (exercises, piano, melodies,
#                          mutations and arrangements of train scores listed in
#                          conf-lite/synthetic_sources_lite2.txt)
#
# Every training score's kern is read back against its own MEI and the score
# dropped below 99 pitch recall and precision / 98 rhythm (--min-kern-*):
# groundtruth the page cannot show is not trained on. Every stage resumes
# where it stopped (builds skip finished scores, synthesis existing files);
# prune drops built scores whose source left their set.
#
# Verovio leaks and a full-core render has taken a machine down: stages run
# one after another, WORKERS processes each (default 8 of 24), niced.
#
# Usage: scripts/build-lite2.sh [stage...]
#   stages: select prune train validation validation-synth synth synth-build summary
#   (default: all, in that order)
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS="${WORKERS:-8}"
# plain single-staff candidates: 6700 balance the grand staff's pages (single
# staff kept 94% at 1.07 pages, grand 81% at ~1.4); the rare scores replace
# as many of them
SINGLE="${SINGLE:-5330}"
GRAND="${GRAND:-12000}"
PDMX="${PDMX:-/home/flux/projects/music_data/mxl}"
CONF=conf-lite
# 3 pages, not lite's old 2: the 2-page cap dropped 28% of the scarce grand staves
GATES=(--max-pages 3 --max-measures-per-system 15 --max-notes-per-measure 40)
QUALITY=(--min-kern-pitch 99 --min-kern-rhythm 98)
VIEWS=(--labels "$CONF" --graph --kern --paginated --abc --profile varied)
MUSICSET=(nice -n 10 uv run musicset)

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

stage_select() {
    if [ ! -f data/sources/holdout.txt ]; then
        { ls data/validation-lite/filter | sed 's/\..*$//'
          ls /home/flux/projects/music_data/dataset_full/processed_validation \
             /home/flux/projects/music_data/dataset_full/processed_test; } | sort -u > data/sources/holdout.txt
    fi
    # every lite score of PDMX, deduplicated by content, holdouts and their
    # copies out, shuffled (seed 7): the candidates (capped) and the whole pool
    [ -f data/sources/lite2-candidates/selection.jsonl ] || "${MUSICSET[@]}" select "$PDMX" \
        data/sources/lite2-candidates --lite --dedup --exclude-list data/sources/holdout.txt \
        --per-staves 12000 12000 --seed 7 --workers "$WORKERS"
    [ -f data/sources/lite2-pool/selection.jsonl ] || "${MUSICSET[@]}" select "$PDMX" \
        data/sources/lite2-pool --lite --dedup --exclude-list data/sources/holdout.txt \
        --seed 7 --workers "$WORKERS"
    # the rare-class lists are kept in conf-lite; they were made from the pool
    # minus the first train and validation sets with:
    #   scripts/rare_scores.py data/sources/lite2-pool/selection.jsonl data/sources/rare-pool.jsonl \
    #       --skip data/train-lite2/filter data/validation-lite2/filter
    #   scripts/choose_rare.py data/sources/rare-pool.jsonl \
    #       conf-lite/rare_validation_lite2.txt conf-lite/rare_train_lite2.txt
    uv run python - "$SINGLE" "$GRAND" <<'EOF'
import json, os, sys
from pathlib import Path

def stems(path):
    return {s for s in Path(path).read_text().splitlines() if s and not s.startswith("#")}

caps, taken = {1: int(sys.argv[1]), 2: int(sys.argv[2])}, {}
paths = {}
for listing in ("data/sources/lite2-pool/selection.jsonl", "data/sources/lite2-candidates/selection.jsonl"):
    for line in open(listing):
        p = Path(json.loads(line)["path"])
        paths.setdefault(p.stem, p)
for p in Path("data/validation-lite/filter").iterdir():
    paths[p.name.split(".")[0]] = p
validation = stems("conf-lite/validation_lite2.txt") | stems("conf-lite/rare_validation_lite2.txt")
train = set()
for line in open("data/sources/lite2-candidates/selection.jsonl"):
    s = json.loads(line)
    k, stem = s["max_staves"], Path(s["path"]).stem
    if stem in validation or taken.get(k, 0) >= caps.get(k, 0):
        continue
    taken[k] = taken.get(k, 0) + 1
    train.add(stem)
train |= stems("conf-lite/rare_train_lite2.txt")
assert not train & validation
sources = Path("conf-lite/synthetic_sources_lite2.txt")
if not sources.exists():
    sources.write_text("# train-lite2 scores that synthetic-lite2's mutate/arrange draw from\n"
                       + "".join(f"{s}\n" for s in sorted(train)))
synth = stems(sources)

def sync(directory, wanted):
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    have = {p.name.split(".")[0]: p for p in d.iterdir() if p.is_symlink()}
    for stem, link in have.items():
        if stem not in wanted:
            link.unlink()
    for stem in wanted - have.keys():
        src = paths[stem]
        os.symlink(src.resolve(), d / src.name)
    print(f"{directory}: {len(wanted)} scores")

sync("data/train-lite2/filter", train)
# (links only: validation-synth's generated files there stay)
sync("data/validation-lite2/filter", validation)
sync("data/sources/synth-sources-lite2", synth)
print(f"train: {taken} plain candidates by staff count")
EOF
}

stage_prune() {
    # built scores whose source left the set (a changed split)
    uv run python - <<'EOF'
from pathlib import Path
from musicset.pipeline import Layout, _remove_score
for root in ("data/train-lite2", "data/validation-lite2", "data/synthetic-lite2"):
    layout = Layout(Path(root))
    if not layout.manifest.exists():
        continue
    sources = {p.name.removesuffix(".mxl").removesuffix(".musicxml") for p in (layout.root / "filter").iterdir()}
    gone = [m.stem for m in layout.manifest.glob("*.json") if m.stem not in sources]
    for score in gone:
        _remove_score(layout, score)
    print(f"{root}: {len(gone)} scores pruned")
EOF
}

stage_train() {
    "${MUSICSET[@]}" build data/train-lite2/filter data/train-lite2 "${VIEWS[@]}" --seed 0 \
        "${GATES[@]}" "${QUALITY[@]}" --workers "$WORKERS"
}

stage_validation() {
    # kern quality recorded, not gated: evaluation scores against the MEI
    "${MUSICSET[@]}" build data/validation-lite2/filter data/validation-lite2 "${VIEWS[@]}" --seed 1 \
        "${GATES[@]}" --workers "$WORKERS"
}

stage_validation_synth() {
    # synthetic validation scores (files, not links, in validation-lite2/filter;
    # indices from 900000) for the classes no clean real score carries
    local out=data/validation-lite2/filter synth=("${MUSICSET[@]}" synth)
    "${synth[@]}" exercise "$out" -n 40 --staves 1 --profile rare --seed 301 --start 900000 \
        --workers "$WORKERS"
    "${synth[@]}" exercise "$out" -n 20 --staves 2 --max-measures 12 --profile rare --seed 302 \
        --start 910000 --workers "$WORKERS"
    stage_validation
}

stage_synth() {
    local out=data/synthetic-lite2/filter synth=("${MUSICSET[@]}" synth)
    local sources=data/sources/synth-sources-lite2
    mkdir -p "$out"
    # single staff
    "${synth[@]}" exercise "$out" -n 3900 --staves 1 --profile rare --seed 201 --workers "$WORKERS"
    "${synth[@]}" exercise "$out" -n 1200 --staves 1 --meters rare --max-measures 8 --profile rare \
        --seed 202 --start 10000 --workers "$WORKERS"
    "${synth[@]}" melody "$out" -n 800 --seed 203 --workers "$WORKERS"
    "${synth[@]}" easy "$out" -n 400 --seed 204 --workers "$WORKERS"
    # grand staff
    "${synth[@]}" exercise "$out" -n 2500 --staves 2 --max-measures 12 --profile rare --seed 205 \
        --start 20000 --workers "$WORKERS"
    "${synth[@]}" piano "$out" -n 4000 --profile rare --seed 206 --workers "$WORKERS"
    # variations of train scores (single and grand, as listed)
    "${synth[@]}" mutate "$out" --sources "$sources" -n 1500 --profile rare --seed 207 \
        --workers "$WORKERS"
    "${synth[@]}" arrange "$out" --sources "$sources" -n 1500 --profile rare --seed 208 \
        --workers "$WORKERS"
}

stage_synth_build() {
    "${MUSICSET[@]}" build data/synthetic-lite2/filter data/synthetic-lite2 "${VIEWS[@]}" --seed 2 \
        "${GATES[@]}" "${QUALITY[@]}" --workers "$WORKERS"
}

stage_summary() {
    uv run python scripts/dataset_summary.py data/train-lite2 data/validation-lite2 data/synthetic-lite2
    uv run python scripts/class_coverage.py data/train-lite2 data/synthetic-lite2 data/validation-lite2 \
        --below 20
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(select prune train validation validation-synth synth synth-build summary)
for s in "${stages[@]}"; do
    log "== $s"
    "stage_${s//-/_}"
done
log "== done"
