# End-to-end training guide

How to go from a pile of MusicXML files to a trained detector + GNN and a
measurable `**kern` output. Each stage lists the command, what it consumes and
produces, and how to check it worked before moving on.

```
MXL corpus ─▶ curate ─▶ render/export ─▶ annotations ─▶ graphs
                                                          │
              detector (vision) ◀── imgs + annotations ───┤
                     │ frozen                             │
              GNN (graph) ◀── imgs + annotations + graphs ┘
                     │
              **kern serialization ─▶ compare-kern / eval
```

Conventions: all commands run from the repo root with `uv run`. A "dataset
directory" always means the parallel triplet `imgs/` (PNG), `annotations/`
(JSON), `graphs/` (PyG `.pt`), plus `mei/` and `krn/` groundtruth where noted.

## Prerequisites

- `uv sync` completed (PyTorch ≥ 2.7 CUDA wheels, Verovio ≥ 6.0, music21).
- A GPU. Vision training defaults to batch 4 @ 640px; the GNN runs full-page
  graphs at batch 2.
- A source corpus of `.mxl` files.

## Stage 0 — Curate the corpus

```
uv run filter-mxl        # keep only scores whose render contains target classes
uv run inject-mids       # stamp stable id= attributes into every MXL element
```

`inject-mids` is what makes ground-truth graphs possible: the ids survive
Verovio's MXL→MEI/SVG conversion, so a bounding box in the SVG can be traced
back to the MEI element it draws. Skipping it breaks `generate-graphs`.

Optional augmentation (choose per experiment):

| Command | What it does |
| --- | --- |
| `synthetic-arranger` | One random per-measure mutation per real score |
| `synthetic-generator` | Aggressive whole-score probabilistic mutations |
| `synthetic-writer` | Brand-new scale/chord exercises from scratch |

## Stage 1 — Render and export groundtruth

```
uv run render-dataset    # MXL → per-page SVG + PNG
uv run export-dataset    # MXL → MEI (+ Verovio's own .krn attempt)
```

The MEI is the real groundtruth used everywhere downstream. The exported
`.krn` from Verovio is unreliable (see `data/validation-test/krn/` for what
failure looks like) — use `compare-kern` against the MEI instead of diffing
kern text.

## Stage 2 — Annotations and graphs

For a dataset directory that keeps its curated sources in `filter/`, the
whole render → extract cycle (Stages 1–2) is wrapped in one script that
first wipes all derived data:

```
./regenerate-dataset.sh data/train-small                 # svgs + imgs + annotations
WITH_GRAPHS=1 ./regenerate-dataset.sh data/validation-x  # + mei + krn + graphs
```

Use it whenever the symbol/annotation configuration changes — it also
verifies the annotation count against the page count and retries the
extraction (transient mass failures right after the render stage have been
observed; failures are logged as warnings per file).

The underlying commands, for custom layouts:

```
uv run extract-annotations --svg_dir <d>/svgs --img_dir <d>/imgs --out_dir <d>/annotations
uv run generate-graphs <d>/mei <d>/annotations --out_dir <d>/graphs --roles_file conf/structure.json
```

Produces per-page bbox JSONs (vision targets) and per-score `.pt` graphs with
`edge_index`, `y` (6 edge classes: 0 none / 1 structural / 2 modifier /
3 temporal / 4 sync-text / 5 simultaneity), `node_ids`, `json_files`
(GNN targets).

**Gotchas (will silently bite):**

- Both commands **skip existing output files** and still count them as
  successes. Re-running after a config change requires deleting the old
  outputs first.
- Annotations and graphs must be regenerated **together**: node-ID collisions
  are resolved positionally at load time, so `.pt` files built against older
  JSONs stop matching.
- Adding a glyph class touches four config files and must append to the end
  of `target_classes` — see "Adding a new glyph class" in the README.

Sanity check: open a few pages in `visualize-dataset` (boxes) and
`visualize-graphs` (edges overlaid on the score).

## Stage 3 — Train the vision detector

The detector is trained in an **evaluate-driven loop**: a base run on real
data first, then per-class evaluation decides what synthetic data to
generate for the weak classes, and only then the synthetic fine-tune runs.

### 3a. Base training from scratch (real data only)

```
./train.sh          # supervisor: relaunches `train-model --resume` until clean exit
# equivalent foreground run: uv run train-model --train-config conf/train.yaml
```

Data dirs, class list (`conf/config.json`), epochs etc. come from
`conf/train.yaml`. Checkpoints land in `checkpoints/run_NNN`
(auto-incrementing); TensorBoard logs in `runs/`. After *any* change to the
class list you either train from scratch (this stage) or widen an existing
checkpoint with `uv run retrain-model --old-num-classes N` — mixing an old
checkpoint with a new class count without one of these will not load.

### 3b. Evaluate per-class mAP to find the weak classes

```
uv run eval-model --checkpoint checkpoints/run_NNN \
                  --img-dir <val>/imgs --ann-dir <val>/annotations
```

This prints per-class mAP/IoU (grouped via `conf/hierarchy.json`). Three
distinct causes of a weak class, with different remedies:

- **Rare in the corpus** (few training examples) → generate synthetic data
  targeting it (3c).
- **Badly annotated** (extraction bug, wrong bbox) → inspect Stage-2 output
  with `visualize-dataset` for that class first; more data won't fix wrong
  boxes.
- **Below the input's resolving power** → if the class differs from a common
  one only by a few pixels at `input_size` 640, data stops helping once
  localization is solid. Diagnose by matching predictions to GT boxes: when
  every GT box has a high-IoU prediction but the class votes go to the
  look-alike, you've hit this. Seen with `clefG8va` (a ~3 px "8" above an
  otherwise identical treble clef): three synthetic rounds took it
  0.019 → 0.117 → 0.317 and plateaued while its data exceeded `clefG8vb`'s
  (which sits at 0.986). The fix is input resolution / `scale_ranges` work,
  not a fourth round.

### 3c. Generate targeted synthetic data

Use the synthetic generators to oversample the weak classes —
`synthetic-writer` (from-scratch exercises), `synthetic-generator`
(aggressive mutations), `synthetic-arranger` (mild per-measure mutations) —
into a separate dataset directory, then run
`./regenerate-dataset.sh data/synthetic-<name>` to render and annotate it.

### 3d. Synthetic fine-tune and consolidation

```
uv run train-model --train-config conf/fine_tune.yaml      # + synthetic 1:3, higher lr/gamma
uv run train-model --train-config conf/consolidation.yaml  # mostly real again, low lr
```

Each stage warm-starts from the previous run's checkpoint via the
`fine_tune:` key in its YAML — **update those paths** (`checkpoints/run_NNN/`)
to the run the previous stage actually produced, and point
`synthetic_img_dir`/`synthetic_ann_dir` at the 3c output. Re-run 3b after
each stage; iterate 3c–3d until the weak classes converge.

**Troubleshooting:** a `CUBLAS_STATUS_NOT_INITIALIZED` / CUDA init error at
the first training step (seen on newer GPUs) is an environment problem, not
a model problem — upgrade the pinned wheels with
`uv lock --upgrade-package torch --upgrade-package torchvision && uv sync`.

### 3e. Class expansion (adding glyphs after a trained run)

When classes are appended to `conf/config.json` (see the README's
"how to add a glyph"), widen the newest consolidated checkpoint instead of
retraining from scratch:

```
uv run retrain-model --train-config conf/update_model.yaml   # old_num_classes: N
```

`retrain-model` copies the old head weights positionally (hence *append*
classes, never reorder), gives the new slots the focal bias prior, and runs a
two-stage schedule: frozen backbone at `lr` to fit the new heads, then full
unfreeze at `lr_unfrozen` (stage-2 checkpoints land in `run_NNN/stage2/`;
both stages crash-resume independently). Before launching, make sure **every
annotation source actually contains the new labels** — real *and* synthetic:
a synthetic set extracted before the new SMuFL mapping contributes zero
signal even if its SVGs contain the glyph, and re-extracting is cheap
(`rm -rf <set>/annotations`, then `extract-annotations` — no re-render
needed). Then re-enter the 3b–3d loop for the new classes. Reference run:
107 → 109 (`clefG8vb`/`clefG8va`, 2026-07-17): expansion `run_004` valsmall
mAP@.50 0.704 (vs 0.687 before — no regression), octave-boost consolidations
`run_005`/`run_006` → 0.724, `clefG8vb` 0.986, `clefG8va` capped at 0.317 by
input resolution (see 3b).

## Stage 4 — Train the GNN (phased)

The GNN trains in four phases that release one constraint at a time — the
full rationale, monitoring guide, and hard-won lessons live in
[graph.md](graph.md). Operationally each phase is one run of:

```
./train.sh uv run train-gnn --train-config conf/train_gnn.yaml --resume
```

with the per-phase config (pre-create the next `checkpoints_gnn/run_NNN`
so `--resume` targets it rather than the previous run):

| Phase | config | Notes |
| --- | --- | --- |
| 1 | `conf/train_gnn_phase1.yaml` | GNN + RoI head from scratch, frozen detector |
| 2 | `conf/train_gnn_phase2.yaml` | PANet adapts as a vision→graph adaptation layer |
| 3 | `conf/train_gnn_phase3.yaml` | in reserve — only if phase 2 plateaus early |
| 4 | `conf/train_gnn_phase4.yaml` | GT-box noise, narrows the gap to detector boxes |
| 5 | `conf/train_gnn_phase5.yaml` | class expansion: after a detector class change (3e) |

Update each phase file's `fine_tune:` to point at the previous phase's
actual run directory before launching.

Phase 5 exists because a class change ripples differently through the GNN
than through the detector: glyph classes are GNN *inputs* (a
`class_embedding` row each), not outputs, so there is no head to widen and
no frozen stage — `old_num_classes:` in the yaml makes `train-gnn` load only
the old checkpoint's `gnn.*` weights, widen the embedding, and seed each new
row from a look-alike class via `new_class_templates:` (octave clefs copy
`clefG` — same structural role). The old wrapper's adapted neck is
deliberately dropped: the detector underneath moved, and re-adapting the
neck to it (with `unfreeze: neck` + the phase-4 jitter) is the actual work
of this phase. Regenerate annotations *and* graphs (Stage 2) before it so
the new classes exist as nodes.

Common settings: `detector_checkpoint:` → the Stage-3 consolidation run;
`img_dir`/`ann_dir`/`graph_dir` → the Stage-2 triplet; `config:` stays
`conf/config.json`. `fine_tune:` accepts an HF checkpoint dir and
warm-starts the whole wrapper. **After any unfrozen phase, re-run
`eval-model` on the detector**: the same backbone serves detection at
inference, and joint fine-tuning can silently degrade box quality.

Candidate edges come from `heuristics.generate_axis_aware_edges` (vertical,
staff-lane horizontal, kNN safety net); the GNN classifies candidates, so
recall is capped by the heuristics — if an edge type never appears as a
candidate, no amount of training recovers it.

Monitoring (eval runs every `eval_steps`; TensorBoard in `runs_gnn/`):
watch per-edge-class F1 (all six classes) *and* the topology deltas.
Topology deltas frozen at identical values across evals mean the model has
collapsed into an attractor (all-background, or the inverse rare-class
flood) — see graph.md §4 for the diagnosis recipe and §5 for the fixes.
Phase-1 reference: macro-F1 0.825 after 30 epochs (~2.2 h on an RTX 5090
at ~20 it/s).

## Stage 5 — Validate the serializer (no models involved)

Before judging the models, pin down the serializer's ceiling on perfect
input by running it directly on ground-truth graphs:

```
uv run validate-groundtruth --graph_dir <d>/graphs --json_dir <d>/annotations \
                            --out_dir <d>/krn-me --roles_file conf/structure.json
uv run compare-kern --mei_dir <d>/mei --krn_dir <d>/krn-me
```

Current ceiling on `data/validation-test`: **99.9% pitch / 99.9% rhythm**
(see "Serializer status & TODO" in the README for what the last 0.1% is).
Any end-to-end score is bounded above by this number — regressions here are
serializer bugs, not model problems.

## Stage 6 — End-to-end test on *predicted* graphs

This is the step that answers "does the graph model work":

```
uv run validate-predictions --graph_dir <d>/graphs --json_dir <d>/annotations \
    --img_dir <d>/imgs --out_dir <d>/krn-pred \
    --gnn_checkpoint checkpoints_gnn/run_NNN
uv run compare-kern --mei_dir <d>/mei --krn_dir <d>/krn-pred
```

`validate-predictions` mirrors `validate-groundtruth` but replaces the
ground-truth edge labels with GNN predictions: per page it letterboxes the
image, runs the (wrapper-checkpoint) detector features + heuristic
candidates + GNN, maps predicted edges back to annotation node ids, injects
the one scaffold the GNN structurally cannot predict (system→measure —
system nodes are excluded from the per-system groups), and hands the result
to `MinimalHumdrumSerializer`. Compare against the ceiling (`compare-kern`
on `validate-groundtruth` output for the *same* dataset) to separate
serializer limitations from model errors.

Predicted edges pass through `graph/graph_repair.py` before serialization:
guarded spatial heuristics that recover the *geometric certainties* the GNN
loses (note→notehead→stem chains, meterSig digits, keySig accidentals,
staff context, orphaned events) without ever overriding a model prediction.
Ownership means structural/modifier parenthood — temporal/sync neighbors
never satisfy a guard — and class checks are exact (`"note"` must not
prefix-match `"noteheadBlack"`; that single bug cost 42 pitch points).
Repairs are individually toggleable (`enabled`/`loose`) for ablations;
`--no-repair` gives the raw-GNN baseline.

Reference numbers on `data/validation-small` (GT boxes; ceiling with GT
edges 95.5% pitch / 94.3% rhythm): the 107-class stack reached 89.6% /
76.6% (median file 97.9% pitch), up from 39.5% / 23.2% without repairs.
The 109-class stack (detector `run_006` + phase-5 GNN at f1_macro 0.856)
initially landed at 89.7% / 76.6% with individual files swinging up to
±26 pitch points; edge-level diagnosis of those swings (compare predicted
edges against the graph `.pt` labels, per relation family) attributed them
to missed `measure→staff` links and stems/flags blocked from repair by the
any-edge guard, and the second repair iteration (repair 10 + selective
structural guards) brought the stack to 91.6% / 78.3%. Two measured
negatives worth remembering: chord→note recovery by containment pulls in
other-voice notes (net-negative, default-off as repair 11), and temporal
chain bridging changes nothing (the serializer's cx fallback already
orders identically — default-off as repair 12). Serializer hardening on top of the repairs, in order: grace detection
relative to the page's median notehead height (one small-notehead render
style serialized a whole file as grace notes); C-clef staff lines measured
from the glyph's position (`*clefC1`–`*clefC5`; one 6-part early-music
score went 57%/51% → 97%/95% at the ceiling); and staff-identity tracking
for optimized layouts (pages buffered, spines sized from the widest
system, reduced systems' rows mapped to parts by monotone DP over printed
clef/key evidence, hidden parts padded with full-measure rests — full
measures still map by order, and the predicted path y-filters rows and
rejects double-mapped parts so bad edges can't desynchronize spines).
Current stack: **93.2% / 80.4%** against a ceiling of **98.4% / 97.7%**
(validation-small); held-out test-small: 82.3% / 75.0% against a
90.0% / 89.0% ceiling. Investigating the worst files showed no
actually-missing systems — the residue is broken sources (gestural-only
accidentals, tablature) and orchestral scores whose full part set never
appears in one system (identification needs label OCR).
Caveat for future measurements: GNN inference is nondeterministic (cuDNN),
giving a ~±0.5 per-file noise floor end-to-end — validate heuristics on
the deterministic ceiling path or on aggregate deltas well above noise.

**Held-out generalization** (`data/test-small`, 224 unseen scores / 484
pages / 82.6k notes, curated from `raw/` with `filter-mxl`, 2026-07-17):
detector mAP@.50 0.722 / .50:.95 0.637 (validation-small: 0.724/0.644) —
no overfitting. Ceiling 90.0% / 89.0% with **median file 100% / 100%**
(200 of 223 files ≥99% pitch); end-to-end 82.3% / 75.0%. The model-path
cost over the ceiling matches validation-small, confirming the GNN +
repairs transfer. The aggregate residue is two huge orchestral scores
whose full part set never appears in a single system — geometric identity
cannot name their rows (label OCR territory); together they hold ~12% of
the corpus notes.

**Stage 6b — the detection-driven path** (`validate-detections`, the
honest `infer-omr` measurement: detector boxes only). Baseline was 5.1% /
3.2% with only 104/201 scores serializing — structural detection at
inference thresholds misses 64% of measures, 44% of systems, 38% of staff
rows. `graph/bbox_repair.py` recovers the skeleton before the graph is
built: band-NMS FP filtering (systems/rows never y-overlap; one 0.30-score
"blanket" system turned a solo piece into a phantom duet), systems from
uncovered structure bands, measures from barline x-clusters (barlines
detect at ~0.86), staff rows from fixed-line-clef geometry (a clef bbox
determines its row EXACTLY — constants have zero variance), and cells from
measure×row intersections. Together with graph repair 13 (orphan
beam/tuplet/layer subtrees → containing staff; on detections, container
chains break constantly): **29.0% / 23.3%**, 201/201 scores, median 33%,
best files 90–93%. The GT-box path is unaffected by any of it. The
remaining gap is the edge model meeting detection noise it never saw —
v0.2.0: train the GNN on detected boxes with GT-matched edge labels.

> **Roadmap:** `MinimalHumdrumSerializer` stays the minimal variant;
> `graph_repair` (edges) and `bbox_repair` (detections) carry the
> hardening iterations. Missing-system recovery on the GT-box path was
> investigated and not needed, but IS needed and implemented on the
> detection path (bbox repairs 1/4/5). Temporal bridging: measured, no
> effect (repair 12, default-off). Next: GNN trained on detections
> (v0.2.0 headline), instrument-label OCR for never-full orchestral
> scores. `validate-predictions`/`validate-detections` keep every stage
> swappable for A/B.

Also useful: `graph/eval.py` (SER/CER/LER on kern text) for
sequence-level comparison.

## Quick reference — what feeds what

| Artifact | Produced by | Consumed by |
| --- | --- | --- |
| `imgs/*.png` | `render-dataset` | vision + GNN training, inference |
| `annotations/*.json` | `extract-annotations` | vision targets, GNN nodes, serializer bboxes |
| `mei/*.mei` | `export-dataset` | `generate-graphs`, `compare-kern` |
| `graphs/*.pt` | `generate-graphs` | GNN targets, `validate-groundtruth` |
| `checkpoints/run_NNN` | `train-model` | `train-gnn` (frozen), `infer-model` |
| `checkpoints_gnn/run_NNN` | `train-gnn` | Stage-6 inference |
| `krn-me/*.krn` | `validate-groundtruth` | `compare-kern`, `eval.py` |
