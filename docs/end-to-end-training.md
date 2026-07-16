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

This prints per-class mAP/IoU (grouped via `conf/hierarchy.json`). Two
distinct causes of a weak class, with different remedies:

- **Rare in the corpus** (few training examples) → generate synthetic data
  targeting it (3c).
- **Badly annotated** (extraction bug, wrong bbox) → inspect Stage-2 output
  with `visualize-dataset` for that class first; more data won't fix wrong
  boxes.

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

## Stage 4 — Train the GNN

```
uv run train-gnn --train-config conf/train_gnn.yaml
```

Edit `conf/train_gnn.yaml` first:

- `detector_checkpoint:` → your Stage-3 consolidation run
  (e.g. `checkpoints/run_025`). The detector is **frozen**; the GNN trains a
  GATv2 edge classifier on RoI features pooled from the detector's PANet
  neck.
- `img_dir` / `ann_dir` / `graph_dir` → your Stage-2 dataset triplet.
- `config:` should stay `conf/config.json` (the `gelato_config.json` default
  in `graph/train.py` is stale — the file doesn't exist).

Candidate edges come from `heuristics.generate_axis_aware_edges` (vertical,
staff-lane horizontal, kNN safety net); the GNN classifies candidates, so
recall is capped by the heuristics — if an edge type never appears as a
candidate, no amount of training recovers it.

Monitor `runs_gnn/` in TensorBoard; checkpoints land in
`checkpoints_gnn/run_NNN`. Look at per-edge-class F1, not just loss —
class 0 (no-edge) dominates the candidate set.

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

This is the step that answers "does the graph model work". The intended flow:

1. Run the detector on validation pages (`uv run infer-model` emits per-page
   JSON + PNG), or reuse ground-truth annotations to isolate the GNN's
   contribution from detector noise.
2. Build candidate edges with the same heuristics used in training, run the
   GNN to get per-edge class predictions.
3. Feed `MinimalHumdrumSerializer(edge_index, edge_predictions, node_roles,
   node_ids)` exactly as `validate_groundtruth.py` does — it is agnostic to
   whether `edge_predictions` are ground truth or model output.
4. Score with `compare-kern` (pitch/rhythm vs MEI) and `graph/eval.py`
   (SER/CER/LER on kern text).

> **Status:** step 2's glue is currently stale. `graph/infer.py` predates the
> current serializer — its imports reference a removed `serialization`
> module and are partly commented out. The cleanest path is a small
> `validate-predictions` CLI cloned from `validate_groundtruth.py` that
> loads a GNN checkpoint, replaces `y` with predicted edge classes over the
> heuristic candidates, and reuses everything else unchanged. Until that
> exists, Stage 6 cannot run.

A useful intermediate metric while that glue is being built: the GNN
trainer's edge-classification report on a held-out dataset directory —
if per-class edge F1 is low, no serializer will save the output.

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
