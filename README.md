# kakigori

An end-to-end **Optical Music Recognition (OMR)** pipeline that turns scanned or rendered music score pages into [Humdrum `**kern`](https://www.humdrum.org/) representations.

The system is split into three cooperating stages:

1. **Vision** — a CenterNet-style multi-scale detector finds every musical primitive (noteheads, stems, beams, accidentals, clefs, dynamics, lyrics, barlines, …) on a page as a class + bounding box.
2. **Graph** — a graph neural network reads the detector's output and predicts the typed relationships between primitives (which stem belongs to which notehead, which notes share a beam, which lyric attaches to which note, which measures synchronize across staves).
3. **Serialization** — the predicted score graph is converted to `**kern` text.

The training corpus comes from [musicset](../musicset): it selects and mutates real MusicXML files, generates synthetic scores, engraves them with Verovio (fixed or varied fonts, staff sizes and line weights), and extracts ground-truth bounding boxes, graph edges and `**kern`.

## Quick start — pretrained weights & inference

Download the released weights from Hugging Face
([detector](https://huggingface.co/vibetuned/kakigori-vision) ·
[full pipeline](https://huggingface.co/vibetuned/kakigori-omr); model cards
live in [model_cards/](model_cards/)):

```bash
./scripts/download-weights.sh
# -> checkpoints/release/vision/model.safetensors  (detector, 109 classes)
# -> checkpoints/release/omr/model.safetensors     (GNN wrapper)
```

Run the full pipeline on a PDF score (or a directory of page `.png`s with
`--images pages/` instead of `--pdf`):

```bash
uv run infer-omr --pdf score.pdf \
                 --detector-checkpoint checkpoints/release/vision \
                 --gnn-checkpoint checkpoints/release/omr \
                 --output-dir out/
# out/score.krn                        — the Humdrum **kern transcription
# out/page_NNNN.png + page_NNNN.json   — per-page detections
# --split writes one kern per page (page_NNNN.krn) instead of score.krn
```

Detections only (vision stage by itself):

```bash
uv run infer-model --checkpoint checkpoints/release/vision \
                   --pdf score.pdf --output-dir out/
```

Both scripts read the class list from `conf/config.json`; `infer-omr` also
reads `conf/structure.json` for the serializer's node roles. Expect the
best results on digitally rendered pages — real scans are outside the
training distribution for now. Honest v0.1.0 expectation for the fully
detection-driven path: the best scores transcribe at ~90% pitch accuracy,
the median is ~33% — the edge model has not yet been trained on detection
noise (see the roadmap in the TODO section); with ground-truth boxes the
same graph+serializer stack reaches 93% / 80%.

## Repository layout

```
src/kakigori/
├── vision/    # Multi-scale primitive detector (training, eval, inference, viewer)
├── graph/     # Phase-2 GNN over detector output + Humdrum serializer
└── matching/  # Exploratory flow-matching image restoration (currently unused)
```

Datasets (rendering, annotations, graphs, **kern groundtruth, scan-like
augmentation, synthetic scores) come from [musicset](../musicset), a dependency
of this package: see [docs/modules/dataset.md](docs/modules/dataset.md).

Per-module documentation lives in [docs/modules/](docs/modules/):
- [vision.md](docs/modules/vision.md)
- [graph.md](docs/modules/graph.md)
- [dataset.md](docs/modules/dataset.md) (datasets with musicset)
- [matching.md](docs/modules/matching.md)

For the full corpus → detector → GNN → `**kern` recipe, see the
[end-to-end training guide](docs/end-to-end-training.md). The graph model's
design, phase curriculum, and training lessons are explained in
[docs/graph.md](docs/graph.md).

## Pipeline

```
       .mxl files (real, or musicset synth)
                │
   ┌────────────┴────────────┐
   │ musicset select         │  lite gates, voice caps, holdout exclusion
   │ musicset synth          │  optional: exercises, piano, counterpoint,
   │                         │   mutations of real scores
   └────────────┬────────────┘
                │
   ┌────────────┴────────────┐
   │ musicset build          │  stable ids → Verovio (fixed or varied
   │   --labels conf-lite    │   engraving) → PNG pages, bbox JSONs,
   │   --graph --kern        │   MEI, ground-truth graphs, **kern
   └────────────┬────────────┘
                │
       ┌────────┴────────┐
       │                 │
   ┌───┴────┐       ┌────┴─────┐
   │ vision │       │  graph   │
   │ train  │ ─────▶│  train   │  GNN trains on top of frozen detector
   │ eval   │       │          │
   │ infer  │       │  infer   │  detector → RoI → GNN → **kern
   └────────┘       └──────────┘
```

`musicset view` browses a dataset (pages with boxes, graph edges, the score
graph, augmentation next to real scans). `visualize-model` (under `vision/`) is
an interactive detector inspector with bbox / heatmap / feature-map views.

## Configuration

The pipeline reads several JSON configs from `conf/`:

| File | Purpose |
| --- | --- |
| `config.json` | `target_classes` list — the class vocabulary used everywhere |
| `hierarchy.json` | Class groupings for grouped visibility toggles and TensorBoard layout |
| `smufl_mapping.json` | Maps SMuFL glyph hrefs to semantic class names |
| `structure.json` | `node_roles` — splits classes into temporal anchors / modifiers / sync / context |
| `gelato_config.json` | GNN-specific target-class list (consumed by `graph/train.py`) |
| `train_gnn.yaml` | Default flags for the GNN trainer |

### Adding a new glyph class

Four config files need the new class (e.g. adding SMuFL U+E52D as `dynamicMF`):

1. `smufl_mapping.json` — codepoint → class name (`"E52D": "dynamicMF"`); use the
   [SMuFL canonical name](https://w3c.github.io/smufl/latest/tables/index.html).
2. `config.json` — **append to the END of `target_classes`**. Never insert
   mid-list: `retrain-model` widens the detector's classification head
   positionally (old weights are copied into the first N slots), so inserting
   shifts every later class index and silently breaks existing checkpoints.
3. `hierarchy.json` — add to the appropriate visibility/TensorBoard group.
4. `structure.json` — add to the right `node_roles` bucket.

Then rebuild the datasets (`./regenerate-dataset.sh data/<set>`): musicset
fingerprints the label set, so a changed conf re-projects the bbox JSONs and
graphs from the stored canonical pages without re-rendering, and graphs are
always rebuilt against the new JSONs (annotation IDs are de-collided
positionally, so a graph only matches the JSONs it was built from). Finally
rerun `validate-groundtruth` + `compare-kern` and check the pitch/rhythm
totals are unchanged. Note that the vision model
must be fine-tuned (`retrain-model --old-num-classes N`) before the *detector*
can emit the new class; the ground-truth pipeline works immediately.

## CLI entry points

All scripts are installed as `pyproject.toml` `[project.scripts]`, so once the package is installed they are on `$PATH`.

### Datasets

Built with musicset (installed with this package; `uv run musicset --help`):

| Command | What it does |
| --- | --- |
| `musicset select` | Filter and sample sources: `--lite` gates, voice caps, holdout exclusion |
| `musicset build` | Engrave and annotate: PNG pages, bbox JSONs (`--labels conf-lite`), MEI, graphs (`--graph`), **kern (`--kern`); `--profile varied` for render variety |
| `musicset synth` | Synthetic MusicXML: exercises, piano pieces, counterpoint, mutations of real scores |
| `musicset augment` | Scan-like copies of a built dataset (boxes moved with the ink) |
| `./regenerate-dataset.sh` | `musicset build` with this repo's layout and confs (see the script) |

### Inspection

| Command | What it shows |
| --- | --- |
| `musicset view` | Pages + boxes, graph edges, score graph, augmentation vs real scans |
| `visualize-model` | Detector inspector: boxes, heatmaps, feature maps, comparison with ground truth (`--ann-dir`) |

### Training

| Command | What it does |
| --- | --- |
| `calculate-areas` | Print median normalized bbox area per class to pick scale thresholds |
| `train-model` | Train the vision detector from scratch or warm-start |
| `retrain-model` | Class-expansion fine-tune: widen the cls head, two-stage freeze schedule |
| `eval-model` | Compute per-class mAP and log to TensorBoard |
| `infer-model` | Run the detector across a PDF, emit PNG + JSON per page |
| `infer-omr` | Full pipeline on a PDF or page images: detect → predict edges → repair → `**kern` |
| `train-gnn` | Train the Phase-2 GNN on top of a frozen detector |
| `train-joint` | Joint GNN + detection training: one checkpoint for detection and edges |
| `validate-groundtruth` | Run the serializer directly on GT graphs to debug it independently of the GNN |
| `validate-predictions` | End-to-end validation on GT boxes with predicted edges + repairs |
| `validate-detections` | End-to-end validation on detected boxes (the honest `infer-omr` path) |
| `compare-kern` | Score generated `**kern` against groundtruth MEI: per-measure pitch and rhythm match rates, total and median file |
| `./pipeline-lite.sh` | Lite detector, unattended: base → synthetic fine-tune → consolidation, eval after each stage; resumable (run it in tmux, `tail -f OUT/pipeline.log`) |
| `./pipeline-gnn.sh` | Lite GNN on a trained detector, unattended: phase 1 → phase 2 → joint, kern evaluations (GT boxes and detections) after each stage; resumable |

Detector and GNN-wrapper checkpoints carry their layout (backbone, detection
levels, input canvas, scale ranges): `eval-model`, `infer-*`, `visualize-model`
and `validate-*` take either a detector checkpoint or a GNN / joint run (whose
own detector is then used).

## Serializer status & TODO

The `**kern` serializer is validated by running it on ground-truth graphs and
comparing the result against the source MEI:

```
validate-groundtruth --graph_dir data/validation-test/graphs \
                     --json_dir data/validation-test/annotations \
                     --out_dir data/validation-test/krn-me \
                     --roles_file conf/structure.json
compare-kern --mei_dir data/validation-test/mei --krn_dir data/validation-test/krn-me
```

`compare-kern` matches per-staff/per-measure multisets of sounding pitches
(letter, octave, alteration) and of pitch+duration, so it catches accidental,
key-signature, octave, dots, and duration regressions. Current score on
`data/validation-test`: **99.9% pitch / 99.9% rhythm** over ~6000 notes.
Mid-piece clef, key signature, and meter changes are detected from each
system's printed restatements (plus in-measure `gClefChange`/`fClefChange`
glyphs) and emitted as `*clefX` / `*k[...]` / `*M...` interpretation rows.
Dynamics are serialized into a per-staff `**dynam` spine (192/194 markings
match the MEI on the validation set; the two misses are `cresc.` text
directives, which have no glyph class). Pedal spans become `*ped` / `*Xped`
interpretation rows at the press/release positions (span counts match the
MEI exactly on the validation set).

Known gaps, roughly by impact:

- [ ] Tuplet ratios are inferred from the member count, assuming uniform
      durations — mixed-duration tuplets misscale. The `tupletNum` digit glyph
      is not read. (Causes the 2 remaining verovio rhythm warnings.)
- [ ] A pedal span is attached to the measure where it starts, so a span
      crossing a barline gets its `*Xped` at the end of that measure instead
      of in the following one (no such span exists in the validation set yet).
- [ ] `compare-kern`'s MEI side still applies the *initial* key signature per
      staff; a groundtruth piece that truly modulates would confuse the
      checker even though the serializer now follows the change.
- [ ] A few isolated high-ledger-line notes get the wrong octave from the
      geometric pitch estimate (~1% of notes on dense scans).
- [ ] Ties whose far end is missing from the annotations leave unbalanced
      `[` / `]` markers (2 unmatched opens in `QmbiBpt…`).
- [ ] Optimized-layout scores (systems hiding resting staves) scramble the
      index-based staff→spine mapping — needs staff-identity tracking
      across systems (planned serializer iteration).
- [ ] Tablature (`6stringTabClef`) and percussion clefs are out of scope
      for the kern serializer.
- [x] Retrain the vision detector for the appended octave clefs — done
      (`run_004` class expansion → `run_006` octave consolidations, valsmall
      mAP@.50 0.724). `clefG8vb` AP 0.986; `clefG8va` plateaus at 0.317:
      localization is perfect (IoU ≈ 0.9) but the ~3 px "8" above the clef is
      below the 640 px input's resolving power, so classification votes split
      toward `clefG`/`clefG8vb` — same family as the thin-shape item below.
- [x] Retrain the GNN with the 109-class embedding — done (GNN `run_004` via
      `conf/train_gnn_phase5.yaml`: `old_num_classes` + `new_class_templates`
      seed the new embedding rows from `clefG`; f1_macro 0.856, the best of
      all phases).
- [x] Second graph-repair iteration against the per-file swings — done:
      edge-level diagnosis traced them to missed `measure→staff` links and
      guard-blocked stem/flag repairs; repair 10 + selective structural
      guards take end-to-end from 89.7% / 76.6% to 91.6% / 78.3%.
      Temporal-chain bridging was implemented and measured: zero effect
      (repair 12, default-off); chord→note containment measured net-negative
      (repair 11, default-off).
- [x] Grace-note misdetection on small-notehead render styles — fixed: the
      staff-space-only threshold flagged one whole file as grace notes
      (rhythm 0%); the test now also requires the head to be small relative
      to the page's median notehead. Current totals: end-to-end
      **91.8% / 78.8%**, ceiling 95.5% / 94.6%.
- [x] Investigated the "whole-region loss" files — none actually had missing
      systems (structure counts matched GT everywhere). Real causes, fixed
      or closed: `QmbkhcJML…` was an early-music score whose C clefs sit on
      different lines per part (soprano C1, mezzo C2…) while the vision
      class is just `clefC` — the serializer now measures the clef's staff
      line from the glyph position (`_resolve_clef_class`, `*clefC1`–`C5`),
      taking that file from 57%/51% to 97%/95% at the ceiling. `Qmbkw5eNmB…`
      is a broken source: 42 gestural-only accidentals (`accid.ges`), zero
      printed ones, none in the SVG — unrecoverable by any OMR, same class
      as the tablature file. Current totals: **end-to-end 92.7% / 80.1%**,
      ceiling 96.8% / 96.1%.
- [x] Staff-identity tracking for optimized layouts — done: the serializer
      now buffers all pages, sizes the spine set from the widest system,
      maps reduced systems' rows onto parts by monotone DP over printed
      evidence (clef incl. measured C-clef line, key-accidental count), and
      pads hidden parts with full-measure rests. Full measures still map by
      order (immune to row-matching noise); on the predicted path the row
      list is y-filtered by the system bbox and double-mapped parts are
      dropped, or bad edges desynchronize spines. Ceilings: validation-small
      96.8/96.1 → **98.4/97.7**, test-small 86.7/85.4 → **90.0/89.0** (zero
      per-file regressions); end-to-end validation-small **93.2/80.4**,
      test-small 82.3/75.0.
- [ ] Large orchestral scores where the full part set NEVER appears in one
      system (MEI numbers 25–31 parts, ≤16 visible concurrently) can't be
      identified geometrically — needs instrument-label OCR to name rows.
      Two such scores cap the test-small aggregate.
- [x] Detection-driven measurement + bbox repair (pre-release gate) — done:
      `validate-detections` measures the honest `infer-omr` path, and
      `graph/bbox_repair.py` recovers detector-missed structure (band-NMS
      FP filtering; systems from uncovered bands; measures from barline
      x-clusters; staff rows from fixed-line-clef geometry, which is exact;
      cells from measure×row intersections) plus graph repair 13 (orphan
      beam/tuplet/layer subtrees → containing staff). validation-small:
      5.1%/3.2% (104/201 scores) → **29.0%/23.3%** (201/201, median 33%,
      best files 90–93%); GT-box path unchanged.
- [ ] **v0.2.0 headline**: train the GNN on DETECTED boxes with
      groundtruth-matched edges (match GT nodes onto detections to derive
      labels). The detection-driven gap is now dominated by the edge model
      meeting detection noise it never saw — box-jitter (σ=1.5 px) does not
      simulate misses, false positives, or class confusion.
- [ ] End-to-end comparisons carry a per-file noise floor (~±0.5) from
      nondeterministic GNN inference (cuDNN); judge serializer heuristics by
      the deterministic ceiling path or aggregate deltas well above noise.
- [ ] Thin/small-glyph class group (`beamBroken`, `stem32`, `ledgerLines`,
      `meterSig`, digit time signatures, and now the `clefG8va` "8"): frequent
      or well-localized but low AP — needs `scale_ranges`/input-resolution
      investigation rather than more data.
- [ ] Double-dot detection is a bbox-width heuristic (width vs. staff space);
      unusual render scales could misclassify.
- [ ] Unpitched/percussion notation (MEI `loc`-based notes) is not supported.
- [ ] Fingering, tempo, and text directives (`cresc.`, hairpins…) are detected
      by the vision stage but not serialized; text directives also lack a
      glyph class, so they are invisible to the `**dynam` spine.

## Requirements

- Python ≥ 3.14
- PyTorch ≥ 2.7 (CUDA 13.0 wheels are pinned in `pyproject.toml`)
- Verovio ≥ 6.0
- A GPU for training/inference of any reasonable size

Heavy native dependencies (Verovio, CairoSVG, music21, OpenCV, PySide6) make the install non-trivial — `uv` is recommended.

## License

[AGPL-3.0-only](LICENSE).
