# kakigori

An end-to-end **Optical Music Recognition (OMR)** pipeline that turns scanned or rendered music score pages into [Humdrum `**kern`](https://www.humdrum.org/) representations.

The system is split into three cooperating stages:

1. **Vision** — a CenterNet-style multi-scale detector finds every musical primitive (noteheads, stems, beams, accidentals, clefs, dynamics, lyrics, barlines, …) on a page as a class + bounding box.
2. **Graph** — a graph neural network reads the detector's output and predicts the typed relationships between primitives (which stem belongs to which notehead, which notes share a beam, which lyric attaches to which note, which measures synchronize across staves).
3. **Serialization** — the predicted score graph is converted to `**kern` text.

A large **dataset** subsystem produces the training corpus by filtering and mutating real MusicXML files, generating fully synthetic exercises, rendering them with Verovio, and extracting ground-truth bounding boxes and graph edges.

## Repository layout

```
src/kakigori/
├── dataset/    # MXL → SVG/PNG + annotation JSONs + ground-truth graphs
├── vision/    # Multi-scale primitive detector (training, eval, inference, viewer)
├── graph/     # Phase-2 GNN over detector output + Humdrum serializer
└── matching/  # Exploratory flow-matching image restoration (currently unused)
```

Per-module documentation lives in [docs/modules/](docs/modules/):
- [vision.md](docs/modules/vision.md)
- [graph.md](docs/modules/graph.md)
- [dataset.md](docs/modules/dataset.md)
- [matching.md](docs/modules/matching.md)

## Pipeline

```
       .mxl files (real or synthetic)
                │
   ┌────────────┴────────────┐
   │ filter-mxl              │  drop files lacking target SMuFL classes
   │ inject-mids             │  stamp every MXL element with a stable id=
   │ synthetic-{arranger,    │  optional augmentation:
   │   generator, writer}   │   per-measure / whole-score / from-scratch
   └────────────┬────────────┘
                │
   ┌────────────┴────────────┐
   │ render-dataset          │  Verovio → SVG → CairoSVG → PNG (per page)
   │ export-dataset          │  Verovio → MEI + Humdrum .krn
   └────────────┬────────────┘
                │
   ┌────────────┴────────────┐
   │ extract-classes         │  bootstrap target-class vocabulary
   │ extract-annotations    │  SVG → per-page bbox JSON (vision targets)
   │ generate-graphs        │  MEI + bbox JSON → PyG .pt (GNN ground truth)
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

`visualize-dataset`, `visualize-graphs`, and `visualize-graphs-pysigma` are PySide6 viewers for inspecting each stage's outputs. `visualize-model` (under `vision/`) is an interactive detector inspector with bbox / heatmap / feature-map views.

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

Then regenerate the derived data. Both extraction steps **silently skip
existing output files** (and still count them as successes), so delete the old
outputs first:

```
rm <annotations_dir>/*.json && extract-annotations --svg_dir … --img_dir … --out_dir <annotations_dir>
rm <graphs_dir>/*.pt      && generate-graphs <mei_dir> <annotations_dir> --out_dir <graphs_dir>
```

Regenerating the graphs is not optional: annotation IDs are de-collided
positionally at load time, so `.pt` graphs built against the old JSONs stop
matching the new ones. Finally rerun `validate-groundtruth` + `compare-kern`
and check the pitch/rhythm totals are unchanged. Note that the vision model
must be fine-tuned (`retrain-model --old-num-classes N`) before the *detector*
can emit the new class; the ground-truth pipeline works immediately.

## CLI entry points

All scripts are installed as `pyproject.toml` `[project.scripts]`, so once the package is installed they are on `$PATH`.

### Dataset construction

| Command | What it does |
| --- | --- |
| `filter-mxl` | Keep only MXL files whose Verovio render contains target classes |
| `inject-mids` | Add stable `id=` attributes to every MusicXML element |
| `synthetic-arranger` | One random per-measure mutation per score |
| `synthetic-generator` | Aggressive whole-score probabilistic mutations |
| `synthetic-writer` | Generate brand-new piano scale/chord exercises from scratch |
| `render-dataset` | MXL → per-page SVG + PNG via Verovio + CairoSVG |
| `export-dataset` | MXL → MEI + Humdrum `.krn` via Verovio |
| `extract-classes` | Scan SVGs and emit the union of class tokens |
| `extract-annotations` | Compute per-class bounding boxes from SVG → JSON |
| `generate-graphs` | Build PyG ground-truth graphs from MEI + bbox JSON |

### Inspection

| Command | What it shows |
| --- | --- |
| `visualize-dataset` | PNG + bbox annotations with class visibility toggles |
| `visualize-graphs` | Graph edges overlaid on the score image |
| `visualize-graphs-pysigma` | Force-directed Sigma.js view of graph topology |
| `visualize-model` | Detector inspector: boxes, heatmaps, feature maps |

### Training

| Command | What it does |
| --- | --- |
| `calculate-areas` | Print median normalized bbox area per class to pick scale thresholds |
| `train-model` | Train the vision detector from scratch or warm-start |
| `retrain-model` | Class-expansion fine-tune: widen the cls head, two-stage freeze schedule |
| `eval-model` | Compute per-class mAP and log to TensorBoard |
| `infer-model` | Run the detector across a PDF, emit PNG + JSON per page |
| `train-gnn` | Train the Phase-2 GNN on top of a frozen detector |
| `validate-groundtruth` | Run the serializer directly on GT graphs to debug it independently of the GNN |
| `compare-kern` | Score generated `**kern` against groundtruth MEI: per-measure pitch and rhythm match rates |

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
