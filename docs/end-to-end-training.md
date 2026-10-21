# End-to-end training guide

How to go from a pile of MusicXML files to a trained detector + GNN and a
measured `**kern` output. Each stage lists the commands, what they consume and
produce, and how to check the result before moving on. The design and the
reasons behind each choice are in [vision.md](vision.md) (detector) and
[graph.md](graph.md) (GNN, repairs, serializer).

```
MXL corpus ─▶ musicset select / synth ─▶ musicset build ─▶ imgs + annotations + graphs + mei + kern
                                                                  │
   pipeline-lite.sh:  detector  base ─▶ fine-tune ─▶ consolidation ◀┤
                                                    │               │
   pipeline-gnn.sh:   GNN  phase 1 ─▶ phase 2 ─▶ joint ─▶ stage D ◀─┘
                                                             │ one checkpoint: boxes + edges
                 validate-groundtruth / -predictions / -detections, infer-omr
                                                   │
                         **kern ─▶ compare-kern, musicset view --pred-kern
```

This guide follows the **LITE track** (`conf-lite/`, 99 classes, one part
with at most two staves, one voice per staff), the 0.2.0 proving ground.
The full track (`conf/`, 109 classes) uses the same tools with its own
configs; its 0.1.x recipe and numbers are kept at the end.

Conventions: all commands run from the repo root with `uv run`. A "dataset
directory" holds `imgs/` (PNG), `annotations/` (JSON), `graphs/` (JSON),
`mei/` (the real ground truth) and `kern/`, all built by
[musicset](modules/dataset.md) from the curated sources in `filter/`.

## The whole queue

`pipeline-full.sh` runs Stages 3 and 4 back to back, unattended: about two
and a half days on the DGX Spark. Every stage is evaluated, and the queue
resumes after a reboot when rerun.

```
tmux new-session -d -s full './pipeline-full.sh checkpoints_lite_896s checkpoints_gnn_lite_896s'
tail -f pipeline-full.log
```

Before a long run, smoke-test the whole queue at 10 steps per stage on small
symlinked sets (the usage block in `pipeline-full.sh`; ~10 minutes), on
this machine before pushing. Run it with `MAX_STALLED_RESTARTS=0` so that a
crash fails at once.

`train.sh` resumes a crashed stage for as long as it makes progress: it
gives up only after 3 restarts in a row that saved no new checkpoint
(`MAX_STALLED_RESTARTS`), i.e. a bug. Hardware faults every few hours are
resumed indefinitely.

## Prerequisites

- `uv sync` completed. musicset is a sibling checkout (`../musicset`) and
  brings Verovio ≥ 6.3 and music21. Run musicset's
  `scripts/install-fonts.sh` once per machine.
- A GPU. At 640×896 the detector trains at batch 4 in bf16. On an RTX 5090
  a step takes 51 ms; on the DGX Spark (GB10) it takes 297 ms (see
  "Training on the DGX Spark").
- A source corpus of `.mxl` files.

## Stage 0 — Curate the corpus

```
uv run musicset select ../music_data/mxl data/train-lite/filter --lite --limit 9500 \
    --exclude-from data/validation-lite/filter --copy
```

`--lite` keeps one part, at most two staves, one voice per staff, no
tablature or percussion, and no gestural-only accidentals. `--max-voices` /
`--per-voice` shape a full-track selection. `--exclude-from` keeps holdouts
out.

Synthetic scores are MusicXML too. Choose them per experiment:

| Command | What it does |
| --- | --- |
| `musicset synth exercise` | Scales and chord progressions; `--profile rare` / `detector` weights the weak classes |
| `musicset synth piano` | Melody/chords over accompaniment patterns on functional harmony |
| `musicset synth counterpoint` | Species counterpoint from a pool (`musicset counterpoint-pool`): piano, duet with C clefs, one staff, choir |
| `musicset synth mutate` / `arrange` | Real scores with rare notation injected / one mutation per measure |

The current sets:
- `data/train-lite`: 9,500 scores, 10,387 pages.
- `data/validation-lite`: 500 scores, 555 pages.
- `data/synthetic-lite`: 5,991 scores, 10,497 pages. Exercises, short
  rare-meter exercises and piano pieces on the `rare` profile, plus 1,500
  mutations and arrangements of train-lite.

## Stages 1-2 — Engrave, annotate, graphs

```
./regenerate-dataset.sh data/train-lite                    # varied engraving, seed 0
SEED=1 ./regenerate-dataset.sh data/validation-lite
./regenerate-dataset.sh data/synthetic-lite
```

This runs one `musicset build` per set (label set `CONF=conf-lite` by
default). It produces:
- stable ids;
- Verovio engraving. `varied` draws a font, staff size, line weights,
  spacing and page per score; `fixed` uses Verovio's default look;
- PNG pages and bbox JSONs under the label set;
- MEI and `**kern`;
- graphs: `edge_index`, `node_ids`, `json_files`, and `y` with the six edge
  classes (0 none, 1 structural, 2 modifier, 3 temporal, 4 sync-text,
  5 simultaneity).

Builds are resumable and never stale:
- Each score's manifest records its settings. A changed conf re-projects
  annotations and graphs from the stored canonical pages without
  re-rendering, so graphs always match the JSONs they were built from.
- Rejections (a staff off the page, or the optional density gates) are
  listed in `failures.jsonl`.
- Adding a glyph class touches four config files and must append to the end
  of `target_classes`. See "Adding a new glyph class" in the README.

Sanity checks:
- `uv run musicset view data/<set> --labels lite` shows boxes, graph edges
  and the score graph.
- The Kern tab engraves the ground-truth `**kern` next to the source pages.
- Stage 5's ceiling path audits the ground truth: a score far below 95%
  with ground-truth graphs has a broken source or annotation, not a model
  problem.

## Stage 3 — Train the detector

```
./pipeline-lite.sh conf-lite/train_640x896.yaml checkpoints_lite_896s
tail -f checkpoints_lite_896s/pipeline.log
```

`pipeline-lite.sh` chains the three stages unattended. Each stage
warm-starts from the previous one through `--fine-tune`, and each stage is
evaluated on validation-lite (`eval_<stage>.log`):

| stage | config | data | lr | epochs | current run |
| --- | --- | --- | --- | --- | --- |
| base | `conf-lite/train_640x896.yaml` (regression slots) | train-lite | 2e-3 | 40 | `run_001` |
| fine-tune | `conf-lite/fine_tune.yaml` | 1 real : 3 synthetic-lite | 2e-4 | 10 | `run_002` |
| consolidation | `conf-lite/consolidation.yaml` | 9 real : 1 synthetic-lite | 2e-4 | 30 | `run_003` |

How the pipeline runs:
- Every stage runs under `train.sh`, which relaunches `train-model
  --resume` after a crash.
- `.started_<stage>` / `.done_<stage>` markers in the output directory let
  a rerun after a reboot skip finished stages and resume an interrupted
  one.
- `FT_CONFIG`, `CONS_CONFIG`, `NUM_WORKERS` and `LOGGING_STEPS` override
  the defaults.
- The fine-tune and consolidation configs inherit the canvas and the scale
  ranges from their checkpoint (`input_size: null`, `scale_ranges: null`).
  Their own `fine_tune:` paths are overridden by the pipeline.

**Read the per-class table** in `eval_<stage>.log`, or run it yourself:

```
uv run eval-model --checkpoint checkpoints_lite_896/run_003 \
    --img-dir data/validation-lite/imgs --ann-dir data/validation-lite/annotations \
    --config conf-lite/config.json --hierarchy conf-lite/hierarchy.json
```

- Compare runs over a fixed class set (the 80 classes with validation
  ground truth), not the printed global mean, which skips zero-mAP classes.
- A class whose mAP worker crashed shows `[!] Crash` and scores 0. The
  pipeline retries the evaluation up to three times.

**A weak class has one of three causes,** each with a different remedy:
- **Rare in the corpus** (few training examples): generate synthetic data
  that targets it (Stage 0, `musicset synth`), rebuild `synthetic-lite`,
  and rerun the fine-tune and consolidation stages.
- **Badly annotated** (an extraction bug, a wrong box): inspect it with
  `musicset view` first. More data won't fix wrong boxes.
- **Below the input's resolving power**: data stops helping once
  localization is solid.
  - Diagnose it by matching predictions to ground-truth boxes. If every
    ground-truth box has a high-IoU prediction but the class votes go to a
    look-alike, this is the cause.
  - It was seen with `clefG8va` on the full track: a ~3 px "8" above a
    treble clef. Three synthetic rounds took it 0.019 → 0.117 → 0.317,
    then it plateaued.
  - The fix is resolution, which is why the canvas went from 640×640 to
    640×896.
  - Whenever the canvas changes, scale the level ranges by the area factor
    ([vision.md](vision.md) §5, lesson 3).

**Inspect a detector** with `visualize-model --checkpoint <run> --img_dir
<set>/imgs --ann-dir <set>/annotations`. It shows misses, false positives
and wrong-class detections per page, with the worst pages first.

**Troubleshooting:** a `CUBLAS_STATUS_NOT_INITIALIZED` / CUDA init error at
the first training step is an environment problem, not a model problem.
Upgrade the pinned wheels with `uv lock --upgrade-package torch
--upgrade-package torchvision && uv sync`.

## Stage 4 — Train the GNN and the joint model

```
./pipeline-gnn.sh checkpoints_lite_896s/run_003 checkpoints_gnn_lite_896s
tail -f checkpoints_gnn_lite_896s/pipeline.log
```

`pipeline-gnn.sh` runs four stages on the trained detector, each one
warm-starting from the previous (rationale in [graph.md](graph.md) §3):

| stage | command | config | current run |
| --- | --- | --- | --- |
| phase 1: detector frozen | `train-gnn` | `conf-lite/train_gnn_phase1.yaml` | `run_001` |
| phase 2: the neck adapts | `train-gnn` | `conf-lite/train_gnn_phase2.yaml` | `run_002` |
| joint: full backbone + detection loss | `train-joint` | `conf-lite/train_joint_full.yaml` | `run_003` |
| stage D: on its own detections, plus the keep head | `train-joint` | `conf-lite/train_stage_d.yaml` | `run_004` |

Before stage D, `cache-detections` stores the joint detector's boxes for
every training page (`<out>/detections-train`, a few minutes).

**The last run is the deliverable** (stage D; the joint run before it). It
is one checkpoint for both halves: pass it as both `--detector-checkpoint`
and `--gnn-checkpoint`.
`eval-model`, `visualize-model` and the `infer-*` / `validate-*` tools also
accept it as a detector.

After each stage the pipeline scores kern on validation-lite (Stage 5):
- `krn-gt-<stage>`: ground-truth boxes, predicted edges;
- `krn-det-<stage>`: detected boxes, after phase 2, joint and stage D;
- `krn-det-prune-staged`: stage D with the keep head pruning
  (`--prune-threshold`, `PRUNE` 0.5);
- each with a `.compare.txt`, plus `eval_joint.log` for the joint
  detector.

Settings:
- `P1_CONFIG`, `P2_CONFIG`, `JOINT_CONFIG` and `D_CONFIG` override the
  configs; `TRAIN_IMGS` sets the pages to cache detections for.
- `VAL` sets the evaluation set.
- `EXTRA_ARGS` is appended to every training command. Smoke-test a change
  before a day-long run:

  ```
  VAL=<small set> EXTRA_ARGS="--max-steps 30 --eval-steps 30 --save-steps 30" \
      ./pipeline-gnn.sh checkpoints_lite_896/run_003 <scratch dir>
  ```

**Monitoring.** TensorBoard is in `<out>/tb`. Watch:
- per-edge-class F1;
- the topology deltas. Deltas frozen at identical values across evals mean
  the model has collapsed into an attractor: all-background, or the
  inverse rare-class flood. See graph.md §4 for the diagnosis and §5 for
  the fixes.

The joint run logs its edge and detection losses separately.

**After a detector class change** (full track): train-gnn's phase 5
widens the GNN's class embedding instead of retraining (graph.md §3, "in
reserve"). Regenerate the annotations *and* graphs first, so the new
classes exist as nodes.

## Training on the DGX Spark

The Spark (`dgx-spark@spark-458f.local`, GB10, 119 GB unified memory) is
about 5× slower than an RTX 5090 but free to run for a day. Each pipeline
takes about a day there:
- detector: 25 h, at 3.0 it/s;
- GNN: 21 h (phase 1 7.0 h, phase 2 3.8 h, joint 9.9 h).

```
# once, from this machine: the three datasets (~11 GB, ~2 min)
rsync -a data/train-lite data/validation-lite data/synthetic-lite \
    dgx-spark@spark-458f.local:kakigori/data/

# on the Spark, after pushing the code from here
ssh dgx-spark@spark-458f.local
export PATH="$HOME/.local/bin:$PATH"            # uv is not on the non-interactive PATH
cd ~/musicset && git pull && cd ~/kakigori && git pull && uv sync
tmux new-session -d -s full './pipeline-full.sh checkpoints_lite_896s checkpoints_gnn_lite_896s'
tail -f pipeline-full.log

# back here, when it is done (the detection cache can stay there)
rsync -a --exclude detections-train dgx-spark@spark-458f.local:kakigori/checkpoints_lite_896s \
    dgx-spark@spark-458f.local:kakigori/checkpoints_gnn_lite_896s .
```

The pipelines resume after a reboot: relaunch the same tmux command. The
evaluations in a pipeline log are scored with the code of the time. When
the serializer or the repairs change, rescore the runs (Stage 5) before
comparing.

## Stage 5 — Measure: three paths to `**kern`

Three paths over the same validation set separate where errors come from
(graph.md §7):

```
VAL=data/validation-lite
RUN=checkpoints_gnn_lite_896/run_003

# ceiling: the serializer alone, on ground-truth graphs
uv run validate-groundtruth --graph_dir $VAL/graphs --json_dir $VAL/annotations \
    --out_dir $VAL/krn-me --roles_file conf-lite/structure.json

# GT boxes: the GNN + edge repairs, on ground-truth boxes
uv run validate-predictions --graph_dir $VAL/graphs --json_dir $VAL/annotations \
    --img_dir $VAL/imgs --out_dir $VAL/krn-gt --gnn_checkpoint $RUN \
    --config conf-lite/config.json --roles_file conf-lite/structure.json

# detections: the whole pipeline, as at inference
uv run validate-detections --img_dir $VAL/imgs --mei_dir $VAL/mei --out_dir $VAL/krn-det \
    --detector-checkpoint $RUN --gnn-checkpoint $RUN \
    --config conf-lite/config.json --roles_file conf-lite/structure.json

uv run compare-kern --mei_dir $VAL/mei --krn_dir $VAL/krn-det
```

- **`compare-kern` output:** pitch and rhythm recall (`TOTAL`), pitch
  precision and F1 with the output's note count relative to the ground
  truth (`PITCH PRECISION / F1 (1.15x notes out)`), and the median file.
  - Recall alone hides extra notes. Always read the precision line.
  - Scores that produced no kern are listed and excluded from the totals.
- **Ablations:**
  - `--no-repair` disables the edge repairs.
  - `--no-bbox-repair` disables the box repairs.
  - Each repair is also individually toggleable in code, for A/B runs.
- **Noise:** GNN inference is nondeterministic (cuDNN), about ±0.5 per
  file. Validate heuristics on the deterministic ceiling path, or on
  aggregate deltas well above the noise.

Current numbers (validation-lite, 500 scores, 57,398 notes; joint run
`checkpoints_gnn_lite_896/run_003`):

| path | pitch recall | rhythm recall | pitch precision | pitch F1 | median file (pitch) |
| --- | --- | --- | --- | --- | --- |
| ceiling | 99.8 | 99.6 | 99.8 | 99.8 | 100.0 |
| GT boxes | 97.2 | 93.2 | 90.4 | 93.7 | 99.1 |
| detections | 82.3 | 77.5 | 71.7 | 76.6 | 92.4 |

**Finding the bottleneck:** substitute ground-truth boxes for one part of
the detection path at a time (systems, rows, measures, glyphs), keeping
everything else, and see which swap recovers the most (graph.md §7). The
current lever is measure segmentation.

## Stage 6 — Look at the output

**On a labelled set:**

```
uv run musicset view data/validation-lite --labels lite --pred-kern data/validation-lite/krn-det
```

The Kern tab shows, side by side:
- the source pages;
- the ground-truth kern and the predicted kern, both engraved by Verovio
  at the same width;
- each kern's scores against the MEI.

Order the scores worst first to find failure patterns. Checkboxes hide
any column, and the kern texts can be shown below.

**On real scans** (no ground truth):

```
uv run infer-omr --pdf <score>.pdf --detector-checkpoint $RUN --gnn-checkpoint $RUN \
    --config conf-lite/config.json --roles_file conf-lite/structure.json \
    --output-dir inference_out/<name> --split
uv run python scripts/qa_views.py inference_out/<name>
```

`qa_views.py` writes, per page:
- `overlay/`: the scan with the structure the serializer reads after box
  repair. Rows show with their assumed five staff lines, along with
  systems, measures and clefs. A misplaced row is the first thing to look
  for.
- `render/`: the predicted kern engraved on white.

`musicset view <set> --scans <dir>` puts real scans next to the
augmentation preview.

## Full track (0.1.x) reference

The full track (`conf/`, ConvNeXt-B DINOv3 detector, 109 classes) was
trained with the same stages, run by hand:

- **Detector:** `./train.sh` with `conf/train.yaml`, then
  `conf/fine_tune.yaml` and `conf/consolidation.yaml`, updating each
  `fine_tune:` path.
- **Class expansion:** `retrain-model` with `conf/update_model.yaml`
  (`old_num_classes: N`). It copies the old head weights positionally
  (hence *append* classes, never reorder) and gives the new slots the focal
  bias prior. It then trains with a frozen backbone, then fully unfrozen.
  - Before launching, make sure every annotation source contains the new
    labels, synthetic sets included. Rebuilding with the new conf
    re-projects without re-rendering.
  - Reference: 107 → 109 classes (`run_004`, then `run_006` at mAP@.50
    0.724).
- **GNN:** phases 1, 2, 4 and 5 (`conf/train_gnn_phase*.yaml`) with
  `./train.sh uv run train-gnn --train-config ... --resume`. Pre-create the
  next `checkpoints_gnn/run_NNN`, and point each `fine_tune:` at the
  previous phase. Phase-1 reference: macro-F1 0.825 after 30 epochs
  (~2.2 h on an RTX 5090).

Results (recall only):

| set | ceiling | GT boxes + predicted edges | detections |
| --- | --- | --- | --- |
| validation-small | 98.4 / 97.7 | 93.2 / 80.4 | 29.0 / 23.3 (median 33) |
| test-small (224 unseen scores) | 90.0 / 89.0 | 82.3 / 75.0 | — |

Lessons from that track that still apply:
- **Class checks in repairs must be exact.** `"note"` must not
  prefix-match `"noteheadBlack"`; that single bug cost 42 pitch points.
- **Ownership means structural or modifier parenthood.** Temporal and
  sync neighbours never satisfy a repair guard.
- **Two measured negatives.** Chord → note recovery by containment pulls
  in other-voice notes (repair 11, default off). Temporal chain bridging
  changes nothing (repair 12, default off).
- **Serializer hardening:**
  - grace notes are detected relative to the page's median notehead
    height;
  - C-clef staff lines are measured from the glyph's position;
  - staff identity is tracked for optimized layouts (reduced systems
    mapped to parts by clef and key evidence, hidden parts padded with
    rests).
- **The test-small residue** is broken sources (gestural-only accidentals,
  tablature) and orchestral scores whose full part set never appears in
  one system. Naming their rows needs instrument-label OCR.

`graph/eval.py` (SER/CER/LER on the kern text) gives a sequence-level
comparison.

## Quick reference — what feeds what

| Artifact | Produced by | Consumed by |
| --- | --- | --- |
| `<set>/imgs`, `annotations`, `graphs`, `mei`, `kern` | `musicset build` (`regenerate-dataset.sh`) | training, validation, `compare-kern`, `musicset view` |
| `checkpoints_lite_896s/run_NNN` | `pipeline-lite.sh` (`train-model`) | `pipeline-gnn.sh`, `eval-model`, `visualize-model` |
| `checkpoints_gnn_lite_896s/run_NNN` | `pipeline-gnn.sh` (`train-gnn`, `train-joint`) | `validate-*`, `infer-omr`; the last run (stage D) as detector and GNN |
| `<out>/pipeline.log`, `eval_<stage>.log` | the pipelines | you, with `tail -f` |
| `<out>/krn-{gt,det,det-prune}-<stage>/` + `.compare.txt` | `pipeline-gnn.sh` | stage comparisons |
| `<out>/detections-train/` | `cache-detections` | stage D training (`det_dir`) |
| `pipeline-full.log` | `pipeline-full.sh` | you, with `tail -f` |
| `krn-me/`, `krn-gt/`, `krn-det/` | `validate-groundtruth` / `-predictions` / `-detections` | `compare-kern`, `musicset view --pred-kern` |
| `inference_out/<name>/` | `infer-omr --split` | `scripts/qa_views.py` |
