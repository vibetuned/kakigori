# LITE track results — October 2026

The 0.2.0 LITE track ([roadmap](roadmap-0.2.0.md)) was rebuilt between
2026-10-18 and 2026-10-21, and four things changed:

- the data, now built with musicset and varied renders;
- the backbone, ReGLA on one grey channel instead of ConvNeXt DINOv3;
- the decoding, now multi-label;
- the input canvas, 640×896 and page-shaped instead of a 640 square.

The GNN curriculum and the joint model were then retrained on top.
This page records what was measured, the evaluation caveats behind the
numbers, and the bugs found on the way.

## Summary

| | July (ConvNeXt, fixed renders) | now (ReGLA 640×896, varied renders) |
| --- | --- | --- |
| detector params | 60.5M | **4.97M** |
| detector mAP@.50, 80 classes, multi-label | — | **0.795** (0.731 with the same recipe at 640) |
| serializer ceiling (GT graphs) | 99.7 / 99.4 | 99.8 / 99.6, median 100 / 100 |
| GT boxes + predicted edges | 93.8 / 89.5 | **94.4 / 90.8**, median 99.1 / 97.1 |
| detection-driven kern | 49.6 / 44.2, median 53.8 (joint, pruned) | **53.4 / 47.5**, median 53.6 / 47.5 (joint) |
| joint model stems / barlines | collapsed (0.06 / 0.11) | **0.99 / 0.99** |

The kern columns are pitch / rhythm match on validation-lite (500
scores) as scored by `compare-kern`.

The July and current numbers come from different renders of the same
scores. They show the trend, not a strict A/B.

The detector improved a lot, but the detection-driven kern is only 3.8
points above July's, and its median is the same. It sits far below the
stage-D gate (median ≥ 80%). No same-data GNN baseline was run on the
640 detector, so the July run is the only comparison.

The bottleneck is now the graph side meeting detection noise. That is
exactly what stage D (training the GNN on its own detections) targets.

## Data

`data/train-lite` (9,500 scores, 10,387 pages) and
`data/validation-lite` (500 scores, 555 pages) were rebuilt in place
with musicset:

```
musicset build ... --profile varied
```

The seed is 0 for train and 1 for validation. The `varied` profile
samples engraving options per score: page size and margins, fonts,
spacing, and headers/footers.

`data/synthetic-lite` was rebuilt with `musicset synth`: 5,991 scores
and 10,497 pages. It is made of:

- exercises, short rare-meter exercises and piano pieces, using the
  `rare` profile (key changes for `keyAccidNatural`, whole-measure
  rests, fermatas, rare meters);
- 1,500 mutations and arrangements of train-lite.

Kakigori's own dataset module was removed.

Old models evaluated on the varied validation set drop 25–30% from
their fixed-render scores. Results before 2026-10-18 are therefore not
directly comparable.

## Evaluation changes (read before comparing numbers)

- **Multi-label decoding is now the default.** `build_targets` marks
  every class centred on a grid cell as positive, but the decoder used to
  keep only the argmax class per cell. Coincident boxes therefore
  suppressed each other:
  - measure / staff / layer on a single staff;
  - `mRest` over its whole-rest glyph;
  - a one-sharp `keySig` over its `keyAccidSharp`.

  `decode_model_outputs(multi_label=True)` emits every class over the
  threshold, sharing the cell's box. On run_006 the mean over the fixed
  80 classes goes from 0.627 to **0.731**. Gains include mRest .08 → 1.00,
  clefC → 1.00, measure/staff/layer ≈ .5 → .93 and system .55 → .73.
  page-margin pays for it (.71 → .46): its cell's single regression box
  belongs to a smaller class. `--no-multi-label` restores the old
  decoding.
- **Compare runs over a fixed class set.** `eval-model`'s global mean
  averages only classes with mAP > 0, so a run that loses a class
  entirely is not penalised. All the comparisons below average over the
  80 classes with ground truth in validation-lite, counting a missing
  class as 0. Some also give the instance-weighted mean, or the mean
  over the 59 classes with ≥ 20 validation instances. Classes with 1–5
  instances (`turn`, `timeSig0`, `fing`, …) swing by ±0.5 between runs.
- **eval-model worker crashes zero a class.** A per-class mAP worker can
  exit with code 255; the class then scores 0 and leaves the global
  mean. This happened to `repeatMark` and `dynamicForte` once each.
  Grep eval logs for `[!] Crash` before comparing; `pipeline-lib.sh`
  retries.
- **Letterbox back-mapping fixed.** Boxes were mapped back to page
  pixels with the unrounded letterbox scale, while the page had been
  resized with rounded sizes. The drift reached about 2 page pixels at
  the far edge, enough to fail IoU on 3 px stems and barlines.
  - Affected: `infer-omr` / graph inference coordinates, `infer-model`
    JSONs and the visualizer.
  - Not affected: training and eval-model, which both work in canvas
    space.
  - There is now one shared letterbox implementation
    (`vision/dataset.letterbox_image`, `page_to_canvas`,
    `canvas_to_page`).
- **compare-kern** now prints the median file. It also reports scores
  without kern output: those are excluded from the total, not scored 0.

## Detector: ReGLA backbone

The detector now uses ReGLA-M (arXiv 2602.05262, `vision/regla.py`),
stages 1–3, trained from scratch on one grey channel:

- 4.63M backbone parameters; the detector has 4.97M, against 60.5M with
  ConvNeXt-B DINOv3;
- 33.5 ms / 3.58 GiB per batch-4 training step at 640 on an RTX 5090,
  against 41.2 ms / 5.70 GiB for ConvNeXt.

The learning rates were tuned from scratch. Each sweep reports
validation mAP@.50 after 2–3k steps:

| stage | sweep | chosen |
| --- | --- | --- |
| base | 1e-4 .215 · 3e-4 .258 · 1e-3 .329 · 2e-3 .376 · 4e-3 .384 | 2e-3 |
| synthetic fine-tune (1:3) | 2e-4 .653 · 5e-4 .639 · 1e-3 .635 · 2e-3 .633 (weak classes flat) | 2e-4 |
| consolidation (9:1) | 5e-5 .650 · 1e-4 .658 · 2e-4 .671 · 4e-4 .667 | 2e-4 |

**Clean backbone A/B.** ConvNeXt-B DINOv3 was retrained on the same
train-lite (run_007, lr 1e-4) and compared with the ReGLA base (run_004),
using multi-label decoding over the 80 classes:

- ConvNeXt 0.435, ReGLA **0.678**;
- ReGLA is better on 63 classes and ConvNeXt on 6, none by more than
  0.1;
- under argmax decoding, ConvNeXt seemed to win on system, staff and
  mRest, but that was an artifact of argmax.

The ConvNeXt fine-tune and consolidation stages were not run.

## Input canvas and detection levels

The detector now stores its canvas (`input_wh`) and per-level scale
ranges (`level_areas`) in its weights. Every tool reads them from the
checkpoint: eval, inference, the visualizer, the GNN datasets and later
training stages. Scale ranges are pixel areas in fractions of a 640×640
canvas, so they keep their meaning at square 640.

**Warm-start probe.** Each variant got 5 consolidation epochs from
run_006 (13.5k steps, 9:1, lr 2e-4); a square-640 control got the same
treatment.

| variant | 80 classes | instance-weighted | weak 20 |
| --- | --- | --- | --- |
| control, 640×640 | .726 | .924 | .376 |
| 448×640 (same resolution, 0.70× pixels) | .726 | .921 | .392 |
| 640×896 (1.4× linear), pixel-area ranges | .693 | .944 | .397 |
| **640×896, ranges ×1.96 (levels kept)** | **.767** | **.947** | **.470** |
| stage 4 grown (stride-32 head) | .713 | .926 | .322 |

- **Higher resolution works once each class keeps its detection level.**
  With pixel-area ranges, the glyphs that grow past a boundary moved to
  another head and collapsed in the short warm start: accidentalFlat,
  keyAccidNatural and timeSig1 past 1e-4; clefG8vb and grpSym past
  1e-3. When resolution changes, scale the ranges by the area factor.
  `conf-lite/train_640x896.yaml` does this.
- **448×640 is a free speed option.** It matches square 640 at 0.70× the
  pixels.
- **Stage 4 is deprioritised.** It pushed system to .99 and
  measure/staff to .98. But its new fusion block disturbs the stride-16
  path, and tuplet, volta and page-margin dropped. Multi-label decoding
  had already closed most of the structure gap.
- **A model needs training on its canvas.** Zero-shot on the new
  canvases is poor (.62 at 448×640, .46 at 640×896): the page now
  touches the canvas edge, which the square letterbox never showed the
  model.

Forward pass per page at batch 8 on the RTX 5090: 1.45 ms at 448×640,
2.11 ms at 640×640, 2.96 ms at 640×896. Stage 4 adds about 10% and
takes the detector to 10.3M parameters. At batch 1 the GPU is
launch-bound at about 7.5 ms whatever the canvas.

## Detector: 640×896 from scratch

The full recipe was re-run on the 640×896 canvas with the ×1.96 ranges:
`pipeline-lite.sh conf-lite/train_640x896.yaml checkpoints_lite_896` on
the DGX Spark, 25 h, no crashes. Each stage is compared with the same
stage at 640 (multi-label decoding, mAP@.50 unless noted):

| stage | 80 classes | ≥ 20 instances (59) | instance-weighted | mAP@.50:.95 |
| --- | --- | --- | --- | --- |
| base: 640 → 896 | .678 → **.742** | .768 → **.829** | .924 → **.947** | .515 → **.597** |
| fine-tune | .710 → **.790** | .773 → **.847** | .924 → **.949** | .553 → **.654** |
| consolidation | .731 → **.795** | .780 → **.856** | .925 → **.948** | .575 → **.662** |

Each 896 stage is better than its 640 counterpart on about 50 classes and
worse on 8–11, mostly classes with 1–5 validation instances.

Selected classes after consolidation, 640 → 896:

| class | 640 | 896 | | class | 640 | 896 |
| --- | --- | --- | --- | --- | --- | --- |
| timeSig2 | .561 | .855 | | system | .730 | .808 |
| timeSig9 | .123 | .274 | | system-staff | .224 | .405 |
| meterSig | .359 | .523 | | page-margin | .461 | .590 |
| keySig | .760 | .883 | | measure / staff | .927 | .951 |
| fermata | .488 | .686 | | tuplet | .547 | .652 |
| mordent | .272 | .580 | | beam8 | .510 | .678 |
| chord | .829 | .889 | | beam16 | .352 | .405 |
| note | .953 | .977 | | beamBroken | .344 | .452 |
| flag8thUp | .718 | .755 | | beam32 | .009 | .065 |

**Still weak** (classes with ≥ 20 instances): beam32 .06, timeSig9 .27,
beam16 .41, system-staff .41, beamBroken .45, meterSig .52, page-margin
.59. Beams are the main remaining family.

**Watch:**

- `system` is lower at 896 in the base and fine-tune stages (.78 vs .93);
  consolidation brings it back to .81.
- `flag16thUp` is lower at 896 in every stage (16 instances).

## GNN and joint model on the 640×896 detector

The run was `pipeline-gnn.sh checkpoints_lite_896/run_003
checkpoints_gnn_lite_896` on the DGX Spark, about 21 h with no crashes:

| stage | time |
| --- | --- |
| phase 1, detector frozen | 7.0 h |
| phase 2, neck adapts | 3.8 h |
| joint, full backbone + detection loss | 9.9 h |

Results on validation-lite (500 scores):

| | GT boxes + predicted edges | detection-driven |
| --- | --- | --- |
| phase 1 | 94.1 / 90.1, median 98.8 / 96.6 | — |
| phase 2 | 94.2 / 90.2, median 98.9 / 96.5 | 53.1 / 47.3, median 53.8 / 47.8 (499 of 500 serialize) |
| **joint** | **94.4 / 90.8**, median 99.1 / 97.1 | **53.4 / 47.5**, median 53.6 / 47.5 (500 of 500) |

**Joint model:**

- **Edges:** F1 macro 0.689 (structural .938, modifier .809, temporal
  .876, simultaneity .518; sync-text is empty in lite by construction),
  level with July's best (0.690).
- **Detector:** mAP@.50 over 79 classes .800, against .794 for the
  standalone detector. Stems and barlines stay at 0.99. In July every
  ConvNeXt joint run collapsed them (barlineSingle 0.99 → 0.11, stems
  → 0.06).

**Detection-driven score spread (joint):**

- 33 scores reach ≥ 80% pitch, 10 reach ≥ 95%, and 2 are perfect;
- the quartiles are 42.2% and 65.8%;
- the ceiling (serializer on GT graphs) is 99.8 / 99.6, with 494 scores
  at ≥ 95%.

Joint and phase 2 are within noise of each other on detections (+0.3).

**Serializer fix (musicset `views/kern.py`, `_reference_measure`).**

- **The bug.** Spines were created from the leftmost measure of the
  reference system. On predicted graphs that measure can be missing
  some measure → staff edges, which gave either:
  - no spines: the whole score fails, with the misleading "Error: No
    valid systems found across any page";
  - too few spines: a whole staff is dropped.
- **The fix.** Spines now come from the leftmost measure with a staff
  cell on every row, falling back to the one with the most staff cells.
  On ground-truth graphs the choice is unchanged.
- **Effect on detections.** Scores only went up. In phase 2, 19 scores
  gained more than 5 points (one went from 4% to 98%) and none dropped;
  the mean gain was +1.7.
- **Scores in the pipeline logs.** The Spark's own `pipeline.log`
  predates the fix: it shows 51.2 / 53.0, with 3 / 9 scores that did not
  serialize. The numbers above are re-runs with the fix.

## Bugs fixed

- **Coincident boxes lost to argmax decoding:** multi-label is now the
  default (above).
- **Letterbox back-mapping drift:** about 2 page px (above).
- **CIoU NaN.** A predicted box of size exp(−40) rounds to zero width
  *and* height in xyxy, and `atan(w/h)` = `atan(0/0)` then poisons every
  weight: bf16 training has no grad scaler to skip a NaN gradient. It
  appeared on pages touching the canvas edge, in a warm start at
  640×896. `CIoULoss` now floors the predicted size at 1e-6 of the
  canvas.
- **Serializer reference measure** (above).
- **`validate-detections` silent failure:** a score with no edges now
  logs a warning.
- **`eval-model` on a bare weights directory:** it no longer needs a
  `checkpoint-N` folder.

## Infrastructure

- **Checkpoints carry their layout:** backbone, detection levels,
  bottom-up neck, canvas and scale ranges.
  - The inference CLIs lost `--input-size`, `--out-indices` and
    `--use-bottom-up`.
  - `load_detector` also accepts a GNN or joint run and uses the
    detector inside it, so a joint run serves as its own detector in
    `eval-model`, `infer-*`, `visualize-model` and `validate-detections`.
  - Pre-October checkpoints load as square 640 with the default ranges.
- **Unattended pipelines.** `pipeline-lite.sh` runs the detector stages
  and `pipeline-gnn.sh` the GNN stages, sharing
  `scripts/pipeline-lib.sh`.
  - Every stage runs under `train.sh`. Markers let a pipeline resume
    after a reboot. Evaluations run after each stage, and one log is
    meant to be followed with `tail -f`.
  - `pipeline-gnn.sh` accepts `VAL` and `EXTRA_ARGS` for smoke tests.
- **Visualizer.** `visualize-model --ann-dir` compares predictions with
  the ground truth: false positives, wrong-class detections and misses,
  with per-page precision, recall and F1. It can rank pages worst first
  and show backbone or neck feature maps. Multi-label decoding is a
  toggle.
- **DGX Spark (GB10).** It is GPU-bound and about 5× slower than the
  RTX 5090: a 640×896 training step takes 297 ms against 51 ms.
  - Measured throughput: detector 3.0 it/s; GNN phase 1 4.9 it/s,
    phase 2 4.7 it/s, joint 2.0 it/s.
  - A full detector or GNN pipeline takes about a day there.

## Open issues and next steps

1. **Stage D: train the GNN on its own detections.** This means
   Hungarian matching plus a curriculum from GT boxes, to jittered
   boxes, to the model's own detections. The detector gained 6 mAP
   points over the same recipe at 640, yet detection-driven kern is only
   3.8 pitch points above July's, with the same median: the graph side
   meeting detection noise is now the bottleneck. Start with a breakdown
   of where the detection-path files lose notes (structure,
   bbox_repair, edges), using the 42–66% interquartile range.
2. **Beams** (beam16/32, beamBroken) are the weakest detection family
   and decide durations in beamed groups.
3. **system-staff (.41) and page-margin (.59).** Stage 4 helps
   structure but needs a from-scratch run; it is deprioritised.
4. **The pruned 91-class set** (July's stage-C winner) has not been
   re-tested on ReGLA.
5. **Export to Core ML / the ANE.** It needs a tensor-returning wrapper
   around `MusicDetector`; `LayerNorm2d` already switches to its
   permute-free form when traced. 448×640 is the low-cost canvas
   option.
6. **Lower the `rare` profile's ornament rate.** After the 640 fine-tune,
   fermatas collapsed (.51 → .10), read as mordent, turn or trill. At
   896 fermata recovers to .69.
