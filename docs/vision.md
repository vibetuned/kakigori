# The vision detector: design, training, and lessons learned

This document explains *how the detection half of the pipeline works and how
we train it*: the model, the target assignment, the training recipe, how to
evaluate it, and the lessons that shaped the current design. It is the
companion of [graph.md](graph.md). For a file-by-file reference see
[modules/vision.md](modules/vision.md); for the commands see
[end-to-end-training.md](end-to-end-training.md); for the measurements behind
the current choices see [lite-results-2026-10.md](lite-results-2026-10.md).

## 1. What the detector does

The detector finds every musical primitive on a page as a class + box. The
lite label set (`conf-lite/config.json`) has 99 classes:
- glyphs: noteheads, stems, flags, beams, rests, accidentals, clefs,
  key and time signatures, dynamics, ornaments;
- the structure the serializer reads:
  - `system` (all staves of a line of music);
  - `system-staff` (one staff's five-line row across the system);
  - `measure`;
  - `staff` (one staff within one measure);
  - `layer` (one voice within that staff and measure).

Those boxes become the GNN's nodes ([graph.md](graph.md)) and the serializer's
geometry: pitch is read from a notehead's position on its row's five lines,
so the precision of a few boxes (rows above all) matters more than the mAP
of most classes.

## 2. The model

```
page ─▶ letterbox into the canvas (640×896, grey) ─▶ ReGLA-M stages 1-3
     ─▶ PANet neck (128 ch) ─▶ one decoupled head per level (strides 4/8/16)
     ─▶ class heatmaps + box regression per cell
```

- **Canvas.** The page is fitted, undistorted, into a page-shaped
  (width, height) canvas, and the rest is padded grey.
  - The current detector uses 640×896, where an A4 page fills 634×896.
    The old square 640 letterbox gave it 453×640, so that is 1.4× the
    linear resolution.
  - 448×640 is the low-cost option: the same resolution as the square 640
    canvas with 0.7× the pixels.
  - Both sides must divide by the coarsest stride. One shared implementation
    (`vision/dataset.letterbox_image`, `page_to_canvas`, `canvas_to_page`)
    serves every tool.
- **Backbone.** ReGLA-M ([arXiv 2602.05262](https://arxiv.org/abs/2602.05262),
  `vision/regla.py`), stages 1–3, one grey channel, trained from scratch:
  4.63M parameters (the detector has 4.97M; the 0.1.x ConvNeXt-B DINOv3 had
  60.5M).
  - Stages 1–2 are ELRF blocks (two small depthwise convolutions with a
    linear bottleneck).
  - Stage 3 is attention: a conditional position encoding, then RGMA, a
    ReLU linear attention gated by a sigmoid value, global over all tokens,
    computed in fp32.
  - `LayerNorm2d` uses the fused kernel in training and a permute-free form
    when traced or exported (Core ML / the ANE).
- **Neck.** PANet top-down fusion (optionally bottom-up) over any number of
  levels. A fourth level (ReGLA stage 4, stride 32) is supported but not
  used: it was tested and is deprioritised (§6).
- **Heads.** One decoupled head per level: a class heatmap
  (sigmoid, focal-loss prior) and a box regression per cell and
  *regression slot* (offset of the centre inside the cell, log width and
  height). Each class reads its box from its slot (§3).
- **Stored with the weights:** the canvas (`input_wh`), each level's
  scale range (`level_areas`) and each class's regression slot
  (`reg_slot`). `detector_for_checkpoint` also reads the
  backbone, the level count and the bottom-up path from the weight names,
  so no tool takes architecture flags.
  - `load_detector` also accepts a GNN or joint run and uses the detector
    inside it.
  - Pre-October checkpoints load as square 640 with the default ranges;
    checkpoints from before 2026-10-22 with one shared slot.

## 3. Targets and decoding

**Assignment (`vision/trainer.OMRTrainer.build_targets`).** Each ground-truth
box goes to one level by its area, and to the grid cell holding its centre.
- **Scale ranges** are box areas in fractions of a 640×640 canvas
  (pixel area / 640²): plain normalized areas at square 640, and the same
  pixel sizes on any other canvas.
  - At 640×640 the defaults are `(0, 1e-4), (1e-4, 1e-3), (1e-3, 2)`.
  - The 640×896 detector scales them by the area factor (896/640)² = 1.96,
    so that every class stays on the level it was tuned on (§5, lesson 3).
- **Multi-label classes, one box per slot.** A cell's class map marks
  *every* class centred there as positive. Its regression targets hold one
  box per slot (`model.regression_slots`), the smallest of the boxes
  sharing that cell and slot:
  - each structure class (system, system-staff, measure, staff, layer,
    page-margin) has its own slot;
  - the composites (note, chord, keySig, meterSig, mRest, tuplet) share
    one;
  - every other glyph uses slot 0.

  With one shared slot, 21% of the training boxes shared their cell with
  another box of their level and lost their target to it (staff and
  measure under their layer, a notehead under its note, a beam centred on
  a note, a system over its only row). With the slots, 0.4% do.
- **Resolved, not written over.** The smallest box per (cell, slot) is
  chosen explicitly (§5, lesson 8).
- **Losses:** `DynamicFocalLoss` on the class maps (gamma annealed 2 → 4
  over training) plus 5 × CIoU on the positive cells. The CIoU floors the
  predicted box size at 1e-6 of the canvas (§5, lesson 4).

**Decoding (`vision/utils.decode_model_outputs`).** Sigmoid, confidence
threshold, then class-aware NMS at IoU 0.5.
- **Multi-label by default:** every class over the threshold at a cell is
  emitted, sharing the cell's box. Coincident boxes need it, because one
  cell carries several classes:
  - measure / staff / layer on a single staff;
  - `mRest` over its whole-rest glyph;
  - a one-sharp `keySig` over its `keyAccidSharp`.
- **Each class takes its box from its own slot**, so coincident classes
  no longer share one box.
- **The OMR path decodes the temporal anchors argmax-only**
  (`argmax_only` for note and chord in `graph/infer.detect_page_nodes`).
  As a secondary label they carry a notehead's box, and become duplicate
  events that NMS cannot merge with the real one.
- `--no-multi-label` restores the argmax decoding for comparisons with old
  numbers.

## 4. Training recipe

Three stages, each warm-started from the previous one. `pipeline-lite.sh`
chains them unattended (resumable, one log to `tail -f`):

| stage | config | data | lr | epochs |
| --- | --- | --- | --- | --- |
| base | `conf-lite/train_640x896.yaml` (`regression_slots: true`) | train-lite (10,387 pages) | 2e-3 | 40 |
| synthetic fine-tune | `conf-lite/fine_tune.yaml` | 1 real : 3 synthetic-lite | 2e-4 | 10 |
| consolidation | `conf-lite/consolidation.yaml` | 9 real : 1 synthetic | 2e-4 | 30 |

- **Common settings:** batch 4, bf16, cosine schedule with 10% warmup,
  gradients clipped at 1.0, and the scanner augmentations of `OMRDataset`
  (brightness/contrast, Gaussian noise, coarse dropout, JPEG,
  salt-and-pepper, illumination gradients; no morphology, which erases
  sub-pixel stems and barlines).
- **Learning rates** come from 2–3k-step sweeps. A from-scratch ReGLA wants
  20× the 1e-4 used to fine-tune ConvNeXt, and the synthetic stages gain
  nothing above 2e-4.
- **Inheritance:** the fine-tune and consolidation configs set
  `input_size: null` and `scale_ranges: null`, so they inherit the canvas
  and ranges from their checkpoint. Setting either one moves a model to
  another canvas or another set of ranges; ReGLA is fully convolutional.
- **The synthetic set** targets the weak classes: exercises, short
  rare-meter exercises, piano pieces, and mutations/arrangements of real
  scores, built with `musicset synth` and the `rare` profile.
- **Cost:** about 25 h for the three stages on a DGX Spark (GB10), where a
  640×896 step takes 297 ms. An RTX 5090 is about 5× faster (51 ms).

## 5. Lessons learned

1. **Decode the way you trained.** Training marks every class centred on a
   cell positive, but argmax decoding kept one class per cell. The coincident
   classes were then reported as weak, and "hard cap" diagnoses were written
   for them. mRest's "semantic cap" was really an argmax casualty: multi-label
   decoding took it from 0.08 to 1.00, and the 80-class mean from 0.627 to
   0.731. The same artifact made ConvNeXt look better on system, staff and
   mRest; under multi-label decoding ReGLA wins on 63 of 80 classes.
2. **Map boxes back with the resize you applied.** Page coordinates were
   recovered with the unrounded letterbox scale while the page had been
   resized with rounded sizes. That is up to 2 page pixels of drift, enough
   to fail IoU on 3 px stems and barlines. Training and eval-model work in
   canvas space and were unaffected; everything in page space (inference,
   the visualizer, the GNN tools) was.
3. **Changing resolution changes which head sees a class.** With
   pixel-area ranges, glyphs that grew past a level boundary at 1.4×
   resolution moved to another head:
   - accidentalFlat, keyAccidNatural and timeSig1 crossed 1e-4;
   - clefG8vb and grpSym crossed 1e-3.

   In a warm start those classes collapsed (80-class mean 0.693). Scaling
   the ranges by the area factor kept every class on its level and gave
   0.767. Do the same whenever the canvas changes.
4. **A degenerate box NaNs bf16 training.** A predicted box of size
   exp(−40) rounds to zero width and height, and CIoU's `atan(w/h)` becomes
   `atan(0/0)`. With bf16 there is no grad scaler to skip that step, so the
   NaN reaches every weight. It appeared on page edges, where a square-letterbox model had
   never seen content; hence the 1e-6 size floor in `CIoULoss`.
5. **A new canvas needs training, not just a flag.** Zero-shot on a
   page-shaped canvas the square-trained model scored 0.62 (448×640) and
   0.46 (640×896), because the page now touches the canvas edge. Five
   consolidation epochs recovered it.
6. **Natural-image pretraining is not what OMR needs.** A 12× smaller
   from-scratch ReGLA on one grey channel beat ConvNeXt-B DINOv3 by about
   0.24 mAP@.50 on the same data (multi-label decoding, 80 classes), and it
   kept detection intact through joint training. Every July ConvNeXt joint
   run collapsed stems and barlines (barlineSingle 0.99 → 0.11).
7. **Read the per-class table, not the global mean.**
   - eval-model's global mean only averages classes with a non-zero mAP,
     so a run that loses a class entirely is not penalised. Compare runs
     over a fixed class set (the 80 classes with validation ground truth)
     and per class.
   - A per-class mAP worker can crash and silently score its class 0
     (`[!] Crash` in the log). `pipeline-lib.sh` retries; check before
     comparing.
   - Classes with 1–5 validation instances swing by ±0.5 between runs.
8. **Never resolve collisions by write order on the GPU.** `build_targets`
   sorted boxes largest first so that, writing them in one vectorized
   assignment, the smallest would win a shared cell. With repeated indices
   a CUDA index assignment keeps an arbitrary write, separately per
   channel.
   - Measured: the last write survived 65% of the time.
   - In 12% of the shared cells the four regression channels came from
     different boxes: one box's centre with another's size.

   Until 2026-10-22 a fifth of the boxes trained toward partly random,
   partly corrupt targets. Collisions are now resolved before writing, so
   every index is written once.

## 6. Current model and where it is weak

`checkpoints_lite_896/run_003` (consolidation), validation-lite, multi-label
decoding. It was trained with one shared regression slot and the
write-order collisions of lesson 8; its successor with slots is
`checkpoints_lite_896s`.

| | 80 classes | ≥ 20 instances | instance-weighted | mAP@.50:.95 |
| --- | --- | --- | --- | --- |
| 640×896 (current) | 0.795 | 0.856 | 0.948 | 0.662 |
| same recipe at 640×640 | 0.731 | 0.780 | 0.925 | 0.575 |

The joint GNN run (`checkpoints_gnn_lite_896/run_003`) carries a further
trained copy of this detector: 0.800 over 79 classes, with no stem or
barline collapse.

**Still weak:**
- **Beams:** beam32 0.06, beam16 0.41, beamBroken 0.45, beam8 0.68. They
  are thin and diagonal, and in beamed groups they decide durations.
- **timeSig9:** 0.27.
- **meterSig:** 0.52.
- **system-staff:** 0.41. The detector also places rows a median 3.6
  pitch steps off; the graph side now rebuilds row geometry from the clefs
  (bbox repair 7, [graph.md](graph.md) §6).
- **page-margin:** 0.59.

**Tested and not adopted:** ReGLA stage 4 with a stride-32 head. In a warm
start it raised system to .99 and measure/staff to .98, but its new fusion
block disturbed the stride-16 path (tuplet, volta and page-margin dropped).
Multi-label decoding had already closed most of the structure gap.

**Deployment notes:**
- 448×640 halves the compute at the old square resolution (1.45 vs
  2.96 ms per page batched on an RTX 5090).
- A Core ML export needs a wrapper that returns tensors (`MusicDetector`
  returns a list of dicts).

## 7. Inspecting a detector

- `visualize-model --checkpoint <run> --img_dir <set>/imgs --ann-dir
  <set>/annotations` shows, per page:
  - boxes, heatmaps, and backbone or neck feature maps;
  - a comparison with the ground truth: false positives, wrong-class
    detections and misses, with per-page precision, recall and F1;
  - a pass that ranks pages worst first.

  Multi-label decoding is a toggle.
- `eval-model --checkpoint <run>` gives per-class mAP in TensorBoard. It
  accepts a GNN or joint run too.
