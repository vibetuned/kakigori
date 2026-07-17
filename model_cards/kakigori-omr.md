---
license: agpl-3.0
base_model: timm/convnext_base.dinov3_lvd1689m
pipeline_tag: image-to-text
library_name: pytorch
tags:
- optical-music-recognition
- music
- music-notation
- graph-neural-network
- humdrum
- kern
---

# kakigori — end-to-end optical music recognition

Full OMR pipeline of the [kakigori](https://github.com/vibetuned/kakigori)
project: score page images → Humdrum `**kern`. This repository holds the
**graph model** (a GNN wrapper that turns detections into a music graph);
it works together with the [detector](https://huggingface.co/vibetuned/kakigori-vision).

## How it works

1. **Detection** ([kakigori-vision](https://huggingface.co/vibetuned/kakigori-vision)):
   109 classes of glyphs and layout structure per page.
2. **Graph construction** (this model): a GATv2 edge classifier over
   per-system candidate edges. Node features = RoI-Align features from the
   detector backbone through the model's own PANet neck (adapted for the
   relational task — the detection neck stays untouched), normalized box
   coordinates, and a learned class embedding. Six edge classes: none,
   structural, modifier, temporal, sync-text, simultaneity.
3. **Graph repair**: guarded spatial heuristics restore geometric
   certainties the edge model may miss (sub-glyph containment,
   measure→staff cells, orphaned events) without ever overriding a model
   prediction.
4. **Serialization**: a Humdrum `**kern` serializer with staff-identity
   tracking for optimized layouts, geometric C-clef line resolution,
   mid-piece clef/key/meter changes, dynamics + pedal spines, beams,
   tuplets, ties across system breaks.

## Training

Trained on ~2.5k music graphs (8.8k pages) derived from MEI groundtruth in
a five-phase curriculum: frozen-vision GNN training → PANet-neck adaptation
→ (full unfreeze, held in reserve) → box-jitter robustness (σ = 1.5 px
noise on boxes) → class-expansion adaptation after the detector grew to
109 classes. Edge macro-F1: **0.856** on held-out graphs (structural 0.93,
sync-text 0.89, simultaneity 0.84, temporal 0.79, modifier 0.70).

## Evaluation

`compare-kern` matches per-staff/per-measure multisets of sounding pitches
and pitch+duration pairs against MEI groundtruth. "Ceiling" uses
ground-truth boxes and edges (serializer alone); "end-to-end" uses
ground-truth boxes with predicted edges + repairs.

| set | ceiling pitch/rhythm | end-to-end pitch/rhythm |
| --- | --- | --- |
| validation-small (49k notes) | 98.4% / 97.7% | 93.2% / 80.4% |
| test-small (83k notes, unseen) | 90.0% / 89.0% | 82.3% / 75.0% |

Median file on the unseen set: 100% / 100% (200 of 223 files ≥99% pitch).

## Known limitations

- Inference from **detected** boxes (the `infer-omr` path) is rougher than
  the ground-truth-box numbers above; the box-jitter curriculum narrows
  but does not close that gap.
- Large orchestral scores whose full part set never appears in a single
  system cannot be staff-identified geometrically (needs instrument-label
  OCR) — the two such scores in the test set score far below average.
- Tablature and percussion notation are out of scope; scores whose
  accidentals exist only gesturally in the source (never printed) are
  unrecoverable by any OMR.
- GNN inference is nondeterministic on GPU (cuDNN): per-file results
  fluctuate by ~±0.5 points between runs.

## Usage

```bash
# weights -> checkpoints/release/{vision,omr}/model.safetensors
uv run infer-omr --pdf score.pdf \
                 --detector-checkpoint checkpoints/release/vision \
                 --gnn-checkpoint checkpoints/release/omr \
                 --output-dir out/
# outputs: out/score.krn + per-page detections (png + json)
```

License: AGPL-3.0-only (the DINOv3 backbone weights carry their own
upstream license).
