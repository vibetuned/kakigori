# The score graph: design, training, and lessons learned

This document explains *how the graph half of the pipeline works and how we
train it* — the concepts, the phase curriculum, and the debugging lessons
that took the GNN from flat-zero F1 to a working edge classifier. For a
file-by-file reference see [modules/graph.md](modules/graph.md); for the
operational commands see [end-to-end-training.md](end-to-end-training.md).

## 1. The graph design

A page of music is modeled as a typed graph over the detector's primitives.
Node roles (configured in `conf/structure.json`) shape what may connect to
what:

- **Temporal anchors** (notes, chords, rests) — string together
  left-to-right to form the chronological Humdrum backbone via *temporal*
  edges.
- **Structural components & modifiers** (stems, beams, accidentals, dots,
  ornaments) — only ever connect to temporal anchors, via *structural* or
  *modifier* edges. A stem never connects directly to a clef.
- **Context globals** (clefs, key/meter signatures, dynamics, pedals) —
  dictate staff state; they attach to measures or staves rather than to
  individual notes.
- **Virtual nodes** — a "sink node" is not a physical glyph, so it is
  injected into the PyG object at graph-build time to give the GATv2 a
  definitive end-of-sequence target.

Six **edge classes** encode the relationships:

| Class | Name | Example |
| --- | --- | --- |
| 0 | no edge | (the overwhelming majority of candidate pairs) |
| 1 | structural | staff → note, note → notehead, beam → note |
| 2 | modifier | note → accidental, note → dots, note → slur |
| 3 | temporal | note → next event in the same voice |
| 4 | sync-text | syllable/verse → the note it is sung on |
| 5 | simultaneity | events that share an onset across staves |

The **kern serializer consumes exactly these semantics**: class-1 descent
collects events, class-3 topological order forms the timeline, class-2
children decorate tokens, and class-5 groups become simultaneous spine rows.
Whatever the GNN misses, the serializer degrades to spatial fallbacks — so
edge F1 translates quite directly into kern quality.

## 2. What the model sees

The GNN never touches raw pixels itself. Per page:

1. The **frozen detector** (ConvNeXt + PANet) produces multi-scale feature
   maps for the letterboxed 640×640 page — the same resolution and
   letterbox convention it was trained with (this matters; see lesson 1).
2. The page is **split into systems**; the GNN learns per system, not per
   page. Losses from all systems on a page flow through the one shared
   feature map, so a page's backbone gradient is naturally weighted by its
   system count.
3. Per node: a 7×7 **RoI-aligned** visual feature (256-d) + its box
   coordinates **normalized to [0,1]** (4-d) + a learned class embedding
   (32-d) → 292-d node feature.
4. **Candidate edges** come from geometric heuristics (vertical column
   alignment, staff-lane horizontal raycasts, a kNN safety net). The GNN
   classifies candidates; anything the heuristics never propose can never
   be recalled — heuristic recall is the model's ceiling.
5. A 3-layer **GATv2** contextualizes nodes; a concat-pair MLP classifies
   each candidate into the six classes, trained with class-weighted focal
   loss.

## 3. The training curriculum

Training runs in four phases, each releasing one constraint at a time so a
failure is attributable:

| Phase | Vision | Boxes | Knobs (`conf/train_gnn.yaml`) |
| --- | --- | --- | --- |
| 1 | frozen | ground truth | `unfreeze: "none"` |
| 2 | PANet neck trains (adaptation layer) | ground truth | `unfreeze: "neck"`, `vision_lr`, `fine_tune: <phase-1 run>` |
| 3 | full backbone trains | ground truth | `unfreeze: "full"` — only if phase 2 plateaus |
| 4 | best of 2/3 | GT + noise | `box_jitter: <px>` simulates detector localization error |

Rationale: the neck is the cheapest place for the vision features to adapt
to the relational task, and keeping ConvNeXt frozen protects the detector —
the same backbone serves detection at inference, so **after any unfrozen
phase re-run `eval-model` and check detector mAP for drift**. Phase 4
narrows the gap to inference, where boxes come from the detector instead of
annotations; true predicted-box training would additionally require
matching GT nodes onto detections to re-derive edge labels — measure the
gap before building that machinery.

Each phase warm-starts from the previous run via `fine_tune:` (accepts an
HF checkpoint dir). Unfrozen vision params train in their own optimizer
param group at `vision_lr` (~1e-5), far below the GNN head's LR.

## 4. How to tell it is learning (and the two failure attractors)

Watch three things every eval (`eval_steps` in the yaml):

- **Per-class F1** (`f1_structural` … `f1_simultaneity`). Structural moves
  first; modifier is historically the laggard.
- **Topology deltas** (`δ_components`, `δ_triangles`, `δ_density`) — they
  compare the *predicted* active graph against GT structure. If they sit
  **frozen at identical values across evals**, the model is not changing
  its predictions: it has collapsed into an attractor.
- **The prediction histogram** (run a checkpoint over a few pages and
  `argmax` the logits). Cheap, and it distinguishes the two degenerate
  states that per-class F1 alone can hide:
  - *All-background*: predicts class 0 everywhere. Normal for the first
    epochs; a problem only if it persists past LR warmup.
  - *Rare-class flood*: never predicts class 0 at all (we hit this — 92% of
    candidates got class 5). Caused by too-aggressive class weighting; see
    lesson 5.

Healthy runs here looked like: eval loss halves in the first epoch,
`f1_structural` clears 0.3 by epoch ~1.5, macro-F1 crosses 0.5 around
epoch 6 and 0.8 by epoch ~20, and δ_components falls monotonically
(190 → 2.6 over phase 1).

## 5. Lessons learned (each of these was a real blocker)

1. **Feed the detector the resolution it was trained on.** The dataset must
   letterbox pages to the detector's `input_size` with the *same*
   aspect-preserving grey-padding convention as vision training. Full-res
   pages cost 15× the compute *and* put every RoI feature out of
   distribution. (Fixed in `OMRFullPageDataset`.)
2. **No Python `.item()` loops on CUDA tensors in the step path.** Two of
   them (`map_gt_to_candidates`, `split_into_systems`) each caused
   thousands of GPU syncs per step; together with (1) the fixes took the
   step rate from 2.8 s/it to ~20 it/s. Vectorize with `searchsorted` key
   matching and lookup-table remaps.
3. **Remap edges when nodes are filtered.** Graph nodes without an
   annotation bbox are dropped when building the box tensor; edge indices
   must be remapped onto the filtered ordering or every edge after a
   dropped node silently points at the wrong box. This corrupted targets
   invisibly — the old dict-based code neither crashed nor warned.
4. **Normalize heterogeneous node features.** Raw pixel coordinates
   (0–640) concatenated with unit-scale RoI features stall GATv2 training
   completely — three epochs of near-zero F1, fixed the moment coords were
   divided by the canvas size. If inputs mix scales, either normalize them
   or put a LayerNorm at the model's front door.
5. **Focal loss has two sharp edges.** (a) Compute the modulating factor
   from the *unweighted* CE: `pt = exp(-α·CE)` makes low-α classes look
   "easy" even when wrong, suppressing their gradient twice. (b) Do not
   set the background alpha too low — 0.05 vs ~1.5 for rare classes made
   flooding rare classes loss-optimal (the class-5 flood above). We run
   `[0.4, 1.0, 2.0, 1.0, 3.0, 1.5]`.
6. **Custom wrappers need a custom `prediction_step`.** `GNNPhase2Model`
   has no `forward()`; HF's default eval path calls `model(**inputs)` and
   dies — invisible until `eval_strategy` is actually enabled (it defaults
   to `"no"`!). The override routes eval through `compute_loss` and hands
   `compute_metrics` the edge logits/targets pair.
7. **Operational**: on flaky consumer hardware, keep `save_steps` well
   inside the crash cadence; the `--resume` path must discard partial
   checkpoints (no `trainer_state.json`) or a crash during a save
   crash-loops the supervisor; and never `pkill -f` with a pattern your own
   command line contains — kill by PID.

## 6. Phase 1 reference results

30 epochs, frozen detector (`checkpoints/run_003`), GT boxes,
`train-small` (2500 scores / 8783 pages), ~2.2 h wall-clock on an RTX 5090:

| Metric | Final |
| --- | --- |
| f1_structural | 0.892 |
| f1_sync_text | 0.855 |
| f1_simultaneity | 0.795 |
| f1_temporal | 0.769 |
| f1_modifier | 0.646 |
| **f1_macro** | **0.825** |
| δ_components | 2.6 |

Modifier edges are the weak class — and the serializer needs them for
accidentals/dots/ornaments, so they are the first thing to check in later
phases.
