# The score graph: design, training, and lessons learned

This document explains *how the graph half of the pipeline works and how we
train it*: the concepts, the curriculum up to the joint model, how detections
become `**kern`, how we measure it, and the lessons that got each part
working. Its companion for the detector is [vision.md](vision.md). For a
file-by-file reference see [modules/graph.md](modules/graph.md); for the
commands see [end-to-end-training.md](end-to-end-training.md).

## 1. The graph design

A page of music is modeled as a typed graph over the detector's primitives.
Node roles (`conf-lite/structure.json`; `conf/` for the full track) shape what
may connect to what:

- **Temporal anchors** (notes, chords, rests) string together left-to-right
  into the chronological Humdrum backbone via *temporal* edges.
- **Structural components & modifiers** (stems, beams, accidentals, dots,
  ornaments) only ever connect to temporal anchors, via *structural* or
  *modifier* edges. A stem never connects directly to a clef.
- **Context globals** (clefs, key/meter signatures, dynamics, pedals) dictate
  staff state; they attach to measures or staves, not to individual notes.
- **Structure** (system → measure → staff cell → layer) is itself
  structural edges; it carries the events.

Six **edge classes** encode the relationships:

| Class | Name | Example |
| --- | --- | --- |
| 0 | no edge | (the overwhelming majority of candidate pairs) |
| 1 | structural | staff → layer → note, note → notehead, beam → note |
| 2 | modifier | note → accidental, note → dots, note → slur |
| 3 | temporal | note → next event in the same voice |
| 4 | sync-text | syllable/verse → the note it is sung on (none in lite) |
| 5 | simultaneity | events that share an onset across staves |

The **kern serializer** (`MinimalHumdrumSerializer`, in musicset) consumes
exactly these semantics:
- class-1 descent collects each staff cell's events;
- class-3 topological order forms the timeline;
- class-2 children decorate the tokens;
- class-5 groups become simultaneous spine rows.

Whatever the GNN misses, the serializer falls back on geometry, so edge F1
translates directly into kern quality.

## 2. The model

The GNN never touches raw pixels. It reads the detector's features at the
boxes it is given, ground truth or detected, and classifies the candidate
edges between them (`graph/model.py`):

```
page ─▶ detector backbone + PANet neck ─▶ 3 feature maps (strides 4/8/16, 128 ch)
                                                │
boxes of one system ─▶ MultiScaleRoIAlign 7×7 ──┘─▶ Linear 6272→256, ELU ─┐
                       box relative to the system, ÷ canvas width (4) ────┤
                       class embedding (99 → 32) ─────────────────────────┼─▶ node x (294)
                       detector score, synthesized by bbox repair (2) ────┘
node x + candidate edges ─▶ GATv2 ×3 (4 heads × 64 = 256) ─▶ node h (256)
candidate (u, v) ─▶ [h_u ‖ h_v] (512) ─▶ MLP 512→256→128→6 ─▶ edge class logits
node h ─▶ MLP 256→128→1 ─▶ keep logit (does the box belong in the score?)
```

| part | module | parameters |
| --- | --- | --- |
| detector (backbone, neck, heads) | `MusicDetector` ([vision.md](vision.md) §2) | 4.98M |
| RoI projection | `GraphVisualExtractor` | 1.61M |
| class embedding + 3 GATv2 layers + edge MLP + keep head | `ScoreGraphReconstructor` | 0.62M |

Checkpoints from before 2026-10-22 have a 292-d node input and no keep
head; `_load_gnn` reads both from the weights.

The model in detail:

1. **The detector's features.** The detector runs once per page, which is
   letterboxed into *its* canvas. That canvas is stored in its checkpoint:
   640×896 for the current detector, square 640 before October. It must be
   the canvas the detector was trained on (lesson 1). The GNN reads the
   three fused PANet maps, the ones the detection heads read.
2. **Systems.** The page is split into systems, and each system is one
   graph with its own forward pass. Losses from all systems on a page flow
   through the one shared feature map, so a page's backbone gradient is
   naturally weighted by its system count.
3. **The node feature** (294-d) has four parts:
   - **What the box looks like (256-d).** `MultiScaleRoIAlign` pools a
     7×7 grid from the level the box's size selects. That's the FPN rule
     on the canvas: a box whose √area is under 112 px reads stride 4, so
     nearly every glyph does; under 224 px reads stride 8; larger boxes
     (measures, rows, systems) read stride 16. A linear layer with ELU projects the
     6,272 values to 256.
   - **Where it is (4-d).** The box relative to its system, divided by the
     canvas width. Both axes are divided by the width, which keeps the
     aspect.
   - **What it is (32-d).** A learned embedding of its detector class.
     Classes are GNN *inputs*, which is why a class change needs phase 5
     (§3).
   - **How sure the detector was (2-d).** Its score, and whether bbox
     repair synthesized the box (score 0). Ground-truth boxes are score 1,
     not synthesized. Their input weights start at zero and stay frozen
     there through the ground-truth stages, where the values are constant;
     they train from stage D on (§3).
4. **Candidate edges** come from geometric heuristics
   (`heuristics.generate_axis_aware_edges`):
   - **Vertical:** pairs whose centres are within 1.5× their mean width
     horizontally, for structural and modifier links. In a pair with a
     *span* box (system, row, measure, staff cell, layer, beam, tuplet), the
     narrower box's centre must lie within the wider box's x-span, padded
     by a quarter of its height and at most 10 px. Clefs, key and time
     signatures and barlines are exempt.
   - **Horizontal:** a raycast to the 3 nearest boxes to the right in the
     same staff lane, in both directions, for the temporal flow, plus the
     same ray over the temporal anchors alone (notes, chords, rests). Lanes
     are the system's staff rows, found by clustering the staff cells'
     centres. Containers are neither sources nor targets of the ray.
   - **Safety net:** the 3 nearest neighbours, under the same span rule.

   The heuristics are ranking-based, so they don't depend on the canvas
   scale. The GNN only classifies candidates, so anything the heuristics
   never propose can never be recalled: heuristic recall is the model's
   ceiling (lesson 12).
5. **Message passing.** Three `GATv2Conv` layers (4 heads of 64,
   concatenated, plus self-loops) run over the candidate edges themselves,
   with ELU and dropout 0.2 between layers. Attention lets each node weigh
   its candidate neighbours instead of averaging them. Most candidates
   exist in both directions, so context flows both ways.
6. **Edge classification.** Each candidate (u, v) is classified from the
   *ordered* concatenation of its two node states. Ground-truth edges are
   directed (parent → child, earlier → later), so (v, u) is a separate
   candidate whose label is usually "no edge": the order is how the MLP
   learns direction. Training uses class-weighted focal loss over all
   candidates (lesson 5). Inference takes the argmax per candidate and keeps
   the non-zero classes.
7. **The keep head.** A small MLP on each node state gives the logit that
   the box belongs in the score. It is trained only where it has targets:
   detected boxes matched to the ground truth (stage D). At inference,
   `--prune-threshold` drops the events, measures and barlines it scores
   below the threshold, before graph repair can reattach them.

**Wrappers.**
- `GNNPhase2Model` holds the detector, the RoI extractor and the GNN.
  `unfreeze` (`none` / `neck` / `full`) decides which detector parts train;
  the detector stays in eval mode either way.
- `JointModel` (`graph/train_joint.py`) also trains the detection heads and
  adds the detection loss, so one checkpoint serves both halves (§3).

## 3. The training curriculum

Three runs, each releasing one constraint, chained unattended by
`pipeline-gnn.sh` on a trained detector:

| Stage | Vision | Boxes | Config (`conf-lite/`) |
| --- | --- | --- | --- |
| phase 1 | frozen | ground truth | `train_gnn_phase1.yaml` (`unfreeze: none`, lr 1e-3, 30 epochs) |
| phase 2 | PANet neck adapts | ground truth | `train_gnn_phase2.yaml` (`unfreeze: neck`, lr 3e-4, 15 epochs) |
| joint | full backbone + detection loss | GT + 1.5 px jitter | `train_joint_full.yaml` (15 epochs) |
| stage D | as joint | 70% the model's own detections, 30% GT + jitter | `train_stage_d.yaml` (8 epochs) |

- **Phases 1–2.** The neck is the cheapest place for the vision features to
  adapt to the relational task. Each stage warm-starts the whole wrapper
  from the previous run (`fine_tune:`). Unfrozen vision parameters train in
  their own optimizer group at `vision_lr` (1e-5).
- **The joint model** (`train-joint`, roadmap stage C) trains *one* network
  with two losses, so the same checkpoint serves detection and edges at
  inference:

  ```
  L = 10 · mean over systems(L_edges) + 0.5 · (L_cls + 5 · L_reg)(page)
  ```

  - The detection loss anchors the shared backbone and neck, which makes
    full unfreezing safe.
  - Three learning-rate groups: GNN 3e-4, detection heads 3e-5,
    backbone/neck 1e-5.
  - The box jitter feeds only the GNN branch; detection targets stay clean.
  - Detection is supervised on the *full* annotation set (lesson 9).
  - With ReGLA the joint detector *improves* slightly (mAP 0.800 vs 0.794
    standalone) and keeps stems and barlines at 0.99. Every July ConvNeXt
    joint run collapsed them.
- **Stage D** (roadmap stage D) trains the joint model on the boxes it meets
  at inference:
  - `cache-detections` stores the joint detector's boxes, after bbox
    repair, for every training page.
  - Each page then trains on them with probability `det_prob` (0.7).
  - The detections are matched to the page's ground-truth nodes per class
    (`utils.match_detections`): Hungarian on IoU, at least 0.5 for
    structure and 0.3 for glyphs.
    - Variants of one object family (beams, stems, noteheads, rests,
      barlines, flags, accidentals, key accidentals, time digits, clefs,
      dynamics) match as one group: a wrong variant is still the object.
    - A same-class detection always wins: multi-label decoding often emits
      two variants on one object (rest8th and rest16th on one rest), and the
      wrong one is the duplicate to drop.
  - A matched box takes its ground-truth node's edges and keep target 1.
    An unmatched one (a false positive, or a duplicate such as one beam
    decoded as beam8 and beam16) gets no edges, so its candidates train
    toward "no edge", and keep target 0. About 87% of the ground-truth
    edges survive the transfer.
  - The keep head's BCE (`node_weight` 1.0) joins the edge and detection
    losses. Ground-truth pages carry no keep targets.
    - False positives are ~3% of the prunable detections, so keep-0 targets
      weigh 5× and the prunable classes (events, measures, barlines) 2×.
      Otherwise the head would settle on "keep" everywhere.
  - Validation runs on detections only (`det_prob` 1), so the edge and keep
    metrics compare across evaluations.

**In reserve (full track, 0.1.x):**
- phase 3, full backbone without the detection loss: superseded by the
  joint model;
- phase 4, GT-box jitter without joint training;
- phase 5, class expansion after a detector change. Classes are GNN
  *inputs*, so `old_num_classes:` loads only the `gnn.*` weights, widens the
  embedding, and seeds new rows from a look-alike (`new_class_templates:
  {clefG8va: clefG}`).

## 4. How to tell it is learning (and the two failure attractors)

Watch these every eval (`eval_steps` in the yaml):

- **Per-class F1** (`f1_structural` … `f1_simultaneity`). Structural moves
  first; modifier is historically the laggard. In lite, sync-text reads 0
  by construction.
- **Topology deltas** (`δ_components`, `δ_triangles`, `δ_density`). They
  compare the *predicted* active graph against GT structure. If they sit
  **frozen at identical values across evals**, the model is not changing
  its predictions: it has collapsed into an attractor.
- **What the serializer will see** (`metrics.GraphTopologyEvaluator`):
  - `event_reach_recall`: the events a staff cell reaches through the
    predicted structural edges, out of those it reaches in the ground
    truth.
  - `multi_staff_event_rate`: events reached from two or more staff cells
    (other staff or other measure: the duplicate sources of §6).
  - `multi_owner_rate`: layers with several staff parents, beams and
    tuplets with several layers, events with several containers.
  - `temporal_branch_rate`: events with more than one temporal successor
    or predecessor.
- **The keep head** (stage D): `prune_fp_precision` / `_recall` / `_f1` on
  the prunable classes, and `keep_node_acc`.
- **The prediction histogram.** Run a checkpoint over a few pages and
  `argmax` the logits. It's cheap, and it distinguishes the two degenerate
  states that per-class F1 alone can hide:
  - *All-background*: predicts class 0 everywhere. Normal for the first
    epochs; a problem only if it persists past LR warmup.
  - *Rare-class flood*: never predicts class 0 at all (we hit this: 92% of
    candidates got class 5). It comes from too-aggressive class weighting;
    see lesson 5.

Healthy runs look like this:
- eval loss halves in the first epoch;
- `f1_structural` clears 0.3 by epoch ~1.5;
- macro-F1 crosses 0.5 around epoch 6 and 0.8 by epoch ~20;
- δ_components falls monotonically.

## 5. Lessons learned (each of these was a real blocker)

1. **Feed the detector the canvas it was trained on.** The dataset
   letterboxes pages into the detector's own canvas
   (`checkpoint_input_size(detector_checkpoint)`), with the same
   aspect-preserving grey-padding convention as vision training. Full-res
   pages cost 15× the compute *and* put every RoI feature out of
   distribution.
2. **No Python `.item()` loops on CUDA tensors in the step path.** Two of
   them (`map_gt_to_candidates`, `split_into_systems`) each caused
   thousands of GPU syncs per step. Together with (1), fixing them took the
   step rate from 2.8 s/it to ~20 it/s. Vectorize with `searchsorted` key
   matching and lookup-table remaps.
3. **Remap edges when nodes are filtered.** Graph nodes without an
   annotation bbox are dropped when building the box tensor, so edge indices
   must be remapped onto the filtered ordering. Otherwise every edge after a
   dropped node silently points at the wrong box.
4. **Normalize heterogeneous node features.** Raw pixel coordinates
   concatenated with unit-scale RoI features stalled GATv2 training
   completely (three epochs of near-zero F1). It was fixed the moment the
   coordinates were divided by the canvas size.
5. **Focal loss has two sharp edges.**
   - (a) Compute the modulating factor from the *unweighted* CE:
     `pt = exp(-α·CE)` makes low-α classes look "easy" even when wrong.
   - (b) Don't set the background alpha too low. 0.05 against ~1.5 for
     rare classes made flooding the rare classes loss-optimal.

   We run `[0.4, 1.0, 2.0, 1.0, 3.0, 1.5]`.
6. **Custom wrappers need a custom `prediction_step`.** `GNNPhase2Model` has
   no `forward()`, and HF's default eval path dies on it. That's invisible
   until `eval_strategy` is actually enabled (it defaults to `"no"`).
7. **Operational.**
   - Keep `save_steps` well inside the crash cadence.
   - `--resume` must discard partial checkpoints; the trainers check
     `trainer_state.json`, and the joint trainer also checks that the
     safetensors header loads.
   - Never `pkill -f` a pattern your own command line contains.
8. **Jitter is a GNN-side augmentation only.** Jittering the boxes before
   *both* branches of the joint model destroyed detection localization
   (mAP@.75 0.58 → 0.21).
9. **When two tasks share supervision, check both label spaces.** The GNN
   dataset only yields graph-node boxes, so classes that never become nodes
   (note anchors, barlines) were absent from the joint detection loss. That
   actively unlearned them (note AP 0.93 → 0.01) until the detection targets
   used the full annotation set. Non-graph annotations now also join the
   GNN as isolated nodes, so both losses share one node space.
10. **Log what you optimize.** `GNNTrainer` had no `log()` override, so the
    joint trainer's per-loss metrics were silently dropped for five runs.
11. **Resolve node ids the way the graph builder does.** Annotations that
    share an id (a note and its notehead, a single-staff system and its row)
    give the first one the id and every later one a pseudo-id
    `{id}_{class}_{n}`. The validation tools resolved them; until 2026-10-22
    the GNN *training* dataset mapped raw ids, where the last annotation
    wins.
    - Every note's graph node carried its notehead's box and class.
    - The notehead node and the rows were dropped.
    - The GNN never saw a `note` node in training, while inference has
      both notes and noteheads.

    `dataset.resolved_page_annotations` now reproduces the builder's ids.
12. **Measure candidate recall, not only edge F1.** Edge F1 is measured on
    candidates, so edges the heuristics never propose are invisible to it.
    Two flaws hid behind a temporal F1 of 0.876, while only 74.7% of the
    ground-truth temporal edges were candidates at all (68.5% on
    detections):
    - **Per-cell lanes.** A lane per staff cell split each row into several
      wherever cell centres differed by a pixel: 74% of the detected rows.
    - **Containers on the ray.** Staff, measure and layer centres sit
      mid-measure on the staff line and took the ray's slots.

    The fixes raised temporal candidate recall to 91.9% / 82.9%, with
    11–14% fewer candidates. Once note anchors became real training nodes
    (lesson 11), each note's own notehead, stem and beam took the slots
    again (87.5% / 81.5%). A second ray over the temporal anchors alone
    gives 99.8% / 99.9%, for 5% more candidates.
13. **Bound what a wide box can reach.** A layer is a measure wide, so the
    1.5× vertical window gave it candidates among the neighbouring
    measures' beams and notes. The GNN linked enough of them that 18% of
    the detection path's output notes were the same note emitted in a
    second measure. Span boxes now pair only with what lies under them,
    which halves the cross-measure container candidates; the serializer
    rule below catches the rest.

## 6. From detections to `**kern`

At inference (`infer-omr`, and `validate-detections` on a labelled set),
the boxes come from the detector:

1. **Detect.** Detection is multi-label, with the temporal anchors argmax-only
   (see [vision.md](vision.md) §3).
2. **Repair the boxes** (`graph/bbox_repair.py`; each repair numbered and
   toggleable, listed in the order they run). These recover the structure
   the serializer needs:
   - 6: structure false-positive filtering: fragment rows, band-NMS on
     systems and rows (they never overlap vertically), and duplicate
     measures and cells.
   - **8: duplicate anchors.** A note inside a note, or a chord inside a
     chord, is the detector firing twice. A chord's member notes, inside it,
     are legitimate.
   - **7: row geometry from the clefs.** Each detected row's vertical extent
     is set from the leftmost fixed-line clef on it. The detector places
     rows a median 3.6 pitch steps off; the clef-derived row is within half
     a step 96% of the time, and every system restates its clefs.
   - 5: a row from a clef that has none.
   - 1: systems from uncovered bands.
   - 4: measures from barline x-clusters.
   - 2: measures from stacks of staff cells.
   - 3: a staff cell at each measure × row.
3. **Predict edges** with the GNN. With `--prune-threshold`, the events,
   measures and barlines the keep head scores below the threshold are
   dropped, with their edges, before graph repair could reattach them.
   Then inject the system → measure edges (system nodes are outside the
   per-system groups).
4. **Repair the edges** (`graph/graph_repair.py`): guarded spatial
   heuristics restore the geometric certainties the GNN loses. Examples are
   note → notehead → stem chains, meterSig digits, keySig accidentals, and
   orphan subtrees under their containing staff. A repair never overrides a
   model prediction; `--no-repair` gives the raw-GNN baseline.
5. **Serialize.** Four rules make the serializer robust to predicted
   graphs; ground-truth graphs serialize byte-identically with or without
   them:
   - **The reference measure.** Spines are created from the reference
     system's leftmost measure that has a staff cell on every row. A first
     measure missing its measure → staff edges used to give no spines (the
     whole score failed with "No valid systems") or too few (a staff
     dropped).
   - **Events reached by several staff cells.** On grand staves ~75% of
     anchors are reachable from *both* staff cells through predicted edges.
     Each cell emitted them, reading the pitch against its own row: a
     duplicate about two octaves off. Such an event now stays with the cell
     whose band holds its noteheads, where bands are split at the midpoint
     between rows.
     - An event only one cell reaches stays with that cell. Cross-staff
       notes are drawn on the other staff but belong to their layer's
       staff; moving them cost 1.5 points of recall.
   - **Signature glyphs by band.** A clef, key signature, key accidental,
     meter or clef change counts for a staff only if it's drawn in that
     staff's band. Otherwise a grand staff's bass clef, linked to the treble
     cell, became a clef change of the treble staff.
   - **Events reached from several measures.** A predicted edge across a
     barline (a beam, note or layer under two measures) made both measures
     emit its subtree: 11.8k repeated notes on validation-lite, 18% of the
     output. Such an event is now emitted once, by the reaching measure
     whose box holds it (the narrowest, if several do), or else the nearest
     one (`_measure_owners`, per page).
     - On the joint model, without retraining: notes out 1.15× → 0.94×,
       precision 71.7 → 85.5, F1 76.6 → 83.0, recall 82.3 → 80.6.
     - Most of the recall lost comes from scores with broken structure
       (e.g. one detected system for a whole page), where a duplicate had
       landed in the measure the comparison expected.

## 7. Measuring it

Three paths, from the same validation set, separate the stages' errors:

| path | command | measures |
| --- | --- | --- |
| ceiling | `validate-groundtruth` | the serializer alone, on ground-truth graphs |
| GT boxes | `validate-predictions` | the GNN + edge repairs, on ground-truth boxes |
| detections | `validate-detections` | the whole pipeline (detector, box repairs, GNN, repairs) |

**`compare-kern`** scores each against the MEI:
- **Recall**, with pitch and pitch + duration matched as multisets per
  (staff, measure).
- **Precision and F1**, plus the output's note count relative to the
  ground truth. Extra notes only show here: until October every number
  quoted was recall, which is how duplicated notes went unnoticed.
- **The median file**, and the scores that produced no kern (excluded from
  the totals, not scored 0).

The comparison code lives in musicset (`musicset.evaluate`).

**Finding the bottleneck: oracle substitution.** Swap ground-truth boxes in
for one part of the detection path at a time, keeping the same GNN, repairs
and serializer, and see which swap recovers the most:
- Before repair 7, ground-truth *rows* alone took the detection path from
  53.4 to 84.0 (recall). Ground-truth systems or measures added at most 3.
- With repair 7, measure segmentation became the next lever: ground-truth
  measures give 89.0. Miscounted measures (85 of 500 scores, mostly one
  extra) shift every comparison after them, while those scores are 97.9%
  right per staff.

**Looking at it:**
- `musicset view <set> --labels lite --pred-kern <dir>`, Kern tab: the
  source pages, the ground-truth kern and the predicted kern engraved side
  by side, worst first on request.
- `scripts/qa_views.py`: structure overlays and engravings for real scans
  without ground truth.

## 8. Reference results

LITE track, validation-lite (500 scores, 57,398 notes). Detector
`checkpoints_lite_896/run_003`, joint model `checkpoints_gnn_lite_896/run_003`,
the repairs and serializer of 2026-10-21. These models predate lessons
11–13, the regression slots and stage D. The queue retraining with all of
them is `checkpoints_lite_896s` → `checkpoints_gnn_lite_896s`.

| path | pitch recall | rhythm recall | pitch precision | pitch F1 | median file (pitch) |
| --- | --- | --- | --- | --- | --- |
| ceiling (GT graphs) | 99.8 | 99.6 | 99.8 | 99.8 | 100.0 |
| GT boxes + predicted edges | 97.2 | 93.2 | 90.4 | 93.7 | 99.1 |
| detections | 82.3 | 77.5 | 71.7 | 76.6 | 92.4 |

Joint-model edges on held-out pages: F1 macro 0.689 (structural 0.938,
modifier 0.809, temporal 0.876, simultaneity 0.518).

Training cost on a DGX Spark:

| stage | time |
| --- | --- |
| phase 1 | 7.0 h |
| phase 2 | 3.8 h |
| joint | 9.9 h |

Roadmap stage D's gate is a detection-path median ≥ 80%. It is met on
recall: 92.4%. Precision is the open front: 1.15× the notes come out, from
the extra anchors that remain after the fixes above.

The 0.1.x full-track reference (validation-small, GT boxes):
- 93.2 / 80.4 against a ceiling of 98.4 / 97.7;
- detection-driven 29.0 / 23.3, all recall.
