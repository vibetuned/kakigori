# graph/ — module map

## Pipeline overview

This subpackage is the "relational" half of the OMR pipeline. The vision stack (`kakigori.vision`) detects musical primitives as bounding boxes on full pages; `graph/` then takes those detections (plus, at training time, MEI annotations) and learns the typed edges that turn a bag of glyphs into a structured score, which is finally serialized to Humdrum `**kern`.

The flow is roughly:

1. `parsers.GroundTruthGraphBuilder` reads an MEI file plus per-page JSON detections and emits ground-truth typed edges (6 classes: 0=no edge, 1=structural, 2=modifier, 3=temporal, 4=sync-text (syl/verse), 5=simultaneity). These are saved as `.pt` files (one per score) containing `edge_index`, `y`, `node_ids`, `json_files`.
2. `dataset.OMRFullPageDataset` pairs images + JSON annotations + saved graphs into PyG-compatible samples.
3. `model.ScoreGraphReconstructor` (GATv2) and `model.GraphVisualExtractor` (MultiScaleRoIAlign over the detector's PANet features) form the GNN. `model.GNNPhase2Model` wraps them with a frozen detector.
4. `train.py` (CLI) trains the GNN via `trainer.GNNTrainer` using `losses.MultiClassEdgeFocalLoss`, `heuristics.generate_axis_aware_edges` for candidate proposal, and `metrics.GraphTopologyEvaluator` plus `compute_gnn_metrics` for evaluation.
5. `infer.py` (`infer-omr`) runs the full pipeline on PDFs or page images: detector → RoI → GNN → predicted edges → `graph_repair` → `serializers.MinimalHumdrumSerializer` → `**kern`.
6. `validate_groundtruth.py` (CLI) is the offline path that re-uses `MinimalHumdrumSerializer` directly on GT graphs to sanity-check the serializer on perfect data.
7. `eval.py` scores predicted `.krn` against ground-truth `.krn` with SER/CER/LER.

## Modules

### `__init__.py`
**Role:** Empty package marker; no public re-exports.
**Preconditions:** None.
**Produces / used by:** Makes `kakigori.graph` importable.

### `parsers.py`
**Role:** Build the typed ground-truth edge set for a score by combining an MEI tree with per-page JSON detections. Walks the MEI XML hierarchy to assign edge types (1 structural / 2 modifier / 3 temporal / 4 sync-text / 5 simultaneity) and adds spatial fallbacks for relationships MEI does not express in the XML tree (system→measure, staff→clef/keySig, note→dots, note→stem with unison tie-breaking). Spatial fallbacks skip extractor-only annotations that carry no `id` (system-staff boxes, page furniture) — they cannot be graph nodes.
**Key classes/functions:** `GroundTruthGraphBuilder.__init__` (loads MEI + JSON pages, handles ID collisions by minting `pseudo_id`s), `build_edges` (returns the list of `(u_id, v_id, edge_class)` tuples), `get_pyg_labels` (maps a candidate `edge_index` tensor to a `y` label tensor using the GT dict).
**Preconditions:** An MEI file with `xml:id`s matching JSON `id` fields; per-page JSON with `annotations[].{id, class, bbox}`; a `node_roles` config grouping classes into `temporal_anchors`, `modifiers`, `synchronization_text`, `context_globals`.
**Produces / used by:** Driven by `dataset/generate_graphs.py` to build the per-score `.pt` graphs (which then feed `dataset.OMRFullPageDataset`).

### `dataset.py`
**Role:** PyTorch `Dataset` that aligns a page image, its JSON detections, and the pre-built ground-truth graph (`.pt`). Letterboxes every page to the detector's training resolution (`input_size`, default 640 — aspect-preserving, grey padding, same convention as `vision/dataset._letterbox`) and remaps boxes into canvas coordinates; feeding full-resolution pages would put the frozen detector's features out of distribution.
**Key classes/functions:** `OMRFullPageDataset.__init__` matches `(img, json, graph)` triplets by stem (tolerates `_pageN` suffix); `__getitem__` returns `{image, boxes, labels, edges (E,3), file_name}` with boxes/labels strictly ordered to match `node_ids` from the graph, and **edge indices remapped onto the filtered box ordering** — graph nodes without an annotation bbox are dropped, so raw graph indices would silently pair edges with the wrong boxes.
**Preconditions:** Three parallel directories: `img_dir/*.png`, `json_dir/*.json`, `graph_dir/*.pt`. `.pt` must contain `edge_index`, `y`, `node_ids`. `class_list` (from `conf/config.json["target_classes"]`) drives label encoding.
**Produces / used by:** Used by `train.py` together with `utils.omr_collate_fn`.

### `eval.py`
**Role:** Token/character/system-level error rate comparison between predicted and ground-truth `**kern` strings, using `jiwer`.
**Key classes/functions:** `evaluate_humdrum_output(predicted_kern_list, ground_truth_kern_list)` returns `(ser, cer, ler)`.
**Preconditions:** Two parallel lists of `**kern` strings (one per system or per file).
**Produces / used by:** Prints metrics, returns the triple. Not wired into the trainer; intended as an end-to-end OMR quality check.

### `heuristics.py`
**Role:** Candidate-edge generation and GT-to-candidate label mapping. The GNN classifies *candidate* edges rather than the full `N²` pairs, so these heuristics are what gates recall vs. compute.
**Key classes/functions:**
- `generate_axis_aware_edges(boxes, labels, class_to_idx, k_neighbors=3)` — three-phase: vertical (column-aligned modifier candidates), horizontal (staff-lane-aware right-ward raycasts for temporal flow), and a kNN safety net.
- `generate_text_candidate_edges` — vertical-raycast lyric→notehead pairing.
- `map_gt_to_candidates(edge_index, gt_targets)` — assigns class 0–5 to each candidate; fully vectorized ((u,v) pairs encoded as scalar keys + `searchsorted`) because the previous per-candidate `.item()` loop caused thousands of GPU syncs per training step.
- `pure_pytorch_knn_graph` — torch-only replacement for `torch_geometric.nn.knn_graph` to avoid the optional C extension.
**Preconditions:** `boxes` in `[x1,y1,x2,y2]`, `labels` as int class indices, `class_to_idx` containing at least `"staff"` (and `"Lyric"`/`"Notehead"` for the text variant).
**Produces / used by:** Called by `trainer.GNNTrainer.compute_loss` and by `infer.py`.

### `infer.py`
**Role:** End-to-end inference CLI (`infer-omr`): PDF or page images → detector → GNN edges → graph repair → `**kern`. Rewritten for v0.1.0 on the `validate_predictions` mechanics — the earlier prototype inferencer classes it contained were dead code.
**Key classes/functions:**
- `detect_page_nodes(detector, page_image, ...)` — runs the detector on a letterboxed page and returns serializer-format nodes (`id`/`class`/`bbox`/`cx`/`cy`) in original page coordinates.
- `main()` — loads the detection checkpoint and the GNN wrapper separately (two-model deployment: the wrapper's adapted neck only feeds RoI features), then per page: detect → `predict_page_edges` → `_inject_system_measure_edges` → `repair_page_edges`, and finally one `MinimalHumdrumSerializer` pass over all pages.
**Preconditions:** detector checkpoint dir, GNN wrapper run dir, `conf/config.json`, `conf/structure.json`; `pypdfium2` for `--pdf` input.
**Produces / used by:** Writes `page_NNNN.png`/`page_NNNN.json` (dataset annotation format, same as `infer-model`) and `<stem>.krn` into `--output-dir`.

### `losses.py`
**Role:** Class-balanced focal loss for the 6-way edge classifier (down-weighting class 0 "No Edge" which dominates ~97% of candidates).
**Key classes/functions:** `EdgeFocalLoss` and `MultiClassEdgeFocalLoss` — CE with `(1-p_t)^gamma` modulator and per-class `alpha` weights; the second registers `alpha` as a buffer for device safety. The modulating factor is derived from the **unweighted** CE: computing `pt = exp(-α·CE)` makes low-α classes look "easy" even when confidently wrong and suppresses their gradient twice (this caused a rare-class flood; see docs/graph.md lesson 5).
**Preconditions:** `alpha_weights` of length `num_edge_classes` (6). Keep `alpha[0]` moderate (currently 0.4): too small and never-predicting-class-0 becomes loss-optimal.
**Produces / used by:** `MultiClassEdgeFocalLoss` is instantiated inside `trainer.GNNTrainer`. `EdgeFocalLoss` is referenced only by the unused training helpers in `train_tools.py`.

### `metrics.py`
**Role:** Macro-topology metrics that compare the GT vs. predicted *active* graph (edges with class > 0) using NetworkX.
**Key classes/functions:** `GraphTopologyEvaluator.update(edge_index, gt_edges, pred_edges, num_nodes)` tracks |Δ connected components|, |Δ triangles|, |Δ density|; `.compute()` averages them; `.reset()` clears state between eval epochs.
**Preconditions:** Per-system tensors of candidate edges and class predictions/targets.
**Produces / used by:** Driven by `trainer.GNNTrainer.evaluate`, results injected into the HF Trainer metrics dict.

### `model.py`
**Role:** Defines the three neural building blocks of Phase 2.
**Key classes/functions:**
- `GraphVisualExtractor` — `MultiScaleRoIAlign` over PANet feature maps (`'0','1','2'`), flatten + Linear → 256-d node visual feature. Expects boxes in normalized `[cx,cy,w,h]`, denormalizes internally.
- `ScoreGraphReconstructor` — 3-layer GATv2 backbone + concat-pair MLP edge classifier producing logits for `num_edge_classes` (6). Owns the `class_embedding` lookup.
- `GNNPhase2Model` — wrapper around detector + RoI extractor + GNN with an `unfreeze` mode implementing the phase curriculum: `"none"` (detector fully frozen), `"neck"` (PANet trains as an adaptation layer, ConvNeXt frozen), `"full"` (backbone + neck train). The detector stays in eval mode in every phase (LayerNorm-based, so eval only disables droppath while unfrozen weights still receive gradients); `train()` returns `self` per the `nn.Module` contract.
**Preconditions:** `node_in_dim = roi_dim(256) + bbox(4) + class_embed_dim(32) = 292`; bbox coords are normalized to [0,1] by the trainer before concat. Detector must already be loaded with weights before wrapping.
**Produces / used by:** Instantiated in `train.py`; the wrapper is what `GNNTrainer` calls.

### `old_serializers.py`
**Role:** Historical sketches of the Humdrum serializer — multiple iterations stacked in one file. Contains a `ContextTracker` (clef/key-sig/measure accidental state machine), a procedural `generate_kern_stream`, free-floating `_collapse_primitives` / `_resolve_semantics` / `_calculate_pitch` helpers, an earlier `HumdrumSerializer` class, then a second `ContextTracker` + `HumdrumSerializer` "First try" pair.
**Key classes/functions:** `ContextTracker` (×2), `HumdrumSerializer` (×2), `generate_kern_stream`, `_collapse_primitives`, `_resolve_semantics`, `_calculate_pitch`.
**Preconditions:** Several functions are defined as free functions but reference `self` (e.g. `_collapse_primitives`, `_resolve_semantics`, `_calculate_pitch`) — they were clearly meant to be methods of some serializer class.
**Produces / used by:** Nothing in the current package imports from this file. Some symbols referenced by `infer.py` (`generate_kern_stream`, `_collapse_primitives`) live here, which suggests `infer.py` was written against this older design. **Flagged as likely dead code** kept for reference.

### `serializers.py`
**Role:** Current Humdrum `**kern` serializer. Walks the predicted (or GT) graph, extracts clef/key/meter from structural children of each staff, collects temporal-anchor events via Class-1 descendants ordered by Class-3 topological sort (with `cx` tie-break), analyzes each event (note/chord/rest/mRest) into a typed info dict, resolves ambiguous durations against the time signature, estimates pitch from `cy` vs. staff bbox + clef, and aligns spines across staves by spatial-cluster `cx`.
**Key classes/functions:** `Measure`, `Spine` (with classmethod `create_from_measure` and helpers `_extract_key_signature`, `_extract_meter_signature`, `_get_system_descendants`, `_parse_meter`), `HumdrumContext` (`_synchronize_measures`, `merge_spines`), `MinimalHumdrumSerializer` (`add_page`, `_collect_staff_events`, `_analyze_note`, `_analyze_chord`, `_resolve_durations`, `_position_to_kern_pitch`, `_info_to_kern`, `export_to_krn`). Module-level constants: `DIATONIC`, `CLEF_BOTTOM_LINE`, `REST_KERN`, `NOTEHEAD_BASE_DURATION`, `FLAG_DURATION`, `STEM_DURATION`, `ACCIDENTAL_KERN`, `EVENT_CLASSES`, `SUB_GLYPH_CLASSES`.
**Preconditions:** A `(edge_index, edge_predictions, pyg_node_ids, node_roles)` tuple plus the per-page list of node dicts (`id`, `class`, `bbox`, `cx`, `cy`). `node_roles` must define `temporal_anchors`.
**Produces / used by:** Used by `validate_groundtruth.py` (CLI) and intended as the runtime serializer for inference. Produces a single tab-separated `**kern` string per multi-page score.

### `train.py`
**Role:** CLI entry point for the GNN training phases (see docs/graph.md §3 for the curriculum).
**Scenario:** "I have a trained detector and a directory of GT graphs; train the GNN on top of the vision stack, progressively unfreezing it."
**Key classes/functions:** `parse_args` (loads `conf/train_gnn.yaml` defaults if present, then argparse — including the phase knobs `--unfreeze {none,neck,full}`, `--vision-lr`, `--box-jitter`, `--eval-steps`), `main` (creates numbered `run_XXX` output dirs, loads `conf/config.json["target_classes"]`, builds `OMRFullPageDataset` with 95/5 split, instantiates `MusicDetector` + `GraphVisualExtractor` + `ScoreGraphReconstructor` wrapped in `GNNPhase2Model(unfreeze=...)`, configures `TrainingArguments` for cosine schedule + fp16 + `eval_strategy="steps"` — the HF default is `"no"`, i.e. no eval at all — builds a param-group AdamW when vision is unfrozen (GNN head at `lr`, vision at `vision_lr`), then runs `GNNTrainer` with `alpha_weights = [0.4, 1.0, 2.0, 1.0, 3.0, 1.5]`). `--fine-tune` warm-starts the whole wrapper from a previous phase's HF checkpoint dir via `vision.utils.load_checkpoint`. The `--resume` path discards partial checkpoints (missing `trainer_state.json`) so a crash during a save cannot crash-loop the supervisor.
**Preconditions:** `--detector-checkpoint` is **required** (raises ValueError otherwise). Needs `--img-dir`, `--ann-dir`, `--graph-dir`, `--config`. Supports `--resume` / `--resume-from-checkpoint` / `--fine-tune`.
**Produces / used by:** Writes HF Trainer checkpoints into `checkpoints_gnn/run_XXX/`, TensorBoard logs into `runs_gnn/run_XXX/`, and a final `gnn_final.pth` (raw GNN state dict).

### `train_tools.py`
**Role:** Looks like an earlier playground of training helpers — three separate functions that have since been superseded.
**Key classes/functions:**
- `slice_page_to_systems` — eager system slicer that also crops the image tensor.
- `train_phase3_end_to_end` — manual end-to-end (detector + RoI + GNN all trainable) loop; references undefined `EdgeFocalLoss`, `split_into_systems`, `generate_axis_aware_edges`, and `class_to_idx`.
- `train_gnn_phase2` — manual PyG `DataLoader` training loop using `EdgeFocalLoss`.
- `split_into_systems` (a third copy, in addition to the ones in `utils.py`).
**Preconditions:** None enforced; functions assume globals that aren't imported in this file.
**Produces / used by:** Nothing in the package imports from `train_tools`. **Flagged as likely dead code** / superseded by `train.py` + `trainer.py` + `utils.py`.

### `trainer.py`
**Role:** Custom HF `Trainer` subclass that implements the phase loss: for each page in the batch, run the detector backbone+neck once (gradient flow matches the wrapper's `unfreeze` mode — `no_grad` only around the frozen parts), then iterate the systems, run the (trainable) RoI extractor, generate candidate edges, run the GNN, and accumulate focal loss across systems. Box coordinates are normalized to [0,1] before entering the node features (unnormalized pixel coords stall GATv2 learning entirely); the heuristics keep the pixel-space copy they were tuned for. Optional `box_jitter` adds gaussian pixel noise to GT boxes during training (phase-4 curriculum).
**Key classes/functions:** `GNNTrainer` (with `compute_loss`, a now-unused alternative `compute_loss2`, `prediction_step` — routes evaluation through `compute_loss` since `GNNPhase2Model` has no `forward()`, and returns the *edge* logits/targets pair for metrics — and `evaluate` which drives the topology evaluator). `compute_gnn_metrics(eval_pred)` produces per-edge-class F1 (all six classes, stable `labels=range(6)`) + macro F1 via `sklearn.classification_report`.
**Preconditions:** Model wrapper exposes `.detector.backbone`, `.detector.neck`, `.roi_extractor`, `.gnn`, `.gnn.class_embedding`, and `.unfreeze`. Batches come from `omr_collate_fn`. Needs `class_list` so it can rebuild `class_to_idx`.
**Produces / used by:** Used exclusively by `train.py`. The dummy-loss-times-zero trick when no valid system exists keeps AMP happy.

### `utils.py`
**Role:** Data-pipeline glue: the HF Trainer collator and the page→systems splitter.
**Key classes/functions:** `omr_collate_fn` (preserves the variable-size structure as Python lists); `split_into_systems` (groups primitives into the y-range of each `system` bbox and re-indexes GT edges via a vectorized global→local lookup tensor — the earlier per-edge `.item()` dict loop forced thousands of GPU syncs per training step). Two unused alternative implementations `split_into_systems2` and `split_into_systems3` are also defined.
**Preconditions:** `gt_edges` is an `(E,3)` tensor `[u_global, v_global, class]`; `class_to_idx["system"]` must exist.
**Produces / used by:** `omr_collate_fn` is set as the `data_collator` in `train.py`; `split_into_systems` is called by `trainer.GNNTrainer.compute_loss`. The `2`/`3` variants are unused.

### `validate_groundtruth.py`
**Role:** CLI that runs the current serializer on the ground-truth graphs to verify the serializer end-to-end (no GNN involved).
**Scenario:** "Given the GT `.pt` files, do we get sensible `**kern`? If not, the bug is in the serializer, not the GNN."
**Key classes/functions:** `_load_and_resolve_nodes` (reproduces the exact pseudo-ID collision logic from `GroundTruthGraphBuilder` so node IDs match the saved graph), `_build_page_nodes` (filters and adds `cx`/`cy`), `process_single_graph` (drives `MinimalHumdrumSerializer.add_page` per page then `.export_to_krn`), `main` (argparse).
**Preconditions:** `--graph_dir` of `.pt` files, each carrying `edge_index`, `y`, `node_ids`, `json_files`; `--json_dir` containing those JSONs; `--roles_file` (default `conf/structure.json`).
**Produces / used by:** Writes one `.krn` per `.pt` into `--out_dir`. Logs success count.
