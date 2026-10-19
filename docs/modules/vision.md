# vision/ — module map

## Pipeline overview

The `vision` package implements a CenterNet-style multi-scale OMR detector. `dataset.py` loads rendered score pages and their bbox annotations into letterboxed tensors; `model.py` (with helpers in `layers.py`, backbone in `regla.py`) wraps a ReGLA backbone on one grey channel, a PANet neck, and three decoupled heads that emit classification heatmaps + box regressions at three scales; `trainer.py` (with `losses.py`) assigns each ground-truth box to a scale by area, then computes focal classification loss plus CIoU regression loss. `train.py` / `retrain.py` are CLI entry points around that trainer; `eval.py` measures per-class mAP; `infer.py` runs the trained model over a PDF; `visualize.py` is an interactive PySide6 GUI for inspecting boxes, heatmaps and feature maps. `utils.py` holds shared helpers (collate, checkpoint loading, NMS-based decoding, ratio sampler), and `calculate_areas.py` is an offline tool for choosing the per-scale area thresholds the trainer consumes.

## Modules

### `__init__.py`
**Role:** Empty package marker; no exported symbols.
**Preconditions:** None.
**Produces / used by:** Makes `kakigori.vision` importable.

### `calculate_areas.py`
**Role:** Offline CLI tool that scans every annotation JSON in a directory and reports the median normalized bbox area per class, plus a suggested P2/P3/P4 grid assignment.
**Key classes/functions:** `main()` (argparse: `--ann-dir`).
**Preconditions:** A directory of `*.json` annotations in the project's `{width, height, annotations:[{class, bbox:[x1,y1,x2,y2]}]}` format.
**Produces / used by:** Prints a table to stdout. Output is meant to inform the `--scale-ranges` choice fed to `train.py` / `retrain.py` and `OMRTrainer.DEFAULT_SCALE_RANGES`.

### `dataset.py`
**Role:** `OMRDataset` PyTorch `Dataset` that pairs `*.png` images with `*.json` annotations, normalizes boxes to `(cx, cy, w, h)`, letterboxes images to a square canvas, and optionally applies an albumentations + OpenCV "scanned document" augmentation pipeline.
**Key classes/functions:** `OMRDataset`, `_letterbox(image, size, boxes)`.
**Preconditions:** Matching `<stem>.png` and `<stem>.json` pairs under `img_dir` and `ann_dir`; an explicit `class_list` so unknown classes are silently dropped.
**Produces / used by:** Returns `(image_tensor, {"boxes", "labels"})` samples consumed by `train.py`, `retrain.py`, and `eval.py`. `_letterbox` is also reused by `infer.py`.

### `eval.py`
**Role:** CLI script that evaluates a trained checkpoint's mAP on a held-out set and logs results to TensorBoard. Runs GPU inference in a single loop, then computes per-class mAP in isolated worker processes (with a 5-minute timeout) to insulate the main process from torchmetrics' C++ matcher crashing or hanging on classes with many detections. The global means average only classes with a non-zero mAP, so a run that drops a class to 0 (or crashes it) is not penalised for it: compare runs per class, or over a fixed class set. `--multi-label` decodes every class over the threshold per cell instead of the argmax (see `utils.py`).
**Key classes/functions:** `main()`, `_evaluate_isolated`, `_evaluate_target_func`, `format_ground_truth`, `get_checkpoint_step`.
**Preconditions:** A trained checkpoint dir (`--checkpoint`), validation `--img-dir` / `--ann-dir`, a `conf/config.json` listing `target_classes`, a `conf/hierarchy.json` for grouped TensorBoard layout, and model architecture flags (`--use-bottom-up`, `--out-indices`) matching the checkpoint.
**Produces / used by:** Logs `mAP/IoU_0.50`, `mAP/IoU_0.75`, `mAP/IoU_0.50_0.95`, and per-class `mAP_Class/<name>` scalars to TensorBoard under `<tb-dir>/<checkpoint-name>`; prints the same to stdout. Force-exits with `os._exit(0)` to defeat lingering multiprocessing semaphores.

### `infer.py`
**Role:** CLI script that renders every page of a PDF and runs the detector over it, emitting per-page dataset-format JSON annotations alongside the rendered PNGs.
**Key classes/functions:** `main()`, `pdf_to_pages` (uses `pypdfium2`), `preprocess` (letterbox + tensorize + record padding meta). Boxes map back to page pixels with the meta's per-axis `scale_x`/`scale_y` (the rounded resize), not the unrounded `scale`, which drifts up to half a canvas pixel (~2 page px) at the far edge.
**Preconditions:** A trained checkpoint (`--checkpoint`), an input `--pdf`, `conf/config.json` for class names, and model architecture flags matching the checkpoint.
**Produces / used by:** Writes `page_NNNN.png` and `page_NNNN.json` files under `--output-dir`. The JSONs match `OMRDataset`'s annotation schema, so they can be fed straight back into the dataset pipeline. Reused by `visualize.py` via its `preprocess` helper.

### `layers.py`
**Role:** Building blocks for the detector — depthwise-separable convs, a PANet neck (top-down with optional bottom-up path), and a decoupled (cls + reg) head with focal-prior bias initialization.
**Key classes/functions:** `DepthwiseSeparableConv`, `PANetNeck`, `DecoupledHead`.
**Preconditions:** Three feature maps of channel sizes given by `in_channels_list` from the backbone.
**Produces / used by:** Imported by `model.py`.

### `losses.py`
**Role:** The two loss functions consumed by `OMRTrainer`: `CIoULoss` (regression via torchvision's `complete_box_iou_loss`) and `DynamicFocalLoss` (binary focal loss whose `gamma` is linearly annealed from `base_gamma` to `max_gamma` based on a `progress` argument supplied by the trainer).
**Key classes/functions:** `CIoULoss`, `DynamicFocalLoss`.
**Preconditions:** `progress` in `[0.0, 1.0]` from caller; boxes in `(cx, cy, w, h)` for CIoU.
**Produces / used by:** Used by `trainer.py`.

### `model.py`
**Role:** `MusicDetector` — the end-to-end network: backbone (ReGLA `regla_m` by default, one grey input channel, trained from scratch; or a timm model name such as the 0.1.x `convnext_base.dinov3_lvd1689m`, which takes grey pages repeated to RGB) → `PANetNeck` (hidden_dim=128) → three `DecoupledHead`s, one per fused scale. `detect_backbone(state_dict)` reads which backbone a checkpoint holds from its weight names and shapes; `utils.detector_for_checkpoint()` builds a matching detector, so every loader (eval, infer, retrain, GNN, joint) follows the checkpoint and the released 0.1.x weights still load.
**Key classes/functions:** `MusicDetector(num_classes, use_bottom_up, out_indices, backbone)`, `detect_backbone`.
**Preconditions:** `num_classes` must match the checkpoint at load time; `out_indices` selects which three backbone stages to extract (`[0,1,2]`: strides 4/8/16; ReGLA builds only the stages it returns).

### `regla.py`
**Role:** ReGLA (Li et al., arXiv 2602.05262): stem to 1/4, stages 1-2 of ELRF blocks (DW3x3 -> 1x1 to 4C -> DW5x5 -> 1x1 back to C, residual: a 7x7 receptive field from two small depthwise convolutions), stages 3-4 of attention blocks (CPE, RGMA, a local separable 3x3 convolution). RGMA: v = sigmoid(W_v x) is both value and gate, Q, K = ReLU(W_qk x); attn = Q(K^T v) / Q(K^T 1), gated by v, normalized, projected, residual: global attention linear in the number of tokens, computed in fp32. Variants `regla_t/s/m/l/x` follow the paper's Table 1.
**Departures from the paper's Fig. 2:** ELRF is a linear bottleneck (no GELU after the final 1x1); RGMA has a pre-norm and a 1x1 output projection (without it and with a depthwise local module the parameter counts were 72% of Table 1; the projection brings regla_m to 88%, the separable local module to 106%, all variants 104-111%); Q and K are scaled by 1/sqrt(N) before the token sums, which cancels in the ratio and keeps fp16 from overflowing at stride-4 token counts (ANE export); the local module after RGMA is separable (DW3x3 + norm + GELU + 1x1) where the paper used a depthwise 3x3 (a candidate ablation).
**Chosen here (not in the paper, no code or supplement available):** two-conv stem, 3x3 stride-2 downsampling, channel LayerNorm (the GNN trainers keep the detector in eval mode and run it one page at a time; `norm="bn"` exists), single-head attention, layer scale 1e-2. `LayerNorm2d` uses PyTorch's fused kernel behind a permute in training (fp32 under bf16 autocast) and a permute-free form only when traced or exported for Core ML / the ANE; both read the same parameters (the hand-written form in training cost 55.6 vs 33.7 ms and 4.80 vs 3.58 GiB per step).
**Key classes/functions:** `ReGLA(variant, in_chans, out_indices, norm, heads)`, `ELRF`, `RGMA`, `AttentionBlock`, `LayerNorm2d`, `VARIANTS`.
**Produces / used by:** Feature maps at the requested stages (`channels` lists their widths); stages past the last requested one are not built. regla_m stages 1-3: 4.63M parameters (the detector 4.97M, vs 60.5M with ConvNeXt-B; regla_l: 11.7M). A batch-4 640px bf16 training step on an RTX 5090: regla_m 33.5 ms / 3.58 GiB, regla_l 47.1 ms / 6.05 GiB, ConvNeXt-B 41.2 ms / 5.70 GiB. `MusicDetector` returns a list of dicts, so a Core ML export needs a wrapper that returns tensors.

### `retrain.py`
**Role:** CLI script for *class-expansion* fine-tuning: load weights trained with a smaller class vocabulary, surgically widen the classification 1x1 conv on each head to the new `num_classes`, then run a two-stage training schedule (Stage 1: backbone+neck frozen for `--freeze-epochs`; Stage 2: unfrozen at `--lr-unfrozen` for the remaining epochs).
**Key classes/functions:** `parse_args()`, `main()`.
**Preconditions:** `--fine-tune <path>` is REQUIRED — this script's whole purpose is to load a prior checkpoint and expand its head. Also requires `--old-num-classes` to match that checkpoint, a current `conf/config.json` (whose `target_classes` defines the new, larger vocabulary), and optional synthetic-data dirs when `--use-synthetic` is set.
**Produces / used by:** Checkpoint subdirectories `run_NNN/checkpoint-XXXX/` under `--output-dir`, plus TensorBoard logs under `<logging-dir>/run_NNN/`. Final model written via `trainer_unfrozen.save_model()`.

### `train.py`
**Role:** Primary training CLI — initializes a fresh `MusicDetector` (or loads weights via `--fine-tune` for simple warm-start without head expansion), wraps it in `OMRTrainer`, and runs HF Trainer with cosine LR + bf16. Supports `--resume` / `--resume-from-checkpoint` for HF-Trainer-style resumption. Reads YAML defaults from `--train-config` first so CLI args can override them.
**Key classes/functions:** `parse_args()`, `main()`.
**Preconditions:** Image/annotation directories on disk; `conf/config.json` with `target_classes`. `--fine-tune` is optional here (unlike `retrain.py`); if omitted the model starts from scratch with `--backbone` (default `regla_m`; a timm name starts from its pretrained weights). With `--fine-tune` the backbone comes from the checkpoint.
**Produces / used by:** `run_NNN/` directories under `--output-dir` containing HF checkpoints, plus TensorBoard scalars under `<logging-dir>/run_NNN/`.

### `trainer.py`
**Role:** `OMRTrainer` subclasses HF `Trainer` to implement the OMR-specific loss. `build_targets` assigns each ground-truth box to exactly one of the three feature scales based on normalized box area (`scale_ranges`), then rasterizes class one-hots and regression targets into per-scale grid maps. `compute_loss` decodes predictions back to `(cx, cy, w, h)` (sigmoid on offsets, exp on w/h) and combines `DynamicFocalLoss` (classification, with `progress` from `state.global_step / state.max_steps`) and `CIoULoss` (regression, only on positive cells). Also overrides `log()` to surface `train/cls_loss` and `train/reg_loss`, and `_get_train_sampler()` to honor an injected `custom_sampler` (used for synthetic/real ratio mixing).
**Key classes/functions:** `OMRTrainer`, `OMRTrainer.build_targets`, `OMRTrainer.compute_loss`.
**Preconditions:** Model whose forward returns the 3-dict list shape produced by `MusicDetector`; data collator that emits `{"pixel_values", "labels"}` (see `omr_collate_fn`).
**Produces / used by:** Used by `train.py` and `retrain.py`. Default scale buckets are `[(0, 1e-4), (1e-4, 1e-3), (1e-3, 2.0)]` and can be overridden by `--scale-ranges`.

### `utils.py`
**Role:** Shared helpers — `RatioSampler` (multi-dataset sampler anchored to dataset 0 for mixing real + synthetic), `omr_collate_fn` (stacks images and keeps per-sample target dicts as a list), `load_checkpoint` (resolves a HF Trainer dir, a single-run dir, or a direct weights file, supporting `model.safetensors` and `pytorch_model.bin`), and `decode_model_outputs` (sigmoid + exp box decoding, confidence threshold, class-aware `batched_nms`). By default each cell emits only its argmax class; `multi_label=True` emits every class over the threshold, sharing the cell's box. Training marks every class centred on a cell positive, so coincident boxes (measure/staff/layer of one staff, mRest over its whole-rest glyph, a one-accidental keySig) only survive multi-label decoding (run_006: 0.627 -> 0.731 mAP@.50 over the 80 validation-lite classes).
**Key classes/functions:** `RatioSampler`, `omr_collate_fn`, `load_checkpoint`, `decode_model_outputs`.
**Preconditions:** For `decode_model_outputs`, raw model output list-of-dicts as `MusicDetector` returns.
**Produces / used by:** Used by `train.py`, `retrain.py`, `eval.py`, `infer.py`, `visualize.py`.

### `visualize.py`
**Role:** Interactive PySide6 GUI (`ModelVisualizer`) for exploring model behavior on a folder of images. Works with ReGLA and legacy ConvNeXt checkpoints (the backbone is read from the weights). Four view modes: (a) bounding boxes with per-class color and visibility toggles grouped by `conf/hierarchy.json`, (b) classification heatmaps per scale, (c) feature maps from the neck or the backbone stages, (d) comparison with the ground truth (`--ann-dir`): `match_page` matches per class at IoU 0.5 and draws false positives, wrong-class detections and misses, with per-page precision/recall/F1 and a "rank pages, worst F1 first" pass over the folder. Inference runs in a background thread, results are cached per image, and the confidence slider and multi-label toggle re-decode from cached raw outputs without re-running the model.
**Key classes/functions:** `ModelVisualizer`, `match_page`, `page_scores`, `ResizableGraphicsView`, `InferenceSignals`, `_jet_rgba`, `_class_color`, `pil2qpixmap`, `main()`.
**Preconditions:** A checkpoint (`--checkpoint`), an image directory (`--img_dir`, scans `*.png/*.jpg/*.jpeg` recursively) or one `--image`, optionally `--ann-dir` for the comparison mode, `conf/config.json`, `conf/hierarchy.json`. Requires a display.
**Produces / used by:** No files written; this is purely an interactive viewer. Reuses `infer.preprocess` and `utils.decode_model_outputs`.
