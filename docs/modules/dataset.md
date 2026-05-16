# dataset/ — module map

## Pipeline overview

The `dataset/` subpackage builds the OMR training corpus end-to-end. Raw MusicXML (`.mxl`) inputs are first filtered (`filter_mxl.py`) for files containing target SMuFL classes, optionally mutated to amplify rare symbols (`synthetic_arranger.py`, `synthetic_generator.py`) or replaced by fully synthetic scale/chord exercises (`synthetic_writer.py`), and stamped with deterministic IDs (`inject_mxl_ids.py`). The prepared MXL files are rendered to per-page SVG + PNG via Verovio (`render_dataset.py`), and also exported to MEI and Humdrum `.krn` (`export_dataset.py`). From the rendered SVGs, geometric bounding boxes are extracted per-class into JSON annotations (`extract_annotations.py`), with class-discovery (`extract_classes.py`) and frequency auditing (`frequencies.py`) tools. Finally, MEI + bbox JSON are paired to build PyTorch Geometric graphs (`generate_graphs.py`). The PNG/JSON pairs feed the vision detector under `vision/`, and the `.pt` graphs feed the GNN under `graph/`. Three Qt-based viewers (`visualize_dataset.py`, `visualize_graphs.py`, `visualize_graphs_pysigma.py`) let humans inspect annotations and graphs.

## Modules

### Input curation

#### `filter_mxl.py`
**Role:** Renders each input `.mxl` to SVG in memory with Verovio and keeps only files whose SVG contains at least one class from `target_classes` in the config; runs each Verovio call in an isolated child process with a 15s timeout to survive C++ crashes.
**Preconditions:** Input directory of `.mxl` files; `conf/config.json` with `target_classes`.
**Produces:** Copies of matching `.mxl` files in `--output_dir`, capped at `--num_files`.
**Position in pipeline:** Stage 1 — selects a manageable subset of raw scores before any further processing.

#### `inject_mxl_ids.py`
**Role:** Walks the MusicXML tree inside each `.mxl` and writes a unique `id="<stem>-<n>"` attribute onto every element that lacks one, producing IDs that survive Verovio's render (which copies XML IDs to SVG `g` elements).
**Preconditions:** Input directory of `.mxl` files.
**Produces:** New `.mxl` files with injected IDs in `--out-dir` (skips already-present outputs for safe resume).
**Position in pipeline:** Stage 2 — runs after filtering and before rendering, so SVG nodes can later be matched back to MEI nodes by ID in graph generation.

### Synthetic generation

#### `synthetic_arranger.py`
**Role:** Loads existing MXL scores with music21 and applies exactly one random per-measure mutation chosen from: add articulations, add ornament/arpeggio, add dynamic, octave-shift (8va/8vb) with compensating transposition, replace durations with 16th/half/whole blocks, or swap in two triplet groups. Preserves time/key/clef metadata.
**Preconditions:** Input directory of `.mxl` files.
**Produces:** `<stem>_arranged.musicxml` files (written via `score.write("musicxml")`).
**Position in pipeline:** Optional augmentation that diversifies existing scores at the measure level.

#### `synthetic_generator.py`
**Role:** Loads existing MXL scores and applies score-wide, probabilistic mutations: dense 16ths, dotted rhythms, 32nds, triplet tuplets, ornaments (trill/mordent/turn/fingering), articulations, dynamics, chord build-out with optional arpeggios, lyrics, rehearsal marks, stem directions, and part-level slurs/ottavas. Coarser and more aggressive than `synthetic_arranger.py` (whole-score rather than per-measure single mutation).
**Preconditions:** Input directory of `.mxl` files.
**Produces:** `synthetic_<original_name>.mxl` in the output directory, plus a `svg_music21_mapping.json` reference file. Stops once `--num_files` successes are reached.
**Position in pipeline:** Alternative augmentation path targeting harder/denser scores with rarer symbol coverage.

#### `synthetic_writer.py`
**Role:** Generates fully new piano scores from scratch (no input MXL). Picks a scale type (major / harmonic minor / melodic minor / chromatic / circle-of-fifths chord progression), builds a 2-staff RH-treble / LH-bass layout with realistic fingerings, ties, slurs, dynamics, ornaments and arpeggios, then equalizes the two hands via LCM measure repetition. Carefully preserves intentional ties around `makeNotation`.
**Preconditions:** None (it invents content).
**Produces:** `synthetic_score_NNNNNN.mxl` files in the output directory.
**Position in pipeline:** Pure-synthetic source used to oversample classes that are rare in real-world MXL (fingerings, scales, specific dynamics/articulations).

### Rendering and conversion

#### `render_dataset.py`
**Role:** Renders each `.mxl` to one SVG per page with Verovio (with `svgViewBox`, fixed `pageWidth=2100`, fixed `xmlIdSeed=42`) and converts each SVG to PNG with CairoSVG. Each render runs in an isolated subprocess with 30s timeout to survive Verovio crashes/hangs.
**Preconditions:** `.mxl` files (typically post-`inject_mxl_ids` so SVGs carry stable IDs).
**Produces:** `<stem>_pageN.svg` in `--svg_dir`, `<stem>_pageN.png` in `--img_dir`.
**Position in pipeline:** Core rendering step — its PNGs are the vision inputs, its SVGs are parsed for annotations.

#### `export_dataset.py`
**Role:** Loads each `.mxl` with Verovio and exports two alternate formats: MEI (via `getMEI`) and Humdrum `.krn` (via `getHumdrumFile`, with fd-1 redirected to `/dev/null` to mute C++ log spam). Runs each conversion in an isolated `spawn` subprocess with retries.
**Preconditions:** `.mxl` files (same set as rendering).
**Produces:** `<stem>.mei` in `--mei-dir`, `<stem>.krn` in `--krn-dir`.
**Position in pipeline:** Provides the MEI ground-truth structure consumed by `generate_graphs.py`; `.krn` is an extra by-product for downstream/Humdrum tooling.

### Annotation extraction

#### `extract_classes.py`
**Role:** One-shot class discovery: scans every Verovio SVG and collects the union of all `class` tokens on `<g>` elements plus SMuFL-mapped `<use href=...>` glyphs. Applies the same `barLine` and `keyAccid` disambiguation heuristics as `extract_annotations.py`, but only emits the sorted class name list (no bboxes).
**Preconditions:** Rendered SVG directory; `conf/smufl_mapping.json` (optional).
**Produces:** Single JSON file (default `available_classes.json`) containing the sorted class vocabulary.
**Position in pipeline:** Used once to bootstrap / audit the `target_classes` entry in `conf/config.json`.

#### `extract_annotations.py`
**Role:** Heavy-lifting bbox extractor. Parses each Verovio SVG, walks `<defs>` glyph paths, accumulates absolute transform matrices through the SVG hierarchy, and computes pixel-space bboxes for every `<g>` whose class is in `target_classes`. Adds derived sub-classes (`barlineSingle/Double/Dashed/Final`, `beam8/16/32/Broken`, `stem8/16/32/4`, `system-staff`), and re-classifies SMuFL `accidental*` glyphs inside `keySig` ancestors as `keyAccid*`. Handles `<use>`, `<path>`, `<rect>`, `<line>`, `<text>` (with text-anchor/font-size), and `<ellipse>`.
**Preconditions:** SVG dir, matching PNG dir (for image w/h), `conf/config.json` (target_classes), `conf/smufl_mapping.json`.
**Produces:** Per-page `<stem>.json` files containing `{image, width, height, annotations: [{class, bbox, id?}]}`.
**Position in pipeline:** Stage 5 — turns rendered SVGs into the bbox dataset consumed by the vision detector and by graph generation.

#### `frequencies.py`
**Role:** Audits annotation JSONs and prints a sorted class-frequency table; emits a pruned JSON list of classes whose count meets `--min-count` to paste back into `conf/config.json`.
**Preconditions:** Directory of annotation JSONs from `extract_annotations.py`.
**Produces:** Stdout report only (no files written).
**Position in pipeline:** Diagnostic / config-tuning helper used after annotation extraction to balance classes and refresh the `target_classes` list.

### Graph generation

#### `generate_graphs.py`
**Role:** Pairs each `.mei` with all matching `<stem>_page*.json` annotation files and feeds them into `kakigori.graph.parsers.GroundTruthGraphBuilder` to compute ground-truth edges. Builds a string-id → integer-index map, packs edges into a PyG-style `edge_index` tensor, and saves a `{edge_index, y, node_ids, num_nodes, mei_file, json_files}` dict via `torch.save`. Each build runs in an isolated `spawn` subprocess with retries.
**Preconditions:** MEI directory (from `export_dataset.py`), JSON annotation directory (from `extract_annotations.py`), `conf/structure.json` (with `node_roles`).
**Produces:** One `<stem>.pt` PyG graph file per MEI in `--out-dir`.
**Position in pipeline:** Final stage — produces the graph artifacts consumed by the GNN training code under `graph/`.

### Visualization

#### `visualize_dataset.py`
**Role:** PySide6 desktop viewer for the PNG + JSON pairs. Draws every annotation bbox over its image, with a sidebar of grouped per-class visibility toggles driven by `conf/hierarchy.json`. Supports zoom/pan, prev/next image, and keyboard shortcuts (`A/D`, arrow keys).
**Preconditions:** Image directory, annotation JSON directory, `conf/hierarchy.json`, `conf/config.json` (target_classes).
**Produces:** Interactive UI only.
**Position in pipeline:** Manual QA tool for the bbox dataset.

#### `visualize_graphs.py`
**Role:** PySide6 desktop viewer that overlays the generated graph on top of the source PNG. Loads the `.pt` graph for the current image, finds each node's bbox via the matching annotation JSON (replaying the `parsers.py` collision-safe pseudo-ID rule), draws node boxes plus center-to-center edge lines colored by edge class (1=Structural, 2=Modifier, 3=Temporal, 4=Sync). Per-layer visibility toggles.
**Preconditions:** Image dir, annotation JSON dir, graph `.pt` dir.
**Produces:** Interactive UI only.
**Position in pipeline:** Visual sanity-check for graph generation, geometrically aligned with the score image.

#### `visualize_graphs_pysigma.py`
**Role:** Alternative graph viewer that renders each `.pt` as an interactive force-directed Sigma.js graph (via `ipysigma`) embedded in a `QWebEngineView`. Iterates over `.pt` files (not images), offers sidebar controls for component selection, min-degree filter, max-node cap, and isolated-node toggle, plus a "Show JSON" dialog that dumps the filtered subgraph. Components are sorted by page and average vertical position of `system` nodes.
**Preconditions:** Graph `.pt` dir; also reads matching annotation JSON for node class labels.
**Produces:** Temporary HTML files (per render) inside a `TemporaryDirectory`; interactive UI only.
**Position in pipeline:** Topology-focused (non-spatial) inspector for graph structure, complementing the bbox-overlaid `visualize_graphs.py`.

### Package marker

#### `__init__.py`
**Role:** Empty package marker that makes `kakigori.dataset` importable.
**Preconditions:** None.
**Produces:** Nothing.
**Position in pipeline:** N/A.
