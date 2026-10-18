"""End-to-end OMR inference: score pages -> Humdrum **kern.

Runs the full pipeline on a PDF or a directory of page images:

  1. detector checkpoint      — glyph/structure detection on each page
  2. GNN wrapper checkpoint   — edge prediction over the detections
                                (its own adapted PANet neck feeds the RoI
                                features; the detection neck stays intact)
  3. graph repairs            — spatial certainties (graph_repair.py)
  4. MinimalHumdrumSerializer — staff-identity-aware **kern export

Usage:
    infer-omr --pdf score.pdf \
              --detector-checkpoint checkpoints/run_006 \
              --gnn-checkpoint checkpoints_gnn/run_004 \
              --output-dir out/
    infer-omr --images pages/ ...     # directory of pre-rendered .png pages

Outputs into --output-dir: page_NNNN.png + page_NNNN.json (detections in
dataset annotation format, same as infer-model) and score.krn.
"""

# Standard library imports
import json
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
from PIL import Image
from tqdm import tqdm

# Local imports
from musicset.views.kern import MinimalHumdrumSerializer
from .graph_repair import repair_page_edges
from .bbox_repair import repair_page_boxes
from .validate_predictions import (
    _load_gnn, predict_page_edges, _inject_system_measure_edges,
)
from kakigori.vision.model import MusicDetector
from kakigori.vision.utils import detector_for_checkpoint, load_checkpoint, decode_model_outputs
from kakigori.vision.infer import pdf_to_pages, preprocess

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@torch.inference_mode()
def detect_page_nodes(detector, page_image: Image.Image, page_idx: int,
                      class_list: list, device, input_size: int,
                      conf_thresh: float, iou_thresh: float) -> list:
    """Detect glyphs on one page; return serializer-format nodes in
    ORIGINAL page coordinates (the serializer's geometry space)."""
    tensor, meta = preprocess(page_image, input_size, device)
    outputs = detector(tensor)
    preds = decode_model_outputs(outputs, conf_thresh, iou_thresh, input_size)[0]

    nodes = []
    boxes = preds["boxes"].cpu()
    labels = preds["labels"].cpu()
    scores = preds["scores"].cpu()
    for i in range(boxes.shape[0]):
        x1 = float(((boxes[i, 0] - meta["pad_x"]) / meta["scale"]).clamp(min=0.0))
        y1 = float(((boxes[i, 1] - meta["pad_y"]) / meta["scale"]).clamp(min=0.0))
        x2 = float(((boxes[i, 2] - meta["pad_x"]) / meta["scale"]).clamp(max=float(meta["orig_w"])))
        y2 = float(((boxes[i, 3] - meta["pad_y"]) / meta["scale"]).clamp(max=float(meta["orig_h"])))
        nodes.append({
            "id": f"p{page_idx}n{i}",
            "class": class_list[int(labels[i])],
            "score": round(float(scores[i]), 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "cx": (x1 + x2) / 2.0,
            "cy": (y1 + y2) / 2.0,
        })
    return nodes


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end OMR: PDF or page images -> Humdrum **kern."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", type=str, help="Input PDF score.")
    src.add_argument("--images", type=str,
                     help="Directory of pre-rendered page images (.png), page order = sorted names.")
    parser.add_argument("--detector-checkpoint", type=str, required=True,
                        help="Detection checkpoint dir (e.g. checkpoints/run_006)")
    parser.add_argument("--gnn-checkpoint", type=str, required=True,
                        help="GNN wrapper run dir (e.g. checkpoints_gnn/run_004)")
    parser.add_argument("--config", type=str, default="conf/config.json")
    parser.add_argument("--roles_file", type=str, default="conf/structure.json")
    parser.add_argument("--output-dir", type=str, default="inference_out")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300, help="DPI for PDF rendering.")
    parser.add_argument("--split", action="store_true",
                        help="Write one .krn per page (page_NNNN.krn) instead of "
                             "a single document-level kern")
    parser.add_argument("--no-repair", action="store_true",
                        help="Disable the spatial graph-repair heuristics")
    parser.add_argument("--no-bbox-repair", action="store_true",
                        help="Disable detection-level structure box repairs")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    with open(args.config) as f:
        class_list = json.load(f)["target_classes"]
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    with open(args.roles_file) as f:
        node_roles = json.load(f)

    detector = detector_for_checkpoint(args.detector_checkpoint, len(class_list))
    load_checkpoint(detector, args.detector_checkpoint, device)
    gnn_wrapper = _load_gnn(args.gnn_checkpoint, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        pages = pdf_to_pages(Path(args.pdf), dpi=args.dpi)
        stem = Path(args.pdf).stem
    else:
        img_paths = sorted(Path(args.images).glob("*.png"))
        if not img_paths:
            raise SystemExit(f"No .png pages found in {args.images}")
        pages = [Image.open(p).convert("RGB") for p in img_paths]
        stem = Path(args.images).name

    # 1. Detect nodes and predict edges page by page
    all_edges, all_pages, node_ids = [], [], []
    for page_idx, page_image in enumerate(
            tqdm(pages, desc="Detect + predict edges", unit="page")):
        img_path = output_dir / f"page_{page_idx + 1:04d}.png"
        page_image.save(img_path)

        page_nodes = detect_page_nodes(
            detector, page_image, page_idx, class_list, device,
            args.input_size, args.conf_thresh, args.iou_thresh,
        )
        with open(output_dir / f"page_{page_idx + 1:04d}.json", "w") as f:
            json.dump({
                "image": img_path.name,
                "width": page_image.width,
                "height": page_image.height,
                "annotations": [
                    {"class": n["class"], "score": n["score"], "bbox": n["bbox"]}
                    for n in page_nodes
                ],
            }, f, indent=2)

        if not page_nodes:
            continue
        if not args.no_bbox_repair:
            repair_page_boxes(page_nodes, page_prefix=f"p{page_idx}")
        page_edges = predict_page_edges(
            gnn_wrapper, page_nodes, img_path, class_to_idx, device,
            input_size=args.input_size,
        )
        _inject_system_measure_edges(page_nodes, page_edges)
        if not args.no_repair:
            repair_page_edges(page_nodes, page_edges)

        all_edges.extend(page_edges)
        all_pages.append(page_nodes)
        node_ids.extend(n["id"] for n in page_nodes)

    if not all_edges:
        raise SystemExit("No edges predicted — nothing to serialize.")

    # 2. Serialize — the whole document, or one kern per page with --split
    def serialize(page_groups: list, edges: list, ids: list) -> str:
        known = set(ids)
        serializer = MinimalHumdrumSerializer(
            [(u, v, c) for u, v, c in edges if u in known and v in known], node_roles
        )
        for page_nodes in page_groups:
            serializer.add_page(page_nodes)
        return serializer.export_to_krn()

    if args.split:
        written = 0
        for page_idx, page_nodes in enumerate(all_pages):
            ids = [n["id"] for n in page_nodes]
            id_set = set(ids)
            page_edges = [(u, v, c) for u, v, c in all_edges
                          if u in id_set and v in id_set]
            kern_stream = serialize([page_nodes], page_edges, ids)
            if kern_stream.startswith("Error"):
                logger.warning(f"page {page_idx + 1}: {kern_stream}")
                continue
            with open(output_dir / f"page_{page_idx + 1:04d}.krn", "w",
                      encoding="utf-8") as f:
                f.write(kern_stream)
                f.write("\n")
            written += 1
        logger.info(f"Done. Wrote {written}/{len(all_pages)} per-page .krn files "
                    f"to {output_dir}/ ({len(node_ids)} detections).")
    else:
        kern_stream = serialize(all_pages, all_edges, node_ids)
        if kern_stream.startswith("Error"):
            raise SystemExit(f"Serialization failed: {kern_stream}")
        krn_path = output_dir / f"{stem}.krn"
        with open(krn_path, "w", encoding="utf-8") as f:
            f.write(kern_stream)
            f.write("\n")
        logger.info(f"Done. Wrote {krn_path} ({len(node_ids)} detections, "
                    f"{len(all_edges)} edges, {len(pages)} pages).")


if __name__ == "__main__":
    main()
