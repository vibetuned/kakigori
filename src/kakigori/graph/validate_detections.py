"""Detection-driven end-to-end validation: page images -> detector ->
GNN edges -> **kern, scored against MEI groundtruth with compare-kern.

This is the honest `infer-omr` measurement: unlike validate-predictions
(ground-truth boxes, predicted edges), nothing here touches the
annotations — the node set comes entirely from the detector.

Usage:
    validate-detections --img_dir data/validation-small/imgs \
                        --mei_dir data/validation-small/mei \
                        --out_dir data/validation-small/krn-det \
                        --detector-checkpoint checkpoints/run_006 \
                        --gnn-checkpoint checkpoints_gnn/run_004
    compare-kern --mei_dir data/validation-small/mei \
                 --krn_dir data/validation-small/krn-det
"""

# Standard library imports
import re
import json
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
from PIL import Image
from tqdm import tqdm

# Local imports
from .serializers import MinimalHumdrumSerializer
from .graph_repair import repair_page_edges
from .bbox_repair import repair_page_boxes
from .infer import detect_page_nodes
from .validate_predictions import (
    _load_gnn, predict_page_edges, _inject_system_measure_edges,
)
from kakigori.vision.model import MusicDetector
from kakigori.vision.utils import load_checkpoint

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _score_pages(img_dir: Path) -> dict:
    """Group page images by score stem, sorted by NUMERIC page number."""
    scores = {}
    for p in img_dir.glob("*_page*.png"):
        m = re.match(r"(.+)_page(\d+)$", p.stem)
        if not m:
            continue
        scores.setdefault(m.group(1), []).append((int(m.group(2)), p))
    return {
        stem: [p for _, p in sorted(pages)]
        for stem, pages in scores.items()
    }


def process_score(stem: str, page_paths: list, out_dir: Path, node_roles: dict,
                  detector, gnn_wrapper, class_list: list, class_to_idx: dict,
                  device, conf_thresh: float, iou_thresh: float,
                  input_size: int, bbox_repair: bool = True,
                  graph_repair: bool = True) -> bool:
    try:
        all_edges, all_pages, node_ids = [], [], []
        for page_idx, img_path in enumerate(page_paths):
            page_image = Image.open(img_path).convert("RGB")
            page_nodes = detect_page_nodes(
                detector, page_image, page_idx, class_list, device,
                input_size, conf_thresh, iou_thresh,
            )
            if not page_nodes:
                continue
            if bbox_repair:
                repair_page_boxes(page_nodes, page_prefix=f"p{page_idx}")
            page_edges = predict_page_edges(
                gnn_wrapper, page_nodes, img_path, class_to_idx, device,
                input_size=input_size,
            )
            _inject_system_measure_edges(page_nodes, page_edges)
            if graph_repair:
                repair_page_edges(page_nodes, page_edges)
            all_edges.extend(page_edges)
            all_pages.append(page_nodes)
            node_ids.extend(n["id"] for n in page_nodes)

        if not all_edges:
            return False

        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
        rows = [(id_to_idx[u], id_to_idx[v], c) for u, v, c in all_edges
                if u in id_to_idx and v in id_to_idx]
        edge_index = torch.tensor(
            [[r[0] for r in rows], [r[1] for r in rows]], dtype=torch.long)
        edge_predictions = torch.tensor([r[2] for r in rows], dtype=torch.long)

        serializer = MinimalHumdrumSerializer(
            edge_index=edge_index,
            edge_predictions=edge_predictions,
            node_roles=node_roles,
            pyg_node_ids=node_ids,
        )
        for page_nodes in all_pages:
            serializer.add_page(page_nodes)

        kern_stream = serializer.export_to_krn()
        if kern_stream.startswith("Error"):
            logger.warning(f"{stem}: {kern_stream}")
            return False

        with open(out_dir / f"{stem}.krn", "w", encoding="utf-8") as f:
            f.write(kern_stream)
            f.write("\n")
        return True
    except Exception as e:
        logger.warning(f"Failed to process {stem}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Detection-driven end-to-end validation (no groundtruth "
        "boxes): detector -> GNN -> **kern."
    )
    parser.add_argument("--img_dir", type=str, required=True, help="Dir with page .png images")
    parser.add_argument("--mei_dir", type=str, required=True,
                        help="Dir with .mei groundtruth (defines the score list)")
    parser.add_argument("--out_dir", type=str, required=True, help="Output dir for .krn")
    parser.add_argument("--detector-checkpoint", type=str, required=True)
    parser.add_argument("--gnn-checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="conf/config.json")
    parser.add_argument("--roles_file", type=str, default="conf/structure.json")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--no-bbox-repair", action="store_true",
                        help="Disable detection-level box repairs (A/B baseline)")
    parser.add_argument("--no-repair", action="store_true",
                        help="Disable the spatial graph-repair heuristics")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip scores whose .krn already exists (crash resume)")
    args = parser.parse_args()

    with open(args.config) as f:
        class_list = json.load(f)["target_classes"]
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    with open(args.roles_file) as f:
        node_roles = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = MusicDetector(num_classes=len(class_list))
    load_checkpoint(detector, args.detector_checkpoint, device)
    gnn_wrapper = _load_gnn(args.gnn_checkpoint, device)

    img_dir, out_dir = Path(args.img_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems = sorted(p.stem for p in Path(args.mei_dir).glob("*.mei"))
    pages_by_score = _score_pages(img_dir)

    success = 0
    for stem in tqdm(stems, desc="Detection-driven kern"):
        if args.skip_existing and (out_dir / f"{stem}.krn").exists():
            success += 1
            continue
        page_paths = pages_by_score.get(stem)
        if not page_paths:
            logger.warning(f"No page images for {stem}")
            continue
        if process_score(stem, page_paths, out_dir, node_roles, detector,
                         gnn_wrapper, class_list, class_to_idx, device,
                         args.conf_thresh, args.iou_thresh, args.input_size,
                         bbox_repair=not args.no_bbox_repair,
                         graph_repair=not args.no_repair):
            success += 1
    logger.info(f"Finished. Generated {success}/{len(stems)} detection-driven Humdrum files.")


if __name__ == "__main__":
    main()
