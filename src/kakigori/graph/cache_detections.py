"""Detect every page of a dataset once and store its boxes, after bbox repair,
for stage-D GNN training (OMRFullPageDataset ``det_dir``): the GNN then
learns on the boxes it meets at inference, false positives and synthesized
structure included.

Usage:
    cache-detections --img-dir data/train-lite/imgs --out-dir data/train-lite/detections \
                     --detector-checkpoint checkpoints_gnn_lite_896/run_003 \
                     --config conf-lite/config.json

Writes <out-dir>/<page stem>.json: {"nodes": [{"id", "class", "score",
"bbox" (page pixels, xyxy), "synthesized"}]}. Pages already cached are
skipped, so an interrupted run resumes.
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

# First party imports
from kakigori.vision.utils import load_detector

# Local folder imports
from .infer import detect_page_nodes
from .model import is_synthesized
from .bbox_repair import repair_page_boxes

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Cache a detector's boxes per page for stage-D training.")
    parser.add_argument("--img-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--detector-checkpoint", type=str, required=True,
                        help="A detector run, or a GNN / joint run (its detector)")
    parser.add_argument("--config", type=str, default="conf-lite/config.json")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    args = parser.parse_args()

    with open(args.config) as f:
        class_list = json.load(f)["target_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(args.detector_checkpoint, len(class_list), device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages = sorted(Path(args.img_dir).glob("*.png"))
    todo = [p for p in pages if not (out_dir / f"{p.stem}.json").exists()]
    logger.info(f"{len(pages)} pages, {len(pages) - len(todo)} already cached")

    for img_path in tqdm(todo, desc="Cache detections", mininterval=30):
        nodes = detect_page_nodes(
            detector, Image.open(img_path).convert("RGB"), 0, class_list, device,
            None, args.conf_thresh, args.iou_thresh,
        )
        if nodes:
            repair_page_boxes(nodes, page_prefix="p0")
        record = {"nodes": [
            {"id": n["id"], "class": n["class"], "score": n["score"], "bbox": n["bbox"],
             "synthesized": is_synthesized(n)}
            for n in nodes
        ]}
        tmp = out_dir / f"{img_path.stem}.json.tmp"
        tmp.write_text(json.dumps(record))
        tmp.replace(out_dir / f"{img_path.stem}.json")
    logger.info(f"Cached detections for {len(pages)} pages in {out_dir}")


if __name__ == "__main__":
    main()
