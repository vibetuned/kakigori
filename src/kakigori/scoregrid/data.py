"""Training data for the scoregrid decoder: the detection path's systems as
node tokens, each paired with the ground-truth matrix of the system it
detects.

Per page: detection and bbox repair (the boxes inference sees), the GNN's
node states over each detected system, and per detected system a record:

  cls [N]            class index
  geom [N, 5]        centre x and y and size in the system box (fractions),
                     and staff steps from the top line of the node's row
                     (0 top line, 8 bottom line; the row's box is the
                     clef-derived one of bbox repair 7)
  row [N]            index of the node's row in the system (-1: no row)
  meas [N]           index of the detected measure holding it, left to
                     right (-1: none)
  conf [N]           detector score (0: synthesized by bbox repair)
  roi, gnn [N, 256]  RoI feature and GNN node state (fp16)
  target             the matched ground-truth SystemGrid's arrays (IoU >= 0.5
                     between the systems' vertical bands), else None
  score, page, order identify it; ``gt_systems`` per score lists every
  ground-truth system so the unmatched ones count as misses in evaluation.

Usage:
    build-scoregrid-data --data data/validation-lite --out data/scoregrid/validation-lite \\
                         --checkpoint checkpoints_gnn_lite_896/run_003
"""

# Standard library imports
import json
import logging
import argparse
from pathlib import Path

# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

# First party imports
from kakigori.graph.infer import detect_page_nodes
from kakigori.graph.model import node_features, is_synthesized
from kakigori.graph.utils import split_into_systems
from kakigori.vision.dataset import canvas_to_page, page_to_canvas, letterbox_image
from kakigori.graph.heuristics import generate_axis_aware_edges
from kakigori.graph.bbox_repair import repair_page_boxes
from kakigori.graph.validate_detections import _score_pages
from kakigori.graph.validate_predictions import _load_gnn

# Local folder imports
from .grid import systems_from_mei

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_MIN_IOU = 0.5
SHARD_SIZE = 2000


def _row_geometry(boxes, labels, row_idx: int):
    """Per node: its row (the nearest system-staff centre) and its staff
    steps from that row's top line."""
    n = boxes.shape[0]
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0
    rows = boxes[labels == row_idx]
    if rows.shape[0] == 0:
        return torch.full((n,), -1, dtype=torch.long), torch.zeros(n)
    rows = rows[torch.argsort(rows[:, 1])]
    row_cy = (rows[:, 1] + rows[:, 3]) / 2.0
    row = torch.argmin(torch.abs(cy.unsqueeze(1) - row_cy.unsqueeze(0)), dim=1)
    top, height = rows[row, 1], (rows[row, 3] - rows[row, 1]).clamp(min=1e-3)
    return row, ((cy - top) / (height / 8.0)).clamp(-24.0, 32.0)


def _measure_of(boxes, labels, measure_idx: int):
    """Per node: the detected measure holding its centre, measures left to
    right (the narrowest if several; -1: none)."""
    n = boxes.shape[0]
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0
    measures = boxes[labels == measure_idx]
    if measures.shape[0] == 0:
        return torch.full((n,), -1, dtype=torch.long)
    measures = measures[torch.argsort(measures[:, 0])]
    inside = (cx.unsqueeze(1) >= measures[:, 0]) & (cx.unsqueeze(1) <= measures[:, 2])
    width = (measures[:, 2] - measures[:, 0]).unsqueeze(0).expand(n, -1)
    width = torch.where(inside, width, torch.full_like(width, float("inf")))
    best = torch.argmin(width, dim=1)
    return torch.where(inside.any(dim=1), best, torch.full_like(best, -1))


@torch.no_grad()
def page_systems(wrapper, page_nodes: list, img_path: Path, class_to_idx: dict, device) -> list:
    """The detected systems of one page as records (without targets), in
    reading order."""
    canvas, geo = letterbox_image(Image.open(img_path).convert("L"), wrapper.detector.input_size)
    img = TF.to_tensor(canvas).to(device)
    num_known = wrapper.gnn.class_embedding.num_embeddings
    page_nodes = [n for n in page_nodes if class_to_idx[n["class"]] < num_known]
    if len(page_nodes) < 2:
        return []
    boxes = page_to_canvas(torch.tensor([n["bbox"] for n in page_nodes], dtype=torch.float32), geo).to(device)
    labels = torch.tensor([class_to_idx[n["class"]] for n in page_nodes], dtype=torch.long, device=device)
    fused = wrapper.detector.neck(wrapper.detector.backbone(img.unsqueeze(0)))
    feat_dict = {str(i): f for i, f in enumerate(fused)}

    records = []
    empty = torch.empty((0, 3), dtype=torch.long, device=device)
    for group in split_into_systems(boxes, labels, empty, class_to_idx):
        b, lab, idx = group["abs_boxes"], group["labels"], group["node_indices"]
        if len(b) < 2:
            continue
        sx1, sy1, sx2, sy2 = (float(v) for v in group["system_bbox"])
        roi = wrapper.roi_extractor(feat_dict, [b], img.shape[-2:])
        rel = b.clone()
        rel[:, [0, 2]] -= sx1
        rel[:, [1, 3]] -= sy1
        local = idx.cpu().tolist()
        x = node_features(
            wrapper.gnn, roi, rel / geo["canvas_w"], lab,
            scores=torch.tensor([float(page_nodes[i].get("score", 1.0)) for i in local], device=device),
            synthesized=torch.tensor([float(is_synthesized(page_nodes[i])) for i in local], device=device),
        )
        cand = generate_axis_aware_edges(rel, lab, class_to_idx)
        h = wrapper.gnn.node_states(x, cand) if cand.shape[1] else torch.zeros_like(roi)

        sw, sh = max(sx2 - sx1, 1e-3), max(sy2 - sy1, 1e-3)
        cx, cy = (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0
        row, steps = _row_geometry(b.cpu(), lab.cpu(), class_to_idx["system-staff"])
        geom = torch.stack([
            (cx - sx1) / sw, (cy - sy1) / sh, (b[:, 2] - b[:, 0]) / sw, (b[:, 3] - b[:, 1]) / sh,
            steps.to(device),
        ], dim=1)
        page_box = canvas_to_page(torch.tensor([[sx1, sy1, sx2, sy2]]), geo)[0].tolist()
        records.append({
            "system_box": page_box,
            "cls": lab.cpu().short(),
            "geom": geom.cpu().half(),
            "row": row.char(),
            "meas": _measure_of(b.cpu(), lab.cpu(), class_to_idx["measure"]).short(),
            "conf": torch.tensor([float(page_nodes[i].get("score", 1.0)) for i in local]).half(),
            "synth": torch.tensor([is_synthesized(page_nodes[i]) for i in local]),
            "roi": roi.cpu().half(),
            "gnn": h.cpu().half(),
        })
    records.sort(key=lambda r: r["system_box"][1])
    return records


def _band_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU of the boxes' vertical extents, [len(a), len(b)]."""
    top = torch.maximum(a[:, None, 1], b[None, :, 1])
    bottom = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (bottom - top).clamp(min=0)
    union = (a[:, 3] - a[:, 1])[:, None] + (b[:, 3] - b[:, 1])[None, :] - inter
    return inter / union.clamp(min=1e-6)


def match_systems(records: list, gt_boxes: list) -> list:
    """Per record: the position in ``gt_boxes`` ([(anything, box)], the
    page's ground-truth systems) of the system it detects, one-to-one by the
    IoU of their vertical bands >= SYSTEM_MIN_IOU, greedy; -1 if none.
    Systems group their nodes by band (split_into_systems), so a detected
    system box covering part of the width still holds the whole system
    (with a 2-d IoU, 15% of the validation systems failed to match)."""
    if not records or not gt_boxes:
        return [-1] * len(records)
    iou = _band_iou(torch.tensor([r["system_box"] for r in records]),
                    torch.tensor([b for _, b in gt_boxes], dtype=torch.float32))
    out, taken = [-1] * len(records), set()
    for flat in torch.argsort(iou.flatten(), descending=True).tolist():
        r, g = divmod(flat, iou.shape[1])
        if iou[r, g] < SYSTEM_MIN_IOU:
            break
        if out[r] == -1 and g not in taken:
            out[r] = g
            taken.add(g)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build scoregrid decoder data from the detection path.")
    parser.add_argument("--data", type=str, required=True, help="Dataset dir (imgs/, annotations/, mei/)")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Joint / GNN run: its detector detects, its GNN gives the node states")
    parser.add_argument("--detector-checkpoint", type=str, default=None,
                        help="Another detector (default: the checkpoint's own)")
    parser.add_argument("--config", type=str, default="conf-lite/config.json")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N scores (smoke tests)")
    args = parser.parse_args()

    # First party imports
    from kakigori.vision.utils import load_detector

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    class_list = json.load(open(args.config))["target_classes"]
    class_to_idx = {c: i for i, c in enumerate(class_list)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(args.detector_checkpoint or args.checkpoint, len(class_list), device)
    wrapper = _load_gnn(args.checkpoint, device)

    pages_by_score = _score_pages(data / "imgs")
    scores = sorted(p.stem for p in (data / "mei").glob("*.mei"))
    if args.limit:
        scores = scores[:args.limit]
    shard, n_shard, stats = [], 0, {"systems": 0, "matched": 0, "gt_systems": 0, "scores": 0}
    index = {}
    for score in tqdm(scores, desc="Scoregrid data", mininterval=30):
        grids, header = systems_from_mei(data / "mei" / f"{score}.mei", data / "annotations")
        stats["gt_systems"] += len(grids)
        stats["scores"] += 1
        order = 0
        for page_idx, img_path in enumerate(pages_by_score.get(score, [])):
            nodes = detect_page_nodes(detector, Image.open(img_path).convert("RGB"), page_idx,
                                      class_list, device, None, args.conf_thresh, args.iou_thresh)
            if not nodes:
                continue
            repair_page_boxes(nodes, page_prefix=f"p{page_idx}")
            records = page_systems(wrapper, nodes, img_path, class_to_idx, device)
            gt_page = [(i, box) for i, (p, box) in enumerate(header["boxes"]) if p == page_idx]
            matched = match_systems(records, gt_page)
            for record, m in zip(records, matched, strict=True):
                gt_index = gt_page[m][0] if m >= 0 else -1
                record.update(
                    score=score, page=page_idx, order=order, gt_index=gt_index,
                    target=grids[gt_index].arrays() if gt_index >= 0 else None,
                    numbers=grids[gt_index].numbers if gt_index >= 0 else [],
                )
                order += 1
                stats["systems"] += 1
                stats["matched"] += gt_index >= 0
                shard.append(record)
        index[score] = {"gt_systems": len(grids), "meter": header["meter"], "staves": header["staves"]}
        if len(shard) >= SHARD_SIZE:
            torch.save(shard, out / f"shard_{n_shard:03d}.pt")
            shard, n_shard = [], n_shard + 1
    if shard:
        torch.save(shard, out / f"shard_{n_shard:03d}.pt")
    (out / "index.json").write_text(json.dumps({"stats": stats, "scores": index}))
    logger.info(f"{stats['systems']} detected systems ({stats['matched']} matched) for "
                f"{stats['gt_systems']} ground-truth systems in {stats['scores']} scores -> {out}")


if __name__ == "__main__":
    main()
