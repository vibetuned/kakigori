"""Training data for the scoregrid decoder: the detection path's systems as
node tokens, each paired with the ground-truth matrix of the system it
detects.

Per page: detection and bbox repair (the boxes inference sees), the GNN's
node states over each detected system, and per detected system a record:

  cls [N]            class index
  geom [N, 5]        centre x and y and size in the system span (fractions),
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
from .grid import N_PITCH, METER_COUNTS, METER_UNITS, systems_from_mei, grid_from_measures

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SYSTEM_MIN_IOU = 0.5
SHARD_SIZE = 2000

# The diatonic step (octave * 7 + step) of each clef's top staff line,
# measured from the training geometry: G2 38 (F5), F4 26 (A3), the octave
# clefs an octave from G2 (the MEI holds sounding pitch); C clefs by the line
# their centre sits on
CLEF_TOP_LINE = {"G2": 38, "F4": 26, "G2^": 45, "G2v": 31, "C1": 36, "C2": 34, "C3": 32, "C4": 30, "C5": 28}
CLEF_KIND = {
    "clefG": "G2", "gClefChange": "G2", "clefF": "F4", "fClefChange": "F4",
    "clefG8va": "G2^", "clefG8vb": "G2v", "clefC": "C", "cClefChange": "C",
}
NOTEHEADS = ("noteheadBlack", "noteheadHalf", "noteheadWhole")


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


def candidate_pitches(cls, geom, row, class_list: list) -> torch.Tensor:
    """Per node: the pitch (diatonic + 1; 0 none) a notehead's staff position
    implies under the clef in effect at it, the nearest clef or clef change
    to its left on its row (else the row's first). The decoder can copy it
    whatever the clef: the clef-to-pitch mapping is not left to learn from
    the few bass-clef examples, and mid-system clef changes are followed."""
    names = [class_list[int(c)] for c in cls]
    cx, steps = geom[:, 0].float(), geom[:, 4].float()
    clefs = []
    for i, name in enumerate(names):
        if name in CLEF_KIND and int(row[i]) >= 0:
            kind = CLEF_KIND[name]
            if kind == "C":
                kind = f"C{min(5, max(1, 5 - round(float(steps[i]) / 2)))}"
            clefs.append((float(cx[i]), int(row[i]), kind))
    out = torch.zeros(len(names), dtype=torch.int16)
    for i, name in enumerate(names):
        if name not in NOTEHEADS or int(row[i]) < 0:
            continue
        on_row = [c for c in clefs if c[1] == int(row[i])]
        if not on_row:
            continue
        left = [c for c in on_row if c[0] <= float(cx[i])]
        clef = max(left)[2] if left else min(on_row)[2]
        diatonic = CLEF_TOP_LINE[clef] - round(float(steps[i]))
        if 0 <= diatonic < N_PITCH - 1:
            out[i] = diatonic + 1
    return out


def system_meter(cls, geom, row, conf, class_list: list):
    """(count, unit) printed at the start of a system, or None: common and
    cut time are 4/4 and 2/2; otherwise the time-signature digits of the
    first row at the leftmost signature, those above the middle line read
    left to right as the count, those below as the unit. Multi-label decoding
    puts several digit classes on one digit box: each position keeps its
    most confident one."""
    names = [class_list[int(c)] for c in cls]
    if "timeSigCommon" in names:
        return (4, 4)
    if "timeSigCutCommon" in names:
        return (2, 2)
    best: dict = {}
    for i, name in enumerate(names):
        if name.startswith("timeSig") and name[7:].isdigit():
            key = (int(row[i]), round(float(geom[i, 0]) * 200), round(float(geom[i, 4])))
            if key not in best or float(conf[i]) > best[key][0]:
                best[key] = (float(conf[i]), float(geom[i, 0]), float(geom[i, 4]), int(row[i]), int(name[7:]))
    digits = [d[1:] for d in best.values()]
    if not digits:
        return None
    first_row = min(d[2] for d in digits)
    digits = [d for d in digits if d[2] == first_row]
    x0 = min(d[0] for d in digits)
    digits = [d for d in digits if d[0] <= x0 + 0.03]  # one signature, not a later change
    upper = "".join(str(d[3]) for d in sorted(d for d in digits if d[1] < 4))
    lower = "".join(str(d[3]) for d in sorted(d for d in digits if d[1] >= 4))
    if not upper or not lower:
        return None
    count, unit = int(upper), int(lower)
    if not 1 <= count < METER_COUNTS or unit not in METER_UNITS[1:]:
        return None
    return (count, unit)


def assign_meters(records: list, class_list: list) -> None:
    """Each record's ``meter`` (count, unit index): the last one printed at
    or before its system in the score (systems before the first printed meter
    take that one), (0, 0) when none is found."""
    by_score: dict = {}
    for r in records:
        by_score.setdefault(r["score"], []).append(r)
    for systems in by_score.values():
        systems.sort(key=lambda r: (r["page"], r["order"]))
        printed = [system_meter(r["cls"], r["geom"], r["row"], r["conf"], class_list) for r in systems]
        current = next((m for m in printed if m is not None), None)
        for r, m in zip(systems, printed, strict=True):
            current = m or current
            r["meter"] = (current[0], METER_UNITS.index(current[1])) if current else (0, 0)


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
        synth_list = [is_synthesized(page_nodes[i]) for i in local]
        x = node_features(
            wrapper.gnn, roi, rel / geo["canvas_w"], lab,
            scores=torch.tensor([float(page_nodes[i].get("score", 1.0)) for i in local], device=device),
            synthesized=torch.tensor([float(s) for s in synth_list], device=device),
        )
        cand = generate_axis_aware_edges(rel, lab, class_to_idx)
        h = wrapper.gnn.node_states(x, cand) if cand.shape[1] else torch.zeros_like(roi)

        cx, cy = (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0
        unsynth = ~torch.tensor(synth_list, dtype=torch.bool, device=device)
        if not unsynth.any():
            unsynth = torch.ones_like(unsynth)
        nx1, nx2 = float(cx[unsynth].min()), float(cx[unsynth].max())
        sw, sh = max(nx2 - nx1, 1e-3), max(sy2 - sy1, 1e-3)
        row, steps = _row_geometry(b.cpu(), lab.cpu(), class_to_idx["system-staff"])
        geom = torch.stack([
            (cx - nx1) / sw, (cy - sy1) / sh, (b[:, 2] - b[:, 0]) / sw, (b[:, 3] - b[:, 1]) / sh,
            steps.to(device),
        ], dim=1)
        page_box = canvas_to_page(torch.tensor([[sx1, sy1, sx2, sy2]]), geo)[0].tolist()
        span_box = canvas_to_page(torch.tensor([[nx1, sy1, nx2, sy2]]), geo)[0].tolist()
        records.append({
            "system_box": page_box,
            "x_span": [span_box[0], span_box[2]],
            "cls": lab.cpu().short(),
            "geom": geom.cpu().half(),
            "row": row.char(),
            "meas": _measure_of(b.cpu(), lab.cpu(), class_to_idx["measure"]).short(),
            "conf": torch.tensor([float(page_nodes[i].get("score", 1.0)) for i in local]).half(),
            "synth": torch.tensor(synth_list),
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


def augment(data_dir: Path, class_list: list) -> None:
    """Add ``cand`` and ``meter`` to the records of a dataset built before
    them (every shard of the dir in memory: a score's systems may span two
    shards)."""
    shards = sorted(data_dir.glob("shard_*.pt"))
    loaded = [torch.load(s, weights_only=False) for s in shards]
    records = [r for shard in loaded for r in shard]
    for r in records:
        r["cand"] = candidate_pitches(r["cls"], r["geom"], r["row"], class_list)
    assign_meters(records, class_list)
    for path, shard in zip(shards, loaded, strict=True):
        tmp = path.with_suffix(".tmp")
        torch.save(shard, tmp)
        tmp.replace(path)
    found = sum(r["meter"][0] > 0 for r in records)
    logger.info(f"{data_dir}: {len(records)} systems augmented, meter found for {found}")


def main():
    parser = argparse.ArgumentParser(description="Build scoregrid decoder data from the detection path.")
    parser.add_argument("--data", type=str, help="Dataset dir (imgs/, annotations/, mei/)")
    parser.add_argument("--out", type=str)
    parser.add_argument("--augment", type=str, nargs="+", default=None,
                        help="Instead: add candidate pitches and meters to these built dirs")
    parser.add_argument("--checkpoint", type=str,
                        help="Joint / GNN run: its detector detects, its GNN gives the node states")
    parser.add_argument("--detector-checkpoint", type=str, default=None,
                        help="Another detector (default: the checkpoint's own)")
    parser.add_argument("--config", type=str, default="conf-lite/config.json")
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0, help="Only the first N scores (smoke tests)")
    args = parser.parse_args()
    class_list = json.load(open(args.config))["target_classes"]
    if args.augment:
        for data_dir in args.augment:
            augment(Path(data_dir), class_list)
        return

    # First party imports
    from kakigori.vision.utils import load_detector

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
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
        order, score_records = 0, []
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
                if gt_index >= 0:
                    target_grid = grid_from_measures(
                        header["system_measures"][gt_index], header["staves"],
                        header["ann_by_id"], tuple(record["x_span"]),
                    )
                    target, numbers = target_grid.arrays(), target_grid.numbers
                else:
                    target, numbers = None, []
                record.update(score=score, page=page_idx, order=order, gt_index=gt_index,
                              target=target, numbers=numbers)
                record["cand"] = candidate_pitches(record["cls"], record["geom"], record["row"], class_list)
                order += 1
                stats["systems"] += 1
                stats["matched"] += gt_index >= 0
                score_records.append(record)
        assign_meters(score_records, class_list)
        shard.extend(score_records)
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
