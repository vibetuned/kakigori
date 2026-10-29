"""Staff rows placed by bbox repair (rule 7) against the ground-truth rows,
grouped by the clef that placed them: top and bottom errors in staff spaces
(one pitch step = half a space), the rows the candidate pitches are read from.
"<clef> (G height)" simulates sizing a non-G row with the page's median
G-clef row height instead.

Usage:
  uv run python scripts/scoregrid/row_check.py --detector checkpoints_gnn_lite_896/run_003 \\
      --data data/synthetic-grand-lite data/validation-lite --pages 250
"""
import json
import random
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image

from kakigori.graph.infer import detect_page_nodes
from kakigori.vision.utils import load_detector
from kakigori.graph.bbox_repair import CLEF_ROW_CONSTANTS, repair_page_boxes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--detector", default="checkpoints_gnn_lite_896/run_003")
    parser.add_argument("--config", default="conf-lite/config.json")
    parser.add_argument("--data", nargs="+", default=["data/synthetic-grand-lite", "data/validation-lite"])
    parser.add_argument("--pages", type=int, default=250, help="Random pages per dataset")
    args = parser.parse_args()
    class_list = json.load(open(args.config))["target_classes"]
    dev = torch.device("cuda")
    det = load_detector(args.detector, len(class_list), dev)
    pages = []
    for d in args.data:
        imgs = sorted(Path(d, "imgs").glob("*.png"))
        random.Random(0).shuffle(imgs)
        pages += [(Path(d), p) for p in imgs[:args.pages]]
    err = defaultdict(list)
    for d, img in pages:
        nodes = detect_page_nodes(det, Image.open(img).convert("RGB"), 0, class_list, dev, None, 0.3, 0.5)
        repair_page_boxes(nodes, page_prefix="p0")
        annotations = json.loads((d / "annotations" / f"{img.stem}.json").read_text())["annotations"]
        gt_rows = [a["bbox"] for a in annotations if a["class"] == "system-staff"]
        if not gt_rows:
            continue
        clefs = [n for n in nodes if n["class"] in CLEF_ROW_CONSTANTS]
        rows = [n for n in nodes if n["class"] == "system-staff"]

        def clef_of(row):
            on = [c for c in clefs if row["bbox"][1] - 5 <= c["cy"] <= row["bbox"][3] + 5]
            return min(on, key=lambda c: c["cx"])["class"] if on else None

        g_heights = sorted(r["bbox"][3] - r["bbox"][1] for r in rows if clef_of(r) == "clefG")
        page_h = g_heights[len(g_heights) // 2] if g_heights else None
        for row in rows:
            clef = clef_of(row)
            if clef is None:
                continue
            _, y1, _, y2 = row["bbox"]
            gt = min(gt_rows, key=lambda g: abs((g[1] + g[3]) / 2 - (y1 + y2) / 2))
            space = (gt[3] - gt[1]) / 4
            if abs((gt[1] + gt[3]) / 2 - (y1 + y2) / 2) > 2 * space:
                continue
            err[clef].append(((y1 - gt[1]) / space, (y2 - gt[3]) / space))
            if page_h is not None and clef != "clefG":
                err[clef + " (G height)"].append(((y1 - gt[1]) / space, (y1 + page_h - gt[3]) / space))
    p90 = lambda v: sorted(abs(x) for x in v)[int(0.9 * len(v))]
    print("clef        rows  top error (median, |p90|)   bottom error   within 1/4 space (top & bottom)")
    for clef, e in sorted(err.items(), key=lambda kv: -len(kv[1])):
        top, bot = [a for a, _ in e], [b for _, b in e]
        ok = sum(abs(a) <= 0.25 and abs(b) <= 0.25 for a, b in e) / len(e)
        print(f"{clef:10s} {len(e):5d}   {statistics.median(top):+.2f} ({p90(top):.2f})            "
              f"{statistics.median(bot):+.2f} ({p90(bot):.2f})     {100 * ok:.0f}%")


if __name__ == "__main__":
    main()
