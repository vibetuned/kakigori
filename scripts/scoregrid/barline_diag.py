"""Where a scoregrid checkpoint's barline errors are (one sampling step), on
the single- or two-staff systems of a scoregrid set: true barlines with no
predicted one within a bin (the system's first, last or a middle one), and
predicted ones with no true one within a bin (2-4 bins off: displaced; else
invented); each also split by whether a barline detection is within a bin
(a missed barline the detector also missed is the detector's).

Usage:
  uv run python scripts/scoregrid/barline_diag.py checkpoints_scoregrid/balanced-cand-drop/best.pt
"""
import json
import argparse
from pathlib import Path
from collections import Counter

import torch

from kakigori.scoregrid.grid import B_MAX
from kakigori.scoregrid.model import ScoreGridDecoder, sample
from kakigori.scoregrid.train import collate, to_device, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data", default="data/scoregrid/validation-lite")
    parser.add_argument("--staves", type=int, choices=(1, 2), default=1)
    parser.add_argument("--config", default="conf-lite/config.json")
    args = parser.parse_args()
    class_list = json.load(open(args.config))["target_classes"]
    is_barline = torch.tensor([c.startswith("barline") for c in class_list])
    dev = torch.device("cuda")
    recs = [r for r in load_records(Path(args.data)) if int(r["target"]["staff"].sum()) == args.staves]
    near = lambda x, ys, d=1: any(abs(x - y) <= d for y in ys)
    for ckpt in args.checkpoints:
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model = ScoreGridDecoder(**st["config"]).to(dev).eval()
        model.load_state_dict(st["model"])
        A = Counter()
        for i in range(0, len(recs), 32):
            b = to_device(collate(recs[i:i + 32]), dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _, bar, _ = sample(model, b, steps=1)
            detected = (is_barline[b["cls"].cpu()] & b["valid"].cpu())
            bins = (b["geom"][..., 0].float().cpu() * B_MAX).long().clamp(0, B_MAX - 1)
            for t, p, dm, db in zip((b["barline"] > 0).cpu(), (bar > 0).cpu(), detected, bins):
                true, pred, dets = (t.nonzero().flatten().tolist(), p.nonzero().flatten().tolist(),
                                    db[dm].tolist())
                A["true"] += len(true)
                for x in true:
                    if near(x, pred):
                        continue
                    A["missed " + ("last" if x == max(true) else "first" if x == min(true) else "middle")] += 1
                    A["missed, detected" if near(x, dets) else "missed, not detected"] += 1
                for y in pred:
                    if near(y, true):
                        continue
                    A["extra, displaced" if near(y, true, 4) else "extra, invented"] += 1
                    A["extra, on a detection" if near(y, dets) else "extra, no detection"] += 1
        print(f"{ckpt}: {A['true']} true barlines on {len(recs)} {args.staves}-staff systems")
        for kind in ("missed", "extra"):
            print(f"  {kind}: " + " | ".join(f"{k.split(kind)[1].strip(', ')} {v}"
                                            for k, v in sorted(A.items()) if k.startswith(kind)))


if __name__ == "__main__":
    main()
