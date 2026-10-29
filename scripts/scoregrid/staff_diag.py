"""Per-staff decoding accuracy of scoregrid checkpoints (one sampling step) on
the single- or two-staff systems of a scoregrid set: barline count, notes in
the right bin (or put on the other staff) and, on those hits, pitch,
alteration and duration.

Usage:
  uv run python scripts/scoregrid/staff_diag.py --staves 2 \\
      checkpoints_scoregrid/gnn-balanced/best.pt checkpoints_scoregrid/balanced-cand/best.pt
"""
import argparse
from pathlib import Path
from collections import Counter

import torch

from kakigori.scoregrid.grid import NOTE
from kakigori.scoregrid.model import DUR, EVENT, ALTER0, PITCH0, ScoreGridDecoder, sample
from kakigori.scoregrid.train import collate, to_device, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+")
    parser.add_argument("--data", default="data/scoregrid/validation-lite")
    parser.add_argument("--staves", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    dev = torch.device("cuda")
    recs = [r for r in load_records(Path(args.data)) if int(r["target"]["staff"].sum()) == args.staves]
    for ckpt in args.checkpoints:
        st = torch.load(ckpt, map_location=dev, weights_only=False)
        model = ScoreGridDecoder(**st["config"]).to(dev).eval()
        model.load_state_dict(st["model"])
        A = Counter()
        for i in range(0, len(recs), 32):
            b = to_device(collate(recs[i:i + 32]), dev)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                cells, bar, _ = sample(model, b, steps=1)
            tc = b["cells"]
            te, pe = tc[..., EVENT], cells[..., EVENT]
            A["sys"] += b["cls"].shape[0]
            A["bar_ok"] += int((bar.sum(1) == b["barline"].sum(1)).sum())
            for s in range(args.staves):
                note = te[..., s] == NOTE
                hit = note & (pe[..., s] == NOTE)
                A[f"n{s}"] += int(note.sum())
                A[f"hit{s}"] += int(hit.sum())
                A[f"other{s}"] += int((note & (pe[..., s] != NOTE) & (pe[..., 1 - s] == NOTE)).sum())
                A[f"pitch{s}"] += int(((cells[..., s, PITCH0] == tc[..., s, PITCH0]) & hit).sum())
                A[f"alter{s}"] += int(((cells[..., s, ALTER0] == tc[..., s, ALTER0]) & hit).sum())
                A[f"dur{s}"] += int(((cells[..., s, DUR] == tc[..., s, DUR]) & hit).sum())
        pct = lambda a, b: f"{100 * A[a] / max(1, A[b]):.1f}%"
        print(f"{ckpt}: {A['sys']} {args.staves}-staff systems, barline count right {pct('bar_ok', 'sys')}")
        for s in range(args.staves):
            name = "staff" if args.staves == 1 else ("upper", "lower")[s]
            print(f"  {name}: notes in the right bin {pct(f'hit{s}', f'n{s}')} "
                  f"(on the other staff {pct(f'other{s}', f'n{s}')}), on hits: pitch {pct(f'pitch{s}', f'hit{s}')}, "
                  f"alteration {pct(f'alter{s}', f'hit{s}')}, duration {pct(f'dur{s}', f'hit{s}')}")


if __name__ == "__main__":
    main()
