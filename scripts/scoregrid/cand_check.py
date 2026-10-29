"""How often a target note's pitches equal the candidate pitches
(data.candidate_pitches) of the noteheads pooled into its cell, the same bin
on the same staff: the input the decoder can copy. Per staff of the single-
or two-staff systems of a scoregrid set, with the lowest pitch's offset
(in diatonic steps) when it is wrong.

Usage:
  uv run python scripts/scoregrid/cand_check.py --data data/scoregrid/synthetic-grand-lite --staves 2
"""
import argparse
from pathlib import Path
from collections import Counter

from kakigori.scoregrid.grid import NOTE, B_MAX, S_MAX
from kakigori.scoregrid.model import EVENT, ALTER0, PITCH0
from kakigori.scoregrid.train import collate, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", default="data/scoregrid/validation-lite")
    parser.add_argument("--staves", type=int, choices=(1, 2), default=1)
    parser.add_argument("--limit", type=int, default=3000, help="Systems to check (0: all)")
    args = parser.parse_args()
    recs = [r for r in load_records(Path(args.data)) if int(r["target"]["staff"].sum()) == args.staves]
    if args.limit:
        recs = recs[:args.limit]
    A = Counter()
    for i in range(0, len(recs), 64):
        b = collate(recs[i:i + 64])
        bins = (b["geom"][..., 0].float() * B_MAX).long().clamp(0, B_MAX - 1)
        staff = b["row"].clamp(0, S_MAX - 1)
        cand = b["cand"].long()
        for j in range(cand.shape[0]):
            tc = b["cells"][j]
            for s in range(args.staves):
                for x in (tc[:, s, EVENT] == NOTE).nonzero().flatten().tolist():
                    want = sorted(int(p) for p in tc[x, s, PITCH0:ALTER0] if p > 0)
                    here = b["valid"][j] & (bins[j] == x) & (staff[j] == s) & (cand[j] > 0)
                    got = sorted(set(int(c) for c in cand[j][here]))
                    A[s, "n"] += 1
                    if not got:
                        A[s, "none"] += 1
                        continue
                    A[s, "exact"] += got == want
                    if got[0] == want[0]:
                        A[s, "lowest"] += 1
                    else:
                        A[s, got[0] - want[0]] += 1
    for s in range(args.staves):
        n = max(1, A[s, "n"])
        offsets = sorted(((k, v) for (t, k), v in A.items() if t == s and isinstance(k, int)), key=lambda kv: -kv[1])
        name = "staff" if args.staves == 1 else ("upper", "lower")[s]
        print(f"{name}: {A[s, 'n']} notes | no candidate {100 * A[s, 'none'] / n:.1f}% | "
              f"candidates = pitches {100 * A[s, 'exact'] / n:.1f}% | lowest pitch agrees {100 * A[s, 'lowest'] / n:.1f}% | "
              f"lowest off by: " + ", ".join(f"{k:+d} {100 * v / n:.1f}%" for k, v in offsets[:6]))


if __name__ == "__main__":
    main()
