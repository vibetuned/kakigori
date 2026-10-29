"""compare-kern metrics of decoded **kern directories, split by the score's
staff count (single vs grand staff) or, on single-staff scores, by the first
clef of the ground-truth MEI.

Usage:
  uv run python scripts/scoregrid/kern_split.py [--by staves|clef] \\
      balanced=checkpoints_scoregrid/gnn-balanced/krn-val-k1 \\
      cand=checkpoints_scoregrid/balanced-cand/krn-val-k1
"""
import re
import json
import argparse
import statistics
from pathlib import Path
from collections import defaultdict

from musicset.evaluate import compare_file, output_notes


def first_clef(mei: Path) -> str:
    """G2, F4, G2v8, C3, ...: shape, line and octave displacement."""
    m = re.search(r'<(?:staffDef|clef)\b[^>]*?(?:clef\.)?shape="(\w)"[^>]*', mei.read_text())
    if not m:
        return "?"
    tag = m.group(0)
    line = re.search(r'(?:clef\.)?line="(\d)"', tag)
    dis = re.search(r'(?:clef\.)?dis="(\d+)"', tag)
    place = re.search(r'(?:clef\.)?dis\.place="(\w+)"', tag)
    name = m.group(1) + (line.group(1) if line else "")
    if dis:
        name += ("^" if place and place.group(1) == "above" else "v") + dis.group(1)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="NAME=KRN_DIR")
    parser.add_argument("--by", choices=("staves", "clef"), default="staves")
    parser.add_argument("--data", default="data/validation-lite", help="Dataset with mei/")
    parser.add_argument("--index", default="data/scoregrid/validation-lite/index.json")
    parser.add_argument("--real-only", action="store_true",
                        help="Skip synthetic scores (kind_<index> names, e.g. validation-lite2's)")
    args = parser.parse_args()
    mei_dir = Path(args.data) / "mei"
    staves = {s: v["staves"] for s, v in json.loads(Path(args.index).read_text())["scores"].items()
              if not (args.real_only and re.match(r"[a-z]+_\d", s))}
    if args.by == "staves":
        groups = {s: "single" if k == 1 else "grand" for s, k in staves.items()}
    else:
        groups = {s: first_clef(mei_dir / f"{s}.mei") for s, k in staves.items() if k == 1}
    names = sorted(set(groups.values()), key=lambda g: -sum(v == g for v in groups.values()))
    print(f"scores: " + ", ".join(f"{sum(v == g for v in groups.values())} {g}" for g in names))
    for name, d in (a.split("=", 1) for a in args.runs):
        totals = defaultdict(lambda: [0, 0, 0, 0, 0])
        files, missing = defaultdict(list), defaultdict(int)
        for score, group in groups.items():
            krn = Path(d) / f"{score}.krn"
            if not krn.exists():
                missing[group] += 1
                continue
            p, a, r, b = compare_file(mei_dir / f"{score}.mei", krn)
            t = totals[group]
            for i, v in enumerate((p, a, r, b, output_notes(krn))):
                t[i] += v
            if p:
                files[group].append(100 * a / p)
        for group in names:
            pt, pm, rt, rm, out = totals[group]
            rec, prec = 100 * pm / max(1, pt), 100 * pm / max(1, out)
            f1 = 2 * rec * prec / max(1e-9, rec + prec)
            median = statistics.median(files[group]) if files[group] else 0
            print(f"{name:20s} {group:8s} recall {rec:5.1f} rhythm {100 * rm / max(1, rt):5.1f} "
                  f"precision {prec:5.1f} F1 {f1:5.1f} notes out {out / max(1, pt):.2f}x "
                  f"median {median:5.1f} ({pt} notes, {missing[group]} without kern)")


if __name__ == "__main__":
    main()
