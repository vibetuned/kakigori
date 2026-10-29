"""Compare decoded **kern runs without the measure-numbering cascade: kern F1
counts notes per numbered measure, so one missing or extra barline shifts
every later measure of the score and roughly halves that file's score. Per
run: how many files have the ground truth's measure count, and F1 on the files
where every run has it (the content, barline luck aside). Single and grand
staff apart.

Usage:
  uv run python scripts/scoregrid/paired_f1.py \\
      balanced=checkpoints_scoregrid/gnn-balanced/krn-val-k1 \\
      cand-drop=checkpoints_scoregrid/balanced-cand-drop/krn-val-k1
"""
import re
import json
import argparse
from pathlib import Path

from musicset.evaluate import compare_file, output_notes


def barlines(krn: Path) -> int:
    return sum(1 for line in krn.read_text().splitlines() if line.startswith("="))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("runs", nargs="+", help="NAME=KRN_DIR")
    parser.add_argument("--data", default="data/validation-lite", help="Dataset with mei/ and kern/")
    parser.add_argument("--index", default="data/scoregrid/validation-lite/index.json")
    parser.add_argument("--real-only", action="store_true",
                        help="Skip synthetic scores (kind_<index> names, e.g. validation-lite2's)")
    args = parser.parse_args()
    data = Path(args.data)
    staves = {s: v["staves"] for s, v in json.loads(Path(args.index).read_text())["scores"].items()
              if not (args.real_only and re.match(r"[a-z]+_\d", s))}
    runs = [(name, Path(d)) for name, d in (a.split("=", 1) for a in args.runs)]
    truth = {s: barlines(data / "kern" / f"{s}.krn") for s in staves}
    right = {name: {s for s in staves if (d / f"{s}.krn").exists() and barlines(d / f"{s}.krn") == truth[s]}
             for name, d in runs}
    for n, label in ((1, "single"), (2, "grand")):
        files = [s for s in staves if staves[s] == n]
        common = [s for s in files if all(s in right[name] for name, _ in runs)]
        print(f"{label}: {len(files)} files, {len(common)} with the right measure count in every run")
        for name, d in runs:
            pt = pm = out = 0
            for s in common:
                p, a, _, _ = compare_file(data / "mei" / f"{s}.mei", d / f"{s}.krn")
                pt, pm, out = pt + p, pm + a, out + output_notes(d / f"{s}.krn")
            rec, prec = 100 * pm / max(1, pt), 100 * pm / max(1, out)
            f1 = 2 * rec * prec / max(1e-9, rec + prec)
            print(f"  {name:14s} right count {sum(s in right[name] for s in files):3d}/{len(files)} | "
                  f"on common files: recall {rec:5.1f} precision {prec:5.1f} F1 {f1:5.1f}")


if __name__ == "__main__":
    main()
