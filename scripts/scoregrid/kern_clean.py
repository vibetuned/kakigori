"""How clean is a directory of **kern? Per file: can Verovio engrave it; per
measure: do all **kern spines carry the same total duration, and does it fill
the meter (the *M of the header; a lone rest counts as a full-measure rest; the
first measure may be a pickup).

Usage:
  uv run python scripts/scoregrid/kern_clean.py \\
      ground-truth=data/validation-lite/kern balanced=checkpoints_scoregrid/gnn-balanced/krn-val-k1
"""
import re
import sys
import subprocess
from fractions import Fraction
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

ENGRAVE = r"""
import sys, verovio
tk = verovio.toolkit()
tk.setOptions({"adjustPageHeight": True})
ok = tk.loadData(open(sys.argv[1]).read())
print("OK" if ok and tk.getPageCount() > 0 and tk.renderToSVG(1) else "FAIL")
"""


def engraves(path: Path) -> bool:
    try:
        run = subprocess.run([sys.executable, "-c", ENGRAVE, str(path)], capture_output=True, text=True, timeout=60)
        return run.stdout.strip().endswith("OK")
    except subprocess.TimeoutExpired:
        return False


def token_quarters(tok: str):
    """Duration of a kern token in quarters (None: null, grace or no duration)."""
    if tok in (".", "") or tok.startswith(("*", "!", "=")):
        return None
    sub = tok.split(" ")[0]
    if "q" in sub:
        return None
    m = re.match(r"[^\d]*(\d+)(\.*)", sub)
    if not m:
        return None
    n = int(m.group(1))
    base = Fraction(8) if n == 0 else Fraction(4, n)
    dots = sub.count(".")  # kern allows the dot after the pitch (8g.), as compare-kern counts
    return base * (2 - Fraction(1, 2 ** dots))


def measures(path: Path):
    meter, spines, cur, out, kern_cols = None, 0, None, [], []
    for line in path.read_text().splitlines():
        if not line or line.startswith("!"):
            continue
        cells = line.split("\t")
        if line.startswith("**"):
            # only the **kern spines carry durations (the ground truth adds **dynam)
            kern_cols = [i for i, c in enumerate(cells) if c == "**kern"]
            spines = len(kern_cols)
            continue
        cells = [cells[i] for i in kern_cols if i < len(cells)]
        m = re.match(r"\*M(\d+)/(\d+)", cells[0])
        if m:
            meter = Fraction(4 * int(m.group(1)), int(m.group(2)))
            continue
        if line.startswith("*"):
            continue
        if line.startswith("="):
            if cur is not None:
                out.append(cur)
            cur = {"sums": [Fraction(0)] * spines, "events": [0] * spines, "rests": [0] * spines}
            continue
        if cur is None:
            cur = {"sums": [Fraction(0)] * spines, "events": [0] * spines, "rests": [0] * spines}
        for i, tok in enumerate(cells[:spines]):
            q = token_quarters(tok)
            if q is not None:
                cur["sums"][i] += q
                cur["events"][i] += 1
                cur["rests"][i] += "r" in tok
    return meter, out


def main():
    stats = {}
    for name, d in [a.split("=", 1) for a in sys.argv[1:]]:
        files = sorted(Path(d).glob("*.krn"))
        with ThreadPoolExecutor(16) as pool:
            ok = list(pool.map(engraves, files))
        agree = fill = total = 0
        for f in files:
            meter, ms = measures(f)
            for i, m in enumerate(ms):
                # a spine whose only event is a rest is a full-measure rest
                sums = [s for s, e, r in zip(m["sums"], m["events"], m["rests"]) if not (e == 1 and r == 1)]
                if not sums:
                    agree += 1
                    fill += 1
                    total += 1
                    continue
                total += 1
                same = len(set(sums)) == 1
                agree += same
                # first measure may be a pickup: spines must agree, need not fill
                fill += same and (meter is None or sums[0] == meter or i == 0)
        stats[name] = {
            "files": len(files),
            "Verovio engraves": f"{100 * sum(ok) / max(1, len(files)):.1f}%",
            "measures": total,
            "spines agree": f"{100 * agree / max(1, total):.1f}%",
            "fills the meter": f"{100 * fill / max(1, total):.1f}%",
        }
    for name, s in stats.items():
        print(f"{name:22s} " + " | ".join(f"{k} {v}" for k, v in s.items()))


if __name__ == "__main__":
    main()
