"""What a musicset build holds: scores and pages by staff count (single /
grand staff) and by source kind (synthetic generators), why scores were
rejected, and how well each score's kern reads back against its MEI.

Usage: uv run python scripts/dataset_summary.py data/train-lite2 [data/...]
"""
# Standard library imports
import re
import sys
import json
from pathlib import Path
from collections import Counter

STAFF_DEF = re.compile(r"<staffDef\b")


def kind(score: str) -> str:
    m = re.match(r"([a-z]+)_\d", score)
    return m.group(1) if m else "real"


def staves(mei: Path) -> int:
    text = mei.read_text(encoding="utf-8")
    first = text.find("</scoreDef>")
    return len(STAFF_DEF.findall(text[:first if first > 0 else None]))


def summarize(root: Path) -> None:
    manifests = [json.loads(p.read_text()) for p in sorted((root / "manifest").glob("*.json"))]
    by = Counter()
    pages = Counter()
    quality = []
    for m in manifests:
        n = staves(root / "mei" / f"{m['score']}.mei")
        key = (kind(m["score"]), "single" if n == 1 else "grand" if n == 2 else f"{n} staves")
        by[key] += 1
        pages[key] += len(m["pages"])
        if "kern_quality" in m:
            quality.append(m["kern_quality"])
    print(f"== {root}: {len(manifests)} scores, {sum(pages.values())} pages")
    for (k, s), n in sorted(by.items()):
        print(f"  {k:9s} {s:7s} {n:6d} scores {pages[(k, s)]:6d} pages")
    for s in ("single", "grand"):
        total = sum(n for (_, ss), n in by.items() if ss == s)
        print(f"  {'all':9s} {s:7s} {total:6d} scores {sum(p for (_, ss), p in pages.items() if ss == s):6d} pages")
    if quality:
        perfect = sum(q["pitch"] == q["rhythm"] == q["precision"] == 100 for q in quality)
        mean = {k: sum(q[k] for q in quality) / len(quality) for k in ("pitch", "rhythm", "precision")}
        print(f"  kern vs MEI: mean pitch {mean['pitch']:.2f} rhythm {mean['rhythm']:.2f} "
              f"precision {mean['precision']:.2f}; {perfect} of {len(quality)} scores perfect")
    failures = root / "failures.jsonl"
    if failures.exists():
        reasons = Counter()
        for line in failures.read_text().splitlines():
            error = json.loads(line).get("error") or ""
            reason = re.sub(r"[\d.]+", "N", error.split(": ", 1)[-1])[:60]
            reasons[reason] += 1
        print(f"  rejected / failed: {sum(reasons.values())}")
        for reason, n in reasons.most_common(8):
            print(f"    {n:6d}  {reason}")


def main() -> None:
    for d in sys.argv[1:]:
        summarize(Path(d))


if __name__ == "__main__":
    main()
