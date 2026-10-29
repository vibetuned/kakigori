"""Boxes per class of a label set in detection datasets: which classes a
build lacks or barely has, side by side across sets.

Usage: uv run python scripts/class_coverage.py [--config conf-lite/config.json] \\
           data/train-lite2 data/synthetic-lite2 data/validation-lite2 [--below 200]
"""
# Standard library imports
import json
import argparse
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor


def count(path: Path) -> Counter:
    return Counter(a["class"] for a in json.loads(path.read_text())["annotations"])


def boxes(dataset: Path) -> Counter:
    total = Counter()
    with ProcessPoolExecutor(8) as ex:
        for c in ex.map(count, sorted((dataset / "annotations").glob("*.json")), chunksize=64):
            total += c
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--config", type=Path, default=Path("conf-lite/config.json"))
    parser.add_argument("--below", type=int, default=200, help="flag classes under this in a set")
    args = parser.parse_args()
    classes = json.loads(args.config.read_text())["target_classes"]
    counts = {d.name: boxes(d) for d in args.datasets}
    names = list(counts)
    print(f"{'class':24s}" + "".join(f"{n:>20s}" for n in names))
    for cls in classes:
        row = [counts[n][cls] for n in names]
        flag = " <-- missing" if 0 in row else " <-- rare" if min(row) < args.below else ""
        print(f"{cls:24s}" + "".join(f"{v:20d}" for v in row) + flag)
    for n in names:
        extra = sorted(set(counts[n]) - set(classes))
        if extra:
            print(f"{n}: classes outside {args.config}: {extra}")


if __name__ == "__main__":
    main()
