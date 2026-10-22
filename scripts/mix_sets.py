"""A training set of symlinks: every score of BASE plus the scores listed in
LIST (one id per line, # comments) from EXTRA, with their pages,
annotations, graphs, MEI and kern. Used for the GNN's mixed training set
(conf-lite/gnn_mix_synthetic.txt: train-lite is 5% grand-staff pages, so
the GNN met piano scores mostly in validation). Rerunnable: existing links
stay.

Usage: uv run python scripts/mix_sets.py OUT BASE EXTRA LIST
   e.g. uv run python scripts/mix_sets.py data/gnn-mix-lite data/train-lite \\
            data/synthetic-lite conf-lite/gnn_mix_synthetic.txt
"""
# Standard library imports
import os
import sys
import json
from pathlib import Path

PER_PAGE = (("imgs", ".png"), ("annotations", ".json"))
PER_SCORE = (("graphs", ".json"), ("mei", ".mei"), ("kern", ".krn"))


def link(src: Path, dst: Path) -> None:
    if src.exists() and not dst.exists():
        os.symlink(src.resolve(), dst)


def add_score(src: Path, out: Path, score: str) -> int:
    """Link one score's files; returns its page count."""
    graph = src / "graphs" / f"{score}.json"
    if graph.exists():
        pages = [name.removesuffix(".json") for name in json.loads(graph.read_text())["json_files"]]
    else:
        pages = [p.stem for p in (src / "imgs").glob(f"{score}_page*.png")]
    for page in pages:
        for sub, ext in PER_PAGE:
            link(src / sub / f"{page}{ext}", out / sub / f"{page}{ext}")
    for sub, ext in PER_SCORE:
        link(src / sub / f"{score}{ext}", out / sub / f"{score}{ext}")
    return len(pages)


def main() -> None:
    out, base, extra, listing = (Path(a) for a in sys.argv[1:5])
    for sub, _ in PER_PAGE + PER_SCORE:
        (out / sub).mkdir(parents=True, exist_ok=True)
    base_scores = sorted(p.stem for p in (base / "mei").glob("*.mei"))
    extra_scores = [
        line.strip() for line in listing.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    pages = sum(add_score(base, out, s) for s in base_scores)
    extra_pages = sum(add_score(extra, out, s) for s in extra_scores)
    print(f"{out}: {len(base_scores)} scores ({pages} pages) from {base}, "
          f"{len(extra_scores)} scores ({extra_pages} pages) from {extra}")


if __name__ == "__main__":
    main()
