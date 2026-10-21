"""Small symlinked copies of a training and a validation set, for a smoke
test of the whole training queue on this machine before pushing a change
(the recipe is in pipeline-full.sh): N scores each, every page of a score,
with their annotations and graphs, and the validation set's MEI and kern.

Usage: uv run python scripts/smoke_sets.py OUT_DIR [TRAIN_SET] [VAL_SET]
Creates OUT_DIR/train and OUT_DIR/val.
"""
# Standard library imports
import os
import sys
import json
import random
from pathlib import Path

TRAIN_SCORES, VAL_SCORES = 30, 6


def subset(src: Path, dst: Path, n: int, extra: tuple = ()) -> None:
    graphs = sorted((src / "graphs").glob("*.json"))
    random.Random(1).shuffle(graphs)
    for sub in ("imgs", "annotations", "graphs", *extra):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    taken = 0
    for graph in graphs:
        stems = [name.removesuffix(".json") for name in json.loads(graph.read_text())["json_files"]]
        needed = [src / "imgs" / f"{s}.png" for s in stems]
        needed += [src / sub / f"{graph.stem}{'.mei' if sub == 'mei' else '.krn'}" for sub in extra]
        if not all(p.exists() for p in needed):
            continue
        os.symlink(graph.resolve(), dst / "graphs" / graph.name)
        for s in stems:
            os.symlink((src / "imgs" / f"{s}.png").resolve(), dst / "imgs" / f"{s}.png")
            os.symlink((src / "annotations" / f"{s}.json").resolve(), dst / "annotations" / f"{s}.json")
        for sub in extra:
            name = f"{graph.stem}{'.mei' if sub == 'mei' else '.krn'}"
            os.symlink((src / sub / name).resolve(), dst / sub / name)
        taken += 1
        if taken == n:
            break
    print(f"{dst}: {taken} scores, {len(list((dst / 'imgs').iterdir()))} pages")


def main() -> None:
    out = Path(sys.argv[1])
    train = Path(sys.argv[2] if len(sys.argv) > 2 else "data/train-lite")
    val = Path(sys.argv[3] if len(sys.argv) > 3 else "data/validation-lite")
    subset(train, out / "train", TRAIN_SCORES)
    subset(val, out / "val", VAL_SCORES, extra=("mei", "kern"))


if __name__ == "__main__":
    main()
