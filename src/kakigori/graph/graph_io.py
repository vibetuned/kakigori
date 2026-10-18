"""Ground-truth graph files.

musicset writes them as JSON (``musicset build --graph``); the old dataset
pipeline wrote ``.pt``. Both hold the same fields: ``node_ids``,
``num_nodes``, ``edge_index`` (2 x E), ``y`` (E edge classes), ``mei_file``
and ``json_files`` (the score's annotation files in page order).
"""

# Standard library imports
import json
from pathlib import Path

# Third party imports
import torch

GRAPH_SUFFIXES = (".json", ".pt")


def load_graph(path) -> dict:
    """A graph dict with ``edge_index`` and ``y`` as long tensors."""
    path = Path(path)
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        data["edge_index"] = torch.tensor(data["edge_index"], dtype=torch.long).reshape(2, -1)
        data["y"] = torch.tensor(data["y"], dtype=torch.long)
        return data
    return torch.load(path, weights_only=False)


def find_graph(graph_dir, stem: str):
    """The graph file of a score (``x``) or page (``x_page2``), or None."""
    graph_dir = Path(graph_dir)
    for name in dict.fromkeys((stem.split("_page")[0], stem)):
        for suffix in GRAPH_SUFFIXES:
            path = graph_dir / f"{name}{suffix}"
            if path.exists():
                return path
    return None


def graph_files(graph_dir) -> list:
    """Every graph in a directory, one per score (JSON preferred), sorted."""
    by_stem = {}
    for suffix in reversed(GRAPH_SUFFIXES):
        for path in Path(graph_dir).glob(f"*{suffix}"):
            by_stem[path.stem] = path
    return [by_stem[stem] for stem in sorted(by_stem)]
