# Standard library imports
import json
import random
from pathlib import Path

# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from kakigori.graph.graph_io import find_graph, load_graph
from kakigori.graph.utils import match_detections
from kakigori.vision.dataset import canvas_size, page_to_canvas, letterbox_image


def resolved_page_annotations(json_dir, page_files: list, page_file: str) -> list:
    """A page's annotations with the ids the graph builder gave them
    (musicset.views.graph.graph_nodes): the first annotation with an id keeps
    it, every later one sharing it gets the pseudo-id {id}_{class}_{n}, n the
    number of id'd annotations before it in the score's pages (build order,
    the graph's json_files)."""
    seen: set = set()
    count = 0
    for name in page_files:
        with open(Path(json_dir) / name, "r", encoding="utf-8") as f:
            annotations = json.load(f).get("annotations", [])
        for ann in annotations:
            if "id" not in ann:
                continue
            if ann["id"] in seen:
                ann["id"] = f"{ann['id']}_{ann['class']}_{count}"
            seen.add(ann["id"])
            count += 1
        if name == page_file:
            return annotations
    # a page the graph does not list (not built with it): raw ids
    with open(Path(json_dir) / page_file, "r", encoding="utf-8") as f:
        return json.load(f).get("annotations", [])


class OMRFullPageDataset(Dataset):
    """Pages as (image, nodes, edges): ground-truth boxes and graph edges, or
    (``det_dir``, with probability ``det_prob``) the detector's own boxes
    for the page (cache-detections) with the ground truth carried onto them:
    stage D trains the GNN on what it meets at inference.

    Detected boxes are matched to the page's ground-truth nodes per class
    (utils.match_detections); a matched box takes its ground-truth node's
    edges and keep target 1, an unmatched one (a false positive or a
    duplicate) has no edges, so every candidate touching it trains toward
    "no edge", and keep target 0. Items carry per-node ``scores`` and
    ``synthesized`` (ground truth: 1 and 0), ``keep`` (-1: no target) and
    ``from_detections``.
    """

    def __init__(self, img_dir, json_dir, graph_dir, class_list, input_size=640,
                 class_remap=None, det_dir=None, det_prob=0.0):
        self.class_list = list(class_list)
        self.det_dir = Path(det_dir) if det_dir else None
        self.det_prob = det_prob
        self.img_dir = Path(img_dir)
        self.json_dir = Path(json_dir)
        self.graph_dir = Path(graph_dir)
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        # Optional annotation-class -> config-class mapping (e.g. noise
        # classes merged into a single "sink" class). Remapped classes are
        # never graph nodes: they lose their edges and re-enter only as
        # isolated detection-supervised nodes.
        self.class_remap = class_remap or {}
        # The detector was trained on pages letterboxed into its canvas
        # (detector.input_size); feeding full-resolution pages would put its
        # features out of distribution (and cost ~15x the compute)
        self.input_size = canvas_size(input_size)

        # 1. Match the file triplets (Image, JSON, Graph)
        self.samples = []
        for img_path in sorted(self.img_dir.glob("*.png")):
            stem = img_path.stem
            json_path = self.json_dir / f"{stem}.json"

            # One graph per score ('score.json' for 'score_page1.png')
            graph_path = find_graph(self.graph_dir, stem)

            if json_path.exists() and graph_path is not None:
                self.samples.append(
                    {"img": img_path, "json": json_path, "graph": graph_path}
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # --- A. Load Vision Data ---
        # Letterbox into the detector's canvas, as the detector was trained
        canvas, geo = letterbox_image(Image.open(sample["img"]).convert("L"), self.input_size)
        img_tensor = TF.to_tensor(canvas)  # Shape: (1, height, width)

        # --- B. Load Relational Data ---
        graph_data = load_graph(sample["graph"])
        edge_index = graph_data["edge_index"]  # Shape: (2, E)
        edge_labels = graph_data["y"]  # Shape: (E)
        node_ids = graph_data["node_ids"]  # List of strings

        # --- C. Align Modalities ---
        # Graph node ids as the graph builder resolved them: annotations that
        # share an id (a note and its notehead, a single-staff system and its
        # row) give the first one the id and every later one a pseudo-id
        # {id}_{class}_{n}, n counted over the score's pages in build order.
        # Mapping raw ids instead (the last annotation won) gave a note's
        # graph node its notehead's box and class and dropped the notehead
        # and the system rows, so the GNN never saw a note anchor in training
        # while inference (validate-*, infer-omr) has both.
        page_files = graph_data.get("json_files") or [sample["json"].name]
        page_anns = resolved_page_annotations(self.json_dir, page_files, sample["json"].name)
        ann_map = {ann["id"]: ann for ann in page_anns if "id" in ann}

        boxes = []
        labels = []
        valid_node_indices = []

        # Strictly order the boxes and labels to match the graph's node_ids index!
        for i, node_id in enumerate(node_ids):
            if node_id in ann_map:
                ann = ann_map[node_id]
                cls = ann["class"]
                # Classes absent from the config (pruned) or remapped to a
                # noise class must not stay graph nodes: skipping them here
                # also drops their edges via the keep mask below.
                if cls not in self.class_to_idx or cls in self.class_remap:
                    continue
                boxes.append(ann["bbox"])
                labels.append(self.class_to_idx[cls])
                valid_node_indices.append(i)

        # Harmonize the node space with detection: append every annotation
        # that never became a graph node as an ISOLATED node — present in
        # the box/label set, referenced by no GT edge, so every candidate
        # touching it trains toward class 0. Without this, such classes
        # (a) get their neck features sacrificed during joint training
        # (barlineSingle 0.99 -> 0.10: no edge-loss pressure protects them)
        # and (b) appear at inference (detections include them) despite
        # never being seen in edge training. ONLY annotations that are not
        # graph nodes qualify (with resolved ids, a note anchor IS one: its
        # edges must be learned, not suppressed). Appended AFTER the
        # graph-matched nodes so edge index
        # remapping is untouched. Membership is tested against the KEPT
        # graph nodes (not raw node_ids) so classes dropped from the graph
        # by class_remap fall through to isolation as their remapped class.
        kept_node_ids = {node_ids[i] for i in valid_node_indices}
        for ann in page_anns:
            cls = self.class_remap.get(ann.get("class"), ann.get("class"))
            if cls not in self.class_to_idx:
                continue
            if ann.get("id") in kept_node_ids:
                continue
            boxes.append(ann["bbox"])
            labels.append(self.class_to_idx[cls])

        # Remap boxes into letterboxed canvas coordinates
        if boxes:
            boxes_tensor = page_to_canvas(torch.tensor(boxes, dtype=torch.float32), geo)
        else:
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Format edges as (E, 3) where columns are [u_idx, v_idx, edge_class].
        # Graph node indices MUST be remapped onto the filtered box ordering:
        # nodes without an annotation bbox are dropped above, which shifts
        # every index after them — using raw graph indices silently pairs
        # edges with the wrong boxes.
        if edge_index.numel() > 0 and valid_node_indices:
            remap = torch.full((len(node_ids),), -1, dtype=torch.long)
            remap[torch.tensor(valid_node_indices, dtype=torch.long)] = torch.arange(
                len(valid_node_indices)
            )
            u = remap[edge_index[0]]
            v = remap[edge_index[1]]
            keep = (u >= 0) & (v >= 0)
            edges_tensor = torch.stack(
                [u[keep], v[keep], edge_labels[keep].long()], dim=1
            )
        else:
            edges_tensor = torch.empty((0, 3), dtype=torch.long)

        # Full annotation set for DETECTION supervision (joint training).
        # The graph-filtered boxes above exclude classes that never become
        # graph nodes (note anchors, barlines): supervising detection on
        # them actively unlearned those classes (note AP 0.93 -> 0.01).
        det_boxes, det_labels = [], []
        for ann in page_anns:
            cls = self.class_remap.get(ann.get("class"), ann.get("class"))
            if cls in self.class_to_idx:
                det_boxes.append(ann["bbox"])
                det_labels.append(self.class_to_idx[cls])
        if det_boxes:
            det_boxes_tensor = page_to_canvas(torch.tensor(det_boxes, dtype=torch.float32), geo)
        else:
            det_boxes_tensor = torch.empty((0, 4), dtype=torch.float32)

        n = len(labels_tensor)
        item = {
            "image": img_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "edges": edges_tensor,
            "det_boxes": det_boxes_tensor,
            "det_labels": torch.tensor(det_labels, dtype=torch.long),
            "file_name": sample["img"].name,
            "scores": torch.ones(n),
            "synthesized": torch.zeros(n),
            "keep": torch.full((n,), -1.0),
            "from_detections": False,
        }
        if self.det_dir is not None and self.det_prob > 0 and random.random() < self.det_prob:
            self._use_detections(item, sample["img"].stem, geo)
        return item

    def _use_detections(self, item: dict, stem: str, geo: dict) -> None:
        """Swap the item's nodes for the page's cached detections, the
        ground-truth edges and keep targets carried onto them."""
        path = self.det_dir / f"{stem}.json"
        if not path.exists():
            return
        nodes = [n for n in json.loads(path.read_text())["nodes"] if n["class"] in self.class_to_idx]
        if len(nodes) < 2:
            return
        boxes = page_to_canvas(torch.tensor([n["bbox"] for n in nodes], dtype=torch.float32), geo)
        labels = torch.tensor([self.class_to_idx[n["class"]] for n in nodes], dtype=torch.long)
        matched = match_detections(boxes, labels, item["boxes"], item["labels"], self.class_list)
        edges = [
            (matched[u], matched[v], c)
            for u, v, c in item["edges"].tolist()
            if u in matched and v in matched
        ]
        keep = torch.zeros(len(nodes))
        keep[list(matched.values())] = 1.0
        item.update(
            boxes=boxes,
            labels=labels,
            edges=torch.tensor(edges, dtype=torch.long).reshape(-1, 3),
            scores=torch.tensor([float(n["score"]) for n in nodes]),
            synthesized=torch.tensor([float(n.get("synthesized", False)) for n in nodes]),
            keep=keep,
            from_detections=True,
        )
