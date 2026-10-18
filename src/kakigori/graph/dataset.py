# Standard library imports
import json
from pathlib import Path

# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from kakigori.graph.graph_io import find_graph, load_graph


class OMRFullPageDataset(Dataset):
    def __init__(self, img_dir, json_dir, graph_dir, class_list, input_size=640,
                 class_remap=None):
        self.img_dir = Path(img_dir)
        self.json_dir = Path(json_dir)
        self.graph_dir = Path(graph_dir)
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        # Optional annotation-class -> config-class mapping (e.g. noise
        # classes merged into a single "sink" class). Remapped classes are
        # never graph nodes: they lose their edges and re-enter only as
        # isolated detection-supervised nodes.
        self.class_remap = class_remap or {}
        # The detector was trained on letterboxed input_size images; feeding
        # full-resolution pages would put its features out of distribution
        # (and cost ~15x the compute)
        self.input_size = input_size

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
        # Letterbox to the detector's training resolution: aspect-preserving
        # resize on a grey canvas, same convention as vision/dataset._letterbox
        image = Image.open(sample["img"]).convert("L")
        orig_w, orig_h = image.size
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (self.input_size, self.input_size), 114)
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas.paste(image, (pad_x, pad_y))
        img_tensor = TF.to_tensor(canvas)  # Shape: (1, input_size, input_size)

        # --- B. Load Relational Data ---
        graph_data = load_graph(sample["graph"])
        edge_index = graph_data["edge_index"]  # Shape: (2, E)
        edge_labels = graph_data["y"]  # Shape: (E)
        node_ids = graph_data["node_ids"]  # List of strings

        # --- C. Align Modalities ---
        with open(sample["json"], "r") as f:
            json_data = json.load(f)

        # Create a fast lookup dict mapping xml:id -> bounding box annotation
        ann_map = {
            ann["id"]: ann for ann in json_data.get("annotations", []) if "id" in ann
        }

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
        # never being seen in edge training. ONLY classes whose ids never
        # appear in the graph node space qualify: 'note' shares its xml:id
        # with its notehead (the graph node exists, shadowed), and notes
        # MUST keep edge extrapolation at inference — teaching
        # note-labeled -> class 0 would suppress the serializer's event
        # anchors. Appended AFTER the graph-matched nodes so edge index
        # remapping is untouched. Membership is tested against the KEPT
        # graph nodes (not raw node_ids) so classes dropped from the graph
        # by class_remap fall through to isolation as their remapped class.
        kept_node_ids = {node_ids[i] for i in valid_node_indices}
        for ann in json_data.get("annotations", []):
            cls = self.class_remap.get(ann.get("class"), ann.get("class"))
            if cls not in self.class_to_idx:
                continue
            if ann.get("id") in kept_node_ids:
                continue
            boxes.append(ann["bbox"])
            labels.append(self.class_to_idx[cls])

        # Remap boxes into letterboxed canvas coordinates
        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            boxes_tensor[:, [0, 2]] = boxes_tensor[:, [0, 2]] * scale + pad_x
            boxes_tensor[:, [1, 3]] = boxes_tensor[:, [1, 3]] * scale + pad_y
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
        for ann in json_data.get("annotations", []):
            cls = self.class_remap.get(ann.get("class"), ann.get("class"))
            if cls in self.class_to_idx:
                det_boxes.append(ann["bbox"])
                det_labels.append(self.class_to_idx[cls])
        if det_boxes:
            det_boxes_tensor = torch.tensor(det_boxes, dtype=torch.float32)
            det_boxes_tensor[:, [0, 2]] = det_boxes_tensor[:, [0, 2]] * scale + pad_x
            det_boxes_tensor[:, [1, 3]] = det_boxes_tensor[:, [1, 3]] * scale + pad_y
        else:
            det_boxes_tensor = torch.empty((0, 4), dtype=torch.float32)

        return {
            "image": img_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "edges": edges_tensor,
            "det_boxes": det_boxes_tensor,
            "det_labels": torch.tensor(det_labels, dtype=torch.long),
            "file_name": sample["img"].name,
        }
