# Standard library imports
import json
from pathlib import Path

# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class OMRFullPageDataset(Dataset):
    def __init__(self, img_dir, json_dir, graph_dir, class_list, input_size=640):
        self.img_dir = Path(img_dir)
        self.json_dir = Path(json_dir)
        self.graph_dir = Path(graph_dir)
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        # The detector was trained on letterboxed input_size images; feeding
        # full-resolution pages would put its features out of distribution
        # (and cost ~15x the compute)
        self.input_size = input_size

        # 1. Match the file triplets (Image, JSON, Graph)
        self.samples = []
        for img_path in sorted(self.img_dir.glob("*.png")):
            stem = img_path.stem
            json_path = self.json_dir / f"{stem}.json"

            # Handle the naming convention (e.g., 'score_page1' vs 'score')
            stem_no_page = stem.split("_page")[0]
            graph_path = self.graph_dir / f"{stem_no_page}.pt"
            if not graph_path.exists():
                graph_path = self.graph_dir / f"{stem}.pt"

            if json_path.exists() and graph_path.exists():
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
        image = Image.open(sample["img"]).convert("RGB")
        orig_w, orig_h = image.size
        scale = min(self.input_size / orig_w, self.input_size / orig_h)
        new_w, new_h = int(round(orig_w * scale)), int(round(orig_h * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (self.input_size, self.input_size), (114, 114, 114))
        pad_x = (self.input_size - new_w) // 2
        pad_y = (self.input_size - new_h) // 2
        canvas.paste(image, (pad_x, pad_y))
        img_tensor = TF.to_tensor(canvas)  # Shape: (3, input_size, input_size)

        # --- B. Load Relational Data ---
        graph_data = torch.load(sample["graph"], weights_only=False)
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
                boxes.append(ann["bbox"])
                labels.append(self.class_to_idx[ann["class"]])
                valid_node_indices.append(i)

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

        return {
            "image": img_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "edges": edges_tensor,
            "file_name": sample["img"].name,
        }
