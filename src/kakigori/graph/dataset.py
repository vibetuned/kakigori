# Standard library imports
import json
from pathlib import Path

# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class OMRFullPageDataset(Dataset):
    def __init__(self, img_dir, json_dir, graph_dir, class_list):
        self.img_dir = Path(img_dir)
        self.json_dir = Path(json_dir)
        self.graph_dir = Path(graph_dir)
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}

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
        image = Image.open(sample["img"]).convert("RGB")
        img_tensor = TF.to_tensor(image)  # Shape: (3, H, W)

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

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Format edges as (E, 3) where columns are [u_idx, v_idx, edge_class]
        # This is the exact format expected by our `split_into_systems` function
        if edge_index.numel() > 0:
            edges_tensor = torch.cat([edge_index, edge_labels.unsqueeze(0)], dim=0).t()
        else:
            edges_tensor = torch.empty((0, 3), dtype=torch.long)

        return {
            "image": img_tensor,
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "edges": edges_tensor,
            "file_name": sample["img"].name,
        }
