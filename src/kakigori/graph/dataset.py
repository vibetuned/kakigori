# Third party imports
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.data import Data, InMemoryDataset


class PianoStaffDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # List of your raw JSON/XML annotation files and MobileNet outputs
        return ["staff_001.json", "staff_002.json"]

    @property
    def processed_file_names(self):
        return ["piano_graphs.pt"]

    def process(self):
        data_list = []

        for raw_file in self.raw_file_names:
            # 1. Load your MobileNetV5 bounding boxes and classes
            boxes, classes = self._load_mobilenet_outputs(raw_file)

            # 2. Build the Node Feature tensor (x) and Position tensor (pos)
            x = self._encode_node_features(boxes, classes)
            pos = torch.tensor([[b.cx, b.cy] for b in boxes], dtype=torch.float)

            # 3. Generate Candidate Edges (e.g., connect each node to its 15 nearest neighbors)
            edge_index = knn_graph(pos, k=15, loop=False)

            # 4. Assign Ground Truth Labels (y) to the candidate edges based on your XML annotations
            y = self._assign_edge_labels(edge_index, raw_file)

            # 5. Construct the PyG Data object
            data = Data(x=x, pos=pos, edge_index=edge_index, y=y)
            data_list.append(data)

        self.save(data_list, self.processed_paths[0])


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
