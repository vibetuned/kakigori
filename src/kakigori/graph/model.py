# Third party imports
import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign


class GraphVisualExtractor(nn.Module):
    def __init__(self, featmap_names=['0', '1', '2'], roi_size=(7, 7), in_channels=128, out_channels=256):
        super().__init__()
        self.roi_size = roi_size
        
        # 1. Replace standard roi_align with MultiScaleRoIAlign
        # This automatically maps boxes to the correct PANet scale (P3, P4, P5)
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=featmap_names,
            output_size=roi_size,
            sampling_ratio=2 # Standard practice to avoid aliasing during pooling
        )
        
        # Flattens the 7x7 RoI grid into a single 1D feature vector for the GNN Node
        self.fc = nn.Linear(in_channels * roi_size[0] * roi_size[1], out_channels)
        self.activation = nn.ELU()

    def forward(self, feature_maps_dict, normalized_boxes, image_size):
        """
        feature_maps_dict: dict mapping string keys (e.g., '0', '1', '2') to tensors 
                           from the ConvNeXt/DINOv3 PANetNeck
        normalized_boxes: list of tensors per batch item, shape (N, 4) -> [cx, cy, w, h]
        image_size: Tuple (height, width) of the input image to the network
        """
        abs_boxes_list = []
        
        for boxes in normalized_boxes:
            if boxes.numel() == 0:
                abs_boxes_list.append(torch.empty((0, 4), dtype=torch.float32, device=boxes.device))
                continue
            
            # 1. Denormalize boxes back to absolute pixel coordinates
            cx = boxes[:, 0] * image_size[1]
            cy = boxes[:, 1] * image_size[0]
            w  = boxes[:, 2] * image_size[1]
            h  = boxes[:, 3] * image_size[0]
            
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            
            # 2. Format as [x1, y1, x2, y2]. 
            # Notice we DON'T need the batch index anymore! MultiScaleRoIAlign handles 
            # that automatically based on the list structure.
            formatted_boxes = torch.stack([x1, y1, x2, y2], dim=1)
            abs_boxes_list.append(formatted_boxes)
        
        # Catch empty batches to prevent crashes
        if all(b.shape[0] == 0 for b in abs_boxes_list):
            # Grab the device from the first feature map in the dictionary
            first_map = next(iter(feature_maps_dict.values()))
            return torch.empty(0, self.fc.out_features, device=first_map.device)
            
        # MultiScaleRoIAlign needs to know the original image shape to calculate strides properly
        image_shapes = [image_size for _ in range(len(normalized_boxes))]
        
        # 3. Extract the visual features across all scales automatically
        extracted_rois = self.roi_align(
            feature_maps_dict, 
            abs_boxes_list, 
            image_shapes
        )
        
        # 4. Flatten and project to form the GNN Node feature 'x'
        flattened = extracted_rois.flatten(start_dim=1)
        node_features = self.activation(self.fc(flattened))
        
        return node_features


# Assuming 'fused_features' is the list [P3, P4, P5] from your PANetNeck
#features_dict = {str(i): feat for i, feat in enumerate(fused_features)}

# Now pass it to the new extractor
#node_visual_features = visual_extractor(features_dict, normalized_boxes, (640, 640))

# Standard library imports
import json
from pathlib import Path

# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch_geometric.data import Data

# Local folder imports
from .model import GraphVisualExtractor, ScoreGraphReconstructor
from .heuristics import generate_axis_aware_edges, generate_text_candidate_edges
from .serializer import HumdrumSerializer


class OMRGraphInferencer:
    def __init__(self, gnn_checkpoint, roi_extractor_checkpoint, class_list, device="cpu"):
        self.device = torch.device(device)
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}
        
        # 1. Load the RoI Extractor (The bridge from pixels to node features)
        self.roi_extractor = GraphVisualExtractor().to(self.device)
        self.roi_extractor.load_state_dict(torch.load(roi_extractor_checkpoint, map_location=self.device))
        self.roi_extractor.eval()
        
        # 2. Load the GATv2 Edge Classifier
        # Node input dim = RoI features (256) + BBox coords (4) + Class Embedding (e.g., 32)
        node_in_dim = 256 + 4 + 32 
        self.gnn = ScoreGraphReconstructor(node_in_dim=node_in_dim).to(self.device)
        self.gnn.load_state_dict(torch.load(gnn_checkpoint, map_location=self.device))
        self.gnn.eval()
        
        # 3. Class Embedding lookup (translates class ID into a trainable vector)
        self.class_embedding = torch.nn.Embedding(len(class_list), 32).to(self.device)

    def _build_pyg_data(self, image_tensor, annotations):
        """Converts raw JSON annotations and the image into a PyTorch Geometric graph."""
        nodes_meta = []
        boxes = []
        labels = []
        
        for i, ann in enumerate(annotations):
            boxes.append(ann["bbox"]) # Assuming absolute [x1, y1, x2, y2] relative to the system crop
            labels.append(self.class_to_idx[ann["class"]])
            nodes_meta.append({"id": i, "class": ann["class"], "bbox": ann["bbox"]})
            
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # --- A. Node Features (x) ---
        with torch.no_grad():
            # 1. Extract visual features from the raw image using the bounding boxes
            # We pass a dummy feature map here assuming RoI extractor handles a raw image directly for inference
            # (Or you pass the MobileNet backbone here if doing end-to-end)
            roi_features = self.roi_extractor(image_tensor.unsqueeze(0), [boxes_tensor])
            
            # 2. Get class embeddings
            class_embeds = self.class_embedding(labels_tensor)
            
            # 3. Concatenate to create the final node feature vector 'x'
            x = torch.cat([roi_features, boxes_tensor, class_embeds], dim=1)
            
        # --- B. Candidate Edges (edge_index) ---
        # Run the X/Y axis sorting and the Vertical Raycasting for text
        structural_edges = generate_axis_aware_edges(boxes_tensor, labels_tensor, self.class_to_idx)
        text_edges = generate_text_candidate_edges(boxes_tensor, labels_tensor, self.class_to_idx)
        
        if text_edges.numel() > 0:
            edge_index = torch.cat([structural_edges, text_edges], dim=1)
        else:
            edge_index = structural_edges
            
        return Data(x=x, edge_index=edge_index), nodes_meta

    @torch.inference_mode()
    def process_system(self, image_path, json_path):
        """Runs the GNN on a single system and outputs the **kern string."""
        # 1. Load Data
        image = Image.open(image_path).convert("RGB")
        image_tensor = TF.to_tensor(image).to(self.device)
        
        with open(json_path) as f:
            data = json.load(f)
        
        # 2. Build the Graph
        pyg_data, nodes_meta = self._build_pyg_data(image_tensor, data["annotations"])
        
        # 3. Predict Edges
        edge_logits = self.gnn(pyg_data.x, pyg_data.edge_index)
        edge_predictions = torch.argmax(edge_logits, dim=1) # 0, 1, 2, 3, or 4
        
        # 4. Serialize to Humdrum
        serializer = HumdrumSerializer(nodes_meta, pyg_data.edge_index, edge_predictions)
        kern_matrix = serializer.generate_kern_matrix()
        
        return kern_matrix

# Usage example:
# inferencer = OMRGraphInferencer("gnn.pth", "roi.pth", ["clef", "notehead", "stem", ...])
# kern_output = inferencer.process_system("system_01.png", "system_01.json")

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class ScoreGraphReconstructor(nn.Module):
    def __init__(self, node_in_dim, hidden_dim=256, num_edge_classes=5, num_heads=4, dropout=0.2):
        super().__init__()
        
        # 1. GATv2 Message Passing Layers
        # We divide hidden_dim by num_heads so the concatenated output remains 'hidden_dim'
        self.conv1 = GATv2Conv(node_in_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # 2. The Edge Classifier (MLP Head)
        # It takes the concatenated features of the Source Node and Destination Node (hidden_dim * 2)
        self.edge_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, num_edge_classes)
        )

    def forward(self, x, edge_index):
        """
        x: Node features (Number of Nodes, node_in_dim) -> RoI + BBox + Class
        edge_index: Candidate edges (2, Number of Edges) -> From your Axis-Aware heuristics
        """
        
        # --- PHASE 1: Contextualize the Nodes (Message Passing) ---
        # The nodes look at their neighbors and update their internal representations
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        
        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = self.dropout(h)
        
        h = self.conv3(h, edge_index)
        # Now 'h' contains deep, graph-aware features for every musical symbol
        
        # --- PHASE 2: Construct Edge Features ---
        # edge_index[0] contains the Source nodes (u)
        # edge_index[1] contains the Destination nodes (v)
        row, col = edge_index
        
        node_u = h[row] # Features of the source nodes (E, hidden_dim)
        node_v = h[col] # Features of the destination nodes (E, hidden_dim)
        
        # Concatenate them to represent the "bridge" between the two symbols
        edge_features = torch.cat([node_u, node_v], dim=1) # Shape: (E, hidden_dim * 2)
        
        # --- PHASE 3: Classify the Edges ---
        edge_logits = self.edge_classifier(edge_features) # Shape: (E, num_edge_classes)
        
        return edge_logits