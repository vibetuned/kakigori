# Third party imports
import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign

class GNNPhase2Model(nn.Module):
    """
    Wraps the Vision Backbone (frozen), RoI Extractor (frozen), 
    and Graph Network (training) into a single module.
    """
    def __init__(self, detector, roi_extractor, gnn):
        super().__init__()
        self.detector = detector
        self.roi_extractor = roi_extractor
        self.gnn = gnn
        
        # Phase 2: Strictly freeze the vision backbone and RoI extractor
        self.detector.eval()
        
        for p in self.detector.parameters():
            p.requires_grad = False
            
    def train(self, mode=True):
        """Ensure vision components stay in eval mode."""
        super().train(mode)
        self.detector.eval()
        
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



# Usage example:
# inferencer = OMRGraphInferencer("gnn.pth", "roi.pth", ["clef", "notehead", "stem", ...])
# kern_output = inferencer.process_system("system_01.png", "system_01.json")

# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class ScoreGraphReconstructor(nn.Module):
    def __init__(self, node_in_dim, num_classes, class_embed_dim=32, hidden_dim=256, num_edge_classes=5, num_heads=4, dropout=0.2):
        super().__init__()
        
        # 1. Class Embedding Layer
        # Maps the integer class ID (0 to num_classes-1) to a dense vector
        self.class_embedding = nn.Embedding(num_classes, class_embed_dim)

        # 2. GATv2 Message Passing Layers
        # We divide hidden_dim by num_heads so the concatenated output remains 'hidden_dim'
        self.conv1 = GATv2Conv(node_in_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        self.conv3 = GATv2Conv(hidden_dim, hidden_dim // num_heads, heads=num_heads, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # 3. The Edge Classifier (MLP Head)
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