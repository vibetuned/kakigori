# Third party imports
import torch
import torch.nn as nn
from torchvision.ops import MultiScaleRoIAlign

class GNNPhase2Model(nn.Module):
    """
    Wraps the Vision Backbone, RoI Extractor, and Graph Network into a
    single module. The vision side is frozen by default; `unfreeze` opens
    it up progressively for the later GNN training phases:
      - "none": detector fully frozen (phase 1)
      - "neck": PANet neck trains as an adaptation layer, ConvNeXt frozen (phase 2)
      - "full": backbone + neck both train (phase 3)
    """
    def __init__(self, detector, roi_extractor, gnn, unfreeze="none"):
        super().__init__()
        self.detector = detector
        self.roi_extractor = roi_extractor
        self.gnn = gnn
        self.unfreeze = unfreeze

        self.detector.eval()
        for p in self.detector.parameters():
            p.requires_grad = False
        if unfreeze in ("neck", "full"):
            for p in self.detector.neck.parameters():
                p.requires_grad = True
        if unfreeze == "full":
            for p in self.detector.backbone.parameters():
                p.requires_grad = True

    def train(self, mode=True):
        """Keep the detector in eval mode in every phase: ConvNeXt/PANet are
        LayerNorm-based, so eval only disables dropout/droppath — weights
        still receive gradients when unfrozen."""
        super().train(mode)
        self.detector.eval()
        return self
        
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

    def forward(self, feature_maps_dict, boxes_list, image_size):
        """
        feature_maps_dict: dict mapping string keys (e.g., '0', '1', '2') to tensors
                           from the detector's neck
        boxes_list: list of tensors per batch item, shape (N, 4): absolute
                    (x1, y1, x2, y2) pixels on the input canvas, as every caller
                    passes them. Until 2026-10-29 this denormalized them as if
                    they were (cx, cy, w, h) fractions: every RoI landed far off
                    the image, read zeros, and every node got the same feature
                    (the fc's bias) — GNNs and decoders trained before then never
                    saw the image, and their fc weights never got a gradient.
        image_size: Tuple (height, width) of the input image to the network
        """
        abs_boxes_list = [
            boxes.float() if boxes.numel() else torch.empty((0, 4), dtype=torch.float32, device=boxes.device)
            for boxes in boxes_list
        ]

        # Catch empty batches to prevent crashes
        if all(b.shape[0] == 0 for b in abs_boxes_list):
            # Grab the device from the first feature map in the dictionary
            first_map = next(iter(feature_maps_dict.values()))
            return torch.empty(0, self.fc.out_features, device=first_map.device)

        # MultiScaleRoIAlign needs to know the original image shape to calculate strides properly
        image_shapes = [image_size for _ in range(len(boxes_list))]

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

# Node features: RoI (256) + box (4) + class embedding (32), then the
# detection features: the detector's score and whether bbox repair
# synthesized the box (score 0). Ground-truth boxes are score 1, not
# synthesized. Checkpoints from before the detection features have 292.
ROI_DIM, BOX_DIM, CLASS_EMBED_DIM, DETECTION_DIM = 256, 4, 32, 2
NODE_IN_DIM = ROI_DIM + BOX_DIM + CLASS_EMBED_DIM + DETECTION_DIM


def is_synthesized(node: dict) -> bool:
    """A box bbox_repair added (it carries score 0 and a 'bbr' id)."""
    return "bbr" in str(node.get("id", "")) and node.get("score", 1.0) == 0.0


def node_features(gnn, roi_feats, norm_boxes, labels, scores=None, synthesized=None):
    """The GNN's node input: [RoI, box, class embedding(, score, synthesized)],
    the detection features only for a GNN built with them."""
    parts = [roi_feats, norm_boxes, gnn.class_embedding(labels)]
    base_dim = sum(part.shape[1] for part in parts)
    if gnn.node_in_dim == base_dim + DETECTION_DIM:
        n = labels.shape[0]
        if scores is None:
            scores = roi_feats.new_ones(n)
        if synthesized is None:
            synthesized = roi_feats.new_zeros(n)
        parts.append(torch.stack([scores.to(roi_feats), synthesized.to(roi_feats)], dim=1))
    return torch.cat(parts, dim=1)


class ScoreGraphReconstructor(nn.Module):
    def __init__(self, node_in_dim, num_classes, class_embed_dim=32, hidden_dim=256, num_edge_classes=6, num_heads=4, dropout=0.2,
                 node_head=False):
        super().__init__()
        self.node_in_dim = node_in_dim

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

        # 4. The node keep head: does this box belong in the score? Trained on
        # detected boxes matched to the ground truth (stage D); the logit of
        # keeping it
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
        ) if node_head else None

        # Zero-init the detection-feature columns (score, synthesized) so
        # constant ground-truth values (1, 0) do not act as an uncalibrated
        # bias before stage D trains on detections
        if node_in_dim == NODE_IN_DIM:
            with torch.no_grad():
                self.conv1.lin_l.weight[:, -DETECTION_DIM:].zero_()
                self.conv1.lin_r.weight[:, -DETECTION_DIM:].zero_()

    def freeze_detection_features(self) -> None:
        """Keep the detection-feature input columns of conv1 at zero while
        training on ground-truth boxes only (phases 1-2 and stage C joint)."""
        if self.node_in_dim != NODE_IN_DIM:
            return
        with torch.no_grad():
            self.conv1.lin_l.weight[:, -DETECTION_DIM:].zero_()
            self.conv1.lin_r.weight[:, -DETECTION_DIM:].zero_()

        def _zero_det_grad(grad):
            grad = grad.clone()
            grad[:, -DETECTION_DIM:] = 0.0
            return grad

        self.conv1.lin_l.weight.register_hook(_zero_det_grad)
        self.conv1.lin_r.weight.register_hook(_zero_det_grad)

    def node_states(self, x, edge_index):
        """The graph-aware node states h (N, hidden_dim) after message passing
        over the candidate edges (also the scoregrid decoder's input)."""
        # The nodes look at their neighbors and update their internal representations
        h = self.conv1(x, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        h = self.conv2(h, edge_index)
        h = F.elu(h)
        h = self.dropout(h)

        return self.conv3(h, edge_index)

    def forward(self, x, edge_index, return_nodes=False):
        """
        x: Node features (Number of Nodes, node_in_dim) -> RoI + BBox + Class
        edge_index: Candidate edges (2, Number of Edges) -> From your Axis-Aware heuristics
        """

        # --- PHASE 1: Contextualize the Nodes (Message Passing) ---
        h = self.node_states(x, edge_index)
        
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

        if return_nodes:
            node_logits = self.node_classifier(h).squeeze(1) if self.node_classifier is not None else None
            return edge_logits, node_logits
        return edge_logits