# Third party imports
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Import your previously built components
# from models import ModelA_Detector, GraphVisualExtractor, ScoreGraphReconstructor
# from heuristics import generate_axis_aware_edges, generate_text_candidate_edges
# from serialization import generate_kern_stream, _collapse_primitives
# from context import ContextTracker


class FullPageOMRPipeline:
    def __init__(self, detector, roi_extractor, gnn, class_list, device="cuda"):
        self.device = torch.device(device)
        self.class_list = class_list
        self.class_to_idx = {c: i for i, c in enumerate(class_list)}

        # We assume models are already loaded with their weights and set to eval()
        self.detector = detector.to(self.device).eval()
        self.roi_extractor = roi_extractor.to(self.device).eval()
        self.gnn = gnn.to(self.device).eval()

    @torch.inference_mode()
    def process_page(self, image_path, node_roles):
        """Processes a full page image and returns the complete **kern document."""
        image = Image.open(image_path).convert("RGB")
        image_tensor = (
            TF.to_tensor(image).unsqueeze(0).to(self.device)
        )  # Shape: (1, C, H, W)

        # --- 1. Model A: Full Page Detection ---
        # Assuming the detector returns absolute pixel coordinates [x1, y1, x2, y2]
        predictions = self.detector(image_tensor)[0]
        all_boxes = predictions["boxes"]
        all_labels = predictions["labels"]
        all_scores = predictions["scores"]

        # Filter by confidence threshold (e.g., > 0.5)
        keep = all_scores > 0.5
        all_boxes = all_boxes[keep]
        all_labels = all_labels[keep]

        # --- 2. System Isolation & Coordinate Translation ---
        system_idx = self.class_to_idx["system"]  # From gelato_config.json
        system_mask = all_labels == system_idx

        system_boxes = all_boxes[system_mask]
        primitive_boxes = all_boxes[~system_mask]
        primitive_labels = all_labels[~system_mask]

        # Sort systems vertically top-to-bottom on the page
        system_boxes = system_boxes[system_boxes[:, 1].argsort()]

        full_page_kern = []

        # --- 3. Process Each System Independently ---
        for sys_idx, sys_box in enumerate(system_boxes):
            sx1, sy1, sx2, sy2 = sys_box

            # Find all primitives whose center point (cy) falls within this system's vertical boundaries
            centers_y = (primitive_boxes[:, 1] + primitive_boxes[:, 3]) / 2.0
            in_system_mask = (centers_y >= sy1) & (centers_y <= sy2)

            sys_primitives_boxes = primitive_boxes[in_system_mask].clone()
            sys_primitives_labels = primitive_labels[in_system_mask]

            if len(sys_primitives_boxes) == 0:
                continue

            # CRITICAL: Translate absolute page coordinates to relative system coordinates
            sys_primitives_boxes[:, 0] -= sx1  # x1
            sys_primitives_boxes[:, 1] -= sy1  # y1
            sys_primitives_boxes[:, 2] -= sx1  # x2
            sys_primitives_boxes[:, 3] -= sy1  # y2

            # Crop the image tensor to just this system
            # Coordinates must be integers for tensor slicing
            crop_x1, crop_y1, crop_x2, crop_y2 = map(int, [sx1, sy1, sx2, sy2])
            system_crop = image_tensor[:, :, crop_y1:crop_y2, crop_x1:crop_x2]

            # --- 4. Model B: Graph Generation ---
            # Reconstruct the node tracking dictionary expected by your serializer
            nodes_meta = []
            for i in range(len(sys_primitives_boxes)):
                box = sys_primitives_boxes[i].cpu().tolist()
                label_name = self.class_list[sys_primitives_labels[i].item()]
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2

                nodes_meta.append(
                    {"id": i, "class": label_name, "bbox": box, "cx": cx, "cy": cy}
                )

            # A. Extract visual features using the *cropped* image and *relative* boxes
            roi_features = self.roi_extractor(
                self.detector.extract_features(
                    system_crop
                ),  # Assuming a feature extraction method
                [sys_primitives_boxes],
                system_crop.shape[-2:],
            )

            # B. Build PyG inputs (x, edge_index)
            class_embeds = self.gnn.class_embedding(sys_primitives_labels)
            x = torch.cat([roi_features, sys_primitives_boxes, class_embeds], dim=1)

            # Generate candidate edges using your spatial heuristics based on relative coordinates
            edge_index = generate_axis_aware_edges(
                sys_primitives_boxes, sys_primitives_labels, self.class_to_idx
            )

            # C. Predict Edges
            edge_logits = self.gnn(x, edge_index)
            edge_predictions = torch.argmax(edge_logits, dim=1)

            # --- 5. Semantic Serialization ---
            # Instantiate the serializer graph with your predicted edges
            # (Assuming you updated HumdrumSerializer to accept this format)
            # Third party imports
            from serialization import HumdrumSerializer

            serializer = HumdrumSerializer(nodes_meta, edge_index, edge_predictions)

            super_nodes = serializer._collapse_primitives(node_roles)

            # Sort the raw nodes left-to-right for the context tracker sweep
            sorted_nodes = sorted(nodes_meta, key=lambda n: n["cx"])

            # Generate the **kern string for this specific system
            system_kern_string = generate_kern_stream(
                sorted_nodes, super_nodes, node_roles
            )

            # Append system marker and the decoded string
            full_page_kern.append(f"!! System {sys_idx + 1}")
            full_page_kern.append(system_kern_string)

        return "\n".join(full_page_kern)
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
#from .model import GraphVisualExtractor, ScoreGraphReconstructor
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
