import torch
import torch.nn as nn
from transformers import Trainer
from sklearn.metrics import classification_report

# Import your custom modules
from .losses import MultiClassEdgeFocalLoss
from .heuristics import generate_axis_aware_edges, map_gt_to_candidates
from .metrics import GraphTopologyEvaluator
from .utils import split_into_systems




class GNNTrainer(Trainer):
    def __init__(self, alpha_weights, class_list, box_jitter=0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the multi-class focal loss
        self.loss_fn = MultiClassEdgeFocalLoss(alpha_weights=alpha_weights)
        self.topology_evaluator = GraphTopologyEvaluator()
        self.class_list = class_list
        # Phase 4 curriculum: gaussian noise (in px) added to GT boxes during
        # training to simulate detector localization error
        self.box_jitter = box_jitter

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        images = inputs["images"]
        boxes_list = inputs["boxes"]
        labels_list = inputs["labels"]
        edges_list = inputs["edges"]
        
        device = self.args.device
        
        # FIX 1: Initialize as a standard float, not an isolated tensor
        total_loss = 0.0 
        
        all_logits = []
        all_targets = []
        valid_systems = 0
        
        class_to_idx = {c: i for i, c in enumerate(self.class_list)} 
        
        unfreeze = getattr(model, "unfreeze", "none")

        for i in range(len(images)):
            img = images[i].to(device)
            abs_boxes = boxes_list[i].to(device)
            labels = labels_list[i].to(device)
            page_edges = edges_list[i].to(device)

            # We only need at least 2 boxes to form an edge.
            if len(abs_boxes) < 2:
                continue

            # Phase 4: simulate detector localization noise on GT boxes
            if model.training and self.box_jitter > 0:
                abs_boxes = abs_boxes + torch.randn_like(abs_boxes) * self.box_jitter
                abs_boxes[:, 2:] = torch.maximum(abs_boxes[:, 2:], abs_boxes[:, :2] + 1.0)

            # --- 1. VISION DOMAIN: Full Page Extraction ---
            # Gradient flow matches the phase: backbone only in "full",
            # neck also in "neck" (PANet as an adaptation layer)
            if unfreeze == "full":
                features = model.detector.backbone(img.unsqueeze(0))
            else:
                with torch.no_grad():
                    features = model.detector.backbone(img.unsqueeze(0))
            if unfreeze in ("neck", "full"):
                fused_features = model.detector.neck(features)
            else:
                with torch.no_grad():
                    fused_features = model.detector.neck(features)
            feat_dict = {str(idx): feat for idx, feat in enumerate(fused_features)}

            # --- 2. BRIDGE: Isolate Systems ---
            system_groups = split_into_systems(abs_boxes, labels, page_edges, class_to_idx)
            
            for sys_data in system_groups:
                sys_abs_boxes = sys_data['abs_boxes']
                sys_labels = sys_data['labels']
                sys_gt_targets = sys_data['edge_targets']
                sx1, sy1, sx2, sy2 = sys_data['system_bbox']
                
                if len(sys_abs_boxes) < 2:
                    continue
                
                # FIX 2: Removed torch.no_grad() here! The RoI Extractor MUST train in Phase 2.
                roi_feats = model.roi_extractor(feat_dict, [sys_abs_boxes], img.shape[-2:])
                
                sys_rel_boxes = sys_abs_boxes.clone()
                sys_rel_boxes[:, 0] -= sx1
                sys_rel_boxes[:, 1] -= sy1
                sys_rel_boxes[:, 2] -= sx1
                sys_rel_boxes[:, 3] -= sy1

                # Node features must share a scale: raw pixel coords (0..640)
                # swamp the unit-scale RoI features and embeddings inside the
                # GATv2 projections. Heuristics keep the pixel-space copy.
                norm_boxes = sys_rel_boxes / img.shape[-1]

                class_embeds = model.gnn.class_embedding(sys_labels)
                x = torch.cat([roi_feats, norm_boxes, class_embeds], dim=1)

                # FIX 3: Passed class_to_idx to heuristics
                candidate_edge_index = generate_axis_aware_edges(sys_rel_boxes, sys_labels, class_to_idx)
                
                # FIX 4: Prevent NaN loss if heuristics found absolutely zero candidate edges
                if candidate_edge_index.shape[1] == 0:
                    continue
                    
                y_targets = map_gt_to_candidates(candidate_edge_index, sys_gt_targets)
                
                # GNN Forward Pass
                edge_logits = model.gnn(x, candidate_edge_index)
                
                loss = self.loss_fn(edge_logits, y_targets)
                total_loss = total_loss + loss
                valid_systems += 1
                
                if return_outputs or not model.training:
                    all_logits.append(edge_logits)
                    all_targets.append(y_targets)
                    
                    if not model.training:
                        preds = torch.argmax(edge_logits, dim=1)
                        self.topology_evaluator.update(
                            edge_index=candidate_edge_index,
                            gt_edges=y_targets,
                            pred_edges=preds,
                            num_nodes=len(sys_abs_boxes)
                        )
        
        # FIX 5: The "Dummy Loss Trick" 
        # If no valid systems were found, we multiply 0.0 by the sum of all model parameters.
        # This creates a valid computational graph with 0.0 gradients, preventing the AMP crash.
        if valid_systems > 0:
            total_loss = total_loss / valid_systems
        else:
            total_loss = sum(p.sum() for p in model.parameters() if p.requires_grad) * 0.0
            
        if return_outputs:
            if all_logits:
                concat_logits = torch.cat(all_logits, dim=0)
                concat_targets = torch.cat(all_targets, dim=0)
                outputs = {"logits": concat_logits, "labels": concat_targets}
            else:
                outputs = {"logits": torch.empty((0, 6), device=device), "labels": torch.empty(0, device=device)}
            return total_loss, outputs
            
        return total_loss

    def compute_loss2(self, model, inputs, return_outputs=False, **kwargs):
        """
        Intercepts batched pages, extracts full-page vision features, 
        slices the data into independent system graphs, and accumulates the loss.
        """
        images = inputs["images"]
        boxes_list = inputs["boxes"]
        labels_list = inputs["labels"]
        edges_list = inputs["edges"]
        
        device = self.args.device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        all_logits = []
        all_targets = []
        valid_systems = 0
        
        # We need the class-to-index mapping to find the 'system' bounding boxes
        # Assuming you passed class_list to the Trainer or can access it globally
        class_to_idx = {c: i for i, c in enumerate(self.class_list)} 
        
        for i in range(len(images)):
            img = images[i].to(device)
            abs_boxes = boxes_list[i].to(device)
            labels = labels_list[i].to(device)
            page_edges = edges_list[i].to(device)
            
            if len(abs_boxes) < 2 or len(page_edges) == 0:
                continue
            
            # --- 1. VISION DOMAIN: Full Page Extraction (Frozen) ---
            with torch.no_grad():
                features = model.detector.backbone(img.unsqueeze(0))
                fused_features = model.detector.neck(features)
                feat_dict = {str(idx): feat for idx, feat in enumerate(fused_features)}

            # --- 2. BRIDGE: Isolate Systems & Re-index Edges ---
            # This isolates primitives, re-maps global edge indices to local system indices,
            # and returns a list of dictionaries for each valid system on the page.
            system_groups = split_into_systems(abs_boxes, labels, page_edges, class_to_idx)
            
            for sys_data in system_groups:
                sys_abs_boxes = sys_data['abs_boxes']
                sys_labels = sys_data['labels']
                sys_gt_targets = sys_data['edge_targets'] # Locally re-indexed!
                sx1, sy1, sx2, sy2 = sys_data['system_bbox']
                
                if len(sys_abs_boxes) < 2:
                    continue
                
                # A. Extract RoI Features using ABSOLUTE page coordinates
                with torch.no_grad():
                    roi_feats = model.roi_extractor(feat_dict, [sys_abs_boxes], img.shape[-2:])
                
                # B. Translate to RELATIVE coordinates purely for the GNN's spatial math
                sys_rel_boxes = sys_abs_boxes.clone()
                sys_rel_boxes[:, 0] -= sx1
                sys_rel_boxes[:, 1] -= sy1
                sys_rel_boxes[:, 2] -= sx1
                sys_rel_boxes[:, 3] -= sy1
                
                # --- 3. RELATIONAL DOMAIN: System Graph ---
                class_embeds = model.gnn.class_embedding(sys_labels)
                x = torch.cat([roi_feats, sys_rel_boxes, class_embeds], dim=1)
                
                # Generate candidates using RELATIVE coordinates
                candidate_edge_index = generate_axis_aware_edges(sys_rel_boxes, sys_labels)
                y_targets = map_gt_to_candidates(candidate_edge_index, sys_gt_targets)
                
                # GNN Forward Pass
                edge_logits = model.gnn(x, candidate_edge_index)
                
                # Accumulate Loss for this specific system
                loss = self.loss_fn(edge_logits, y_targets)
                total_loss = total_loss + loss
                valid_systems += 1
                
                # Store outputs for metrics calculation
                if return_outputs or not model.training:
                    all_logits.append(edge_logits)
                    all_targets.append(y_targets)
                    
                    if not model.training:
                        preds = torch.argmax(edge_logits, dim=1)
                        self.topology_evaluator.update(
                            edge_index=candidate_edge_index,
                            gt_edges=y_targets,
                            pred_edges=preds,
                            num_nodes=len(sys_abs_boxes)
                        )
        
        if valid_systems > 0:
            total_loss = total_loss / valid_systems
            
        if return_outputs:
            if all_logits:
                concat_logits = torch.cat(all_logits, dim=0)
                concat_targets = torch.cat(all_targets, dim=0)
                outputs = {"logits": concat_logits, "labels": concat_targets}
            else:
                outputs = {"logits": torch.empty((0, 6), device=device), "labels": torch.empty(0, device=device)}
            return total_loss, outputs
            
        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Route evaluation through compute_loss: GNNPhase2Model has no
        forward(), and the default prediction_step would call model(**inputs).
        Returns (loss, edge_logits, edge_targets) so compute_metrics receives
        the edge-classification pair rather than the page-level node labels."""
        with torch.no_grad():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return (loss.detach(), None, None)
        return (loss.detach(), outputs["logits"].detach(), outputs["labels"].detach())

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Overrides the standard evaluation loop to inject the custom graph topology metrics 
        into the Hugging Face logging dictionary.
        """
        self.topology_evaluator.reset()
        
        # This calls compute_loss with model.training = False, populating our topology_evaluator
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        
        # Retrieve and format the topology results
        topology_results = self.topology_evaluator.compute()
        for metric_name, value in topology_results.items():
            formatted_name = metric_name.lower().replace(" ", "_").replace("Δ", "delta")
            metrics[f"{metric_key_prefix}/{formatted_name}"] = value
            
        self.log(metrics)
        return metrics

def compute_gnn_metrics(eval_pred):
    """Calculates standard classification metrics from the concatenated logits."""
    logits, labels = eval_pred
    
    # Ensure they are numpy arrays
    if isinstance(logits, tuple):
        logits = logits[0]
        
    preds = logits.argmax(axis=1)

    # Full edge taxonomy emitted by parsers.GroundTruthGraphBuilder:
    # 0 none, 1 structural, 2 modifier, 3 temporal, 4 sync-text (syl/verse),
    # 5 simultaneity (same-onset events, consumed by the kern serializer)
    target_names = ['No Edge', 'Structural', 'Modifier', 'Temporal', 'SyncText', 'Simultaneity']

    report = classification_report(
        labels,
        preds,
        labels=list(range(len(target_names))),
        target_names=target_names,
        zero_division=0,
        output_dict=True
    )

    return {
        "f1_structural": report['Structural']['f1-score'],
        "f1_modifier": report['Modifier']['f1-score'],
        "f1_temporal": report['Temporal']['f1-score'],
        "f1_sync_text": report['SyncText']['f1-score'],
        "f1_simultaneity": report['Simultaneity']['f1-score'],
        "f1_macro": report['macro avg']['f1-score']
    }