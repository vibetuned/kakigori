import torch
import torch.nn as nn
from transformers import Trainer
from sklearn.metrics import classification_report

# Import your custom modules
# from .losses import MultiClassEdgeFocalLoss
# from .heuristics import generate_axis_aware_edges, map_gt_to_candidates
# from .topology import GraphTopologyEvaluator




class GNNTrainer(Trainer):
    def __init__(self, alpha_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize the multi-class focal loss
        self.loss_fn = MultiClassEdgeFocalLoss(alpha_weights=alpha_weights)
        self.topology_evaluator = GraphTopologyEvaluator()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Intercepts the batched lists from omr_collate_fn, computes the graphs, 
        and calculates the focal loss.
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
        
        for i in range(len(images)):
            img = images[i].to(device)
            boxes = boxes_list[i].to(device)
            labels = labels_list[i].to(device)
            edges = edges_list[i].to(device)
            
            if len(boxes) < 2 or len(edges) == 0:
                continue
            
            # 1. Vision Extraction (Frozen)
            with torch.no_grad():
                roi_feats = model.roi_extractor(img.unsqueeze(0), [boxes], img.shape[-2:])
            
            # 2. Build Node Features
            class_embeds = model.gnn.class_embedding(labels)
            x = torch.cat([roi_feats, boxes, class_embeds], dim=1)
            
            # 3. Graph Generation & Target Mapping
            candidate_edge_index = generate_axis_aware_edges(boxes, labels)
            y_targets = map_gt_to_candidates(candidate_edge_index, edges)
            
            # 4. GNN Forward Pass
            edge_logits = model.gnn(x, candidate_edge_index)
            
            # Accumulate Loss
            loss = self.loss_fn(edge_logits, y_targets)
            total_loss = total_loss + loss
            valid_systems += 1
            
            # Store outputs for metrics calculation
            if return_outputs or not model.training:
                all_logits.append(edge_logits)
                all_targets.append(y_targets)
                
                # --- Custom Topology Evaluation (Validation Only) ---
                if not model.training:
                    preds = torch.argmax(edge_logits, dim=1)
                    self.topology_evaluator.update(
                        edge_index=candidate_edge_index,
                        gt_edges=y_targets,
                        pred_edges=preds,
                        num_nodes=len(boxes)
                    )
        
        # Average loss over the batch
        if valid_systems > 0:
            total_loss = total_loss / valid_systems
            
        if return_outputs:
            if all_logits:
                concat_logits = torch.cat(all_logits, dim=0)
                concat_targets = torch.cat(all_targets, dim=0)
                outputs = {"logits": concat_logits, "labels": concat_targets}
            else:
                outputs = {"logits": torch.empty((0, 5), device=device), "labels": torch.empty(0, device=device)}
            return total_loss, outputs
            
        return total_loss

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
    
    target_names = ['No Edge', 'Structural', 'Modifier', 'Temporal', 'Sync']
    
    report = classification_report(
        labels, 
        preds, 
        target_names=target_names, 
        zero_division=0, 
        output_dict=True
    )
    
    return {
        "f1_structural": report['Structural']['f1-score'],
        "f1_modifier": report['Modifier']['f1-score'],
        "f1_temporal": report['Temporal']['f1-score'],
        "f1_sync": report['Sync']['f1-score'],
        "f1_macro": report['macro avg']['f1-score']
    }