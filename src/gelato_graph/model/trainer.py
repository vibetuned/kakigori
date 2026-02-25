"""Modular OMR Trainer using Hugging Face Transformers.

This module decouples the training logic and loss computation from the 
main loop, providing a cleaner, standard implementation.
"""

import torch
import torch.nn as nn
from transformers import Trainer
from .losses import FocalLoss, CIoULoss


class OMRTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        self.ciou_loss_fn = CIoULoss()

    def build_targets(self, targets_list, feature_sizes, num_classes, device):
        """Assign ground-truth boxes to grid cells based on object scale."""
        batch_size = len(targets_list)
        cls_targets = []
        reg_targets = []
        obj_masks = []
        
        # 1. Define Area Thresholds (Normalized 0.0 to 1.0)
        # Assumes 3 feature maps from highest-res (P3) to lowest-res (P5).
        # Area = width * height. A box covering 20% width and 20% height has an area of 0.04.
        # - Scale 0 (e.g., 64x64 grid): For tiny objects like dots and noteheads (< 4% of image area)
        # - Scale 1 (e.g., 32x32 grid): For medium objects like short stems and rests (4% to 16%)
        # - Scale 2 (e.g., 16x16 grid): For large objects like Clefs, beams, and brackets (> 16%)
        scale_ranges = [(0.0, 0.04), (0.04, 0.16), (0.16, 2.0)] 

        for scale_idx, (fh, fw) in enumerate(feature_sizes):
            # Initialize empty target tensors for this specific scale
            cls_map = torch.zeros(batch_size, num_classes, fh, fw, device=device)
            reg_map = torch.zeros(batch_size, 4, fh, fw, device=device)
            obj_mask = torch.zeros(batch_size, fh, fw, dtype=torch.bool, device=device)
            
            # Get the min and max allowed area for this feature map
            min_area, max_area = scale_ranges[scale_idx]

            for b, tgt in enumerate(targets_list):
                boxes = tgt["boxes"].to(device)
                labels = tgt["labels"].to(device)

                if boxes.numel() == 0:
                    continue
                    
                # 2. Calculate the area of every ground truth box
                # Since boxes are normalized (cx, cy, w, h), w is index 2, h is index 3
                box_areas = boxes[:, 2] * boxes[:, 3]
                
                # 3. Filter boxes: Keep only the ones that belong to this feature scale
                valid_mask = (box_areas >= min_area) & (box_areas < max_area)
                valid_boxes = boxes[valid_mask]
                valid_labels = labels[valid_mask]
                
                if valid_boxes.numel() == 0:
                    continue

                # 4. Map the valid box centers strictly to this grid's coordinates
                gi = (valid_boxes[:, 0] * fw).long().clamp(0, fw - 1)
                gj = (valid_boxes[:, 1] * fh).long().clamp(0, fh - 1)

                for n in range(valid_boxes.shape[0]):
                    j, i = gj[n], gi[n]
                    
                    # Assign the filtered ground truth to the grid cell
                    cls_map[b, valid_labels[n], j, i] = 1.0
                    reg_map[b, :, j, i] = valid_boxes[n]
                    obj_mask[b, j, i] = True

            cls_targets.append(cls_map)
            reg_targets.append(reg_map)
            obj_masks.append(obj_mask)

        return cls_targets, reg_targets, obj_masks

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation for the OMR model."""
        images = inputs.get("pixel_values")
        targets_list = inputs.get("labels")
        
        # Determine num_classes from the model head or config
        # Assuming the first head's out_channels is num_classes
        num_classes = model.heads[0].cls_branch[-1].out_channels
        device = images.device

        outputs = model(images)
        
        feature_sizes = [(o["cls"].shape[2], o["cls"].shape[3]) for o in outputs]
        cls_targets, reg_targets, obj_masks = self.build_targets(
            targets_list, feature_sizes, num_classes, device
        )

        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)

        for scale_idx, out in enumerate(outputs):
            # Classification loss
            total_cls = total_cls + self.focal_loss_fn(out["cls"], cls_targets[scale_idx])

            # Regression loss
            mask = obj_masks[scale_idx]
            if mask.any():
                pos_indices = mask.nonzero(as_tuple=False)
                b_idx, r_idx, c_idx = pos_indices[:, 0], pos_indices[:, 1], pos_indices[:, 2]
                
                pred_pos = out["reg"][b_idx, :, r_idx, c_idx]
                tgt_pos = reg_targets[scale_idx][b_idx, :, r_idx, c_idx]
                
                total_reg = total_reg + self.ciou_loss_fn(
                    pred_pos.unsqueeze(-1).unsqueeze(-1),
                    tgt_pos.unsqueeze(-1).unsqueeze(-1),
                )

        reg_weight = 5.0
        total_loss = total_cls + reg_weight * total_reg

        # Accumulate custom metrics; they are flushed via the log() override below.
        self._custom_metrics = {
            "train/cls_loss": total_cls.detach().item(),
            "train/reg_loss": total_reg.detach().item(),
        }

        return (total_loss, outputs) if return_outputs else total_loss

    def log(self, logs: dict, start_time: float | None = None) -> None:
        """Merge custom metrics into every log call made by the Trainer."""
        if hasattr(self, "_custom_metrics"):
            logs.update(self._custom_metrics)
            self._custom_metrics = {}
        super().log(logs, start_time=start_time)
