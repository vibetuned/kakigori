"""Modular OMR Trainer using Hugging Face Transformers.

This module decouples the training logic and loss computation from the 
main loop, providing a cleaner, standard implementation.
"""

import torch
import torch.nn as nn
from transformers import Trainer
from .losses import CIoULoss, DynamicFocalLoss


class OMRTrainer(Trainer):
    DEFAULT_SCALE_RANGES = [(0.0, 0.0001), (0.0001, 0.001), (0.001, 2.0)] #[(0.0, 0.0002), (0.0002, 0.002), (0.002, 2.0)]

    def __init__(self, cls_weight=1.0, reg_weight=5.0, scale_ranges=None, base_gamma=2.0, max_gamma=4.0, custom_sampler=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.scale_ranges = scale_ranges if scale_ranges is not None else self.DEFAULT_SCALE_RANGES
        self.focal_loss_fn = DynamicFocalLoss(alpha=0.25, base_gamma=base_gamma, max_gamma=max_gamma)
        self.ciou_loss_fn = CIoULoss()
        self.custom_sampler = custom_sampler

    def _get_train_sampler(self, *args, **kwargs):
        if self.custom_sampler is not None:
            return self.custom_sampler
        return super()._get_train_sampler(*args, **kwargs)

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
        # scale_ranges = [(0.0, 0.04), (0.04, 0.16), (0.16, 2.0)] 
        # P2 (Tiny - < 0.0002): artic, stem, notehead, tie, flag.
        # P3 (Medium - 0.0002 to 0.002): note, slur, keySig, chord, clef, beam.
        # P4 (Large - > 0.002): tuplet, staff, measure, system, page-margin.
        
        for scale_idx, (fh, fw) in enumerate(feature_sizes):
            # Initialize empty target tensors for this specific scale
            cls_map = torch.zeros(batch_size, num_classes, fh, fw, device=device)
            reg_map = torch.zeros(batch_size, 4, fh, fw, device=device)
            obj_mask = torch.zeros(batch_size, fh, fw, dtype=torch.bool, device=device)
            
            # Get the min and max allowed area for this feature map
            min_area, max_area = self.scale_ranges[scale_idx]

            for b, tgt in enumerate(targets_list):
                boxes = tgt["boxes"].to(device)
                labels = tgt["labels"].to(device)

                if boxes.numel() == 0:
                    continue
                    
                # Calculate the area of every ground truth box
                box_areas = boxes[:, 2] * boxes[:, 3]
                
                # Filter boxes: Keep only the ones that belong to this feature scale
                valid_mask = (box_areas >= min_area) & (box_areas < max_area)
                valid_boxes = boxes[valid_mask]
                valid_labels = labels[valid_mask]
                valid_areas = box_areas[valid_mask] 
                
                if valid_boxes.numel() == 0:
                    continue

                # --- THE MAGIC FIX: Area Sorting ---
                # Sort descending so the smallest objects are processed LAST
                sort_idx = torch.argsort(valid_areas, descending=True)
                valid_boxes = valid_boxes[sort_idx]
                valid_labels = valid_labels[sort_idx]

                # Map the valid box centers strictly to this grid's coordinates
                gi = valid_boxes[:, 0] * fw
                gj = valid_boxes[:, 1] * fh
                
                # Get the integer grid cell coordinates
                gi_idx = gi.long().clamp(0, fw - 1)
                gj_idx = gj.long().clamp(0, fh - 1)

                # --- NO MORE FOR LOOPS: Vectorized Grid Assignment ---
                
                # Calculate all local offsets simultaneously
                tx = gi - gi_idx
                ty = gj - gj_idx
                
                # 1. Update Class Map
                # PyTorch handles the multi-dimensional mapping across the batch instantly
                cls_map[b, valid_labels, gj_idx, gi_idx] = 1.0
                
                # 2. Update Regression Map
                reg_map[b, 0, gj_idx, gi_idx] = tx
                reg_map[b, 1, gj_idx, gi_idx] = ty
                reg_map[b, 2, gj_idx, gi_idx] = valid_boxes[:, 2]
                reg_map[b, 3, gj_idx, gi_idx] = valid_boxes[:, 3]
                
                # 3. Update Object Mask
                obj_mask[b, gj_idx, gi_idx] = True

            cls_targets.append(cls_map)
            reg_targets.append(reg_map)
            obj_masks.append(obj_mask)

        return cls_targets, reg_targets, obj_masks

    def log(self, logs: dict, start_time: float | None = None) -> None:
        """Merge custom metrics into every log call made by the Trainer."""
        if hasattr(self, "_custom_metrics"):
            logs.update(self._custom_metrics)
            self._custom_metrics = {}
        super().log(logs, start_time=start_time)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Custom loss computation for the OMR model."""
        images = inputs.get("pixel_values")
        targets_list = inputs.get("labels")
        
        num_classes = model.heads[0].cls_branch[-1].out_channels
        device = images.device

        outputs = model(images)
        
        feature_sizes = [(o["cls"].shape[2], o["cls"].shape[3]) for o in outputs]
        cls_targets, reg_targets, obj_masks = self.build_targets(
            targets_list, feature_sizes, num_classes, device
        )

        total_cls = torch.tensor(0.0, device=device)
        total_reg = torch.tensor(0.0, device=device)

        # Calculate progress (0.0 to 1.0) using HF Trainer state
        max_steps = self.state.max_steps
        current_step = self.state.global_step
        progress = current_step / max_steps if max_steps > 0 else 0.0

        for scale_idx, out in enumerate(outputs):
            # Pass the progress variable to the new DynamicFocalLoss
            total_cls = total_cls + self.focal_loss_fn(out["cls"], cls_targets[scale_idx], progress=progress)

            # Regression loss (CIoU)
            mask = obj_masks[scale_idx]
            mask = obj_masks[scale_idx]
            if mask.any():
                pos_indices = mask.nonzero(as_tuple=False)
                b_idx, r_idx, c_idx = pos_indices[:, 0], pos_indices[:, 1], pos_indices[:, 2]
                
                pred_pos = out["reg"][b_idx, :, r_idx, c_idx]
                tgt_pos = reg_targets[scale_idx][b_idx, :, r_idx, c_idx]
                
                fw, fh = out["reg"].shape[3], out["reg"].shape[2]
                
                # THE FIX: Convert (tx, ty) back to global (cx, cy)
                # Apply sigmoid to force the network's offset prediction strictly between 0 and 1
                pred_cx = (pred_pos[:, 0].sigmoid() + c_idx) / fw 
                pred_cy = (pred_pos[:, 1].sigmoid() + r_idx) / fh

                # THE FIX: Apply exponential to guarantee strictly positive width/height
                pred_w = torch.exp(pred_pos[:, 2])
                pred_h = torch.exp(pred_pos[:, 3])
                
                tgt_cx = (tgt_pos[:, 0] + c_idx) / fw
                tgt_cy = (tgt_pos[:, 1] + r_idx) / fh
                tgt_w = tgt_pos[:, 2]
                tgt_h = tgt_pos[:, 3]
                
                pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=1)
                tgt_boxes = torch.stack([tgt_cx, tgt_cy, tgt_w, tgt_h], dim=1)
                
                total_reg = total_reg + self.ciou_loss_fn(
                    pred_boxes,
                    tgt_boxes,
                )

        total_loss = (self.cls_weight * total_cls) + (self.reg_weight * total_reg)

        self._custom_metrics = {
            "train/cls_loss": total_cls.detach().item(),
            "train/reg_loss": total_reg.detach().item(),
        }

        return (total_loss, outputs) if return_outputs else total_loss