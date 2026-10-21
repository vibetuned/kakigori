"""Modular OMR Trainer using Hugging Face Transformers.

This module decouples the training logic and loss computation from the
main loop, providing a cleaner, standard implementation.
"""

# Third party imports
import torch
import torch.nn as nn
from transformers import Trainer

# Local folder imports
from .losses import CIoULoss, DynamicFocalLoss


# Scale ranges are box areas in fractions of a 640x640 canvas (pixel area /
# 640**2): on the square 640 canvas they are the normalized areas they always
# were, and on any other canvas they keep meaning the same size in pixels,
# which is what decides the stride a box needs. The detector carries its
# ranges (model.DEFAULT_SCALE_RANGES unless set at its first training)
REFERENCE_AREA = 640 * 640


def parse_scale_ranges(value) -> list[tuple[float, float]] | None:
    """Scale ranges from a config: pairs, or a flat list of (min, max) floats."""
    if value is None:
        return None
    if all(isinstance(v, (int, float)) for v in value):
        if len(value) % 2:
            raise ValueError(f"scale ranges need (min, max) pairs, got {len(value)} numbers")
        value = [value[i:i + 2] for i in range(0, len(value), 2)]
    return [(float(lo), float(hi)) for lo, hi in value]


def regression_loss(outputs, reg_targets, obj_masks, ciou_loss_fn) -> torch.Tensor:
    """CIoU over the positive (cell, slot) entries of every level, the
    prediction read from the target's slot (OMRTrainer.build_targets)."""
    total = torch.tensor(0.0, device=outputs[0]["reg"].device)
    for out, reg_target, mask in zip(outputs, reg_targets, obj_masks):
        if not mask.any():
            continue
        b_idx, s_idx, r_idx, c_idx = mask.nonzero(as_tuple=True)
        channels = 4 * s_idx
        pred = torch.stack([out["reg"][b_idx, channels + k, r_idx, c_idx] for k in range(4)], dim=1)
        tgt = torch.stack([reg_target[b_idx, channels + k, r_idx, c_idx] for k in range(4)], dim=1)
        fw, fh = out["reg"].shape[3], out["reg"].shape[2]

        # Offsets inside the cell through a sigmoid; positive sizes through an
        # exp (clamped against fp16 overflow)
        pred_boxes = torch.stack([
            (pred[:, 0].sigmoid() + c_idx) / fw,
            (pred[:, 1].sigmoid() + r_idx) / fh,
            torch.exp(pred[:, 2].clamp(max=10.0)),
            torch.exp(pred[:, 3].clamp(max=10.0)),
        ], dim=1)
        tgt_boxes = torch.stack([
            (tgt[:, 0] + c_idx) / fw,
            (tgt[:, 1] + r_idx) / fh,
            tgt[:, 2],
            tgt[:, 3],
        ], dim=1)
        total = total + ciou_loss_fn(pred_boxes, tgt_boxes)
    return total


class OMRTrainer(Trainer):
    def __init__(
        self,
        cls_weight=1.0,
        reg_weight=5.0,
        scale_ranges=None,
        base_gamma=2.0,
        max_gamma=4.0,
        custom_sampler=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        # Ranges given here (a first training) are stored in the detector;
        # otherwise the ones it was trained with are reused
        if scale_ranges is not None:
            self.model.set_scale_ranges(scale_ranges)
        self.scale_ranges = self.model.scale_ranges
        self.focal_loss_fn = DynamicFocalLoss(
            alpha=0.25, base_gamma=base_gamma, max_gamma=max_gamma
        )
        self.ciou_loss_fn = CIoULoss()
        self.custom_sampler = custom_sampler

    def _get_train_sampler(self, *args, **kwargs):
        if self.custom_sampler is not None:
            return self.custom_sampler
        return super()._get_train_sampler(*args, **kwargs)

    def build_targets(self, targets_list, feature_sizes, num_classes, device, canvas=(640, 640),
                      reg_slot=None):
        """Assign ground-truth boxes to grid cells based on object scale.

        Each box goes to the level its area selects and to the cell holding
        its centre. The class map is multi-label: every class centred on a
        cell is positive there. The regression map holds one box per cell and
        regression slot (``reg_slot``: each class's slot, None = one slot):
        of the boxes sharing a cell and slot, the smallest.

        ``canvas``: (width, height) of the input, for the pixel areas the
        scale ranges are expressed in (see REFERENCE_AREA).

        Returns per level: cls (B, C, H, W), reg (B, 4 * slots, H, W) and the
        positive mask (B, slots, H, W).
        """
        if len(self.scale_ranges) != len(feature_sizes):
            raise ValueError(
                f"{len(self.scale_ranges)} scale ranges for {len(feature_sizes)} detection levels: "
                "set scale_ranges with one (min, max) pair per level"
            )
        if reg_slot is None:
            reg_slot = torch.zeros(num_classes, dtype=torch.long)
        reg_slot = reg_slot.to(device)
        num_slots = int(reg_slot.max()) + 1
        area_factor = canvas[0] * canvas[1] / REFERENCE_AREA
        batch_size = len(targets_list)
        cls_targets = []
        reg_targets = []
        obj_masks = []

        for scale_idx, (fh, fw) in enumerate(feature_sizes):
            cls_map = torch.zeros(batch_size, num_classes, fh, fw, device=device)
            reg_map = torch.zeros(batch_size, 4 * num_slots, fh, fw, device=device)
            obj_mask = torch.zeros(batch_size, num_slots, fh, fw, dtype=torch.bool, device=device)

            # The min and max box area for this level
            min_area, max_area = self.scale_ranges[scale_idx]

            for b, tgt in enumerate(targets_list):
                boxes = tgt["boxes"].to(device)
                labels = tgt["labels"].to(device)

                if boxes.numel() == 0:
                    continue

                box_areas = boxes[:, 2] * boxes[:, 3] * area_factor
                valid_mask = (box_areas >= min_area) & (box_areas < max_area)
                valid_boxes = boxes[valid_mask]
                valid_labels = labels[valid_mask]
                valid_areas = box_areas[valid_mask]

                if valid_boxes.numel() == 0:
                    continue

                # The box centres on this level's grid
                gi = valid_boxes[:, 0] * fw
                gj = valid_boxes[:, 1] * fh
                gi_idx = gi.long().clamp(0, fw - 1)
                gj_idx = gj.long().clamp(0, fh - 1)

                # Multi-label classes: setting 1.0 twice is harmless
                cls_map[b, valid_labels, gj_idx, gi_idx] = 1.0

                # One box per (cell, slot), the smallest. Resolved here, not by
                # write order: a CUDA index assignment with repeated indices
                # keeps an arbitrary write, per channel (measured: the last
                # write kept 65% of the time, and the four channels of 12% of
                # the shared cells taken from different boxes)
                slots = reg_slot[valid_labels]
                key = (slots * fh + gj_idx) * fw + gi_idx
                by_area = torch.argsort(valid_areas, descending=True, stable=True)
                by_key = by_area[torch.argsort(key[by_area], stable=True)]
                sorted_key = key[by_key]
                last = torch.ones_like(sorted_key, dtype=torch.bool)
                last[:-1] = sorted_key[1:] != sorted_key[:-1]
                win = by_key[last]

                s, y, x = slots[win], gj_idx[win], gi_idx[win]
                reg_map[b, 4 * s, y, x] = gi[win] - x
                reg_map[b, 4 * s + 1, y, x] = gj[win] - y
                reg_map[b, 4 * s + 2, y, x] = valid_boxes[win, 2]
                reg_map[b, 4 * s + 3, y, x] = valid_boxes[win, 3]
                obj_mask[b, s, y, x] = True

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
            targets_list, feature_sizes, num_classes, device,
            canvas=(images.shape[-1], images.shape[-2]), reg_slot=outputs[0]["reg_slot"],
        )

        total_cls = torch.tensor(0.0, device=device)

        # Calculate progress (0.0 to 1.0) using HF Trainer state
        max_steps = self.state.max_steps
        current_step = self.state.global_step
        progress = current_step / max_steps if max_steps > 0 else 0.0

        for scale_idx, out in enumerate(outputs):
            # Pass the progress variable to the new DynamicFocalLoss
            total_cls = total_cls + self.focal_loss_fn(
                out["cls"], cls_targets[scale_idx], progress=progress
            )

        total_reg = regression_loss(outputs, reg_targets, obj_masks, self.ciou_loss_fn)
        total_loss = (self.cls_weight * total_cls) + (self.reg_weight * total_reg)

        self._custom_metrics = {
            "train/cls_loss": total_cls.detach().item(),
            "train/reg_loss": total_reg.detach().item(),
        }

        return (total_loss, outputs) if return_outputs else total_loss
