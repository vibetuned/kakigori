"""Evaluation Script for EdgeMusicDetector OMR model.

Calculates Mean Average Precision (mAP) on a validation dataset.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
#from torchvision.ops import nms
from torchvision.ops import batched_nms
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from transformers.trainer_utils import get_last_checkpoint

from .model import EdgeMusicDetector
from .dataset import OMRDataset
from .collator import omr_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def decode_batch_for_eval(outputs, conf_thresh, iou_thresh, input_size):
    """Decodes multi-scale outputs into absolute canvas coordinates [x1, y1, x2, y2]."""
    batch_size = outputs[0]["cls"].shape[0]
    device = outputs[0]["cls"].device
    batch_preds = []

    for b in range(batch_size):
        b_boxes, b_scores, b_labels = [], [], []

        for out in outputs:
            cls_map = out["cls"][b].sigmoid()  # (C, H, W)
            reg_map = out["reg"][b]            # (4, H, W)

            max_scores, max_classes = cls_map.max(dim=0)
            pos = (max_scores >= conf_thresh).nonzero(as_tuple=False)

            if pos.numel() == 0:
                continue

            scores = max_scores[pos[:, 0], pos[:, 1]]
            cls_idx = max_classes[pos[:, 0], pos[:, 1]]

            # Extract normalized coordinates
            # FIX: Decode tx, ty offsets using sigmoid and grid indices
            W, H = reg_map.shape[2], reg_map.shape[1]
            
            # Notice we drop the leading '0,' because reg_map is only 3D here: (4, H, W)
            # Channel 0: tx, Channel 1: ty
            cx_n = (reg_map[0, pos[:, 0], pos[:, 1]].sigmoid() + pos[:, 1]) / W
            cy_n = (reg_map[1, pos[:, 0], pos[:, 1]].sigmoid() + pos[:, 0]) / H
            
            # Channel 2: tw, Channel 3: th
            bw_n = reg_map[2, pos[:, 0], pos[:, 1]].exp()
            bh_n = reg_map[3, pos[:, 0], pos[:, 1]].exp()

            # Convert to absolute pixel coordinates on the padded canvas
            cx = cx_n * input_size
            cy = cy_n * input_size
            bw = bw_n * input_size
            bh = bh_n * input_size

            x1 = cx - bw / 2
            y1 = cy - bh / 2
            x2 = cx + bw / 2
            y2 = cy + bh / 2

            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            b_boxes.append(boxes)
            b_scores.append(scores)
            b_labels.append(cls_idx)

        if b_boxes:
            b_boxes = torch.cat(b_boxes, dim=0)
            b_scores = torch.cat(b_scores, dim=0)
            b_labels = torch.cat(b_labels, dim=0)

            # Apply PyTorch Vision's highly optimized NMS
            keep = batched_nms(b_boxes, b_scores, b_labels, iou_thresh)
            
            batch_preds.append({
                "boxes": b_boxes[keep],
                "scores": b_scores[keep],
                "labels": b_labels[keep]
            })
        else:
            batch_preds.append({
                "boxes": torch.empty((0, 4), device=device),
                "scores": torch.empty((0,), device=device),
                "labels": torch.empty((0,), dtype=torch.long, device=device)
            })

    return batch_preds


def format_ground_truth(targets_list, input_size, device):
    """Converts normalized (cx, cy, w, h) to absolute (x1, y1, x2, y2)."""
    batch_targets = []
    for tgt in targets_list:
        gt_boxes = tgt["boxes"].to(device)
        gt_labels = tgt["labels"].to(device)

        if gt_boxes.numel() > 0:
            cx = gt_boxes[:, 0] * input_size
            cy = gt_boxes[:, 1] * input_size
            w = gt_boxes[:, 2] * input_size
            h = gt_boxes[:, 3] * input_size

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            abs_boxes = torch.stack([x1, y1, x2, y2], dim=1)
        else:
            abs_boxes = torch.empty((0, 4), device=device)

        batch_targets.append({
            "boxes": abs_boxes,
            "labels": gt_labels
        })
    return batch_targets


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMR model mAP.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--img-dir", type=str, required=True, help="Validation images")
    parser.add_argument("--ann-dir", type=str, required=True, help="Validation annotations")
    parser.add_argument("--config", type=str, default="gelato_config.json")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-thresh", type=float, default=0.05) # Keep low for mAP calculation
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--use-bottom-up", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Config & Dataset ---
    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    
    # Note: augment=False so we evaluate on clean images
    val_dataset = OMRDataset(
        img_dir=args.img_dir, ann_dir=args.ann_dir, 
        class_list=class_list, input_size=args.input_size, augment=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=omr_collate_fn
    )

    # --- Load Model ---
    model = EdgeMusicDetector(num_classes=len(class_list), use_bottom_up=args.use_bottom_up)
    ckpt = Path(args.checkpoint)
    if ckpt.is_dir():
        last = get_last_checkpoint(str(ckpt))
        if last is not None:
            ckpt = Path(last)
        logger.info(f"  Using last checkpoint: {ckpt}")

    for name in ("model.safetensors", "pytorch_model.bin"):
        weights = (ckpt / name) if ckpt.is_dir() else ckpt
        if weights.exists():
            if weights.suffix == ".safetensors":
                from safetensors.torch import load_file
                state = load_file(weights, device=str(device))
            else:
                state = torch.load(weights, map_location=device, weights_only=False)
            model.load_state_dict(state)
            logger.info(f"  Loaded weights: {weights}")
            break
    else:
        logger.warning("No weights file found — using random weights.")
            
    model.to(device).eval()

    # --- Initialize Metric ---
    metric = MeanAveragePrecision(
        box_format="xyxy", 
        iou_type="bbox",
        # Default is [1, 10, 100]. We scale this up for dense OMR pages!
        max_detection_thresholds=[100, 1000, 5000],
        class_metrics=True 
    )
    
    # Silence the warning since we actively expect thousands of detections
    metric.warn_on_many_detections = False

    # --- Evaluation Loop ---
    logger.info(f"Evaluating on {len(val_dataset)} images...")
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Calculating mAP"):
            images = batch["pixel_values"].to(device)
            targets_list = batch["labels"]

            # 1. Forward Pass
            outputs = model(images)

            # 2. Decode Predictions & Format Ground Truths
            preds = decode_batch_for_eval(outputs, args.conf_thresh, args.iou_thresh, args.input_size)
            targets = format_ground_truth(targets_list, args.input_size, device)

            # 3. Update Metric
            metric.update(preds, targets)

    # --- Compute and Print Results ---
    results = metric.compute()
    logger.info("\n--- Evaluation Results ---")
    logger.info(f"mAP @ IoU=0.50      : {results['map_50'].item():.4f}")
    logger.info(f"mAP @ IoU=0.75      : {results['map_75'].item():.4f}")
    logger.info(f"mAP @ IoU=0.50:0.95 : {results['map'].item():.4f}")

    logger.info("\n--- Class-wise mAP @ IoU=0.50:0.95 ---")
    
    if 'classes' in results:
        classes_present = results['classes'].tolist()
        map_50_per_class = results['map_per_class'].tolist()
        
        # Pair up the string names with their scores
        class_scores = []
        for cls_idx, score in zip(classes_present, map_50_per_class):
            cls_name = class_list[int(cls_idx)]
            class_scores.append((cls_name, score))
            
        # Sort from highest performing to lowest performing
        class_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Print a cleanly formatted table
        for cls_name, score in class_scores:
            # The <25 pads the class name with spaces so the scores align perfectly
            logger.info(f"{cls_name:<25}: {score:.4f}")
    else:
        logger.warning("No class-wise metrics were returned by the evaluator.")


if __name__ == "__main__":
    main()