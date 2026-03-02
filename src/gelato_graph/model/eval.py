import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

from .model import EdgeMusicDetector
from .dataset import OMRDataset
from .utils import decode_model_outputs, load_checkpoint, omr_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_checkpoint_step(checkpoint_dir):
    """Extracts the step number from the checkpoint directory path."""
    checkpoint_path = Path(checkpoint_dir)
    checkpoints = [int(c.name.split("-")[-1]) for c in checkpoint_path.iterdir() if c.name.startswith("checkpoint-")]
    checkpoints.sort()
    return checkpoint_path.name, checkpoints[-1]

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
    parser.add_argument("--hierarchy", type=str, default="hierarchy.json")
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--conf-thresh", type=float, default=0.05) 
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--use-bottom-up", action="store_true")
    # Trainer args
    parser.add_argument("--out-indices", type=int, nargs=3, default=[0, 1, 2],
        help="Three backbone feature map indices to extract (default: 0 1 2).",
    )
    # Tensorboard logging
    parser.add_argument("--tb-dir", type=str, default="runs", help="TensorBoard log directory")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        config = json.load(f)
    class_list = config["target_classes"]
    
    with open(args.hierarchy) as f:
        hierarchy = json.load(f)
    
    val_dataset = OMRDataset(
        img_dir=args.img_dir, ann_dir=args.ann_dir, 
        class_list=class_list, input_size=args.input_size, augment=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=omr_collate_fn
    )

    model = EdgeMusicDetector(num_classes=len(class_list), use_bottom_up=args.use_bottom_up, out_indices=args.out_indices)
    
    load_checkpoint(model, args.checkpoint, device)
    checkpoint_name, step = get_checkpoint_step(args.checkpoint)

    metric = MeanAveragePrecision(
        box_format="xyxy", 
        iou_type="bbox",
        max_detection_thresholds=[100, 1000, 5000],
        class_metrics=True 
    )
    metric.warn_on_many_detections = False

    logger.info(f"Evaluating on {len(val_dataset)} images...")
    
    # NEW: Move the metric to the GPU so the update loop doesn't bottleneck on CPU transfers
    metric.to(device)
    
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Calculating mAP"):
            images = batch["pixel_values"].to(device)
            targets_list = batch["labels"]

            outputs = model(images)

            preds = decode_model_outputs(outputs, args.conf_thresh, args.iou_thresh, args.input_size)
            targets = format_ground_truth(targets_list, args.input_size, device)

            # PyTorch Lightning metrics are much faster when both tensors and the metric are on the same device
            metric.update(preds, targets)

    results = metric.compute()
    
    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=args.tb_dir + "/" + checkpoint_name)

    logger.info("\n--- Evaluation Results ---")
    logger.info(f"mAP @ IoU=0.50      : {results['map_50'].item():.4f}")
    logger.info(f"mAP @ IoU=0.75      : {results['map_75'].item():.4f}")
    logger.info(f"mAP @ IoU=0.50:0.95 : {results['map'].item():.4f}")

    # Log overall mAP metrics to TensorBoard
    writer.add_scalar("mAP/IoU_0.50", results['map_50'].item(), step)
    writer.add_scalar("mAP/IoU_0.75", results['map_75'].item(), step)
    writer.add_scalar("mAP/IoU_0.50_0.95", results['map'].item(), step)

    logger.info("\n--- Class-wise mAP @ IoU=0.50:0.95 ---")
    
    if 'classes' in results:
        classes_present = results['classes'].tolist()
        map_50_per_class = results['map_per_class'].tolist()
        
        # Create a dictionary of {class_name: score} for easy lookup
        class_scores = {class_list[int(cls_idx)]: score 
                        for cls_idx, score in zip(classes_present, map_50_per_class)}
        
        # --- 1. Define the Custom TensorBoard Layout ---
        layout = {"Grouped_Class_mAP": {}}
        tracked_classes = set()
        
        for group_name, grouped_classes in hierarchy.items():
            # Create a list of the exact tags that will go into this chart
            tags = [f"mAP_Class/{cls_name}" for cls_name in grouped_classes if cls_name in class_scores]
            
            if tags:
                # Tell TensorBoard to put these tags on a single "Multiline" chart
                layout["Grouped_Class_mAP"][group_name] = ["Multiline", tags]
                tracked_classes.update(grouped_classes)

        # Catch uncategorized classes so they aren't left out
        uncategorized_tags = [f"mAP_Class/{cls_name}" for cls_name in class_scores if cls_name not in tracked_classes]
        if uncategorized_tags:
            layout["Grouped_Class_mAP"]["Uncategorized"] = ["Multiline", uncategorized_tags]

        # Apply the layout to the writer
        writer.add_custom_scalars(layout)

        # --- 2. Log the actual values using standard add_scalar ---
        for cls_name, score in class_scores.items():
            logger.info(f"{cls_name:<25}: {score:.4f}")
            # Because we defined the layout above, TensorBoard will automatically 
            # route these individual scalars into the correct multi-line charts!
            writer.add_scalar(f"mAP_Class/{cls_name}", score, step)
            
    else:
        logger.warning("No class-wise metrics were returned by the evaluator.")

    writer.close()

if __name__ == "__main__":
    main()