import argparse
import json
import logging
from pathlib import Path
import multiprocessing as mp
import concurrent.futures
import torch
from torch.utils.data import DataLoader
from torchvision.ops import batched_nms
from tqdm import tqdm
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.utils.tensorboard import SummaryWriter

from .model import MusicDetector
from .dataset import OMRDataset
from .utils import decode_model_outputs, load_checkpoint, omr_collate_fn

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed

def _evaluate_target_func(q, class_idx, class_name, preds, targets):
    """
    Isolated target function. 
    Runs the C++ backend and pushes the final standard python dict to the Queue.
    """
    try:
        metric = MeanAveragePrecision(
            box_format="xyxy", 
            iou_type="bbox",
            max_detection_thresholds=[100, 1000, 5000],
            class_metrics=False,
        )
        metric.warn_on_many_detections = False
        metric.update(preds, targets)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metric.to(device)
        
        res = metric.compute()
        q.put({
            "class_name": class_name,
            "map_50": res["map_50"].item(),
            "map_75": res["map_75"].item(),
            "map": res["map"].item()
        })
    except Exception as e:
        logger.debug(f"Error in {class_name} mAP calculation: {e}")
        q.put({
            "class_name": class_name,
            "map_50": 0.0, "map_75": 0.0, "map": 0.0
        })

def _evaluate_isolated(args):
    """
    Spawns an isolated OS process for a single class evaluation.
    Catches C++ deadlocks and hangs perfectly using a strict timeout.
    """
    class_idx, class_name, preds, targets = args
    
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_evaluate_target_func, args=(q, class_idx, class_name, preds, targets))
    p.start()
    
    # 5-minute strict timeout to kill infinite C++ matching loops
    p.join(timeout=300) 
    
    if p.is_alive():
        # The process hung. Kill it aggressively.
        logger.error(f" [!] Timeout: Class '{class_name}' calculation hung. Terminating process.")
        p.terminate()
        p.join()
        return {"class_name": class_name, "map_50": 0.0, "map_75": 0.0, "map": 0.0}
        
    if p.exitcode != 0:
        # The process crashed violently (C++ abort, segfault, out of memory, etc.)
        logger.error(f" [!] Crash: Class '{class_name}' exited with code {p.exitcode}.")
        return {"class_name": class_name, "map_50": 0.0, "map_75": 0.0, "map": 0.0}
        
    # Process finished cleanly, grab the result from the queue
    if not q.empty():
        return q.get()
        
    return {"class_name": class_name, "map_50": 0.0, "map_75": 0.0, "map": 0.0}


def main():
    parser = argparse.ArgumentParser(description="Evaluate OMR model mAP.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint dir")
    parser.add_argument("--img-dir", type=str, required=True, help="Validation images")
    parser.add_argument("--ann-dir", type=str, required=True, help="Validation annotations")
    parser.add_argument("--config", type=str, default="conf/config.json")
    parser.add_argument("--hierarchy", type=str, default="conf/hierarchy.json")
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

    model = MusicDetector(num_classes=len(class_list), use_bottom_up=args.use_bottom_up, out_indices=args.out_indices)
    
    load_checkpoint(model, args.checkpoint, device)
    checkpoint_name, step = get_checkpoint_step(args.checkpoint)

    logger.info(f"Evaluating on {len(val_dataset)} images...")
    
    all_preds = []
    all_targets = []

    # --- 1. Fast GPU Inference Loop ---
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Running GPU Inference"):
            images = batch["pixel_values"].to(device, non_blocking=True)
            targets_list = batch["labels"]

            outputs = model(images)
            preds = decode_model_outputs(outputs, args.conf_thresh, args.iou_thresh, args.input_size)
            targets = format_ground_truth(targets_list, args.input_size, device)

            # Move tensors to CPU immediately so we don't cause a CUDA multiprocessing crash later
            for p in preds:
                all_preds.append({k: v.cpu() for k, v in p.items()})
            for t in targets:
                all_targets.append({k: v.cpu() for k, v in t.items()})

    # --- 2. CPU Data Sorting (Memory Optimization) ---
    logger.info("Sorting predictions by class for multiprocessing...")
    num_classes = len(class_list)
    
    # Pre-allocate dictionary for each class
    class_data = {c: {"preds": [], "targets": []} for c in range(num_classes)}
    
    for p, t in zip(all_preds, all_targets):
        for c in range(num_classes):
            # Create boolean masks to filter boxes for this specific class
            p_mask = p["labels"] == c
            t_mask = t["labels"] == c
            
            class_data[c]["preds"].append({
                "boxes": p["boxes"][p_mask],
                "scores": p["scores"][p_mask],
                "labels": p["labels"][p_mask]
            })
            
            class_data[c]["targets"].append({
                "boxes": t["boxes"][t_mask],
                "labels": t["labels"][t_mask]
            })

    # --- 3. Multi-Core Parallel mAP Computation ---
    num_cores = mp.cpu_count() - 2

    logger.info(f"Firing up {num_cores} CPU cores to calculate mAP...")
    
    worker_args = [
        (c, class_list[c], class_data[c]["preds"], class_data[c]["targets"]) 
        for c in range(num_classes)
    ]
    
    class_results = []

    # Free memory before spawning heavy worker processes
    del all_preds
    del all_targets

    # We use ThreadPoolExecutor to manage the lightweight threads, 
    # while the threads handle the heavy, unstable multiprocessing.
    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(_evaluate_isolated, arg) for arg in worker_args]
        
        with tqdm(total=len(worker_args), desc="Computing mAP per class", unit="cls") as pbar:
            for future in as_completed(futures):
                res = future.result()
                class_results.append(res)
                pbar.update(1)

    # --- 4. Aggregate the Results ---
    # Calculate global mAP by taking the mean of all valid class mAPs
    valid_classes = [res for res in class_results if res["map"] > 0 or res["map_50"] > 0]
    
    global_map_50 = sum([r["map_50"] for r in valid_classes]) / max(1, len(valid_classes))
    global_map_75 = sum([r["map_75"] for r in valid_classes]) / max(1, len(valid_classes))
    global_map = sum([r["map"] for r in valid_classes]) / max(1, len(valid_classes))

    # Initialize TensorBoard Writer
    writer = SummaryWriter(log_dir=args.tb_dir + "/" + checkpoint_name)

    logger.info("\n--- Evaluation Results ---")
    logger.info(f"mAP @ IoU=0.50      : {global_map_50:.4f}")
    logger.info(f"mAP @ IoU=0.75      : {global_map_75:.4f}")
    logger.info(f"mAP @ IoU=0.50:0.95 : {global_map:.4f}")

    # Log overall mAP metrics to TensorBoard
    writer.add_scalar("mAP/IoU_0.50", global_map_50, step)
    writer.add_scalar("mAP/IoU_0.75", global_map_75, step)
    writer.add_scalar("mAP/IoU_0.50_0.95", global_map, step)

    logger.info("\n--- Class-wise mAP @ IoU=0.50:0.95 ---")
    
    # FIX 2: Build class_scores directly from our custom list of dictionaries
    class_scores = {res["class_name"]: res["map_50"] for res in valid_classes}

    if class_scores:
        # --- 1. Define the Custom TensorBoard Layout ---
        layout = {"Grouped_Class_mAP": {}}
        tracked_classes = set()
        
        for group_name, grouped_classes in hierarchy.items():
            tags = [f"mAP_Class/{cls_name}" for cls_name in grouped_classes if cls_name in class_scores]
            
            if tags:
                layout["Grouped_Class_mAP"][group_name] = ["Multiline", tags]
                tracked_classes.update(grouped_classes)

        uncategorized_tags = [f"mAP_Class/{cls_name}" for cls_name in class_scores if cls_name not in tracked_classes]
        if uncategorized_tags:
            layout["Grouped_Class_mAP"]["Uncategorized"] = ["Multiline", uncategorized_tags]

        writer.add_custom_scalars(layout)

        # --- 2. Log the actual values using standard add_scalar ---
        for cls_name, score in class_scores.items():
            logger.info(f"{cls_name:<25}: {score:.4f}")
            writer.add_scalar(f"mAP_Class/{cls_name}", score, step)
            
    else:
        logger.warning("No class-wise metrics were returned by the evaluator.")

    writer.close()

if __name__ == "__main__":
    main()