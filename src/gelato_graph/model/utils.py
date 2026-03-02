import logging
from pathlib import Path

import torch
from torchvision.ops import batched_nms
from transformers.trainer_utils import get_last_checkpoint

logger = logging.getLogger(__name__)

class RatioSampler(torch.utils.data.Sampler):
    """
    Samples from multiple datasets with specific ratios.
    For example, ratios=[1, 4] means 1 sample from dataset 0 for every 4 from dataset 1.
    """
    def __init__(self, dataset_lengths, ratios):
        self.dataset_lengths = dataset_lengths
        self.ratios = ratios
        
        self.offsets = [0]
        for i in range(len(dataset_lengths) - 1):
            self.offsets.append(self.offsets[-1] + dataset_lengths[i])
            
        # Base length is determined by the bottleneck dataset to prevent over-inflating epochs.
        # This guarantees 1 pass of the limiting dataset + proportionally matching samples from the others.
        self.base_len = min([max(1, l // r) for l, r in zip(dataset_lengths, ratios) if r > 0])
        self.total_size = sum([self.base_len * r for r in ratios])
        
    def __iter__(self):
        iters = []
        for i, l in enumerate(self.dataset_lengths):
            if l == 0:
                raise ValueError(f"Dataset at index {i} has length 0, which is not supported by RatioSampler.")
            num_samples = self.base_len * self.ratios[i]
            if num_samples > l:
                repeats = (num_samples // l) + 1
                perm = torch.cat([torch.randperm(l) for _ in range(repeats)])[:num_samples]
            else:
                perm = torch.randperm(l)[:num_samples]
            iters.append(iter(perm.tolist()))
            
        for _ in range(self.base_len):
            for i, r in enumerate(self.ratios):
                for _ in range(r):
                    yield next(iters[i]) + self.offsets[i]
                    
    def __len__(self):
        return self.total_size

def omr_collate_fn(batch):
    """Collate images and targets into a dictionary for Trainer.
    
    Returns:
        dict: {
            "pixel_values": (B, 3, H, W) float32 tensor
            "labels": list[dict] of length B, annotations
        }
    """
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return {
        "pixel_values": images,
        "labels": list(targets)
    }


def load_checkpoint(model, checkpoint_path: str, device: torch.device, eval: bool = True):
    """Resolve a checkpoint path and load weights into the model.
    
    Handles:
      - HF Trainer output dirs (finds the last checkpoint-NNNN subdirectory)
      - Direct paths to model.safetensors or pytorch_model.bin
      - Single-run directories containing weight files
    
    Args:
        eval: If True (default), sets the model to eval mode. Set to False for training.
    """
    ckpt = Path(checkpoint_path)
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

    model.to(device)
    if eval:
        model.eval()

def decode_model_outputs(outputs, conf_thresh, iou_thresh, input_size):
    """
    Decodes multi-scale raw model outputs into absolute pixel bounding boxes.
    Handles dynamic batch sizes and dynamic feature map resolutions.
    
    Returns:
        list[dict]: A list of length BatchSize, containing dicts with "boxes", "scores", and "labels".
    """
    batch_size = outputs[0]["cls"].shape[0]
    device = outputs[0]["cls"].device
    
    all_boxes = [[] for _ in range(batch_size)]
    all_scores = [[] for _ in range(batch_size)]
    all_labels = [[] for _ in range(batch_size)]

    for out in outputs:
        cls_map = out["cls"].sigmoid()  
        reg_map = out["reg"]            
        
        B, _, H, W = reg_map.shape

        max_scores, max_classes = cls_map.max(dim=1) 
        mask = max_scores >= conf_thresh
        pos = mask.nonzero(as_tuple=False)
        
        if pos.numel() == 0:
            continue
            
        b_idx, y_idx, x_idx = pos[:, 0], pos[:, 1], pos[:, 2]

        scores = max_scores[b_idx, y_idx, x_idx]
        cls_idx = max_classes[b_idx, y_idx, x_idx]

        tx = reg_map[b_idx, 0, y_idx, x_idx].sigmoid()
        ty = reg_map[b_idx, 1, y_idx, x_idx].sigmoid()
        tw = reg_map[b_idx, 2, y_idx, x_idx].exp()
        th = reg_map[b_idx, 3, y_idx, x_idx].exp()

        cx_n = (tx + x_idx) / W
        cy_n = (ty + y_idx) / H

        cx = cx_n * input_size
        cy = cy_n * input_size
        bw = tw * input_size
        bh = th * input_size

        x1 = cx - bw / 2
        y1 = cy - bh / 2
        x2 = cx + bw / 2
        y2 = cy + bh / 2

        boxes = torch.stack([x1, y1, x2, y2], dim=1)
        
        for i in range(len(b_idx)):
            batch_item = b_idx[i].item()
            all_boxes[batch_item].append(boxes[i:i+1])
            all_scores[batch_item].append(scores[i:i+1])
            all_labels[batch_item].append(cls_idx[i:i+1])

    batch_preds = []
    for b in range(batch_size):
        if len(all_boxes[b]) > 0:
            b_boxes = torch.cat(all_boxes[b], dim=0)
            b_scores = torch.cat(all_scores[b], dim=0)
            b_labels = torch.cat(all_labels[b], dim=0)

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