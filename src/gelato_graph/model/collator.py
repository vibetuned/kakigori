"""Custom collate function for the OMR object-detection dataset."""

import torch


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
