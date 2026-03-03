import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import box_convert, complete_box_iou_loss

# ---------------------------------------------------------------------------
# Regression Loss
# ---------------------------------------------------------------------------

class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        # preds, targets shapes: (N, 4) where 4 is (cx, cy, w, h)
        
        # 1. Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        # We use the built-in PyTorch operator for safety and speed
        preds_xyxy = box_convert(preds, in_fmt='cxcywh', out_fmt='xyxy')
        targets_xyxy = box_convert(targets, in_fmt='cxcywh', out_fmt='xyxy')
        
        # 2. Compute official C++ optimized CIoU
        # This safely handles gradient instability, aspect ratio clipping, and division by zero
        loss = complete_box_iou_loss(preds_xyxy, targets_xyxy, reduction=self.reduction)
        
        return loss


# ---------------------------------------------------------------------------
# Classification Loss
# ---------------------------------------------------------------------------

class DynamicFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, base_gamma=2.0, max_gamma=3.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.base_gamma = base_gamma
        self.max_gamma = max_gamma
        self.reduction = reduction

    def forward(self, inputs, targets, progress=0.0):
        """
        progress: A float between 0.0 (start) and 1.0 (end).
        """
        current_gamma = self.base_gamma + (self.max_gamma - self.base_gamma) * progress
        
        # BCE with Logits is highly optimized and numerically stable
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        probs = torch.sigmoid(inputs)
        
        # THE MEMORY HACK: (1 - p_t) is mathematically identical to abs(targets - probs)
        # This avoids creating multiple massive intermediate tensors
        modulating_factor = torch.abs(targets - probs) ** current_gamma
        
        # THE SPEED HACK: torch.where is faster and lighter than arithmetic for binary masks
        alpha_weight = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
        
        # Combine pieces
        focal_loss = alpha_weight * modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss