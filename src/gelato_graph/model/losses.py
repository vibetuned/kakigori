import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits from the classification branch (Batch, Classes, H, W)
        # targets: one-hot encoded ground truth (Batch, Classes, H, W)
        
        # Use BCEWithLogits for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Get the predicted probabilities (sigmoid of logits)
        probs = torch.sigmoid(inputs)
        
        # Calculate the modulating factor: (1 - p_t)^gamma
        # p_t is the probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # Apply alpha weighting (optional, balances positive/negative examples)
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Combine pieces
        focal_loss = alpha_weight * modulating_factor * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss



class CIoULoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, preds, targets):
        # preds, targets shapes: (Batch, 4, H, W) where 4 is (cx, cy, w, h)
        
        # Clamp w,h to a small positive value for numerical stability
        # Use functional approach (no in-place ops) to keep autograd happy
        pred_cx = preds[:, 0:1]
        pred_cy = preds[:, 1:2]
        pred_w = preds[:, 2:3].clamp(min=1e-6)
        pred_h = preds[:, 3:4].clamp(min=1e-6)
        preds = torch.cat([pred_cx, pred_cy, pred_w, pred_h], dim=1)
        
        # 1. Convert (cx, cy, w, h) to (x1, y1, x2, y2) for intersection math
        preds_x1 = preds[:, 0] - preds[:, 2] / 2
        preds_y1 = preds[:, 1] - preds[:, 3] / 2
        preds_x2 = preds[:, 0] + preds[:, 2] / 2
        preds_y2 = preds[:, 1] + preds[:, 3] / 2
        
        tgt_x1 = targets[:, 0] - targets[:, 2] / 2
        tgt_y1 = targets[:, 1] - targets[:, 3] / 2
        tgt_x2 = targets[:, 0] + targets[:, 2] / 2
        tgt_y2 = targets[:, 1] + targets[:, 3] / 2
        
        # 2. Calculate standard IoU
        inter_x1 = torch.max(preds_x1, tgt_x1)
        inter_y1 = torch.max(preds_y1, tgt_y1)
        inter_x2 = torch.min(preds_x2, tgt_x2)
        inter_y2 = torch.min(preds_y2, tgt_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        preds_area = preds[:, 2] * preds[:, 3]
        tgt_area = targets[:, 2] * targets[:, 3]
        union_area = preds_area + tgt_area - inter_area + 1e-16
        
        iou = inter_area / union_area
        
        # 3. Calculate Center Distance Penalty
        center_dist_sq = (preds[:, 0] - targets[:, 0])**2 + (preds[:, 1] - targets[:, 1])**2
        
        # Find the smallest enclosing box
        enclose_x1 = torch.min(preds_x1, tgt_x1)
        enclose_y1 = torch.min(preds_y1, tgt_y1)
        enclose_x2 = torch.max(preds_x2, tgt_x2)
        enclose_y2 = torch.max(preds_y2, tgt_y2)
        enclose_diagonal_sq = (enclose_x2 - enclose_x1)**2 + (enclose_y2 - enclose_y1)**2 + 1e-16
        
        # 4. Calculate Aspect Ratio Penalty (v)
        atan_preds = torch.atan(preds[:, 2] / (preds[:, 3] + 1e-16))
        atan_tgts = torch.atan(targets[:, 2] / (targets[:, 3] + 1e-16))
        v = (4 / (math.pi ** 2)) * (atan_tgts - atan_preds)**2
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + 1e-16)
            
        # 5. Final CIoU
        ciou = iou - (center_dist_sq / enclose_diagonal_sq) - alpha * v
        ciou_loss = 1 - ciou
        
        if self.reduction == 'mean':
            return ciou_loss.mean()
        return ciou_loss.sum()