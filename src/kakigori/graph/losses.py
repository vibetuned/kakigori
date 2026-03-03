# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # Alpha is a tensor of weights for each class (0 through 4)
        # e.g., torch.tensor([0.1, 1.0, 1.0, 1.0, 1.0]) to heavily penalize Class 0
        self.alpha = alpha 

    def forward(self, inputs, targets):
        """
        inputs: Shape (E, 5) - The raw logits from the GNN edge classifier
        targets: Shape (E) - The ground truth edge classes (0 to 4)
        """
        # Standard cross entropy gives us the negative log likelihood
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        
        # Get the probabilities of the true class
        pt = torch.exp(-ce_loss)
        
        # Focal modulating factor: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss