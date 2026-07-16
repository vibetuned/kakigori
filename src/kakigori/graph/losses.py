# Third party imports
import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
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
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")

        # Get the probabilities of the true class
        pt = torch.exp(-ce_loss)

        # Focal modulating factor: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# Third party imports
import torch.nn as nn
import torch.nn.functional as F


class MultiClassEdgeFocalLoss(nn.Module):
    def __init__(self, alpha_weights, gamma=2.0, reduction="mean"):
        """
        alpha_weights: A 1D tensor of length C (number of classes)
        """
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha_weights)

    def forward(self, logits, targets):
        # logits: (E, 5)
        # targets: (E)

        # Weighted CE for the loss magnitude...
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")

        # ...but the focal factor must come from the TRUE probability p_t,
        # i.e. the unweighted CE. Deriving it from the weighted CE makes
        # pt ~ exp(-alpha*CE): for a low-alpha class even a wrong prediction
        # looks "easy" and its gradient is suppressed twice over.
        with torch.no_grad():
            pt = torch.exp(-F.cross_entropy(logits, targets, reduction="none"))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss
