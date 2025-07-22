import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicWeightedLoss(nn.Module):
    """Dynamic weighted loss combining classification and regression losses.

    Applies cross-entropy loss for classification and binary cross-entropy for
    regression tasks, with configurable weights for chase and attack predictions.

    Args:
        class_weights (torch.Tensor): Weights for classification loss (shape: [num_classes]).
        chase_weight (float, optional): Weight for positive chase predictions. Default: 10.0.
        attack_weight (float, optional): Weight for positive attack predictions. Default: 15.0.
        reg_weight (float, optional): Weight for regression loss. Default: 0.5.
        label_smoothing (float, optional): Label smoothing for classification. Default: 0.1.
    """
    def __init__(self, class_weights: torch.Tensor, chase_weight: float = 10.0,
                 attack_weight: float = 15.0, reg_weight: float = 0.5, label_smoothing: float = 0.1):
        super().__init__()
        self.cls_criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
        self.chase_weight = chase_weight
        self.attack_weight = attack_weight
        self.reg_weight = reg_weight

    def forward(self, cls_pred: torch.Tensor, reg_pred: torch.Tensor,
                cls_true: torch.Tensor, reg_true: torch.Tensor) -> torch.Tensor:
        """Compute the combined loss.

        Args:
            cls_pred (torch.Tensor): Classification logits (batch_size, num_classes).
            reg_pred (torch.Tensor): Regression predictions (batch_size, 2).
            cls_true (torch.Tensor): True class labels (batch_size,).
            reg_true (torch.Tensor): True regression labels (batch_size, 2).

        Returns:
            torch.Tensor: Combined loss value.
        """
        cls_loss = self.cls_criterion(cls_pred, cls_true.squeeze())
        chase_loss = F.binary_cross_entropy_with_logits(
            reg_pred[:, 0], reg_true[:, 0], weight=torch.where(reg_true[:, 0] == 1, self.chase_weight, 1.0)
        )
        attack_loss = F.binary_cross_entropy_with_logits(
            reg_pred[:, 1], reg_true[:, 1], weight=torch.where(reg_true[:, 1] == 1, self.attack_weight, 1.0)
        )
        return cls_loss + self.reg_weight * (chase_loss + attack_loss)