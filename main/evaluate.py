import torch
from torch.utils.data import DataLoader
from .model import ImprovedSTNet
from .loss import DynamicWeightedLoss
from typing import Dict, Any

def evaluate_model(model: ImprovedSTNet, loader: DataLoader, criterion: DynamicWeightedLoss,
                  device: torch.device) -> Dict[str, Any]:
    """Evaluate the model on a dataset.

    Args:
        model (ImprovedSTNet): The model to evaluate.
        loader (DataLoader): DataLoader for the dataset.
        criterion (DynamicWeightedLoss): Loss function.
        device (torch.device): Device for computation.

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics (loss, accuracy).
    """
    model.eval()
    total_loss = 0.0
    cls_correct = 0
    total_samples = 0

    with torch.no_grad():
        for features, cls_labels, reg_labels in loader:
            features = features.to(device)
            cls_labels = cls_labels.to(device).squeeze()
            reg_labels = reg_labels.to(device)

            cls_pred, reg_pred = model(features)
            loss = criterion(cls_pred, reg_pred, cls_labels, reg_labels)
            total_loss += loss.item() * features.size(0)

            cls_pred = cls_pred.argmax(dim=1)
            cls_correct += (cls_pred == cls_labels).sum().item()
            total_samples += features.size(0)

    avg_loss = total_loss / total_samples
    cls_accuracy = cls_correct / total_samples

    return {
        'loss': avg_loss,
        'cls_accuracy': cls_accuracy
    }