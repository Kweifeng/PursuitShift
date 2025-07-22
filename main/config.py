from dataclasses import dataclass
from typing import List
import torch

@dataclass
class ModelConfig:
    """Configuration for the ImprovedSTNet model and training.

    Attributes:
        input_size (int): Number of input features per time step.
        num_classes (int): Number of classes for classification.
        hidden_size (int): Size of hidden layers.
        dropout (float): Dropout probability.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for optimizer.
        epochs (int): Number of training epochs.
        class_weights (List[float]): Weights for classification loss.
        chase_weight (float): Weight for chase regression loss.
        attack_weight (float): Weight for attack regression loss.
        reg_weight (float): Weight for regression loss.
        label_smoothing (float): Label smoothing for classification loss.
        device (torch.device): Device for computation.
    """
    input_size: int
    num_classes: int
    hidden_size: int = 128
    dropout: float = 0.5
    batch_size: int = 16
    learning_rate: float = 1e-5
    epochs: int = 150
    class_weights: List[float] = None
    chase_weight: float = 10.0
    attack_weight: float = 15.0
    reg_weight: float = 0.5
    label_smoothing: float = 0.1
    device: torch.device = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.class_weights is None:
            self.class_weights = [1.0] * self.num_classes
        self.class_weights = torch.tensor(self.class_weights).to(self.device)