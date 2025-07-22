import torch
from .model import ImprovedSTNet

def get_device() -> torch.device:
    """Get the appropriate device (CUDA if available, else CPU).

    Returns:
        torch.device: Device to use for computations.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def initialize_model(input_size: int, num_classes: int, hidden_size: int = 128,
                     dropout: float = 0.5, device: torch.device = None) -> ImprovedSTNet:
    """Initialize the ImprovedSTNet model.

    Args:
        input_size (int): Number of input features per time step.
        num_classes (int): Number of classes for classification output.
        hidden_size (int, optional): Size of hidden layers. Default: 128.
        dropout (float, optional): Dropout probability. Default: 0.5.
        device (torch.device, optional): Device to initialize model on. Default: None (uses get_device()).

    Returns:
        ImprovedSTNet: Initialized model.
    """
    if device is None:
        device = get_device()
    model = ImprovedSTNet(input_size, num_classes, hidden_size, dropout).to(device)
    return model