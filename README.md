# PursuitShift
A deep learning framework specifically designed for real-time classification and prediction of the search-to-pursuit transition in mice predation behavior.



<img width="554" height="261" alt="image" src="https://github.com/user-attachments/assets/44b4d57e-0761-4411-8e6a-5f8c941d54ae" />


STNet: Improved Spatio-Temporal Network
STNet is a PyTorch implementation of a spatio-temporal neural network designed for sequence modeling tasks, combining temporal processing, GRU, attention, and convolutional layers. It supports both classification and regression outputs, making it suitable for tasks involving time-series data with spatial components.
Installation
pip install stnet

Or install from source:
git clone https://github.com/Kweifeng/PursuitShift.git
cd stnet
pip install .

Requirements

Python >= 3.7
PyTorch >= 1.9.0

See requirements.txt for details.
Usage
import torch
from stnet.model import ImprovedSTNet
from stnet.utils import get_device

# Initialize model
device = get_device()
model = ImprovedSTNet(input_size=9, num_classes=3).to(device)

# Example input: batch_size=16, seq_len=12, input_size=9
x = torch.randn(16, 12, 9).to(device)

# Forward pass
cls_pred, reg_pred = model(x)
print(cls_pred.shape)  # [16, 3]
print(reg_pred.shape)  # [16, 2]

See examples/example_usage.py for a complete example.
Modules

SpatioTemporalAttention: Self-attention mechanism for capturing relationships across time steps.
ImprovedSTNet: Main model combining temporal, GRU, attention, and convolutional layers.
DynamicWeightedLoss: Custom loss function for combined classification and regression tasks.

License
This project is licensed under the MIT License. See the LICENSE file for details.
Contributing
Contributions are welcome! Please open an issue or submit a pull request on GitHub.
