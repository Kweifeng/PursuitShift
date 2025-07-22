# PursuitShift
A deep learning framework specifically designed for real-time classification and prediction of the search-to-pursuit transition in mice predation behavior.



<img width="554" height="261" alt="image" src="https://github.com/user-attachments/assets/44b4d57e-0761-4411-8e6a-5f8c941d54ae" />


STNet: Improved Spatio-Temporal Network
STNet is a PyTorch implementation of a spatio-temporal neural network designed for sequence modeling tasks, combining temporal processing, GRU, attention, and convolutional layers. It supports both classification and regression outputs, making it suitable for tasks involving time-series data with spatial components.
Installation
pip install stnet

Or install from source:
git clone https://github.com/Kweifeng/PursuitShift.git

# Requirements

Python >= 3.7
PyTorch >= 2.0.4

See requirements.txt for details.
Usage
import torch
from stnet.model import ImprovedSTNet
from stnet.utils import get_device

# License
This project is licensed under the MIT License.
See the LICENSE file for details.
