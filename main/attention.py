import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalAttention(nn.Module):
    """Spatio-Temporal Attention mechanism for sequence data.

    This module applies self-attention to capture relationships across time steps
    and feature dimensions in spatio-temporal data.

    Args:
        in_dim (int): Input feature dimension.
    """
    def __init__(self, in_dim: int):
        super().__init__()
        self.query = nn.Linear(in_dim, in_dim)
        self.key = nn.Linear(in_dim, in_dim)
        self.value = nn.Linear(in_dim, in_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention mechanism.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, in_dim).

        Returns:
            torch.Tensor: Output tensor after attention, same shape as input.
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = torch.softmax(Q @ K.transpose(1, 2) / (x.size(-1) ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ V