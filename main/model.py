import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SpatioTemporalAttention

class ImprovedSTNet(nn.Module):
    """Improved Spatio-Temporal Network for sequence classification and regression.

    Combines temporal processing, GRU, attention, and convolutional layers to model
    spatio-temporal data, producing both classification and regression outputs.

    Args:
        input_size (int): Number of input features per time step.
        num_classes (int): Number of classes for classification output.
        hidden_size (int, optional): Size of hidden layers. Default: 128.
        dropout (float, optional): Dropout probability. Default: 0.5.
    """
    def __init__(self, input_size: int, num_classes: int, hidden_size: int = 128, dropout: float = 0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, hidden_size)
        )
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.attention = SpatioTemporalAttention(hidden_size)
        self.conv_block = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, groups=1),
            nn.GELU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.res_conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the ImprovedSTNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Classification logits (batch_size, num_classes)
                and regression outputs (batch_size, 2).
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(-1, x.size(-1))
        temporal_features = self.temporal(x)
        temporal_features = temporal_features.view(batch_size, seq_len, -1)
        gru_out, _ = self.gru(temporal_features)
        attn_out = self.attention(gru_out)
        conv_input = attn_out.permute(0, 2, 1)
        conv_out = self.conv_block(conv_input)
        residual = self.res_conv(conv_input)
        conv_out = conv_out + residual
        pooled = F.adaptive_avg_pool1d(conv_out, 1).squeeze(2)
        return self.classifier(pooled), self.regressor(pooled)