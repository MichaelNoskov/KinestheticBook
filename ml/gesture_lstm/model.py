from __future__ import annotations

import torch
import torch.nn as nn


class GestureLSTM(nn.Module):
    def __init__(
        self,
        n_features: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        num_classes: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            n_features,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


class GestureCNNLSTM(nn.Module):
    """Conv1D front-end plus LSTM classifier."""

    def __init__(
        self,
        n_features: int = 6,
        conv_channels: int = 64,
        hidden_size: int = 64,
        num_layers: int = 1,
        num_classes: int = 3,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, conv_channels // 2, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(conv_channels // 2, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            conv_channels,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)
