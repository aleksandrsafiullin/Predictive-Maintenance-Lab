"""Numeric, multi-horizon signal head over the existing causal recurrent encoder."""
from __future__ import annotations

import torch
from torch import nn

from pdm.models.recurrent import RecurrentEncoder


class SignalRecurrent(nn.Module):
    def __init__(self, architecture: str, input_size: int, hidden_size: int, n_horizons: int):
        super().__init__()
        self.encoder = RecurrentEncoder(input_size, hidden_size, architecture=architecture)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))
