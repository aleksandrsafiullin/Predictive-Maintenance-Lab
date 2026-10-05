"""Causal discrete RED-entry hazard heads over real-length recurrent histories."""

import torch
from torch import nn

from pdm.models.recurrent import RecurrentEncoder


class RedEntryRecurrent(nn.Module):
    def __init__(self, architecture, input_size, hidden_size, n_horizons):
        super().__init__()
        self.encoder = RecurrentEncoder(input_size, hidden_size, architecture=architecture)
        self.head = nn.Linear(hidden_size, n_horizons)

    def forward(self, x, lengths):
        if torch.any(lengths < 1) or torch.any(lengths > x.shape[1]):
            raise ValueError("Actual sequence lengths must be within stored window")
        return self.head(self.encoder(x, lengths))
