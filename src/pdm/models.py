from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pdm.losses import weibull_median_rul


class RecurrentEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        architecture: str = "gru",
    ) -> None:
        super().__init__()
        arch = architecture.lower()
        if arch not in {"gru", "lstm"}:
            raise ValueError("architecture must be gru or lstm")
        self.architecture = arch
        rnn_cls = nn.GRU if arch == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, h = self.rnn(x)
        if self.architecture == "lstm":
            h = h[0]
        return h[-1]


class RULHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(hidden_size // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_size // 2, 8), 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(h)).squeeze(-1)


class WeibullHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(hidden_size // 2, 8)),
            nn.ReLU(),
            nn.Linear(max(hidden_size // 2, 8), 2),
        )

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(h)
        lam = F.softplus(raw[:, 0]) + 1e-6
        k = F.softplus(raw[:, 1]) + 1e-6
        return lam, k


class PDMNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        architecture: str = "gru",
        head: str = "rul",
        dropout: float = 0.1,
        time_scale_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.encoder = RecurrentEncoder(input_size, hidden_size, num_layers, architecture)
        self.head_type = head
        self.time_scale_s = float(time_scale_s)
        if head == "rul":
            self.head = RULHead(hidden_size, dropout)
        elif head == "weibull":
            self.head = WeibullHead(hidden_size, dropout)
        else:
            raise ValueError("head must be rul or weibull")

    def forward(self, x: torch.Tensor):
        h = self.encoder(x)
        if self.head_type == "rul":
            return self.head(h)
        return self.head(h)

    def predicted_rul_s(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "rul":
            norm = self.forward(x)
            return norm * self.time_scale_s
        lam, k = self.forward(x)
        return weibull_median_rul(lam, k, self.time_scale_s)
