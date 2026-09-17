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

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
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
        self.architecture = architecture
        self.state_mode = "window_reset"
        self.n_nodes = int(hidden_size) * int(num_layers)
        self.node_order = [f"L{layer + 1}:h{unit + 1}" for layer in range(num_layers) for unit in range(hidden_size)]
        self.is_synthetic = False
        self.encoder = RecurrentEncoder(input_size, hidden_size, num_layers, architecture)
        self.head_type = head
        self.time_scale_s = float(time_scale_s)
        if head == "rul":
            self.head = RULHead(hidden_size, dropout)
        elif head == "weibull":
            self.head = WeibullHead(hidden_size, dropout)
        else:
            raise ValueError("head must be rul or weibull")

    def forward(self, x: torch.Tensor, lengths=None):
        h = self.encoder(x, lengths)
        if self.head_type == "rul":
            return self.head(h)
        return self.head(h)

    def predicted_rul_s(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "rul":
            norm = self.forward(x)
            return norm * self.time_scale_s
        lam, k = self.forward(x)
        return weibull_median_rul(lam, k, self.time_scale_s)


def build_model(
    *,
    architecture: str,
    input_size: int,
    hidden_size: int = 64,
    num_layers: int = 1,
    head: str = "rul",
    dropout: float = 0.1,
    time_scale_s: float = 1.0,
    graph=None,
    n_nodes: int | None = None,
    leak: float = 0.2,
    spectral_radius: float = 0.9,
    input_scale: float = 0.1,
    seed: int = 42,
    state_mode: str = "window_reset",
    provenance: dict | None = None,
    parent_provenance: dict | None = None,
    node_order=None,
    frozen_weights=None,
) -> nn.Module:
    """Construct GRU/LSTM ``PDMNet`` or a fly/random reservoir.

    Reservoirs require an in-memory ``graph`` (or ``frozen_weights`` from a saved
    ``weights.npz``). ``W_in`` / ``W_res`` / ``b_res`` are never rebuilt from seed
    when ``frozen_weights`` is set.
    """
    arch = str(architecture or "").strip().lower()
    if arch in {"gru", "lstm"}:
        return PDMNet(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            architecture=arch,
            head=head,
            dropout=dropout,
            time_scale_s=time_scale_s,
        )
    if arch == "fly_connectome_reservoir":
        from pdm.models.fly_reservoir import FlyConnectomeReservoir

        return FlyConnectomeReservoir(
            graph,
            input_size,
            head=head,
            leak=float(leak),
            spectral_radius=float(spectral_radius),
            input_scale=float(input_scale),
            seed=int(seed),
            state_mode=state_mode,
            time_scale_s=time_scale_s,
            node_order=node_order,
            n_nodes=n_nodes,
            provenance=provenance,
            frozen_weights=frozen_weights,
        )
    if arch == "random_reservoir":
        from pdm.models.random_reservoir import RandomReservoir

        return RandomReservoir(
            graph,
            input_size,
            head=head,
            leak=float(leak),
            spectral_radius=float(spectral_radius),
            input_scale=float(input_scale),
            seed=int(seed),
            state_mode=state_mode,
            time_scale_s=time_scale_s,
            node_order=node_order,
            n_nodes=n_nodes,
            provenance=provenance,
            parent_provenance=parent_provenance,
            frozen_weights=frozen_weights,
        )
    raise ValueError(f"Unknown architecture {architecture!r}")
