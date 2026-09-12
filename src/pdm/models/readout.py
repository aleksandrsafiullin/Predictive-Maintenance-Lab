from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LinearReadout(nn.Module):
    """Trainable linear readout. Weights: W_x (n_nodes), W_u (input_size), b.

    raw = W_x @ x_T + W_u @ u_T + b  # direct input term required
    """

    def __init__(self, n_nodes: int, input_size: int, out_features: int = 1) -> None:
        super().__init__()
        n_nodes = int(n_nodes)
        input_size = int(input_size)
        out_features = int(out_features)
        if n_nodes < 1 or input_size < 1 or out_features < 1:
            raise ValueError("n_nodes, input_size, and out_features must be positive")
        self.n_nodes = n_nodes
        self.input_size = input_size
        self.out_features = out_features
        self.W_x = nn.Parameter(torch.empty(out_features, n_nodes))
        self.W_u = nn.Parameter(torch.empty(out_features, input_size))
        self.b = nn.Parameter(torch.empty(out_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1.0 / max(self.n_nodes, 1) ** 0.5
        nn.init.uniform_(self.W_x, -bound, bound)
        nn.init.uniform_(self.W_u, -bound, bound)
        nn.init.uniform_(self.b, -bound, bound)

    def forward_raw(self, x_T: torch.Tensor, u_T: torch.Tensor) -> torch.Tensor:
        """Affine readout before any activation: [B, out] or [out] if unbatched."""
        squeeze = False
        if x_T.dim() == 1:
            x_T = x_T.unsqueeze(0)
            u_T = u_T.unsqueeze(0)
            squeeze = True
        raw = F.linear(x_T, self.W_x, None) + F.linear(u_T, self.W_u, self.b)
        if squeeze:
            return raw.squeeze(0)
        return raw

    def forward(self, x_T: torch.Tensor, u_T: torch.Tensor) -> torch.Tensor:
        return self.forward_raw(x_T, u_T)

    def load_ridge_vector(self, w: np.ndarray) -> None:
        """Load a 1D ridge solution ``[n_nodes + input_size + 1]`` into ``W_x``, ``W_u``, ``b``."""
        if self.out_features != 1:
            raise ValueError("load_ridge_vector is for a 1-output (bearings) readout")
        vec = np.asarray(w, dtype=np.float64).reshape(-1)
        expected = self.n_nodes + self.input_size + 1
        if vec.size != expected:
            raise ValueError(f"ridge vector has {vec.size} entries, expected {expected}")
        with torch.no_grad():
            self.W_x.copy_(
                torch.tensor(vec[: self.n_nodes], dtype=self.W_x.dtype, device=self.W_x.device).unsqueeze(0)
            )
            self.W_u.copy_(
                torch.tensor(
                    vec[self.n_nodes : self.n_nodes + self.input_size],
                    dtype=self.W_u.dtype,
                    device=self.W_u.device,
                ).unsqueeze(0)
            )
            self.b.copy_(torch.tensor(vec[-1:], dtype=self.b.dtype, device=self.b.device))


def fit_ridge(Z: np.ndarray, y: np.ndarray, alpha: float = 0.001) -> tuple[np.ndarray, float]:
    """Solve ``(Z^T Z + alpha I) w = Z^T y``. Returns ``(w, residual_mean_sq)``.

    ``Z``: ``[N, n_nodes + input_size + 1]`` (state concat input concat 1).
    ``y``: ``[N]`` normalized RUL (NOT raw ``target_rul_s``).
    """
    z = np.asarray(Z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {z.shape}")
    target = np.asarray(y, dtype=np.float64).reshape(-1)
    if z.shape[0] != target.shape[0]:
        raise ValueError(f"Z rows {z.shape[0]} != y length {target.shape[0]}")
    n_features = z.shape[1]
    gram = z.T @ z + float(alpha) * np.eye(n_features, dtype=np.float64)
    rhs = z.T @ target
    try:
        w = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(gram, rhs, rcond=None)
    residual = target - z @ w
    residual_mean_sq = float(np.mean(residual * residual))
    return w, residual_mean_sq
