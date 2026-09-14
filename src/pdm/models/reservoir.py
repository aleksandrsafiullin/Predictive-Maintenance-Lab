from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from pdm.losses import weibull_median_rul
from pdm.models.readout import LinearReadout

# Bearings reservoir forward uses ReLU on the affine readout with NO extra Softplus.
# That diverges from RULHead (Softplus) so a ridge solve on y_norm is exact.


def forward_states(
    inputs: torch.Tensor,
    W_in: torch.Tensor,
    W_res: torch.Tensor,
    b_res: torch.Tensor,
    alpha: float,
    x0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Leaky ESN state update. **This is the only tanh kernel** in the package.

    ``x[t] = (1 - alpha) * x[t-1] + alpha * tanh(W_res @ x[t-1] + W_in @ u[t] + b_res)``

    ``W_res[i, j]`` is the weight of edge ``j → i`` (source is column j). Do not transpose.

    ``inputs``: ``[T, input_size]`` or ``[B, T, input_size]``.
    Returns states ``[T, n_nodes]`` or ``[B, T, n_nodes]``.
    """
    squeeze_batch = False
    if inputs.dim() == 2:
        inputs = inputs.unsqueeze(0)
        squeeze_batch = True
        if x0 is not None and x0.dim() == 1:
            x0 = x0.unsqueeze(0)
    elif inputs.dim() != 3:
        raise ValueError(f"inputs must be [T, F] or [B, T, F], got shape {tuple(inputs.shape)}")
    batch, n_steps, _feat = inputs.shape
    n_nodes = W_res.shape[0]
    if x0 is None:
        x = inputs.new_zeros(batch, n_nodes)
    else:
        x = x0.to(dtype=inputs.dtype, device=inputs.device)
        if x.dim() == 1:
            x = x.unsqueeze(0).expand(batch, -1).contiguous()
        elif x.shape[0] == 1 and batch > 1:
            x = x.expand(batch, -1).contiguous()
    leak = float(alpha)
    keep = 1.0 - leak
    frames: list[torch.Tensor] = []
    for t in range(n_steps):
        u_t = inputs[:, t, :]
        # F.linear(x, W_res) = x @ W_res.T = (W_res @ x.T).T — column j is the source.
        preact = F.linear(x, W_res, None) + F.linear(u_t, W_in, b_res)
        x = keep * x + leak * torch.tanh(preact)
        frames.append(x)
    states = torch.stack(frames, dim=1)
    if squeeze_batch:
        return states.squeeze(0)
    return states


class LeakyESN(nn.Module):
    """Leaky Echo State Network: fixed W_in, W_res, b_res; trainable readout."""

    def __init__(
        self,
        W_in: torch.Tensor | None = None,
        W_res: torch.Tensor | None = None,
        b_res: torch.Tensor | None = None,
        *,
        alpha: float = 0.2,
        state_mode: str = "window_reset",
        head: str = "rul",
        time_scale_s: float = 1.0,
        readout: LinearReadout | None = None,
    ) -> None:
        super().__init__()
        if W_in is None or W_res is None or b_res is None:
            raise ValueError("W_in, W_res, and b_res are required")
        w_in = _as_float_tensor(W_in)
        w_res = _as_float_tensor(W_res)
        bias = _as_float_tensor(b_res)
        if w_res.ndim != 2 or w_res.shape[0] != w_res.shape[1]:
            raise ValueError(f"W_res must be square [N, N], got {tuple(w_res.shape)}")
        n_nodes = int(w_res.shape[0])
        if w_in.ndim != 2 or w_in.shape[0] != n_nodes:
            raise ValueError(f"W_in must be [N, F] with N={n_nodes}, got {tuple(w_in.shape)}")
        if bias.ndim != 1 or bias.shape[0] != n_nodes:
            raise ValueError(f"b_res must be [N] with N={n_nodes}, got {tuple(bias.shape)}")
        head_norm = str(head or "").strip().lower()
        if head_norm not in {"rul", "weibull"}:
            raise ValueError("head must be rul or weibull")
        mode = str(state_mode or "window_reset").strip().lower()
        if mode not in {"window_reset", "continuous"}:
            raise ValueError(f"unsupported state_mode={state_mode!r}")
        self.alpha = float(alpha)
        self.leak = self.alpha
        self.state_mode = mode
        self.rul_transform = "linear"
        self.rul_reference_s = 60.0
        self.head_type = head_norm
        self.time_scale_s = float(time_scale_s)
        self.n_nodes = n_nodes
        self.input_size = int(w_in.shape[1])
        # Frozen reservoir matrices — never nn.Parameter (AdamW must not see them).
        self.register_buffer("W_in", w_in)
        self.register_buffer("W_res", w_res)
        self.register_buffer("b_res", bias)
        out_features = 1 if head_norm == "rul" else 2
        self.readout = readout if readout is not None else LinearReadout(n_nodes, self.input_size, out_features)

    def forward_states(self, inputs: torch.Tensor, x0: torch.Tensor | None = None) -> torch.Tensor:
        """
        inputs: [T, input_size] or [B, T, input_size]
        returns: states [T, n_nodes] or [B, T, n_nodes]

        x[t] = (1-alpha)*x[t-1] + alpha*tanh(W_res @ x[t-1] + W_in @ u[t] + b_res)
        W_res[i,j] = edge j→i (source is j, target is i)
        state_mode='window_reset' zeros x at start of each window unless ``x0`` is given.
        """
        if self.state_mode == "window_reset" and x0 is None:
            x0 = None
        return forward_states(inputs, self.W_in, self.W_res, self.b_res, self.alpha, x0)

    def forward_raw(self, x_T: torch.Tensor, u_T: torch.Tensor) -> torch.Tensor:
        """Raw affine output before ReLU / Softplus (contribution identity uses this)."""
        return self.readout.forward_raw(x_T, u_T)

    def _postprocess(self, raw: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.head_type == "rul":
            if self.rul_transform == "log1p":
                value = torch.expm1(torch.clamp(raw, min=0.0, max=20.0)) * self.rul_reference_s / self.time_scale_s
                return value if raw.dim() == 1 else value.squeeze(-1)
            if self.rul_transform != "linear":
                raise ValueError(f"Unsupported RUL transform: {self.rul_transform}")
            # ReLU, not Softplus — diverges from RULHead so ridge on y_norm is exact.
            if raw.dim() == 1:
                return F.relu(raw)
            return F.relu(raw).squeeze(-1)
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        lam = F.softplus(raw[:, 0]) + 1e-6
        k = F.softplus(raw[:, 1]) + 1e-6
        return lam, k

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Process a batch of windows.

        Bearings / ``head='rul'``: 1D ``[B]`` non-negative normalized RUL as
        ``relu(W_out @ x_T + W_u @ u_T + b)`` — **no extra Softplus**.
        Filters / ``head='weibull'``: tuple ``(lam, k)`` after Softplus like WeibullHead.
        """
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, F] input, got shape {tuple(x.shape)}")
        states = self.forward_states(x)
        return self._postprocess(self.forward_raw(states[:, -1, :], x[:, -1, :]))

    def forward_with_states(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor | tuple[torch.Tensor, torch.Tensor], list[torch.Tensor]]:
        """Return ``(prediction, states_list)`` where each list item is ``[T, n_nodes]``."""
        if x.dim() != 3:
            raise ValueError(f"expected [B, T, F] input, got shape {tuple(x.shape)}")
        states = self.forward_states(x)
        pred = self._postprocess(self.forward_raw(states[:, -1, :], x[:, -1, :]))
        states_list = [states[i] for i in range(states.shape[0])]
        return pred, states_list

    def predicted_rul_s(self, x: torch.Tensor) -> torch.Tensor:
        if self.head_type == "rul":
            return self.forward(x) * self.time_scale_s
        lam, k = self.forward(x)
        return weibull_median_rul(lam, k, self.time_scale_s)


def _as_float_tensor(value: torch.Tensor | object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).contiguous()
    return torch.tensor(value, dtype=torch.float32)
