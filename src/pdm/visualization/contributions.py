"""Contribution decomposition for reservoir readout.

Ridge (bearings) is solved in normalized RUL space (y = target_rul_s / time_scale_s),
matching Smooth L1. The identity test is NOT in seconds and NOT after Softplus.

At each frame t, using the pre-display linear output:
  intercept = b (broadcast to [out_features])
  input_contributions[j] = W_u[:, j] * u[t, j]  (shape [out_features])
  neuron_contributions[i] = W_x[:, i] * x[t, i]  (shape [out_features])

Sum: intercept + sum(input_contributions) + sum(neuron_contributions) == raw[t]
within 1e-5. Do NOT use softplus(raw), Weibull median, or raw * time_scale_s in the sum.
"""

from __future__ import annotations

import numpy as np

from pdm.models.readout import LinearReadout


def decompose_contributions(
    states: np.ndarray,
    inputs: np.ndarray,
    readout: LinearReadout,
    *,
    lazy: bool = False,
) -> dict[str, np.ndarray]:
    """Decompose pre-activation linear readout into intercept / input / neuron terms.

    Returns {
        'intercept': np.ndarray [out_features],
        'input': np.ndarray [T, F, out_features],    # input_contributions[t,j] = W_u[:,j]*u[t,j]
        'neuron': np.ndarray [T, N, out_features],   # neuron_contributions[t,i] = W_x[:,i]*x[t,i]
        'raw': np.ndarray [T, out_features],         # raw linear output pre-activation
    }
    # Verify: for each t: intercept + sum(input[t]) + sum(neuron[t]) == raw[t] within 1e-5
    """
    x = np.asarray(states, dtype=np.float64)
    u = np.asarray(inputs, dtype=np.float64)
    if x.ndim != 2 or u.ndim != 2:
        raise ValueError(f"states must be [T, N] and inputs [T, F], got {x.shape} and {u.shape}")
    if x.shape[0] != u.shape[0]:
        raise ValueError(f"states frames {x.shape[0]} != inputs frames {u.shape[0]}")
    w_x = readout.W_x.detach().cpu().numpy().astype(np.float64, copy=False)
    w_u = readout.W_u.detach().cpu().numpy().astype(np.float64, copy=False)
    bias = readout.b.detach().cpu().numpy().astype(np.float64, copy=False)
    if w_x.ndim != 2 or w_u.ndim != 2 or bias.ndim != 1:
        raise ValueError("readout weights must be W_x [out, N], W_u [out, F], b [out]")
    out_features = int(w_x.shape[0])
    n_nodes = int(w_x.shape[1])
    n_in = int(w_u.shape[1])
    if x.shape[1] != n_nodes:
        raise ValueError(f"states N={x.shape[1]} != W_x columns {n_nodes}")
    if u.shape[1] != n_in:
        raise ValueError(f"inputs F={u.shape[1]} != W_u columns {n_in}")
    if w_u.shape[0] != out_features or bias.shape[0] != out_features:
        raise ValueError("W_x, W_u, and b must share out_features")

    n_steps = int(x.shape[0])
    intercept = np.asarray(bias, dtype=np.float64).reshape(out_features)
    # input[t, j, o] = W_u[o, j] * u[t, j]
    input_contrib = u[:, :, None] * w_u.T[None, :, :]
    raw = (x @ w_x.T) + (u @ w_u.T) + intercept[None, :]
    if lazy:
        neuron = np.zeros((n_steps, 0, out_features), dtype=np.float32)
    else:
        # neuron[t, i, o] = W_x[o, i] * x[t, i]
        neuron = (x[:, :, None] * w_x.T[None, :, :]).astype(np.float32, copy=False)
    return {
        "intercept": intercept.astype(np.float32, copy=False),
        "input": input_contrib.astype(np.float32, copy=False),
        "neuron": neuron,
        "raw": raw.astype(np.float32, copy=False),
    }
