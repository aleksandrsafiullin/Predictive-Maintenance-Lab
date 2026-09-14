"""Causal equipment steps and inspectable terms from the saved reservoir."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from pdm.preprocessing import FEATURE_PIPELINE_VERSION, apply_preprocessor
from pdm.visualization.contributions import decompose_contributions
from pdm.visualization.trace import predict_with_trace


def simulate_step(features, unit_id, timestamp_s, model, prep, history_length, previous_trace=None) -> dict[str, Any]:
    """Score only this unit's observed prefix. Ground truth never enters inference."""
    prefix = features.loc[
        (features["unit_id"].astype(str) == str(unit_id))
        & (pd.to_numeric(features["timestamp_s"], errors="coerce") <= float(timestamp_s))
    ].sort_values("timestamp_s").copy()
    if getattr(model, "state_mode", None) == "continuous":
        return continuous_trace(prefix, model, prep, history_length, previous_trace=previous_trace)
    trace = predict_with_trace(
        prefix, str(unit_id), model, prep, history_length=history_length, device="cpu"
    )
    if trace["status"] != "predicted":
        return trace
    # Diagnostic decomposition only. States still come from the shared ESN kernel.
    states = torch.from_numpy(trace["states"])
    inputs = torch.from_numpy(trace["inputs"])
    previous = torch.cat([torch.zeros_like(states[:1]), states[:-1]], dim=0)
    with torch.no_grad():
        trace["processing"] = {
            "input_drive": F.linear(inputs, model.W_in).numpy(),
            "recurrent_drive": F.linear(previous, model.W_res).numpy(),
            "previous_state": previous.numpy(),
            "bias": model.b_res.detach().numpy().copy(),
            "leak": float(model.alpha),
        }
    return trace


def continuous_trace(prefix, model, prep, warmup, *, previous_trace=None):
    """Exact chronological ESN states. Cached prefixes only accelerate append-only steps."""
    frame = prefix.sort_values("timestamp_s").copy().reset_index(drop=True)
    from pdm.visualization.trace import _display_from_raw, _empty_trace

    if frame.empty:
        return _empty_trace(model, list(model.node_order), 0)
    if prep.feature_pipeline_version != FEATURE_PIPELINE_VERSION:
        raise ValueError("Continuous forecasting requires the saved raw-first feature pipeline")
    if prep.dataset_id == "filters":
        from pdm.windows import recompute_filter_gap_before

        frame = recompute_filter_gap_before(frame, gap_multiplier=prep.gap_multiplier,
                                           sampling_interval_s=prep.sampling_interval_s, causal=True)
    if "unit_id" in frame and frame["unit_id"].astype(str).nunique() != 1:
        raise ValueError("Continuous history must belong to one bearing")
    ts = frame["timestamp_s"].to_numpy(dtype=float)
    if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        raise ValueError("Measurement timestamps must be finite and strictly increasing")
    if "gap_before" in frame:
        gaps = np.flatnonzero(frame["gap_before"].fillna(False).to_numpy(dtype=bool))
        if len(gaps):
            frame = frame.iloc[int(gaps[-1]):].reset_index(drop=True)
            ts = frame["timestamp_s"].to_numpy(dtype=float)
    encoded = apply_preprocessor(prep, frame)
    inputs = encoded[prep.feature_names].to_numpy(dtype=np.float32, copy=True)
    previous_trace = previous_trace or {}
    old_inputs = previous_trace.get("inputs")
    old_ts = previous_trace.get("timestamps_s")
    old_n = len(old_inputs) if old_inputs is not None else 0
    identity = (id(model), model.W_in._version, model.W_res._version, model.b_res._version, float(model.alpha))
    reuse = (old_n > 0 and old_n < len(inputs) and old_ts is not None
             and np.array_equal(ts[:old_n], old_ts)
             and np.array_equal(inputs[:old_n], old_inputs)
             and previous_trace.get("model_identity") == identity)
    model.eval()
    device = model.W_in.device
    with torch.no_grad():
        x = torch.from_numpy(inputs).to(device)
        if reuse:
            cached = torch.from_numpy(previous_trace["states"]).to(device)
            fresh = model.forward_states(x[old_n:], x0=cached[-1])
            states = torch.cat([cached, fresh], dim=0)
        else:
            states = model.forward_states(x)
        raw = model.forward_raw(states, x)
        raw_rul = _display_from_raw(model, raw).detach().cpu().numpy()
        prev = states[-2:-1] if len(states) > 1 else torch.zeros_like(states[-1:])
        processing = {
            "input_drive": F.linear(x[-1:], model.W_in).cpu().numpy(),
            "recurrent_drive": F.linear(prev, model.W_res).cpu().numpy(),
            "previous_state": prev.cpu().numpy(),
            "bias": model.b_res.detach().cpu().numpy().copy(), "leak": float(model.alpha),
        }
    states_np = states.cpu().numpy()
    contributions = decompose_contributions(states_np, inputs, model.readout)
    contributions["raw"] = raw.cpu().numpy()
    ready = len(frame) >= int(warmup)
    return {
        "status": "predicted" if ready else "Collecting history",
        "predicted_rul_s": float(raw_rul[-1]) if ready else None,
        "raw_prediction": raw[-1].cpu().numpy(), "raw_rul_s": raw_rul,
        "states": states_np, "inputs": inputs, "processing": processing,
        "contributions": contributions, "timestamps_s": ts,
        "frame_map": [{"timestamp_s": float(t), "frame_index": i} for i, t in enumerate(ts)],
        "node_order": list(model.node_order), "n_history": len(frame),
        "valid_history_reason": "" if ready else "insufficient_length",
        "model_identity": identity,
    }


def neuron_details(trace, model, prep, node_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Actual last-step incoming products, with source IDs and signed model weights."""
    i = int(node_index)
    inputs = trace["inputs"][-1]
    previous = trace["processing"]["previous_state"][-1]
    win = model.W_in[i].detach().cpu().numpy()
    wres = model.W_res[i].detach().cpu().numpy()
    sensors = pd.DataFrame({
        "Feature": prep.feature_names,
        "Normalized input": inputs,
        "Input weight": win,
        "Input contribution": inputs * win,
    })
    source = np.flatnonzero(wres)
    neighbors = pd.DataFrame({
        "Source body ID": [model.node_order[j] for j in source],
        "Previous state": previous[source],
        "Recurrent weight": wres[source],
        "Recurrent contribution": previous[source] * wres[source],
    })
    if not neighbors.empty:
        order = neighbors["Recurrent contribution"].abs().sort_values(ascending=False).index
        neighbors = neighbors.loc[order].reset_index(drop=True)
    return sensors, neighbors
