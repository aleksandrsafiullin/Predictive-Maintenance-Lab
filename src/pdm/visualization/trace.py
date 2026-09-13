from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from pdm.architectures import RESERVOIR_ARCHITECTURES
from pdm.losses import weibull_median_rul
from pdm.models.reservoir import LeakyESN
from pdm.models.reservoir import forward_states as leaky_forward_states
from pdm.predict import prepare_history_window
from pdm.preprocessing import Preprocessor
from pdm.visualization.contributions import decompose_contributions

# Re-export so tests can assert `trace.leaky_forward_states is reservoir.forward_states`.
# predict / ridge / this module all call LeakyESN.forward_states → leaky_forward_states.
__all__ = ["predict_with_trace", "leaky_forward_states", "run_trace_job"]


def is_reservoir_module(model: nn.Module) -> bool:
    arch = str(getattr(model, "architecture", "") or "").strip().lower()
    if arch in RESERVOIR_ARCHITECTURES:
        return True
    return isinstance(model, LeakyESN)


def predict_with_trace(
    history_rows: pd.DataFrame,
    unit_id: str,
    model: nn.Module,
    prep: Preprocessor,
    *,
    lazy: bool = False,
    history_length: int | None = None,
    prediction_index: int | None = None,
    device: str | torch.device | None = None,
    gap_multiplier: float | None = None,
    sampling_interval_s: float | None = None,
) -> dict[str, Any]:
    """Causal reservoir trace. Uses ``model.forward_states`` (the shared leaky kernel).

    ``history_rows`` must contain only measurements at times ``<= t``. The last
    ``history_length`` eligible rows form one window; ``state_mode='window_reset'``
    starts from zeros for that window (same as ordinary predict).
    """
    node_order = [str(n) for n in list(getattr(model, "node_order", []) or [])]
    n_hist = len(history_rows)
    empty = _empty_trace(model, node_order, n_hist)
    if not is_reservoir_module(model):
        empty["status"] = "traces require reservoir model"
        empty["valid_history_reason"] = "traces require a reservoir run"
        return empty

    hist_len = int(history_length) if history_length is not None else max(int(n_hist), 1)
    prepared = prepare_history_window(
        history_rows,
        prep,
        hist_len,
        gap_multiplier=gap_multiplier,
        sampling_interval_s=sampling_interval_s,
    )
    if not prepared["ok"]:
        empty["status"] = prepared["status"]
        empty["valid_history_reason"] = prepared["valid_history_reason"]
        empty["n_history"] = prepared["n_history"]
        return empty

    window: pd.DataFrame = prepared["window"]
    arr: np.ndarray = prepared["inputs"]
    uid = str(unit_id or _unit_id_from_window(window))
    dev = _resolve_device(model, device)
    model = model.to(dev)
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).to(dev)
    with torch.no_grad():
        # Shared leaky kernel — LeakyESN.forward_states → leaky_forward_states.
        states_t = model.forward_states(x)
        raw_seq = model.forward_raw(states_t.squeeze(0), x.squeeze(0))
        raw_last = model.forward_raw(states_t[:, -1, :], x[:, -1, :])
        display = _display_from_raw(model, raw_last)

    value = float(display.detach().cpu().reshape(-1)[0].item())
    if not np.isfinite(value) or value < 0:
        empty["status"] = "No valid prediction"
        empty["n_history"] = prepared["n_history"]
        return empty

    states_np = states_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    inputs_np = np.asarray(arr, dtype=np.float32)
    raw_seq_np = raw_seq.detach().cpu().numpy().astype(np.float32, copy=False)
    if raw_seq_np.ndim == 1:
        raw_seq_np = raw_seq_np.reshape(states_np.shape[0], -1)
    contrib = decompose_contributions(states_np, inputs_np, model.readout, lazy=lazy)
    contrib["raw"] = np.asarray(raw_seq_np, dtype=np.float32)
    raw_prediction = raw_last.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
    frame_map = _frame_map(window, uid, prediction_index)
    return {
        "predicted_rul_s": value,
        "raw_prediction": raw_prediction,
        "states": states_np,
        "inputs": inputs_np,
        "contributions": contrib,
        "frame_map": frame_map,
        "status": "predicted",
        "node_order": node_order,
        "n_history": prepared["n_history"],
        "valid_history_reason": "",
    }


def run_trace_job(
    dataset_id: str,
    run_id: str,
    unit_id: str | None,
    *,
    should_stop: Callable[[], bool] | None = None,
    lazy: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Write ``runs/<dataset_id>/<run_id>/traces/<unit_id>/``. Stop → ``cancelled``."""
    from pdm.data.prepare import load_processed
    from pdm.paths import dataset_runs, run_traces_dir
    from pdm.train import load_trained_model
    from pdm.visualization.export import save_trace

    if should_stop is not None and should_stop():
        return {"status": "cancelled"}
    rdir = dataset_runs(dataset_id) / run_id
    model, prep, meta = load_trained_model(rdir, device=device, which="best")
    if should_stop is not None and should_stop():
        return {"status": "cancelled"}
    if not is_reservoir_module(model):
        return {
            "status": "completed",
            "trace_note": "traces require reservoir model",
        }
    packed = load_processed(dataset_id)
    features = packed["features"]
    if unit_id is None:
        raise ValueError("trace job requires unit_id")
    hist = features[features["unit_id"].astype(str) == str(unit_id)].copy()
    if hist.empty:
        raise ValueError(f"no measurements for unit_id={unit_id!r}")
    hist.attrs["raw_features"] = True
    if should_stop is not None and should_stop():
        return {"status": "cancelled"}
    hist_len = int(meta["history_length"])
    trace = predict_with_trace(
        hist,
        str(unit_id),
        model,
        prep,
        lazy=lazy,
        history_length=hist_len,
        device=device,
    )
    if should_stop is not None and should_stop():
        return {"status": "cancelled"}
    if trace["status"] != "predicted":
        return {"status": "completed", "trace_status": trace["status"], "unit_id": str(unit_id)}
    dest = run_traces_dir(dataset_id, run_id, str(unit_id))
    save_trace(
        dest,
        unit_id=str(unit_id),
        run_id=str(run_id),
        dataset_id=str(dataset_id),
        architecture=str(getattr(model, "architecture", meta.get("architecture", ""))),
        graph_hash=getattr(model, "graph_hash", meta.get("graph_hash")),
        n_nodes=int(getattr(model, "n_nodes", meta.get("n_nodes") or 0)),
        history_length=hist_len,
        graph_mode=str(getattr(model, "graph_mode", meta.get("graph_mode") or "")),
        is_synthetic=bool(getattr(model, "is_synthetic", True)),
        states=trace["states"],
        inputs=trace["inputs"],
        contributions=trace["contributions"],
        frame_map=trace["frame_map"],
        node_order=trace["node_order"],
        predicted_rul_s=trace["predicted_rul_s"],
        raw_prediction=trace["raw_prediction"],
        status=trace["status"],
        time_scale_s=float(getattr(model, "time_scale_s", prep.time_scale_s)),
        head=str(getattr(model, "head_type", "")),
    )
    return {"status": "completed", "trace_dir": str(dest), "unit_id": str(unit_id)}


def _display_from_raw(model: nn.Module, raw_last: torch.Tensor) -> torch.Tensor:
    """Display RUL seconds from last-step linear raw. Same postprocess as ``predicted_rul_s``."""
    scale = float(getattr(model, "time_scale_s", 1.0))
    post = model._postprocess(raw_last)
    if getattr(model, "head_type", "rul") == "rul":
        return post * scale
    lam, k = post
    return weibull_median_rul(lam, k, scale)


def _frame_map(window: pd.DataFrame, unit_id: str, prediction_index: int | None) -> list[dict[str, Any]]:
    ts = window["timestamp_s"].to_numpy(dtype=np.float64)
    end_ts = float(ts[-1]) if ts.size else 0.0
    pred_idx = int(prediction_index) if prediction_index is not None else int(len(window) - 1)
    frames: list[dict[str, Any]] = []
    for i, stamp in enumerate(ts.tolist()):
        frames.append(
            {
                "frame_index": int(i),
                "window_offset": int(i),
                "timestamp_s": float(stamp),
                "window_end_timestamp_s": end_ts,
                "unit_id": str(unit_id),
                "prediction_index": pred_idx,
            }
        )
    return frames


def _empty_trace(model: nn.Module, node_order: list[str], n_history: int) -> dict[str, Any]:
    n_nodes = int(getattr(model, "n_nodes", 0) or 0)
    n_in = int(getattr(model, "input_size", 0) or 0)
    out_f = int(getattr(getattr(model, "readout", None), "out_features", 1) or 1)
    return {
        "predicted_rul_s": None,
        "raw_prediction": np.zeros((out_f,), dtype=np.float32),
        "states": np.zeros((0, n_nodes), dtype=np.float32),
        "inputs": np.zeros((0, n_in), dtype=np.float32),
        "contributions": {
            "intercept": np.zeros((out_f,), dtype=np.float32),
            "input": np.zeros((0, n_in, out_f), dtype=np.float32),
            "neuron": np.zeros((0, n_nodes, out_f), dtype=np.float32),
            "raw": np.zeros((0, out_f), dtype=np.float32),
        },
        "frame_map": [],
        "status": "Collecting history",
        "node_order": node_order,
        "n_history": int(n_history),
        "valid_history_reason": "",
    }


def _unit_id_from_window(window: pd.DataFrame) -> str:
    if "unit_id" in window.columns and len(window):
        return str(window["unit_id"].iloc[-1])
    return ""


def _resolve_device(model: nn.Module, device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    try:
        return next(model.parameters()).device
    except StopIteration:
        buf = getattr(model, "W_res", None)
        if isinstance(buf, torch.Tensor):
            return buf.device
        return torch.device("cpu")
