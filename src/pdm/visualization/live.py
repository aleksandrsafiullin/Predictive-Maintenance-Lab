from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from pdm.architectures import is_reservoir
from pdm.io_util import atomic_write_bytes, atomic_write_json, read_json
from pdm.paths import worker_dir
from pdm.visualization.trace import is_reservoir_module

LIVE_ACTIVITY_JSON = "live_activity.json"
LIVE_ACTIVITY_NPZ = "live_activity.npz"
LIVE_NODE_CAP = 512
DOWNSAMPLE_CAPTION = "Activity downsampled for display"


def live_activity_path() -> Path:
    """Small JSON sidecar under ``runs/_worker/``. Arrays live in the sibling npz."""
    return worker_dir() / LIVE_ACTIVITY_JSON


def live_activity_npz_path() -> Path:
    return worker_dir() / LIVE_ACTIVITY_NPZ


def clear_live_activity() -> None:
    for path in (live_activity_path(), live_activity_npz_path()):
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def load_live_activity() -> dict[str, Any] | None:
    meta_path = live_activity_path()
    if not meta_path.exists():
        return None
    try:
        meta = dict(read_json(meta_path))
    except Exception:  # noqa: BLE001
        return None
    npz_path = live_activity_npz_path()
    if npz_path.exists() and str(meta.get("status") or "") != "not_reservoir":
        try:
            with np.load(npz_path, allow_pickle=False) as packed:
                if "states" in packed.files:
                    meta["states"] = np.array(packed["states"], copy=True)
                if "inputs" in packed.files:
                    meta["inputs"] = np.array(packed["inputs"], copy=True)
        except Exception:  # noqa: BLE001
            return meta
    return meta


def write_live_activity(
    *,
    architecture: str,
    status: str,
    states: np.ndarray | None = None,
    inputs: np.ndarray | None = None,
    node_order: Sequence[str] | None = None,
    predicted_rul_s: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write meta JSON (no large arrays) and optional npz. Never status.json."""
    meta: dict[str, Any] = {
        "architecture": str(architecture or ""),
        "status": str(status),
        "node_order": [str(n) for n in (node_order or [])],
        "predicted_rul_s": None if predicted_rul_s is None else float(predicted_rul_s),
        "phase": "training",
    }
    if extra:
        meta.update(extra)
    dest = live_activity_path()
    if states is None or status == "not_reservoir":
        npz = live_activity_npz_path()
        if npz.exists():
            try:
                npz.unlink()
            except OSError:
                pass
        atomic_write_json(dest, meta)
        return dest
    arr = np.asarray(states, dtype=np.float32)
    inp = np.asarray(inputs if inputs is not None else np.zeros((arr.shape[0], 0)), dtype=np.float32)
    meta["n_frames"] = int(arr.shape[0])
    meta["n_features"] = int(inp.shape[1]) if inp.ndim == 2 else 0
    buf = io.BytesIO()
    np.savez(buf, states=arr, inputs=inp)
    atomic_write_bytes(live_activity_npz_path(), buf.getvalue())
    atomic_write_json(dest, meta)
    return dest


def downsample_nodes(
    states: np.ndarray,
    node_order: Sequence[str],
    *,
    cap: int = LIVE_NODE_CAP,
) -> tuple[np.ndarray, list[str], np.ndarray, bool]:
    """Evenly spaced node subset. Deterministic. Returns states, names, index, downsampled."""
    arr = np.asarray(states, dtype=np.float32)
    names = [str(n) for n in node_order]
    if arr.ndim != 2:
        return arr, names, np.arange(len(names), dtype=np.int64), False
    n = int(arr.shape[1])
    if n <= int(cap):
        return arr, names[:n] if names else [str(i) for i in range(n)], np.arange(n, dtype=np.int64), False
    idx = np.unique(np.linspace(0, n - 1, num=int(cap), dtype=np.int64))
    kept_names = [names[i] if i < len(names) else str(int(i)) for i in idx.tolist()]
    return arr[:, idx], kept_names, idx, True


def write_training_live_snapshot(
    model: nn.Module,
    window_x: torch.Tensor | np.ndarray | None,
    *,
    architecture: str | None = None,
    dataset_id: str | None = None,
    run_id: str | None = None,
    epoch: int | None = None,
    unit_id: str | None = None,
    node_order: Sequence[str] | None = None,
    window_timestamps_s: Sequence[float] | np.ndarray | None = None,
    should_stop: Any | None = None,
) -> bool:
    """Snapshot one train-split window. GRU/LSTM writes ``not_reservoir`` (no fake states).

    Uses ``model.eval()`` + ``no_grad`` + shared ``forward_states``. Returns True when
    a reservoir activity array was written.
    """
    if should_stop is not None and callable(should_stop) and should_stop():
        return False
    arch = str(architecture or getattr(model, "architecture", "") or "").strip().lower()
    if not is_reservoir_module(model) and not is_reservoir(arch):
        write_live_activity(architecture=arch or "gru", status="not_reservoir")
        return False
    if window_x is None:
        return False
    if isinstance(window_x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(window_x)).float()
    else:
        x = window_x.detach().float()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.dim() != 3 or x.shape[0] < 1:
        return False
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        buf = getattr(model, "W_res", None)
        dev = buf.device if isinstance(buf, torch.Tensor) else torch.device("cpu")
    x = x[:1].contiguous().to(dev)
    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            states_t = model.forward_states(x)
            pred = None
            try:
                rul = model.predicted_rul_s(x)
                pred = float(rul.detach().cpu().reshape(-1)[0].item())
                if not np.isfinite(pred) or pred < 0:
                    pred = None
            except Exception:  # noqa: BLE001
                pred = None
    finally:
        if was_training:
            model.train()
    states_np = states_t.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    inputs_np = x.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)
    names = [str(n) for n in (node_order if node_order is not None else (getattr(model, "node_order", None) or []))]
    if not names:
        names = [str(i) for i in range(int(states_np.shape[1]))]
    states_np, names, _idx, downsampled = downsample_nodes(states_np, names)
    n_t = int(states_np.shape[0])
    stamps: list[float] = []
    if window_timestamps_s is not None:
        raw_ts = np.asarray(window_timestamps_s, dtype=np.float64).reshape(-1)
        if raw_ts.size == n_t:
            stamps = [float(v) for v in raw_ts.tolist()]
    end_ts = float(stamps[-1]) if stamps else None
    frame_map = [
        {
            "frame_index": int(i),
            "window_offset": int(i),
            "timestamp_s": stamps[i] if stamps else float(i),
            "window_end_timestamp_s": end_ts if end_ts is not None else float(max(n_t - 1, 0)),
            "unit_id": str(unit_id or ""),
            "prediction_index": int(max(n_t - 1, 0)),
        }
        for i in range(n_t)
    ]
    extra: dict[str, Any] = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "epoch": epoch,
        "unit_id": str(unit_id or ""),
        "downsampled": bool(downsampled),
        "downsample_caption": DOWNSAMPLE_CAPTION if downsampled else "",
        "frame_map": frame_map,
        "history_length": n_t,
        "n_nodes_display": int(states_np.shape[1]),
    }
    write_live_activity(
        architecture=arch,
        status="ok",
        states=states_np,
        inputs=inputs_np,
        node_order=names,
        predicted_rul_s=pred,
        extra=extra,
    )
    return True
