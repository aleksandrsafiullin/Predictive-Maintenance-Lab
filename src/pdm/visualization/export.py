from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np

from pdm.io_util import atomic_write_bytes, atomic_write_json, read_json


def save_trace(
    trace_dir: Path,
    unit_id: str,
    run_id: str,
    dataset_id: str,
    architecture: str,
    graph_hash: str | None,
    n_nodes: int,
    history_length: int,
    graph_mode: str,
    is_synthetic: bool,
    states: np.ndarray,
    inputs: np.ndarray,
    contributions: dict,
    frame_map: list[dict],
    node_order: list[str],
    *,
    predicted_rul_s: float | None = None,
    raw_prediction: np.ndarray | None = None,
    status: str = "predicted",
    time_scale_s: float | None = None,
    head: str | None = None,
) -> None:
    """Write meta.json, states.npz, contributions.npz, frame_map.json atomically."""
    dest = Path(trace_dir)
    dest.mkdir(parents=True, exist_ok=True)
    raw_arr = contributions.get("raw")
    if raw_prediction is None and raw_arr is not None and len(np.asarray(raw_arr)):
        raw_prediction = np.asarray(raw_arr)[-1]
    meta: dict[str, Any] = {
        "run_id": str(run_id),
        "dataset_id": str(dataset_id),
        "architecture": str(architecture),
        "graph_hash": graph_hash,
        "n_nodes": int(n_nodes),
        "history_length": int(history_length),
        "graph_mode": str(graph_mode),
        "is_synthetic": bool(is_synthetic),
        "unit_id": str(unit_id),
        "node_order": [str(n) for n in node_order],
        "status": str(status),
        "predicted_rul_s": None if predicted_rul_s is None else float(predicted_rul_s),
    }
    if raw_prediction is not None:
        meta["raw_prediction"] = [float(v) for v in np.asarray(raw_prediction, dtype=np.float64).reshape(-1)]
    if time_scale_s is not None:
        meta["time_scale_s"] = float(time_scale_s)
    if head is not None:
        meta["head"] = str(head)
    atomic_write_json(dest / "meta.json", meta)
    _atomic_npz(
        dest / "states.npz",
        states=np.asarray(states, dtype=np.float32),
        inputs=np.asarray(inputs, dtype=np.float32),
    )
    _atomic_npz(
        dest / "contributions.npz",
        intercept=np.asarray(contributions["intercept"], dtype=np.float32),
        input=np.asarray(contributions["input"], dtype=np.float32),
        neuron=np.asarray(contributions["neuron"], dtype=np.float32),
        raw=np.asarray(contributions["raw"], dtype=np.float32),
    )
    atomic_write_json(dest / "frame_map.json", list(frame_map))


def load_trace(trace_dir: Path) -> dict[str, Any]:
    """Load saved trace artifacts. Returns same keys as predict_with_trace."""
    dest = Path(trace_dir)
    meta = read_json(dest / "meta.json")
    with np.load(dest / "states.npz", allow_pickle=False) as packed:
        states = np.array(packed["states"], copy=True)
        inputs = np.array(packed["inputs"], copy=True)
    with np.load(dest / "contributions.npz", allow_pickle=False) as packed:
        contributions = {
            "intercept": np.array(packed["intercept"], copy=True),
            "input": np.array(packed["input"], copy=True),
            "neuron": np.array(packed["neuron"], copy=True),
            "raw": np.array(packed["raw"], copy=True),
        }
    frame_map = read_json(dest / "frame_map.json")
    raw_pred = meta.get("raw_prediction")
    if raw_pred is None and len(contributions["raw"]):
        raw_prediction = np.asarray(contributions["raw"][-1], dtype=np.float32)
    else:
        raw_prediction = np.asarray(raw_pred if raw_pred is not None else [], dtype=np.float32)
    pred = meta.get("predicted_rul_s")
    return {
        "predicted_rul_s": None if pred is None else float(pred),
        "raw_prediction": raw_prediction,
        "states": states,
        "inputs": inputs,
        "contributions": contributions,
        "frame_map": frame_map,
        "status": str(meta.get("status") or "predicted"),
        "node_order": [str(n) for n in (meta.get("node_order") or [])],
        "meta": meta,
        "n_history": int(meta.get("history_length") or 0),
        "valid_history_reason": "",
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    atomic_write_bytes(path, buf.getvalue())
