from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdm.connectome.provenance import GRAPH_MODE_SYNTHETIC, SYNTHETIC_DISCLAIMER
from pdm.io_util import read_json
from pdm.replay import slice_predictions_to_replay_time

EXPLORER_DISCLAIMER = (
    "Computational activity in a connectome-based reservoir. "
    "This is not a biophysical simulation or recorded activity of a living fly brain."
)
EXPLORER_MODES = (
    "Overview",
    "Equipment replay",
    "Inside prediction window",
    "Alert inspection",
)
INLINE_TRACE_MAX_NODES = 128
WORKER_BUSY_MESSAGE = "A heavy job is already running. Please wait."
RESERVOIR_REQUIRED_MESSAGE = "Neural Activity Explorer requires a reservoir run."


def synthetic_banner_required(rec: Mapping[str, Any] | None, scene: Mapping[str, Any] | None = None) -> bool:
    """True when the selected run is a synthetic fixture (or labeled synthetic)."""
    rec = rec or {}
    scene = scene or {}
    if bool(rec.get("is_synthetic")) or bool(scene.get("is_synthetic")):
        return True
    for mode in (rec.get("graph_mode"), scene.get("graph_mode")):
        if str(mode or "").strip().lower() == GRAPH_MODE_SYNTHETIC:
            return True
    return False


def fallback_positions(nodes: Sequence[str]) -> dict[str, list[float]]:
    """Ring layout so the scene is never blank when xyz / layout.json are missing."""
    ids = [str(n) for n in nodes]
    n = max(len(ids), 1)
    out: dict[str, list[float]] = {}
    for i, nid in enumerate(ids):
        angle = 2.0 * math.pi * i / n
        out[nid] = [math.cos(angle), math.sin(angle), 0.0]
    return out


def _as_xyz(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = float(value[0])
    y = float(value[1])
    z = float(value[2]) if len(value) > 2 else 0.0
    return [x, y, z]


def load_scene_from_run(rdir: Path, node_order: Sequence[str] | None = None) -> dict[str, Any]:
    """Nodes, edges, and coordinates from connectome artifacts. Never a blank graph."""
    cdir = Path(rdir) / "connectome"
    nodes: list[str] = []
    edges: list[dict[str, Any]] = []
    positions: dict[str, list[float]] = {}
    graph_mode = ""
    is_synthetic = False

    graph_path = cdir / "graph.json"
    if graph_path.exists():
        payload = read_json(graph_path)
        nodes = [str(n) for n in (payload.get("node_order") or payload.get("nodes") or [])]
        edges = [dict(e) for e in (payload.get("edges") or [])]

    layout_path = cdir / "layout.json"
    if layout_path.exists():
        layout = read_json(layout_path)
        raw_pos = layout.get("positions") if isinstance(layout, dict) else None
        if not isinstance(raw_pos, dict):
            raw_pos = layout if isinstance(layout, dict) else {}
        for key, val in (raw_pos or {}).items():
            if key == "node_order":
                continue
            xyz = _as_xyz(val)
            if xyz is not None:
                positions[str(key)] = xyz
        if not nodes and layout.get("node_order"):
            nodes = [str(n) for n in layout["node_order"]]

    prov_path = cdir / "provenance.json"
    if prov_path.exists():
        prov = read_json(prov_path)
        graph_mode = str(prov.get("graph_mode") or "")
        is_synthetic = bool(prov.get("is_synthetic", graph_mode == GRAPH_MODE_SYNTHETIC))

    if node_order:
        order = [str(n) for n in node_order]
        extra = [n for n in nodes if n not in order]
        nodes = order + extra
    if not nodes:
        nodes = list(positions.keys())
    if not positions:
        positions = fallback_positions(nodes)
    else:
        for nid in nodes:
            if nid not in positions:
                positions[nid] = [0.0, 0.0, 0.0]
    return {
        "nodes": nodes,
        "edges": edges,
        "positions": positions,
        "graph_mode": graph_mode,
        "is_synthetic": is_synthetic,
        "disclaimer": SYNTHETIC_DISCLAIMER if is_synthetic or graph_mode == GRAPH_MODE_SYNTHETIC else "",
    }


def build_explorer_payload(
    *,
    nodes: Sequence[str],
    edges: Sequence[Mapping[str, Any]] | None,
    positions: Mapping[str, Sequence[float]] | None,
    states: Any,
    frame_map: Sequence[Mapping[str, Any]] | None,
    flags: Mapping[str, Any] | None,
    labels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """JSON-serializable args for the Neural Activity Explorer component."""
    arr = np.asarray(states, dtype=float) if states is not None else np.zeros((0, 0), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        arr = np.zeros((0, len(nodes)), dtype=float)
    pos: dict[str, list[float]] = {}
    for key, val in (positions or {}).items():
        xyz = _as_xyz(val)
        if xyz is not None:
            pos[str(key)] = xyz
    payload: dict[str, Any] = {
        "nodes": [str(n) for n in nodes],
        "edges": [dict(e) for e in (edges or [])],
        "positions": pos,
        "states": arr.tolist(),
        "frame_map": [dict(f) for f in (frame_map or [])],
        "flags": dict(flags or {}),
    }
    if labels is not None:
        payload["labels"] = dict(labels)
    return payload


def collect_alert_rows(rdir: Path, unit_id: str | None) -> list[dict[str, Any]]:
    """Alert episode rows for Alert inspection. Empty list if none; never raises."""
    from pdm.experiments import (
        legacy_evaluation_artifacts,
        list_evaluations,
        resolve_evaluation_artifacts,
    )

    paths: list[Path] = []
    try:
        for ev in list_evaluations(rdir):
            art = resolve_evaluation_artifacts(rdir, ev.get("eval_id"))
            alerts = art.get("alerts")
            if alerts is not None:
                paths.append(Path(alerts))
        legacy = legacy_evaluation_artifacts(rdir).get("alerts")
        if legacy is not None:
            paths.append(Path(legacy))
    except Exception:  # noqa: BLE001
        return []

    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any]] = set()
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except Exception:  # noqa: BLE001
            continue
        if unit_id and "unit_id" in frame.columns:
            frame = frame[frame["unit_id"].astype(str) == str(unit_id)]
        for rec in frame.to_dict(orient="records"):
            key = (
                rec.get("unit_id"),
                rec.get("timestamp_s"),
                rec.get("type") or rec.get("alert_status"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec)
    return rows


def causal_prefix_rows(frame: pd.DataFrame, timestamp_s: float) -> pd.DataFrame:
    """Keep rows with timestamp_s ≤ T. No future measurements or predictions."""
    if frame is None or getattr(frame, "empty", True) or "timestamp_s" not in frame.columns:
        return frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
    ts = pd.to_numeric(frame["timestamp_s"], errors="coerce")
    return frame.loc[ts <= float(timestamp_s)].copy()


def alert_jump_target(episode: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Read-only unit + timestamp + stored predicted RUL from an alert episode."""
    if not episode:
        return None
    ts = episode.get("timestamp_s")
    if ts is None or (isinstance(ts, float) and pd.isna(ts)):
        return None
    rul = episode.get("predicted_rul_s")
    stored = None
    try:
        if rul is not None and not (isinstance(rul, float) and pd.isna(rul)):
            stored = float(rul)
    except (TypeError, ValueError):
        stored = None
    unit = episode.get("unit_id")
    return {
        "unit_id": None if unit is None else str(unit),
        "timestamp_s": float(ts),
        "predicted_rul_s": stored,
    }


def stored_alert_prediction(
    episode: Mapping[str, Any] | None,
    predictions: pd.DataFrame | None = None,
    *,
    unit_id: str | None = None,
    timestamp_s: float | None = None,
) -> float | None:
    """Stored predicted_rul_s at ≤ T. Never a rescore that uses future rows."""
    target = alert_jump_target(episode)
    if target and target.get("predicted_rul_s") is not None:
        return float(target["predicted_rul_s"])
    t = timestamp_s if timestamp_s is not None else (target or {}).get("timestamp_s")
    uid = unit_id if unit_id is not None else (target or {}).get("unit_id")
    if predictions is None or t is None:
        return None
    pred = predictions
    if uid is not None and "unit_id" in pred.columns:
        pred = pred[pred["unit_id"].astype(str) == str(uid)]
    sliced = slice_predictions_to_replay_time(pred, float(t))
    if sliced.empty or "predicted_rul_s" not in sliced.columns:
        return None
    val = sliced.iloc[-1].get("predicted_rul_s")
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def slice_trace_to_alert(
    trace: Mapping[str, Any] | None,
    episode: Mapping[str, Any] | None,
    *,
    timestamp_s: float | None = None,
    predictions: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Restrict a saved trace to frames with timestamp_s ≤ alert time.

    Predicted RUL is the stored evaluation/alert value, not a rescore from future rows.
    """
    rec = dict(trace or {})
    target = alert_jump_target(episode) or {}
    t = timestamp_s if timestamp_s is not None else target.get("timestamp_s")
    frames = list(rec.get("frame_map") or [])
    if t is None:
        rec["frame_map"] = [dict(f) for f in frames]
        rec["predicted_rul_s"] = stored_alert_prediction(
            episode, predictions, timestamp_s=t, unit_id=target.get("unit_id")
        )
        return rec
    t = float(t)
    keep_idx = [
        i
        for i, frm in enumerate(frames)
        if frm.get("timestamp_s") is not None and float(frm["timestamp_s"]) <= t
    ]
    rec["frame_map"] = [dict(frames[i]) for i in keep_idx]
    states = rec.get("states")
    arr = np.asarray(states) if states is not None else None
    if arr is not None and arr.ndim == 2 and arr.shape[0] == len(frames):
        rec["states"] = arr[keep_idx] if keep_idx else arr[0:0]
    inputs = rec.get("inputs")
    inp = np.asarray(inputs) if inputs is not None else None
    if inp is not None and inp.ndim == 2 and inp.shape[0] == len(frames):
        rec["inputs"] = inp[keep_idx] if keep_idx else inp[0:0]
    rec["predicted_rul_s"] = stored_alert_prediction(
        episode,
        predictions,
        timestamp_s=t,
        unit_id=target.get("unit_id"),
    )
    return rec
