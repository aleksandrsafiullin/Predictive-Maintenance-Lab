"""Saved numeric signal inference from an observed equipment prefix only."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import torch

from pdm.data.project_prepare import load_snapshot
from pdm.models.signal_recurrent import SignalRecurrent
from pdm.project_zones import is_beyond, resolve_thresholds
from pdm.signal_training import _predict, _segments, load_signal_run

_resolved_thresholds = resolve_thresholds


@lru_cache(maxsize=12)
def _load_model(project_id: str, run_id: str, artifact_digest: str, root: str):
    # Hash is checked before reaching this cache; root separates test stores.
    run = load_signal_run(project_id, run_id)
    params = run["params"]
    if run["engine_id"] in {"gru", "lstm"}:
        model = SignalRecurrent(run["engine_id"], 1, params["hidden_size"], len(params["horizons_s"]))
        state = torch.load(run["artifact_path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    return joblib.load(run["artifact_path"])


def forecast_prefix(project_id: str, run_id: str, unit_id: str, as_of_s: float,
                    thresholds: dict | None = None) -> dict:
    """Score only rows at/before as_of; never feed hidden future observations.

    ``thresholds`` replaces the run's saved zone rule for display and red-crossing
    only; model inputs and forecast values do not depend on it.
    """
    from pdm.projects import project_store

    if not np.isfinite(float(as_of_s)):
        raise ValueError("Replay time must be finite")
    run = load_signal_run(project_id, run_id)
    data = load_snapshot(project_id, run["snapshot_id"])
    unit = data["features"][data["features"].unit_id.astype(str) == str(unit_id)].sort_values("timestamp_s")
    if str(unit_id) not in set(map(str, data["split"].get("test", []))):
        raise ValueError("Replay is available only for this run's held-out test units")
    prefix = unit[unit.timestamp_s.astype(float) <= float(as_of_s)].copy()
    observed = [{"timestamp_s": float(t), "signal": float(v)}
                for t, v in zip(prefix.timestamp_s, prefix.signal, strict=True)]
    rule_schema = {**run["schema"], "thresholds": dict(thresholds)} if thresholds else run["schema"]
    threshold = _resolved_thresholds(rule_schema, prefix)
    result = {"as_of_s": float(as_of_s), "observed_prefix": observed, "points": [],
              "thresholds": threshold, "crossing": {"status": "unavailable", "time_s": None},
              "project_id": project_id, "run_id": run_id, "snapshot_id": run["snapshot_id"],
              "calibration_status": run.get("interval_status", "unavailable"),
              "status": "unavailable", "reason": None}
    if prefix.empty:
        result["reason"] = "No observations at this time"
        return result
    # A model is issued from an observed sample, never from an arbitrary slider time.
    issued = float(prefix.timestamp_s.iloc[-1])
    result["as_of_s"] = issued
    segments = _segments(prefix, str(unit_id))
    segment = segments[-1]
    history = int(run["params"]["history_length"])
    if len(segment) < history:
        result["reason"] = f"Need {history} observations since the last gap"
        return result
    signal = segment.signal.to_numpy(np.float32)[-history:]
    x = signal.reshape(1, history, 1)
    model = _load_model(project_id, run_id, run["artifacts"][run["artifact"]], str(project_store().root))
    frame = {"x": x, "y": np.zeros((1, len(run["params"]["horizons_s"])), np.float32)}
    pred, lower, upper = _predict(model, run["engine_id"], frame, run["scaler"])
    points = []
    for j, horizon in enumerate(run["params"]["horizons_s"]):
        value = float(pred[0, j])
        if not np.isfinite(value):
            continue
        points.append({"target_time_s": issued + float(horizon), "value": value,
                       "lower": float(lower[0, j]) if lower is not None else None,
                       "upper": float(upper[0, j]) if upper is not None else None})
    result["points"] = points
    result["status"] = "available" if points else "unavailable"
    result["reason"] = None if points else "No supported forecast points"
    current = float(prefix.signal.iloc[-1])
    already_red = (bool(is_beyond(current, threshold["red"], threshold["direction"]))
                   if threshold["status"] == "available" else False)
    if already_red:
        result["crossing"] = {"status": "already_red", "time_s": issued}
    elif threshold["status"] != "available":
        result["crossing"] = {"status": "unavailable", "time_s": None}
    elif not points:
        result["crossing"] = {"status": "unavailable", "time_s": None}
    else:
        red = threshold["red"]
        hit = next((point for point in points if is_beyond(point["value"], red, threshold["direction"])), None)
        result["crossing"] = {"status": "predicted" if hit else "none_within_horizon",
                              "time_s": hit["target_time_s"] if hit else None}
    return result
