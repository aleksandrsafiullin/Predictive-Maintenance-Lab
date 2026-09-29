"""Saved numeric signal inference from an observed equipment prefix only."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import torch

from pdm.data.project_prepare import load_snapshot
from pdm.models.signal_recurrent import SignalRecurrent
from pdm.signal_training import _predict, _segments, load_signal_run


def _resolved_thresholds(schema: dict, prefix) -> dict:
    rule = dict(schema.get("thresholds") or {})
    mode = rule.get("mode", "absolute")
    direction = rule.get("direction", "above")
    if direction not in {"above", "below"}:
        return {"mode": mode, "direction": direction, "yellow": None, "red": None,
                "status": "unavailable", "reason": "Unsupported threshold direction"}
    if mode == "absolute":
        try:
            red = float(rule["red"])
        except (KeyError, TypeError, ValueError):
            red = None
        try:
            yellow = float(rule["yellow"]) if rule.get("yellow") is not None else None
        except (TypeError, ValueError):
            yellow = None
        if red is None or not np.isfinite(red) or (yellow is not None and not np.isfinite(yellow)):
            return {"mode": mode, "direction": direction, "yellow": None, "red": None,
                    "status": "unavailable", "reason": "No valid saved signal thresholds"}
        return {"mode": mode, "direction": direction, "yellow": yellow, "red": red,
                "status": "available", "reason": None}
    if mode == "initial_baseline_multiple":
        n = int(rule.get("baseline_n", 5))
        if len(prefix) < n:
            return {"mode": mode, "direction": direction, "yellow": None, "red": None,
                    "status": "unavailable", "reason": "Waiting for the initial causal baseline"}
        first = prefix.iloc[:n].signal.to_numpy(float)
        baseline = float(np.median(first))
        yellow = max(baseline + float(rule.get("onset_sigma", 3.0)) * float(np.std(first)),
                     float(rule.get("onset_ratio", 1.25)) * baseline)
        red = float(rule.get("red_ratio", 2.0)) * baseline
        return {"mode": mode, "direction": direction, "yellow": yellow, "red": red,
                "status": "available", "reason": None, "baseline": baseline,
                "baseline_n": n}
    return {"mode": mode, "direction": direction, "yellow": None, "red": None,
            "status": "unavailable", "reason": "Unsupported saved threshold rule"}


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


def forecast_prefix(project_id: str, run_id: str, unit_id: str, as_of_s: float) -> dict:
    """Score only rows at/before as_of; never feed hidden future observations."""
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
    threshold = _resolved_thresholds(run["schema"], prefix)
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
    already_red = (current >= threshold["red"] if threshold["direction"] == "above"
                   else current <= threshold["red"]) if threshold["status"] == "available" else False
    if already_red:
        result["crossing"] = {"status": "already_red", "time_s": issued}
    elif threshold["status"] != "available":
        result["crossing"] = {"status": "unavailable", "time_s": None}
    elif not points:
        result["crossing"] = {"status": "unavailable", "time_s": None}
    else:
        red = threshold["red"]
        hit = next((point for point in points if (point["value"] >= red if threshold["direction"] == "above"
                                                  else point["value"] <= red)), None)
        result["crossing"] = {"status": "predicted" if hit else "none_within_horizon",
                              "time_s": hit["target_time_s"] if hit else None}
    return result
