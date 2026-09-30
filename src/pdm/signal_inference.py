"""Saved numeric signal inference from an observed equipment prefix only."""
from __future__ import annotations

from functools import lru_cache

import joblib
import numpy as np
import torch

from pdm.data.project_prepare import load_snapshot
from pdm.models.signal_full_cns import SignalFullCNS
from pdm.models.signal_recurrent import SignalRecurrent
from pdm.project_zones import is_beyond, resolve_thresholds
from pdm.signal_training import _predict, _segments, load_signal_run

_resolved_thresholds = resolve_thresholds


def training_cadence(data: dict) -> float | None:
    """Use Training clocks only, never the hidden Test suffix, for rollout timing."""
    train = data["features"][data["features"].unit_id.astype(str).isin(map(str, data["split"]["train"]))]
    deltas = []
    for uid in train.unit_id.astype(str).unique():
        for segment in _segments(train, uid):
            deltas.extend(np.diff(segment.timestamp_s.to_numpy(float)))
    return float(np.median(deltas)) if deltas else None


def _extend_forecast(model, run: dict, signal: np.ndarray, points: list[dict],
                     issued: float, cadence: float | None, steps: int, threshold: dict,
                     history_times: np.ndarray, *, should_stop=None, progress_cb=None) -> dict:
    """Roll saved heads forward in blocks, feeding only synthetic history back.

    Sparse heads are interpolated on the Training cadence for model inputs only.
    Chart points remain actual model outputs. Conditional one-block quantiles are
    not propagated as though they were calibrated multi-block intervals.
    """
    horizons = np.asarray(run["params"]["horizons_s"], float)
    span = float(horizons[-1])
    info = {"method": "recursive", "direct_through_s": issued + span,
            "status": "unavailable", "reason": None, "steps": steps,
            "interpolated_inputs": False}
    tolerance = run["params"].get("target_tolerance_s", 1e-6)
    if (cadence is None or not np.isfinite(cadence) or cadence <= 0
            or span < cadence or abs(span / cadence - round(span / cadence)) * cadence > tolerance
            or not np.allclose(np.diff(history_times), cadence, atol=tolerance, rtol=0)):
        info["reason"] = "Continuation needs regular history and a final model horizon aligned to the Training cadence."
        return info
    block_steps = round(span / cadence)
    # Limit work even for models with very distant sparse forecast heads.
    if block_steps > 4096:
        info["reason"] = "The saved horizon is too sparse for recursive continuation."
        return info
    grid = np.arange(1, block_steps + 1) * cadence
    info["interpolated_inputs"] = not all(np.any(np.isclose(horizons, t, atol=tolerance, rtol=0)) for t in grid)
    search_end = issued + max(steps * cadence, span)
    tail = max(10 * cadence, span)
    info.update(search_through_s=search_end, post_crossing_s=tail, cadence_s=cadence)
    block = points[:]
    origin = issued
    history = signal.copy()
    while True:
        _check_cancelled(should_stop)
        if len(block) != len(horizons) or any(p.get("value") is None or not np.isfinite(p["value"]) for p in block):
            info.update(status="stopped", reason="The model returned a nonfinite forecast; continuation stopped.")
            return info
        hit = None
        if threshold["status"] == "available":
            if is_beyond(float(signal[-1]), threshold["red"], threshold["direction"]):
                hit = issued
            else:
                hit = next((p["target_time_s"] for p in points
                            if is_beyond(p["value"], threshold["red"], threshold["direction"])), None)
        end = origin + span
        target_end = hit + tail if hit is not None else search_end
        info.update(status="running", predicted_through_s=end, target_through_s=target_end)
        if end >= target_end - tolerance:
            info.update(status="complete", reason="red_tail" if hit is not None else "search_limit")
            return info
        if progress_cb:
            progress_cb(points, info)
        _check_cancelled(should_stop)
        synthetic = np.interp(grid, np.r_[0.0, horizons],
                              [float(history[-1]), *[p["value"] for p in block]])
        history = np.concatenate([history, synthetic])[-len(signal):].astype(np.float32)
        if not np.isfinite(history).all():
            info.update(status="stopped", reason="Recursive history exceeded the model's numeric range.")
            return info
        origin = end
        frame = {"x": history.reshape(1, len(signal), 1),
                 "y": np.zeros((1, len(horizons)), np.float32)}
        pred, _, _ = _predict(model, run["engine_id"], frame, run["scaler"],
                              **({"should_stop": should_stop} if should_stop else {}))
        if not np.isfinite(pred).all():
            info.update(status="stopped", reason="The model returned a nonfinite forecast; continuation stopped.")
            return info
        block = [{"target_time_s": origin + float(h), "value": float(pred[0, j]),
                  "lower": None, "upper": None, "kind": "recursive"}
                 for j, h in enumerate(horizons)]
        points.extend(block)


def _check_cancelled(should_stop) -> None:
    if should_stop and should_stop():
        raise InterruptedError("Signal forecast cancelled")


def _crossing(current: float, threshold: dict, points: list[dict], issued: float,
              *, complete: bool = True) -> dict:
    if threshold["status"] != "available":
        return {"status": "unavailable", "time_s": None}
    if is_beyond(current, threshold["red"], threshold["direction"]):
        return {"status": "already_red", "time_s": issued}
    hit = next((p for p in points if is_beyond(p["value"], threshold["red"], threshold["direction"])), None)
    status = "none_within_horizon" if points else "unavailable"
    return {"status": "predicted" if hit else (status if complete else "pending"),
            "time_s": hit["target_time_s"] if hit else None}


@lru_cache(maxsize=12)
def _load_model(project_id: str, run_id: str, artifact_digest: str, root: str):
    # Hash is checked before reaching this cache; root separates test stores.
    run = load_signal_run(project_id, run_id)
    params = run["params"]
    if run["engine_id"] in {"gru", "lstm"}:
        model = SignalRecurrent(run["engine_id"], 1, params["hidden_size"], len(params["horizons_s"]),
                                residual_forecast=params.get("residual_forecast", False))
        state = torch.load(run["artifact_path"], map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model
    model = joblib.load(run["artifact_path"])
    if run["engine_id"] == "full_cns":
        if not isinstance(model, SignalFullCNS) or model.provenance != run["connectome"]:
            raise ValueError("Saved Full MaleCNS model differs from its connectome provenance")
    return model


def forecast_prefix(project_id: str, run_id: str, unit_id: str, as_of_s: float,
                    thresholds: dict | None = None, *, rollout_steps: int | None = None,
                    should_stop=None, progress_cb=None) -> dict:
    """Score only rows at/before as_of; never feed hidden future observations.

    ``thresholds`` replaces the run's saved zone rule for display and red-crossing
    only; forecast values at shared timestamps do not depend on it. Optional
    rollout searches up to ``rollout_steps`` Training intervals, or continues
    for at least ten intervals after the first red entry. It never forces one.
    """
    from pdm.projects import project_store

    _check_cancelled(should_stop)
    if not np.isfinite(float(as_of_s)):
        raise ValueError("Replay time must be finite")
    if rollout_steps is not None and (isinstance(rollout_steps, bool)
                                     or not isinstance(rollout_steps, int) or not 1 <= rollout_steps <= 960):
        raise ValueError("Recursive search requires 1–960 steps")
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
              "red_entry_corridor": {"status": "unavailable", "reason": "No trained event-time interval"},
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
    _check_cancelled(should_stop)
    model = _load_model(project_id, run_id, run["artifacts"][run["artifact"]], str(project_store().root))
    frame = {"x": x, "y": np.zeros((1, len(run["params"]["horizons_s"])), np.float32)}
    pred, lower, upper = _predict(model, run["engine_id"], frame, run["scaler"],
                                 **({"should_stop": should_stop} if should_stop else {}))
    points = []
    for j, horizon in enumerate(run["params"]["horizons_s"]):
        value = float(pred[0, j])
        if not np.isfinite(value):
            continue
        points.append({"target_time_s": issued + float(horizon), "value": value,
                       "lower": float(lower[0, j]) if lower is not None else None,
                       "upper": float(upper[0, j]) if upper is not None else None})
    current = float(prefix.signal.iloc[-1])

    def publish(partial_points: list[dict], info: dict) -> None:
        _check_cancelled(should_stop)
        if progress_cb:
            progress_cb({**result, "points": [dict(p) for p in partial_points], "rollout": dict(info),
                         "status": "available", "crossing": _crossing(current, threshold, partial_points, issued,
                                                                         complete=False)})

    if rollout_steps is not None:
        result["rollout"] = _extend_forecast(
            model, run, signal, points, issued, training_cadence(data), rollout_steps, threshold,
            segment.timestamp_s.to_numpy(float)[-history:], should_stop=should_stop, progress_cb=publish)
    _check_cancelled(should_stop)
    result["points"] = points
    result["status"] = "available" if points else "unavailable"
    result["reason"] = None if points else "No supported forecast points"
    result["crossing"] = _crossing(current, threshold, points, issued)
    return result
