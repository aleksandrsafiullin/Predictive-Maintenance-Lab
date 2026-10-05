"""Predeclared, bounded signal study settings derived from Train clocks only."""
from __future__ import annotations

import math

import numpy as np

from pdm.signal_training import ENGINES, _segments, average_training_duration_s


def funnel_training_profile(snapshot: dict, engine_id: str) -> dict:
    if engine_id not in ENGINES:
        raise ValueError(f"Unsupported numeric signal engine: {engine_id}")
    segments = [segment for uid in snapshot["split"]["train"]
                for segment in _segments(snapshot["features"], str(uid))]
    deltas = [value for segment in segments
              for value in np.diff(segment.timestamp_s.to_numpy(float))
              if np.isfinite(value) and value > 0]
    if not deltas:
        raise ValueError("Training needs increasing observation clocks")
    cadence = float(np.median(deltas))
    history = min(8, max(2, max(map(len, segments)) - 1))
    end_steps = max(1, math.ceil(average_training_duration_s(snapshot) / cadence - 1e-9))
    steps = np.unique(np.rint(np.geomspace(1, end_steps, 24)).astype(int))
    tolerance = max(0.000001, min(cadence * 0.01, 0.01))
    horizons = []
    for step in steps:
        horizon = float(step * cadence)
        supported = False
        for segment in segments:
            times = segment.timestamp_s.to_numpy(float)
            desired = times[history - 1:] + horizon
            left = np.searchsorted(times, desired - tolerance, side="left")
            right = np.searchsorted(times, desired + tolerance, side="right")
            if np.any((right - left == 1) & (left < len(times))):
                supported = True
                break
        if supported:
            horizons.append(horizon)
    if not horizons:
        raise ValueError("Training has no supported targets after the history window")
    return {
        "forecast_mode": "joint_residual_paths",
        "history_length": history,
        "horizons_s": horizons,
        "epochs": 30,
        "hidden_size": 32,
        "batch_size": 64,
        "learning_rate": 0.001,
        "max_iter": 80,
        "seed": 20261002,
        "residual_forecast": engine_id in {"gru", "lstm"},
        "cv_folds": 3,
        "max_windows_per_unit": 128,
        "path_samples": 256,
        "nominal_coverage": 0.9,
    }


def learned_bearings_training_profile(snapshot: dict, engine_id: str) -> dict | None:
    """Use the direct learned distribution for admitted nonnegative bearing RMS."""
    if snapshot["schema"].get("signal_label") != "Combined max-axis RMS":
        return None
    from pdm.learned_trajectory import learned_params
    features=snapshot["features"][snapshot["features"].unit_id.astype(str).isin(map(str,snapshot["split"]["train"]))]
    return learned_params(engine_id, {"forecast_mode":"learned_joint_trajectories",
                         "history_length":8,"hidden_size":128,"num_layers":2,"epochs":150,
                         "max_windows_per_unit":256,"training_samples":32,"path_samples":256,"batch_size":32,"patience":25,
                         "seed":20261002,"width_weight":.2,"miss_weight":5.,"event_weight":.5,
                         "energy_weight":1.,"phase_weight":.5},features)
