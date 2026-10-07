"""The shared physical signal contract; no evaluator metadata is a feature."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

TASK = "probabilistic_signal_forecast"
SUPPORTED_ENGINES = ("persistence", "local_trend", "quantile_boosting", "gru")
SPLITS = ("train", "validation", "calibration", "test")
REFERENCE_ORIGIN_POLICY = (
    "baseline-v1:sha256-evaluation-anchor:common-history-60:full-future-60"
)
DENSE_V2_REFERENCE_ORIGIN_POLICY = (
    "dense-v2:earliest-causal-common-history-60:sha256-physical-unit-id"
)
BOUNDED_TREND_MODE = "bounded_trend_v1"
TREND_CENTER_PROTOCOL = "relative_shared_lead_v1"
TREND_CORRIDOR_PROTOCOL = "fixed_point_decision_v1"
BALANCED_TRAIN_PROTOCOL = "balanced_rolling_reference_v1"


def is_bounded_trend(config: dict) -> bool:
    return config.get("forecast_mode") == BOUNDED_TREND_MODE


def is_reference_balanced(config: dict) -> bool:
    return config.get("training_population_protocol") == BALANCED_TRAIN_PROTOCOL


def trend_config(config: dict, H: int | None = None) -> dict:
    cfg = extended_config(config, config.get("max_horizon", 60) if H is None else H)
    return default_config(**{**cfg, "forecast_mode": BOUNDED_TREND_MODE,
                             "center_protocol": TREND_CENTER_PROTOCOL,
                             "corridor_protocol": TREND_CORRIDOR_PROTOCOL,
                             "quantiles": None, "nominal": None, "coverage_guarantee": False,
                             "output_kind": "decision_corridor", "output_slots": ["lower", "center", "upper"]})


def is_dense_v2(config: dict) -> bool:
    return config.get("horizon_protocol") == "dense-v2"


def report_horizons_for(H: int) -> list[int]:
    if isinstance(H, bool) or not isinstance(H, int) or not 1 <= H <= 4096:
        raise ValueError("Dense-v2 H must be an integer in 1..4096")
    return sorted({h for h in (5, 15, 30, 60) if h <= H} | set(range(60, H + 1, 60)) | {H})


def reference_origin_policy(config: dict) -> str:
    return DENSE_V2_REFERENCE_ORIGIN_POLICY if is_dense_v2(config) else REFERENCE_ORIGIN_POLICY


def extended_config(config: dict, H: int) -> dict:
    """Opt into a newly trained direct grid without modifying an old config."""
    reports = report_horizons_for(H)
    return default_config(**{**config, "horizon_protocol": "dense-v2", "max_horizon": H,
                             "horizons_s": [float(config.get("cadence_s", 60.0)) * h for h in range(1, H + 1)],
                             "report_horizons": reports})


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_threshold_direction(direction: str) -> None:
    if not isinstance(direction, str) or direction not in ("above", "below"):
        raise ValueError("threshold_direction must be 'above' or 'below'")


def threshold_crossed(values, threshold, direction="above"):
    """Inclusive single-observation event rule in the physical signal domain."""
    validate_threshold_direction(direction)
    return np.less_equal(values, threshold) if direction == "below" else np.greater_equal(values, threshold)


def default_config(**overrides: Any) -> dict:
    config = {
        "task": TASK,
        "engine_id": "gru",
        "target": "vibration_rms_g",
        "unit": "g",
        "schema": ["unit_id", "timestamp_s", "vibration_rms_g"],
        "cadence_s": 60.0,
        "clock_tolerance_s": 1e-6,
        "positive_domain": True,
        "history_length": 60,
        "common_history_length": 60,
        "max_horizon": 60,
        "horizons_s": [60.0 * h for h in range(1, 61)],
        "report_horizons": [5, 15, 30, 60],
        "quantiles": [0.05, 0.50, 0.95],
        "nominal": 0.90,
        "scale_floor": 0.02,
        "seed": 42,
        "origin_stride": 5,
        "max_origins_per_unit": 128,
        "hidden_size": 64,
        "layers": 1,
        "learning_rate": 0.001,
        "batch_size": 64,
        "max_epochs": 80,
        "patience": 10,
        "gradient_clip": 1.0,
        "boosting_max_iter": 140,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 35,
        "boosting_learning_rate": 0.075,
        "l2_regularization": 1.0,
        "yellow": 0.55,
        "red": 0.80,
        "width_budget": 0.30,
    }
    config.update(overrides)
    if is_bounded_trend(config):
        if "quantiles" not in overrides:
            config["quantiles"] = None
        if "nominal" not in overrides:
            config["nominal"] = None
        for key, value in {"center_protocol": TREND_CENTER_PROTOCOL,
                           "corridor_protocol": TREND_CORRIDOR_PROTOCOL,
                           "trend_basis_size": 16, "center_loss_weight": 1.0,
                           "point_excess_loss_weight": 1.0,
                           "validation_rolling_weight": .5,
                           "validation_reference_weight": .5, "coverage_guarantee": False,
                           "output_kind": "decision_corridor", "output_slots": ["lower", "center", "upper"]}.items():
            config.setdefault(key, value)
    # The dense grid follows an explicitly changed cadence, never the display horizon.
    if "cadence_s" in overrides and "horizons_s" not in overrides:
        config["horizons_s"] = [float(config["cadence_s"]) * h for h in range(1, config["max_horizon"] + 1)]
    if "target" in overrides and "schema" not in overrides:
        config["schema"] = ["unit_id", "timestamp_s", config["target"]]
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    validate_threshold_direction(config.get("threshold_direction", "above"))
    if "forecast_mode" in config and not is_bounded_trend(config):
        raise ValueError("Unsupported forecast_mode")
    if "training_population_protocol" in config and (not is_reference_balanced(config) or not is_bounded_trend(config)):
        raise ValueError("Unsupported training_population_protocol or non-bounded forecast mode")
    if is_bounded_trend(config):
        if not is_dense_v2(config) or config.get("center_protocol") != TREND_CENTER_PROTOCOL or config.get("corridor_protocol") != TREND_CORRIDOR_PROTOCOL:
            raise ValueError("Bounded trend requires the frozen center/corridor and dense-v2 protocols")
        if config.get("trend_basis_size") != 16:
            raise ValueError("The first bounded trend decoder has 16 fixed basis functions")
        if config.get("quantiles") is not None or config.get("nominal") is not None or config.get("coverage_guarantee") is not False:
            raise ValueError("Decision bounds have no quantile estimates or nominal guarantee")
        if config.get("output_kind") != "decision_corridor" or config.get("output_slots") != ["lower", "center", "upper"]:
            raise ValueError("Decision output slots must be explicitly lower/center/upper")
        if not np.isfinite(config["width_budget"]) or not 0 < config["width_budget"] <= .30:
            raise ValueError("Decision corridor FULL width_budget must be in (0,0.30]")
        for key in ("center_loss_weight", "point_excess_loss_weight"):
            if config.get(key) != 1.0:
                raise ValueError(f"Frozen bounded trend objective requires {key}=1")
        for key in ("validation_rolling_weight", "validation_reference_weight"):
            if config.get(key) != .5:
                raise ValueError(f"Frozen bounded trend selection requires {key}=0.5")
    elif any(key in config for key in ("center_protocol", "corridor_protocol")):
        raise ValueError("Center/corridor protocol requires explicit forecast_mode")
    if config.get("task") != TASK or config.get("engine_id") not in SUPPORTED_ENGINES:
        raise ValueError("Unsupported probabilistic signal task or engine")
    if config.get("schema") != ["unit_id", "timestamp_s", config.get("target")]:
        raise ValueError("Only the declared single observed signal schema is supported")
    if not isinstance(config.get("target"), str) or not config["target"] or not config.get("unit"):
        raise ValueError("Target name and physical unit are required")
    cadence, tolerance = float(config["cadence_s"]), float(config["clock_tolerance_s"])
    if not np.isfinite(cadence) or cadence <= 0 or not 0 <= tolerance < cadence / 2:
        raise ValueError("Cadence and explicit clock tolerance are invalid")
    if config["positive_domain"] is not True:
        raise ValueError("This release supports the positive single-channel profile only")
    if "horizon_protocol" in config and not is_dense_v2(config):
        raise ValueError("Unsupported horizon_protocol")
    horizon = config.get("max_horizon")
    if is_dense_v2(config):
        report_horizons_for(horizon)
    elif horizon != 60:
        raise ValueError("The first comparison contract requires dense 60-lead forecasts/common history")
    if config.get("common_history_length") != 60:
        raise ValueError("Common causal history must remain 60 observations")
    length = config["history_length"]
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= 60:
        raise ValueError("history_length must be an integer in 1..60")
    if not np.array_equal(np.asarray(config["horizons_s"]), cadence * np.arange(1, horizon + 1)):
        raise ValueError("Unsupported horizon grid")
    if config["report_horizons"] != (report_horizons_for(horizon) if is_dense_v2(config) else [5, 15, 30, 60]):
        raise ValueError("Only H=5/15/30/60 has a first-version calibration protocol")
    if not is_bounded_trend(config) and (config["quantiles"] != [0.05, 0.5, 0.95] or config["nominal"] != 0.90):
        raise ValueError("Only q05/q50/q95 and nominal=0.90 are supported")
    for key in ("scale_floor", "learning_rate", "boosting_learning_rate", "gradient_clip"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in (
        "origin_stride", "max_origins_per_unit", "hidden_size", "layers", "batch_size",
        "max_epochs", "patience", "boosting_max_iter", "max_leaf_nodes", "min_samples_leaf",
    ):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["layers"] != 1:
        raise ValueError("The first GRU matrix has one recurrent layer")
    canonical_hash(config)  # Reject non-JSON/nonfinite parameters before artifact binding.


def config_hash(config: dict) -> str:
    validate_config(config)
    return canonical_hash(config)


def unit_balanced_weights(mask: np.ndarray, units: np.ndarray) -> np.ndarray:
    """Equal units, equal supported leads within unit, equal origins within lead."""
    mask = np.asarray(mask, dtype=bool)
    units = np.asarray(units, dtype=str)
    if mask.ndim != 2 or units.shape != (mask.shape[0],):
        raise ValueError("Expected (origins, leads) mask and one unit per origin")
    weights = np.zeros(mask.shape, dtype=float)
    active = [u for u in np.unique(units) if mask[units == u].any()]
    for unit in active:
        rows = np.flatnonzero(units == unit)
        counts = mask[rows].sum(axis=0)
        leads = counts > 0
        weights[rows] = np.divide(
            mask[rows], len(active) * int(leads.sum()) * counts,
            out=np.zeros_like(mask[rows], dtype=float), where=leads,
        )
    return weights
