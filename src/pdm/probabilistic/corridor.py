"""Fixed point decision bounds and empirical support; no nominal guarantee."""
from __future__ import annotations

import numpy as np

from .contract import canonical_hash, is_bounded_trend, unit_balanced_weights, validate_config

DECISION_SCOPE = "independent_physical_reference:empirical_point_containment;not_simultaneous_coverage"


def decision_outputs(center, config):
    validate_config(config)
    center = np.asarray(center, dtype=float)
    if not is_bounded_trend(config) or not np.isfinite(center).all() or (center <= 0).any():
        raise ValueError("Decision center requires the explicit positive bounded trend protocol")
    radius = config["width_budget"] / 2
    with np.errstate(over="ignore", invalid="ignore"):
        outputs = np.stack((center * (1 - radius), center, center * (1 + radius)), axis=-1)
    if not np.isfinite(outputs).all() or (outputs <= 0).any() or (np.diff(outputs, axis=-1) < 0).any():
        raise ValueError("Expected finite ordered positive decision bounds")
    return outputs


def _check_prefix(H, config):
    if isinstance(H, bool) or not isinstance(H, int) or not 1 <= H <= config["max_horizon"]:
        raise ValueError("Decision prefix exceeds the saved direct grid")


def corridor_prefix(assessment, H, config):
    validate_config(config)
    _check_prefix(H, config)
    result = {"requested_H": H, "assessment_H": H, "calibration_H": H,
              "conservative_prefix": False, "calibrated": False, "nominal": None,
              "coverage_guarantee": False, "status": "unassessed_decision_corridor",
              "scope": "unassessed_point_decision_bounds", "n": 0, "rank": None,
              "issued_units": 0, "supported_units": 0, "known_targets": 0,
              "complete_paths": 0, "partial_paths": 0, "no_future_paths": 0,
              "point_coverage": None, "center_mae": None, "full_relative_width_max": None}
    if assessment is None:
        return result
    from .calibration import validate_calibrator
    validate_calibrator(assessment)
    if assessment.get("assessment_kind") != "empirical_point_decision_v1" or assessment["config_hash"] != canonical_hash(config):
        raise ValueError("Decision assessment protocol/config mismatch")
    # JSON stores an empty (0,H) grid as []; reconstruct its saved dense shape.
    units = np.asarray(assessment["assessment_physical_units"], dtype=str)
    shape = (len(units), config["max_horizon"])
    def grid(key, dtype):
        array = np.asarray(assessment[key], dtype=dtype)
        if not len(units) and not array.size:
            array = array.reshape(shape)
        if array.shape != shape:
            raise ValueError("Decision assessment grid/unit shape mismatch")
        return array[:, :H]
    mask = grid("target_support_mask", bool)
    if (np.diff(mask.astype(int), axis=1) > 0).any():
        raise ValueError("Decision assessment target masks must be continuous prefixes")
    inside = grid("point_containment_mask", bool)
    errors = grid("center_absolute_error", float)
    # Arrays use origin record order, which can differ from sorted identity metadata.
    if not len(units):
        if assessment["full_relative_width_max"] is not None or assessment["full_relative_width_max_by_lead"] != [None] * config["max_horizon"]:
            raise ValueError("Zero-issued assessment width must be unavailable")
        result.update(status="no_issued_reference_origins", scope=DECISION_SCOPE)
        return result
    weights = unit_balanced_weights(mask, units)
    active = mask.any(axis=1)
    complete = mask.all(axis=1)
    n = int(mask[:, -1].sum())
    widths = np.asarray(assessment["full_relative_width_max_by_lead"], dtype=float)
    if widths.shape != (config["max_horizon"],) or not np.isfinite(widths).all() or (widths < 0).any():
        raise ValueError("Decision assessment per-lead width metadata is invalid")
    result.update(status="empirically_assessed_decision_corridor", scope=DECISION_SCOPE,
                  n=n, issued_units=len(units), supported_units=int(active.sum()),
                  known_targets=int(mask.sum()), complete_paths=int(complete.sum()),
                  partial_paths=int((active & ~complete).sum()), no_future_paths=int((~active).sum()),
                  point_coverage=float((inside * weights).sum()) if mask.any() else None,
                  center_mae=float((errors * weights).sum()) if mask.any() else None,
                  full_relative_width_max=float(widths[:H].max()))
    return result


def decision_band(outputs, config, H, *, assessment=None, model_hash=None):
    validate_config(config)
    _check_prefix(H, config)
    outputs = np.asarray(outputs, dtype=float)
    if outputs.shape[-2:] != (config["max_horizon"], 3):
        raise ValueError("Decision output shape does not match the frozen grid")
    if not np.isfinite(outputs).all() or (outputs <= 0).any() or (np.diff(outputs, axis=-1) < 0).any():
        raise ValueError("Expected finite ordered positive decision bounds")
    center = outputs[..., 1]
    expected = decision_outputs(center, config)
    if not np.allclose(outputs, expected, rtol=1e-12, atol=0):
        raise ValueError("Decision slots must be the frozen lower/center/upper bounds")
    if assessment is not None and model_hash is not None and assessment["model_hash"] != model_hash:
        raise ValueError("Decision assessment model binding mismatch")
    projection = corridor_prefix(assessment, H, config)
    return {"outputs": outputs.copy(), "center": center.copy(),
            "lower": outputs[..., 0].copy(), "upper": outputs[..., 2].copy(),
            "pointwise_lower": outputs[..., 0].copy(), "pointwise_upper": outputs[..., 2].copy(),
            "trajectory_lower": outputs[..., :H, 0].copy(), "trajectory_upper": outputs[..., :H, 2].copy(),
            "H": H, "output_kind": "decision_corridor", "forecast_mode": config["forecast_mode"],
            "coverage_guarantee": False, "nominal": None, "calibrated": False,
            "empirically_assessed": assessment is not None and projection["issued_units"] > 0, "assessment": projection,
            "status": projection["status"], "scope": projection["scope"],
            "n": projection["n"], "rank": None, "quality_accepted": False,
            "calibrator_hash": assessment["calibrator_hash"] if assessment else None,
            "warnings": ["decision_bounds_have_no_nominal_coverage_guarantee"]}


def center_objective(y, center, mask, weights, config):
    """Raw positive targets, mean relative center error and mean point excess."""
    active = np.asarray(mask, dtype=bool)
    if not active.any():
        raise ValueError("No supported decision targets")
    log_y = np.log(np.asarray(y)[active])
    log_c = np.log(np.asarray(center)[active])
    half = config["width_budget"] / 2
    error = np.abs(log_y - log_c)
    excess = np.maximum(log_c + np.log1p(-half) - log_y, 0) + np.maximum(log_y - log_c - np.log1p(half), 0)
    return float(np.sum(np.asarray(weights)[active] * (error + excess)))
