"""Finite-rank independent calibration, explicitly scoped to one reference origin."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .contract import is_bounded_trend, is_dense_v2, reference_origin_policy
from .models import QUANTILES, json_hash, predict, validate_model

REFERENCE_SCOPE = "one_independent_physical_unit:sha256_reference_origin:complete_60_future"


def _hash_windows(windows):
    digest = hashlib.sha256()
    for key in ("x", "y", "mask", "physical_units", "origin_s"):
        array = np.asarray(windows[key])
        digest.update(key.encode())
        if array.dtype.kind in "OUS":
            digest.update(json.dumps(array.tolist(), sort_keys=True).encode())
        else:
            digest.update(str(array.shape).encode())
            digest.update(np.ascontiguousarray(array).tobytes())
    digest.update(str(windows.get("dataset_hash", "")).encode())
    digest.update(str(windows.get("origin_policy", "")).encode())
    return digest.hexdigest()


def _calibrator_hash(calibrator):
    return json_hash({k: v for k, v in calibrator.items() if k != "calibrator_hash"})


def fit_calibrator(frozen_model, calibration_windows, config):
    validate_model(frozen_model)
    cfg = frozen_model["config"]
    dense_v2 = is_dense_v2(cfg)
    if dense_v2 and (config != cfg or calibration_windows.get("config") != cfg):
        raise ValueError("Dense-v2 Calibration configuration must equal the frozen model configuration")
    # Optional event semantics bind new rules without changing old Above payloads.
    new_rule = cfg.get("threshold_direction", "above") == "below" or "zone_rule_hash" in cfg
    for candidate, label in ((config, "Calibration configuration"),
                             (calibration_windows.get("config", {}), "Calibration window configuration")):
        if candidate.get("threshold_direction", "above") != cfg.get("threshold_direction", "above"):
            raise ValueError(f"{label} mismatch: threshold_direction")
        if ("zone_rule_hash" in candidate or "zone_rule_hash" in cfg) and candidate.get("zone_rule_hash") != cfg.get("zone_rule_hash"):
            raise ValueError(f"{label} mismatch: zone_rule_hash")
        if new_rule:
            for key in ("red", "yellow"):
                if key not in candidate or candidate[key] != cfg[key]:
                    raise ValueError(f"{label} mismatch: {key}")
    # Every modelling/calibration setting binds to the frozen contract.
    for key in ("target", "unit", "cadence_s", "history_length", "max_horizon", "report_horizons", "nominal", "scale_floor", "positive_domain"):
        if key in config and config[key] != cfg[key]:
            raise ValueError(f"Calibration configuration mismatch: {key}")
    if calibration_windows.get("split", "calibration") != "calibration":
        raise ValueError("Only the independent Calibration split can fit corrections")
    policy = calibration_windows.get("origin_policy", "")
    if policy != reference_origin_policy(cfg):
        raise ValueError("Calibration requires the frozen SHA256 reference-origin policy")
    for key in ("target", "unit", "cadence_s", "schema", "history_length", "max_horizon", "positive_domain"):
        if key in calibration_windows.get("config", {}) and calibration_windows["config"][key] != cfg[key]:
            raise ValueError(f"Calibration window configuration mismatch: {key}")
    if "config" not in calibration_windows or calibration_windows.get("config_hash") != json_hash(calibration_windows["config"]):
        raise ValueError("Calibration window config hash mismatch")
    source = frozen_model["training_summary"].get("source_binding", {})
    if not source or not calibration_windows.get("dataset_hash") or not calibration_windows.get("profile"):
        raise ValueError("Calibration requires explicit frozen source provenance")
    for key, value in source.items():
        if calibration_windows.get(key) != value:
            raise ValueError(f"Calibration source provenance mismatch: {key}")
    manifest = calibration_windows.get("origin_manifest")
    if not isinstance(manifest, dict):
        raise ValueError("Calibration requires a signed reference origin manifest")
    if isinstance(manifest, dict):
        unsigned = {key: value for key, value in manifest.items() if key != "origin_hash"}
        if manifest.get("origin_hash") != json_hash(unsigned) or manifest.get("origin_policy") != reference_origin_policy(cfg):
            raise ValueError("Calibration origin manifest integrity/policy mismatch")
        origin_contract = manifest.get("origin_contract", {})
        if dense_v2 and origin_contract.get("horizon_protocol") != "dense-v2":
            raise ValueError("Calibration origin manifest contract mismatch: horizon_protocol")
        if dense_v2:
            for key in ("target", "unit", "schema", "cadence_s", "clock_tolerance_s", "positive_domain",
                        "common_history_length", "max_horizon"):
                if origin_contract.get(key) != cfg[key]:
                    raise ValueError(f"Calibration origin manifest contract mismatch: {key}")
        if new_rule and "threshold_direction" not in origin_contract:
            raise ValueError("Calibration origin manifest contract mismatch: threshold_direction")
        if origin_contract.get("threshold_direction", "above") != cfg.get("threshold_direction", "above"):
            raise ValueError("Calibration origin manifest contract mismatch: threshold_direction")
        if ("zone_rule_hash" in origin_contract or "zone_rule_hash" in cfg) and origin_contract.get("zone_rule_hash") != cfg.get("zone_rule_hash"):
            raise ValueError("Calibration origin manifest contract mismatch: zone_rule_hash")
        if new_rule:
            for key in ("red", "yellow"):
                if key not in origin_contract or origin_contract[key] != cfg[key]:
                    raise ValueError(f"Calibration origin manifest contract mismatch: {key}")
        for key in ("cadence_s", "max_horizon", "common_history_length"):
            if manifest.get(key) != cfg[key]:
                raise ValueError(f"Calibration origin manifest contract mismatch: {key}")
        for key, value in source.items():
            if manifest.get(key) != value:
                raise ValueError(f"Calibration origin manifest source mismatch: {key}")
        actual_records = [{"unit_id": str(u), "physical_unit_id": str(p), "origin_s": float(o)} for u, p, o in zip(calibration_windows["units"], calibration_windows["physical_units"], calibration_windows["origin_s"])]
        if manifest.get("records") != actual_records:
            raise ValueError("Calibration origin records mismatch")
    x = np.asarray(calibration_windows["x"], dtype=float)
    y = np.asarray(calibration_windows["y"], dtype=float)
    mask = np.asarray(calibration_windows["mask"], dtype=bool)
    units = np.asarray(calibration_windows["physical_units"]).astype(str)
    if len(units) != len(set(units)):
        raise ValueError("Calibration requires one origin per independent physical unit")
    fitted_units = set(frozen_model["training_summary"]["train_units"]) | set(frozen_model["training_summary"]["validation_units"])
    if fitted_units & set(units):
        raise ValueError("Calibration units overlap Train/Validation units")
    if x.shape != (len(units), cfg["history_length"]) or y.shape != (len(units), cfg["max_horizon"]) or mask.shape != y.shape:
        raise ValueError("Calibration grid/history shape mismatch")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Calibration history must be finite nonnegative")
    if dense_v2:
        if (x <= 0).any() or not np.isfinite(y[mask]).all() or (y[mask] <= 0).any():
            raise ValueError("Dense-v2 Calibration supported observations must be finite positive")
        # A prefix cannot resume after a missing observation or telemetry barrier.
        if (np.diff(mask.astype(int), axis=1) > 0).any():
            raise ValueError("Dense-v2 Calibration target masks must be continuous prefixes")
        if is_bounded_trend(cfg):
            return _fit_decision_assessment(frozen_model, calibration_windows, cfg, x, y, mask, units)
        return _fit_dense_v2(frozen_model, calibration_windows, cfg, x, y, mask, units)
    if not mask.all() or not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Reference Calibration needs complete finite common 60-step future paths")
    n = len(units)
    rank = math.ceil((n + 1) * cfg["nominal"])
    available = n > 0 and rank <= n
    calibrator = {"model_hash": frozen_model["model_hash"], "config_hash": json_hash(cfg),
                  "preprocessing_hash": json_hash(frozen_model["preprocessing"]),
                  "calibration_hash": _hash_windows(calibration_windows),
                  "calibration_snapshot_hash": calibration_windows.get("dataset_hash"),
                  "calibration_physical_units": sorted(units.tolist()),
                  "origin_policy": policy, "origin_manifest": calibration_windows.get("origin_manifest", []),
                  "scope": REFERENCE_SCOPE, "status": "calibrated_reference_scope" if available else "insufficient_calibration_data",
                  "n": n, "rank": rank, "nominal": cfg["nominal"], "scale_floor": cfg["scale_floor"],
                  "positive_domain": cfg["positive_domain"], "target": cfg["target"], "unit": cfg["unit"],
                  "cadence_s": cfg["cadence_s"], "history_length": cfg["history_length"],
                  "quantiles": QUANTILES.tolist(), "max_horizon": 60, "report_horizons": cfg["report_horizons"],
                  "warnings": ["small_calibration_sample"] if n < 50 else [],
                  "c_point": None, "c_path": {str(h): None for h in cfg["report_horizons"]}}
    if available:
        raw = predict(frozen_model, x)
        low, high = raw[..., 0], raw[..., 2]
        width = np.maximum(high - low, cfg["scale_floor"])
        residuals = np.maximum(np.maximum(low - y, y - high), 0) / width
        calibrator["c_point"] = np.sort(residuals, axis=0)[rank - 1].tolist()
        calibrator["c_path"] = {str(h): float(np.sort(residuals[:, :h].max(axis=1))[rank - 1]) for h in cfg["report_horizons"]}
    calibrator["calibrator_hash"] = _calibrator_hash(calibrator)
    return calibrator


def _fit_decision_assessment(model, windows, cfg, x, y, mask, units):
    from .corridor import DECISION_SCOPE, corridor_prefix
    outputs = predict(model, x)
    low, center, high = outputs[..., 0], outputs[..., 1], outputs[..., 2]
    inside = mask & (y >= low) & (y <= high)
    errors = np.where(mask, np.abs(np.where(mask, y, 0) - center), 0)
    counts = mask.sum(axis=0).astype(int)
    coverage = [float(inside[:, j].sum() / counts[j]) if counts[j] else None for j in range(cfg["max_horizon"])]
    mae = [float(errors[:, j].sum() / counts[j]) if counts[j] else None for j in range(cfg["max_horizon"])]
    cal = {"horizon_protocol": "dense-v2", "forecast_mode": cfg["forecast_mode"],
           "assessment_kind": "empirical_point_decision_v1", "output_kind": "decision_corridor",
           "coverage_guarantee": False, "nominal": None, "config": dict(cfg),
           "model_hash": model["model_hash"], "config_hash": json_hash(cfg),
           "preprocessing_hash": json_hash(model["preprocessing"]), "calibration_hash": _hash_windows(windows),
           "calibration_snapshot_hash": windows.get("dataset_hash"),
           "calibration_physical_units": sorted(units.tolist()), "assessment_physical_units": units.tolist(),
           "origin_policy": windows["origin_policy"], "origin_manifest": windows["origin_manifest"],
           "scope": DECISION_SCOPE, "status": "empirically_assessed_decision_corridor" if len(units) else "no_issued_reference_origins",
           "n": len(units), "rank": None, "scale_floor": cfg["scale_floor"],
           "positive_domain": cfg["positive_domain"], "target": cfg["target"], "unit": cfg["unit"],
           "cadence_s": cfg["cadence_s"], "history_length": cfg["history_length"],
           "output_slots": ["lower", "center", "upper"], "max_horizon": cfg["max_horizon"],
           "report_horizons": cfg["report_horizons"], "warnings": ["no_nominal_coverage_guarantee"],
           "n_by_lead": counts.tolist(), "point_coverage_by_lead": coverage, "center_mae_by_lead": mae,
           "target_support_mask": mask.tolist(), "point_containment_mask": inside.tolist(),
           "center_absolute_error": errors.tolist(),
           "full_relative_width_max_by_lead": np.max((high - low) / np.maximum(np.abs(center), cfg["scale_floor"]), axis=0).tolist() if len(center) else [None] * cfg["max_horizon"],
           "full_relative_width_max": float(np.max((high - low) / np.maximum(np.abs(center), cfg["scale_floor"]))) if center.size else None}
    cal["calibrator_hash"] = _calibrator_hash(cal)
    cal["assessment_by_horizon"] = {str(h): corridor_prefix(cal, h, cfg) for h in cfg["report_horizons"]}
    cal["calibrator_hash"] = _calibrator_hash(cal)
    return cal


def _fit_dense_v2(model, windows, cfg, x, y, mask, units):
    """Same residual/rank method, with an independent complete-prefix population."""
    raw = predict(model, x)
    low, high = raw[..., 0], raw[..., 2]
    width = np.maximum(high - low, cfg["scale_floor"])
    safe_y = np.where(mask, y, 0.0)
    residuals = np.maximum(np.maximum(low - safe_y, safe_y - high), 0) / width
    counts = mask.sum(axis=0).astype(int)
    ranks = np.ceil((counts + 1) * cfg["nominal"]).astype(int)
    point = [float(np.sort(residuals[mask[:, j], j])[ranks[j] - 1])
             if counts[j] > 0 and ranks[j] <= counts[j] else None for j in range(cfg["max_horizon"])]
    n_by_h, rank_by_h, status_by_h, path = {}, {}, {}, {}
    for h in cfg["report_horizons"]:
        complete = mask[:, :h].all(axis=1)
        n = int(complete.sum())
        rank = math.ceil((n + 1) * cfg["nominal"])
        available = n > 0 and rank <= n
        n_by_h[str(h)], rank_by_h[str(h)] = n, rank
        status_by_h[str(h)] = "calibrated_reference_scope" if available else "insufficient_calibration_data"
        path[str(h)] = float(np.sort(residuals[complete, :h].max(axis=1))[rank - 1]) if available else None
    n = len(units)
    cal = {"horizon_protocol": "dense-v2", "model_hash": model["model_hash"], "config_hash": json_hash(cfg),
           "preprocessing_hash": json_hash(model["preprocessing"]), "calibration_hash": _hash_windows(windows),
           "calibration_snapshot_hash": windows.get("dataset_hash"), "calibration_physical_units": sorted(units.tolist()),
           "origin_policy": windows["origin_policy"], "origin_manifest": windows["origin_manifest"],
           "scope": "one_independent_physical_unit:earliest_causal_reference_origin:complete_prefix_by_H",
           "status": "calibrated_reference_scope" if any(v is not None for v in path.values()) else "insufficient_calibration_data",
           "n": n, "rank": math.ceil((n + 1) * cfg["nominal"]), "nominal": cfg["nominal"],
           "scale_floor": cfg["scale_floor"], "positive_domain": cfg["positive_domain"],
           "target": cfg["target"], "unit": cfg["unit"], "cadence_s": cfg["cadence_s"], "history_length": cfg["history_length"],
           "quantiles": QUANTILES.tolist(), "max_horizon": cfg["max_horizon"], "report_horizons": cfg["report_horizons"],
           "warnings": ["small_calibration_sample"] if n < 50 else [], "c_point": point, "c_path": path,
           "n_by_horizon": n_by_h, "rank_by_horizon": rank_by_h, "status_by_horizon": status_by_h,
           "n_by_lead": counts.tolist(), "rank_by_lead": ranks.tolist(),
           "point_status_by_lead": ["calibrated_reference_scope" if c is not None else "insufficient_calibration_data" for c in point],
           "target_support_mask": mask.tolist()}
    cal["calibrator_hash"] = _calibrator_hash(cal)
    return cal


def validate_calibrator(calibrator, model=None):
    if calibrator["calibrator_hash"] != _calibrator_hash(calibrator):
        raise ValueError("Calibrator was mutated or has an invalid hash")
    if model is not None:
        validate_model(model)
        if is_bounded_trend(model["config"]) != (calibrator.get("assessment_kind") == "empirical_point_decision_v1"):
            raise ValueError("Decision assessment and nominal calibrator protocols are incompatible")
        if calibrator["model_hash"] != model["model_hash"] or calibrator["config_hash"] != json_hash(model["config"]):
            raise ValueError("Model/config incompatible with frozen calibrator")
        if calibrator["preprocessing_hash"] != json_hash(model["preprocessing"]):
            raise ValueError("Preprocessing incompatible with frozen calibrator")
    return calibrator


def calibration_prefix(calibrator, requested_h: int, config: dict) -> dict:
    """Select an available larger prefix; never borrow a shorter correction."""
    if is_bounded_trend(config):
        from .corridor import corridor_prefix
        return corridor_prefix(calibrator, requested_h, config)
    if isinstance(requested_h, bool) or not isinstance(requested_h, int) or not 1 <= requested_h <= config["max_horizon"]:
        raise ValueError("Requested prefix exceeds the saved direct horizon grid")
    candidates = [h for h in config["report_horizons"] if h >= requested_h]
    if calibrator is not None:
        validate_calibrator(calibrator)
        if calibrator["config_hash"] != json_hash(config):
            raise ValueError("Calibration prefix config binding mismatch")
    available = [h for h in candidates if calibrator is not None and calibrator["c_path"].get(str(h)) is not None]
    h = min(available or candidates)
    calibrated = bool(available)
    n = (calibrator.get("n_by_horizon", {}).get(str(h), calibrator["n"]) if calibrator else 0)
    rank = (calibrator.get("rank_by_horizon", {}).get(str(h), calibrator["rank"]) if calibrator else None)
    status = (calibrator.get("status_by_horizon", {}).get(str(h), calibrator["status"]) if calibrator else "uncalibrated")
    return {"requested_H": requested_h, "calibration_H": h, "conservative_prefix": h > requested_h,
            "calibrated": calibrated, "status": status, "n": n, "rank": rank,
            "scope": calibrator["scope"] if calibrator else "raw_marginal_diagnostic_only",
            "reason": "next_available_complete_prefix" if calibrated and h > requested_h else
                      "exact_complete_prefix" if calibrated else "no_available_complete_prefix_correction"}


def apply_calibrator(raw_quantiles, calibrator, H, *, model_hash=None, config=None):
    if is_bounded_trend(config or {}):
        from .corridor import decision_band
        return decision_band(raw_quantiles, config, H, assessment=calibrator, model_hash=model_hash)
    if calibrator is not None and calibrator.get("assessment_kind") == "empirical_point_decision_v1":
        raise ValueError("Decision assessment requires its explicit bounded configuration")
    dense_v2 = is_dense_v2(config or {}) or (calibrator is not None and is_dense_v2(calibrator))
    if dense_v2:
        return _apply_dense_v2(raw_quantiles, calibrator, H, model_hash=model_hash, config=config)
    raw = np.asarray(raw_quantiles, dtype=float)
    if raw.shape[-2:] != (60, 3) or not np.isfinite(raw).all() or (raw < 0).any() or (np.diff(raw, axis=-1) < 0).any():
        raise ValueError("Expected finite physical ordered nonnegative quantiles on frozen 60-step grid")
    if H not in (5, 15, 30, 60):
        raise ValueError("Unsupported H; no arbitrary interpolation of calibration correction")
    low, median, high = raw[..., 0], raw[..., 1], raw[..., 2]
    result = {"raw_quantiles": raw.copy(), "q05": low.copy(), "q50": median.copy(), "q95": high.copy(),
              "H": H, "status": "uncalibrated", "scope": "raw_marginal_diagnostic_only", "nominal": None,
              "pointwise_lower": low.copy(), "pointwise_upper": high.copy(),
              "trajectory_lower": low[..., :H].copy(), "trajectory_upper": high[..., :H].copy(),
              "calibrated": False, "calibrator_hash": None, "n": 0, "warnings": []}
    if calibrator is None:
        return result
    validate_calibrator(calibrator)
    if model_hash is not None and calibrator["model_hash"] != model_hash:
        raise ValueError("Model hash mismatch; calibrator invalidated")
    result.update(status=calibrator["status"], scope=calibrator["scope"], nominal=calibrator["nominal"],
                  n=calibrator["n"], warnings=calibrator["warnings"], calibrator_hash=calibrator["calibrator_hash"])
    correction = calibrator["c_path"].get(str(H))
    if correction is None or calibrator["c_point"] is None:
        # Retain raw diagnostics, but do not label an invented zero correction calibrated.
        return result
    amplitude = np.maximum(high - low, calibrator["scale_floor"])
    point_c = np.asarray(calibrator["c_point"])
    point_low = low - point_c * amplitude
    path_low = low[..., :H] - correction * amplitude[..., :H]
    if calibrator["positive_domain"]:
        point_low = np.maximum(point_low, 0)
        path_low = np.maximum(path_low, 0)
    result.update(pointwise_lower=point_low, pointwise_upper=high + point_c * amplitude,
                  trajectory_lower=path_low, trajectory_upper=high[..., :H] + correction * amplitude[..., :H],
                  calibrated=True)
    return result


def _apply_dense_v2(raw_quantiles, calibrator, H, *, model_hash, config):
    grid = config or calibrator
    raw = np.asarray(raw_quantiles, dtype=float)
    if raw.shape[-2:] != (grid["max_horizon"], 3) or not np.isfinite(raw).all() or (raw < 0).any() or (np.diff(raw, axis=-1) < 0).any():
        raise ValueError("Expected finite ordered physical quantiles on the frozen dense-v2 grid")
    if H not in grid["report_horizons"]:
        raise ValueError("Unsupported H; no interpolation of calibration correction")
    low, median, high = raw[..., 0], raw[..., 1], raw[..., 2]
    result = {"raw_quantiles": raw.copy(), "q05": low.copy(), "q50": median.copy(), "q95": high.copy(),
              "H": H, "horizon_protocol": "dense-v2", "status": "uncalibrated", "scope": "raw_marginal_diagnostic_only",
              "nominal": None, "pointwise_lower": low.copy(), "pointwise_upper": high.copy(),
              "trajectory_lower": low[..., :H].copy(), "trajectory_upper": high[..., :H].copy(),
              "pointwise_calibrated_mask": np.zeros(grid["max_horizon"], dtype=bool),
              "calibrated": False, "calibrator_hash": None, "n": 0, "rank": None, "warnings": []}
    if calibrator is None:
        return result
    validate_calibrator(calibrator)
    if not is_dense_v2(calibrator) or calibrator["max_horizon"] != grid["max_horizon"]:
        raise ValueError("Calibrator horizon protocol/grid mismatch")
    if config is not None and calibrator["config_hash"] != json_hash(config):
        raise ValueError("Calibrator config binding mismatch")
    if model_hash is not None and calibrator["model_hash"] != model_hash:
        raise ValueError("Model hash mismatch; calibrator invalidated")
    correction = calibrator["c_path"][str(H)]
    result.update(status=calibrator["status_by_horizon"][str(H)], scope=calibrator["scope"],
                  n=calibrator["n_by_horizon"][str(H)], rank=calibrator["rank_by_horizon"][str(H)],
                  warnings=calibrator["warnings"], calibrator_hash=calibrator["calibrator_hash"])
    point_known = np.array([c is not None for c in calibrator["c_point"]])
    point_c = np.array([c if c is not None else 0.0 for c in calibrator["c_point"]])
    amplitude = np.maximum(high - low, calibrator["scale_floor"])
    result.update(pointwise_lower=np.maximum(low - point_c * amplitude, 0),
                  pointwise_upper=high + point_c * amplitude, pointwise_calibrated_mask=point_known)
    if correction is not None:
        result.update(trajectory_lower=np.maximum(low[..., :H] - correction * amplitude[..., :H], 0),
                      trajectory_upper=high[..., :H] + correction * amplitude[..., :H],
                      calibrated=True, nominal=calibrator["nominal"])
    return result


def save_calibrator(calibrator, directory):
    validate_calibrator(calibrator)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "calibration.json"
    if path.exists():
        raise FileExistsError("Frozen calibration artifacts are immutable")
    path.write_text(json.dumps(calibrator, indent=2, allow_nan=False))
    return path


def load_calibrator(directory, *, model=None):
    path = Path(directory)
    if path.is_dir():
        path /= "calibration.json"
    return validate_calibrator(json.loads(path.read_text()), model)


def band_availability(band, config):
    """Diagnose full trajectory width without changing any numerical bound."""
    low = np.asarray(band["trajectory_lower"], dtype=float)
    high = np.asarray(band["trajectory_upper"], dtype=float)
    decision = band.get("output_kind") == "decision_corridor"
    median = np.asarray(band["center"] if decision else band["q50"], dtype=float)[..., :low.shape[-1]]
    relative = (high - low) / np.maximum(np.abs(median), config["scale_floor"])
    maximum = relative.max(axis=-1)
    available = np.isfinite(low).all(axis=-1) & np.isfinite(high).all(axis=-1)
    wide = maximum > config["width_budget"] + (1e-12 if decision else 0)
    accepted = available & ~wide & bool(band.get("calibrated", False))
    status = np.where(~available, "unavailable_forecast", np.where(wide, "wide_interval", band["status"]))
    if np.size(maximum) == 1:
        return {"available": bool(np.asarray(available).reshape(-1)[0]),
                "wide_interval": bool(np.asarray(wide).reshape(-1)[0]),
                "accepted": bool(np.asarray(accepted).reshape(-1)[0]),
                "max_relative_full_width": float(np.asarray(maximum).reshape(-1)[0]),
                "status": str(np.asarray(status).reshape(-1)[0])}
    return {"available": available, "wide_interval": wide, "accepted": accepted,
            "max_relative_full_width": maximum, "status": status}
