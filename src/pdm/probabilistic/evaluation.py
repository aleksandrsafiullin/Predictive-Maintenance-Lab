"""Physical metrics with distinct reference and rolling statistical populations."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .calibration import apply_calibrator, band_availability, validate_calibrator
from .contract import (
    is_bounded_trend,
    is_dense_v2,
    threshold_crossed,
    unit_balanced_weights,
    validate_threshold_direction,
)
from .models import QUANTILES, baseline_predictions, json_hash, predict_details
from .windows import _supplied_barriers


def wilson_interval(successes, n):
    if n == 0:
        return None
    z = 1.959963984540054
    fraction = successes / n
    denominator = 1 + z * z / n
    center = (fraction + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(fraction * (1 - fraction) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def unit_bootstrap_interval(values, units, *, seed=42, repetitions=1000):
    """Resample complete physical units, not overlapping forecasts or target rows."""
    values, units = np.asarray(values, dtype=float), np.asarray(units).astype(str)
    group = [values[units == u].mean() for u in np.unique(units)]
    if not group:
        return None
    rng = np.random.default_rng(seed)
    samples = rng.choice(group, size=(repetitions, len(group)), replace=True).mean(axis=1)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def _mean(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    keep = weights > 0
    if not keep.any():
        return None
    return float(np.sum(values[keep] * weights[keep]) / weights[keep].sum())


def _distribution(values, weights, prefix):
    keep = np.asarray(weights) > 0
    if not keep.any():
        return {f"{prefix}_{stat}": None for stat in ("mean", "median", "p90")}
    v, w = np.asarray(values)[keep], np.asarray(weights)[keep]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumulative = np.cumsum(w) / w.sum()
    return {f"{prefix}_mean": float(np.sum(v * w) / w.sum()),
            f"{prefix}_median": float(v[min(np.searchsorted(cumulative, .5), len(v) - 1)]),
            f"{prefix}_p90": float(v[min(np.searchsorted(cumulative, .9), len(v) - 1)])}


def _interval_score(y, low, high, mask, alpha=.1):
    # Unsupported values never enter arithmetic, even if they contain NaN/Inf.
    safe_y = np.where(mask, y, 0)
    return high - low + (2 / alpha) * (np.maximum(low - safe_y, 0) + np.maximum(safe_y - high, 0))


def _skill(model_mae, baseline_mae):
    if model_mae is None or baseline_mae is None:
        return None, "no_supported_targets"
    if baseline_mae == 0:
        return None, "zero_baseline_mae"
    return 1 - model_mae / baseline_mae, None


def _point_metrics(y, q, low, high, mask, units, floor, persistence, trend, calibrated):
    weights = unit_balanced_weights(mask, units)
    all_weights = unit_balanced_weights(np.ones(mask.shape, dtype=bool), units)
    safe_y = np.where(mask, y, 0)
    residual = safe_y[..., None] - q
    pinball = np.maximum(QUANTILES * residual, (QUANTILES - 1) * residual)
    covered = (safe_y >= low) & (safe_y <= high)
    mae = _mean(np.abs(safe_y - q[..., 1]), weights)
    pmae = _mean(np.abs(safe_y - persistence), weights)
    tmae = _mean(np.abs(safe_y - trend), weights)
    pskill, preason = _skill(mae, pmae)
    tskill, treason = _skill(mae, tmae)
    width = high - low
    relative = width / np.maximum(np.abs(q[..., 1]), floor)
    result = {"mae": mae, "mae_persistence": pmae, "mae_local_trend": tmae,
              "skill_persistence": pskill, "skill_persistence_reason": preason,
              "skill_local_trend": tskill, "skill_local_trend_reason": treason,
              "pinball_q05": _mean(pinball[..., 0], weights), "pinball_q50": _mean(pinball[..., 1], weights),
              "pinball_q95": _mean(pinball[..., 2], weights), "pinball": _mean(pinball.mean(-1), weights),
              "point_coverage": _mean(covered, weights) if calibrated else None,
              "diagnostic_raw_or_band_coverage": _mean(covered, weights),
              "interval_score": _mean(_interval_score(y, low, high, mask), weights),
              "raw_interval_score": _mean(_interval_score(y, q[..., 0], q[..., 2], mask), weights),
              "known_targets": int(mask.sum()), "supported_origins": int(mask.any(axis=1).sum()),
              "supported_units": int(len(np.unique(units[mask.any(axis=1)]))),
              "no_support_reason": "no_supported_targets" if not mask.any() else None}
    result.update(_distribution(width, all_weights, "width_g"))
    result.update(_distribution(relative, all_weights, "relative_full_width"))
    return result


def _unit_mean(values, units):
    values = np.asarray(values, dtype=float)
    return float(np.mean([values[units == u].mean() for u in np.unique(units)])) if len(values) else None


def _past_status_at(windows, i):
    statuses = windows.get("past_status")
    if isinstance(statuses, list) and len(statuses) == len(windows["x"]):
        return statuses[i]
    return {"first_event_status": "first_event_unknown", "first_event_unknown": True}


def red_window(forecast, past_status, *, red=.8, cadence_s=60, direction="above"):
    """Discrete first-single-RED geometry conditional on full trajectory coverage."""
    validate_threshold_direction(direction)
    if not np.isfinite(red) or not np.isfinite(cadence_s) or cadence_s <= 0:
        raise ValueError("Finite RED threshold and positive cadence required")
    past = past_status if isinstance(past_status, dict) else {"first_event_status": past_status}
    result = {"status": None, "earliest_s": None, "latest_s": None, "earliest_index": None,
              "latest_index": None, "cadence_s": cadence_s, "threshold": red,
              "interpretation": "conditional_discrete_band_geometry;not_failure_time_probability"}
    if direction == "below":
        result["threshold_direction"] = direction
    if past.get("already_red") or past.get("first_event_status") == "already_observed":
        result["status"] = "already_observed"
        result["first_red_s"] = past.get("first_red_s")
        return result
    if past.get("first_event_unknown") or past.get("first_event_status") not in ("not_observed", "never_observed"):
        result["status"] = "first_event_unknown"
        return result
    low = np.asarray(forecast["trajectory_lower"], dtype=float).reshape(-1)
    high = np.asarray(forecast["trajectory_upper"], dtype=float).reshape(-1)
    if not len(low) or low.shape != high.shape or not np.isfinite(low).all() or not np.isfinite(high).all():
        result["status"] = "unavailable_forecast"
        return result
    possible = np.flatnonzero(threshold_crossed(low if direction == "below" else high, red, direction))
    guaranteed = np.flatnonzero(threshold_crossed(high if direction == "below" else low, red, direction))
    if not len(possible):
        result["status"] = "no_crossing_in_horizon"
        return result
    earliest = int(possible[0] + 1)
    result.update(earliest_index=earliest, earliest_s=earliest * cadence_s)
    if not len(guaranteed):
        result["status"] = "open_right"
        return result
    latest = int(guaranteed[0] + 1)
    result.update(status="bounded", latest_index=latest, latest_s=latest * cadence_s)
    return result


def raw_validation_metrics(frozen_model, windows):
    """Same raw unit-balanced metrics without prefix loops or forecast tables."""
    if windows.get("split") != "validation":
        raise ValueError("Raw Validation metrics require the explicit Validation split")
    return evaluate(frozen_model, None, windows, protocol="rolling", raw_metrics_only=True)["raw_metrics"]


def evaluate(frozen_model, calibrator, windows, *, protocol="reference", metadata=None, observed_stream=None,
             raw_metrics_only=False):
    if protocol == "external":
        protocol = "external_reference"
    if protocol not in ("reference", "rolling", "external_reference", "external_rolling"):
        raise ValueError("Unknown evaluation protocol")
    reference = protocol.endswith("reference")
    cfg = frozen_model["config"]
    dense_v2 = is_dense_v2(cfg)
    direction = cfg.get("threshold_direction", "above")
    validate_threshold_direction(direction)
    if calibrator is not None:
        validate_calibrator(calibrator, frozen_model)
    for key in ("target", "unit", "cadence_s", "history_length", "max_horizon", "positive_domain", "schema"):
        if key in windows.get("config", {}) and windows["config"][key] != cfg[key]:
            raise ValueError(f"Evaluator contract mismatch: {key}")
    window_config = windows.get("config", {})
    if window_config.get("horizon_protocol") != cfg.get("horizon_protocol"):
        raise ValueError("Evaluator contract mismatch: horizon_protocol")
    if window_config.get("threshold_direction", "above") != direction:
        raise ValueError("Evaluator contract mismatch: threshold_direction")
    for key in ("red", "yellow", "zone_rule_hash"):
        if (key in window_config or key == "zone_rule_hash" and key in cfg) and window_config.get(key) != cfg.get(key):
            raise ValueError(f"Evaluator contract mismatch: {key}")
    source = frozen_model["training_summary"].get("source_binding", {})
    if not protocol.startswith("external"):
        for key, value in source.items():
            if windows.get(key) != value:
                raise ValueError(f"Evaluation source provenance mismatch; use explicit external protocol: {key}")
    elif source.get("profile") is not None and windows.get("profile") != source["profile"]:
        raise ValueError("External single-channel physical profile mismatch")
    x, y = np.asarray(windows["x"], dtype=float), np.asarray(windows["y"], dtype=float)
    mask = np.asarray(windows["mask"], dtype=bool)
    units = np.asarray(windows["physical_units"]).astype(str)
    unit_ids = np.asarray(windows["units"]).astype(str)
    origins = np.asarray(windows["origin_s"], dtype=float)
    if y.shape != (len(x), cfg["max_horizon"]) or mask.shape != y.shape or len(units) != len(x):
        raise ValueError("Evaluation grid/unit mismatch")
    if not np.isfinite(y[mask]).all() or (y[mask] < 0).any() or dense_v2 and (y[mask] <= 0).any():
        raise ValueError("Invalid supported evaluation target")
    if reference and len(set(units)) != len(units):
        raise ValueError("Reference Wilson population requires one origin per independent unit")
    if is_bounded_trend(cfg):
        if window_config != cfg:
            raise ValueError("Decision evaluator requires the frozen full configuration")
        if (np.diff(mask.astype(int), axis=1) > 0).any():
            raise ValueError("Decision evaluation target masks must be continuous prefixes")
        return _evaluate_decision(frozen_model, calibrator, windows, protocol, raw_metrics_only)
    # This is the ONLY predictor input. Metadata and future arrays are evaluator-only.
    details = predict_details(frozen_model, x)
    q = details["quantiles"]
    persistence = baseline_predictions(x, "persistence", cfg["max_horizon"])[..., 1]
    trend = baseline_predictions(x, "local_trend", cfg["max_horizon"])[..., 1]
    floor = cfg["scale_floor"]
    if raw_metrics_only:
        raw = _point_metrics(y, q, q[..., 0], q[..., 2], mask, units, floor, persistence, trend, False)
        return {"raw_metrics": raw}
    metric_rows, lead_rows, prediction_frames, trajectory_rows = [], [], [], []
    ids = [json_hash([frozen_model["model_hash"], calibrator["calibrator_hash"] if calibrator else None,
                      unit, float(origin), history.tolist()])[:32] for unit, origin, history in zip(units, origins, x)]
    calibration_status = "uncalibrated"
    for h in cfg["report_horizons"]:
        band = apply_calibrator(q, calibrator, h, model_hash=frozen_model["model_hash"], config=cfg)
        calibration_status = band["status"]
        if protocol.startswith("external"):
            band["scope"] = "external_distribution:no_nominal_transfer_assumed:" + band["scope"]
        low, high = band["trajectory_lower"], band["trajectory_upper"]
        complete = mask[:, :h].all(axis=1)
        partial = mask[:, :h].any(axis=1) & ~complete
        safe_y = np.where(mask[:, :h], y[:, :h], 0)
        inside = (safe_y >= low) & (safe_y <= high)
        path_covered = inside.all(axis=1)
        availability = band_availability(band, cfg)
        max_relative_width = np.asarray(availability["max_relative_full_width"]).reshape(len(x))
        wide = np.asarray(availability["wide_interval"], dtype=bool).reshape(len(x))
        accepted = np.asarray(availability["accepted"], dtype=bool).reshape(len(x))
        metrics = _point_metrics(y[:, :h], q[:, :h], low, high, mask[:, :h], units, floor,
                                 persistence[:, :h], trend[:, :h], band["calibrated"])
        if band["calibrated"] and complete.any():
            coverage = _unit_mean(path_covered[complete], units[complete])
            successes = int(path_covered[complete].sum())
            path_n = int(complete.sum())
            n = int(len(np.unique(units[complete])))
            path_successes = successes
            if not reference:
                successes = None
            uncertainty = wilson_interval(successes, n) if reference else unit_bootstrap_interval(path_covered[complete], units[complete], seed=cfg["seed"])
        else:
            coverage, successes, n, uncertainty = None, None, int(len(np.unique(units[complete]))), None
            path_n, path_successes = int(complete.sum()), None
        metrics.update(H=h, protocol=protocol, status=band["status"], calibrated=band["calibrated"],
                       scope=band["scope"], whole_path_coverage=coverage, successes=successes, n=n,
                       n_definition="independent_physical_units", path_origins=path_n, path_successes=path_successes,
                       ci95_lower=uncertainty[0] if uncertainty else None, ci95_upper=uncertainty[1] if uncertainty else None,
                       uncertainty_method="wilson_independent_units" if reference else "unit_block_bootstrap",
                       uncertainty_units=int(len(np.unique(units[complete]))),
                       whole_path_reason="not_evaluable" if not complete.any() else (None if band["calibrated"] else band["status"]),
                       complete_paths=int(complete.sum()), partial_paths=int(partial.sum()),
                       no_future_paths=int((~mask[:, :h].any(axis=1)).sum()), origins=len(x), units=len(set(units)),
                       wide_fraction=_unit_mean(wide, units), accepted_fraction=_unit_mean(accepted, units),
                       max_relative_full_width_mean=_unit_mean(max_relative_width, units),
                       mae_all=metrics["mae"], mae_accepted=None)
        accepted_mask = mask[:, :h] & accepted[:, None]
        if accepted_mask.any():
            metrics["mae_accepted"] = _mean(np.abs(safe_y - q[:, :h, 1]), unit_balanced_weights(accepted_mask, units))
        # Coverage inside complete paths uses only the stated complete population.
        full_mask = np.repeat(complete[:, None], h, axis=1)
        metrics["point_coverage_within_complete_paths"] = _mean(inside, unit_balanced_weights(full_mask, units)) if band["calibrated"] else None
        metric_rows.append(metrics)
        for i in range(len(x)):
            event = red_window({"trajectory_lower": low[i], "trajectory_upper": high[i]}, _past_status_at(windows, i), red=cfg["red"], cadence_s=cfg["cadence_s"], direction=direction)
            if dense_v2 and not band["calibrated"]:
                event.update(status="unavailable_calibration", earliest_s=None, latest_s=None,
                             earliest_index=None, latest_index=None)
            trajectory_rows.append({"forecast_id": ids[i], "physical_unit_id": units[i], "unit_id": unit_ids[i],
                                    "origin_s": origins[i], "H": h, "complete": bool(complete[i]),
                                    "evaluation_status": "evaluable" if complete[i] else "not_evaluable",
                                    "path_covered": bool(path_covered[i]) if complete[i] and band["calibrated"] else None,
                                    "wide_interval": bool(wide[i]), "accepted": bool(accepted[i]),
                                    "max_relative_full_width": float(max_relative_width[i]), "red_status": event["status"],
                                    "red_earliest_s": event["earliest_s"], "red_latest_s": event["latest_s"]})
        if len(x):
            origin_index = np.repeat(np.arange(len(x)), h)
            lead_index = np.tile(np.arange(h), len(x))
            plow, phigh = band["pointwise_lower"][:, :h], band["pointwise_upper"][:, :h]
            actual = np.where(mask[:, :h], y[:, :h], np.nan)
            current = x[:, -1]
            states = np.where(threshold_crossed(current, cfg["red"], direction), "Red",
                              np.where(threshold_crossed(current, cfg["yellow"], direction), "Yellow", "Green"))
            prediction_frames.append(pd.DataFrame({
                "forecast_id": np.asarray(ids)[origin_index], "model_hash": frozen_model["model_hash"],
                "calibrator_hash": band["calibrator_hash"], "dataset_hash": windows.get("dataset_hash"),
                "release_id": windows.get("release_id"), "unit_id": unit_ids[origin_index],
                "physical_unit_id": units[origin_index], "origin_s": origins[origin_index],
                "lead": lead_index + 1, "lead_s": (lead_index + 1) * cfg["cadence_s"],
                "timestamp_s": origins[origin_index] + (lead_index + 1) * cfg["cadence_s"],
                "H": h, "status": band["status"], "scope": band["scope"], "nominal": band["nominal"],
                "q05": q[:, :h, 0].ravel(), "q50": q[:, :h, 1].ravel(), "q95": q[:, :h, 2].ravel(),
                "pointwise_lower": plow.ravel(), "pointwise_upper": phigh.ravel(),
                "trajectory_lower": low.ravel(), "trajectory_upper": high.ravel(),
                "raw_crossing": details["raw_crossing"][:, :h].ravel(),
                "raw_q05_before_sort": details["raw_quantiles_before_sort"][:, :h, 0].ravel(),
                "raw_q50_before_sort": details["raw_quantiles_before_sort"][:, :h, 1].ravel(),
                "raw_q95_before_sort": details["raw_quantiles_before_sort"][:, :h, 2].ravel(),
                "actual": actual.ravel(), "mask": mask[:, :h].ravel(),
                "wide_interval": wide[origin_index], "accepted": accepted[origin_index],
                "observed_state": states[origin_index], "protocol": protocol,
                "first_event_status": np.asarray([_past_status_at(windows, i).get("first_event_status", "first_event_unknown") for i in range(len(x))])[origin_index]}))
    point_band = apply_calibrator(q, calibrator, cfg["max_horizon"], model_hash=frozen_model["model_hash"], config=cfg)
    for lead in range(cfg["max_horizon"]):
        sl = slice(lead, lead + 1)
        row = _point_metrics(y[:, sl], q[:, sl], point_band["pointwise_lower"][:, sl], point_band["pointwise_upper"][:, sl],
                             mask[:, sl], units, floor, persistence[:, sl], trend[:, sl],
                             bool(point_band["pointwise_calibrated_mask"][lead]) if dense_v2 else point_band["calibrated"])
        row.update(lead=lead + 1, lead_s=(lead + 1) * cfg["cadence_s"], protocol=protocol,
                   status=calibrator["point_status_by_lead"][lead] if dense_v2 and calibrator else point_band["status"])
        lead_rows.append(row)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    # Join evaluator-only mechanism labels AFTER predictions. They never enter fit/predict.
    if callable(metadata):
        metadata = metadata()
    if isinstance(metadata, (str, bytes)):
        metadata = pd.read_csv(metadata)
    if metadata is not None and len(predictions):
        meta = metadata.copy() if isinstance(metadata, pd.DataFrame) else pd.DataFrame(metadata)
        key = "physical_unit_id" if "physical_unit_id" in meta else "unit_id"
        if key not in meta or meta[key].duplicated().any():
            raise ValueError("Evaluator metadata must have one row per declared unit")
        columns = [key] + [c for c in ("mechanism", "scenario", "before_event") if c in meta]
        predictions = predictions.merge(meta[columns], on=key, how="left", validate="many_to_one")
    slices = []
    if len(predictions):
        for dimension in ("observed_state", "unit_id", "mechanism", "scenario", "before_event"):
            if dimension not in predictions:
                continue
            for (h, value), subset in predictions.groupby(["H", dimension], dropna=False):
                slice_calibrated = (calibrator is not None and calibrator["c_path"].get(str(int(h))) is not None) if dense_v2 else point_band["calibrated"]
                known = subset[subset["mask"]]
                if len(known):
                    errors = np.abs(known["actual"] - known["q50"])
                    covered = (known["actual"] >= known["trajectory_lower"]) & (known["actual"] <= known["trajectory_upper"])
                    by_lead = pd.DataFrame({"unit": known["physical_unit_id"], "lead": known["lead"], "error": errors, "covered": covered}).groupby(["unit", "lead"]).mean()
                    by_unit = by_lead.groupby("unit").mean()
                    mae, coverage = float(by_unit["error"].mean()), float(by_unit["covered"].mean())
                else:
                    mae, coverage = None, None
                within = (subset["actual"] >= subset["trajectory_lower"]) & (subset["actual"] <= subset["trajectory_upper"])
                cases = subset.assign(within=within).groupby("forecast_id", sort=False).agg(
                    unit=("physical_unit_id", "first"), supported=("mask", "sum"), row_count=("mask", "size"),
                    success=("within", "all"), wide=("wide_interval", "first"))
                cases["complete"] = (cases["supported"] == h) & (cases["row_count"] == h)
                complete_cases = cases[cases["complete"]]
                slice_units = subset["physical_unit_id"].nunique()
                # Widths describe all issued forecasts, including unsupported/wide cases.
                group_sizes = subset.groupby(["physical_unit_id", "lead"])["forecast_id"].transform("count")
                lead_counts = subset.groupby("physical_unit_id")["lead"].transform("nunique")
                slice_weights = (1 / (slice_units * lead_counts * group_sizes)).to_numpy()
                widths = (subset["trajectory_upper"] - subset["trajectory_lower"]).to_numpy()
                relative = widths / np.maximum(np.abs(subset["q50"].to_numpy()), floor)
                path_coverage = _unit_mean(complete_cases["success"].to_numpy(), complete_cases["unit"].to_numpy()) if len(complete_cases) and slice_calibrated else None
                ci = None
                if path_coverage is not None:
                    ci = wilson_interval(int(complete_cases["success"].sum()), len(complete_cases)) if reference else unit_bootstrap_interval(complete_cases["success"], complete_cases["unit"], seed=cfg["seed"])
                slice_row = {"H": int(h), "dimension": dimension, "value": str(value), "mae": mae,
                             "point_coverage": coverage if slice_calibrated else None,
                             "whole_path_coverage": path_coverage,
                             "n": complete_cases["unit"].nunique(), "path_origins": len(complete_cases),
                             "successes": int(complete_cases["success"].sum()) if reference and path_coverage is not None else None,
                             "ci95_lower": ci[0] if ci else None, "ci95_upper": ci[1] if ci else None,
                             "uncertainty_method": "wilson_independent_units" if reference else "unit_block_bootstrap",
                             "known_targets": len(known), "origins": subset["forecast_id"].nunique(),
                             "units": slice_units,
                             "wide_fraction": _unit_mean(cases["wide"].to_numpy(), cases["unit"].to_numpy())}
                slice_row.update(_distribution(widths, slice_weights, "width_g"))
                slice_row.update(_distribution(relative, slice_weights, "relative_full_width"))
                slices.append(slice_row)
    all_weights = unit_balanced_weights(np.ones(mask.shape, dtype=bool), units)
    raw_metrics = _point_metrics(y, q, q[..., 0], q[..., 2], mask, units, floor, persistence, trend, False)
    excluded = windows.get("admission", {}).get("excluded_units", [])
    admission = windows.get("admission", {})
    history_availability = admission.get("history_availability_by_unit", [])
    if reference:
        denominator = admission.get("total_units")
        numerator = admission.get("accepted_units", len(set(units)))
        availability = numerator / denominator if denominator else None
        availability_population = "all_source_physical_units:causal_reference_history_admission" if dense_v2 else "all_source_physical_units:complete_reference_path_admission"
    else:
        valid_counts = [r for r in history_availability if r.get("recorded_origins", 0) > 0]
        availability = float(np.mean([r["available_origins"] / r["recorded_origins"] for r in valid_counts])) if valid_counts else None
        numerator = sum(r["available_origins"] for r in valid_counts)
        denominator = sum(r["recorded_origins"] for r in valid_counts)
        availability_population = "unit_balanced_causal_history_availability_at_recorded_origins"
    refusal_counts = {}
    for item in excluded:
        reason = item.get("reason", "unknown")
        refusal_counts[reason] = refusal_counts.get(reason, 0) + 1
    assessment = {"red": cfg["red"], "yellow": cfg["yellow"], "cadence_s": cfg["cadence_s"]}
    if direction == "below":
        assessment["threshold_direction"] = direction
    if "zone_rule_hash" in cfg:
        assessment["zone_rule_hash"] = cfg["zone_rule_hash"]
    if dense_v2:
        assessment.update(horizon_protocol="dense-v2", max_horizon=cfg["max_horizon"])
    summary = {"protocol": protocol, "model_hash": frozen_model["model_hash"],
               "calibrator_hash": calibrator["calibrator_hash"] if calibrator else None,
               "calibration_status": calibration_status, "dataset_hash": windows.get("dataset_hash"),
               "origins": len(x), "units": len(set(units)), "known_targets": int(mask.sum()),
               "complete_60_paths": int(mask.all(axis=1).sum()), "excluded_units": excluded,
               "raw_crossing_rate": _mean(details["raw_crossing"], all_weights),
               "raw_uncalibrated_pinball": raw_metrics["pinball"],
               "forecast_available_fraction": availability,
               "availability_population": availability_population,
               "availability_numerator": numerator, "availability_denominator": denominator,
               "availability_reason": None if availability is not None else "missing_admission_denominator",
               "refusal_reasons": refusal_counts,
               "rolling_scope_note": "reference_calibration_does_not_guarantee_all_sequential_forecasts" if not reference else None,
               "external_scope_note": "outside_checked_distribution;nominal_transfer_not_assumed" if protocol.startswith("external") else None,
               "threshold_assessment_hash": json_hash(assessment),
               "alert_metrics": {"status": "not_estimable", "reason": "requires_explicit_episode_policy_and_observed_operating_stream"}}
    if dense_v2:
        from .horizons import target_support
        summary.pop("complete_60_paths")
        summary.update(horizon_protocol="dense-v2", max_horizon=cfg["max_horizon"],
                       complete_max_horizon_paths=int(mask.all(axis=1).sum()),
                       calibration_status_by_horizon={str(row["H"]): row["status"] for row in metric_rows},
                       support_by_lead=target_support(mask, units))
    if callable(observed_stream):
        observed_stream = observed_stream()
    if observed_stream is not None:
        summary["alert_metrics"] = evaluate_alert_episodes(predictions, observed_stream, cfg)
    horizons = pd.DataFrame(metric_rows)
    return {"summary": summary, "predictions": predictions, "metrics_by_horizon": horizons,
            "metrics_by_lead": pd.DataFrame(lead_rows), "trajectory_table": pd.DataFrame(trajectory_rows),
            "rolling_table": horizons.copy() if not reference else pd.DataFrame(),
            "raw_metrics": raw_metrics, "slices": pd.DataFrame(slices), "admission": windows.get("admission", {})}


def _decision_metrics(y, outputs, mask, units, x, cfg, persistence, trend):
    """Array-only physical-unit metrics, with unknown cells excluded before arithmetic."""
    low, center, high = outputs[..., 0], outputs[..., 1], outputs[..., 2]
    safe = np.where(mask, y, 0)
    weights = unit_balanced_weights(mask, units)
    all_weights = unit_balanced_weights(np.ones(mask.shape, dtype=bool), units)
    inside = (safe >= low) & (safe <= high)
    mae = _mean(np.abs(safe - center), weights)
    pmae, tmae = _mean(np.abs(safe - persistence), weights), _mean(np.abs(safe - trend), weights)
    ps, pr = _skill(mae, pmae)
    ts, tr = _skill(mae, tmae)
    coverage = _mean(inside, weights)
    # Empirical raw RED observations form a separate supported population.
    # This measures containment, not the correctness of predicted event geometry.
    red_mask = mask & threshold_crossed(safe, cfg["red"], cfg.get("threshold_direction", "above"))
    red_weights = unit_balanced_weights(red_mask, units)
    miss = mask & ~inside
    index = np.arange(1, mask.shape[1] + 1)[None, :]
    resets = np.maximum.accumulate(np.where(~miss, index, 0), axis=1)
    lengths = np.where(miss, index - resets, 0)
    starts = miss & ~np.column_stack((np.zeros(len(mask), dtype=bool), miss[:, :-1]))
    longest = lengths.max(axis=1) if mask.shape[1] else np.zeros(len(mask))
    mean_run = np.divide(miss.sum(axis=1), starts.sum(axis=1), out=np.zeros(len(mask), dtype=float), where=starts.sum(axis=1) > 0)
    active = mask.any(axis=1)
    mean_duration = _unit_mean(mean_run[active], units[active])
    max_duration = int(longest[active].max()) if active.any() else None
    current = x[:, -1, None]
    result = {"mae": mae, "center_mae": mae, "center_relative_mae": _mean(np.abs(safe - center) / np.maximum(np.abs(safe), cfg["scale_floor"]), weights),
              "mae_persistence": pmae, "mae_local_trend": tmae,
              "skill_persistence": ps, "skill_persistence_reason": pr,
              "skill_local_trend": ts, "skill_local_trend_reason": tr,
              "point_coverage": coverage, "diagnostic_raw_or_band_coverage": coverage,
              "red_point_coverage": _mean(inside, red_weights),
              "red_supported_targets": int(red_mask.sum()),
              "red_supported_units": len(np.unique(units[red_mask.any(axis=1)])),
              "red_no_support_reason": None if red_mask.any() else "no_supported_raw_red_targets",
              "miss_rate": 1 - coverage if coverage is not None else None,
              "direction_accuracy": _mean(np.sign(safe - current) == np.sign(center - current), weights),
              "change_mae": mae, "direction_definition": "sign of future value minus last observed value;exact zero is flat",
              "mean_miss_run_leads": mean_duration, "max_miss_run_leads": max_duration,
              "mean_miss_run_s": mean_duration * cfg["cadence_s"] if mean_duration is not None else None,
              "max_miss_run_s": max_duration * cfg["cadence_s"] if max_duration is not None else None,
              "known_targets": int(mask.sum()), "supported_origins": int(active.sum()),
              "supported_units": len(np.unique(units[active])),
              "no_support_reason": None if mask.any() else "no_supported_targets",
              "output_kind": "decision_corridor", "nominal": None, "coverage_guarantee": False}
    width = high - low
    relative = width / np.maximum(np.abs(center), cfg["scale_floor"])
    result.update(_distribution(width, all_weights, "width_g"))
    result.update(_distribution(relative, all_weights, "relative_full_width"))
    result["full_relative_width_max"] = float(relative.max()) if relative.size else None
    return result


def _evaluate_decision(model, assessment, windows, protocol, raw_only):
    from .horizons import target_support
    cfg = model["config"]
    x, y, mask = np.asarray(windows["x"]), np.asarray(windows["y"]), np.asarray(windows["mask"], dtype=bool)
    units = np.asarray(windows["physical_units"]).astype(str)
    uids, origins = np.asarray(windows["units"]), np.asarray(windows["origin_s"])
    outputs = predict_details(model, x)["outputs"]
    persistence = baseline_predictions(x, "persistence", cfg["max_horizon"])[..., 1]
    trend = baseline_predictions(x, "local_trend", cfg["max_horizon"])[..., 1]
    raw = _decision_metrics(y, outputs, mask, units, x, cfg, persistence, trend)
    # Rolling Validation must return before report-H loops, tables, or assessment projections.
    if raw_only:
        return {"raw_metrics": raw}
    rows, leads, frames, trajectories = [], [], [], []
    ids = [json_hash([model["model_hash"], assessment["calibrator_hash"] if assessment else None,
                      str(uid), float(origin), history.tolist()])[:32] for uid, origin, history in zip(uids, origins, x)]
    for h in cfg["report_horizons"]:
        band = apply_calibrator(outputs, assessment, h, model_hash=model["model_hash"], config=cfg)
        scope = ("external_distribution:empirical_transfer_not_assumed:" if protocol.startswith("external") else "") + band["scope"]
        hm = mask[:, :h]
        complete, active = hm.all(axis=1), hm.any(axis=1)
        covered = ((y[:, :h] >= outputs[:, :h, 0]) & (y[:, :h] <= outputs[:, :h, 2])).all(axis=1)
        metrics = _decision_metrics(y[:, :h], outputs[:, :h], hm, units, x, cfg, persistence[:, :h], trend[:, :h])
        metrics.update(H=h, protocol=protocol, status=band["status"], scope=scope, calibrated=False,
                       n=len(np.unique(units[hm[:, -1]])), issued_units=len(set(units)),
                       whole_path_coverage=_unit_mean(covered[complete], units[complete]),
                       whole_path_reason="diagnostic_only_no_coverage_guarantee" if complete.any() else "no_complete_paths",
                       complete_paths=int(complete.sum()), partial_paths=int((active & ~complete).sum()),
                       no_future_paths=int((~active).sum()), origins=len(x), units=len(set(units)),
                       accepted_fraction=0., wide_fraction=0., mae_all=metrics["mae"], mae_accepted=None)
        rows.append(metrics)
        for i in range(len(x)):
            event = red_window({"trajectory_lower": outputs[i, :h, 0], "trajectory_upper": outputs[i, :h, 2]},
                               _past_status_at(windows, i), red=cfg["red"], cadence_s=cfg["cadence_s"], direction=cfg.get("threshold_direction", "above"))
            trajectories.append({"forecast_id": ids[i], "physical_unit_id": units[i], "unit_id": uids[i],
                                 "origin_s": origins[i], "H": h, "complete": bool(complete[i]),
                                 "evaluation_status": "evaluable" if complete[i] else "not_evaluable",
                                 "path_covered": bool(covered[i]) if complete[i] else None,
                                 "wide_interval": False, "accepted": False,
                                 "max_relative_full_width": float(np.max((outputs[i, :h, 2] - outputs[i, :h, 0]) / np.maximum(outputs[i, :h, 1], cfg["scale_floor"]))),
                                 "red_status": event["status"], "red_earliest_s": event["earliest_s"], "red_latest_s": event["latest_s"],
                                 "red_interpretation": "decision_bound_crossing_geometry;no_event_time_guarantee"})
        oi = np.repeat(np.arange(len(x)), h)
        li = np.tile(np.arange(h), len(x))
        frames.append(pd.DataFrame({"forecast_id": np.asarray(ids)[oi], "model_hash": model["model_hash"],
                                    "calibrator_hash": band["calibrator_hash"], "dataset_hash": windows.get("dataset_hash"),
                                    "unit_id": uids[oi], "physical_unit_id": units[oi], "origin_s": origins[oi],
                                    "lead": li + 1, "lead_s": (li + 1) * cfg["cadence_s"],
                                    "timestamp_s": origins[oi] + (li + 1) * cfg["cadence_s"], "H": h,
                                    "status": band["status"], "scope": scope, "output_kind": "decision_corridor",
                                    "nominal": None, "coverage_guarantee": False,
                                    "center": outputs[:, :h, 1].ravel(), "corridor_lower": outputs[:, :h, 0].ravel(),
                                    "corridor_upper": outputs[:, :h, 2].ravel(),
                                    "trajectory_lower": outputs[:, :h, 0].ravel(), "trajectory_upper": outputs[:, :h, 2].ravel(),
                                    "pointwise_lower": outputs[:, :h, 0].ravel(), "pointwise_upper": outputs[:, :h, 2].ravel(),
                                    "actual": np.where(hm, y[:, :h], np.nan).ravel(), "mask": hm.ravel(),
                                    "accepted": False, "wide_interval": False, "protocol": protocol}))
    for j in range(cfg["max_horizon"]):
        row = _decision_metrics(y[:, j:j+1], outputs[:, j:j+1], mask[:, j:j+1], units, x, cfg, persistence[:, j:j+1], trend[:, j:j+1])
        row.update(lead=j+1, lead_s=(j+1)*cfg["cadence_s"], protocol=protocol,
                   n=len(np.unique(units[mask[:, j]])), status="empirically_assessed_decision_corridor" if assessment else "unassessed_decision_corridor")
        leads.append(row)
    summary = {"protocol": protocol, "model_hash": model["model_hash"],
               "calibrator_hash": assessment["calibrator_hash"] if assessment else None,
               "calibration_status": "empirically_assessed_decision_corridor" if assessment else "unassessed_decision_corridor",
               "dataset_hash": windows.get("dataset_hash"), "origins": len(x), "units": len(set(units)),
               "known_targets": int(mask.sum()), "output_kind": "decision_corridor", "forecast_mode": cfg["forecast_mode"],
               "nominal": None, "coverage_guarantee": False, "quality_accepted": False,
               "horizon_protocol": "dense-v2", "max_horizon": cfg["max_horizon"], "support_by_lead": target_support(mask, units),
               "threshold_assessment_hash": json_hash({k: cfg.get(k) for k in ("red", "yellow", "threshold_direction", "zone_rule_hash", "forecast_mode", "center_protocol", "corridor_protocol")}),
               "alert_metrics": {"status": "not_estimable", "reason": "decision_corridor_has_no_event_time_coverage_guarantee"},
               "threshold_geometry_interpretation": "decision_bound_crossing_geometry;no_event_time_guarantee"}
    summary.update({key: raw[key] for key in ("red_point_coverage", "red_supported_targets", "red_supported_units", "red_no_support_reason")})
    horizons = pd.DataFrame(rows)
    return {"summary": summary, "raw_metrics": raw, "predictions": pd.concat(frames, ignore_index=True),
            "metrics_by_horizon": horizons, "metrics_by_lead": pd.DataFrame(leads),
            "trajectory_table": pd.DataFrame(trajectories), "rolling_table": horizons.copy() if protocol.endswith("rolling") else pd.DataFrame(),
            "slices": pd.DataFrame(), "admission": windows.get("admission", {})}


def evaluate_alert_episodes(predictions, observed_stream, config):
    """Frozen H=30 possible-crossing policy; actuals enter only post-prediction."""
    direction = config.get("threshold_direction", "above")
    validate_threshold_direction(direction)
    policy = {"H": 30, "upper_threshold": config["red"], "width_budget": config["width_budget"],
              "origin_stride": config["origin_stride"], "cadence_s": config["cadence_s"],
              "require_calibrated": True, "first_event": "single_observed_red", "episodes": "successive_issued_warnings"}
    if direction == "below":
        policy["lower_threshold"] = policy.pop("upper_threshold")
        policy["threshold_direction"] = direction
    if "zone_rule_hash" in config:
        policy["zone_rule_hash"] = config["zone_rule_hash"]
    if is_dense_v2(config):
        policy["horizon_protocol"] = "dense-v2"
    result = {"policy": policy, "policy_hash": json_hash(policy), "status": "not_estimable",
              "observed_operating_hours": 0.0, "warning_episodes": 0, "false_episodes": 0,
              "censored_episodes": 0, "false_episodes_per_observed_operating_hour": None,
              "observed_first_events": 0, "detected_events": 0, "event_recall": None,
              "lead_time_s_mean": None, "lead_times_s": [], "censored_first_events": 0,
              "no_event_right_censored_units": 0, "suppressed_wide_origins": 0,
              "suppressed_unknown_first_origins": 0}
    if is_dense_v2(config) and config["max_horizon"] < policy["H"]:
        result.update(reason="alert_policy_horizon_exceeds_saved_grid", observed_units=0,
                      units_no_forecast=0, true_episodes=0)
        return result
    stream = observed_stream.copy() if isinstance(observed_stream, pd.DataFrame) else pd.DataFrame(observed_stream)
    required = {"timestamp_s", config["target"]}
    if not required <= set(stream.columns):
        raise ValueError("Alert evaluator needs timestamp and physical observed signal")
    key = "physical_unit_id" if "physical_unit_id" in stream else "unit_id"
    if key not in stream:
        raise ValueError("Alert evaluator requires explicit physical unit identity")
    selected = predictions[predictions["H"] == policy["H"]] if "H" in predictions else pd.DataFrame()
    result.update(observed_units=0, units_no_forecast=0, true_episodes=0)
    if config.get("observed_profile") is True and key == "physical_unit_id" and "unit_id" in stream:
        ambiguous = []
        for physical, rows in stream.groupby(key, sort=True):
            if rows["unit_id"].nunique() < 2:
                continue
            times = pd.to_numeric(rows["timestamp_s"], errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
                ambiguous.append(str(physical))
                continue
            spans = sorted((float(pd.to_numeric(unit["timestamp_s"]).min()), float(pd.to_numeric(unit["timestamp_s"]).max()))
                           for _, unit in rows.groupby("unit_id", sort=True))
            if any(right[0] <= left[1] for left, right in zip(spans, spans[1:])):
                ambiguous.append(str(physical))
        if ambiguous:
            result.update(reason="missing_physical_cycle_chronology", ambiguous_physical_units=ambiguous,
                          observed_units=int(stream[key].nunique()))
            return result
    lead_times = []
    cadence, stride = config["cadence_s"], config["origin_stride"] * config["cadence_s"]
    for unit, rows in stream.groupby(key, sort=True):
        result["observed_units"] += 1
        unit_preds = selected[selected[key].astype(str) == str(unit)] if key in selected else pd.DataFrame()
        result["units_no_forecast"] += int(not len(unit_preds))
        # Preserve original row order: an invalid timestamp is a past-information barrier.
        times = pd.to_numeric(rows["timestamp_s"], errors="coerce").to_numpy(dtype=float)
        values = pd.to_numeric(rows[config["target"]], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(times) & np.isfinite(values) & (values >= 0)
        if direction == "below":
            valid &= values > 0
        if "valid" in rows:
            valid &= rows["valid"].to_numpy(dtype=bool)
        continuous = np.zeros(len(rows), dtype=bool)
        if len(rows) > 1:
            continuous[1:] = valid[1:] & valid[:-1] & np.isclose(np.diff(times), cadence, atol=config["clock_tolerance_s"], rtol=0)
            continuous[1:] &= ~_supplied_barriers(rows)[1:]
        result["observed_operating_hours"] += float(continuous.sum() * cadence / 3600)
        segments = np.cumsum(~continuous)
        red_indices = np.flatnonzero(valid & threshold_crossed(values, config["red"], direction))
        first_index = int(red_indices[0]) if len(red_indices) else None
        past_unknown = bool(first_index is not None and ((~valid[:first_index + 1]).any() or (first_index > 0 and (~continuous[1:first_index + 1]).any())))
        event_s = float(times[first_index]) if first_index is not None and not past_unknown else None
        if first_index is None:
            result["no_event_right_censored_units"] += 1
        elif past_unknown:
            result["censored_first_events"] += 1
        else:
            result["observed_first_events"] += 1
        cases = []
        for origin, forecast in (unit_preds.groupby("origin_s", sort=True) if len(unit_preds) else []):
            indices = np.flatnonzero(valid & (times <= float(origin) + config["clock_tolerance_s"]))
            if not len(indices):
                continue
            current_idx = indices[-1]
            prior_red = np.any(threshold_crossed(values[indices], config["red"], direction))
            unknown = bool((~valid[:current_idx + 1]).any() or (current_idx > 0 and (~continuous[1:current_idx + 1]).any()))
            width = ((forecast["trajectory_upper"] - forecast["trajectory_lower"]) / np.maximum(np.abs(forecast["q50"]), config["scale_floor"])).max()
            wide = bool(width > policy["width_budget"])
            result["suppressed_wide_origins"] += int(wide)
            result["suppressed_unknown_first_origins"] += int(unknown)
            possible = forecast["trajectory_lower" if direction == "below" else "trajectory_upper"]
            warning = bool(not prior_red and not unknown and not wide and forecast["status"].iloc[0] == "calibrated_reference_scope" and threshold_crossed(possible, config["red"], direction).any())
            cases.append({"origin": float(origin), "warning": warning, "segment": int(segments[current_idx])})
        episodes = []
        active = None
        previous = None
        for case in cases:
            consecutive = previous is not None and case["origin"] - previous["origin"] <= stride + config["clock_tolerance_s"] and case["segment"] == previous["segment"]
            if case["warning"]:
                if active is None or not consecutive:
                    active = {"start": case["origin"], "end": case["origin"], "segment": case["segment"]}
                    episodes.append(active)
                else:
                    active["end"] = case["origin"]
            else:
                active = None
            previous = case
        detected = False
        unit_leads = []
        for episode in episodes:
            result["warning_episodes"] += 1
            horizon_end = episode["end"] + policy["H"] * cadence
            seg_times = times[valid & (segments == episode["segment"])]
            seg_end = float(seg_times.max()) if len(seg_times) else episode["end"]
            if event_s is not None and episode["start"] < event_s <= horizon_end:
                detected = True
                unit_leads.append(event_s - episode["start"])
                result["true_episodes"] += 1
            elif seg_end < horizon_end - config["clock_tolerance_s"]:
                result["censored_episodes"] += 1
            else:
                result["false_episodes"] += 1
        if event_s is not None and detected:
            result["detected_events"] += 1
            lead_times.append(max(unit_leads))
    hours = result["observed_operating_hours"]
    result["false_episodes_per_observed_operating_hour"] = result["false_episodes"] / hours if hours > 0 else None
    result["lead_times_s"] = lead_times
    result["lead_time_s_mean"] = float(np.mean(lead_times)) if lead_times else None
    if result["observed_first_events"]:
        result["event_recall"] = result["detected_events"] / result["observed_first_events"]
        result["status"] = "estimated_observed_scope"
        result["reason"] = None
    else:
        result["reason"] = "no_uncensored_observed_first_events"
    return result
