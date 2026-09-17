"""The single monitoring path for seek, sequential replay, UI and exports."""
from __future__ import annotations

import numpy as np

from pdm.history import contiguous_history, resolved_length
from pdm.monitoring.contracts import SCHEMA, observed_prefix
from pdm.monitoring.normality import assess_normality
from pdm.monitoring.quality import assess_quality
from pdm.monitoring.signal_forecast import forecast_signal, median_crossing
from pdm.monitoring.state import initial_state, update_state


def forecast_event(prefix, predictor, profile, quality, *, with_trace=False):
    result = {"event_definition_id": profile["event_definition_id"], "time_basis": profile["time_basis"],
              "status": "unavailable", "point": None, "interval": None, "probabilities": None,
              "operator_probabilities": None, "timing_validated": False,
              "distribution": None,
              "reason": "event_model_not_configured", "diagnostics": None}
    if predictor is None:
        return result
    if quality["data_quality_status"] not in {"valid", "degraded"}:
        result["reason"] = "runtime_quality_refusal"
        return result
    if getattr(predictor.model, "state_mode", None) == "continuous" and getattr(predictor, "interval_profile", None):
        from pdm.replay import _continuous_bearing_predictions

        prediction = _continuous_bearing_predictions(prefix, predictor, predictor.interval_profile).iloc[-1].to_dict()
        prediction["interval_method"] = "empirical_residual_interval"
    else:
        prediction = predictor.predict_from_history(prefix, with_trace=with_trace)
    point = prediction.get("predicted_rul_s")
    result["diagnostics"] = {k: v for k, v in prediction.items() if k in {
        "predicted_rul_s", "raw_rul_s", "forecast_method", "interval_method", "status", "valid_history_reason", "state_mode"}}
    if point is None or not np.isfinite(point):
        result["reason"] = prediction.get("valid_history_reason") or "model_unavailable"
        return result
    result.update(status="limited", point=float(point), reason="forecast_not_validated_for_operational_timing")
    lo, hi = prediction.get("lower_rul_s"), prediction.get("upper_rul_s")
    if lo is not None and hi is not None:
        result["interval"] = {"lower": float(lo), "upper": float(hi),
            "method": prediction["interval_method"], "nominal_level": .9,
            "calibration_status": "unvalidated", "independent_units": 0, "independent_events": 0,
            "coverage_guarantee": False}
    if result["interval"] and getattr(predictor, "interval_profile", None):
        result["interval"].update(calibration_status="empirical_research",
            independent_units=len(predictor.interval_profile.get("calibration_ids", [])),
            independent_events=len(predictor.interval_profile.get("calibration_ids", [])))
    if "weibull_scale_s" in prediction:
        from pdm.monitoring.calibration import weibull_probabilities

        result["probabilities"] = weibull_probabilities(prediction["weibull_scale_s"], prediction["weibull_shape"],
                                                        [5 * profile["nominal_interval"], 15 * profile["nominal_interval"]])
        result["probability_status"] = "insufficient_independent_calibration_data"
        result["distribution"] = {"family": "weibull", "scale": float(prediction["weibull_scale_s"]),
                                  "shape": float(prediction["weibull_shape"]), "time_basis": profile["time_basis"]}
    return result


def monitoring_step(frame, *, profile, reference, quality_policy, state_policy,
                    bundle_id, as_of, previous=None, predictor=None, sensor_model=None):
    prefix = observed_prefix(frame, profile["dataset_id"], as_of)
    uid = str(prefix.iloc[-1].get("unit_id", "unknown")) if len(prefix) else (previous or {}).get("unit_id", "unknown")
    if previous and (previous["unit_id"] != uid or previous["bundle_id"] != bundle_id):
        raise ValueError("Monitoring state belongs to another unit or bundle")
    previous = previous or initial_state(uid, bundle_id)
    quality = assess_quality(prefix, profile, quality_policy, as_of)
    last = prefix.iloc[-1].to_dict() if len(prefix) else {}
    normality = assess_normality(last, reference)
    usable_count = quality["available_measurements"]
    usable_prefix = prefix.iloc[-usable_count:] if usable_count else prefix.iloc[:0]
    segment = contiguous_history(usable_prefix, profile["dataset_id"]) if len(usable_prefix) else usable_prefix
    segment_id = str(segment.timestamp_s.iloc[0]) if len(segment) else (previous.get("segment_id") or "unknown")
    event = forecast_event(usable_prefix, predictor, profile, quality)
    signals = forecast_signal(usable_prefix if usable_count else prefix, sensor_model, as_of, bundle_id=bundle_id) if len(prefix) else []
    if quality["data_quality_status"] not in {"valid", "degraded"} or normality["model_applicability"] != "in_domain":
        for row in signals:
            row.update(point=None, lower=None, upper=None, forecast_status="unavailable",
                       reason="runtime_quality_or_regime_refusal")
    if normality["model_applicability"] != "in_domain":
        event.update(status="unavailable", point=None, interval=None, probabilities=None, distribution=None, reason="unsupported_operating_regime")
    condition = {"quality": quality, "normality": normality, "measurement": last, "segment_id": segment_id}
    state = update_state(previous, condition, event, profile, state_policy, as_of)
    used = 0
    mode = "not_configured"
    if predictor:
        continuous = getattr(predictor.model, "state_mode", None) == "continuous"
        mode = "continuous_since_last_reset" if continuous else (predictor.prep.history_policy or {}).get("mode", f"fixed_{predictor.history_length}")
        required = resolved_length(segment, predictor.prep, predictor.history_length)
        used = len(segment) if continuous else (required if len(segment) >= required else 0)
    elif sensor_model:
        mode = f"sensor_aggregates_{sensor_model['config']['history_min']}_{sensor_model['config']['history_max']}"
        used = min(len(segment), sensor_model["config"]["history_max"]) if len(segment) >= sensor_model["config"]["history_min"] else 0
    duration = float(segment.timestamp_s.iloc[-1] - segment.timestamp_s.iloc[-used]) if used else None
    result = {"schema_version": SCHEMA, "bundle_id": bundle_id, "unit_id": uid, "segment_id": segment_id,
              "as_of": float(as_of), "last_measurement_at": quality["last_measurement_at"], "time_basis": profile["time_basis"],
              "data_quality_status": quality["data_quality_status"], "model_applicability": normality["model_applicability"],
              "forecast_status": "limited" if any(r["point"] is not None for r in signals) else "unavailable",
              "history": {"mode": mode, "available_measurements": quality["available_measurements"],
                          "used_measurements": used, "duration": duration, "feature_context_duration": quality["duration"]},
              "condition": {**state["last_result"], "reference_status": reference["status"]},
              "event_forecast": event, "signal_forecasts": signals, "normality": normality,
              "crossing": median_crossing(signals, state_policy.get("critical_limit")),
              "evaluation_status": "laboratory_research", "quality": quality}
    return result, state


def replay_monitoring(frame, *, as_of=None, should_stop=None, **kwargs):
    """Rebuild from the exact prefix; deterministic IDs make repeated seek idempotent."""
    profile = kwargs["profile"]
    if as_of is None:
        as_of = float(frame.timestamp_s.max())
    prefix = observed_prefix(frame, profile["dataset_id"], as_of)
    state, rows = None, []
    for i, row in enumerate(prefix.itertuples()):
        if should_stop and should_stop():
            raise InterruptedError("Stopped during causal monitoring replay")
        result, state = monitoring_step(prefix.iloc[:i + 1], as_of=float(row.timestamp_s), previous=state, **kwargs)
        rows.append(result)
    if state is not None and as_of > state["last_as_of"]:
        result, state = monitoring_step(prefix, as_of=as_of, previous=state, **kwargs)
        rows.append(result)
    return rows, state
