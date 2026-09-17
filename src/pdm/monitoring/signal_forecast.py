"""Separate, real multi-horizon sensor models. RUL is never a sensor input."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pdm.history import contiguous_history
from pdm.monitoring.contracts import observed_prefix
from pdm.monitoring.normality import assess_normality
from pdm.training_protocol import fingerprint
from pdm.windows import _unit_gap_flags

METHODS = ("persistence", "causal_local_trend", "multi_horizon_quantile_boosting")


def signal_features(segment, profile, *, include_age=False, reference=None, multiscale=True):
    """Only raw signal/current regime and causal aggregates; explicit availability."""
    name = profile["signal_name"]
    y = segment[name].to_numpy(float)
    t = segment.timestamp_s.to_numpy(float)
    last = segment.iloc[-1]
    result = {"current": float(y[-1])}
    for size in ((5, 20, 40, 60) if multiscale else (5, 20)):
        v, times = y[-size:], t[-size:]
        centered = times - times.mean()
        denom = float(centered @ centered)
        result.update({f"mean_{size}": float(v.mean()), f"std_{size}": float(v.std()),
                       f"slope_{size}": float(centered @ (v - v.mean()) / denom) if denom else 0.,
                       f"available_{size}": float(len(y) >= size), f"duration_{size}": float(times[-1] - times[0])})
    for key in (("rpm", "load_kn") if profile["dataset_id"] == "bearings" else ("flow_rate", "dust_feed")):
        result[key] = float(last[key])
    if include_age:
        result["operating_age"] = float(last["operating_age_s"])
    if reference is not None:
        norm = assess_normality(last, reference)
        result["regime_unknown"] = float(norm["model_applicability"] != "in_domain")
        result["normality_available"] = float(norm["score"] is not None)
        result["normality_score"] = float(norm["score"] or 0.)
    return result


def supervised_targets(features, profile, horizons, *, tolerance, history_min=20, history_max=60,
                       include_age=False, reference=None, multiscale=True):
    """Labels stay in the same observed segment; each horizon has its own mask."""
    rows, labels, metadata = [], [], []
    if tolerance < 0 or any(h <= 0 for h in horizons):
        raise ValueError("Invalid target timestamp policy")
    for uid, group in features.groupby("unit_id", sort=True):
        ordered, gaps = _unit_gap_flags(group, profile["dataset_id"])
        gaps = gaps.copy()
        for key in ("maintenance_reset", "segment_reset"):
            if key in ordered:
                gaps |= ordered[key].fillna(False).to_numpy(bool)
        if "segment_id" in ordered:
            gaps[1:] |= ordered.segment_id.to_numpy()[1:] != ordered.segment_id.to_numpy()[:-1]
        for sid, segment in ordered.groupby(np.cumsum(gaps), sort=False):
            segment = segment.reset_index(drop=True)
            ts = segment.timestamp_s.to_numpy(float)
            if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
                raise ValueError("Invalid target clock")
            for end in range(history_min - 1, len(segment)):
                prefix = segment.iloc[max(0, end - history_max + 1):end + 1]
                inputs = signal_features(prefix, profile, include_age=include_age, reference=reference, multiscale=multiscale)
                if not np.isfinite(list(inputs.values())).all():
                    continue
                target = []
                for h in horizons:
                    desired = ts[end] + h
                    candidates = np.flatnonzero((np.abs(ts - desired) <= tolerance + 1e-8) & (ts > ts[end]))
                    # Ambiguous timestamps are refused, not picked to improve error.
                    target.append(float(segment.iloc[candidates[0]][profile["signal_name"]]) if len(candidates) == 1 else np.nan)
                rows.append(inputs)
                labels.append(target)
                metadata.append({"unit_id": str(uid), "segment_id": str(sid), "issued_at": float(ts[end]),
                                 "available_measurements": end + 1})
    return pd.DataFrame(rows), np.asarray(labels, dtype=float).reshape(-1, len(horizons)), pd.DataFrame(metadata)


def fit_sensor(features, train_ids, profile, *, horizons, method, seed=42, tolerance=0.01,
               history_min=20, history_max=60, include_age=False, reference=None,
               multiscale=True, max_iter=60, should_stop=None, on_fit=None):
    if method not in METHODS:
        raise ValueError("Unknown sensor method")
    data = features[features.unit_id.isin(train_ids)]
    x, y, meta = supervised_targets(data, profile, horizons, tolerance=tolerance,
                    history_min=history_min, history_max=history_max, include_age=include_age,
                    reference=reference, multiscale=multiscale)
    if x.empty:
        raise ValueError("No admitted sensor training targets")
    models, support = {}, {}
    for j, h in enumerate(horizons):
        mask = np.isfinite(y[:, j])
        support[str(h)] = {"targets": int(mask.sum()), "independent_units": int(meta.loc[mask, "unit_id"].nunique())}
        if not mask.any():
            continue
        if method == "multi_horizon_quantile_boosting":
            # Every horizon/quantile fit has a visible ledger entry. No random
            # internal validation split and no holdout used for early stopping.
            counts = meta.loc[mask].groupby("unit_id").size()
            weights = meta.loc[mask, "unit_id"].map(1 / counts).to_numpy()
            models[h] = {}
            for quantile in (.05, .5, .95):
                if should_stop and should_stop():
                    raise InterruptedError("Stopped before sensor submodel fit")
                if on_fit:
                    on_fit(h, quantile)
                model = HistGradientBoostingRegressor(loss="quantile", quantile=quantile,
                            max_iter=max_iter, max_leaf_nodes=15, min_samples_leaf=20,
                            l2_regularization=1., early_stopping=False, random_state=seed)
                model.fit(x.loc[mask], y[mask, j], sample_weight=weights / weights.mean())
                models[h][quantile] = model
    config = {"schema_version": "sensor_model_v1", "method": method, "profile": profile,
              "horizons": list(horizons), "tolerance": tolerance, "history_min": history_min,
              "history_max": history_max, "include_age": include_age, "multiscale": multiscale,
              "reference_id": reference.get("reference_id") if reference else None,
              "feature_names": list(x.columns), "quantile_repair": "sort_then_nonnegative",
              "output_transform": "nonnegative_clip", "fit_units": sorted(str(u) for u in train_ids),
              "target_support": support, "seed": seed, "max_iter": max_iter,
              "calibration_status": "unvalidated_pointwise_quantiles", "coverage_guarantee": False,
              "scenario_assumption": "hold_current_observed_conditions"}
    config["sensor_model_run_id"] = fingerprint(config)[:20]
    return {"config": config, "models": models, "reference": reference}


def forecast_signal(frame, sensor_model, as_of, *, bundle_id="unbundled"):
    if sensor_model is None:
        return []
    cfg = sensor_model["config"]
    profile = cfg["profile"]
    prefix = observed_prefix(frame, profile["dataset_id"], as_of)
    if prefix.empty:
        return []
    if not np.isfinite(pd.to_numeric(prefix.timestamp_s, errors="coerce")).all():
        return []
    segment = contiguous_history(prefix, profile["dataset_id"])
    n = len(segment)
    segment_id = str(prefix.iloc[-1].get("segment_id", segment.timestamp_s.iloc[0]))
    segment = segment.iloc[-cfg["history_max"]:]
    reasons = "collecting_sensor_history" if n < cfg["history_min"] else ""
    try:
        features = signal_features(segment, profile, include_age=cfg["include_age"],
                        reference=sensor_model["reference"], multiscale=cfg["multiscale"])
        if not np.isfinite(list(features.values())).all():
            reasons = "invalid_sensor_input"
    except (ValueError, TypeError, KeyError):
        features = {}
        reasons = "invalid_sensor_input"
    x = pd.DataFrame([features]).reindex(columns=cfg["feature_names"])
    rows = []
    for h in cfg["horizons"]:
        reason = reasons
        point = lower = upper = None
        if cfg["target_support"][str(h)]["targets"] == 0:
            reason = "unsupported_target_horizon"
        if not reason:
            if cfg["method"] == "persistence":
                point = max(0., features["current"])
            elif cfg["method"] == "causal_local_trend":
                point = max(0., features["current"] + features["slope_20"] * h)
            else:
                q = [float(sensor_model["models"][h][p].predict(x)[0]) for p in (.05, .5, .95)]
                if np.isfinite(q).all():
                    lower, point, upper = np.maximum(0., np.sort(q)).tolist()
                else:
                    reason = "nonfinite_sensor_output"
        rows.append({"bundle_id": bundle_id, "sensor_model_run_id": cfg["sensor_model_run_id"],
            "unit_id": str(prefix.iloc[-1].get("unit_id", "")),
            "segment_id": segment_id,
            "issued_at": float(prefix.timestamp_s.iloc[-1]), "target_time": float(prefix.timestamp_s.iloc[-1] + h),
            "horizon": h, "time_basis": profile["time_basis"], "signal_name": profile["signal_name"],
            "signal_unit": profile["signal_unit"], "point": point, "lower": lower, "upper": upper,
            "interval_method": "pointwise_quantile_regression_5_95" if lower is not None else None,
            "calibration_status": cfg["calibration_status"], "forecast_status": "unavailable" if reason else "limited",
            "reason": reason, "scenario_assumption": cfg["scenario_assumption"]})
    return rows


def median_crossing(rows, limit):
    if not limit:
        return {"time": None, "reason": "no_applicable_signal_limit"}
    if limit["aggregation_rule"] != "instantaneous":
        return {"time": None, "reason": "discrete_horizons_do_not_establish_sustained_crossing"}
    for row in sorted(rows, key=lambda r: r["target_time"]):
        if row["point"] is None or row["signal_name"] != limit["signal_name"] or row["signal_unit"] != limit["unit"]:
            continue
        hit = row["point"] >= limit["value"] if limit["direction"] == "above" else row["point"] <= limit["value"]
        if hit:
            return {"time": row["target_time"], "reason": "Median forecast threshold crossing", "method": "first_discrete_forecast_point"}
    return {"time": None, "reason": "No crossing predicted within the displayed horizon"}
