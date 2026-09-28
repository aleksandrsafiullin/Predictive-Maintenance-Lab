"""Filter zones are pressure bands; reference and trend remain supplementary.

The visible class uses current measured differential pressure (<300, 300 to <600,
or >=600 Pa) under row-quality and flow/feed gates. Candidate-reference residuals
and comparable-context trends are supplementary evidence only. No event, RUL, or
future measurement defines a zone.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.io_util import atomic_write_json, atomic_write_text, sha256_file
from pdm.monitoring.normality import regime_key
from pdm.paths import runs_root

FILTER_ZONE_POLICY = {
    "version": "filter_sensor_zones_v2",
    "yellow_limit_pa": 300.0,
    "yellow_limit_policy_id": "filter_pressure_warning_300pa_v1",
    "yellow_reference_z": 3.0,
    "trend_points": 5,
    "minimum_trend_change_z": 3.0,
    "minimum_rising_steps": 3,
    "flow_bin_width": 50.0,
    "feed_bin_width": 100.0,
    "source": "provisional_HSE_sensor_pressure_bands_and_reference_adjusted_trend",
    "verification_status": "provisional_laboratory_rule",
    "yellow_limit_source": "provisional_pressure_warning_boundary_pending_expert_validation",
    "green_semantics": "below_provisional_pressure_band_only_not_healthy",
    "pressure_valid_range_pa": [0.0, 2500.0],
    "red_limit_pa": 600.0,
    "red_source": "Monitoring hard limit >=600 Pa; HSE observed event convention >600 Pa",
}


def filter_pressure_band(pressure, flow, feed, *, pressure_quality_valid: bool,
                         row_quality_valid: bool, policy: dict | None = None) -> dict:
    """Shared pressure-only class decision used by runtime and row export."""
    cfg = {**FILTER_ZONE_POLICY, **(policy or {})}
    try:
        p = float(pressure)
    except (TypeError, ValueError):
        p = float("nan")
    try:
        f = float(flow)
    except (TypeError, ValueError):
        f = float("nan")
    try:
        d = float(feed)
    except (TypeError, ValueError):
        d = float("nan")
    pmin, pmax = map(float, cfg["pressure_valid_range_pa"])
    if not pressure_quality_valid or not np.isfinite(p) or not pmin <= p <= pmax:
        return {"status": "unknown", "zone": "unknown", "display_zone": "gray",
                "label": "Assessment unavailable", "reason": "pressure_channel_unavailable_or_invalid"}
    if p >= float(cfg["red_limit_pa"]):
        return {"status": "observed", "zone": "configured_600_pa_limit", "display_zone": "red",
                "label": "Configured laboratory pressure limit reached",
                "reason": "measured_differential_pressure_at_or_above_600_pa"}
    if (not row_quality_valid or not np.isfinite(f) or f <= 0 or not np.isfinite(d)):
        return {"status": "unknown", "zone": "unknown", "display_zone": "gray",
                "label": "Assessment unavailable",
                "reason": "positive_flow_and_finite_feed_context_required"}
    if p >= float(cfg["yellow_limit_pa"]):
        return {"status": "provisional", "zone": "pressure_warning_band", "display_zone": "yellow",
                "label": "Pressure at or above provisional warning band",
                "reason": "measured_pressure_at_or_above_provisional_warning_band"}
    return {"status": "provisional", "zone": "below_provisional_pressure_band", "display_zone": "green",
            "label": "Below provisional pressure warning band",
            "reason": "measured_pressure_below_provisional_warning_band"}


def assess_filter_sensor_zone(frame: pd.DataFrame, *, quality: dict,
                              normality: dict | None = None,
                              policy: dict | None = None) -> dict:
    """Classify the current measured filter state using prefix rows only.

    A candidate-healthy reference may support a yellow pressure deviation. A
    rising trend may support yellow only when the recent measured flow and feed
    remain in the same saved engineering bins. Flow must be positive and finite
    for any non-red classification. The function never reads event/RUL fields.
    """
    cfg = {**FILTER_ZONE_POLICY, **(policy or {})}
    base = {"schema_version": cfg["version"], "status": "unknown",
            "zone": "unknown", "display_zone": "gray", "label": "Assessment unavailable",
            "pressure_pa": None, "flow_rate": None, "dust_feed": None,
            "pressure_trend": None, "reference": None,
            "policy": {k: cfg[k] for k in (
                "version", "yellow_limit_policy_id", "yellow_limit_pa", "yellow_limit_source",
                "green_semantics", "pressure_valid_range_pa", "yellow_reference_z", "trend_points", "flow_bin_width",
                "feed_bin_width", "minimum_trend_change_z", "minimum_rising_steps",
                "source", "verification_status",
                "red_limit_pa", "red_source")}}
    if frame is None or frame.empty:
        base["reason"] = "no_measurement"
        return base
    row = frame.iloc[-1]

    def number(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    pressure, flow, feed = (number(row.get(k)) for k in
                            ("differential_pressure", "flow_rate", "dust_feed"))
    base.update(pressure_pa=pressure if np.isfinite(pressure) else None,
                flow_rate=flow if np.isfinite(flow) else None,
                dust_feed=feed if np.isfinite(feed) else None)
    channel_valid = quality.get("channel_valid") or {}
    pressure_quality_valid = bool(quality.get("critical_channel_usable"))
    row_quality_valid = bool(quality.get("measurement_usable", True)
                             and channel_valid.get("flow_rate", True)
                             and channel_valid.get("dust_feed", True))
    band = filter_pressure_band(pressure, flow, feed,
                                pressure_quality_valid=pressure_quality_valid,
                                row_quality_valid=row_quality_valid, policy=cfg)
    base.update(band)
    if band["zone"] in {"unknown", "configured_600_pa_limit"}:
        return base

    # Reference and trend fields below are supplementary; they cannot recolor it.
    evidence = []
    residual = ((normality or {}).get("residuals") or {}).get("differential_pressure")
    reference_matches_context = ((normality or {}).get("regime_key") == regime_key(row, "filters"))
    if ((normality or {}).get("reference_status") == "provisional" and reference_matches_context and residual
            and np.isfinite([residual.get("expected", np.nan),
                             residual.get("scale", np.nan)]).all()):
        expected, scale = float(residual["expected"]), float(residual["scale"])
        pressure_limit = expected + float(cfg["yellow_reference_z"]) * scale
        base["reference"] = {"status": (normality or {}).get("reference_status", "unknown"),
                             "expected_pa": expected, "robust_scale_pa": scale,
                             "yellow_pressure_pa": pressure_limit,
                             "rule": "candidate_reference_plus_z_times_robust_scale"}
        if pressure > pressure_limit:
            evidence.append("measured_pressure_above_provisional_reference_band")

    n = int(cfg["trend_points"])
    trend_frame = frame.tail(n)
    if len(trend_frame) == n:
        vals = trend_frame[["timestamp_s", "differential_pressure", "flow_rate", "dust_feed"]].apply(
            pd.to_numeric, errors="coerce").to_numpy(float)
        if np.isfinite(vals).all() and np.all(np.diff(vals[:, 0]) > 0):
            flow_bins = np.floor(vals[:, 2] / float(cfg["flow_bin_width"]))
            feed_bins = np.floor(vals[:, 3] / float(cfg["feed_bin_width"]))
            stable_context = len(set(flow_bins)) == 1 and len(set(feed_bins)) == 1 and np.all(vals[:, 2] > 0)
            if stable_context:
                t = vals[:, 0] - vals[:, 0].mean()
                centered_p = vals[:, 1] - vals[:, 1].mean()
                slope = float(t @ centered_p / (t @ t)) if float(t @ t) > 0 else float("nan")
                if np.isfinite(slope):
                    ref = base["reference"]
                    observed_change = float(vals[-1, 1] - vals[0, 1])
                    required_change = (float(cfg["minimum_trend_change_z"]) * ref["robust_scale_pa"]
                                      if ref is not None else None)
                    rising_steps = int(np.sum(np.diff(vals[:, 1]) > 0))
                    significant = bool(ref is not None and observed_change > required_change
                                       and rising_steps >= int(cfg["minimum_rising_steps"]))
                    base["pressure_trend"] = {"direction": "rising" if slope > 0 else "stable_or_falling",
                                               "slope_pa_per_dataset_time": slope,
                                               "observed_change_pa": observed_change,
                                               "minimum_change_pa": required_change,
                                               "rising_steps": rising_steps,
                                               "significant": significant,
                                               "points": n, "context": "same_flow_and_feed_bins"}
    if base["pressure_trend"] and base["pressure_trend"]["significant"]:
        evidence.append("significant_rising_pressure_trend")
    base["supplementary_evidence"] = evidence
    if evidence:
        base["reason"] += ";" + ";".join(evidence)
    return base


def export_filter_zone_labels(*, dataset_version: str | None = None,
                              output_root: Path | None = None,
                              policy: dict | None = None) -> dict:
    """Export all row-admitted filter observations and a hash-bound manifest."""
    from pdm.data.prepare import load_processed

    data = load_processed("filters", dataset_version)
    features, split = data["features"], data["split"]
    overrides = dict(policy or {})
    if "yellow_limit_pa" in overrides and "yellow_limit_policy_id" not in overrides:
        token = f"{float(overrides['yellow_limit_pa']):g}".replace(".", "p")
        overrides["yellow_limit_policy_id"] = f"filter_pressure_warning_{token}pa_v1"
    if "yellow_limit_pa" in overrides and "yellow_limit_source" not in overrides:
        overrides["yellow_limit_source"] = "user_configured_provisional_warning_boundary_pending_expert_validation"
    cfg = {**FILTER_ZONE_POLICY, **overrides}
    if not 0 < float(cfg["yellow_limit_pa"]) < float(cfg["red_limit_pa"]):
        raise ValueError("Filter yellow pressure threshold must be positive and below the red limit")
    quality_path = Path(data["dir"]) / "quality_records.parquet"
    if not quality_path.is_file():
        raise ValueError("Filter zone export requires versioned quality_records.parquet")
    fingerprint = data.get("fingerprint") or {}
    hashed_inputs = {
        "features_hash": Path(data["dir"]) / "features.parquet",
        "units_hash": Path(data["dir"]) / "units.parquet",
        "split_json_hash": Path(data["dir"]) / "split.json",
        "feature_schema_hash": Path(data["dir"]) / "feature_schema.json",
        "quality_records_hash": quality_path,
    }
    drifted = [name for name, path in hashed_inputs.items()
               if not fingerprint.get(name) or not path.is_file()
               or sha256_file(path) != fingerprint.get(name)]
    if drifted:
        raise ValueError("Filter sensor-zone export fingerprint mismatch: " + ", ".join(drifted))
    quality = pd.read_parquet(quality_path)
    keys = ["unit_id", "timestamp_s"]
    if any(key not in quality for key in [*keys, "status"]) or quality.duplicated(keys).any() or features.duplicated(keys).any():
        raise ValueError("Filter row-quality schema or unit/timestamp keys are invalid")
    qcols = [*keys, "status"] + (["reasons"] if "reasons" in quality else [])
    joined = features.merge(quality[qcols], on=keys, how="left", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Row-quality records do not cover filter features exactly")
    source_feature_rows = len(joined)
    source_measurement_rows = len(quality)
    excluded_source_rows = int(quality.status.eq("excluded").sum())
    quality_report = data.get("report", {}).get("quality") or {}
    if source_measurement_rows != int(quality_report.get("source_measurements", source_measurement_rows)):
        raise ValueError("Quality-record count does not match the versioned data-quality report")
    admitted = joined[joined.status.isin(["admitted", "attention"])].copy()
    expected = int((data.get("report", {}).get("quality") or {}).get("admitted_measurements", len(admitted)))
    if len(admitted) != expected:
        raise ValueError(f"Admitted measurement count mismatch: expected {expected}, found {len(admitted)}")
    split_by_unit = {str(uid): name for name in ("train", "validation", "test")
                     for uid in split.get(name, [])}
    missing_units = sorted(set(admitted.unit_id.astype(str)) - set(split_by_unit))
    if missing_units:
        raise ValueError(f"Admitted filter units are absent from the saved split: {missing_units[:5]}")
    admitted["split"] = admitted.unit_id.astype(str).map(split_by_unit)
    pressure = pd.to_numeric(admitted.differential_pressure, errors="coerce")
    flow = pd.to_numeric(admitted.flow_rate, errors="coerce")
    feed = pd.to_numeric(admitted.dust_feed, errors="coerce")
    row_status_ok = admitted.status.isin(["admitted", "attention"])
    bands = [filter_pressure_band(p, f, d,
                pressure_quality_valid=bool(row_status_ok.iloc[i] and np.isfinite(p)
                    and cfg["pressure_valid_range_pa"][0] <= p <= cfg["pressure_valid_range_pa"][1]),
                row_quality_valid=bool(row_status_ok.iloc[i]), policy=cfg)
             for i, (p, f, d) in enumerate(zip(pressure, flow, feed))]
    classes = pd.Series([band["display_zone"] for band in bands], index=admitted.index)
    zone_ids = pd.Series([band["zone"] for band in bands], index=admitted.index)
    reasons = pd.Series([band["reason"] for band in bands], index=admitted.index)
    table = pd.DataFrame({
        "unit_id": admitted.unit_id.astype(str), "split": admitted.split,
        "timestamp_s": pd.to_numeric(admitted.timestamp_s, errors="coerce"),
        "differential_pressure_pa": pressure, "flow_rate_recorded": flow,
        "dust_feed_recorded": feed, "sensor_zone": zone_ids, "display_zone": classes,
        "zone_reason": reasons, "quality_status": admitted.status.astype(str),
    }).sort_values(["unit_id", "timestamp_s"], kind="stable").reset_index(drop=True)
    policy_record = {**cfg, "zone_classes": {
        "green": "pressure < yellow_limit_pa with usable row quality, positive flow and finite feed",
        "yellow": "yellow_limit_pa <= pressure < red_limit_pa with usable row quality, positive flow and finite feed",
        "red": "pressure >= red_limit_pa with usable pressure quality",
        "unknown": "required pressure quality or operating context unavailable",
    }, "classification_inputs": ["differential_pressure", "flow_rate", "dust_feed", "row_quality"],
        "matched_reference_or_trend_changes_class": False, "endpoint_or_rul_used": False}
    fingerprint_sha = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode("utf-8")).hexdigest()
    identity = {"dataset_version": data.get("dataset_version"),
                "dataset_fingerprint_sha256": fingerprint_sha,
                "row_count": len(table), "policy": policy_record}
    artifact_id = "filter_sensor_zones_v2_" + hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    output_root = Path(output_root) if output_root else runs_root() / "_zones" / "filters" / "label_artifacts"
    out_dir = output_root / artifact_id
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_text = table.to_csv(index=False, float_format="%.10g")
    labels_path = out_dir / "labels.csv"
    atomic_write_text(labels_path, csv_text)
    class_names = ("green", "yellow", "red", "unknown")
    display_to_class = {"green": "green", "yellow": "yellow", "red": "red", "gray": "unknown"}
    class_counts_by_row = table.display_zone.map(display_to_class)
    split_counts = {name: {zone: int(class_counts_by_row.loc[part.index].eq(zone).sum()) for zone in class_names}
                    for name, part in table.groupby("split", sort=True)}
    class_counts = {zone: int(class_counts_by_row.eq(zone).sum()) for zone in class_names}
    if sum(class_counts.values()) != len(table):
        raise ValueError("Filter sensor-zone class counts do not sum to exported rows")
    if any(sum(counts.values()) != int((table.split == name).sum())
           for name, counts in split_counts.items()):
        raise ValueError("Filter sensor-zone split counts do not sum to split rows")
    manifest = {
        "schema_version": "filter_sensor_zone_labels_v1", "artifact_id": artifact_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_id": "filters", "dataset_version": data.get("dataset_version"),
        "dataset_fingerprint": fingerprint, "dataset_fingerprint_sha256": fingerprint_sha,
        "policy": policy_record,
        "label_scope": "all row-admitted measurements, including retained attention rows",
        "endpoint_or_rul_used_for_classification": False,
        "future_or_remaining_time_labels_used": False,
        "labels_file": labels_path.name, "labels_sha256": sha256_file(labels_path),
        "row_count": int(len(table)), "source_feature_row_count": int(source_feature_rows),
        "source_measurement_record_count": int(source_measurement_rows),
        "excluded_row_count": excluded_source_rows,
        "unit_count": int(table.unit_id.nunique()),
        "class_counts": class_counts,
        "class_counts_by_split": split_counts,
    }
    atomic_write_json(out_dir / "manifest.json", manifest)
    return {"artifact_id": artifact_id, "directory": str(out_dir), "manifest": manifest}
