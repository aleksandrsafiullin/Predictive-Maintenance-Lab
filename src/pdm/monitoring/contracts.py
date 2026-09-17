"""Versioned endpoints, units and allowlisted runtime observations."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.windows import raw_numeric_columns

SCHEMA = "monitoring_v1"
OUTCOMES = {"observed_failure", "observed_threshold_crossing", "experiment_end_proxy",
            "right_censored", "planned_intervention", "invalid_endpoint"}


def dataset_profile(dataset_id):
    if dataset_id not in {"bearings", "filters"}:
        raise ValueError("Unknown monitoring dataset")
    bearings = dataset_id == "bearings"
    return {
        "dataset_id": dataset_id, "time_basis": "physical_seconds" if bearings else "dataset_internal",
        "time_scale_verified": bearings, "nominal_interval": 60. if bearings else 6.,
        "signal_name": "horizontal_rms" if bearings else "differential_pressure",
        "signal_unit": "g" if bearings else "Pa",
        "event_definition_id": "bearing_experiment_end_proxy_v1" if bearings else "filter_lab_600pa_v1",
        "event_type": "experiment_end_proxy" if bearings else "observed_threshold_crossing",
        "event_convention": "last_recorded_sample" if bearings else "first_measurement_gt_600pa",
        "mode": "laboratory", "operational_status": "laboratory_only",
    }


def unit_verification(dataset_id):
    p = dataset_profile(dataset_id)
    return {**p, "schema_version": "unit_verification_v1",
            "legacy_timestamp_field": "timestamp_s",
            "legacy_conversion": "fragment index * 60" if dataset_id == "bearings" else "Time * 60 (unverified)",
            "sources": ["configs/" + dataset_id + ".yaml", "docs/quality_and_comparison.md"] + (["data/raw/filters/Preventive to Predictive Maintenance dataset.pdf (v1.4, pp3,6,7)"] if dataset_id == "filters" else []),
            "signal_unit_status": "documented",
            "dust_feed_source_unit": "mm3/s, documented; integral uses unverified time" if dataset_id == "filters" else None,
            "unresolved": [] if dataset_id == "bearings" else [
                "Source Time and RUL units not reconciled with acquisition Sampling Hz",
                "Dust feed physical integral not verified; never label internal integral as mass"],
            "action_timing_physical_allowed": p["time_scale_verified"]}


def observation_columns(dataset_id):
    return list(dict.fromkeys(["dataset_id", "unit_id", "origin_unit_id", "segment_id", "timestamp_s",
        "regime_id", "dust", "gap_before", "quality_gap_before", "maintenance_reset", "segment_reset",
        "n_samples", "sample_count_ok", "n_nan", "source_record_id", "source_row", "relpath",
        *raw_numeric_columns(dataset_id)]))


def observed_prefix(frame, dataset_id, as_of):
    """Discard evaluator labels and all future measurements before any computation."""
    if not np.isfinite(as_of):
        raise ValueError("as_of must be finite in the saved time basis")
    if "timestamp_s" not in frame:
        return frame.iloc[:0].copy()
    clock = pd.to_numeric(frame.timestamp_s, errors="coerce")
    selected = frame.loc[(clock <= as_of) | clock.isna()].copy()
    selected["timestamp_s"] = clock.loc[selected.index]
    if "unit_id" in selected and selected.unit_id.nunique() > 1:
        raise ValueError("A monitoring prefix must contain one equipment unit")
    return selected[[c for c in observation_columns(dataset_id) if c in selected]].copy()


def horizon_label(*, as_of, horizon, observation_end, event_time=None, outcome="right_censored"):
    if outcome not in OUTCOMES or horizon <= 0:
        raise ValueError("Invalid outcome or horizon")
    if outcome in {"invalid_endpoint", "planned_intervention"}:
        return None
    if event_time is not None and np.isfinite(event_time):
        if event_time <= as_of:
            return None
        if event_time <= as_of + horizon:
            return 1
    return 0 if observation_end >= as_of + horizon else None
