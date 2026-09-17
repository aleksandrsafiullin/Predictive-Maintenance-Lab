"""Versioned, deterministic admission of measurements and outcome labels.

Raw sources are never rewritten. A rejected measurement remains in the audit,
and its removal creates an explicit boundary in the retained history.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd

from pdm.windows import build_windows, filter_gap_params, raw_numeric_columns

QUALITY_VERSION = "admission_v1"
QUALITY_POLICY = {
    "version": QUALITY_VERSION,
    "censoring": "retain_valid_right_censored",
    "outliers": "diagnostic_only",
    "duplicates": "collapse_identical_reject_conflicting",
    "endpoints": "never_move_after_rejection",
    "missing": "reject_required_nonfinite",
}
QUALITY_POLICY_HASH = hashlib.sha256(json.dumps(QUALITY_POLICY, sort_keys=True).encode()).hexdigest()


def admit_dataset(dataset_id, features, units, split, cfg):
    """Return admitted frames, unchanged membership subsets, and a row audit."""
    f = features.copy().reset_index(drop=True)
    u = units.copy()
    reasons = [set() for _ in range(len(f))]
    warnings = [set() for _ in range(len(f))]
    required = ["timestamp_s", *raw_numeric_columns(dataset_id)]
    missing = sorted(set(required) - set(f.columns))
    if missing:
        raise ValueError(f"Required measurement columns missing: {missing}")
    numeric = f[required].apply(pd.to_numeric, errors="coerce")
    for i in np.flatnonzero(~np.isfinite(numeric.to_numpy()).all(axis=1)):
        reasons[i].add("nonfinite_required_measurement")
    for i in np.flatnonzero(numeric.timestamp_s.to_numpy() < 0):
        reasons[i].add("negative_timestamp")
    if "_quality_errors" in f:
        for i, value in enumerate(f._quality_errors.fillna("")):
            reasons[i].update(str(value).split(";") if value else [])
    if "sample_count_ok" in f:
        for i in np.flatnonzero(~f.sample_count_ok.fillna(False).astype(bool)):
            reasons[i].add("invalid_fragment_length")
    if "n_nan" in f:
        for i in np.flatnonzero(f.n_nan.fillna(1).to_numpy() > 0):
            reasons[i].add("nonfinite_raw_signal")

    for name in ("horizontal_quality_diagnostics", "vertical_quality_diagnostics"):
        if name in f:
            for i, value in enumerate(f[name].fillna("")):
                warnings[i].update(str(value).split(";") if value else [])

    # Compare measurement content, not its filename or quality annotations.
    content = [c for c in f if c not in {"relpath", "_quality_errors", "gap_before", "quality_gap_before", "delta_t_s", "delta_pressure"}]
    for _, g in f[f.duplicated(["unit_id", "timestamp_s"], keep=False)].groupby(["unit_id", "timestamp_s"], dropna=False, sort=False):
        if len(g) < 2:
            continue
        identical = len(g[content].drop_duplicates()) == 1
        indices = g.index[1:] if identical else g.index
        for i in indices:
            reasons[i].add("duplicate_identical" if identical else "duplicate_conflict")

    unit_reasons = {}
    for _, unit in u.iterrows():
        uid = str(unit.unit_id)
        idx = f.index[f.unit_id.astype(str) == uid]
        g = f.loc[idx]
        bad = [i for i in idx if reasons[i] - {"duplicate_identical"}]
        why = set()
        end = unit.get("observation_end_s")
        event = unit.get("event_time_s")
        observed = unit.get("event_observed")
        if observed not in (0, 1) or not _finite(end) or float(end) < 0:
            why.add("invalid_outcome_label")
        if observed == 1 and (not _finite(event) or float(event) < 0 or (_finite(end) and event > end)):
            why.add("invalid_event_time")
        if dataset_id == "bearings" and any(not _finite(f.at[i, "timestamp_s"]) or f.at[i, "timestamp_s"] == event for i in bad):
            why.add("unreliable_recorded_endpoint")
        if dataset_id == "filters" and bad:
            # A missing pressure could hide an earlier crossing. A broken clock
            # also invalidates event/censor durations; never invent a label.
            why.add("unreliable_event_or_censor_time")
        official = unit.get("official_rul_at_prefix_end_s")
        if dataset_id == "filters" and unit.get("author_split") == "author_test":
            if not _finite(official) or official < 0:
                why.add("invalid_official_rul")
        if g.empty or not any(not reasons[i] for i in idx):
            why.add("no_valid_measurements")
        if why:
            unit_reasons[uid] = sorted(why)
            for i in idx:
                reasons[i].update(why)

    accepted = np.asarray([not r for r in reasons], dtype=bool)
    clean = f.loc[accepted].copy()
    clean["quality_gap_before"] = False
    for _, g in f.groupby("unit_id", sort=False):
        rejected_since_last = False
        for i in g.sort_values("timestamp_s", kind="stable").index:
            if not accepted[i]:
                rejected_since_last |= bool(reasons[i] - {"duplicate_identical"})
            else:
                clean.at[i, "quality_gap_before"] = rejected_since_last
                rejected_since_last = False
    if dataset_id == "bearings" and "file_index" in clean:
        # Raw duplicate files can create a false zero-spacing boundary. Once
        # duplicates collapse, derive genuine missing-fragment boundaries again.
        for _, g in clean.groupby("unit_id", sort=False):
            ordered = g.sort_values("timestamp_s")
            clean.loc[ordered.index, "gap_before"] = ordered.file_index.diff().fillna(1).ne(1).to_numpy()
    clean["gap_before"] = clean.get("gap_before", False) | clean.quality_gap_before
    # Robust diagnostics never decide admission and are never model inputs.
    sensor_cols = [c for c in raw_numeric_columns(dataset_id) if c not in {"operating_age_s", "delta_t_s", "rpm", "load_kn"}]
    for _, g in clean.groupby("unit_id", sort=False):
        for col in sensor_cols:
            values = g[col].astype(float)
            med = values.median()
            mad = (values - med).abs().median()
            if mad > 0:
                for i in g.index[(values - med).abs() > 12 * mad]:
                    warnings[i].add("unusual_signal_retained")
        for i in g.index[g.gap_before.astype(bool)]:
            warnings[i].add("history_gap")
    clean = clean.drop(columns=["_quality_errors"], errors="ignore").reset_index(drop=True)
    valid_ids = set(clean.unit_id.astype(str)) - set(unit_reasons)
    u = u[u.unit_id.astype(str).isin(valid_ids)].copy()
    counts = clean.groupby("unit_id").size()
    u["n_measurements"] = u.unit_id.map(counts).astype(int)
    active_split = dict(split)
    for part in ("train", "validation", "test"):
        active_split[part] = [uid for uid in split[part] if str(uid) in valid_ids]
        active_split[f"n_{part}"] = len(active_split[part])
    membership = {str(uid): part for part in ("train", "validation", "test") for uid in split[part]}
    audit = pd.DataFrame({
        "unit_id": f.unit_id.astype(str), "timestamp_s": pd.to_numeric(f.timestamp_s, errors="coerce"),
        "source": f.get("relpath", pd.Series("CSV measurement", index=f.index)),
        "split": f.unit_id.astype(str).map(membership),
        "status": ["excluded" if r else "attention" if w else "admitted" for r, w in zip(reasons, warnings, strict=True)],
        "reasons": [";".join(sorted(r | w)) for r, w in zip(reasons, warnings, strict=True)],
    })
    audit["source_row"] = f.index
    summary = {
        "policy": QUALITY_POLICY, "policy_hash": QUALITY_POLICY_HASH,
        "source_measurements": len(f), "admitted_measurements": int(accepted.sum()),
        "excluded_measurements": int((~accepted).sum()),
        "attention_measurements": int((audit.status == "attention").sum()),
        "source_units": len(units), "admitted_units": len(u), "excluded_units": unit_reasons,
        "reasons": dict(Counter(reason for r, w in zip(reasons, warnings, strict=True) for reason in r | w)),
        "original_split": split,
    }
    return clean, u.reset_index(drop=True), active_split, audit, summary


def _finite(value):
    try:
        return bool(np.isfinite(float(value)))
    except (ValueError, TypeError):
        return False


def training_admission(processed, cfg, history_length, *, events_only=False):
    """Gate both training engines. No imputation may silently admit bad inputs."""
    fp = processed.get("fingerprint") or {}
    if fp.get("quality_policy_hash") != QUALITY_POLICY_HASH:
        raise ValueError("Prepare data with the current Data Quality policy before training.")
    features, units = processed["features"], processed["units"]
    values = features[["timestamp_s", *raw_numeric_columns(cfg["dataset_id"])]].to_numpy(dtype=float)
    if processed.get("dir"):
        from pathlib import Path

        from pdm.io_util import sha256_file

        for name, key in (("features.parquet", "features_hash"), ("units.parquet", "units_hash"), ("quality_records.parquet", "quality_records_hash")):
            if sha256_file(Path(processed["dir"]) / name) != fp.get(key):
                raise ValueError(f"Quality snapshot changed: {name}. Prepare data again.")
    if not np.isfinite(values).all():
        raise ValueError("Admitted measurements contain nonfinite values; prepare data again.")
    split = dict(processed["split"])
    if events_only:
        if cfg["dataset_id"] != "filters":
            raise ValueError("Events-only ablation is supported for filters only")
        observed = set(units.loc[units.event_observed == 1, "unit_id"])
        split["train"] = [uid for uid in split["train"] if uid in observed]
        split["n_train"] = len(split["train"])
    kw = {}
    if cfg["dataset_id"] == "filters":
        k, dt = filter_gap_params(cfg)
        kw = {"gap_multiplier": k, "sampling_interval_s": dt}
    windows = build_windows(features, units, history_length, cfg["dataset_id"], **kw)
    counts = []
    for part in ("train", "validation", "test"):
        w = windows[windows.unit_id.isin(split[part])] if not windows.empty else windows
        if part != "test" and (not split[part] or w.empty):
            raise ValueError(f"No eligible {part} windows after quality admission (history={history_length}).")
        us = units[units.unit_id.isin(split[part])]
        counts.append({"split": part, "units": len(us), "observed_events": int(us.event_observed.sum()), "eligible_windows": len(w)})
    return split, counts


def fragment_diagnostics(signal, *, expected_samples=None, sensor_range=None):
    """Audit only: clipping/stuck suspicion must not delete real fault impulses."""
    values = np.asarray(signal, dtype=float)
    finite = values[np.isfinite(values)]
    reasons = []
    if expected_samples is not None and len(values) != expected_samples:
        reasons.append("fragment_length_mismatch")
    nonfinite = float((~np.isfinite(values)).mean()) if values.size else 1.
    if nonfinite:
        reasons.append("nonfinite_raw_signal")
    constant = bool(finite.size > 1 and np.ptp(finite) == 0)
    if constant:
        reasons.append("constant_fragment_check_sensor")
    saturation = None
    if sensor_range is not None and finite.size:
        low, high = sensor_range
        saturation = float(((finite <= low) | (finite >= high)).mean())
        if saturation:
            reasons.append("at_known_sensor_range")
    repeated = bool(len(values) >= 16 and len(values) % 2 == 0 and np.array_equal(values[:len(values)//2], values[len(values)//2:]))
    if repeated:
        reasons.append("repeated_fragment_halves")
    return {"nonfinite_fraction": nonfinite, "constant_fragment": constant,
            "saturation_fraction": saturation, "repeated_fragment": repeated,
            "diagnostic_reasons": ";".join(reasons), "amplitude_only_exclusion": False}
