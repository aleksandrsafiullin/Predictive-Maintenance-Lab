"""Frozen Train-only, causal inputs for project RED-entry v2.

Padding is storage, never a measurement. Metadata/quality/targets are not model
inputs; explicit known masks can only remove unavailable source values.
"""
from __future__ import annotations

import copy
import re

import numpy as np
import pandas as pd

from pdm.project_zones import resolve_thresholds
from pdm.red_entry_protocol import canonical_json_hash

FEATURE_VERSION = "project_red_entry_features_v2"
MODES = {"age_context", "sensor_only", "hybrid"}
ROLES = {"sensor", "operating_context", "known_age", "quality", "metadata", "target_only"}
# Defense in depth against accidentally assigning a clock/counter sensor role.
CLOCK_NAME = re.compile(r"age|elapsed|timestamp|row_?index|row_?number|record_?(?:index|number|end)|sample_?(?:index|number)|counter|(?:^|_)time(?:_|$)|cycle|lifetime|life_fraction|remaining|rul", re.I)


def _flag(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "1.0"}
    return bool(value)


def _ordered(prefix):
    frame = prefix.copy().reset_index(drop=True)
    if len(frame) and "timestamp_s" in frame:
        times = pd.to_numeric(frame.timestamp_s, errors="coerce").to_numpy(float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Prefix needs finite, strictly increasing timestamps")
    if "unit_id" in frame and frame.unit_id.astype(str).nunique() > 1:
        raise ValueError("Prefix must contain one unit")
    return frame


def _starts(frame):
    starts, start = [], 0
    cycle = next((c for c in ("component_cycle_id", "cycle_id") if c in frame), None)
    for i in range(len(frame)):
        changed = i > 0 and cycle and str(frame[cycle].iloc[i]) != str(frame[cycle].iloc[i - 1])
        if i and (changed or _flag(frame.iloc[i].get("gap_before", False)) or
                  _flag(frame.iloc[i].get("component_replaced", False))):
            start = i
        starts.append(start)
    return starts


def _raw(frame, schema, source_specs):
    result = pd.DataFrame(index=frame.index)
    for name, spec in source_specs.items():
        values = frame.get(name, pd.Series(np.nan, index=frame.index)).copy()
        mask_name = name + "_known" if name != "operating_age_s" else "operating_age_known"
        if mask_name in frame:
            values = values.where(frame[mask_name].map(_flag))
        if spec["kind"] == "numeric":
            values = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
            if spec["role"] == "known_age":
                values = values.where(values >= 0)
        else:
            values = values.where(values.notna() & values.astype(str).str.strip().ne(""))
        result[name] = values
    starts = _starts(frame)
    if "signal" in source_specs:
        signal = pd.to_numeric(result.signal, errors="coerce").to_numpy(float)
        times = pd.to_numeric(frame.get("timestamp_s", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(float)
        slopes, limits, distances = [], [], []
        cycle_start = 0
        cycle_key = next((c for c in ("component_cycle_id", "cycle_id") if c in frame), None)
        for i in range(len(frame)):
            if i and ((cycle_key and str(frame[cycle_key].iloc[i]) != str(frame[cycle_key].iloc[i - 1])) or _flag(frame.iloc[i].get("component_replaced", False))):
                cycle_start = i
            lo = max(starts[i], i - 4)
            s, t = signal[lo:i + 1], times[lo:i + 1]
            good = np.isfinite(s) & np.isfinite(t)
            if good.sum() >= 2 and np.ptp(t[good]) > 0:
                centered = t[good] - np.mean(t[good])
                slope = float(np.dot(centered, s[good] - np.mean(s[good])) / np.dot(centered, centered))
            else:
                slope = np.nan
            threshold = resolve_thresholds(schema, frame.iloc[cycle_start:i + 1])
            limit = threshold.get("red") if threshold["status"] == "available" else np.nan
            limit = np.nan if limit is None else float(limit)
            distance = (limit - signal[i]) if threshold.get("direction", "above") == "above" else (signal[i] - limit)
            slopes.append(slope)
            limits.append(limit)
            distances.append(distance)
        result["signal_trailing_slope"] = slopes
        result["known_red_limit"] = limits
        result["distance_to_red"] = distances
    return result, starts


def fit_feature_state(snapshot, input_mode, *, schema=None):
    """Fit numeric medians/scales and category vocabulary exclusively on Train."""
    if input_mode not in MODES:
        raise ValueError("Unsupported input mode")
    effective = copy.deepcopy(snapshot["schema"] if schema is None else schema)
    train_ids = {str(v) for v in snapshot["split"].get("train", [])}
    frame = snapshot["features"]
    train = frame[frame.unit_id.astype(str).isin(train_ids)].copy()
    if train.empty:
        raise ValueError("Feature fitting requires Train measurements")
    columns = effective.get("columns")
    legacy = not columns
    columns = columns or {"signal": {"role": "sensor"}}
    selected = {}
    for name, spec in columns.items():
        role = spec.get("role")
        if role not in ROLES:
            raise ValueError(f"Unsupported schema role for {name}: {role}")
        allowed = role == "sensor" if input_mode == "sensor_only" else role in {"known_age", "operating_context"} if input_mode == "age_context" else role in {"sensor", "known_age", "operating_context"}
        if role != "known_age" and CLOCK_NAME.search(name):
            allowed = False
        if not allowed or name not in train:
            continue
        dtype = spec.get("dtype", spec.get("type"))
        categorical = dtype in {"categorical", "category", "string", "str", "object"}
        numeric = pd.api.types.is_numeric_dtype(train[name]) or (not categorical and pd.to_numeric(train[name].dropna(), errors="coerce").notna().all())
        selected[name] = {"role": role, "kind": "numeric" if numeric and not categorical else "categorical"}
    raw_parts = [_raw(_ordered(unit.sort_values("timestamp_s", kind="stable")), effective, selected)[0]
                 for _, unit in train.groupby("unit_id", sort=True)]
    raw = pd.concat(raw_parts, ignore_index=True)
    specs, names = {}, []
    for name in raw.columns:
        kind = selected.get(name, {"kind": "numeric"})["kind"]
        if kind == "numeric":
            values = raw[name].to_numpy(float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if len(finite) else 0.0
            filled = np.where(np.isfinite(values), values, median)
            scale = float(np.std(filled)) if len(filled) else 1.0
            specs[name] = {"kind": kind, "impute": median, "mean": float(np.mean(filled)) if len(filled) else 0.0, "scale": scale if scale > 0 else 1.0}
            names.extend([name, name + "__missing"])
        else:
            vocabulary = sorted({str(v) for v in raw[name].dropna()})
            specs[name] = {"kind": kind, "vocabulary": vocabulary}
            names.extend([name + "__category_" + str(i) for i in range(len(vocabulary))] + [name + "__unknown"])
    state = {"feature_version": FEATURE_VERSION, "input_mode": input_mode, "legacy_signal_only": legacy,
             "schema_hash": canonical_json_hash(effective), "schema": effective,
             "source_specs": selected, "preprocessing": specs, "preprocessing_order": list(specs), "feature_names": names,
             "train_unit_ids": sorted(train_ids), "recipe": {"slope_trailing_samples": 5, "sequence_reset": "gap_or_new_cycle", "padding": "right_zeros_with_actual_lengths"}}
    state["state_hash"] = canonical_json_hash(state)
    return state


def transform_prefix(prefix, schema, state):
    """Transform actual prefix rows with frozen state; never refit on inference."""
    if canonical_json_hash(schema) != state["schema_hash"]:
        raise ValueError("Feature schema does not match fitted state")
    expected = state.get("state_hash")
    if expected != canonical_json_hash({k: v for k, v in state.items() if k != "state_hash"}):
        raise ValueError("Feature state integrity mismatch")
    frame = _ordered(prefix)
    raw, starts = _raw(frame, schema, state["source_specs"])
    columns = []
    for name in state["preprocessing_order"]:
        spec = state["preprocessing"][name]
        values = raw[name]
        if spec["kind"] == "numeric":
            arr = values.to_numpy(float)
            missing = ~np.isfinite(arr)
            columns.extend([(np.where(missing, spec["impute"], arr) - spec["mean"]) / spec["scale"], missing.astype(float)])
        else:
            strings = values.astype(str)
            columns.extend([(values.notna() & strings.eq(cat)).to_numpy(float) for cat in spec["vocabulary"]])
            columns.append((values.isna() | ~strings.isin(spec["vocabulary"])).to_numpy(float))
    x = np.column_stack(columns).astype(np.float32) if columns else np.empty((len(frame), 0), dtype=np.float32)
    available = np.zeros(len(frame), dtype=bool)
    supported = np.ones(len(frame), dtype=bool)
    age_available = np.zeros(len(frame), dtype=bool)
    for name, spec in state["source_specs"].items():
        valid = raw[name].notna().to_numpy()
        if spec["role"] == "known_age":
            age_available |= valid
        if spec["kind"] == "categorical":
            known_category = raw[name].astype(str).isin(state["preprocessing"][name]["vocabulary"]).to_numpy()
            supported &= ~valid | known_category
            valid = valid & known_category
        available |= valid
    if "quality_ok" in frame:
        available &= frame.quality_ok.map(_flag).to_numpy(bool)
    if "usable" in frame:
        available &= frame.usable.map(_flag).to_numpy(bool)
    if "quality_status" in frame:
        available &= frame.quality_status.fillna("ok").astype(str).str.lower().isin({"good", "ok", "usable", "valid"}).to_numpy()
    return {"x": x, "feature_names": list(state["feature_names"]), "availability": available,
            "supported_regime": supported, "age_available": age_available,
            "sequence_starts": np.asarray(starts, dtype=np.int64), "input_mode": state["input_mode"]}


def build_windows(snapshot, origins, state, history_length, *, schema=None):
    """Return windows in supplied origins order, with real rows left aligned."""
    if isinstance(history_length, bool) or not isinstance(history_length, int) or history_length < 1:
        raise ValueError("history_length must be a positive integer")
    effective = snapshot["schema"] if schema is None else schema
    origin_frame = origins if isinstance(origins, pd.DataFrame) else pd.DataFrame(origins)
    cache = {}
    for uid, unit in snapshot["features"].groupby("unit_id", sort=False):
        unit = unit.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        cache[str(uid)] = (unit, transform_prefix(unit, effective, state))
    x = np.zeros((len(origin_frame), history_length, len(state["feature_names"])), dtype=np.float32)
    lengths = np.zeros(len(origin_frame), dtype=np.int64)
    available = np.zeros(len(origin_frame), dtype=bool)
    physical_ids = []
    supported = np.ones(len(origin_frame), dtype=bool)
    age_available = np.zeros(len(origin_frame), dtype=bool)
    for j, (_, origin) in enumerate(origin_frame.iterrows()):
        uid = str(origin.unit_id)
        if uid not in cache:
            raise ValueError(f"Origin unit not present: {uid}")
        unit, transformed = cache[uid]
        match = np.flatnonzero(unit.timestamp_s.to_numpy(float) == float(origin.timestamp_s))
        if len(match) != 1:
            raise ValueError("Origin must match exactly one measurement")
        end = int(match[0])
        start = max(int(transformed["sequence_starts"][end]), end - history_length + 1)
        count = end - start + 1
        x[j, :count] = transformed["x"][start:end + 1]
        lengths[j] = count
        available[j] = transformed["availability"][end]
        supported[j] = transformed["supported_regime"][end]
        age_available[j] = transformed["age_available"][end]
        physical_ids.append(str(origin.get("physical_unit_id", unit.iloc[end].get("physical_unit_id", uid))))
    return {"x": x, "lengths": lengths, "availability": available,
            "supported_regime": supported, "age_available": age_available,
            "physical_unit_ids": physical_ids, "feature_names": list(state["feature_names"]),
            "origins": origin_frame.copy(), "input_mode": state["input_mode"]}
