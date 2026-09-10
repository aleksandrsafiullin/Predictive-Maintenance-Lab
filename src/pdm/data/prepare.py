from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from pdm.config import load_dataset_config
from pdm.data.bearings import (
    build_bearing_units,
    extract_bearings_features,
    inspect_bearings,
)
from pdm.data.filters import extract_filters_tables, inspect_filters
from pdm.io_util import atomic_write_json
from pdm.paths import dataset_processed, dataset_raw
from pdm.splits import bearings_split, filters_split

ProgressFn = Callable[[str, dict[str, Any]], None]


def processed_ready(dataset_id: str) -> bool:
    d = dataset_processed(dataset_id)
    return (d / "features.parquet").exists() and (d / "units.parquet").exists() and (d / "split.json").exists()


def load_processed(dataset_id: str) -> dict[str, Any]:
    d = dataset_processed(dataset_id)
    features = pd.read_parquet(d / "features.parquet")
    units = pd.read_parquet(d / "units.parquet")
    split = json.loads((d / "split.json").read_text(encoding="utf-8"))
    report = json.loads((d / "data_report.json").read_text(encoding="utf-8")) if (d / "data_report.json").exists() else {}
    return {"features": features, "units": units, "split": split, "report": report, "dir": d}


def inspect_dataset(dataset_id: str) -> dict[str, Any]:
    if dataset_id == "bearings":
        return inspect_bearings()
    if dataset_id == "filters":
        return inspect_filters()
    raise ValueError(dataset_id)


def prepare_dataset(dataset_id: str, progress: ProgressFn | None = None) -> dict[str, Any]:
    cfg = load_dataset_config(dataset_id)
    out = dataset_processed(dataset_id)
    out.mkdir(parents=True, exist_ok=True)
    if progress:
        progress("inspect", {"dataset_id": dataset_id})
    inspection = inspect_dataset(dataset_id)

    if dataset_id == "bearings":
        if progress:
            progress("features", {"dataset_id": dataset_id})
        features = extract_bearings_features(cfg, progress=progress)
        units = build_bearing_units(features, cfg)
        split = bearings_split(units, cfg)
        sensor_note = (
            "Each CSV is a 1.28 s recording at 25.6 kHz, acquired once per minute. "
            "timestamp_s = file_index * 60 s (fragment interval), not 1.28 s. "
            "Features of a fragment are available only after that fragment is complete. "
            "event_time_s is the last valid fragment (endpoint_definition=last_recorded_sample), "
            "an approximation of test end, not a confirmed industrial failure time."
        )
    else:
        if progress:
            progress("features", {"dataset_id": dataset_id})
        features, units = extract_filters_tables(cfg, progress=progress)
        split = filters_split(units, cfg)
        sensor_note = (
            "CSV Time is converted to seconds with factor "
            f"{cfg.get('time_to_seconds')} (original unit treated as {cfg.get('original_time_unit')}). "
            "PDF documents Sampling in Hz but does not name the Time unit; this conversion is recorded in data_report.json. "
            "Training labels use 600 Pa events vs right-censoring; official test RUL is evaluation-only."
        )

    features.to_parquet(out / "features.parquet", index=False)
    units.to_parquet(out / "units.parquet", index=False)
    atomic_write_json(out / "split.json", split)
    atomic_write_json(out / "inspection.json", inspection)

    report = _data_report(dataset_id, features, units, split, cfg, sensor_note, inspection)
    atomic_write_json(out / "data_report.json", report)
    schema = {
        "dataset_id": dataset_id,
        "feature_columns": [c for c in features.columns],
        "measurement_contract": ["dataset_id", "unit_id", "timestamp_s"],
        "forbidden_model_inputs": [
            "event_time_s",
            "observation_end_s",
            "RUL",
            "split",
            "failure_label",
            "life_fraction",
        ],
    }
    atomic_write_json(out / "feature_schema.json", schema)
    if progress:
        progress("ready", {"dataset_id": dataset_id, "n_units": int(units.shape[0])})
    return {"dir": out, "report": report, "split": split, "n_features_rows": int(len(features))}


def _data_report(dataset_id, features, units, split, cfg, sensor_note, inspection) -> dict[str, Any]:
    n_events = int(units["event_observed"].sum()) if "event_observed" in units.columns else None
    issues = []
    if "gap_before" in features.columns and int(features["gap_before"].sum()):
        issues.append(
            f"{int(features['gap_before'].sum())} gap markers; windows do not cross gaps."
        )
    if dataset_id == "filters" and n_events is not None and n_events < 8:
        issues.append(
            f"Only {n_events} observed 600 Pa events among all units shown; "
            "validation MAE on events is not a full-set quality score."
        )
    if split.get("warning"):
        issues.append(split["warning"])
    ranges = {}
    for col in features.select_dtypes(include="number").columns:
        s = features[col]
        ranges[col] = {
            "min": _num(s.min()),
            "max": _num(s.max()),
            "n_missing": int(s.isna().sum()),
        }
    return {
        "dataset_id": dataset_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_units": int(units.shape[0]),
        "n_measurements": int(features.shape[0]),
        "n_events": n_events,
        "split_counts": {
            "train_units": split.get("n_train"),
            "validation_units": split.get("n_validation"),
            "test_units": split.get("n_test"),
        },
        "units_by_split": {
            "train": split.get("train"),
            "validation": split.get("validation"),
            "test": split.get("test"),
        },
        "sensor_time_note": sensor_note,
        "issues_and_decisions": issues,
        "parameter_ranges": ranges,
        "inspection_summary": {
            k: inspection.get(k)
            for k in inspection
            if k in {"train_columns", "test_columns", "train_units", "test_units", "n_csv", "n_units", "notes", "issues"}
        },
        "history_length_physical": _history_physical(dataset_id, cfg, features),
    }


def _history_physical(dataset_id, cfg, features) -> dict[str, Any]:
    h = int(cfg.get("model", {}).get("history_length", 20))
    if dataset_id == "bearings":
        dt = float(cfg.get("fragment_interval_s", 60))
        return {
            "history_length": h,
            "dt_s": dt,
            "span_s": h * dt,
            "note": f"{h} fragments × {dt}s interval = {h * dt / 60:.0f} minutes of operating time",
        }
    dt = float(features["delta_t_s"].median()) if "delta_t_s" in features.columns else float("nan")
    return {
        "history_length": h,
        "median_dt_s": dt,
        "span_s": h * dt if dt == dt else None,
        "note": f"{h} samples × median Δt",
    }


def _num(x):
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None
