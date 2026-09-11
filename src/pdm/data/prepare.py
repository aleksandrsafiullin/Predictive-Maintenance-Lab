from __future__ import annotations

import hashlib
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pdm.config import load_dataset_config
from pdm.data.bearings import (
    build_bearing_units,
    extract_bearings_features,
    inspect_bearings,
)
from pdm.data.filters import extract_filters_tables, filter_time_scale_meta, inspect_filters
from pdm.io_util import atomic_write_json, read_json, sha256_file, sha256_short
from pdm.paths import dataset_processed
from pdm.preprocessing import FEATURE_PIPELINE_VERSION
from pdm.splits import (
    assert_split_coverage,
    bearings_split,
    filters_split,
    populate_origin_unit_id,
    split_hash,
)
from pdm.windows import (
    GAP_RULE_VERSION,
    count_window_eligibility,
    diagnostic_fullfile_gap_marker_count,
    filter_gap_params,
)

ProgressFn = Callable[[str, dict[str, Any]], None]

HASHED_PROCESSED_FILES = (
    "features.parquet",
    "units.parquet",
    "split.json",
    "feature_schema.json",
)
CORE_PROCESSED_FILES = HASHED_PROCESSED_FILES[:3]


def processed_ready(dataset_id: str) -> bool:
    try:
        d = resolve_processed_dir(dataset_id)
    except FileNotFoundError:
        return False
    man_path = dataset_processed(dataset_id) / "manifest.json"
    if man_path.exists():
        return _version_ready(d)
    return _core_ready(d)


def resolve_processed_dir(dataset_id: str) -> Path:
    """Versioned snapshot from manifest, or legacy root when no manifest exists."""
    root = dataset_processed(dataset_id)
    man_path = root / "manifest.json"
    if not man_path.exists():
        return root
    man = read_json(man_path)
    ver = man.get("current_version")
    if not ver:
        raise FileNotFoundError(
            f"processed manifest {man_path} has no current_version; "
            "re-run prepare (leftover root parquet is ignored)."
        )
    d = root / "versions" / str(ver)
    if not _version_ready(d):
        raise FileNotFoundError(
            f"Processed snapshot versions/{ver} is missing or incomplete under {root}. "
            "Re-run prepare. Leftover files at the dataset root are ignored."
        )
    return d


def load_processed(dataset_id: str) -> dict[str, Any]:
    d = resolve_processed_dir(dataset_id)
    features = pd.read_parquet(d / "features.parquet")
    units = pd.read_parquet(d / "units.parquet")
    split = read_json(d / "split.json")
    report = read_json(d / "data_report.json") if (d / "data_report.json").exists() else {}
    fingerprint = load_processed_fingerprint(d, split, dataset_id)
    schema: dict[str, Any] = {}
    schema_path = d / "feature_schema.json"
    if schema_path.exists():
        schema = read_json(schema_path)
    # Train/eval gate this field from processed_fingerprint.json only.
    # Do not fill from feature_schema.json (schema-only ≠ current).
    return {
        "features": features,
        "units": units,
        "split": split,
        "report": report,
        "dir": d,
        "fingerprint": fingerprint,
        "dataset_version": fingerprint.get("dataset_version"),
        "feature_schema": schema,
        "gap_rule_version": fingerprint.get("gap_rule_version"),
    }


def inspect_dataset(dataset_id: str) -> dict[str, Any]:
    if dataset_id == "bearings":
        return inspect_bearings()
    if dataset_id == "filters":
        return inspect_filters()
    raise ValueError(dataset_id)


def attach_origin_unit_id(units: pd.DataFrame) -> pd.DataFrame:
    """Keep a physical-history key on units.parquet (filters: author_data_no)."""
    if "origin_unit_id" in units.columns:
        return units
    return populate_origin_unit_id(units.copy())


def load_processed_fingerprint(
    processed_dir: Path,
    split: dict[str, Any],
    dataset_id: str,
) -> dict[str, Any]:
    path = processed_dir / "processed_fingerprint.json"
    if path.exists():
        return read_json(path)
    return _fingerprint_from_files(
        dataset_id=dataset_id,
        dataset_version="unversioned",
        processed_dir=processed_dir,
        split=split,
        created_at=None,
    )


def dataset_fingerprint_for_run(
    processed: dict[str, Any],
    split: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy processed fingerprint and set canonical split_hash for the run dir."""
    split = split if split is not None else processed["split"]
    dataset_id = str(split.get("dataset_id") or "")
    fp = dict(processed.get("fingerprint") or {})
    if not fp and processed.get("dir") is not None:
        fp = load_processed_fingerprint(Path(processed["dir"]), split, dataset_id)
    fp["split_hash"] = split_hash(split)
    fp.setdefault("feature_pipeline_version", FEATURE_PIPELINE_VERSION)
    fp.setdefault("split_protocol", split.get("protocol"))
    if split.get("dataset_id"):
        fp.setdefault("dataset_id", split["dataset_id"])
    nested = processed.get("fingerprint") or {}
    if nested.get("gap_rule_version") is not None:
        fp["gap_rule_version"] = nested["gap_rule_version"]
    return fp


def prepare_dataset(dataset_id: str, progress: ProgressFn | None = None) -> dict[str, Any]:
    cfg = load_dataset_config(dataset_id)
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
        ts_meta = filter_time_scale_meta(cfg)
        sensor_note = (
            f"CSV Time is multiplied by time_to_seconds={ts_meta['time_to_seconds']} "
            f"(original_time_unit treated as {ts_meta['original_time_unit']}, unverified). "
            "PDF documents Sampling in Hz but does not name the Time unit. "
            "See time_unit_note in this report and reports/implementation_report.md "
            "(Filters / HSE; §7 Limitations). "
            "Training labels use 600 Pa events vs right-censoring; official test RUL is evaluation-only."
        )

    assert_split_coverage(units, split)
    rec = write_processed_version(
        dataset_id,
        features,
        units,
        split,
        inspection=inspection,
        sensor_note=sensor_note,
        cfg=cfg,
    )
    if progress:
        progress(
            "ready",
            {
                "dataset_id": dataset_id,
                "n_units": int(units.shape[0]),
                "dataset_version": rec["dataset_version"],
            },
        )
    return rec


def write_processed_version(
    dataset_id: str,
    features: pd.DataFrame,
    units: pd.DataFrame,
    split: dict[str, Any],
    *,
    inspection: dict[str, Any] | None = None,
    sensor_note: str = "",
    cfg: dict[str, Any] | None = None,
    processed_root: Path | None = None,
) -> dict[str, Any]:
    """Write an immutable versioned snapshot; bump manifest current_version."""
    cfg = cfg or {}
    inspection = inspection or {}
    units = attach_origin_unit_id(units)
    assert_split_coverage(units, split)
    root = processed_root if processed_root is not None else dataset_processed(dataset_id)
    root.mkdir(parents=True, exist_ok=True)
    versions = root / "versions"
    versions.mkdir(parents=True, exist_ok=True)

    schema = {
        "dataset_id": dataset_id,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "gap_rule_version": GAP_RULE_VERSION,
        "feature_columns": [c for c in features.columns],
        "measurement_contract": ["dataset_id", "unit_id", "timestamp_s"],
        "forbidden_model_inputs": [
            "event_time_s",
            "observation_end_s",
            "RUL",
            "split",
            "failure_label",
            "life_fraction",
            "origin_unit_id",
        ],
    }

    staging = Path(tempfile.mkdtemp(prefix=".prepare_tmp.", dir=str(root)))
    dest: Path | None = None
    try:
        features.to_parquet(staging / "features.parquet", index=False)
        units.to_parquet(staging / "units.parquet", index=False)
        atomic_write_json(staging / "split.json", split)
        atomic_write_json(staging / "feature_schema.json", schema)
        atomic_write_json(staging / "inspection.json", inspection)

        hashes = {name: sha256_file(staging / name) for name in HASHED_PROCESSED_FILES}
        now = time.gmtime()
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", now)
        stamp = time.strftime("%Y%m%dT%H%M%SZ", now)
        combined = hashlib.sha256(
            "".join(hashes[name] for name in HASHED_PROCESSED_FILES).encode("ascii")
        ).hexdigest()
        dataset_version = f"{stamp}_{sha256_short(combined, 8)}"
        dest = versions / dataset_version
        suffix = 0
        while dest.exists():
            suffix += 1
            dest = versions / f"{dataset_version}_{suffix}"
        dataset_version = dest.name

        fingerprint = build_processed_fingerprint(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            file_hashes=hashes,
            split=split,
            created_at=created_at,
            gap_rule_version=GAP_RULE_VERSION,
        )
        report = _data_report(dataset_id, features, units, split, cfg, sensor_note, inspection)
        report.update(
            {
                "dataset_version": dataset_version,
                "split_protocol": split.get("protocol"),
                "split_hash": fingerprint["split_hash"],
                "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
                "gap_rule_version": GAP_RULE_VERSION,
                "features_hash": hashes["features.parquet"],
                "units_hash": hashes["units.parquet"],
                "split_json_hash": hashes["split.json"],
                "feature_schema_hash": hashes["feature_schema.json"],
            }
        )
        atomic_write_json(staging / "processed_fingerprint.json", fingerprint)
        atomic_write_json(staging / "data_report.json", report)
        staging.rename(dest)
        staging = None
        _update_processed_manifest(root, dataset_id, dataset_version, created_at)
    except Exception:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "dir": dest,
        "report": report,
        "split": split,
        "n_features_rows": int(len(features)),
        "dataset_version": dataset_version,
        "fingerprint": fingerprint,
    }


def build_processed_fingerprint(
    *,
    dataset_id: str,
    dataset_version: str,
    file_hashes: dict[str, str],
    split: dict[str, Any],
    created_at: str | None = None,
    gap_rule_version: str | None = None,
) -> dict[str, Any]:
    features_h = file_hashes.get("features.parquet", "")
    units_h = file_hashes.get("units.parquet", "")
    split_h = file_hashes.get("split.json", "")
    schema_h = file_hashes.get("feature_schema.json", "")
    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "created_at": created_at,
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "split_protocol": split.get("protocol"),
        "split_hash": split_hash(split),
        "features_hash": features_h,
        "units_hash": units_h,
        "split_json_hash": split_h,
        "feature_schema_hash": schema_h,
        "features_hash_short": sha256_short(features_h),
        "units_hash_short": sha256_short(units_h),
        "split_json_hash_short": sha256_short(split_h),
        "feature_schema_hash_short": sha256_short(schema_h),
    }
    if gap_rule_version is not None:
        out["gap_rule_version"] = gap_rule_version
    return out


def _fingerprint_from_files(
    *,
    dataset_id: str,
    dataset_version: str,
    processed_dir: Path,
    split: dict[str, Any],
    created_at: str | None,
) -> dict[str, Any]:
    hashes: dict[str, str] = {}
    for name in HASHED_PROCESSED_FILES:
        path = processed_dir / name
        if path.exists():
            hashes[name] = sha256_file(path)
    return build_processed_fingerprint(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        file_hashes=hashes,
        split=split,
        created_at=created_at,
    )


def _core_ready(d: Path) -> bool:
    return all((d / name).exists() for name in CORE_PROCESSED_FILES)


def _version_ready(d: Path) -> bool:
    return _core_ready(d) and (d / "processed_fingerprint.json").exists()


def _update_processed_manifest(
    root: Path,
    dataset_id: str,
    dataset_version: str,
    created_at: str,
) -> dict[str, Any]:
    path = root / "manifest.json"
    if path.exists():
        man = read_json(path)
        versions = list(man.get("versions") or [])
    else:
        versions = []
    if dataset_version not in versions:
        versions.append(dataset_version)
    man = {
        "dataset_id": dataset_id,
        "current_version": dataset_version,
        "versions": versions,
        "updated_at": created_at,
    }
    atomic_write_json(path, man)
    return man


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
    ts_meta = filter_time_scale_meta(cfg) if dataset_id == "filters" else {}
    if dataset_id == "filters" and not ts_meta.get("time_scale_verified"):
        issues.append(
            "Time/RUL scale unverified; time_to_seconds=60 is frozen. "
            "See time_unit_note. Do not treat internal seconds as wall-clock minutes."
        )
    if split.get("warning"):
        issues.append(split["warning"])
    if dataset_id == "filters":
        fh_reason = inspection.get("filters_full_history_reason")
        fh_status = inspection.get("filters_full_history_status")
        if fh_reason:
            issues.append(
                f"filters_full_history disabled (status={fh_status}): {fh_reason}"
            )
    ranges = {}
    for col in features.select_dtypes(include="number").columns:
        s = features[col]
        ranges[col] = {
            "min": _num(s.min()),
            "max": _num(s.max()),
            "n_missing": int(s.isna().sum()),
        }
    history_length = int((cfg.get("model") or {}).get("history_length", 20))
    gap_kw: dict[str, float] = {}
    gap_diagnostics: dict[str, Any] | None = None
    if dataset_id == "filters":
        k, samp = filter_gap_params(cfg)
        gap_kw = {"gap_multiplier": k, "sampling_interval_s": samp}
        n_fullfile = diagnostic_fullfile_gap_marker_count(
            features, gap_multiplier=k, sampling_interval_s=samp
        )
        n_causal = int(features["gap_before"].sum()) if "gap_before" in features.columns else 0
        gap_diagnostics = {
            "eligibility_rule": GAP_RULE_VERSION,
            "n_causal_gap_markers": n_causal,
            "n_fullfile_gap_markers": n_fullfile,
            "fullfile_is_diagnostic_only": True,
        }
    window_counts = count_window_eligibility(
        features, units, history_length, dataset_id, split, **gap_kw
    )
    payload: dict[str, Any] = {
        "dataset_id": dataset_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_units": int(units.shape[0]),
        "n_measurements": int(features.shape[0]),
        "n_events": n_events,
        "history_length": history_length,
        "window_counts": window_counts,
        "events_vs_censoring": _events_vs_censoring(dataset_id, units),
        "observation_time_s": _observation_time_summary(features, units, split),
        "regimes": _regime_counts(units),
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
            if k in {
                "train_columns",
                "test_columns",
                "train_units",
                "test_units",
                "n_csv",
                "n_units",
                "notes",
                "issues",
                "filters_full_history_status",
                "filters_full_history_reason",
                "filters_full_history_enabled",
            }
        },
        "history_length_physical": _history_physical(dataset_id, cfg, features),
        **(
            {
                "time_to_seconds": ts_meta["time_to_seconds"],
                "original_time_unit": ts_meta["original_time_unit"],
                "time_scale_verified": ts_meta["time_scale_verified"],
                "time_unit_note": ts_meta["time_unit_note"],
                "time_scale_sources": ts_meta["time_scale_sources"],
                "filters_full_history_status": inspection.get("filters_full_history_status"),
                "filters_full_history_reason": inspection.get("filters_full_history_reason"),
                "filters_full_history_enabled": False,
            }
            if ts_meta
            else {}
        ),
    }
    if gap_diagnostics is not None:
        payload["gap_diagnostics"] = gap_diagnostics
    return payload


def _events_vs_censoring(dataset_id: str, units: pd.DataFrame) -> dict[str, Any]:
    if "event_observed" in units.columns:
        ev = pd.to_numeric(units["event_observed"], errors="coerce").fillna(0).astype(int)
    else:
        ev = pd.Series(0, index=units.index, dtype=int)
    n_obs = int(ev.sum())
    n_cens = int((ev == 0).sum())
    if "official_rul_at_prefix_end_s" in units.columns:
        rul = pd.to_numeric(units["official_rul_at_prefix_end_s"], errors="coerce")
        has_official = rul.notna() & np.isfinite(rul.to_numpy(dtype="float64"))
    else:
        has_official = pd.Series(False, index=units.index)
    n_official = int(((ev == 0) & has_official).sum())
    n_sensor_end = int(((ev == 0) & ~has_official).sum())
    if dataset_id == "filters":
        note = (
            "Observed 600 Pa events count sensor crossings only. "
            "Author test official RUL is an evaluation label at prefix end — "
            "not an observed event and not a failure at observation_end_s. "
            "Right-censored train units stop at the last measured sample."
        )
    else:
        note = (
            "event_observed uses last_recorded_sample (last valid fragment) as an "
            "endpoint approximation, not a confirmed industrial failure time."
        )
    return {
        "n_event_observed": n_obs,
        "n_right_censored": n_cens,
        "n_official_rul_known_not_observed": n_official,
        "n_sensor_end_without_event_or_official_rul": n_sensor_end,
        "note": note,
    }


def _observation_time_summary(
    features: pd.DataFrame,
    units: pd.DataFrame,
    split: dict[str, Any],
) -> dict[str, Any]:
    part_of: dict[str, str] = {}
    for part in ("train", "validation", "test"):
        for uid in split.get(part) or []:
            part_of[str(uid)] = part
    by_split = {part: 0.0 for part in ("train", "validation", "test")}
    total_span = 0.0
    if not features.empty and "timestamp_s" in features.columns:
        spans = features.groupby("unit_id", sort=False)["timestamp_s"].agg(["min", "max"])
        for uid, row in spans.iterrows():
            lo, hi = _num(row["min"]), _num(row["max"])
            span = 0.0 if lo is None or hi is None else max(0.0, float(hi) - float(lo))
            total_span += span
            part = part_of.get(str(uid))
            if part is not None:
                by_split[part] += span
    total_end = None
    if "observation_end_s" in units.columns and not units.empty:
        total_end = _num(pd.to_numeric(units["observation_end_s"], errors="coerce").sum())
    return {
        "total_span_s": _num(total_span) or 0.0,
        "total_observation_end_s": total_end,
        "by_split": {part: _num(val) or 0.0 for part, val in by_split.items()},
    }


def _regime_counts(units: pd.DataFrame) -> list[dict[str, Any]]:
    if units.empty or "regime_id" not in units.columns:
        return []
    rows: list[dict[str, Any]] = []
    for rid, g in units.groupby("regime_id", sort=True, dropna=False):
        rec: dict[str, Any] = {
            "regime_id": None if pd.isna(rid) else str(rid),
            "n_units": int(len(g)),
        }
        if "event_observed" in g.columns:
            rec["n_event_observed"] = int(
                pd.to_numeric(g["event_observed"], errors="coerce").fillna(0).sum()
            )
        rows.append(rec)
    return rows


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
    factor = float(cfg.get("time_to_seconds", 60.0))
    return {
        "history_length": h,
        "median_dt_s": dt,
        "span_s": h * dt if dt == dt else None,
        "note": (
            f"{h} samples × median Δt_s (internal seconds = original Time × {factor:g}; "
            "unit unconfirmed, not wall-clock minutes)"
        ),
    }


def _num(x):
    try:
        v = float(x)
        if v != v:
            return None
        return v
    except Exception:
        return None
