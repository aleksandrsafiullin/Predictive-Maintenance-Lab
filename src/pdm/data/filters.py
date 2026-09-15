from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pdm.data.archive import (
    mat_fieldnames_are_envelope,
    non_envelope_mat_fieldnames,
    probe_mat,
)
from pdm.features import causal_delta
from pdm.paths import dataset_raw
from pdm.windows import filter_gap_params, gap_before_from_delta_t

ProgressFn = Callable[[str, dict[str, Any]], None]

PRESSURE_LIMIT_PA = 600.0
# Frozen conversion. Not calibrated wall-clock minutes; PDF does not name Time.
FILTER_TIME_TO_SECONDS = 60.0
FILTER_TIME_SCALE_SOURCES = (
    "reports/implementation_report.md (Filters / HSE)",
    "reports/implementation_report.md §7 Limitations",
    "configs/filters.yaml",
)
FILTER_TIME_SCALE_WARNING = (
    "Filter Time/RUL unit compatibility is unconfirmed. The source PDF does not "
    "name the Time column unit. Internal values use Time × 60 (working assumption: "
    "original unit = minutes → seconds). This is not calibrated wall-clock time — "
    "do not read alerts as “X minutes before failure”. Original CSV Time and test "
    "RUL are stored as time_original and official_rul_at_prefix_end_original. "
    "Assumption recorded in reports/implementation_report.md (Filters / HSE; "
    "§7 Limitations); CRAN degradr labels RUL as hours (conflict, not used). "
    "time_to_seconds=60 is frozen; changing it requires a new data version and retrain."
)
FILTERS_CENSORED_MODE = "filters_censored"
FILTERS_FULL_HISTORY_MAT_NAME = "Train_Data_Uncensored.mat"
FILTERS_FULL_HISTORY_STATUSES = ("missing", "unreadable", "readable_needs_protocol")
FILTERS_FULL_HISTORY_ENABLED = False

FORBIDDEN_INPUTS = {
    "rul",
    "remaining_useful_life",
    "event",
    "event_observed",
    "event_time_s",
    "split",
    "failure",
    "life_fraction",
    "percent_life",
    "official_rul_at_prefix_end_s",
    "official_rul_at_prefix_end_original",
}


def locate_filter_csvs(raw_dir: Path | None = None) -> dict[str, Path]:
    raw_dir = raw_dir or dataset_raw("filters")
    train = list(raw_dir.rglob("Train_Data_CSV.csv"))
    test = list(raw_dir.rglob("Test_Data_CSV.csv"))
    if not train or not test:
        found = [str(p.relative_to(raw_dir)) for p in raw_dir.rglob("*") if p.is_file()]
        raise FileNotFoundError(
            f"Expected Train_Data_CSV.csv and Test_Data_CSV.csv under {raw_dir}. Found: {found}"
        )
    return {"train": train[0], "test": test[0], "raw_dir": raw_dir}


def locate_filters_full_history_mat(raw_dir: Path) -> Path | None:
    direct = raw_dir / FILTERS_FULL_HISTORY_MAT_NAME
    if direct.is_file():
        return direct
    hits = sorted(p for p in raw_dir.rglob(FILTERS_FULL_HISTORY_MAT_NAME) if p.is_file())
    return hits[0] if hits else None


def probe_filters_full_history(raw_dir: Path) -> dict[str, Any]:
    """Classify Train_Data_Uncensored.mat. Never invents MATLAB table columns."""
    mat_path = locate_filters_full_history_mat(raw_dir)
    if mat_path is None:
        probe = {
            "path": str(raw_dir / FILTERS_FULL_HISTORY_MAT_NAME),
            "exists": False,
            "load_ok": False,
            "error": "file not found",
            "matlab_header": None,
            "variables": [],
        }
        status = "missing"
    else:
        probe = probe_mat(mat_path)
        status = _filters_full_history_status(probe)
    reason = _filters_full_history_reason(status, probe)
    if status not in FILTERS_FULL_HISTORY_STATUSES:
        raise RuntimeError(f"unexpected filters_full_history_status={status!r}")
    return {
        "filters_full_history_status": status,
        "filters_full_history_reason": reason,
        "filters_full_history_enabled": FILTERS_FULL_HISTORY_ENABLED,
        "mat_probe": probe,
    }


def _filters_full_history_status(probe: dict[str, Any]) -> str:
    if not probe.get("exists"):
        return "missing"
    if not probe.get("load_ok"):
        return "unreadable"
    variables = list(probe.get("variables") or [])
    primary = _primary_experiment_tables(variables)
    if not primary:
        # Numeric sidecars are not the experiment table.
        return "unreadable"
    if any(_mat_variable_opaque_table(v) for v in primary):
        return "unreadable"
    if any(_mat_experiment_table_usable(v) for v in primary):
        return "readable_needs_protocol"
    return "unreadable"


def _primary_experiment_tables(variables: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hits = []
    for var in variables:
        name = str(var.get("name") or "").lower()
        if name == "train_data_uncensored" or "uncensored" in name:
            hits.append(var)
    return hits


def _mat_variable_opaque_table(var: dict[str, Any]) -> bool:
    if var.get("opaque"):
        return True
    ts = str(var.get("matlab_type_system") or "").lower()
    if ts == "mcos":
        return True
    if var.get("scipy_envelope") or mat_fieldnames_are_envelope(var.get("fieldnames")):
        return True
    return False


def _mat_experiment_table_usable(var: dict[str, Any]) -> bool:
    if _mat_variable_opaque_table(var):
        return False
    return _mat_variable_usable(var)


def _mat_variable_usable(var: dict[str, Any]) -> bool:
    if var.get("opaque") or var.get("scipy_envelope"):
        return False
    if str(var.get("matlab_type_system") or "").lower() == "mcos":
        return False
    raw_fields = list(var.get("fieldnames") or [])
    fields = non_envelope_mat_fieldnames(raw_fields)
    if raw_fields and not fields:
        return False
    if str(var.get("matlab_class") or "").lower() in {"table", "timetable"} and not fields:
        return False
    if fields:
        return True
    dtype = str(var.get("dtype") or "")
    if "MatlabOpaque" in dtype or dtype in {"object", "O"}:
        return False
    return var.get("shape") is not None


def _filters_full_history_reason(status: str, probe: dict[str, Any]) -> str:
    mat_name = FILTERS_FULL_HISTORY_MAT_NAME
    if status == "missing":
        return (
            f"{mat_name} was not found in the filters raw directory. "
            "filters_full_history stays disabled; training uses filters_censored CSV only."
        )
    if status == "unreadable":
        vars_ = probe.get("variables") or []
        opaque = [v for v in vars_ if v.get("opaque")]
        classes = [v.get("matlab_class") for v in opaque if v.get("matlab_class")]
        err = probe.get("error")
        if opaque:
            cls = classes[0] if classes else "object"
            names = [v.get("name") for v in opaque]
            return (
                f"{mat_name} opened with scipy.io.loadmat but variable(s) {names} are MATLAB "
                f"MCOS opaque ({cls}); table columns are not readable. "
                "filters_full_history stays disabled. CSV remains the training source. "
                "Do not invent MATLAB table fields."
            )
        extra = f" load error: {err}." if err else " no usable (non-opaque) variables."
        return (
            f"{mat_name} is present but not readable via scipy.io.loadmat.{extra} "
            "filters_full_history stays disabled."
        )
    names = [
        v.get("name")
        for v in _primary_experiment_tables(probe.get("variables") or [])
        if _mat_experiment_table_usable(v)
    ]
    return (
        f"{mat_name} has scipy-readable array/struct variables {names}. "
        "filters_full_history is still disabled: enabling it needs a verified MATLAB table "
        "export, origin_unit_id grouping, and a separate protocol that is not mixed with "
        "filters_censored. That export work is future / out of scope."
    )


def _reject_full_history_training(cfg: dict[str, Any]) -> None:
    mode = str(cfg.get("mode") or FILTERS_CENSORED_MODE)
    flagged = bool(cfg.get("filters_full_history")) or bool(cfg.get("enable_filters_full_history"))
    if mode != FILTERS_CENSORED_MODE or flagged:
        raise ValueError(
            "filters_full_history training is disabled. "
            f"Refusing mode={mode!r} (flags filters_full_history={cfg.get('filters_full_history')!r}). "
            f"Use mode={FILTERS_CENSORED_MODE} (CSV only)."
        )


def inspect_filters(raw_dir: Path | None = None) -> dict[str, Any]:
    paths = locate_filter_csvs(raw_dir)
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    history = probe_filters_full_history(paths["raw_dir"])
    rec = {
        "train_path": str(paths["train"]),
        "test_path": str(paths["test"]),
        "train_columns": list(train.columns),
        "test_columns": list(test.columns),
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "train_units": int(train["Data_No"].nunique()) if "Data_No" in train.columns else None,
        "test_units": int(test["Data_No"].nunique()) if "Data_No" in test.columns else None,
        "mat_files": [p.name for p in paths["raw_dir"].glob("*.mat")],
        "filters_full_history_status": history["filters_full_history_status"],
        "filters_full_history_reason": history["filters_full_history_reason"],
        "filters_full_history_enabled": FILTERS_FULL_HISTORY_ENABLED,
        "mat_probe": history["mat_probe"],
        "notes": [
            "CSV is the primary format; MATLAB MCOS table objects are not decoded into columns.",
            history["filters_full_history_reason"],
            FILTER_TIME_SCALE_WARNING,
        ],
    }
    return rec


def _require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {missing}. Actual columns={list(df.columns)}"
        )


def filter_time_scale_meta(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Documented Time/RUL scale. Factor stays 60 unless a new data version is cut."""
    cfg = cfg or {}
    factor = float(cfg.get("time_to_seconds", FILTER_TIME_TO_SECONDS))
    original = str(cfg.get("original_time_unit") or "minutes")
    verified = bool(cfg.get("time_scale_verified", False))
    note = (
        FILTER_TIME_SCALE_WARNING
        if not verified
        else (
            f"Time/RUL converted with time_to_seconds={factor} "
            f"(original unit {original}); scale marked verified in config."
        )
    )
    return {
        "time_to_seconds": factor,
        "original_time_unit": original,
        "time_scale_verified": verified,
        "time_unit_note": note,
        "time_scale_sources": list(FILTER_TIME_SCALE_SOURCES),
    }


def _time_to_seconds(time_original: np.ndarray, factor: float) -> np.ndarray:
    # factor is the frozen YAML coefficient (60), not a confirmed minute clock.
    return np.asarray(time_original, dtype=np.float64) * float(factor)


def extract_filters_tables(
    cfg: dict[str, Any],
    raw_dir: Path | None = None,
    progress: ProgressFn | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _reject_full_history_training(cfg)
    paths = locate_filter_csvs(raw_dir)
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    _require_columns(
        train,
        ["Data_No", "Differential_pressure", "Flow_rate", "Time", "Dust_feed", "Dust"],
        paths["train"],
    )
    _require_columns(
        test,
        ["Data_No", "Differential_pressure", "Flow_rate", "Time", "Dust_feed", "Dust"],
        paths["test"],
    )
    factor = float(cfg.get("time_to_seconds", FILTER_TIME_TO_SECONDS))
    limit = float(cfg.get("pressure_limit_pa", PRESSURE_LIMIT_PA))
    gap_multiplier, sampling_interval_s = filter_gap_params(cfg)

    train_feat = _measurements_from_csv(
        train,
        split_source="author_train",
        time_factor=factor,
        gap_multiplier=gap_multiplier,
        sampling_interval_s=sampling_interval_s,
    )
    test_feat = _measurements_from_csv(
        test,
        split_source="author_test",
        time_factor=factor,
        gap_multiplier=gap_multiplier,
        sampling_interval_s=sampling_interval_s,
    )

    train_units = _train_unit_table(train_feat, limit)
    test_units = _test_unit_table(test, test_feat, limit, time_factor=factor)

    if progress:
        progress(
            "filters_units",
            {
                "n_train_units": int(train_units.shape[0]),
                "n_test_units": int(test_units.shape[0]),
                "n_train_events": int(train_units["event_observed"].sum()),
            },
        )
    units = pd.concat([train_units, test_units], ignore_index=True)
    feat = pd.concat([train_feat, test_feat], ignore_index=True)
    return feat, units


def _measurements_from_csv(
    df: pd.DataFrame,
    split_source: str,
    time_factor: float,
    gap_multiplier: float,
    sampling_interval_s: float,
) -> pd.DataFrame:
    # Data_No 1–50 is reused in Train_Data_CSV and Test_Data_CSV for different
    # experiments (lengths / Δp series do not match). Namespaced unit_id is the split key.
    ns = "Train" if split_source == "author_train" else "Test"
    identifiers = pd.to_numeric(df["Data_No"], errors="coerce")
    valid_ids = np.isfinite(identifiers) & (identifiers > 0) & (identifiers == np.floor(identifiers))
    if not valid_ids.all():
        rows = (np.flatnonzero(~valid_ids.to_numpy()) + 2).tolist()
        raise ValueError(f"{ns} CSV has invalid equipment identifiers at source rows {rows[:20]}; admission cannot assign these records to a series.")
    df = df.assign(Data_No=identifiers.astype(int))
    rows = []
    for unit_id, g in df.groupby("Data_No", sort=True):
        g = g.copy()
        for column in ("Time", "Differential_pressure", "Flow_rate", "Dust_feed"):
            g[column] = pd.to_numeric(g[column], errors="coerce")
        order_error = "nonmonotonic_source_time" if (np.diff(g.Time.to_numpy()) < 0).any() else ""
        g = g.sort_values("Time", kind="stable").copy()
        t_s = _time_to_seconds(g["Time"].to_numpy(), time_factor)
        dp = g["Differential_pressure"].to_numpy(dtype=np.float64)
        flow = g["Flow_rate"].to_numpy(dtype=np.float64)
        feed = g["Dust_feed"].to_numpy(dtype=np.float64)
        dust = g["Dust"].astype(str).to_numpy()
        d_dp = causal_delta(dp)
        dt = np.zeros_like(t_s)
        if t_s.size > 1:
            dt[1:] = np.diff(t_s)
        # Causal prefix median — same rule as inference / build_windows.
        # causal=False (full-file median) is diagnostics-only; do not store it.
        gap_before = gap_before_from_delta_t(
            dt,
            gap_multiplier=gap_multiplier,
            sampling_interval_s=sampling_interval_s,
            causal=True,
        )
        for i in range(len(g)):
            rows.append(
                {
                    "_quality_errors": order_error or ("missing_dust" if pd.isna(g["Dust"].iloc[i]) or not str(g["Dust"].iloc[i]).strip() else ""),
                    "dataset_id": "filters",
                    "unit_id": f"{ns}_{int(unit_id)}",
                    "author_data_no": int(unit_id),
                    "regime_id": str(dust[i]),
                    "timestamp_s": float(t_s[i]),
                    "time_original": float(g["Time"].iloc[i]),
                    "operating_age_s": float(t_s[i]),
                    "delta_t_s": float(dt[i]),
                    "differential_pressure": float(dp[i]),
                    "delta_pressure": float(d_dp[i]),
                    "flow_rate": float(flow[i]),
                    "dust_feed": float(feed[i]),
                    "dust": str(dust[i]),
                    "gap_before": bool(gap_before[i]),
                    "author_split": split_source,
                }
            )
    return pd.DataFrame(rows)


def _train_unit_table(feat: pd.DataFrame, limit: float) -> pd.DataFrame:
    recs = []
    train = feat[feat["author_split"] == "author_train"]
    for unit_id, g in train.groupby("unit_id"):
        g = g.sort_values("timestamp_s")
        dp_max = float(g["differential_pressure"].max())
        over = g[g["differential_pressure"] > limit]
        event = int(len(over) > 0)
        if event:
            event_time = float(over.iloc[0]["timestamp_s"])
            source = f"first_sample_differential_pressure_gt_{limit}"
        else:
            event_time = float("nan")
            source = "right_censored_preventive_replacement"
        recs.append(
            {
                "dataset_id": "filters",
                "unit_id": unit_id,
                "author_data_no": int(g["author_data_no"].iloc[0]),
                "origin_unit_id": int(g["author_data_no"].iloc[0]),
                "regime_id": str(g["dust"].iloc[0]),
                "n_measurements": int(len(g)),
                "observation_end_s": float(g["timestamp_s"].max()),
                "event_observed": event,
                "event_time_s": event_time,
                "event_source": source,
                "dp_max": dp_max,
                "author_split": "author_train",
                "dust_feed": float(g["dust_feed"].iloc[0]),
                "dust": str(g["dust"].iloc[0]),
            }
        )
    return pd.DataFrame(recs)


def _test_unit_table(
    test_raw: pd.DataFrame,
    feat: pd.DataFrame,
    limit: float,
    time_factor: float,
) -> pd.DataFrame:
    if "RUL" not in test_raw.columns:
        raise ValueError(
            f"Test CSV has no RUL column. Actual columns={list(test_raw.columns)}. "
            "Official prefix-end RUL is required for evaluation, not for model inputs."
        )
    recs = []
    feat_test = feat[feat["author_split"] == "author_test"]
    for data_no, g_raw in test_raw.groupby("Data_No", sort=True):
        g_raw = g_raw.sort_values("Time")
        last = g_raw.iloc[-1]
        unit_id = f"Test_{int(data_no)}"
        g = feat_test[feat_test["unit_id"] == unit_id].sort_values("timestamp_s")
        prefix_end_s = float(g["timestamp_s"].max())
        official_rul_original = pd.to_numeric(last["RUL"], errors="coerce")
        official_rul_s = official_rul_original * time_factor
        dp_max = float(g["differential_pressure"].max())
        recs.append(
            {
                "dataset_id": "filters",
                "unit_id": unit_id,
                "author_data_no": int(data_no),
                "origin_unit_id": int(data_no),
                "regime_id": str(g["dust"].iloc[0]),
                "n_measurements": int(len(g)),
                "observation_end_s": prefix_end_s,
                "event_observed": 0,
                "event_time_s": float("nan"),
                "event_source": "official_RUL_at_prefix_end_evaluation_only",
                "official_rul_at_prefix_end_s": official_rul_s,
                "official_rul_at_prefix_end_original": official_rul_original,
                "dp_max": dp_max,
                "observed_limit_reached": int(dp_max > limit),
                "author_split": "author_test",
                "dust_feed": float(g["dust_feed"].iloc[0]),
                "dust": str(g["dust"].iloc[0]),
            }
        )
    return pd.DataFrame(recs)
