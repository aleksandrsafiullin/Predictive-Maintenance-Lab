from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from pdm.features import causal_delta
from pdm.paths import dataset_raw

ProgressFn = Callable[[str, dict[str, Any]], None]

PRESSURE_LIMIT_PA = 600.0
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


def inspect_filters(raw_dir: Path | None = None) -> dict[str, Any]:
    paths = locate_filter_csvs(raw_dir)
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
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
        "notes": [
            "CSV is the primary format; MATLAB tables in *.mat are MCOS opaque objects "
            "and are not readable via scipy.io.loadmat.",
            "filters_full_history is disabled: Train_Data_Uncensored.mat content/origin "
            "could not be verified with scipy.",
        ],
    }
    return rec


def _require_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {missing}. Actual columns={list(df.columns)}"
        )


def _time_to_seconds(time_original: np.ndarray, factor: float) -> np.ndarray:
    return np.asarray(time_original, dtype=np.float64) * float(factor)


def extract_filters_tables(
    cfg: dict[str, Any],
    raw_dir: Path | None = None,
    progress: ProgressFn | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    factor = float(cfg.get("time_to_seconds", 60.0))
    limit = float(cfg.get("pressure_limit_pa", PRESSURE_LIMIT_PA))

    train_feat = _measurements_from_csv(train, split_source="author_train", time_factor=factor)
    test_feat = _measurements_from_csv(test, split_source="author_test", time_factor=factor)

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


def _measurements_from_csv(df: pd.DataFrame, split_source: str, time_factor: float) -> pd.DataFrame:
    rows = []
    for unit_id, g in df.groupby("Data_No", sort=True):
        g = g.sort_values("Time").copy()
        t_s = _time_to_seconds(g["Time"].to_numpy(), time_factor)
        dp = g["Differential_pressure"].to_numpy(dtype=np.float64)
        flow = g["Flow_rate"].to_numpy(dtype=np.float64)
        feed = g["Dust_feed"].to_numpy(dtype=np.float64)
        dust = g["Dust"].astype(str).to_numpy()
        d_dp = causal_delta(dp)
        dt = np.zeros_like(t_s)
        if t_s.size > 1:
            dt[1:] = np.diff(t_s)
        med_dt = float(np.median(dt[1:])) if dt.size > 1 else 0.0
        gap_before = np.zeros(len(g), dtype=bool)
        if med_dt > 0:
            gap_before[1:] = dt[1:] > 3.0 * med_dt
        for i in range(len(g)):
            rows.append(
                {
                    "dataset_id": "filters",
                    "unit_id": f"{'Train' if split_source == 'author_train' else 'Test'}_{int(unit_id)}",
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
        official_rul_original = float(last["RUL"])
        official_rul_s = official_rul_original * time_factor
        event_time_s = prefix_end_s + official_rul_s
        dp_max = float(g["differential_pressure"].max())
        recs.append(
            {
                "dataset_id": "filters",
                "unit_id": unit_id,
                "author_data_no": int(data_no),
                "regime_id": str(g["dust"].iloc[0]),
                "n_measurements": int(len(g)),
                "observation_end_s": prefix_end_s,
                "event_observed": 0,
                "event_time_s": event_time_s,
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
