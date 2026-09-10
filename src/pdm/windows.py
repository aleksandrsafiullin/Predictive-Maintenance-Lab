from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FORBIDDEN_FEATURE_NAMES = {
    "rul",
    "rul_s",
    "actual_rul_s",
    "event",
    "event_observed",
    "event_time_s",
    "observation_end_s",
    "split",
    "part",
    "life_fraction",
    "percent_life",
    "failure",
    "official_rul_at_prefix_end_s",
    "unit_id",
}


def build_windows(
    features: pd.DataFrame,
    units: pd.DataFrame,
    history_length: int,
    dataset_id: str,
) -> pd.DataFrame:
    """Fixed-length windows. No future padding. No crossing units or time gaps."""
    if history_length < 1:
        raise ValueError("history_length must be >= 1")
    unit_meta = units.set_index("unit_id")
    records: list[dict[str, Any]] = []
    for unit_id, g in features.groupby("unit_id", sort=False):
        g = g.sort_values("timestamp_s").reset_index(drop=True)
        meta = unit_meta.loc[unit_id]
        event_time = meta.get("event_time_s")
        observed_end = float(meta["observation_end_s"])
        event_observed = int(meta.get("event_observed", 0))
        n = len(g)
        gap = g["gap_before"].to_numpy() if "gap_before" in g.columns else np.zeros(n, dtype=bool)
        ts = g["timestamp_s"].to_numpy(dtype=np.float64)
        for end in range(history_length - 1, n):
            start = end - history_length + 1
            if gap[start + 1 : end + 1].any():
                continue
            t = float(ts[end])
            rec: dict[str, Any] = {
                "dataset_id": dataset_id,
                "unit_id": str(unit_id),
                "end_index": int(end),
                "start_index": int(start),
                "timestamp_s": t,
                "input_until_s": t,
            }
            if dataset_id == "bearings":
                et = float(event_time)
                if not np.isfinite(et) or t >= et:
                    continue
                rec["target_rul_s"] = et - t
                rec["event"] = 1
                rec["duration_s"] = et - t
            else:
                duration = observed_end - t
                if duration <= 0:
                    continue
                rec["duration_s"] = float(duration)
                rec["event"] = int(event_observed)
                if event_observed and np.isfinite(event_time) and t < float(event_time):
                    rec["target_rul_s"] = float(event_time) - t
                else:
                    rec["target_rul_s"] = np.nan
            records.append(rec)
    return pd.DataFrame.from_records(records)


def window_matrix(
    features: pd.DataFrame,
    window_row: pd.Series,
    feature_cols: list[str],
) -> np.ndarray:
    g = features[features["unit_id"] == window_row["unit_id"]].sort_values("timestamp_s")
    sl = g.iloc[int(window_row["start_index"]) : int(window_row["end_index"]) + 1]
    arr = sl[feature_cols].to_numpy(dtype=np.float32)
    if arr.shape[0] != int(window_row["end_index"]) - int(window_row["start_index"]) + 1:
        raise RuntimeError("Window row count mismatch")
    return arr
