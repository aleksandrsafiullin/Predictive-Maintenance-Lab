from __future__ import annotations

from collections.abc import Sequence
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
    "official_rul_at_prefix_end_original",
    "unit_id",
    "origin_unit_id",
}

# Raw numeric measurement columns (subtask 1 feature bases, excluding one-hots).
BEARINGS_RAW_NUMERIC_COLUMNS: tuple[str, ...] = (
    "operating_age_s",
    "rpm",
    "load_kn",
    "horizontal_rms",
    "horizontal_std",
    "horizontal_abs_peak",
    "horizontal_peak_to_peak",
    "horizontal_crest_factor",
    "horizontal_kurtosis",
    "vertical_rms",
    "vertical_std",
    "vertical_abs_peak",
    "vertical_peak_to_peak",
    "vertical_crest_factor",
    "vertical_kurtosis",
    "horizontal_band_0",
    "horizontal_band_1",
    "horizontal_band_2",
    "horizontal_band_3",
    "vertical_band_0",
    "vertical_band_1",
    "vertical_band_2",
    "vertical_band_3",
)
FILTERS_RAW_NUMERIC_COLUMNS: tuple[str, ...] = (
    "operating_age_s",
    "delta_t_s",
    "differential_pressure",
    "delta_pressure",
    "flow_rate",
    "dust_feed",
)


def raw_numeric_columns(dataset_id: str | None) -> tuple[str, ...]:
    if dataset_id == "bearings":
        return BEARINGS_RAW_NUMERIC_COLUMNS
    if dataset_id == "filters":
        return FILTERS_RAW_NUMERIC_COLUMNS
    return ()


# Filter Δt gaps. sampling_interval_s = CSV Time step 0.1 (assumed minutes) * time_to_seconds 60.
FILTER_GAP_MULTIPLIER = 3.0
FILTER_SAMPLING_INTERVAL_S = 6.0
GAP_RULE_VERSION = "causal_v1"


def _dataset_id_from_prefix(prefix: pd.DataFrame) -> str | None:
    if "dataset_id" not in prefix.columns or prefix.empty:
        return None
    val = prefix["dataset_id"].iloc[-1]
    if val is None or (isinstance(val, float) and not np.isfinite(val)):
        return None
    text = str(val)
    return None if text == "nan" else text


def filter_gap_params(cfg: dict[str, Any] | None = None) -> tuple[float, float]:
    """``(gap_multiplier, sampling_interval_s)`` from a cfg dict.

    Pass the dataset/run snapshot cfg at prepare/train. ``cfg is None`` loads
    live ``configs/filters.yaml`` — inference should prefer saved preprocessor
    / run values and only call this as a fallback for old snapshots.
    """
    if cfg is None:
        from pdm.config import load_dataset_config

        cfg = load_dataset_config("filters")
    gap = dict(cfg.get("gap") or {})
    k = float(gap.get("gap_multiplier", gap.get("max_dt_factor", FILTER_GAP_MULTIPLIER)))
    samp = float(gap.get("sampling_interval_s", FILTER_SAMPLING_INTERVAL_S))
    if not np.isfinite(k) or k <= 0.0:
        raise ValueError(f"gap_multiplier must be finite and > 0, got {k}")
    if not np.isfinite(samp) or samp <= 0.0:
        raise ValueError(f"sampling_interval_s must be finite and > 0, got {samp}")
    return k, samp


def resolve_filter_gap_params(
    *,
    gap_multiplier: float | None = None,
    sampling_interval_s: float | None = None,
    cfg: dict[str, Any] | None = None,
    allow_yaml_fallback: bool = True,
) -> tuple[float, float]:
    """Prefer explicit/saved values; yaml only if a snapshot field is missing."""
    if gap_multiplier is not None and sampling_interval_s is not None:
        return filter_gap_params(
            {
                "gap": {
                    "gap_multiplier": gap_multiplier,
                    "sampling_interval_s": sampling_interval_s,
                }
            }
        )
    if allow_yaml_fallback:
        yk, ys = filter_gap_params(cfg)
        return (
            float(gap_multiplier) if gap_multiplier is not None else yk,
            float(sampling_interval_s) if sampling_interval_s is not None else ys,
        )
    return (
        FILTER_GAP_MULTIPLIER if gap_multiplier is None else float(gap_multiplier),
        FILTER_SAMPLING_INTERVAL_S if sampling_interval_s is None else float(sampling_interval_s),
    )


def gap_before_from_delta_t(
    delta_t_s: np.ndarray,
    *,
    gap_multiplier: float = FILTER_GAP_MULTIPLIER,
    sampling_interval_s: float = FILTER_SAMPLING_INTERVAL_S,
    causal: bool = True,
) -> np.ndarray:
    """Boolean ``gap_before`` from Δt.

    Causal (prepare parquet, ``build_windows``, inference): ``gap[i] =
    dt[i] > k * median(dt[1:i])``. Empty or non-positive past median falls
    back to ``sampling_interval_s``. Row 0 is never a gap. Future Δt never
    enter the threshold at i.

    Full-file (``causal=False``): one median over ``dt[1:]`` for every row.
    Diagnostics only — must not decide train / val / eval / replay eligibility.
    """
    dt = np.asarray(delta_t_s, dtype=np.float64).reshape(-1)
    n = int(dt.size)
    gap = np.zeros(n, dtype=bool)
    k = float(gap_multiplier)
    fallback = float(sampling_interval_s)
    if n < 2 or not np.isfinite(k) or k <= 0.0:
        return gap
    if not causal:
        past = dt[1:]
        finite = past[np.isfinite(past)]
        med = float(np.median(finite)) if finite.size else float("nan")
        ref = med if np.isfinite(med) and med > 0.0 else fallback
        if not np.isfinite(ref) or ref <= 0.0:
            return gap
        rest = dt[1:]
        gap[1:] = np.isfinite(rest) & (rest > k * ref)
        return gap
    refs = np.full(n, fallback, dtype=np.float64)
    if n >= 3:
        exp = pd.Series(dt[1:], dtype=np.float64).expanding(min_periods=1).median().to_numpy()
        refs[2:] = exp[: n - 2]
    bad = ~np.isfinite(refs) | (refs <= 0.0)
    refs[bad] = fallback
    if not np.isfinite(fallback) or fallback <= 0.0:
        return gap
    rest = dt[1:]
    gap[1:] = np.isfinite(rest) & (rest > k * refs[1:])
    return gap


def recompute_filter_gap_before(
    prefix: pd.DataFrame,
    *,
    gap_multiplier: float | None = None,
    sampling_interval_s: float | None = None,
    causal: bool = True,
) -> pd.DataFrame:
    """Copy of ``prefix`` with filter ``gap_before`` from timestamps 0..t only."""
    if prefix.empty or "timestamp_s" not in prefix.columns:
        return prefix.copy()
    out = prefix.sort_values("timestamp_s")
    ts = out["timestamp_s"].to_numpy(dtype=np.float64)
    dt = np.zeros(len(out), dtype=np.float64)
    if ts.size > 1:
        dt[1:] = np.diff(ts)
    k = FILTER_GAP_MULTIPLIER if gap_multiplier is None else float(gap_multiplier)
    samp = FILTER_SAMPLING_INTERVAL_S if sampling_interval_s is None else float(sampling_interval_s)
    gaps = gap_before_from_delta_t(
        dt, gap_multiplier=k, sampling_interval_s=samp, causal=causal
    )
    out = out.copy()
    out["delta_t_s"] = dt
    out["gap_before"] = gaps
    return out


def window_timestamp_reason(timestamps: np.ndarray) -> str:
    """``valid_history_window`` reason, or ``\"\"`` if finite and strictly increasing."""
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if ts.size == 0 or not np.isfinite(ts).all():
        return "non_finite_timestamps"
    if ts.size >= 2 and not np.all(np.diff(ts) > 0):
        return "timestamps_not_strictly_increasing"
    return ""


def diagnostic_fullfile_gap_marker_count(
    features: pd.DataFrame,
    *,
    gap_multiplier: float = FILTER_GAP_MULTIPLIER,
    sampling_interval_s: float = FILTER_SAMPLING_INTERVAL_S,
) -> int:
    """Full-file-median gap markers. Diagnostics only; never eligibility."""
    if features.empty or "timestamp_s" not in features.columns:
        return 0
    n = 0
    for _, g in features.groupby("unit_id", sort=False):
        ts = g.sort_values("timestamp_s")["timestamp_s"].to_numpy(dtype=np.float64)
        dt = np.zeros(ts.size, dtype=np.float64)
        if ts.size > 1:
            dt[1:] = np.diff(ts)
        n += int(
            gap_before_from_delta_t(
                dt,
                gap_multiplier=gap_multiplier,
                sampling_interval_s=sampling_interval_s,
                causal=False,
            ).sum()
        )
    return n


def _unit_gap_flags(
    g: pd.DataFrame,
    dataset_id: str | None,
    *,
    gap_multiplier: float | None = None,
    sampling_interval_s: float | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sorted unit rows and eligibility ``gap_before``.

    Filters recompute causal flags from timestamps (stored parquet is ignored).
    Bearings keep file-index ``gap_before``; no filter-style Δt median.
    """
    out = g.sort_values("timestamp_s").reset_index(drop=True)
    if dataset_id == "filters":
        out = recompute_filter_gap_before(
            out,
            gap_multiplier=gap_multiplier,
            sampling_interval_s=sampling_interval_s,
            causal=True,
        ).reset_index(drop=True)
    n = len(out)
    gap = out["gap_before"].to_numpy() if "gap_before" in out.columns else np.zeros(n, dtype=bool)
    return out, np.asarray(gap, dtype=bool)


def valid_history_window(
    prefix: pd.DataFrame,
    history_length: int,
    *,
    required_columns: Sequence[str] | None = None,
    dataset_id: str | None = None,
    gap_multiplier: float | None = None,
    sampling_interval_s: float | None = None,
) -> tuple[bool, str]:
    """Whether the last ``history_length`` rows are a valid causal window.

    Returns ``(ok, reason)``. ``reason`` is empty when ``ok`` is True.

    Checks, matching ``build_windows``:
    - prefix length >= ``history_length`` (no future padding)
    - no ``gap_before`` in the window interior: ``gap[start+1:end+1].any()``
      on the prefix sorted by ``timestamp_s`` (``gap_before`` on the first
      row of the window is allowed). Missing ``gap_before`` means no gaps
      for bearings. Filters recompute ``gap_before`` on this prefix
      (``median(Δt[1:i])``, not the full trajectory).
    - ``timestamp_s`` in the window is finite and strictly increasing.
      Equal timestamps are rejected (not unique / not monotonic).
    - required raw numeric columns exist (defaults from ``dataset_id``).
      NaN/Inf in those measurements are allowed: ``transform_frame`` imputes
      them with saved train ``fill_values``. Non-finite timestamps still fail.
    """
    if history_length < 1:
        raise ValueError("history_length must be >= 1")
    if len(prefix) < history_length:
        return False, "insufficient_length"
    if "timestamp_s" not in prefix.columns:
        return False, "missing_timestamp_s"

    g = prefix.sort_values("timestamp_s").reset_index(drop=True)
    ds = dataset_id or _dataset_id_from_prefix(g)
    g, gap = _unit_gap_flags(
        g,
        ds,
        gap_multiplier=gap_multiplier,
        sampling_interval_s=sampling_interval_s,
    )
    n = len(g)
    start = n - history_length
    end = n - 1
    window = g.iloc[start : end + 1]
    if np.asarray(gap[start + 1 : end + 1]).any():
        return False, "gap_in_window"

    ts_reason = window_timestamp_reason(window["timestamp_s"].to_numpy(dtype=np.float64))
    if ts_reason:
        return False, ts_reason

    cols = list(required_columns) if required_columns is not None else list(
        raw_numeric_columns(dataset_id or _dataset_id_from_prefix(g))
    )
    missing = [c for c in cols if c not in window.columns]
    if missing:
        return False, "missing_required_columns"
    return True, ""


def build_windows(
    features: pd.DataFrame,
    units: pd.DataFrame,
    history_length: int,
    dataset_id: str,
    *,
    gap_multiplier: float | None = None,
    sampling_interval_s: float | None = None,
) -> pd.DataFrame:
    """Fixed-length windows. No future padding. No crossing units or time gaps."""
    if history_length < 1:
        raise ValueError("history_length must be >= 1")
    unit_meta = units.set_index("unit_id")
    records: list[dict[str, Any]] = []
    for unit_id, g in features.groupby("unit_id", sort=False):
        g, gap = _unit_gap_flags(
            g,
            dataset_id,
            gap_multiplier=gap_multiplier,
            sampling_interval_s=sampling_interval_s,
        )
        meta = unit_meta.loc[unit_id]
        event_time = meta.get("event_time_s")
        observed_end = float(meta["observation_end_s"])
        event_observed = int(meta.get("event_observed", 0))
        n = len(g)
        ts = g["timestamp_s"].to_numpy(dtype=np.float64)
        for end in range(history_length - 1, n):
            start = end - history_length + 1
            if gap[start + 1 : end + 1].any():
                continue
            if window_timestamp_reason(ts[start : end + 1]):
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
                duration_s = et - t
                rec["target_rul_s"] = duration_s
                rec["event"] = 1
                rec["duration_s"] = duration_s
            elif event_observed:
                et = float(event_time)
                if not np.isfinite(et) or t >= et:
                    continue
                duration_s = et - t
                rec["duration_s"] = float(duration_s)
                rec["event"] = 1
                rec["target_rul_s"] = float(duration_s)
            else:
                if t >= observed_end:
                    continue
                duration_s = observed_end - t
                if duration_s <= 0:
                    continue
                rec["duration_s"] = float(duration_s)
                rec["event"] = 0
                rec["target_rul_s"] = np.nan
            records.append(rec)
    return pd.DataFrame.from_records(records)


def _empty_split_window_counts() -> dict[str, int]:
    return {
        "n_units": 0,
        "n_measurements": 0,
        "eligible_windows": 0,
        "excluded_gap": 0,
        "excluded_post_event": 0,
        "excluded_insufficient_length": 0,
        "excluded_timestamps": 0,
    }


def count_window_eligibility(
    features: pd.DataFrame,
    units: pd.DataFrame,
    history_length: int,
    dataset_id: str,
    split: dict[str, Any] | None = None,
    *,
    gap_multiplier: float | None = None,
    sampling_interval_s: float | None = None,
) -> dict[str, Any]:
    """Classify every measurement as an eligible window end or an exclusion.

    Reasons follow ``build_windows`` skip order: insufficient length, gap in
    the window interior, non-finite / non-unique / non-monotonic timestamps,
    then post-event (``t >= event_time_s`` when observed or bearings;
    ``t >= observation_end_s`` when censored). Does not allocate window
    tensors. Counts are for ``history_length`` at prepare time.
    """
    if history_length < 1:
        raise ValueError("history_length must be >= 1")
    split = split or {}
    by_split = {part: _empty_split_window_counts() for part in ("train", "validation", "test")}
    part_of: dict[str, str] = {}
    for part in ("train", "validation", "test"):
        ids = [str(u) for u in (split.get(part) or [])]
        by_split[part]["n_units"] = int(len(ids))
        for uid in ids:
            part_of[uid] = part

    totals = {
        "eligible": 0,
        "gap": 0,
        "post_event": 0,
        "insufficient_length": 0,
        "timestamps": 0,
    }
    n_candidates = 0
    if features.empty or units.empty:
        return _window_count_payload(history_length, n_candidates, totals, by_split)

    unit_meta = units.set_index("unit_id")
    for unit_id, g in features.groupby("unit_id", sort=False):
        g, gap = _unit_gap_flags(
            g,
            dataset_id,
            gap_multiplier=gap_multiplier,
            sampling_interval_s=sampling_interval_s,
        )
        meta = unit_meta.loc[unit_id]
        n = len(g)
        n_candidates += n
        ts = g["timestamp_s"].to_numpy(dtype=np.float64)
        event_time = meta.get("event_time_s")
        observed_end = float(meta["observation_end_s"])
        event_observed = int(meta.get("event_observed", 0))
        local = {
            "eligible": 0,
            "gap": 0,
            "post_event": 0,
            "insufficient_length": 0,
            "timestamps": 0,
        }
        for end in range(n):
            if end < history_length - 1:
                local["insufficient_length"] += 1
                continue
            start = end - history_length + 1
            if np.asarray(gap[start + 1 : end + 1]).any():
                local["gap"] += 1
                continue
            if window_timestamp_reason(ts[start : end + 1]):
                local["timestamps"] += 1
                continue
            t = float(ts[end])
            if dataset_id == "bearings" or event_observed:
                et = float(event_time) if event_time is not None and pd.notna(event_time) else float("nan")
                if not np.isfinite(et) or t >= et:
                    local["post_event"] += 1
                    continue
            else:
                if t >= observed_end or (observed_end - t) <= 0:
                    local["post_event"] += 1
                    continue
            local["eligible"] += 1
        for key, val in local.items():
            totals[key] += val
        part = part_of.get(str(unit_id))
        if part is None:
            continue
        rec = by_split[part]
        rec["n_measurements"] += n
        rec["eligible_windows"] += local["eligible"]
        rec["excluded_gap"] += local["gap"]
        rec["excluded_post_event"] += local["post_event"]
        rec["excluded_insufficient_length"] += local["insufficient_length"]
        rec["excluded_timestamps"] += local["timestamps"]
    return _window_count_payload(history_length, n_candidates, totals, by_split)


def _window_count_payload(
    history_length: int,
    n_candidates: int,
    totals: dict[str, int],
    by_split: dict[str, dict[str, int]],
) -> dict[str, Any]:
    excluded = {
        "gap": int(totals["gap"]),
        "post_event": int(totals["post_event"]),
        "insufficient_length": int(totals["insufficient_length"]),
        "timestamps": int(totals["timestamps"]),
    }
    return {
        "history_length": int(history_length),
        "n_candidate_ends": int(n_candidates),
        "eligible": int(totals["eligible"]),
        "excluded": excluded,
        "by_split": {
            part: {k: int(v) for k, v in rec.items()} for part, rec in by_split.items()
        },
    }


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
