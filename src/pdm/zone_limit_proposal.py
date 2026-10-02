"""Absolute yellow/red proposal from Training Data early life only."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

EARLY_FRACTION = 0.20
EARLY_MIN_POINTS = 5
MIN_EARLY_VALUES = 8


def propose_absolute_limits(features: pd.DataFrame, train_unit_ids, direction: str) -> dict:
    """One absolute pair for `direction` from the early-life train pool."""
    if direction not in ("above", "below"):
        raise ValueError("Threshold direction must be above or below")
    pool = _early_pool(features, train_unit_ids)
    n = int(pool.size)
    if n < MIN_EARLY_VALUES:
        return {
            "ok": False,
            "direction": direction,
            "n": n,
            "reason": (
                "Need at least "
                f"{MIN_EARLY_VALUES} finite Training Data values in the early-life window. "
                f"This snapshot has {n}."
            ),
        }
    yellow, red = _pair_from_pool(pool, direction)
    return {
        "ok": True,
        "direction": direction,
        "yellow": yellow,
        "red": red,
        "n": n,
        "early_fraction": EARLY_FRACTION,
        "early_min_points": EARLY_MIN_POINTS,
    }


def _early_pool(features: pd.DataFrame, train_unit_ids) -> np.ndarray:
    ids = sorted(set(train_unit_ids))
    if not ids:
        return np.empty(0, dtype=np.float64)
    work = features.loc[:, ["unit_id", "timestamp_s", "signal"]]
    chunks: list[np.ndarray] = []
    for unit_id in ids:
        rows = work.loc[work["unit_id"] == unit_id]
        if rows.empty:
            continue
        timestamp_s = pd.to_numeric(rows["timestamp_s"], errors="coerce").to_numpy(dtype=np.float64)
        signal = pd.to_numeric(rows["signal"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(timestamp_s) & np.isfinite(signal)
        if not finite.any():
            continue
        block = pd.DataFrame({"timestamp_s": timestamp_s[finite], "signal": signal[finite]})
        block = block.sort_values("timestamp_s", kind="mergesort")
        n_i = len(block)
        k_i = min(n_i, max(EARLY_MIN_POINTS, math.ceil(EARLY_FRACTION * n_i)))
        chunks.append(block["signal"].to_numpy(dtype=np.float64)[:k_i])
    if not chunks:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(chunks)


def _pair_from_pool(pool: np.ndarray, direction: str) -> tuple[float, float]:
    m = float(np.quantile(pool, 0.5, method="linear"))
    mad = float(np.median(np.abs(pool - m)))
    robust_sigma = 1.4826 * mad
    min_sep = max(0.5 * robust_sigma, 1e-4 * max(abs(m), 1.0), 1e-6)
    if direction == "above":
        yellow = max(float(np.quantile(pool, 0.90, method="linear")), m)
        red = max(float(np.quantile(pool, 0.99, method="linear")), yellow + min_sep)
    else:
        yellow = min(float(np.quantile(pool, 0.10, method="linear")), m)
        red = min(float(np.quantile(pool, 0.01, method="linear")), yellow - min_sep)
    return float(yellow), float(red)
