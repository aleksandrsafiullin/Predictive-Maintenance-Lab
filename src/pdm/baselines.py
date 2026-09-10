from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from pdm.features import causal_linear_slope


def age_only_baseline_rul(
    operating_age_s: float,
    regime_id: str | None,
    train_units: pd.DataFrame,
) -> float:
    """Median train duration of the same regime minus current age, floored at 0."""
    pool = train_units
    if regime_id is not None and "regime_id" in train_units.columns:
        same = train_units[train_units["regime_id"] == regime_id]
        if len(same):
            pool = same
    durations = pool["observation_end_s"].to_numpy(dtype=np.float64)
    if durations.size == 0:
        return 0.0
    median_life = float(np.median(durations))
    return max(median_life - float(operating_age_s), 0.0)


def filter_trend_baseline(
    timestamps_s: np.ndarray,
    pressures: np.ndarray,
    limit_pa: float = 600.0,
) -> dict[str, Any]:
    """Linear fit on the available past window; time to 600 Pa if slope > 0."""
    slope = causal_linear_slope(np.asarray(timestamps_s), np.asarray(pressures))
    last_p = float(pressures[-1])
    last_t = float(timestamps_s[-1])
    if slope is None or slope <= 0 or not math.isfinite(slope):
        return {
            "predicted_rul_s": None,
            "status": "No finite trend estimate",
            "slope": slope,
        }
    remaining_p = limit_pa - last_p
    if remaining_p <= 0:
        return {"predicted_rul_s": 0.0, "status": "ok", "slope": slope}
    span = max(float(timestamps_s[-1] - timestamps_s[0]), 1.0)
    # Tiny slopes explode into invented huge RULs; treat as unreliable (spec).
    if slope * span < 1.0:  # < 1 Pa of explained rise over the window
        return {"predicted_rul_s": None, "status": "No finite trend estimate", "slope": slope}
    rul = remaining_p / slope
    if (not math.isfinite(rul)) or rul < 0 or rul > 20.0 * span:
        return {"predicted_rul_s": None, "status": "No finite trend estimate", "slope": slope}
    return {"predicted_rul_s": float(rul), "status": "ok", "slope": slope}
