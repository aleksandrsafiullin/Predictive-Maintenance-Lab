"""Past-only derived measurements, shared by training and replay."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.training_protocol import fingerprint


def recipe_names(dataset_id, recipe):
    if recipe == "base_v1":
        return []
    if recipe != "degradation_v1":
        raise ValueError(f"Unknown feature recipe: {recipe}")
    if dataset_id == "bearings":
        return [f"{channel}_{suffix}" for channel in ("horizontal", "vertical") for suffix in (
            "log_rms_slope_5", "log_rms_slope_20", "log_peak_slope_20",
            "kurtosis_slope_20", "log_rms_initial_delta")]
    return ["pressure_slope_5", "pressure_slope_20", "pressure_change_20",
            "pressure_headroom", "dust_accumulated_internal"]


def recipe_metadata(recipe):
    return {"name": recipe, "version": 1, "slope_windows": [5, 20],
            "initial_reference_measurements": 5, "pressure_limit_pa": 600.0,
            "dust_integral": "left endpoint, internal seconds, reset at gap"}


def recipe_hash(recipe):
    return fingerprint(recipe_metadata(recipe))


def _slope(t, y, size):
    # Closed-form rolling least squares after removing the segment time origin.
    t = pd.Series(np.asarray(t, float) - float(t[0]))
    y = pd.Series(np.asarray(y, float))
    n = t.rolling(size, min_periods=1).count()
    sx, sy = t.rolling(size, min_periods=1).sum(), y.rolling(size, min_periods=1).sum()
    den = n * (t*t).rolling(size, min_periods=1).sum() - sx*sx
    num = n * (t*y).rolling(size, min_periods=1).sum() - sx*sy
    return np.divide(num, den, out=np.zeros(len(t)), where=np.asarray(den) > 1e-12)


def enrich_features(frame, dataset_id, recipe):
    if recipe == "base_v1":
        return frame.copy()
    names = recipe_names(dataset_id, recipe)
    out = frame.copy()
    if not len(out):
        for name in names:
            out[name] = pd.Series(dtype=float)
        return out
    # Positional assignment also works with duplicate DataFrame index labels.
    out = out.reset_index(drop=True)
    if "unit_id" not in out:
        groups = [(None, out)]
    else:
        groups = out.groupby("unit_id", sort=False)
    for _, unit in groups:
        unit = unit.sort_values("timestamp_s")
        gaps = unit.get("gap_before", pd.Series(False, index=unit.index)).fillna(False).astype(bool)
        for _, part in unit.groupby(gaps.cumsum(), sort=False):
            t = part.timestamp_s.to_numpy(float)
            if np.any(np.diff(t) <= 0) or not np.isfinite(t).all():
                raise ValueError("Feature recipe requires finite increasing timestamps")
            values = {}
            if dataset_id == "bearings":
                for channel in ("horizontal", "vertical"):
                    rms = np.log1p(part[f"{channel}_rms"].to_numpy(float))
                    peak = np.log1p(part[f"{channel}_abs_peak"].to_numpy(float))
                    for size in (5, 20):
                        values[f"{channel}_log_rms_slope_{size}"] = _slope(t, rms, size)
                    values[f"{channel}_log_peak_slope_20"] = _slope(t, peak, 20)
                    values[f"{channel}_kurtosis_slope_20"] = _slope(t, part[f"{channel}_kurtosis"], 20)
                    reference = np.asarray([np.median(rms[:min(i + 1, 5)]) for i in range(len(rms))])
                    values[f"{channel}_log_rms_initial_delta"] = rms - reference
            else:
                dp = part.differential_pressure.to_numpy(float)
                for size in (5, 20):
                    values[f"pressure_slope_{size}"] = _slope(t, dp, size)
                values["pressure_change_20"] = dp - dp[np.maximum(np.arange(len(dp)) - 19, 0)]
                values["pressure_headroom"] = 600.0 - dp
                feed = part.dust_feed.to_numpy(float)
                values["dust_accumulated_internal"] = np.r_[0.0, np.cumsum(feed[:-1] * np.diff(t))]
            for name, value in values.items():
                if not np.isfinite(value).all():
                    raise ValueError(f"Nonfinite derived feature {name}")
                out.loc[part.index, name] = np.asarray(value)
    out.index = frame.index
    out.attrs = frame.attrs.copy()
    return out
