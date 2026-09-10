from __future__ import annotations

import numpy as np

EPS = 1e-12


def time_domain_features(x: np.ndarray, prefix: str) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {
            f"{prefix}_rms": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_abs_peak": np.nan,
            f"{prefix}_peak_to_peak": np.nan,
            f"{prefix}_crest_factor": np.nan,
            f"{prefix}_kurtosis": np.nan,
        }
    mean = float(np.mean(x))
    rms = float(np.sqrt(np.mean(x * x)))
    std = float(np.std(x, ddof=0))
    abs_peak = float(np.max(np.abs(x)))
    peak_to_peak = float(np.ptp(x))
    crest = abs_peak / rms if rms > EPS else np.nan
    if std > EPS:
        z = (x - mean) / std
        kurtosis = float(np.mean(z ** 4))
    else:
        kurtosis = np.nan
    return {
        f"{prefix}_rms": rms,
        f"{prefix}_std": std,
        f"{prefix}_abs_peak": abs_peak,
        f"{prefix}_peak_to_peak": peak_to_peak,
        f"{prefix}_crest_factor": crest,
        f"{prefix}_kurtosis": kurtosis,
    }


def spectral_band_energy(
    x: np.ndarray,
    fs: float,
    edges_hz: list[float],
    prefix: str,
) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    n = x.size
    out: dict[str, float] = {}
    n_bands = max(0, len(edges_hz) - 1)
    if n < 4 or n_bands == 0:
        for i in range(n_bands):
            out[f"{prefix}_band_{i}"] = np.nan
        return out
    spec = np.fft.rfft(x)
    power = (spec.real ** 2 + spec.imag ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    for i in range(n_bands):
        lo, hi = edges_hz[i], edges_hz[i + 1]
        mask = (freqs >= lo) & (freqs < hi if i < n_bands - 1 else freqs <= hi)
        out[f"{prefix}_band_{i}"] = float(np.sum(power[mask]))
    return out


def apply_log1p_columns(df, columns: list[str]):
    for c in columns:
        if c in df.columns:
            vals = df[c].to_numpy(dtype=np.float64)
            vals = np.where(np.isfinite(vals) & (vals >= 0), np.log1p(vals), vals)
            df[c] = vals
    return df


def causal_delta(values: np.ndarray) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float64)
    if values.size > 1:
        out[1:] = values[1:] - values[:-1]
    return out


def causal_linear_slope(x: np.ndarray, y: np.ndarray) -> float | None:
    """Slope of y vs x using only the provided (already past-only) points."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:
        return None
    vx = np.var(x)
    if vx <= EPS:
        return None
    slope = float(np.cov(x, y, bias=True)[0, 1] / vx)
    if not np.isfinite(slope):
        return None
    return slope
