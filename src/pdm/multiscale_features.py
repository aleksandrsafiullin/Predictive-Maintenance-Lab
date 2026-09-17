"""New recipe versions; base_v1 and degradation_v1 retain their original math."""
from __future__ import annotations

import numpy as np

from pdm.feature_recipes import _slope

NEW_RECIPES = {"multiscale_trend_v1", "multiscale_no_age_v1", "multiscale_trend_v2", "multiscale_no_age_v2"}


def names(dataset_id):
    signals = ["horizontal_rms", "vertical_rms"] if dataset_id == "bearings" else ["differential_pressure"]
    return [f"{s}_{kind}_{n}" for s in signals for n in (5, 20, 40, 60) for kind in ("slope", "available")] + ["regime_unknown"]


def enrich(frame, dataset_id, *, version=2):
    out = frame.copy().reset_index(drop=True)
    signals = ["horizontal_rms", "vertical_rms"] if dataset_id == "bearings" else ["differential_pressure"]
    for name in names(dataset_id):
        out[name] = 0.
    groups = out.groupby("unit_id", sort=False) if "unit_id" in out else [(None, out)]
    for _, group in groups:
        group = group.sort_values("timestamp_s")
        gaps = group.get("gap_before", group.timestamp_s * 0).fillna(False).astype(bool)
        if version >= 2:
            for marker in ("quality_gap_before", "maintenance_reset", "segment_reset"):
                if marker in group:
                    gaps |= group[marker].fillna(False).astype(bool)
            if "segment_id" in group:
                gaps |= group.segment_id.ne(group.segment_id.shift())
        for _, segment in group.groupby(gaps.cumsum(), sort=False):
            for signal in signals:
                y = segment[signal].to_numpy(float)
                for n in (5, 20, 40, 60):
                    slope = _slope(segment.timestamp_s.to_numpy(float), y, n)
                    out.loc[segment.index, f"{signal}_slope_{n}"] = np.asarray(slope) if version >= 2 else slope
                    out.loc[segment.index, f"{signal}_available_{n}"] = (np.arange(len(segment)) + 1 >= n).astype(float)
    out.index = frame.index
    out.attrs = frame.attrs.copy()
    return out


def envelope_features(signal, sampling_rate_hz):
    """Completed fragment only: DC removal, Hann window, Hilbert envelope, PSD energy."""
    from scipy.signal import hilbert

    x = np.asarray(signal, float)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all() or sampling_rate_hz <= 0:
        raise ValueError("A complete finite fragment and verified sample rate are required")
    window = np.hanning(len(x))
    envelope = np.abs(hilbert((x - x.mean()) * window))
    spectrum = np.abs(np.fft.rfft(envelope - envelope.mean())) ** 2 / (sampling_rate_hz * (window @ window))
    return {"envelope_energy": float(spectrum.sum() * sampling_rate_hz / len(x)),
            "envelope_peak": float(envelope.max()), "sample_rate_hz": sampling_rate_hz,
            "convention": "DC removed; Hann; full-band Hilbert; one-sided un-doubled PSD",
            "bearing_fault_frequencies": None, "reason": "verified_geometry_required"}
