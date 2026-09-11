from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_bearing_tables():
    """Labeled synthetic fixture, not the XJTU-SY dataset."""
    rows = []
    for unit, regime, inst, n, gap_at in [
        ("Bearing1_1", "35Hz12kN", 1, 30, None),
        ("Bearing1_2", "35Hz12kN", 2, 28, None),
        ("Bearing1_3", "35Hz12kN", 3, 26, None),
        ("Bearing1_4", "35Hz12kN", 4, 24, None),
        ("Bearing1_5", "35Hz12kN", 5, 22, None),
        ("Bearing2_1", "37.5Hz11kN", 1, 20, 8),
    ]:
        for i in range(1, n + 1):
            if gap_at is not None and i == gap_at:
                continue
            rows.append(
                {
                    "dataset_id": "bearings",
                    "unit_id": unit,
                    "regime_id": regime,
                    "instance": inst,
                    "file_index": i,
                    "timestamp_s": float(i * 60),
                    "operating_age_s": float(i * 60),
                    "rpm": 2100.0 if inst else 2250.0,
                    "load_kn": 12.0,
                    "horizontal_rms": 0.1 + 0.01 * i,
                    "horizontal_std": 0.1,
                    "horizontal_abs_peak": 0.3,
                    "horizontal_peak_to_peak": 0.6,
                    "horizontal_crest_factor": 2.0,
                    "horizontal_kurtosis": 3.0,
                    "vertical_rms": 0.1,
                    "vertical_std": 0.1,
                    "vertical_abs_peak": 0.3,
                    "vertical_peak_to_peak": 0.6,
                    "vertical_crest_factor": 2.0,
                    "vertical_kurtosis": 3.0,
                    "horizontal_band_0": 0.01,
                    "horizontal_band_1": 0.01,
                    "horizontal_band_2": 0.01,
                    "horizontal_band_3": 0.01,
                    "vertical_band_0": 0.01,
                    "vertical_band_1": 0.01,
                    "vertical_band_2": 0.01,
                    "vertical_band_3": 0.01,
                    "gap_before": bool(gap_at is not None and i == gap_at + 1),
                }
            )
    features = pd.DataFrame(rows)
    units = (
        features.groupby("unit_id", as_index=False)
        .agg(
            instance=("instance", "first"),
            regime_id=("regime_id", "first"),
            observation_end_s=("timestamp_s", "max"),
            n_measurements=("timestamp_s", "count"),
        )
        .assign(event_observed=1, dataset_id="bearings")
    )
    units["event_time_s"] = units["observation_end_s"]
    units["endpoint_definition"] = "last_recorded_sample"
    return features, units


@pytest.fixture
def tiny_filter_tables():
    """Labeled synthetic fixture, not the HSE dataset."""
    feat_rows = []
    unit_rows = []
    # Synthetic dust labels for encoding tests — not HSE MATLAB fields.
    dust_a3 = "ISO 12103-1, A3 Medium Test Dust"
    dust_a2 = "ISO 12103-1, A2 Fine Test Dust"
    # 8 train-like + 2 test-like units
    specs = []
    for i in range(1, 9):
        event = i <= 2
        n = 40
        dust = dust_a2 if i % 2 == 0 else dust_a3
        specs.append((f"Filter_{i}", n, event, "author_train", None, dust))
    specs.append(("Filter_101", 30, False, "author_test", 12.0, dust_a3))
    specs.append(("Filter_102", 25, False, "author_test", 8.0, dust_a2))
    for uid, n, event, split, official_rul_min, dust in specs:
        t = np.arange(1, n + 1, dtype=float) * 0.1  # minutes
        dp = np.linspace(10, 650 if event else 400, n)
        for j in range(n):
            feat_rows.append(
                {
                    "dataset_id": "filters",
                    "unit_id": uid,
                    "author_data_no": int(uid.split("_")[1]),
                    "regime_id": "A3",
                    "timestamp_s": float(t[j] * 60.0),
                    "time_original": float(t[j]),
                    "operating_age_s": float(t[j] * 60.0),
                    "delta_t_s": 6.0 if j else 0.0,
                    "differential_pressure": float(dp[j]),
                    "delta_pressure": float(dp[j] - dp[j - 1]) if j else 0.0,
                    "flow_rate": 80.0,
                    "dust_feed": 100.0,
                    "dust": dust,
                    "gap_before": False,
                    "author_split": split,
                }
            )
        ts = t * 60.0
        obs_end = float(ts[-1])
        if event and split == "author_train":
            et = float(ts[np.argmax(dp > 600)])
        else:
            et = float("nan")
        unit_rows.append(
            {
                "dataset_id": "filters",
                "unit_id": uid,
                "author_data_no": int(uid.split("_")[1]),
                "author_split": split,
                "event_observed": int(event and split == "author_train"),
                "event_time_s": et,
                "observation_end_s": obs_end,
                "official_rul_at_prefix_end_s": None if official_rul_min is None else official_rul_min * 60.0,
                "official_rul_at_prefix_end_original": official_rul_min,
                "dust": dust,
                "regime_id": "A3",
            }
        )
    return pd.DataFrame(feat_rows), pd.DataFrame(unit_rows)
