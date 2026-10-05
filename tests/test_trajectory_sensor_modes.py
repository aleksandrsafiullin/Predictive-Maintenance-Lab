"""Causal relative sensor inputs leave the RMS target contract unchanged."""

import numpy as np
import pandas as pd
import pytest

from pdm.trajectory_data import (
    FEATURE_NAMES,
    SENSOR_AVAILABILITY,
    SENSOR_FEATURE_NAMES,
    build_trajectory_frame,
    build_trajectory_prefix,
    causal_features,
)

CONFIG = {"history_length": 3, "horizons_s": [60., 120.]}


def sensor_data():
    features = pd.DataFrame({
        "unit_id": "u", "timestamp_s": np.arange(24) * 60.,
        "signal": np.linspace(.2, 2., 24),
        "gap_before": [True] + [False] * 11 + [True] + [False] * 11,
    })
    for i, name in enumerate(SENSOR_FEATURE_NAMES):
        features[name] = (np.arange(24) + 1.) * (i + 1.)
    features["horizontal_rms"] = features.signal
    features["vertical_rms"] = features.signal / 2.
    return {"features": features, "schema": {
        "thresholds": {"mode": "absolute", "direction": "above", "red": 1.5},
        "trajectory_sensor_features": {
            "schema_version": 1, "columns": list(SENSOR_FEATURE_NAMES),
            "transform": "log1p", "availability": SENSOR_AVAILABILITY,
        },
    }}


@pytest.mark.parametrize("mode,count", [("absolute", 33), ("baseline_relative", 33), ("combined", 53)])
def test_mode_counts_names_and_unmodified_rms_targets(mode, count):
    data = sensor_data()
    original = build_trajectory_frame(data, ["u"], CONFIG)
    frame = build_trajectory_frame(data, ["u"], {**CONFIG, "sensor_feature_mode": mode})
    assert frame["x"].shape[-1] == count
    assert len(frame["feature_names"]) == count
    assert len(set(frame["feature_names"])) == count
    assert frame["feature_names"][:13] == FEATURE_NAMES
    for key in original:
        if key not in ("x", "feature_names"):
            np.testing.assert_array_equal(frame[key], original[key])
    np.testing.assert_array_equal(frame["x"][:, :, :13], original["x"][:, :, :13])
    if mode == "absolute":
        np.testing.assert_array_equal(frame["x"], original["x"])
        assert frame["feature_names"] == original["feature_names"]
    else:
        assert frame["feature_names"] != original["feature_names"]
    if mode == "combined":
        np.testing.assert_array_equal(frame["x"][:, :, :33], original["x"])


@pytest.mark.parametrize("mode", ["baseline_relative", "combined"])
def test_expanding_initial_history_then_fixed_eight_baseline(mode):
    segment = sensor_data()["features"].iloc[:12]
    actual = causal_features(segment, SENSOR_FEATURE_NAMES, mode)
    sensor_log = np.log1p(segment.loc[:, list(SENSOR_FEATURE_NAMES)].to_numpy(float))
    expected = np.stack([row - sensor_log[:min(i + 1, 8)].mean(axis=0)
                         for i, row in enumerate(sensor_log)])
    np.testing.assert_allclose(actual[:, -20:], expected, atol=1e-7, rtol=1e-6)
    np.testing.assert_array_equal(actual[0, -20:], np.zeros(20))
    for end in range(1, 13):
        shortened = causal_features(segment.iloc[:end], SENSOR_FEATURE_NAMES, mode)
        np.testing.assert_array_equal(actual[:end], shortened)


@pytest.mark.parametrize("mode", ["baseline_relative", "combined"])
def test_suffix_invariance_and_every_prefix_matches_frame_after_gap(mode):
    data = sensor_data()
    config = {**CONFIG, "sensor_feature_mode": mode}
    frame = build_trajectory_frame(data, ["u"], config)
    for i, at in enumerate(frame["as_of_s"]):
        prefix = data["features"].loc[data["features"].timestamp_s <= at]
        standalone = build_trajectory_prefix(data, prefix, config)
        np.testing.assert_array_equal(frame["x"][i], standalone["x"][0])
    changed = sensor_data()
    changed["features"].loc[3:, list(SENSOR_FEATURE_NAMES)] *= 100.
    changed["features"]["signal"] = changed["features"].horizontal_rms
    changed_frame = build_trajectory_frame(changed, ["u"], config)
    np.testing.assert_array_equal(frame["x"][0], changed_frame["x"][0])
    # The first eligible origin after the gap starts a fresh initial baseline.
    after_gap = frame["as_of_s"].index(14 * 60.)
    np.testing.assert_array_equal(frame["x"][after_gap, 0, -20:], np.zeros(20))
    segment = data["features"].iloc[12:].reset_index(drop=True)
    np.testing.assert_array_equal(frame["x"][after_gap],
                                  causal_features(segment, SENSOR_FEATURE_NAMES, mode)[:3])


@pytest.mark.parametrize("mode", ["baseline_relative", "combined"])
@pytest.mark.parametrize("fault", ["missing", "nonfinite", "negative", "rms", "declaration"])
def test_relative_modes_keep_audited_sensor_validation(mode, fault):
    data = sensor_data()
    if fault == "missing":
        data["features"] = data["features"].drop(columns="horizontal_band_0")
    elif fault == "nonfinite":
        data["features"].loc[1, "horizontal_band_0"] = np.inf
    elif fault == "negative":
        data["features"].loc[1, "horizontal_band_0"] = -1.
    elif fault == "rms":
        data["features"].loc[1, "horizontal_rms"] += .1
    else:
        data["schema"]["trajectory_sensor_features"]["columns"].append("event_time_s")
    with pytest.raises(ValueError):
        build_trajectory_prefix(data, data["features"].iloc[:3],
                                {**CONFIG, "sensor_feature_mode": mode})


@pytest.mark.parametrize("mode", ["baseline_relative", "combined"])
def test_nonabsolute_requires_explicit_sensor_declaration(mode):
    data = sensor_data()
    del data["schema"]["trajectory_sensor_features"]
    with pytest.raises(ValueError, match="requires declared"):
        build_trajectory_frame(data, ["u"], {**CONFIG, "sensor_feature_mode": mode})
    with pytest.raises(ValueError, match="requires declared"):
        causal_features(data["features"], sensor_feature_mode=mode)


def test_unknown_mode_is_rejected_even_with_no_eligible_rows():
    data = sensor_data()
    data["features"] = data["features"].iloc[:1]
    with pytest.raises(ValueError, match="Invalid sensor_feature_mode"):
        build_trajectory_frame(data, ["u"], {**CONFIG, "sensor_feature_mode": "unknown"})
