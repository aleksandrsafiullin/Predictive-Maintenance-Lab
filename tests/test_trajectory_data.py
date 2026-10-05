import numpy as np
import pandas as pd
import pytest

from pdm.trajectory_data import build_trajectory_frame, build_trajectory_prefix, causal_features


def sample(values, gap=None):
    features = pd.DataFrame(
        {
            "unit_id": "u",
            "physical_unit_id": "physical",
            "timestamp_s": np.arange(len(values)) * 60.0,
            "signal": values,
            "gap_before": gap if gap is not None else [True] + [False] * (len(values) - 1),
        }
    )
    return {
        "features": features,
        "schema": {"thresholds": {"mode": "absolute", "direction": "above", "red": 3.0}},
    }


CONFIG = {"history_length": 2, "horizons_s": [60.0, 120.0, 180.0], "target_tolerance_s": 0.01}


def test_hidden_suffix_cannot_change_encoder_or_origin():
    data = sample([0.4, 0.5, 0.6, 1.0, 4.0])
    original = build_trajectory_frame(data, ["u"], CONFIG)
    changed = sample([0.4, 0.5, 0.6, 100.0, 200.0])
    other = build_trajectory_frame(changed, ["u"], CONFIG)
    np.testing.assert_array_equal(original["x"][0], other["x"][0])
    prefix = build_trajectory_prefix(data, data["features"].iloc[:2], CONFIG)
    np.testing.assert_array_equal(prefix["x"][0], original["x"][0])
    assert prefix["current"][0] == original["current"][0]
    assert not prefix["mask"].any()


def test_exact_event_bucket_and_censored_survival():
    data = sample([0.4, 0.5, 0.6, 4.0])
    frame = build_trajectory_frame(data, ["u"], CONFIG)
    assert frame["event_observed"][0]
    assert np.flatnonzero(frame["event_allowed"][0]).tolist() == [1]
    no_event = build_trajectory_frame(sample([0.4, 0.5, 0.6]), ["u"], CONFIG)
    assert not no_event["event_observed"].any()
    assert no_event["no_entry_prefix"].tolist() == [1, 0]
    assert no_event["event_allowed"][0].tolist() == [False, True, True, True]


def test_gap_does_not_admit_crossing_or_first_entry_warning():
    frame = build_trajectory_frame(
        sample([0.4, 0.5, 4.0, 4.1], [True, False, True, False]), ["u"], CONFIG
    )
    assert not frame["mask"].any()
    assert not frame["event_observed"].any()
    assert frame["warning_eligible"].tolist() == [True, False]
    data = sample([0.4, 0.5, 0.6], [True, False, True])
    with pytest.raises(ValueError, match="Insufficient"):
        build_trajectory_prefix(data, data["features"], CONFIG)


def test_known_context_masks_and_causal_rolls():
    data = sample([0.1, 0.2, 100.0])
    values = causal_features(data["features"])
    shortened = causal_features(data["features"].iloc[:2])
    np.testing.assert_array_equal(values[:2], shortened)
    assert np.isfinite(values).all()
    assert not values[:, 8:].any()
