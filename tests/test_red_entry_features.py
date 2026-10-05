import copy
import json

import numpy as np
import pandas as pd
import pytest

from pdm.red_entry_features import build_windows, fit_feature_state, transform_prefix


def snapshot():
    frame = pd.DataFrame({"unit_id": ["a"] * 4 + ["b"] * 2, "timestamp_s": [0., 1., 2., 3., 0., 1.],
                          "signal": [1., 2., 3., 4., 999., 999.],
                          "operating_age_s": [0., 10., np.nan, 30., 900., 901.],
                          "operating_age_known": [True, True, False, True, True, True],
                          "dust": ["sand", "sand", "soil", "soil", "validation_only", "validation_only"],
                          "gap_before": [False, False, True, False, False, False],
                          "physical_unit_id": ["equipment_a"] * 4 + ["equipment_b"] * 2,
                          "rowindex": range(6), "target": range(6)})
    roles = {"signal": "sensor", "operating_age_s": "known_age", "operating_age_known": "quality",
             "dust": "operating_context", "gap_before": "quality", "physical_unit_id": "metadata",
             "timestamp_s": "metadata", "unit_id": "metadata", "rowindex": "sensor", "target": "target_only"}
    return {"features": frame, "split": {"train": ["a"], "validation": ["b"], "test": []},
            "schema": {"schema_version": 2, "columns": {name: {"role": role} for name, role in roles.items()},
                       "thresholds": {"mode": "absolute", "red": 10., "direction": "above"}}}


def test_modes_and_train_only_state():
    data = snapshot()
    state = fit_feature_state(data, "hybrid")
    assert state["preprocessing"]["signal"]["impute"] == 2.5
    assert state["preprocessing"]["dust"]["vocabulary"] == ["sand", "soil"]
    changed = copy.deepcopy(data)
    changed["features"].loc[4:, "signal"] = -100000.
    changed["features"].loc[4:, "dust"] = "different"
    assert fit_feature_state(changed, "hybrid") == state
    unknown = transform_prefix(data["features"].iloc[4:], data["schema"], state)
    assert np.all(unknown["x"][:, state["feature_names"].index("dust__unknown")] == 1)
    assert not unknown["supported_regime"].any()
    assert state == json.loads(json.dumps(state))
    reloaded = json.loads(json.dumps(state, sort_keys=True))
    np.testing.assert_array_equal(transform_prefix(data["features"].iloc[:4], data["schema"], state)["x"],
                                  transform_prefix(data["features"].iloc[:4], data["schema"], reloaded)["x"])
    sensor = fit_feature_state(data, "sensor_only")
    assert set(sensor["source_specs"]) == {"signal"}
    assert not any("age" in name or "rowindex" in name or "timestamp" in name for name in sensor["feature_names"])
    age = fit_feature_state(data, "age_context")
    assert set(age["source_specs"]) == {"operating_age_s", "dust"}
    assert "signal" not in age["feature_names"]


def test_sensor_only_excludes_context_values_aliases_and_availability():
    data = snapshot()
    for name, values in {
        "load": [0., 1., 2., 3., 4., 5.],
        "known_regime": ["normal"] * 6,
        "load_trailing_slope": [10., 20., 30., 40., 50., 60.],
    }.items():
        data["features"][name] = values
        data["schema"]["columns"][name] = {"role": "operating_context"}
    sensor = fit_feature_state(data, "sensor_only")
    assert set(sensor["source_specs"]) == {"signal"}
    assert set(sensor["preprocessing"]) == {
        "signal", "signal_trailing_slope", "known_red_limit", "distance_to_red",
    }
    prefix = data["features"].iloc[:4].copy()
    expected = transform_prefix(prefix, data["schema"], sensor)
    changed = copy.deepcopy(data)
    for name, spec in data["schema"]["columns"].items():
        if spec["role"] == "operating_context":
            changed["features"][name] = -999. if name == "load" else "unseen"
    changed["features"]["operating_age_s"] = 100000.
    changed["features"]["operating_age_known"] = False
    assert fit_feature_state(changed, "sensor_only") == sensor
    actual = transform_prefix(changed["features"].iloc[:4], data["schema"], sensor)
    for key in ("x", "availability", "supported_regime", "age_available", "sequence_starts"):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual["supported_regime"].all()
    assert not actual["age_available"].any()
    prefix["signal"] = np.nan
    unavailable = transform_prefix(prefix, data["schema"], sensor)
    assert not unavailable["availability"].any()  # Context cannot supply sensor availability.
    origins = pd.DataFrame({"unit_id": ["a"], "timestamp_s": [3.]})
    expected_window = build_windows(data, origins, sensor, 5)
    actual_window = build_windows(changed, origins, sensor, 5)
    for key in ("x", "lengths", "availability", "supported_regime", "age_available"):
        np.testing.assert_array_equal(actual_window[key], expected_window[key])


def test_sensor_only_does_not_derive_features_from_context_named_signal():
    data = snapshot()
    data["schema"]["columns"]["signal"]["role"] = "operating_context"
    data["features"]["vibration"] = data["features"]["signal"]
    data["schema"]["columns"]["vibration"] = {"role": "sensor"}
    state = fit_feature_state(data, "sensor_only")
    assert set(state["source_specs"]) == {"vibration"}
    assert state["feature_names"] == ["vibration", "vibration__missing"]
    changed = data["features"].iloc[:4].copy()
    changed["signal"] = 100000.
    np.testing.assert_array_equal(
        transform_prefix(changed, data["schema"], state)["x"],
        transform_prefix(data["features"].iloc[:4], data["schema"], state)["x"],
    )


@pytest.mark.parametrize("mode", ["sensor_only", "age_context", "hybrid"])
def test_physical_clock_aliases_remain_excluded(mode):
    data = snapshot()
    aliases = ("elapsed_s", "sample_index", "record_end", "cycle_counter", "remaining_rul",
               "time", "operating_age_alias", "lifetime", "row_number")
    for i, name in enumerate(aliases):
        data["features"][name] = np.arange(6, dtype=float)
        data["schema"]["columns"][name] = {
            "role": "sensor" if i % 2 else "operating_context",
        }
    state = fit_feature_state(data, mode)
    assert not set(aliases) & set(state["source_specs"])
    assert not any(name in state["feature_names"] for name in aliases)


@pytest.mark.parametrize("mode", ["age_context", "hybrid"])
def test_context_modes_retain_numeric_and_categorical_inputs(mode):
    data = snapshot()
    data["features"]["load"] = [0., 1., 2., 3., 4., 5.]
    data["schema"]["columns"]["load"] = {"role": "operating_context"}
    state = fit_feature_state(data, mode)
    expected_sources = {"operating_age_s", "dust", "load"}
    if mode == "hybrid":
        expected_sources.add("signal")
    assert set(state["source_specs"]) == expected_sources
    prefix = data["features"].iloc[:1].copy()
    original = transform_prefix(prefix, data["schema"], state)
    for name, value in (("operating_age_s", 500.), ("load", 500.), ("dust", "soil")):
        changed = prefix.copy()
        changed[name] = value
        assert not np.array_equal(original["x"], transform_prefix(changed, data["schema"], state)["x"])
    unknown = transform_prefix(data["features"].iloc[4:], data["schema"], state)
    assert not unknown["supported_regime"].any()
    assert unknown["age_available"].all()


def test_age_changes_input_and_unknown_differs_from_zero():
    data = snapshot()
    state = fit_feature_state(data, "age_context")
    prefix = data["features"].iloc[:1].copy()
    zero = transform_prefix(prefix, data["schema"], state)["x"]
    prefix["operating_age_s"] = 500.
    older = transform_prefix(prefix, data["schema"], state)["x"]
    assert not np.array_equal(zero, older)
    prefix["operating_age_s"] = 0.
    prefix["operating_age_known"] = False
    unknown = transform_prefix(prefix, data["schema"], state)["x"]
    assert unknown[0, state["feature_names"].index("operating_age_s__missing")] == 1.
    assert zero[0, state["feature_names"].index("operating_age_s__missing")] == 0.
    sensor_state = fit_feature_state(data, "sensor_only")
    a = transform_prefix(data["features"].iloc[:1], data["schema"], sensor_state)["x"]
    b = transform_prefix(prefix, data["schema"], sensor_state)["x"]
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("mode", ["sensor_only", "age_context", "hybrid"])
def test_future_perturbation_and_relative_threshold_warmup(mode):
    data = snapshot()
    data["schema"]["thresholds"] = {"mode": "initial_baseline_multiple", "baseline_n": 3, "red_ratio": 2.}
    state = fit_feature_state(data, mode)
    prefix = data["features"].iloc[:2]
    short = transform_prefix(prefix, data["schema"], state)
    full = transform_prefix(data["features"].iloc[:4], data["schema"], state)
    np.testing.assert_array_equal(short["x"], full["x"][:2])
    if mode != "age_context":
        assert short["x"][1, state["feature_names"].index("known_red_limit__missing")] == 1.
        assert full["x"][2, state["feature_names"].index("known_red_limit__missing")] == 0.
    altered = data["features"].iloc[:4].copy()
    altered.loc[2:, "signal"] = 100000.
    altered.loc[2:, "operating_age_s"] = 100000.
    altered.loc[2:, "dust"] = "future_regime"
    np.testing.assert_array_equal(transform_prefix(altered, data["schema"], state)["x"][:2], short["x"])


def test_right_padding_origins_alignment_and_gap_preserves_age():
    data = snapshot()
    state = fit_feature_state(data, "hybrid")
    origins = pd.DataFrame({"unit_id": ["a", "a", "a"], "timestamp_s": [3., 0., 2.],
                            "physical_unit_id": ["equipment_a"] * 3})
    windows = build_windows(data, origins, state, 5)
    assert windows["lengths"].tolist() == [2, 1, 1]
    assert windows["x"].shape == (3, 5, len(state["feature_names"]))
    assert windows["physical_unit_ids"] == ["equipment_a"] * 3
    assert windows["availability"].tolist() == [True] * 3
    assert np.all(windows["x"][1, 1:] == 0)
    full = transform_prefix(data["features"].iloc[:4], data["schema"], state)
    np.testing.assert_array_equal(windows["x"][0, 1], full["x"][3])
    assert full["x"][2, state["feature_names"].index("signal_trailing_slope__missing")] == 1.
    # Gap resets sequence, not imported physical age.
    assert full["x"][3, state["feature_names"].index("operating_age_s")] > full["x"][0, state["feature_names"].index("operating_age_s")]


def test_new_cycle_resets_sequence_and_relative_baseline():
    data = snapshot()
    data["features"]["component_cycle_id"] = ["c1", "c1", "c2", "c2", "c1", "c1"]
    data["schema"]["columns"]["component_cycle_id"] = {"role": "metadata"}
    data["schema"]["thresholds"] = {"mode": "initial_baseline_multiple", "baseline_n": 2, "red_ratio": 2.}
    state = fit_feature_state(data, "sensor_only")
    transformed = transform_prefix(data["features"].iloc[:4], data["schema"], state)
    assert transformed["sequence_starts"].tolist() == [0, 0, 2, 2]
    assert transformed["x"][2, state["feature_names"].index("known_red_limit__missing")] == 1.


def test_legacy_and_frozen_contract():
    data = snapshot()
    del data["schema"]["columns"]
    age = fit_feature_state(data, "age_context")
    transformed = transform_prefix(data["features"].iloc[:1], data["schema"], age)
    assert transformed["x"].shape == (1, 0)
    assert not transformed["availability"].any()
    sensor = fit_feature_state(data, "sensor_only")
    assert set(sensor["source_specs"]) == {"signal"}
    changed = copy.deepcopy(data["schema"])
    changed["thresholds"]["red"] = 11.
    with pytest.raises(ValueError, match="schema"):
        transform_prefix(data["features"].iloc[:1], changed, sensor)
    sensor["feature_names"].reverse()
    with pytest.raises(ValueError, match="integrity"):
        transform_prefix(data["features"].iloc[:1], data["schema"], sensor)
