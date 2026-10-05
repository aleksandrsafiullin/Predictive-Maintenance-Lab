"""Synthetic variable-context integration; no real data or quality claims."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pdm import learned_trajectory as learned
from pdm.io_util import sha256_file
from pdm.trajectory_data import (
    SENSOR_AVAILABILITY,
    SENSOR_FEATURE_NAMES,
    build_trajectory_frame,
    slice_frame,
)


def synthetic_data():
    """Two Train physical groups, one Validation group and an explicit gap."""
    records = []
    for unit, count in (("train-a", 44), ("train-b", 35), ("validation-v", 22)):
        for i in range(count):
            segment = int(unit == "train-a" and i >= 28)
            at = i * 60.0 + (600.0 if segment else 0.0)
            records.append(
                {
                    "unit_id": unit,
                    "physical_unit_id": unit,
                    "timestamp_s": at,
                    "gap_before": i == 0 or (unit == "train-a" and i == 28),
                    "segment_id": segment,
                    "signal": 0.3 + 0.03 * (i % 9) + (1.0 if i in (18, 19) else 0.0),
                }
            )
    features = pd.DataFrame(records)
    for index, name in enumerate(SENSOR_FEATURE_NAMES):
        features[name] = 0.1 + features["signal"] * (index + 1) / 10
    features["horizontal_rms"] = features["signal"]
    features["vertical_rms"] = features["signal"] * 0.8
    return {
        "features": features,
        "schema": {
            "thresholds": {"mode": "absolute", "direction": "above", "red": 1.0},
            "trajectory_sensor_features": {
                "columns": list(SENSOR_FEATURE_NAMES),
                "transform": "log1p",
                "availability": SENSOR_AVAILABILITY,
                "schema_version": 1,
            },
        },
        "split": {"train": ["train-a", "train-b"], "validation": ["validation-v"], "test": []},
    }


def assert_outputs_equal(first, second):
    assert first.keys() == second.keys()
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def assert_old_frame_fields_equal(first, second):
    for key in (
        "y",
        "mask",
        "current",
        "unit_id",
        "physical_unit_id",
        "as_of_s",
        "event_allowed",
        "event_observed",
        "no_entry_prefix",
        "warning_eligible",
        "red_threshold",
        "feature_names",
    ):
        np.testing.assert_array_equal(first[key], second[key])


def resolved_config(engine="gru", mode="variable_causal"):
    params = {
        "horizons_s": [60.0, 120.0, 180.0],
        "history_length": 8,
        "hidden_size": 16,
        "num_layers": 1,
        "rank": 1,
        "epochs": 2,
        "batch_size": 16,
        "path_samples": 16,
        "training_samples": 4,
        "cpu_threads": 1,
        "seed": 932,
        "dropout": 0.0,
        "event_distribution": "finite_horizon_mixture",
        "path_distribution": "coupled_timing_mixture",
        "phase_covariance": "separate",
        "recurrent_context_mode": mode,
        "sensor_feature_mode": "baseline_relative",
        "path_objective_horizons_s": [60.0, 180.0],
        "event_objective_horizons_s": [60.0, 180.0],
    }
    if mode == "variable_causal":
        params["max_history_length"] = 32
    return learned.learned_params(engine, params, synthetic_data()["features"])


def frames(config):
    data = synthetic_data()
    return tuple(
        build_trajectory_frame(data, data["split"][partition], config)
        for partition in ("train", "validation")
    )


@pytest.fixture(scope="module")
def variable_bundle():
    config = resolved_config()
    train, validation = frames(config)
    return learned.fit_learned_model("gru", train, validation, config)


def save_contracts(bundle, directory):
    artifacts = learned.save_learned_bundle(bundle, directory)
    fields = learned._event_distribution_metadata(bundle["model"], bundle["config"])
    for name in ("training_contract.json", "model_input_contract.json"):
        (directory / name).write_text(json.dumps(fields))
        artifacts[name] = sha256_file(directory / name)
    return {
        "dir": directory,
        "engine_id": "gru",
        "params": bundle["config"],
        "scaler": bundle["scaler"],
        "artifacts": artifacts,
        **copy.deepcopy(fields),
    }


@pytest.mark.parametrize("cap", [None, 2, 10])
def test_old_origins_labels_caps_gaps_and_scaler_anchors_are_identical(cap):
    data = synthetic_data()
    fixed = build_trajectory_frame(data, data["split"]["train"], resolved_config(mode="fixed"), cap)
    variable = build_trajectory_frame(data, data["split"]["train"], resolved_config(), cap)
    assert_old_frame_fields_equal(fixed, variable)
    np.testing.assert_array_equal(variable["scaler_x"], fixed["x"])
    for key, value in learned._objective_weights(fixed).items():
        np.testing.assert_array_equal(value, learned._objective_weights(variable)[key])
    assert variable["x"].shape[1:] == (32, 33)
    assert variable["history_lengths"].dtype.kind in "iu"
    assert variable["history_mask"].dtype == bool
    assert np.array_equal(
        variable["history_mask"], np.arange(32)[None] < variable["history_lengths"][:, None]
    )
    first_after_gap = np.flatnonzero(
        (np.asarray(variable["unit_id"]) == "train-a") & (np.asarray(variable["as_of_s"]) == 2700.0)
    )
    if cap is None:
        assert len(first_after_gap) == 1
        assert variable["history_lengths"][first_after_gap[0]] == 8
        assert (~variable["mask"].any(1)).sum() == 3
        assert variable["history_lengths"].max() == 32


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_actual_variable_fit_reload_and_exact_last8_train_scaler(engine, tmp_path):
    config = resolved_config(engine)
    train, validation = frames(config)
    bundle = learned.fit_learned_model(engine, train, validation, config)
    fixed_config = resolved_config(engine, "fixed")
    fixed_train, fixed_validation = frames(fixed_config)
    fixed_bundle = learned.fit_learned_model(engine, fixed_train, fixed_validation, fixed_config)
    assert bundle["scaler"] == fixed_bundle["scaler"]
    assert bundle["selection"]["best_score"] == min(
        row["validation"]["total"] for row in bundle["trace"]
    )
    assert bundle["selection"]["restored_best_checkpoint"]
    assert not bundle["selection"]["test_feedback"]
    for row in bundle["trace"]:
        assert all(
            np.isfinite(value) for part in ("train", "validation") for value in row[part].values()
        )
    artifacts = learned.save_learned_bundle(bundle, tmp_path)
    fields = learned._event_distribution_metadata(bundle["model"], config)
    loaded = learned.load_learned_bundle(
        {"dir": tmp_path, "engine_id": engine, "params": config, "artifacts": artifacts, **fields}
    )
    for name, value in bundle["model"].state_dict().items():
        assert torch.equal(value, loaded["model"].state_dict()[name])
    assert_outputs_equal(
        learned.predict_learned(bundle, validation), learned.predict_learned(loaded, validation)
    )


def test_public_prediction_ignores_future_and_arbitrary_nonfinite_padding(variable_bundle):
    config = resolved_config()
    data = synthetic_data()
    before = slice_frame(build_trajectory_frame(data, ["validation-v"], config), np.array([0]))
    changed_data = {**data, "features": data["features"].copy()}
    future = (changed_data["features"].unit_id == "validation-v") & (
        changed_data["features"].timestamp_s > 420
    )
    changed_data["features"].loc[future, "signal"] = 9.0
    changed_data["features"].loc[future, "horizontal_rms"] = 9.0
    changed_data["features"].loc[future, "vertical_rms"] = 7.2
    after = slice_frame(
        build_trajectory_frame(changed_data, ["validation-v"], config), np.array([0])
    )
    np.testing.assert_array_equal(before["x"], after["x"])
    assert not np.array_equal(before["y"], after["y"])
    assert not np.array_equal(before["event_allowed"], after["event_allowed"])
    baseline = learned.predict_learned(variable_bundle, before)
    assert_outputs_equal(baseline, learned.predict_learned(variable_bundle, after))
    for fill in (1e6, np.nan, np.inf, -np.inf):
        padded = copy.deepcopy(before)
        padded["x"][~padded["history_mask"]] = fill
        padded["raw_x"][~padded["history_mask"]] = fill
        normalized = learned._normalized(padded, variable_bundle["scaler"])
        assert (normalized[~padded["history_mask"]] == 0).all()
        assert np.isfinite(normalized).all()
        assert_outputs_equal(baseline, learned.predict_learned(variable_bundle, padded))


@pytest.mark.parametrize("engine", ["full_cns", "quantile_boosting"])
def test_variable_selector_rejected_for_external_engines(engine):
    with pytest.raises(ValueError):
        resolved_config(engine)
    config = resolved_config()
    with pytest.raises(ValueError):
        learned._make_model(engine, config, 33, 5)


def test_scaler_anchor_tampering_is_not_silently_accepted():
    config = resolved_config()
    train, _ = frames(config)
    train["scaler_x"][0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="scaler"):
        learned._normalization_reference(train, config)


def test_context_metadata_roundtrip_and_same_shaped_wrong_packing_rejection(
    variable_bundle, tmp_path
):
    run = save_contracts(variable_bundle, tmp_path)
    fields = learned._event_distribution_metadata(
        variable_bundle["model"], variable_bundle["config"]
    )
    metadata = json.loads((tmp_path / "learned_model.json").read_text())
    for key, value in fields.items():
        assert metadata[key] == run[key] == value
    for name in ("training_contract.json", "model_input_contract.json"):
        value = json.loads((tmp_path / name).read_text())
        for key, expected in fields.items():
            assert value[key] == expected
    loaded = learned.load_learned_bundle(run)
    for name, value in variable_bundle["model"].state_dict().items():
        assert value.shape == loaded["model"].state_dict()[name].shape
    metadata["recurrent_context_contract"]["encoder_policy"] = "unpacked_last_padded_hidden"
    (tmp_path / "learned_model.json").write_text(json.dumps(metadata))
    run["artifacts"]["learned_model.json"] = sha256_file(tmp_path / "learned_model.json")
    with pytest.raises(ValueError, match="context"):
        learned.load_learned_bundle(run)


CONTEXT_FIELDS = [
    ("recurrent_context_mode", "fixed"),
    ("recurrent_context_contract", {}),
    ("recurrent_context_contract.minimum_real_observations", 2),
    ("recurrent_context_contract.maximum_real_observations", 16),
    ("recurrent_context_contract.tensor_layout", "time_batch_feature"),
    ("recurrent_context_contract.chronological_order", "newest to oldest"),
    ("recurrent_context_contract.padding", "left padding"),
    ("recurrent_context_contract.lengths_dtype", "floating"),
    ("recurrent_context_contract.mask_dtype", "integer"),
    ("recurrent_context_contract.encoder_architecture", "lstm"),
    ("recurrent_context_contract.encoder_policy", "unpacked_last_padded_hidden"),
    ("recurrent_context_contract.gap_policy", "bridge gaps"),
    ("recurrent_context_contract.normalization_reference", "all padded rows"),
    ("recurrent_context_contract.sensor_baseline", "future full-record baseline"),
    ("recurrent_context_contract.feature_history", "truncate then compute features"),
    ("recurrent_context_contract.prediction_seed_policy", "include padding bytes"),
    ("recurrent_context_contract.minimum_eligibility_policy", "admit shorter prefixes"),
]


@pytest.mark.parametrize(
    "surface",
    ["learned_model.json", "manifest", "training_contract.json", "model_input_contract.json"],
)
@pytest.mark.parametrize("field,wrong", CONTEXT_FIELDS)
@pytest.mark.parametrize("operation", ["missing", "contradictory"])
def test_valid_rehash_context_contract_tampering_rejected(
    variable_bundle, tmp_path, surface, field, wrong, operation
):
    run = save_contracts(variable_bundle, tmp_path)
    value = run if surface == "manifest" else json.loads((tmp_path / surface).read_text())
    keys = field.split(".")
    parent = value
    for key in keys[:-1]:
        parent = parent[key]
    if operation == "missing":
        del parent[keys[-1]]
    else:
        parent[keys[-1]] = wrong
    if surface != "manifest":
        (tmp_path / surface).write_text(json.dumps(value))
        run["artifacts"][surface] = sha256_file(tmp_path / surface)
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize(
    "field,value",
    [("recurrent_context_mode", "variable_causal"), ("recurrent_context_contract", {})],
)
def test_context_only_manifest_cannot_bypass_validation(variable_bundle, tmp_path, field, value):
    run = save_contracts(variable_bundle, tmp_path)
    run = {key: run[key] for key in ("dir", "engine_id", "params", "scaler", "artifacts")}
    run[field] = value
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


def test_fixed_missing_selector_exact_state_trace_rng_and_public_reload(tmp_path):
    explicit = resolved_config(mode="fixed")
    missing = dict(explicit)
    missing.pop("recurrent_context_mode")
    missing.pop("max_history_length", None)
    train, validation = frames(explicit)
    old_train, old_validation = frames(missing)
    assert_old_frame_fields_equal(train, old_train)
    np.testing.assert_array_equal(train["x"], old_train["x"])
    first = learned.fit_learned_model("gru", train, validation, explicit)
    first_rng = torch.random.get_rng_state().clone()
    old = learned.fit_learned_model("gru", old_train, old_validation, missing)
    assert torch.equal(first_rng, torch.random.get_rng_state())
    assert first["trace"] == old["trace"]
    assert first["selection"] == old["selection"]
    for key, value in first["model"].state_dict().items():
        assert torch.equal(value, old["model"].state_dict()[key])
    artifacts = learned.save_learned_bundle(old, tmp_path)
    assert "recurrent_context_mode" not in learned._event_distribution_metadata(
        old["model"], missing
    )
    loaded = learned.load_learned_bundle(
        {"dir": tmp_path, "engine_id": "gru", "params": missing, "artifacts": artifacts}
    )
    assert_outputs_equal(
        learned.predict_learned(first, validation), learned.predict_learned(loaded, old_validation)
    )


def test_shared_fixed_variable_origins_use_exact_same_legacy_prediction_seed():
    fixed_config = resolved_config(mode="fixed")
    variable_config = resolved_config()
    for fixed, variable in zip(frames(fixed_config), frames(variable_config), strict=True):
        assert_old_frame_fields_equal(fixed, variable)
        for index in range(len(fixed["x"])):
            assert learned._prediction_identity(
                fixed_config, fixed, index
            ) == learned._prediction_identity(variable_config, variable, index)


def test_older_valid_history_changes_moments_without_changing_seed_and_pads_change_neither(
    variable_bundle,
):
    config = resolved_config()
    _, validation = frames(config)
    index = int(np.flatnonzero(validation["history_lengths"] > 8)[-1])
    baseline = slice_frame(validation, np.array([index]))
    seed = learned._prediction_identity(config, baseline, 0)
    model = variable_bundle["model"].eval()

    def moments(frame):
        with torch.no_grad():
            return model(
                torch.as_tensor(learned._normalized(frame, variable_bundle["scaler"])),
                torch.as_tensor(frame["current"]),
                **learned._context_model_kwargs(config, frame),
            )

    reference = moments(baseline)
    changed = copy.deepcopy(baseline)
    length = int(changed["history_lengths"][0])
    changed["x"][0, length - 9] += 1.0
    np.testing.assert_array_equal(
        changed["x"][0, length - 8 : length], baseline["x"][0, length - 8 : length]
    )
    assert learned._prediction_identity(config, changed, 0) == seed
    actual = moments(changed)
    assert any(not torch.equal(value, actual[key]) for key, value in reference.items())
    before = learned.predict_learned(variable_bundle, baseline)
    after = learned.predict_learned(variable_bundle, changed)
    assert any(not np.array_equal(value, after[key]) for key, value in before.items())
    padded = copy.deepcopy(baseline)
    padded["x"][~padded["history_mask"]] = np.nan
    assert learned._prediction_identity(config, padded, 0) == seed
    padded_moments = moments(padded)
    for key, value in reference.items():
        assert torch.equal(value, padded_moments[key])
    assert_outputs_equal(before, learned.predict_learned(variable_bundle, padded))
