"""Optional optimizer ablation contracts, using isolated synthetic data only."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pdm import learned_trajectory as learned


def _features():
    return pd.DataFrame({"unit_id": ["synthetic"] * 4,
                         "timestamp_s": np.arange(4) * 60., "gap_before": [False] * 4})


def _config(multiplier=1.0):
    return learned.learned_params("gru", {
        "horizons_s": [60., 120., 180.], "history_length": 2,
        "hidden_size": 16, "num_layers": 1, "rank": 2, "epochs": 2,
        "batch_size": 4, "path_samples": 16, "training_samples": 4,
        "cpu_threads": 1, "seed": 913, "dropout": 0.,
        "event_head_learning_rate_multiplier": multiplier,
    }, _features())


@pytest.mark.parametrize("invalid", [True, False, np.bool_(True), None, "2", [], {},
                                    complex(1, 0), 0, -1, float("nan"), float("inf"),
                                    -float("inf")])
def test_invalid_multiplier_rejected(invalid):
    with pytest.raises(ValueError, match="event_head_learning_rate_multiplier"):
        _config(invalid)


@pytest.mark.parametrize("value", [1, 2.5, np.float32(.5), np.int64(3)])
def test_positive_real_multiplier_accepted(value):
    assert _config(value)["event_head_learning_rate_multiplier"] == float(value)


def test_resolved_rate_overflow_rejected():
    with pytest.raises(ValueError, match="Resolved event_head learning rate"):
        learned.learned_params("gru", {"learning_rate": 10.,
                                      "event_head_learning_rate_multiplier": 1e308}, _features())


@pytest.mark.parametrize("legacy_missing", [False, True])
def test_default_optimizer_exact_legacy_steps_and_rng(legacy_missing):
    config = _config()
    if legacy_missing:
        del config["event_head_learning_rate_multiplier"]
    torch.manual_seed(711)
    model = learned._make_model("gru", config, 2, None)
    reference = copy.deepcopy(model)
    before_rng = torch.get_rng_state().clone()
    actual_optimizer = learned._make_optimizer(model, config)
    expected_optimizer = torch.optim.AdamW(reference.parameters(), lr=config["learning_rate"],
                                           weight_decay=config["weight_decay"])
    assert torch.equal(before_rng, torch.get_rng_state())
    assert len(actual_optimizer.param_groups) == 1
    assert [id(p) for p in actual_optimizer.param_groups[0]["params"]] == [id(p) for p in model.parameters()]
    assert actual_optimizer.state_dict() == expected_optimizer.state_dict()
    rng = np.random.default_rng(32)
    expected_rng = np.random.default_rng(32)
    frame = {"x": np.zeros((9, 2, 2)), "physical_unit_id": ["a"] * 9}
    for _ in range(3):
        actual_batches = list(learned._optimizer_batches(frame, 4, rng))
        reference_batches = list(learned._optimizer_batches(frame, 4, expected_rng))
        for (actual_indices, _), (reference_indices, _) in zip(actual_batches, reference_batches):
            np.testing.assert_array_equal(actual_indices, reference_indices)
            actual_optimizer.zero_grad()
            expected_optimizer.zero_grad()
            for index, (actual, expected) in enumerate(zip(model.parameters(), reference.parameters())):
                gradient = torch.full_like(actual, float(actual_indices.sum() + index + 1))
                actual.grad = gradient.clone()
                expected.grad = gradient.clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 5)
            actual_optimizer.step()
            expected_optimizer.step()
    assert rng.bit_generator.state == expected_rng.bit_generator.state
    assert torch.equal(before_rng, torch.get_rng_state())
    for actual, expected in zip(model.parameters(), reference.parameters()):
        assert torch.equal(actual, expected)
        for key in actual_optimizer.state[actual]:
            assert torch.equal(actual_optimizer.state[actual][key], expected_optimizer.state[expected][key])
    assert ("event_head_learning_rate_multiplier" in config) is not legacy_missing


def test_nondefault_groups_exhaustive_identity_and_actual_update():
    config = _config(4)
    model = learned._make_model("gru", config, 2, None).double()
    reference = copy.deepcopy(model)
    optimizer = learned._make_optimizer(model, config)
    legacy = torch.optim.AdamW(reference.parameters(), lr=config["learning_rate"],
                              weight_decay=config["weight_decay"])
    head_ids = {id(p) for p in model.event_head.parameters()}
    groups = optimizer.param_groups
    assert len(groups) == 2
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in model.parameters() if id(p) not in head_ids]
    assert {id(p) for p in groups[1]["params"]} == head_ids
    flattened = [id(p) for group in groups for p in group["params"]]
    assert len(flattened) == len(set(flattened)) == len(list(model.parameters()))
    assert set(flattened) == {id(p) for p in model.parameters()}
    assert [g["lr"] for g in groups] == [config["learning_rate"], 4 * config["learning_rate"]]
    assert all(g["weight_decay"] == config["weight_decay"] for g in groups)
    initial = {name: p.detach().clone() for name, p in model.named_parameters()}
    for actual, expected in zip(model.parameters(), reference.parameters()):
        actual.grad = torch.ones_like(actual)
        expected.grad = torch.ones_like(expected)
    optimizer.step()
    legacy.step()
    for (name, actual), expected in zip(model.named_parameters(), reference.parameters()):
        if id(actual) in head_ids:
            torch.testing.assert_close(initial[name] - actual, 4 * (initial[name] - expected), rtol=1e-10, atol=1e-14)
        else:
            assert torch.equal(actual, expected)
    rates = learned._optimizer_rate_metadata(config)
    assert rates["event_head_decay_per_step"] == 4 * rates["base_decay_per_step"]


def _frame(group, offset):
    x = np.arange(16, dtype=np.float32).reshape(4, 2, 2) / 20 + offset
    observed = np.array([True, False, True, False])
    allowed = np.zeros((4, 4), bool)
    allowed[observed, 1] = True
    allowed[~observed, 3] = True
    y = np.tile([.5, 1.2, 1.4], (4, 1)).astype(np.float32)
    y[~observed] = [.5, .6, .7]
    return {"x": x, "raw_x": x[:, :, :1], "y": y,
            "mask": np.ones((4, 3), bool), "current": np.full(4, .3, np.float32),
            "red_threshold": np.ones(4, np.float32), "event_allowed": allowed,
            "event_observed": observed, "no_entry_prefix": np.where(observed, 0, 3),
            "physical_unit_id": [group] * 4, "feature_names": ["f1", "f2"]}


def test_actual_synthetic_fit_selection_and_exact_saved_reload(tmp_path, monkeypatch):
    config = _config(3)
    train = _frame("synthetic-train", 0)
    validation = _frame("synthetic-validation", .1)
    original_factory = learned._make_optimizer
    groups_seen = []

    def record_factory(model, resolved):
        optimizer = original_factory(model, resolved)
        groups_seen.append([g["lr"] for g in optimizer.param_groups])
        return optimizer

    monkeypatch.setattr(learned, "_make_optimizer", record_factory)
    bundle = learned.fit_learned_model("gru", train, validation, config)
    assert groups_seen == [[config["learning_rate"], config["learning_rate"] * 3]]
    assert len(bundle["trace"]) == 2
    selection = bundle["selection"]
    assert selection["criterion"] == "validation_physical_group_equal_joint_objective"
    assert selection["best_score"] == min(row["validation"]["total"] for row in bundle["trace"])
    assert selection["restored_best_checkpoint"] and not selection["test_feedback"]
    assert not selection["coverage_guarantee"]
    assert selection["optimizer_rates"] == learned._optimizer_rate_metadata(config)
    artifacts = learned.save_learned_bundle(bundle, tmp_path)
    saved = json.loads((tmp_path / "learned_model.json").read_text())
    assert saved["config"] == config
    assert saved["selection"] == selection
    loaded = learned.load_learned_bundle({"dir": tmp_path, "artifacts": artifacts,
                                         "engine_id": "gru", "params": config})
    assert loaded["selection"] == selection and loaded["config"] == config
    for key, value in bundle["model"].state_dict().items():
        assert torch.equal(value, loaded["model"].state_dict()[key])
    before = learned.predict_learned(bundle, validation)
    after = learned.predict_learned(loaded, validation)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_missing_legacy_field_fit_and_bundle_load_preserved(tmp_path):
    config = _config()
    train = _frame("synthetic-train", 0)
    validation = _frame("synthetic-validation", .1)
    explicit = learned.fit_learned_model("gru", train, validation, config)
    legacy_config = {key: value for key, value in config.items()
                     if key != "event_head_learning_rate_multiplier"}
    legacy = learned.fit_learned_model("gru", train, validation, legacy_config)
    assert "event_head_learning_rate_multiplier" not in legacy["config"]
    assert legacy["trace"] == explicit["trace"]
    assert legacy["selection"] == explicit["selection"]
    for key, value in explicit["model"].state_dict().items():
        assert torch.equal(value, legacy["model"].state_dict()[key])
    # Older artifacts lack both the config field and optional resolved provenance.
    del legacy["selection"]["optimizer_rates"]
    artifacts = learned.save_learned_bundle(legacy, tmp_path)
    loaded = learned.load_learned_bundle({"dir": tmp_path, "artifacts": artifacts,
                                         "engine_id": "gru", "params": legacy_config})
    assert loaded["config"] == legacy_config
    assert "optimizer_rates" not in loaded["selection"]
    before = learned.predict_learned(explicit, validation)
    after = learned.predict_learned(loaded, validation)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
