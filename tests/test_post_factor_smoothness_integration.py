"""Tiny synthetic engineering checks; no real data or predictive-quality evidence."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pdm import learned_trajectory as learned
from pdm.io_util import sha256_file
from pdm.models.signal_distribution import SignalDistribution

TERM = "post_factor_smoothness"
CONTRACT = TERM + "_contract"


def config(engine="gru", **overrides):
    features = pd.DataFrame(
        {"unit_id": ["synthetic"] * 5, "timestamp_s": np.arange(5) * 60.0, "gap_before": False}
    )
    return learned.learned_params(
        engine,
        {
            "horizons_s": [60.0, 120.0, 180.0],
            "history_length": 2,
            "hidden_size": 16,
            "num_layers": 1,
            "rank": 2,
            "epochs": 3,
            "batch_size": 5,
            "training_samples": 8,
            "path_samples": 16,
            "cpu_threads": 1,
            "seed": 927,
            "dropout": 0.0,
            "min_delta": 0.0,
            "phase_covariance": "separate",
            "path_distribution": "single",
            "event_distribution": "finite_horizon_mixture",
            "path_objective_horizons_s": [60.0, 180.0],
            "event_objective_horizons_s": [60.0, 180.0],
            "post_factor_smoothness_weight": 5.0,
            **overrides,
        },
        features,
    )


def frame(offset=0.0):
    x = np.arange(20, dtype=np.float32).reshape(5, 2, 2) / 20 + offset
    # Unequal unit sizes and targetless origins in both units intentionally
    # distinguish the causal prior population from evidence-weighted terms.
    mask = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0], [1, 1, 1], [0, 0, 0]], bool)
    return {
        "x": x,
        "raw_x": x[:, :, :1],
        "y": np.tile([0.5, 0.7, 0.9], (5, 1)).astype(np.float32),
        "mask": mask,
        "current": np.arange(5, dtype=np.float32) / 20 + 0.2,
        "red_threshold": np.ones(5, np.float32),
        "event_allowed": np.array([[0, 0, 0, 1]] * 5, bool),
        "event_observed": np.zeros(5, bool),
        "no_entry_prefix": np.array([3, 2, 0, 3, 0]),
        "physical_unit_id": ["A", "A", "A", "B", "B"],
        "feature_names": ["f1", "f2"],
    }


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


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_tiny_fit_augmented_trace_checkpoint_and_exact_reload(engine, tmp_path, monkeypatch):
    cfg = config(engine)
    snapshots = []
    original = learned._make_model
    live = []

    def factory(*args, **kwargs):
        model = original(*args, **kwargs)
        live.append(model)
        return model

    monkeypatch.setattr(learned, "_make_model", factory)

    def report(record):
        if record["stage"] == "training":
            snapshots.append(copy.deepcopy(live[0].state_dict()))

    bundle = learned.fit_learned_model(engine, frame(), frame(0.1), cfg, report=report)
    coefficients = {k: cfg[k + "_weight"] for k in ("energy", "width", "miss", "event")}
    coefficients.update(phase_energy=cfg["phase_weight"], post_factor_smoothness=5.0)
    for record in bundle["trace"]:
        for phase in ("train", "validation"):
            terms = record[phase]
            assert terms[TERM] > 0
            assert all(np.isfinite(v) for v in terms.values())
            assert terms["total"] == pytest.approx(
                sum(coefficients[k] * terms[k] for k in coefficients), rel=2e-6
            )
    selection = bundle["selection"]
    assert selection["best_score"] == min(r["validation"]["total"] for r in bundle["trace"])
    assert selection["restored_best_checkpoint"] and not selection["test_feedback"]
    for key, value in bundle["model"].state_dict().items():
        assert torch.equal(value, snapshots[selection["best_epoch"] - 1][key])
    fields = learned._event_distribution_metadata(bundle["model"], cfg)
    assert selection[CONTRACT] == fields[CONTRACT]
    artifacts = learned.save_learned_bundle(bundle, tmp_path)
    loaded = learned.load_learned_bundle(
        {"dir": tmp_path, "engine_id": engine, "params": cfg, "artifacts": artifacts, **fields}
    )
    for key, value in bundle["model"].state_dict().items():
        assert torch.equal(value, loaded["model"].state_dict()[key])
    expected = learned.predict_learned(bundle, frame(0.1))
    actual = learned.predict_learned(loaded, frame(0.1))
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    assert actual["paths"].shape == (16, 5, 3)
    baseline = original(engine, {**cfg, "post_factor_smoothness_weight": 0.0}, 2, None)
    assert baseline.state_dict().keys() == bundle["model"].state_dict().keys()
    assert sum(p.numel() for p in baseline.parameters()) == sum(
        p.numel() for p in bundle["model"].parameters()
    )
    assert baseline.path_head[-1].out_features == bundle["model"].path_head[-1].out_features
    # Prediction reads the same sampler regardless of the training coefficient.
    zero_prediction_bundle = {**bundle, "config": {**cfg, "post_factor_smoothness_weight": 0.0}}
    for key, value in learned.predict_learned(zero_prediction_bundle, frame(0.1)).items():
        np.testing.assert_array_equal(value, expected[key])


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_restoration_really_replaces_a_later_checkpoint(engine, monkeypatch):
    snapshots, models = [], []
    factory = learned._make_model

    def capture_model(*args, **kwargs):
        model = factory(*args, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(learned, "_make_model", capture_model)

    def report(record):
        if record["stage"] == "training":
            snapshots.append(copy.deepcopy(models[0].state_dict()))

    bundle = learned.fit_learned_model(
        engine, frame(), frame(0.1), config(engine, min_delta=1e10), report=report
    )
    assert bundle["selection"]["best_epoch"] == 1
    assert len(snapshots) == 3
    assert any(not torch.equal(snapshots[0][k], snapshots[-1][k]) for k in snapshots[0])
    for key, value in bundle["model"].state_dict().items():
        assert torch.equal(value, snapshots[0][key])
    assert bundle["selection"]["best_score"] == bundle["trace"][0]["validation"]["total"]


@pytest.mark.parametrize("sampling", ["row_permutation", "physical_group"])
def test_prior_optimizer_uses_all_origin_weights_independent_of_evidence(sampling, monkeypatch):
    pointers = set()
    retained_terms = []
    calls = []
    objective = SignalDistribution.objective
    optimizer_term = learned._optimizer_term

    def capture_objective(self, *args, **kwargs):
        terms = objective(self, *args, **kwargs)
        retained_terms.append(terms[TERM])
        pointers.add(terms[TERM].data_ptr())
        return terms

    def capture_term(values, weights, indices, probabilities):
        if values.data_ptr() in pointers:
            calls.append(
                (
                    weights.copy(),
                    indices.copy(),
                    None if probabilities is None else probabilities.copy(),
                )
            )
        return optimizer_term(values, weights, indices, probabilities)

    monkeypatch.setattr(SignalDistribution, "objective", capture_objective)
    monkeypatch.setattr(learned, "_optimizer_term", capture_term)
    train = frame()
    learned.fit_learned_model(
        "gru", train, frame(0.1), config(epochs=1, batch_size=3, batch_sampling=sampling)
    )
    assert calls
    expected = np.array([1 / 6, 1 / 6, 1 / 6, 1 / 4, 1 / 4], np.float32)
    assert not np.array_equal(expected, learned._objective_weights(train)["energy"])
    for weights, indices, probabilities in calls:
        np.testing.assert_array_equal(weights, expected)
        if sampling == "physical_group":
            np.testing.assert_array_equal(probabilities, expected[indices])
        else:
            assert probabilities is None
    assert expected[2] > 0 and expected[4] > 0


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_default_missing_and_zero_fit_rng_state_trace_and_samples_exact(engine, tmp_path):
    default = config(engine, post_factor_smoothness_weight=0.0)
    missing = {k: v for k, v in default.items() if k != "post_factor_smoothness_weight"}
    resolved_default = learned.learned_params(
        engine,
        missing,
        pd.DataFrame(
            {"unit_id": ["s"] * 4, "timestamp_s": np.arange(4) * 60.0, "gap_before": False}
        ),
    )
    assert resolved_default == default
    bundles, rng_states = [], []
    for cfg in (default, missing, resolved_default):
        bundles.append(learned.fit_learned_model(engine, frame(), frame(0.1), cfg))
        rng_states.append(torch.random.get_rng_state().clone())
    for other, rng in zip(bundles[1:], rng_states[1:]):
        assert torch.equal(rng_states[0], rng)
        assert bundles[0]["trace"] == other["trace"]
        assert bundles[0]["selection"] == other["selection"]
        assert TERM not in other["trace"][0]["train"]
        for key, value in bundles[0]["model"].state_dict().items():
            assert torch.equal(value, other["model"].state_dict()[key])
    old = bundles[1]
    artifacts = learned.save_learned_bundle(old, tmp_path)
    metadata = json.loads((tmp_path / "learned_model.json").read_text())
    assert CONTRACT not in metadata and CONTRACT not in metadata["selection"]
    loaded = learned.load_learned_bundle(
        {"dir": tmp_path, "engine_id": engine, "params": missing, "artifacts": artifacts}
    )
    expected = learned.predict_learned(bundles[0], frame(0.1))
    for bundle in [*bundles[1:], loaded]:
        for key, value in learned.predict_learned(bundle, frame(0.1)).items():
            np.testing.assert_array_equal(value, expected[key])


@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), "5", "0", -1.0, float("nan"), float("inf"), -float("inf")],
)
def test_invalid_coefficients_rejected(value):
    with pytest.raises(ValueError):
        config(post_factor_smoothness_weight=value)


@pytest.mark.parametrize(
    "overrides", [{"phase_covariance": "shared"}, {"path_distribution": "coupled_timing_mixture"}]
)
def test_positive_incompatible_families_rejected(overrides):
    with pytest.raises(ValueError):
        config(**overrides)


@pytest.fixture(scope="module")
def positive_bundle():
    return learned.fit_learned_model("gru", frame(), frame(0.1), config())


@pytest.mark.parametrize(
    "surface",
    [
        "learned_model.json",
        "manifest",
        "training_contract.json",
        "model_input_contract.json",
        "selection",
    ],
)
@pytest.mark.parametrize("operation", ["missing", "contradictory"])
def test_rehashed_missing_or_contradictory_policy_rejected(
    positive_bundle, tmp_path, surface, operation
):
    run = save_contracts(positive_bundle, tmp_path)
    target = "learned_model.json" if surface == "selection" else surface
    saved = run if target == "manifest" else json.loads((tmp_path / target).read_text())
    parent = saved["selection"] if surface == "selection" else saved
    if operation == "missing":
        del parent[CONTRACT]
    else:
        parent[CONTRACT]["origin_weighting"] = "future-evidence rows only"
    if target != "manifest":
        (tmp_path / target).write_text(json.dumps(saved))
        run["artifacts"][target] = sha256_file(tmp_path / target)
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


def test_policy_present_on_all_saved_surfaces(positive_bundle, tmp_path):
    run = save_contracts(positive_bundle, tmp_path)
    expected = positive_bundle["selection"][CONTRACT]
    assert run[CONTRACT] == expected
    for name in ("learned_model.json", "training_contract.json", "model_input_contract.json"):
        assert json.loads((tmp_path / name).read_text())[CONTRACT] == expected
    assert learned.load_learned_bundle(run)["selection"][CONTRACT] == expected


@pytest.mark.parametrize("mutation", ["missing", "best_epoch", "policy_weight"])
def test_positive_full_run_selection_copy_is_mandatory(positive_bundle, tmp_path, mutation):
    run = save_contracts(positive_bundle, tmp_path)
    run["schema_version"] = 1
    run["selection"] = copy.deepcopy(positive_bundle["selection"])
    assert learned.load_learned_bundle(run)["selection"] == run["selection"]
    if mutation == "missing":
        del run["selection"]
    elif mutation == "best_epoch":
        run["selection"]["best_epoch"] += 1
    else:
        run["selection"][CONTRACT]["weight"] = 2.0
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize("name", ["training_contract.json", "model_input_contract.json"])
@pytest.mark.parametrize(
    "field,wrong", [("post_factor_smoothness_weight", 0.0), ("path_objective_horizons_s", [180.0])]
)
def test_rehashed_contract_params_cannot_contradict_policy(
    positive_bundle, tmp_path, name, field, wrong
):
    run = save_contracts(positive_bundle, tmp_path)
    saved = json.loads((tmp_path / name).read_text())
    saved["params"] = copy.deepcopy(positive_bundle["config"])
    (tmp_path / name).write_text(json.dumps(saved))
    run["artifacts"][name] = sha256_file(tmp_path / name)
    assert learned.load_learned_bundle(run)["config"] == positive_bundle["config"]
    saved["params"][field] = wrong
    (tmp_path / name).write_text(json.dumps(saved))
    run["artifacts"][name] = sha256_file(tmp_path / name)
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


def test_explicit_selection_without_schema_still_validated(positive_bundle, tmp_path):
    run = save_contracts(positive_bundle, tmp_path)
    run["selection"] = copy.deepcopy(positive_bundle["selection"])
    assert learned.load_learned_bundle(run)["selection"] == run["selection"]
    del run["selection"][CONTRACT]
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


def test_zero_full_run_without_positive_policy_remains_loadable(tmp_path):
    bundle = learned.fit_learned_model(
        "gru", frame(), frame(0.1), config(post_factor_smoothness_weight=0.0)
    )
    run = save_contracts(bundle, tmp_path)
    run.update(schema_version=1, selection=copy.deepcopy(bundle["selection"]))
    assert CONTRACT not in run and CONTRACT not in run["selection"]
    loaded = learned.load_learned_bundle(run)
    for key, value in learned.predict_learned(bundle, frame(0.1)).items():
        np.testing.assert_array_equal(value, learned.predict_learned(loaded, frame(0.1))[key])
