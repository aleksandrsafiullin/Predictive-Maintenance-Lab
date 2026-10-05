"""Synthetic optimizer contract checks; no real data or model-quality evidence."""

import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from pdm import learned_trajectory as learned


def _features():
    return pd.DataFrame({"unit_id": ["a"] * 4, "timestamp_s": np.arange(4) * 60.,
                         "gap_before": [True, False, False, False]})


def _frame():
    return {
        "physical_unit_id": ["short", "medium", "medium", "long", "long", "long"],
        "x": np.arange(12, dtype=np.float32).reshape(6, 2, 1),
        "mask": np.array([[True], [True], [False], [False], [False], [False]]),
        "event_observed": np.array([True, False, False, False, False, False]),
        "no_entry_prefix": np.array([0, 0, 0, 1, 0, 0]),
        "current": np.array([1., 2., 3., 4., 5., 11.], np.float32),
        "red_threshold": np.full(6, 10., np.float32),
    }


def test_default_and_explicit_legacy_config_and_rng_match():
    default = learned.learned_params("gru", {}, _features())
    explicit = learned.learned_params("gru", {"batch_sampling": "row_permutation"}, _features())
    assert default == explicit
    assert default["batch_sampling"] == "row_permutation"
    frame = _frame()
    rng = np.random.default_rng(732)
    reference = np.random.default_rng(732)
    for _ in range(3):
        batches = list(learned._optimizer_batches(frame, 4, rng))
        expected = np.array_split(reference.permutation(6), 2)
        assert all(p is None and np.array_equal(idx, old)
                   for (idx, p), old in zip(batches, expected))
        assert len(batches) == len(expected)
    assert rng.bit_generator.state == reference.bit_generator.state


@pytest.mark.parametrize("invalid", [None, True, 1, [], {}, "weighted", "PHYSICAL_GROUP"])
def test_config_rejects_invalid_sampling(invalid):
    with pytest.raises(ValueError, match="batch_sampling"):
        learned.learned_params("gru", {"batch_sampling": invalid}, _features())


def test_sampler_budget_stratification_seed_and_identifier_spelling():
    frame = _frame()
    renamed = {**frame, "physical_unit_id": ["z", "a", "a", "m", "m", "m"],
               "current": frame["current"] * 999, "mask": ~frame["mask"]}
    for size in (1, 2, 4, 6, 10):
        batches = list(learned._optimizer_batches(frame, size, np.random.default_rng(31), "physical_group"))
        repeat = list(learned._optimizer_batches(renamed, size, np.random.default_rng(31), "physical_group"))
        expected_sizes = [len(b) for b in np.array_split(np.arange(6), int(np.ceil(6 / size)))]
        assert [len(idx) for idx, _ in batches] == expected_sizes
        assert sum(len(idx) for idx, _ in batches) == 6
        for (idx, p), (other_idx, other_p) in zip(batches, repeat):
            assert np.array_equal(idx, other_idx) and np.array_equal(p, other_p)
            groups = np.array([0, 1, 1, 2, 2, 2])[idx]
            counts = np.bincount(groups, minlength=3)
            assert counts.max() - counts.min() <= 1
            assert np.allclose(p, np.array([1/3, 1/6, 1/6, 1/9, 1/9, 1/9])[idx])
    # Replacement is an actual sampler property, including singleton-group draws.
    singleton = {"x": np.zeros((7, 1, 1)), "physical_unit_id": ["tiny"] + ["large"] * 6}
    idx, _ = next(learned._optimizer_batches(singleton, 7, np.random.default_rng(31), "physical_group"))
    assert np.count_nonzero(idx == 0) >= 3


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4])
def test_exhaustive_stratified_estimator_matches_each_evidence_objective_and_gradient(batch_size):
    frame = _frame()
    members = [[0], [1, 2], [3, 4, 5]]
    probabilities = np.array([1/3, 1/6, 1/6, 1/9, 1/9, 1/9], np.float32)
    q, remainder = divmod(batch_size, 3)
    group_sequences = [groups for groups in itertools.product(range(3), repeat=batch_size)
                       if sorted(groups.count(g) for g in range(3)) ==
                       [q] * (3 - remainder) + [q + 1] * remainder]
    weights = learned._objective_weights(frame)
    values = np.arange(1, 7, dtype=float) ** 2
    expected = {key: np.dot(weight, values) for key, weight in weights.items()}
    accumulated = dict.fromkeys(weights, 0.)
    gradients = dict.fromkeys(weights, 0.)
    mass = 0.
    # Exhaust all allowed group-slot permutations and all within-group row draws.
    for groups in group_sequences:
        for indices in itertools.product(*(members[g] for g in groups)):
            probability = 1 / len(group_sequences) * np.prod([1 / len(members[g]) for g in groups])
            mass += probability
            idx = np.asarray(indices)
            for key, weight in weights.items():
                theta = torch.tensor(1., dtype=torch.float64, requires_grad=True)
                loss = theta * torch.as_tensor(values[idx])
                estimate = learned._optimizer_term(loss, weight, idx, probabilities[idx])
                accumulated[key] += probability * estimate.item()
                gradients[key] += probability * torch.autograd.grad(estimate, theta)[0].item()
    assert mass == pytest.approx(1.)
    assert len(set(round(v, 5) for v in expected.values())) > 1
    for key in weights:
        assert accumulated[key] == pytest.approx(expected[key], rel=2e-7)
        assert gradients[key] == pytest.approx(expected[key], rel=2e-7)


@pytest.mark.parametrize("sampling", ["row_permutation", "physical_group", "legacy_missing"])
def test_optimizer_loop_gradient_scale_epoch_report_and_provenance(monkeypatch, sampling):
    frame = _frame()
    frame.update(y=np.zeros((6, 1), np.float32), event_allowed=np.ones((6, 1), bool),
                 feature_names=["sensor"])
    mode = "row_permutation" if sampling == "legacy_missing" else sampling
    config = learned.learned_params("gru", {"batch_sampling": mode, "horizons_s": [60.],
                                            "epochs": 1, "batch_size": 4}, _features())
    if sampling == "legacy_missing":
        del config["batch_sampling"]
    config["learning_rate"] = 0.  # Synthetic fixed-parameter spy; no model fitting.

    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.theta = torch.nn.Parameter(torch.tensor(1.))

        def forward(self, inputs, current, external):
            assert inputs.dtype == torch.float32 and inputs.shape[-1] == 1
            assert external is None
            return current

        def objective(self, moments, *args, **kwargs):
            return {key: self.theta * moments.square() for key in learned._objective_weights(frame)}

    spy = Spy()
    gradients = []
    monkeypatch.setattr(learned, "_make_model", lambda *args: spy)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_",
                        lambda parameters, bound: gradients.append(float(spy.theta.grad)) or abs(gradients[-1]))
    bundle = learned.fit_learned_model("gru", frame, frame, config)
    weights = learned._objective_weights(frame)
    coefficients = {key: config["phase_weight" if key == "phase_energy" else key + "_weight"]
                    for key in weights}
    expected_totals = dict.fromkeys(weights, 0.)
    expected_gradients = []
    for idx, p in learned._optimizer_batches(frame, 4, np.random.default_rng(config["seed"]), mode):
        terms = {key: learned._optimizer_term(torch.as_tensor(frame["current"][idx] ** 2), weight, idx, p).item()
                 for key, weight in weights.items()}
        expected_gradients.append(sum(coefficients[key] * terms[key] for key in weights) *
                                  (6 / len(idx) if p is None else 1))
        for key in weights:
            expected_totals[key] += terms[key] * (1 if p is None else len(idx) / 6)
    assert gradients == pytest.approx(expected_gradients)
    for key in weights:
        assert bundle["trace"][0]["train"][key] == pytest.approx(expected_totals[key])
        assert bundle["trace"][0]["validation"][key] == pytest.approx(np.dot(weights[key], frame["current"] ** 2))
    assert bundle["trace"][0]["train"]["total"] == pytest.approx(sum(coefficients[key] * expected_totals[key] for key in weights))
    assert bundle["selection"]["batch_sampling"] == mode
    assert bundle["selection"]["optimizer_sampling_policy"]
    assert bundle["selection"]["test_feedback"] is False
