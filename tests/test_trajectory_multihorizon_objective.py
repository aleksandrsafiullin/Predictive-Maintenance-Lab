"""Frozen multi-prefix joint scores; these checks are not model-quality evidence."""

import itertools

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.learned_trajectory import (
    _path_objective_prefix_lengths,
    learned_params,
    load_learned_bundle,
    save_learned_bundle,
)
from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import first_entry_loss, path_losses, path_score_surrogates


def _features():
    return pd.DataFrame({
        "unit_id": ["train"] * 7,
        "timestamp_s": np.arange(7) * 60.0,
        "gap_before": [True] + [False] * 6,
    })


def test_config_default_and_exact_grid_binding_copy_requested_horizons():
    config = learned_params("gru", {}, _features())
    assert config["path_objective_horizons_s"] is None
    assert _path_objective_prefix_lengths(config) is None
    requested = [60, 180, 360]
    config = learned_params("gru", {"path_objective_horizons_s": requested}, _features())
    assert config["path_objective_horizons_s"] == [60.0, 180.0, 360.0]
    assert _path_objective_prefix_lengths(config) == (1, 3, 6)
    requested.append(420)
    assert config["path_objective_horizons_s"] == [60.0, 180.0, 360.0]


@pytest.mark.parametrize("invalid", [
    [], True, False, 60, "60", [True], [60, False], [np.bool_(True)],
    [0], [-60], [float("nan")], [float("inf")], ["60"],
    [60, 60], [180, 60], [61], [60, 420], [60.000000001], [[60]],
])
def test_config_rejects_invalid_prefixes(invalid):
    with pytest.raises(ValueError, match="path_objective_horizons_s"):
        learned_params("gru", {"path_objective_horizons_s": invalid}, _features())


@pytest.mark.parametrize("requested,legacy", [(None, False), (None, True), ([60, 180, 360], False)])
def test_checkpoint_config_binding_and_legacy_reload_without_training(tmp_path, requested, legacy):
    config = learned_params("gru", {
        "path_objective_horizons_s": requested, "hidden_size": 16,
        "num_layers": 1, "rank": 2,
    }, _features())
    if legacy:
        del config["path_objective_horizons_s"]
    model = SignalDistribution("gru", 2, 6, hidden_size=16, num_layers=1, rank=2,
                               event_location=config["event_location"]).eval()
    bundle = {
        "model": model, "config": config, "encoder": None,
        "scaler": {"mean": [0., 0.], "std": [1., 1.]},
        "external_scaler": None, "feature_names": ["a", "b"],
        "selection": {"test_feedback": False}, "trace": [],
    }
    artifacts = save_learned_bundle(bundle, tmp_path)
    run = {"dir": tmp_path, "params": config, "artifacts": artifacts, "engine_id": "gru"}
    loaded = load_learned_bundle(run)
    assert loaded["config"] == config
    assert _path_objective_prefix_lengths(loaded["config"]) == (None if requested is None else (1, 3, 6))
    original = model(torch.zeros(1, 4, 2), torch.tensor([.5]))
    restored = loaded["model"](torch.zeros(1, 4, 2), torch.tensor([.5]))
    assert all(torch.equal(original[name], restored[name]) for name in original)
    with pytest.raises(ValueError, match="config differs"):
        load_learned_bundle({**run, "params": {**config, "path_objective_horizons_s": [120.]}})


def _case():
    torch.manual_seed(19)
    model = SignalDistribution("gru", 2, 6, hidden_size=8, num_layers=1, rank=2).eval()
    moments = model(torch.zeros(3, 4, 2), torch.tensor([0.6, 0.9, 0.4]))
    # Give the tiny fixture appreciable early/late entry mass without fitting.
    moments["event_logits"] = torch.tensor([
        [.1, -.2, .4, -.1, .3, .7, .2],
        [.4, -.3, .1, .2, -.1, .5, .6],
        [.2, .1, -.4, .7, .5, -.2, .3],
    ], requires_grad=True)
    target = torch.tensor([
        [1., 2., 4., 5., 7., 9.],
        [float("nan"), float("nan"), 2., 3., float("nan"), float("nan")],
        [float("nan")] * 6,
    ])
    mask = torch.isfinite(target)
    return model, moments, target, mask


def _objective(model, moments, target, mask, prefixes=None, **extra):
    return model.objective(
        moments, target, mask, 3.0, n_samples=16,
        generator=torch.Generator().manual_seed(42),
        path_objective_prefix_lengths=prefixes, **extra,
    )


def test_none_and_explicit_full_have_identical_values_and_gradients():
    model, moments, target, mask = _case()
    default = _objective(model, moments, target, mask)
    full = _objective(model, moments, target, mask, (6,))
    for name in default:
        assert torch.equal(default[name], full[name])
    variables = (moments["event_logits"], model.path_head[-1].weight)
    left = torch.autograd.grad(default["total"].sum(), variables, retain_graph=True)
    right = torch.autograd.grad(full["total"].sum(), variables)
    assert all(torch.equal(a, b) for a, b in zip(left, right))


def test_multi_prefix_arithmetic_and_gradients_use_one_full_joint_sample(monkeypatch):
    model, moments, target, mask = _case()
    paths, entries = model.sample(
        moments, 3.0, 16, torch.Generator().manual_seed(42), return_entry=True,
    )
    sampled_log_prob = moments["event_logits"].log_softmax(-1).gather(1, entries.T).T
    expected = {name: [] for name in ("energy", "width", "miss")}
    for length in (2, 4, 6):
        values = path_losses(paths[..., :length], target[..., :length], mask[..., :length], signal_scale=3.0)
        surrogates = path_score_surrogates(
            paths[..., :length], target[..., :length], mask[..., :length], sampled_log_prob,
            signal_scale=3.0,
        )
        for name in expected:
            expected[name].append(values[name] + surrogates[name] - surrogates[name].detach())
    original_sample = model.sample
    calls = []

    def counted(*args, **kwargs):
        result = original_sample(*args, **kwargs)
        calls.append((kwargs.get("return_entry", False), (result[0] if isinstance(result, tuple) else result).shape))
        return result

    monkeypatch.setattr(model, "sample", counted)
    actual = _objective(model, moments, target, mask, (2, 4, 6))
    # One unconditional full draw and the existing full conditional phase draw.
    assert calls == [(True, (16, 3, 6)), (False, (16, 3, 6))]
    variables = (moments["event_logits"], model.path_head[-1].weight)
    for name, prefix_values in expected.items():
        average = torch.stack(prefix_values).mean(0)
        torch.testing.assert_close(actual[name], average)
        lhs = torch.autograd.grad(actual[name].sum(), variables, retain_graph=True)
        rhs = torch.autograd.grad(average.sum(), variables, retain_graph=True)
        for a, b in zip(lhs, rhs):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-5)
            assert torch.isfinite(a).all() and a.abs().sum() > 0
        assert actual[name][2] == 0
        # Unknown short prefix is still part of the frozen equal average.
        assert prefix_values[0][1] == 0
        torch.testing.assert_close(actual[name][1], (prefix_values[1][1] + prefix_values[2][1]) / 3)


def test_masked_unknown_suffix_and_full_event_phase_likelihood_are_invariant():
    model, moments, target, mask = _case()
    allowed = torch.zeros(3, 7, dtype=torch.bool)
    allowed[0, 4:6] = True  # Recorded crossing bracket beyond the shorter prefix.
    evidence = dict(event_allowed_mask=allowed, event_observed_mask=torch.tensor([True, False, False]),
                    no_entry_prefix=torch.tensor([0, 4, 0]))
    full = _objective(model, moments, target, mask, **evidence)
    multi = _objective(model, moments, target, mask, (2, 4), **evidence)
    changed = torch.where(mask, target, torch.full_like(target, 1e9))
    replaced = _objective(model, moments, changed, mask, (2, 4), **evidence)
    for name in multi:
        assert torch.equal(multi[name], replaced[name])
        assert torch.isfinite(multi[name]).all()
    exact_event = first_entry_loss(moments["event_logits"], **evidence)
    assert torch.equal(multi["event"], exact_event)
    for name in ("event", "phase_pre", "phase_entry", "phase_post", "phase_energy"):
        assert torch.equal(full[name], multi[name])
    for name in ("event", "phase_energy"):
        left = torch.autograd.grad(full[name].sum(), moments["event_logits"], retain_graph=True, allow_unused=True)[0]
        right = torch.autograd.grad(multi[name].sum(), moments["event_logits"], retain_graph=True, allow_unused=True)[0]
        if left is None:
            assert right is None
        else:
            assert torch.equal(left, right)
    gradient = torch.autograd.grad(multi["total"].sum(), model.path_head[-1].weight)[0]
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("term", ["energy", "width", "miss"])
def test_equal_prefix_surrogates_preserve_exact_expected_categorical_gradient(term):
    # Enumerate every outcome of eight independent binary categorical draws.
    draws = torch.tensor(list(itertools.product([0, 1], repeat=8))).T
    low = torch.tensor([.5, .8, 1.1, .7], dtype=torch.float64)
    high = torch.tensor([2., 4., 6., 8.], dtype=torch.float64)
    paths = torch.where(draws[..., None].bool(), high, low)
    mask = torch.ones(paths.shape[1:], dtype=torch.bool)
    target = torch.full(mask.shape, 9.0, dtype=torch.float64)
    theta = torch.tensor(-.7, dtype=torch.float64, requires_grad=True)
    p = theta.sigmoid()
    lp = torch.where(draws.bool(), p.log(), (1 - p).log())
    probability = lp.sum(0).exp()
    values, surrogates = [], []
    for length in (1, 2, 4):
        values.append(path_losses(paths[..., :length], target[..., :length], mask[..., :length])[term])
        surrogates.append(path_score_surrogates(
            paths[..., :length], target[..., :length], mask[..., :length], lp,
        )[term])
    exact = torch.autograd.grad((probability * torch.stack(values).mean(0)).sum(), theta, retain_graph=True)[0]
    estimated = torch.autograd.grad((probability.detach() * torch.stack(surrogates).mean(0)).sum(), theta)[0]
    torch.testing.assert_close(estimated, exact, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("invalid", [[], [True], [np.bool_(True)], [0], [7], [2, 2], [4, 2], [2.0], "2", 2])
def test_model_rejects_invalid_prefix_lengths(invalid):
    model, moments, target, mask = _case()
    with pytest.raises(ValueError, match="Path objective prefixes"):
        _objective(model, moments, target, mask, invalid)


@pytest.mark.parametrize("target_slice,mask_slice", [(2, 2), (6, 2), (2, 6)])
def test_prefix_scoring_still_requires_full_saved_target_and_mask_grid(target_slice, mask_slice):
    model, moments, target, mask = _case()
    with pytest.raises(ValueError, match="full saved batch/horizon grid"):
        _objective(model, moments, target[..., :target_slice], mask[..., :mask_slice], (2,))
