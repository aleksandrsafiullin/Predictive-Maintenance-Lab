"""Prefix event likelihood retains censoring and the full saved distribution."""

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.learned_trajectory import _path_objective_prefix_lengths, learned_params
from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import first_entry_loss


def _evidence():
    logits = torch.tensor([[.1, -.3, .5, .2, -.4, .7, .3]] * 7,
                          dtype=torch.float64, requires_grad=True)
    allowed = torch.zeros(7, 7, dtype=torch.bool)
    allowed[0, 2] = True  # Recorded entry inside the three-step prefix.
    allowed[1, 5] = True  # Recorded late entry proves survival through it.
    allowed[2, 2:5] = True  # A bracket straddles the declared cutoff.
    observed = torch.tensor([True, True, True, False, False, False, False])
    survival = torch.tensor([0, 0, 0, 1, 4, 6, 0])
    return logits, dict(event_allowed_mask=allowed, event_observed_mask=observed,
                        no_entry_prefix=survival)


def test_prefix_likelihood_and_gradients_match_exact_censored_probabilities():
    logits, evidence = _evidence()
    probability = logits.softmax(-1)
    expected = -torch.stack((probability[0, 2], probability[1, 3:].sum(),
                            probability[2, 2:].sum(), probability[3, 1:].sum(),
                            probability[4, 3:].sum(), probability[5, 3:].sum(),
                            probability[6].sum())).log()
    actual = first_entry_loss(logits, prefix_length=3, **evidence)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-14)
    lhs = torch.autograd.grad(actual.sum(), logits, retain_graph=True)[0]
    rhs = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(lhs, rhs, atol=1e-14, rtol=1e-14)
    assert actual[-1] == 0 and not lhs[-1].any()
    # Unknown time beyond a survival prefix is never padded with a negative.
    assert actual[3] < actual[4]


def test_explicit_full_prefix_preserves_legacy_values_and_gradients_exactly():
    logits, evidence = _evidence()
    legacy = first_entry_loss(logits, **evidence)
    explicit = first_entry_loss(logits, prefix_length=6, **evidence)
    assert torch.equal(legacy, explicit)
    lhs = torch.autograd.grad(legacy.sum(), logits, retain_graph=True)[0]
    rhs = torch.autograd.grad(explicit.sum(), logits)[0]
    assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("invalid", [True, np.bool_(True), 0, -1, 7, 3.0, "3", [3]])
def test_invalid_event_prefix_rejected(invalid):
    logits, evidence = _evidence()
    with pytest.raises(ValueError, match="Event objective prefix"):
        first_entry_loss(logits, prefix_length=invalid, **evidence)


def _features():
    return pd.DataFrame({"unit_id": ["Train"] * 7, "timestamp_s": np.arange(7) * 60,
                         "gap_before": [True] + [False] * 6})


def test_event_prefix_config_is_frozen_separately_from_path_prefixes():
    requested = [180, 360]
    config = learned_params("gru", {"event_objective_horizons_s": requested}, _features())
    requested.append(420)
    assert config["event_objective_horizons_s"] == [180., 360.]
    assert _path_objective_prefix_lengths(config, "event_objective_horizons_s") == (3, 6)
    assert config["path_objective_horizons_s"] is None
    assert _path_objective_prefix_lengths({}, "event_objective_horizons_s") is None


@pytest.mark.parametrize("invalid", [[], [True], [0], [60, 60], [180, 60], [61], [420], [np.nan]])
def test_event_prefix_config_rejects_unbound_horizons(invalid):
    with pytest.raises(ValueError, match="event_objective_horizons_s"):
        learned_params("gru", {"event_objective_horizons_s": invalid}, _features())


def test_averaged_event_prefix_changes_only_event_term_with_exact_gradient():
    torch.manual_seed(9)
    model = SignalDistribution("gru", 2, 6, hidden_size=8, num_layers=1, rank=2).eval()
    moments = model(torch.zeros(7, 4, 2), torch.tensor([.6] * 6 + [4.]))
    logits, evidence = _evidence()
    moments["event_logits"] = logits.float()
    target = torch.ones(7, 6)
    target[-1] = float("nan")
    mask = torch.isfinite(target)

    def objective(prefixes):
        return model.objective(moments, target, mask, 3., n_samples=8,
                               generator=torch.Generator().manual_seed(42),
                               event_objective_prefix_lengths=prefixes, **evidence)

    legacy, full, multi = objective(None), objective((6,)), objective((3, 6))
    assert all(torch.equal(legacy[key], full[key]) for key in legacy)
    expected = (first_entry_loss(moments["event_logits"], prefix_length=3, **evidence)
                + first_entry_loss(moments["event_logits"], prefix_length=6, **evidence)) / 2
    expected = expected * (moments["current_signal"] < 3)
    assert torch.equal(multi["event"], expected)
    for key in set(legacy) - {"event", "total"}:
        assert torch.equal(legacy[key], multi[key])
    lhs = torch.autograd.grad(multi["event"].sum(), logits, retain_graph=True)[0]
    rhs = torch.autograd.grad(expected.sum(), logits)[0]
    assert torch.equal(lhs, rhs)


@pytest.mark.parametrize("invalid", [[], [True], [0], [7], [3, 3], [6, 3], [3.0]])
def test_model_rejects_invalid_event_prefixes(invalid):
    model = SignalDistribution("gru", 2, 6, hidden_size=8, num_layers=1, rank=2).eval()
    moments = model(torch.zeros(1, 4, 2), torch.tensor([.6]))
    with pytest.raises(ValueError, match="Event objective prefixes"):
        model.objective(moments, torch.ones(1, 6), torch.ones(1, 6, dtype=torch.bool),
                        3., n_samples=4, event_objective_prefix_lengths=invalid)
