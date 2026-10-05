import io

import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import first_entry_loss, path_losses


def test_exact_first_entry_and_seeded_reload():
    model = SignalDistribution("gru", 2, 6, hidden_size=12, num_layers=1, rank=2)
    model.eval()
    x = torch.zeros(3, 8, 2)
    current = torch.tensor([0.2, 1.0, 3.0])
    moments = model(x, current)
    paths, entry = model.sample(
        moments, 3.0, 80, torch.Generator().manual_seed(20), return_entry=True
    )
    crossing = paths >= 3.0
    first = torch.where(crossing.any(-1), crossing.int().argmax(-1), 6)
    assert torch.equal(first[:, :2], entry[:, :2])
    assert (entry[:, 2] == -1).all()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = SignalDistribution("gru", 2, 6, hidden_size=12, num_layers=1, rank=2)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    again = restored.sample(restored(x, current), 3.0, 80, torch.Generator().manual_seed(20))
    assert torch.equal(paths, again)


@pytest.mark.parametrize("term", ["energy", "width", "miss"])
def test_path_terms_have_effective_gradient(term):
    raw = torch.tensor([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.8]], [[0.7, 1.0]]], requires_grad=True)
    losses = path_losses(raw, torch.tensor([[5.0, 6.0]]), torch.ones(1, 2, dtype=torch.bool))
    losses[term].sum().backward()
    assert torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0


def test_exact_event_bracket_survival_unknown_gradients():
    logits = torch.zeros(4, 4, requires_grad=True)
    allowed = torch.tensor([[False, True, True, False], [False] * 4, [False] * 4, [False] * 4])
    loss = first_entry_loss(
        logits,
        event_allowed_mask=allowed,
        event_observed_mask=torch.tensor([True, False, False, False]),
        no_entry_prefix=torch.tensor([0, 3, 1, 0]),
    )
    assert torch.allclose(loss, torch.tensor([2.0, 4.0, 4.0 / 3.0, 1.0]).log())
    loss.sum().backward()
    assert logits.grad[:3].abs().sum(-1).min() > 0
    assert logits.grad[3].abs().sum() == 0


def test_unknown_rows_zero_and_masked_nan_safe():
    paths = torch.rand(8, 2, 4, requires_grad=True)
    target = torch.full((2, 4), float("nan"))
    terms = path_losses(paths, target, torch.zeros(2, 4, dtype=torch.bool))
    assert all(torch.equal(v, torch.zeros_like(v)) for v in terms.values())
    sum(v.sum() for v in terms.values()).backward()
    assert torch.equal(paths.grad, torch.zeros_like(paths))


def test_core_objective_gradients_and_already_red_exclusion():
    model = SignalDistribution("lstm", 2, 5, hidden_size=10, num_layers=1, rank=2)
    moments = model(torch.zeros(2, 8, 2), torch.tensor([0.2, 3.0]))
    terms = model.objective(
        moments,
        torch.full((2, 5), 8.0),
        torch.ones(2, 5, dtype=torch.bool),
        3.0,
        no_entry_prefix=torch.tensor([5, 5]),
        n_samples=16,
        generator=torch.Generator().manual_seed(4),
    )
    assert terms["event"][0] > 0
    assert terms["event"][1] == 0
    terms["total"].mean().backward()
    assert model.path_head[-1].weight.grad.abs().sum() > 0
    assert model.event_head.weight.grad.abs().sum() > 0


def test_dense_horizon_has_linear_moment_storage():
    model = SignalDistribution("external", 1, 519, hidden_size=8, rank=2, external_feature_size=3)
    result = model(None, torch.tensor([0.1, 0.2]), external_features=torch.zeros(2, 3))
    assert result["mean"].shape == (2, 519)
    assert result["factors"].shape == (2, 519, 2)
    assert torch.isfinite(model.sample(result, 3.0, 4)).all()


@pytest.mark.parametrize("term", ["width", "miss", "energy"])
def test_operational_terms_also_train_entry_probabilities(term):
    model = SignalDistribution("gru", 2, 5, hidden_size=8, num_layers=1, rank=2)
    moments = model(torch.zeros(1, 8, 2), torch.tensor([0.2]))
    moments["event_logits"].retain_grad()
    terms = model.objective(
        moments,
        torch.full((1, 5), 10.0),
        torch.ones(1, 5, dtype=torch.bool),
        3.0,
        n_samples=16,
        generator=torch.Generator().manual_seed(4),
    )
    terms[term].sum().backward()
    assert moments["event_logits"].grad.abs().sum() > 0


def test_unobserved_suffix_cannot_change_loss_or_get_gradients():
    original = torch.rand(20, 2, 5)
    changed = original.clone()
    changed[:, :, 2:] *= 1000
    changed.requires_grad_()
    mask = torch.tensor([[True, True, False, False, False], [False] * 5])
    target = torch.full((2, 5), 5.0)
    left = path_losses(original, target, mask)
    right = path_losses(changed, target, mask)
    for name in left:
        assert torch.equal(left[name], right[name])
    sum(term.sum() for term in right.values()).backward()
    assert changed.grad[:, :, 2:].abs().sum() == 0
    assert changed.grad[:, 1].abs().sum() == 0


def test_unknown_objective_row_has_zero_categorical_gradient():
    model = SignalDistribution("gru", 2, 4, hidden_size=8, num_layers=1, rank=2)
    moments = model(torch.zeros(2, 8, 2), torch.tensor([0.2, 0.2]))
    moments["event_logits"].retain_grad()
    mask = torch.tensor([[True] * 4, [False] * 4])
    terms = model.objective(
        moments, torch.full((2, 4), 5.0), mask, 3.0, generator=torch.Generator().manual_seed(22)
    )
    terms["total"].sum().backward()
    assert moments["event_logits"].grad[1].abs().sum() == 0


def _zero_residual_moments(current, horizon=4):
    batch = len(current)
    return {
        "event_logits": torch.zeros(batch, horizon + 1),
        "mean": torch.zeros(batch, horizon),
        "entry_mean": torch.zeros(batch, horizon),
        "post_mean": torch.zeros(batch, horizon),
        "diagonal_scale": torch.zeros(batch, horizon),
        "factors": torch.zeros(batch, horizon, 2),
        "current_signal": torch.tensor(current),
    }


def test_phase_emissions_are_causally_anchored_and_unbounded_after_entry():
    model = SignalDistribution("gru", 2, 4, hidden_size=8, num_layers=1, rank=2)
    moments = _zero_residual_moments([0.6, 4.0])
    moments["event_logits"][:, :] = -torch.inf
    moments["event_logits"][:, 1] = 0
    paths, entries = model.sample(moments, 3.0, 4, return_entry=True)
    assert torch.allclose(paths[:, 0, 0], torch.full((4,), 0.6))
    assert torch.allclose(paths[:, 0, 1], torch.full((4,), 3.03))
    assert torch.allclose(paths[:, 0, 2:], torch.full((4, 2), 3.0))
    assert torch.allclose(paths[:, 1], torch.full((4, 4), 4.0))
    assert (entries[:, 1] == -1).all()
    moments["post_mean"][:] = 100
    large = model.sample(moments, 3.0, 4)
    assert (large[:, :, 2:] > 99).all()
    assert (large[:, :, 2:] < 110).all()  # Softplus, no exponential tails.


def test_no_entry_class_has_exact_support_even_with_extreme_residuals():
    model = SignalDistribution("gru", 2, 4, hidden_size=8, num_layers=1, rank=2)
    moments = _zero_residual_moments([0.6])
    moments["event_logits"][:] = -torch.inf
    moments["event_logits"][:, -1] = 0
    moments["mean"][:] = 1000
    paths, entries = model.sample(moments, 3.0, 4, return_entry=True)
    assert (entries == 4).all()
    assert (paths < 3).all()
    assert (paths >= 0).all()


def test_physical_loss_scales_linearly_and_red_normalization_is_frozen():
    raw = torch.tensor([[[0.1, 0.2]], [[0.3, 0.4]], [[0.5, 0.8]], [[0.7, 1.0]]])
    actual, mask = torch.tensor([[5.0, 6.0]]), torch.ones(1, 2, dtype=torch.bool)
    base = path_losses(raw, actual, mask, signal_scale=3.0)
    enlarged = path_losses(raw * 100, actual * 100, mask, signal_scale=3.0)
    normalized = path_losses(raw * 100, actual * 100, mask, signal_scale=300.0)
    for term in base:
        assert torch.allclose(enlarged[term], base[term] * 100, atol=1e-4)
        assert torch.allclose(normalized[term], base[term], atol=1e-6)


def test_phase_masks_do_not_bridge_unknown_bracket_or_target_gaps():
    from pdm.trajectory_objectives import compatible_entry_mask, observed_phase_masks

    logits = torch.zeros(3, 6)
    allowed, known = compatible_entry_mask(
        logits,
        event_allowed_mask=torch.tensor([[False, True, True, False, False, False], [False] * 6, [False] * 6]),
        event_observed_mask=torch.tensor([True, False, False]),
        no_entry_prefix=torch.tensor([0, 2, 0]),
    )
    observed = torch.tensor([[True, True, False, False, True], [True, False, True, True, True], [True] * 5])
    masks = observed_phase_masks(allowed, known, observed, torch.tensor([False] * 3))
    assert torch.equal(masks["pre"], torch.tensor([[True, False, False, False, False], [True, False, False, False, False], [False] * 5]))
    assert not masks["entry"].any()
    assert torch.equal(masks["post"], torch.tensor([[False, False, False, False, True], [False] * 5, [False] * 5]))


@pytest.mark.parametrize("phase,location", [("pre", "mean"), ("entry", "entry_mean"), ("post", "post_mean")])
def test_conditional_phase_energy_trains_only_its_emission_location(phase, location):
    model = SignalDistribution("gru", 2, 4, hidden_size=8, num_layers=1, rank=2)
    moments = _zero_residual_moments([0.2])
    for name in ("mean", "entry_mean", "post_mean"):
        moments[name].requires_grad_()
    # Only entry at index one is compatible; inference is still unconditional.
    terms = model.objective(
        moments, torch.tensor([[0.8, 4.0, 5.0, 6.0]]), torch.ones(1, 4, dtype=torch.bool), 3.0,
        event_allowed_mask=torch.tensor([[False, True, False, False, False]]),
        event_observed_mask=torch.tensor([True]), n_samples=8,
        generator=torch.Generator().manual_seed(33),
    )
    terms["phase_" + phase].sum().backward()
    assert moments[location].grad.abs().sum() > 0
    for name in ("mean", "entry_mean", "post_mean"):
        if name != location:
            assert moments[name].grad.abs().sum() == 0


def test_auxiliary_unknown_is_zero_but_already_red_observed_suffix_trains():
    model = SignalDistribution("gru", 2, 4, hidden_size=8, num_layers=1, rank=2)
    moments = _zero_residual_moments([0.2, 3.5])
    mask = torch.tensor([[True] * 4, [False, False, False, True]])
    terms = model.objective(moments, torch.full((2, 4), 6.0), mask, 3.0, n_samples=8)
    assert terms["phase_energy"][0] == 0
    assert terms["phase_energy"][1] > 0
    assert terms["phase_pre"].sum() == 0
    assert terms["phase_entry"].sum() == 0


def test_conditional_unknown_suffix_changes_neither_loss_nor_gradients():
    model = SignalDistribution("gru", 2, 4, hidden_size=8, num_layers=1, rank=2)
    moments = _zero_residual_moments([0.2])
    moments["post_mean"].requires_grad_()
    mask = torch.tensor([[True, True, True, False]])
    allowed = torch.tensor([[False, True, False, False, False]])
    actual = torch.tensor([[0.8, 4.0, 5.0, float("nan")]])
    kwargs = dict(event_allowed_mask=allowed, event_observed_mask=torch.tensor([True]), n_samples=8)
    left = model.objective(moments, actual, mask, 3.0, generator=torch.Generator().manual_seed(8), **kwargs)
    changed = actual.clone()
    changed[:, 3] = 1e9
    right = model.objective(moments, changed, mask, 3.0, generator=torch.Generator().manual_seed(8), **kwargs)
    for name in left:
        assert torch.equal(left[name], right[name])
    right["phase_energy"].sum().backward()
    assert moments["post_mean"].grad[:, 3].abs().sum() == 0


def test_unobserved_nonfinite_sample_suffix_is_removed_before_arithmetic():
    paths = torch.rand(8, 2, 4)
    mask = torch.tensor([[True, True, False, False], [False] * 4])
    left = path_losses(paths, torch.ones(2, 4), mask)
    paths[:, 0, 2] = float("nan")
    paths[:, 0, 3] = float("inf")
    paths[:, 1] = float("nan")
    paths.requires_grad_()
    right = path_losses(paths, torch.ones(2, 4), mask)
    for name in left:
        assert torch.equal(left[name], right[name])
    sum(v.sum() for v in right.values()).backward()
    assert torch.isfinite(paths.grad).all()
    assert paths.grad[:, 0, 2:].abs().sum() == 0
    assert paths.grad[:, 1].abs().sum() == 0
