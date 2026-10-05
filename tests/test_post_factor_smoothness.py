"""Synthetic model engineering checks; no real-data fits or quality claims."""

import copy
import math
from numbers import Integral

import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import (
    compatible_entry_mask,
    first_entry_loss,
    observed_phase_masks,
    path_losses,
    path_score_surrogates,
)


# Literal pre-intervention implementation, frozen before this source edit.
def legacy_objective(
    self,
    moments,
    actual_future,
    target_mask,
    red_threshold,
    *,
    event_allowed_mask=None,
    event_observed_mask=None,
    no_entry_prefix=None,
    n_samples=64,
    generator=None,
    coverage=0.9,
    width_weight=0.05,
    miss_weight=2.0,
    event_weight=1.0,
    energy_weight=1.0,
    phase_weight=0.5,
    path_objective_prefix_lengths=None,
    event_objective_prefix_lengths=None,
):
    expected_shape = (moments["current_signal"].shape[0], self.n_horizons)
    if actual_future.shape != expected_shape or target_mask.shape != expected_shape:
        raise ValueError("Path targets and masks must match the full saved batch/horizon grid")
    # Prefixes are fixed by the saved config, never by observed targets.
    prefix_lengths = path_objective_prefix_lengths
    if prefix_lengths is None:
        prefix_lengths = (self.n_horizons,)
    elif (
        not isinstance(prefix_lengths, (list, tuple))
        or not prefix_lengths
        or any(isinstance(n, bool) or not isinstance(n, Integral) for n in prefix_lengths)
        or not 0 < prefix_lengths[0]
        or prefix_lengths[-1] > self.n_horizons
        or any(a >= b for a, b in zip(prefix_lengths, prefix_lengths[1:]))
    ):
        raise ValueError("Path objective prefixes must be unique increasing integer lengths within the saved horizon")
    # Draw on the full event/path grid once, retaining the same joint paths
    # and categorical scores for every prefix of this unconditional sample.
    coupled = getattr(self, "path_distribution", "single") == "coupled_timing_mixture"
    if coupled:
        paths, entries, _, joint_score = self.sample(
            moments, red_threshold, n_samples, generator, return_latent=True
        )
    else:
        paths, entries = self.sample(
            moments, red_threshold, n_samples, generator, return_entry=True
        )
    # Exact discrete draws require score-function gradients. Within-row
    # independent baselines prevent high-amplitude post-RED rows adding
    # categorical gradient noise to unrelated pre-RED equipment origins.
    log_prob = moments["event_logits"].log_softmax(-1)
    sampled_log_prob = log_prob.gather(1, entries.T.clamp_min(0)).T
    sampled_log_prob = sampled_log_prob * (entries >= 0)
    if coupled:
        sampled_log_prob = joint_score
    prefix_terms = []
    for length in prefix_lengths:
        prefix_paths = paths[..., :length]
        target, mask = actual_future[..., :length], target_mask[..., :length]
        values = path_losses(
            prefix_paths, target, mask, coverage=coverage, signal_scale=red_threshold,
        )
        surrogates = path_score_surrogates(
            prefix_paths, target, mask, sampled_log_prob,
            coverage=coverage, signal_scale=red_threshold,
        )
        prefix_terms.append({
            name: value + surrogates[name] - surrogates[name].detach()
            for name, value in values.items()
        })
    terms = {
        name: sum(prefix[name] for prefix in prefix_terms) / len(prefix_terms)
        for name in prefix_terms[0]
    }
    event_prefixes = event_objective_prefix_lengths
    if event_prefixes is None:
        event_prefixes = (self.n_horizons,)
    elif (
        not isinstance(event_prefixes, (list, tuple)) or not event_prefixes
        or any(isinstance(n, bool) or not isinstance(n, Integral) for n in event_prefixes)
        or not 0 < event_prefixes[0] or event_prefixes[-1] > self.n_horizons
        or any(a >= b for a, b in zip(event_prefixes, event_prefixes[1:]))
    ):
        raise ValueError("Event objective prefixes must be unique increasing integer lengths within the saved horizon")
    terms["event"] = sum(first_entry_loss(
        moments["event_logits"], event_allowed_mask=event_allowed_mask,
        event_observed_mask=event_observed_mask, no_entry_prefix=no_entry_prefix,
        prefix_length=length,
    ) for length in event_prefixes) / len(event_prefixes)
    threshold = torch.as_tensor(red_threshold, device=actual_future.device).reshape(-1)
    terms["event"] = terms["event"] * (moments["current_signal"] < threshold)
    # Supervised phase energy prevents rare entry/post locations receiving
    # gradients only when an unconditional draw happens to select that phase.
    # This auxiliary conditional score is separate from the proper joint
    # energy diagnostic above, which keeps the original mixture unchanged.
    allowed, known = compatible_entry_mask(
        moments["event_logits"], event_allowed_mask=event_allowed_mask,
        event_observed_mask=event_observed_mask, no_entry_prefix=no_entry_prefix,
    )
    conditional = dict(moments)
    conditional["event_logits"] = moments["event_logits"].masked_fill(~allowed, -torch.inf)
    if coupled:
        conditional_paths, _, _, conditional_score = self.sample(
            moments, red_threshold, n_samples, generator,
            entry_allowed_mask=allowed, return_latent=True,
        )
    else:
        conditional_paths = self.sample(conditional, red_threshold, n_samples, generator)
    phase_masks = observed_phase_masks(
        allowed, known, target_mask, moments["current_signal"] >= threshold,
    )
    active_phases = torch.zeros_like(terms["energy"])
    phase_sum = torch.zeros_like(terms["energy"])
    for phase, observed_mask in phase_masks.items():
        value = path_losses(
            conditional_paths, actual_future, observed_mask,
            coverage=coverage, signal_scale=red_threshold, energy_only=True,
        )["energy"]
        if coupled:
            surrogate = path_score_surrogates(
                conditional_paths, actual_future, observed_mask, conditional_score,
                signal_scale=red_threshold, energy_only=True,
            )["energy"]
            value = value + surrogate - surrogate.detach()
        terms["phase_" + phase] = value
        phase_sum = phase_sum + value
        active_phases = active_phases + observed_mask.any(-1)
    terms["phase_energy"] = phase_sum / active_phases.clamp_min(1)
    terms["total"] = (
        energy_weight * terms["energy"]
        + width_weight * terms["width"]
        + miss_weight * terms["miss"]
        + event_weight * terms["event"]
        + phase_weight * terms["phase_energy"]
    )
    return terms


def make_case(h=3, phase='separate', path='single'):
    torch.manual_seed(412)
    model = SignalDistribution('gru', 2, h, hidden_size=4, num_layers=1,
                               rank=2, dropout=0, phase_covariance=phase,
                               path_distribution=path,
                               event_distribution='finite_horizon_mixture').double()
    x = torch.randn(2, 2, 2, dtype=torch.double)
    moments = model(x, torch.tensor([0.3, 0.5], dtype=torch.double))
    actual = torch.ones(2, h, dtype=torch.double)
    mask = torch.ones_like(actual, dtype=torch.bool)
    return model, moments, actual, mask


def objective(case, **kwargs):
    model, moments, actual, mask = case
    threshold = kwargs.pop('red_threshold', torch.tensor([2., 4.], dtype=torch.double))
    kwargs.setdefault('generator', torch.Generator().manual_seed(74))
    return model.objective(moments, actual, mask, threshold, n_samples=8, **kwargs)


def penalty(case, **kwargs):
    return objective(case, post_factor_smoothness_weight=5., **kwargs)['post_factor_smoothness']


def test_rank_sum_adjacent_mean_row_thresholds_equal_prefix_weights():
    case = make_case()
    factors = torch.tensor([[[1., 2.], [3., 5.], [4., 1.]],
                            [[0., 1.], [2., 2.], [5., 6.]]], dtype=torch.double,
                           requires_grad=True)
    case[1]['post_factors'] = factors
    # Pair squared rank sums are [13,17] and [5,25].
    torch.testing.assert_close(penalty(case), torch.tensor([15/4, 15/16], dtype=torch.double))
    expected = torch.tensor([(0 + 13 + 15)/3/4, (0 + 5 + 15)/3/16], dtype=torch.double)
    torch.testing.assert_close(penalty(case, path_objective_prefix_lengths=[1, 2, 3]), expected)
    old = objective(case)
    new = objective(case, post_factor_smoothness_weight=5.)
    for key in old.keys() - {'total'}:
        assert torch.equal(old[key], new[key])
    assert torch.equal(new['total'], old['total'] + 5 * new['post_factor_smoothness'])


@pytest.mark.parametrize('h', [1, 3])
def test_constant_factors_connected_zero(h):
    case = make_case(h)
    factors = torch.ones(2, h, 2, dtype=torch.double, requires_grad=True)
    case[1]['post_factors'] = factors
    value = penalty(case)
    assert torch.equal(value, torch.zeros(2, dtype=torch.double))
    gradient, = torch.autograd.grad(value.sum(), factors)
    assert torch.equal(gradient, torch.zeros_like(factors))


def test_exact_and_finite_difference_gradients_only_post_factors():
    case = make_case()
    factors = torch.tensor([[[1., 2.], [3., 5.], [4., 1.]],
                            [[0., 1.], [2., 2.], [5., 6.]]], dtype=torch.double,
                           requires_grad=True)
    case[1]['post_factors'] = factors
    value = penalty(case, path_objective_prefix_lengths=[1, 2, 3]).sum()
    unrelated = [v for k, v in case[1].items() if k != 'post_factors' and v.requires_grad]
    gradients = torch.autograd.grad(value, [factors, *unrelated], allow_unused=True)
    expected = torch.zeros_like(factors)
    for length in (2, 3):
        delta = factors.detach()[:, 1:length] - factors.detach()[:, :length-1]
        pair_grad = 2 * delta / (length-1) / 3 / torch.tensor([4., 16.])[:, None, None]
        expected[:, 1:length] += pair_grad
        expected[:, :length-1] -= pair_grad
    torch.testing.assert_close(gradients[0], expected, rtol=1e-14, atol=1e-14)
    assert all(g is None for g in gradients[1:])
    epsilon = 1e-5
    for index in [(0, 0, 0), (0, 1, 1), (1, 2, 1)]:
        with torch.no_grad():
            factors[index] += epsilon
        plus = penalty(case, path_objective_prefix_lengths=[1, 2, 3]).sum().item()
        with torch.no_grad():
            factors[index] -= 2 * epsilon
        minus = penalty(case, path_objective_prefix_lengths=[1, 2, 3]).sum().item()
        with torch.no_grad():
            factors[index] += epsilon
        assert math.isclose((plus-minus)/(2*epsilon), expected[index].item(), abs_tol=1e-9)


def test_penalty_independent_of_future_labels_masks_evidence_and_rng():
    case = make_case()
    expected = penalty(case).detach()
    model, moments, actual, mask = case
    altered = (model, moments, torch.full_like(actual, float('nan')), torch.zeros_like(mask))
    allowed = torch.zeros(2, 4, dtype=torch.bool)
    allowed[:, -1] = True
    observed = torch.ones(2, dtype=torch.bool)
    value = penalty(altered, event_allowed_mask=allowed, event_observed_mask=observed,
                    no_entry_prefix=torch.tensor([3, 3]), generator=torch.Generator().manual_seed(987))
    assert torch.equal(expected, value)


@pytest.mark.parametrize('phase,path', [('shared', 'single'), ('separate', 'single'),
                                      ('separate', 'coupled_timing_mixture')])
def test_literal_default_and_zero_legacy_keys_values_rng_samples_gradients(phase, path):
    def run(kind):
        case = make_case(phase=phase, path=path)
        model, moments, actual, mask = case
        before = copy.deepcopy(model.state_dict())
        generator = torch.Generator().manual_seed(888)
        global_before = torch.random.get_rng_state().clone()
        fn = legacy_objective if kind == 'legacy' else SignalDistribution.objective
        kwargs = {'post_factor_smoothness_weight': 0.} if kind == 'zero' else {}
        terms = fn(model, moments, actual, mask, torch.tensor([2., 4.]), n_samples=8,
                   generator=generator, path_objective_prefix_lengths=[1, 3], **kwargs)
        terms['total'].sum().backward()
        state = generator.get_state().clone()
        global_after = torch.random.get_rng_state().clone()
        samples = model.sample(moments, torch.tensor([2., 4.]), 8, generator)
        assert torch.equal(global_before, global_after)
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
        gradients = {k: None if v.grad is None else v.grad.clone() for k, v in model.named_parameters()}
        return terms, state, samples, gradients
    baseline = run('legacy')
    for kind in ['default', 'zero']:
        terms, state, samples, gradients = run(kind)
        assert terms.keys() == baseline[0].keys()
        assert all(torch.equal(v, baseline[0][k]) for k, v in terms.items())
        assert torch.equal(state, baseline[1])
        assert torch.equal(samples, baseline[2])
        assert all((v is None and baseline[3][k] is None) or
                   (v is not None and torch.equal(v, baseline[3][k])) for k, v in gradients.items())


@pytest.mark.parametrize('weight', [-1., float('nan'), float('inf'), True, False, '5', None])
def test_invalid_coefficient_rejected(weight):
    with pytest.raises(ValueError, match='weight'):
        objective(make_case(), post_factor_smoothness_weight=weight)


@pytest.mark.parametrize('phase,path', [('shared', 'single'), ('separate', 'coupled_timing_mixture')])
def test_positive_family_admission(phase, path):
    with pytest.raises(ValueError, match='requires'):
        penalty(make_case(phase=phase, path=path))


@pytest.mark.parametrize('threshold', [0., -1., float('nan'), float('inf'), [1., 2., 3.]])
def test_threshold_validation(threshold):
    with pytest.raises(ValueError, match='threshold'):
        penalty(make_case(), red_threshold=threshold)


@pytest.mark.parametrize('bad', [None, torch.ones(2, 3, 1), torch.ones(2, 2, 2),
                                 torch.full((2, 3, 2), float('nan'))])
def test_factor_validation(bad):
    case = make_case()
    case[1]['post_factors'] = bad
    with pytest.raises(ValueError, match='Post factors'):
        penalty(case)

def test_single_lead_prefix_connected_zero_with_nonconstant_full_factors():
    case = make_case()
    factors = torch.arange(12., dtype=torch.double).reshape(2, 3, 2).requires_grad_()
    case[1]['post_factors'] = factors
    value = penalty(case, path_objective_prefix_lengths=[1])
    assert torch.equal(value, torch.zeros(2, dtype=torch.double))
    assert torch.equal(torch.autograd.grad(value.sum(), factors)[0], torch.zeros_like(factors))


def test_positive_does_not_consume_rng_and_total_gradient_delta_is_weighted_penalty():
    case = make_case()
    first_generator = torch.Generator().manual_seed(77)
    second_generator = torch.Generator().manual_seed(77)
    old = objective(case, generator=first_generator)
    new = objective(case, generator=second_generator, post_factor_smoothness_weight=5.)
    assert torch.equal(first_generator.get_state(), second_generator.get_state())
    tensors = [value for value in case[1].values() if value.requires_grad]
    old_grad = torch.autograd.grad(old['total'].sum(), tensors, retain_graph=True, allow_unused=True)
    new_grad = torch.autograd.grad(new['total'].sum(), tensors, retain_graph=True, allow_unused=True)
    smooth_grad = torch.autograd.grad(new['post_factor_smoothness'].sum(), tensors, allow_unused=True)
    for a, b, c in zip(old_grad, new_grad, smooth_grad):
        if c is None:
            assert (a is None and b is None) or torch.equal(a, b)
        else:
            torch.testing.assert_close(b - a, 5 * c, rtol=1e-11, atol=1e-13)


def test_scalar_threshold_normalization():
    case = make_case()
    torch.testing.assert_close(penalty(case, red_threshold=2.),
                               penalty(case) * torch.tensor([1., 4.], dtype=torch.double))
