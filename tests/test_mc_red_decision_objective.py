"""Invented synthetic core checks only; no fitting, data access or quality claim."""

import itertools
import math
from numbers import Integral, Real
from unittest.mock import patch

import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import (
    compatible_entry_mask,
    first_entry_loss,
    mc_red_decision_regularizer,
    observed_phase_masks,
    path_losses,
    path_score_surrogates,
    validate_path_band_geometry,
)


# Literal objective frozen before this port, including issued-prefix geometry.
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
    post_factor_smoothness_weight=0.0,
    path_band_geometry="observed_prefix",
):
    validate_path_band_geometry(path_band_geometry)
    if (
        isinstance(post_factor_smoothness_weight, bool)
        or not isinstance(post_factor_smoothness_weight, Real)
        or not math.isfinite(post_factor_smoothness_weight)
        or post_factor_smoothness_weight < 0
    ):
        raise ValueError("Post factor smoothness weight must be finite, nonnegative and nonboolean")
    if post_factor_smoothness_weight > 0 and (
        self.phase_covariance != "separate"
        or getattr(self, "path_distribution", "single") != "single"
    ):
        raise ValueError(
            "Positive post factor smoothness requires separate covariance and single paths"
        )
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
        raise ValueError(
            "Path objective prefixes must be unique increasing integer lengths within the saved horizon"
        )
    if post_factor_smoothness_weight > 0:
        factors = moments.get("post_factors")
        if (
            not isinstance(factors, torch.Tensor)
            or factors.shape != (*expected_shape, self.rank)
            or not torch.isfinite(factors).all()
        ):
            raise ValueError("Post factors must be finite and match [batch, saved horizon, rank]")
        smoothness_threshold = torch.as_tensor(
            red_threshold,
            dtype=factors.dtype,
            device=factors.device,
        ).reshape(-1)
        if (
            smoothness_threshold.numel() not in {1, expected_shape[0]}
            or not torch.isfinite(smoothness_threshold).all()
            or (smoothness_threshold <= 0).any()
        ):
            raise ValueError("RED threshold must be positive, finite and scalar or per-row")
        # Absolute forecast leads, rank-summed adjacent variance, and equal
        # prefix weights. The one-lead zero remains connected to factors.
        smoothness = (
            sum(
                (factors[:, 1:length] - factors[:, : length - 1]).square().sum(-1).mean(-1)
                if length > 1
                else factors[:, :1].sum((1, 2)) * 0
                for length in prefix_lengths
            )
            / len(prefix_lengths)
            / smoothness_threshold.square()
        )
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
            prefix_paths,
            target,
            mask,
            coverage=coverage,
            signal_scale=red_threshold,
            path_band_geometry=path_band_geometry,
        )
        surrogates = path_score_surrogates(
            prefix_paths,
            target,
            mask,
            sampled_log_prob,
            coverage=coverage,
            signal_scale=red_threshold,
            path_band_geometry=path_band_geometry,
        )
        prefix_terms.append(
            {
                name: value + surrogates[name] - surrogates[name].detach()
                for name, value in values.items()
            }
        )
    terms = {
        name: sum(prefix[name] for prefix in prefix_terms) / len(prefix_terms)
        for name in prefix_terms[0]
    }
    event_prefixes = event_objective_prefix_lengths
    if event_prefixes is None:
        event_prefixes = (self.n_horizons,)
    elif (
        not isinstance(event_prefixes, (list, tuple))
        or not event_prefixes
        or any(isinstance(n, bool) or not isinstance(n, Integral) for n in event_prefixes)
        or not 0 < event_prefixes[0]
        or event_prefixes[-1] > self.n_horizons
        or any(a >= b for a, b in zip(event_prefixes, event_prefixes[1:]))
    ):
        raise ValueError(
            "Event objective prefixes must be unique increasing integer lengths within the saved horizon"
        )
    terms["event"] = sum(
        first_entry_loss(
            moments["event_logits"],
            event_allowed_mask=event_allowed_mask,
            event_observed_mask=event_observed_mask,
            no_entry_prefix=no_entry_prefix,
            prefix_length=length,
        )
        for length in event_prefixes
    ) / len(event_prefixes)
    threshold = torch.as_tensor(red_threshold, device=actual_future.device).reshape(-1)
    terms["event"] = terms["event"] * (moments["current_signal"] < threshold)
    # Supervised phase energy prevents rare entry/post locations receiving
    # gradients only when an unconditional draw happens to select that phase.
    # This auxiliary conditional score is separate from the proper joint
    # energy diagnostic above, which keeps the original mixture unchanged.
    allowed, known = compatible_entry_mask(
        moments["event_logits"],
        event_allowed_mask=event_allowed_mask,
        event_observed_mask=event_observed_mask,
        no_entry_prefix=no_entry_prefix,
    )
    conditional = dict(moments)
    conditional["event_logits"] = moments["event_logits"].masked_fill(~allowed, -torch.inf)
    if coupled:
        conditional_paths, _, _, conditional_score = self.sample(
            moments,
            red_threshold,
            n_samples,
            generator,
            entry_allowed_mask=allowed,
            return_latent=True,
        )
    else:
        conditional_paths = self.sample(conditional, red_threshold, n_samples, generator)
    phase_masks = observed_phase_masks(
        allowed,
        known,
        target_mask,
        moments["current_signal"] >= threshold,
    )
    active_phases = torch.zeros_like(terms["energy"])
    phase_sum = torch.zeros_like(terms["energy"])
    for phase, observed_mask in phase_masks.items():
        value = path_losses(
            conditional_paths,
            actual_future,
            observed_mask,
            coverage=coverage,
            signal_scale=red_threshold,
            energy_only=True,
        )["energy"]
        if coupled:
            surrogate = path_score_surrogates(
                conditional_paths,
                actual_future,
                observed_mask,
                conditional_score,
                signal_scale=red_threshold,
                energy_only=True,
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
    if post_factor_smoothness_weight > 0:
        terms["post_factor_smoothness"] = smoothness
        terms["total"] = terms["total"] + post_factor_smoothness_weight * smoothness
    return terms


def regularizer(logits, draws, **kwargs):
    kwargs.setdefault("prefix_length", logits.shape[-1] - 1)
    kwargs.setdefault("already_red", torch.zeros(logits.shape[0], dtype=torch.bool))
    return mc_red_decision_regularizer(logits, draws, **kwargs)


def sorted_cost(sequence, length, admitted, known=True, red=False, coverage=0.9):
    """Independent Python order-statistic oracle, avoiding torch quantile."""
    if red:
        return torch.tensor([0.0, 0.0], dtype=torch.double)
    e = [min(x, length) for x in sequence]
    left = sorted(length + 1 if x == length else x for x in e)
    right = sorted(x + 1 for x in e)
    alpha = (1 - coverage) / 2
    lo = left[math.floor(alpha * (len(e) - 1))]
    hi = right[math.ceil((1 - alpha) * (len(e) - 1))]
    miss = (
        min(max(lo - (length + 1 if j == length else j), 0) + max(j + 1 - hi, 0) for j in admitted)
        if known
        else 0
    )
    return torch.tensor([max(hi - lo, 0), miss], dtype=torch.double)


@pytest.mark.parametrize("samples", [2, 3, 4, 5])
@pytest.mark.parametrize("evidence", ["event", "censored", "unknown", "red"])
def test_enumerated_finite_sample_expected_cost_derivative(samples, evidence):
    logits = torch.tensor([[0.3, -0.2, 0.1]], dtype=torch.double, requires_grad=True)
    q = logits.detach().softmax(-1)[0]
    kwargs = {
        "event": dict(
            event_allowed_mask=torch.tensor([[False, True, False]]),
            event_observed_mask=torch.tensor([True]),
        ),
        "censored": dict(no_entry_prefix=torch.tensor([1])),
        "unknown": {},
        "red": dict(already_red=torch.tensor([True])),
    }[evidence]
    admitted = {"event": [1], "censored": [1, 2], "unknown": [0, 1, 2], "red": [0, 1, 2]}[evidence]
    analytic = torch.zeros(2, 3, dtype=torch.double)
    estimated = torch.zeros_like(analytic)
    for sequence in itertools.product(range(3), repeat=samples):
        draws = torch.tensor(sequence).reshape(samples, 1)
        result = regularizer(logits, draws, **kwargs)
        cost = sorted_cost(sequence, 2, admitted, evidence != "unknown", evidence == "red")
        assert torch.equal(cost, torch.stack([result["width"][0], result["compatible_miss"][0]]))
        weight = q[draws[:, 0]].prod()
        # Exact derivative of product q: counts - S*q. No baseline oracle.
        score = torch.bincount(draws[:, 0], minlength=3) - samples * q
        analytic += weight * cost[:, None] * score[None]
        for i, name in enumerate(("width", "compatible_miss")):
            assert torch.equal(result[name], result[name + "_term"])
            estimated[i] += (
                weight
                * torch.autograd.grad(result[name + "_term"].sum(), logits, retain_graph=True)[0][0]
            )
    torch.testing.assert_close(estimated, analytic, rtol=0, atol=3e-12)


def test_prefix_coarsening_full_scores_priority_unknown_red_and_survival():
    logits = torch.tensor([[0.2, -0.1, 0.4, 0.0, -0.3]], dtype=torch.double, requires_grad=True)
    draws = torch.tensor([[0], [2], [3], [4]])
    result = regularizer(
        logits,
        draws,
        prefix_length=2,
        event_allowed_mask=torch.tensor([[False, False, False, True, False]]),
        event_observed_mask=torch.tensor([True]),
        no_entry_prefix=torch.tensor([4]),
    )
    assert torch.equal(result["compatible_mask"], torch.tensor([[False, False, True]]))
    assert torch.equal(
        result["sampled_full_event_log_score"], logits.log_softmax(-1)[:, draws[:, 0]].T
    )
    assert not torch.equal(
        result["sampled_full_event_log_score"][1], result["sampled_full_event_log_score"][2]
    )
    torch.testing.assert_close(
        result["coarsened_probability"][:, -1], logits.softmax(-1)[:, 2:].sum(-1)
    )
    for count, expected in [
        (0, [True, True, True]),
        (1, [False, True, True]),
        (2, [False, False, True]),
        (4, [False, False, True]),
    ]:
        assert regularizer(logits, draws, prefix_length=2, no_entry_prefix=torch.tensor([count]))[
            "compatible_mask"
        ].tolist() == [expected]
    unknown = regularizer(logits, torch.zeros(4, 1, dtype=torch.long))
    assert unknown["width"].item() == 1 and unknown["compatible_miss"].item() == 0
    survival = regularizer(logits, torch.full((4, 1), 4), no_entry_prefix=torch.tensor([2]))
    assert survival["width"].item() == survival["compatible_miss"].item() == 0
    assert survival["empirical_lower"].item() == 5
    red = regularizer(logits, torch.full((4, 1), -1), already_red=torch.tensor([True]))
    assert red["width"].item() == red["compatible_miss"].item() == 0
    assert (
        torch.autograd.grad((red["width_term"] + red["compatible_miss_term"]).sum(), logits)[0]
        .abs()
        .sum()
        == 0
    )


def case(path="single", phase="shared", h=4):
    torch.manual_seed(811)
    model = SignalDistribution(
        "gru",
        2,
        h,
        hidden_size=4,
        num_layers=1,
        rank=2,
        dropout=0,
        phase_covariance=phase,
        path_distribution=path,
        event_distribution="finite_horizon_mixture",
    ).double()
    moments = model(
        torch.randn(2, 3, 2, dtype=torch.double), torch.tensor([0.2, 1.2], dtype=torch.double)
    )
    actual = torch.ones(2, h, dtype=torch.double)
    return model, moments, actual, torch.ones_like(actual, dtype=torch.bool)


@pytest.mark.parametrize(
    "path,phase",
    [("single", "shared"), ("single", "separate"), ("coupled_timing_mixture", "separate")],
)
@pytest.mark.parametrize("geometry", ["observed_prefix", "issued_prefix"])
def test_literal_legacy_default_zero_values_keys_gradients_rng(path, phase, geometry):
    outputs = []
    for mode in ("legacy", "default", "zero"):
        model, moments, actual, mask = case(path, phase)
        gen = torch.Generator().manual_seed(417)
        state = torch.random.get_rng_state().clone()
        fn = legacy_objective if mode == "legacy" else SignalDistribution.objective
        kwargs = (
            dict(
                red_corridor_width_weight=0.0,
                red_corridor_miss_weight=0.0,
                red_corridor_scale_steps=7.0,
            )
            if mode == "zero"
            else {}
        )
        terms = fn(
            model,
            moments,
            actual,
            mask,
            1.0,
            n_samples=8,
            generator=gen,
            path_band_geometry=geometry,
            path_objective_prefix_lengths=[1, 4],
            event_objective_prefix_lengths=[2, 4],
            **kwargs,
        )
        terms["total"].sum().backward()
        outputs.append(
            (
                terms,
                [p.grad for p in model.parameters()],
                gen.get_state(),
                model.sample(moments, 1.0, 8, gen),
            )
        )
        assert torch.equal(state, torch.random.get_rng_state())
    old = outputs[0]
    for terms, grads, rng, next_samples in outputs[1:]:
        assert terms.keys() == old[0].keys()
        assert all(torch.equal(v, old[0][k]) for k, v in terms.items())
        assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(grads, old[1]))
        assert torch.equal(rng, old[2]) and torch.equal(next_samples, old[3])


@pytest.mark.parametrize("path", ["single", "coupled_timing_mixture"])
def test_positive_reuses_unconditional_draws_marginal_scores_and_equal_event_prefixes(path):
    model, moments, actual, mask = case(path)
    recorded = []
    original = model.sample

    def sample(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("_entries") is None:
            recorded.append((result, kwargs))
        return result

    gen = torch.Generator().manual_seed(79)
    base_gen = torch.Generator().manual_seed(79)
    evidence = dict(
        event_allowed_mask=torch.tensor(
            [[False, False, False, True, False], [True, True, True, True, True]]
        ),
        event_observed_mask=torch.tensor([True, False]),
    )
    base = model.objective(
        moments,
        actual,
        mask,
        1.0,
        n_samples=8,
        generator=base_gen,
        event_objective_prefix_lengths=[1, 4],
        **evidence,
    )
    with patch.object(model, "sample", side_effect=sample):
        terms = model.objective(
            moments,
            actual,
            mask,
            1.0,
            n_samples=8,
            generator=gen,
            red_corridor_width_weight=0.4,
            red_corridor_miss_weight=0.6,
            red_corridor_scale_steps=7.0,
            event_objective_prefix_lengths=[1, 4],
            **evidence,
        )
    assert len(recorded) == 2
    assert recorded[0][1].get("entry_allowed_mask") is None
    assert torch.equal(gen.get_state(), base_gen.get_state())
    entries = recorded[0][0][1]
    if path == "coupled_timing_mixture":
        marginal_cost = terms["red_corridor_width"] + terms["red_corridor_miss"]
        assert (
            torch.autograd.grad(
                marginal_cost.sum(),
                moments["joint_event_logits"],
                allow_unused=True,
                retain_graph=True,
            )[0]
            is None
        )
        assert not torch.equal(
            recorded[0][0][3][..., 0],
            moments["event_logits"]
            .log_softmax(-1)
            .gather(1, entries[:, :1].T.clamp_min(0))
            .T[..., 0],
        )
    expected = [
        regularizer(
            moments["event_logits"],
            entries,
            prefix_length=n,
            already_red=torch.tensor([False, True]),
            **evidence,
        )
        for n in (1, 4)
    ]
    for name, key in [
        ("red_corridor_width", "width_term"),
        ("red_corridor_miss", "compatible_miss_term"),
    ]:
        reference = sum(t[key] for t in expected) / 2 / 7
        assert torch.equal(terms[name], reference)
        torch.testing.assert_close(
            torch.autograd.grad(terms[name].sum(), moments["event_logits"], retain_graph=True)[0],
            torch.autograd.grad(reference.sum(), moments["event_logits"], retain_graph=True)[0],
            rtol=0,
            atol=0,
        )
    assert torch.equal(
        terms["total"],
        terms["energy"]
        + 0.05 * terms["width"]
        + 2 * terms["miss"]
        + terms["event"]
        + 0.5 * terms["phase_energy"]
        + 0.4 * terms["red_corridor_width"]
        + 0.6 * terms["red_corridor_miss"],
    )
    for key in base.keys() - {"total"}:
        assert torch.equal(terms[key], base[key])
    logits = moments["event_logits"]
    old_grad = torch.autograd.grad(base["total"].sum(), logits, retain_graph=True)[0]
    new_grad = torch.autograd.grad(terms["total"].sum(), logits, retain_graph=True)[0]
    prior_grad = torch.autograd.grad(
        (0.4 * terms["red_corridor_width"] + 0.6 * terms["red_corridor_miss"]).sum(),
        logits,
        retain_graph=True,
    )[0]
    torch.testing.assert_close(new_grad - old_grad, prior_grad, rtol=1e-12, atol=1e-12)


def test_H519_S128_float32_bounded_finite_backward_row_independence_no_rng():
    logits = torch.linspace(-1, 1, 520).repeat(3, 1).requires_grad_()
    draws = (torch.arange(128)[:, None].repeat(1, 3) * 3) % 520
    draws[:, 2] = -1
    allowed = torch.zeros(3, 520, dtype=torch.bool)
    allowed[:, 450] = True
    state = torch.random.get_rng_state().clone()
    result = regularizer(
        logits,
        draws,
        event_allowed_mask=allowed,
        event_observed_mask=torch.tensor([True, False, False]),
        already_red=torch.tensor([False, False, True]),
    )
    assert result["compatible_miss"][0] > 0
    assert result["compatible_miss"][1] == 0
    gradient = torch.autograd.grad(
        (result["width_term"] + result["compatible_miss_term"]).sum(), logits, retain_graph=True
    )[0]
    assert gradient.dtype == torch.float32 and torch.isfinite(gradient).all()
    assert not gradient[2].any()
    isolated = torch.autograd.grad(result["width_term"][0], logits)[0]
    assert not isolated[1:].any()
    assert torch.equal(state, torch.random.get_rng_state())
    assert result["audit"]["groups"] == [list(range(a, a + 32)) for a in range(0, 128, 32)]
    for key in ("width", "compatible_miss"):
        assert result[key].dtype == torch.float64
        assert ((result[key] >= 0) & (result[key] <= 520)).all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"red_corridor_width_weight": -1},
        {"red_corridor_width_weight": True},
        {"red_corridor_miss_weight": float("nan")},
        {"red_corridor_miss_weight": float("inf")},
        {"red_corridor_scale_steps": 0},
        {"red_corridor_scale_steps": float("inf")},
        {"red_corridor_scale_steps": False},
        {"n_samples": 1},
        {"n_samples": 2.5},
        {"event_objective_prefix_lengths": [4, 1]},
        {"coverage": 1},
        {"event_observed_mask": torch.tensor([True, False])},
        {"no_entry_prefix": torch.tensor([1.5, 0.0])},
        {
            "event_allowed_mask": torch.zeros(2, 5, dtype=torch.bool),
            "event_observed_mask": torch.tensor([True, False]),
        },
    ],
)
def test_validation_before_rng_or_sampler(kwargs):
    model, moments, actual, mask = case()
    gen = torch.Generator().manual_seed(27)
    state = gen.get_state().clone()
    global_state = torch.random.get_rng_state().clone()
    arguments = dict(n_samples=8, red_corridor_width_weight=1.0, generator=gen)
    arguments.update(kwargs)
    with patch.object(model, "sample", side_effect=AssertionError("sampling before validation")):
        with pytest.raises(ValueError):
            model.objective(moments, actual, mask, 1.0, **arguments)
    assert torch.equal(gen.get_state(), state)
    assert torch.equal(global_state, torch.random.get_rng_state())


@pytest.mark.parametrize(
    "change",
    [
        "half",
        "nan",
        "float_entries",
        "one_sample",
        "negative_entry",
        "too_large_entry",
        "bad_red",
        "bad_allowed",
    ],
)
def test_helper_strict_inputs(change):
    logits = torch.zeros(1, 3)
    draws = torch.zeros(4, 1, dtype=torch.long)
    kwargs = {}
    if change == "half":
        logits = logits.half()
    if change == "nan":
        logits[0, 0] = float("nan")
    if change == "float_entries":
        draws = draws.float()
    if change == "one_sample":
        draws = draws[:1]
    if change == "negative_entry":
        draws.fill_(-1)
    if change == "too_large_entry":
        draws.fill_(3)
    if change == "bad_red":
        kwargs["already_red"] = torch.tensor([0])
    if change == "bad_allowed":
        kwargs.update(event_allowed_mask=torch.ones(1, 3), event_observed_mask=torch.tensor([True]))
    state = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError):
        regularizer(logits, draws, **kwargs)
    assert torch.equal(state, torch.random.get_rng_state())


def test_positive_H519_S128_float32_model_backward_finite():
    model, _, _, _ = case(h=519)
    model = model.float()
    moments = model(torch.zeros(2, 3, 2), torch.tensor([0.2, 1.2]))
    terms = model.objective(
        moments,
        torch.ones(2, 519),
        torch.ones(2, 519, dtype=torch.bool),
        1.0,
        n_samples=128,
        generator=torch.Generator().manual_seed(16),
        red_corridor_width_weight=0.1,
        red_corridor_miss_weight=0.2,
        no_entry_prefix=torch.tensor([400, 0]),
        event_objective_prefix_lengths=[30, 519],
    )
    assert all(torch.isfinite(t).all() for t in terms.values())
    terms["total"].sum().backward()
    assert all(
        p.grad is None or (p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all())
        for p in model.parameters()
    )
