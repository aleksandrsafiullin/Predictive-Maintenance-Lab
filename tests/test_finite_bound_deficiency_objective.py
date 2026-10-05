"""Independent invented CPU inputs; no fit, actual data, or quality claim."""

import math
from unittest.mock import patch

import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import finite_bound_deficiency
from tests.test_mc_red_decision_objective import legacy_objective


def evidence(z, length, admitted=(0,), observed=True, censor=0, red=False, **kwargs):
    allowed = torch.zeros_like(z, dtype=torch.bool)
    allowed[:, list(admitted)] = True
    return finite_bound_deficiency(
        z, prefix_length=length, already_red=torch.full((len(z),), red, device=z.device),
        event_allowed_mask=allowed,
        event_observed_mask=torch.full((len(z),), observed, device=z.device),
        no_entry_prefix=torch.full((len(z),), censor, device=z.device), **kwargs,
    )


@pytest.mark.parametrize("h", [1, 3, 519])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_evidence_and_exclusions(h, dtype):
    z = torch.tensor([[-1000.] * h + [0.]], dtype=dtype, requires_grad=True)
    cases = [(h, (0,), True, 0, False, True),
             (h, tuple(range(h)), True, h, False, True),
             (h, (h,), True, 0, False, False),
             (h, (0,), False, 0, False, False),
             (h, (0,), False, h, False, False),
             (h, (0,), True, 0, True, False)]
    if h > 1:
        cases.extend([(1, (0, 1), True, 0, False, False),
                      (1, (1,), True, 0, False, False),
                      (1, (0,), True, h, False, True)])
    for length, admitted, obs, censor, red, eligible in cases:
        out = evidence(z, length, admitted, obs, censor, red)
        assert bool(out["eligible"].item()) is eligible
        gradient = torch.autograd.grad(out["cost"].sum(), z)[0]
        expected = torch.cat((-torch.softmax(z.double()[:, :length], -1),
                              torch.softmax(z.double()[:, length:], -1)), -1)
        if not eligible:
            assert torch.equal(out["cost"], torch.zeros_like(out["cost"]))
            expected = torch.zeros_like(expected)
        torch.testing.assert_close(gradient.double(), expected, rtol=0, atol=1e-7)
        assert out["compatible_prefix"].shape == (1, length + 1)
        assert torch.equal(out["compatible_prefix"][:, -1],
                           out["compatible_full"][:, length:].any(-1))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytic_and_finite_difference(dtype):
    z = torch.tensor([[.3, -.2, 1., -.1]], dtype=dtype, requires_grad=True)
    out = evidence(z, 2)
    gradient = torch.autograd.grad(out["cost"].sum(), z)[0]
    expected = torch.cat((-torch.softmax(z.double()[:, :2], -1),
                          torch.softmax(z.double()[:, 2:], -1)), -1)
    torch.testing.assert_close(gradient.double(), expected, rtol=0, atol=1e-7)
    eps = 1 / 256 if dtype == torch.float32 else 1e-4
    for coordinate in range(4):
        offset = torch.zeros_like(z)
        offset[0, coordinate] = eps
        numeric = (evidence(z.detach() + offset, 2)["cost"]
                   - evidence(z.detach() - offset, 2)["cost"]) / (2 * eps)
        torch.testing.assert_close(numeric, gradient[:, coordinate].double(), rtol=0, atol=2e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_offset_repair_and_inactive(dtype):
    offset = 2**25 + 16 if dtype == torch.float32 else 2**54 + 16
    z = torch.tensor([[offset, offset - 4, offset - 4, offset - 4]],
                     dtype=dtype, requires_grad=True)
    literal = torch.relu(torch.logsumexp(z[:, 1:], -1) - z[:, 0] - math.log(.025 / .975))
    assert literal.item() == 0
    out = evidence(z, 1)
    assert out["cost"].item() > 0
    reference = torch.tensor([[0., -4., -4., -4.]], dtype=dtype, requires_grad=True)
    expected = evidence(reference, 1)
    assert torch.equal(out["cost"], expected["cost"])
    assert torch.equal(torch.autograd.grad(out["cost"].sum(), z)[0],
                       torch.autograd.grad(expected["cost"].sum(), reference)[0])
    inactive = torch.tensor([[0., -20., -20., -20.]], dtype=dtype, requires_grad=True)
    cost = evidence(inactive, 1)["cost"]
    assert cost.item() == 0
    assert torch.equal(torch.autograd.grad(cost.sum(), inactive)[0], torch.zeros_like(inactive))


@pytest.mark.parametrize("values,length", [([-1, -1, 1], 2), ([-1, 1, 1], 1),
                                          ([-1, -1, 1, 1], 2), ([-1, 1, -1, -1], 2),
                                          ([-1] * 260 + [1] * 260, 260)])
def test_promoted_float32_repeated_extrema(values, length):
    maximum = torch.finfo(torch.float32).max
    z = torch.tensor([[v * maximum for v in values]], requires_grad=True)
    out = evidence(z, length)
    gradient = torch.autograd.grad(out["cost"].sum(), z)[0]
    expected = torch.cat((-torch.softmax(z.double()[:, :length], -1),
                          torch.softmax(z.double()[:, length:], -1)), -1)
    if out["cost"].item() == 0:
        expected = torch.zeros_like(expected)
    torch.testing.assert_close(gradient.double(), expected, rtol=0, atol=1e-7)
    assert gradient.abs().max() <= 1
    assert gradient.double().abs().sum() <= 2 + 1e-7
    excluded = evidence(z, length, observed=False)["cost"]
    assert excluded.item() == 0
    assert torch.equal(torch.autograd.grad(excluded.sum(), z)[0], torch.zeros_like(z))


def test_float64_actual_overflow_rejects():
    maximum = torch.finfo(torch.float64).max
    with pytest.raises(ValueError, match="overflow"):
        evidence(torch.tensor([[-maximum, maximum]], dtype=torch.float64), 1)


@pytest.mark.parametrize("override", [
    {"prefix_length": True}, {"prefix_length": 0}, {"prefix_length": 4},
    {"coverage": True}, {"coverage": float("nan")}, {"coverage": 1 - 4e-8},
    {"already_red": torch.tensor([0])}, {"already_red": torch.tensor([[False]])},
    {"event_allowed_mask": torch.ones(1, 4, dtype=torch.bool)},
    {"event_allowed_mask": torch.zeros(1, 4, dtype=torch.bool),
     "event_observed_mask": torch.tensor([True])},
    {"no_entry_prefix": torch.tensor([1.5])}, {"no_entry_prefix": torch.tensor([-1])},
    {"no_entry_prefix": torch.tensor([True])}, {"no_entry_prefix": torch.tensor([float("inf")])},
])
def test_malformed_evidence_and_coverage(override):
    with pytest.raises(ValueError):
        finite_bound_deficiency(torch.zeros(1, 4), **{
            "prefix_length": 3, "already_red": torch.tensor([False]), **override})


@pytest.mark.parametrize("logits", [torch.zeros(4), torch.zeros(0, 4), torch.zeros(1, 1),
                                    torch.zeros(1, 4, dtype=torch.float16),
                                    torch.tensor([[0., float("nan")]]),
                                    torch.empty(1, 4, device="meta")])
def test_invalid_logits_and_device(logits):
    with pytest.raises(ValueError):
        finite_bound_deficiency(logits, prefix_length=1, already_red=torch.tensor([False]))


def model_case(path="single"):
    torch.manual_seed(183)
    model = SignalDistribution("gru", 2, 4, hidden_size=5, num_layers=1, rank=2,
                               dropout=0, event_distribution="finite_horizon_mixture",
                               phase_covariance="separate", path_distribution=path).double()
    moments = model(torch.randn(2, 3, 2, dtype=torch.float64),
                    torch.tensor([.2, 1.2], dtype=torch.float64))
    target = torch.ones(2, 4, dtype=torch.float64)
    allowed = torch.zeros(2, 5, dtype=torch.bool)
    allowed[:, 0] = True
    arguments = dict(event_allowed_mask=allowed, event_observed_mask=torch.tensor([True, True]),
                     event_objective_prefix_lengths=[1, 4],
                     path_objective_prefix_lengths=[1, 4], n_samples=8)
    return model, moments, target, torch.ones_like(target, dtype=torch.bool), arguments


@pytest.mark.parametrize("path", ["single", "coupled_timing_mixture"])
def test_default_zero_literal_legacy_parity(path):
    outputs = []
    for mode in ("legacy", "default", "zero"):
        model, moments, target, mask, arguments = model_case(path)
        gen = torch.Generator().manual_seed(411)
        global_rng = torch.random.get_rng_state().clone()
        function = legacy_objective if mode == "legacy" else SignalDistribution.objective
        if mode == "zero":
            arguments["red_finite_bound_weight"] = 0.0
        terms = function(model, moments, target, mask, 1., generator=gen, **arguments)
        terms["total"].sum().backward()
        outputs.append((terms, [p.grad for p in model.parameters()], gen.get_state().clone(),
                        model.sample(moments, 1., 8, gen)))
        assert torch.equal(global_rng, torch.random.get_rng_state())
    old = outputs[0]
    for terms, gradients, rng, next_paths in outputs[1:]:
        assert terms.keys() == old[0].keys()
        assert "red_finite_bound" not in terms
        assert all(torch.equal(value, old[0][key]) for key, value in terms.items())
        assert all((a is None and b is None) or torch.equal(a, b)
                   for a, b in zip(gradients, old[1]))
        assert torch.equal(rng, old[2])
        assert torch.equal(next_paths, old[3])


@pytest.mark.parametrize("path", ["single", "coupled_timing_mixture"])
def test_positive_exact_prefix_mean_full_marginal_backward_and_rng(path):
    model, moments, target, mask, arguments = model_case(path)
    outputs = []
    for weight in (0., 2.):
        gen = torch.Generator().manual_seed(617)
        terms = model.objective(moments, target, mask, 1., generator=gen,
                                red_finite_bound_weight=weight,
                                red_corridor_width_weight=.2, red_corridor_miss_weight=.3,
                                post_factor_smoothness_weight=.01 if path == "single" else 0., **arguments)
        outputs.append((terms, gen.get_state().clone()))
    base, positive = outputs[0][0], outputs[1][0]
    z = moments["event_logits"]
    expected = sum(evidence(z, length, red=False)["cost"] * torch.tensor([1., 0.])
                   for length in (1, 4)) / 2
    assert torch.equal(positive["red_finite_bound"], expected)
    assert torch.equal(positive["total"], base["total"] + 2 * expected)
    assert all(torch.equal(positive[name], value) for name, value in base.items() if name != "total")
    assert torch.equal(outputs[0][1], outputs[1][1])
    actual_gradient = torch.autograd.grad(positive["red_finite_bound"].sum(), z, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), z, retain_graph=True)[0]
    assert torch.equal(actual_gradient, expected_gradient)
    if path == "coupled_timing_mixture":
        assert moments["joint_event_logits"].ndim == 3
        torch.testing.assert_close(z.log_softmax(-1),
                                   moments["joint_event_logits"].flatten(1).log_softmax(-1).reshape(2, 3, 5).logsumexp(1),
                                   rtol=0, atol=1e-12)
    positive["total"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("override", [
    {"red_finite_bound_weight": True}, {"red_finite_bound_weight": -1},
    {"red_finite_bound_weight": float("inf")}, {"coverage": 1 - 2e-8},
    {"red_threshold": 0.}, {"red_threshold": float("nan")},
    {"event_objective_prefix_lengths": [4, 1]}, {"event_objective_prefix_lengths": [True]},
    {"event_allowed_mask": torch.ones(2, 5)}, {"event_observed_mask": torch.tensor([1, 1])},
    {"no_entry_prefix": torch.tensor([.5, 0.])},
])
def test_positive_validation_before_sampler(override):
    model, moments, target, mask, arguments = model_case()
    arguments.update({"red_finite_bound_weight": 2., **override})
    threshold = arguments.pop("red_threshold", 1.)
    gen = torch.Generator().manual_seed(617)
    before = gen.get_state().clone()
    with patch.object(model, "sample", side_effect=AssertionError("sample called")):
        with pytest.raises(ValueError):
            model.objective(moments, target, mask, threshold, generator=gen, **arguments)
    assert torch.equal(before, gen.get_state())
