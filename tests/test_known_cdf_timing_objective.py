"""Bounded invented inputs only; no actual data, fit, or quality claim."""
import importlib.util
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from pdm.trajectory_objectives import known_cdf_timing_loss
from tests.test_finite_bound_deficiency_objective import model_case
from tests.test_mc_red_decision_objective import legacy_objective


spec = importlib.util.spec_from_file_location(
    "review_known_cdf", Path(__file__).resolve().parents[1] /
    "output/bearings-learned-funnel-20261002/review/known_cdf_timing_objective_review.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def evidence(z, length, admitted=(1,), observed=True, censor=0, red=False, **kwargs):
    a = torch.zeros_like(z, dtype=torch.bool)
    a[:, list(admitted)] = True
    return dict(prefix_length=length, already_red=torch.full((len(z),), red, device=z.device),
                event_allowed_mask=a, event_observed_mask=torch.full((len(z),), observed, device=z.device),
                no_entry_prefix=torch.full((len(z),), censor, device=z.device), **kwargs)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_evidence_quantifiers_and_exact_zero_gradients(dtype):
    rng = torch.random.get_rng_state().clone()
    for bits, obs, c, red in product(product((False, True), repeat=4), (False, True), range(4), (False, True)):
        admitted = [i for i, bit in enumerate(bits) if bit]
        if obs and not admitted:
            continue
        z = torch.tensor([[.3, -.5, .9, .1]], dtype=dtype, requires_grad=True)
        args = evidence(z, 3, admitted, obs, c, red)
        out = known_cdf_timing_loss(z, **args)
        ref = reference.known_cdf(z, **args)
        assert all(torch.equal(out[k], ref[k]) for k in ref)
        g = torch.autograd.grad(out["cost"].sum(), z, retain_graph=True)[0]
        assert torch.equal(g, torch.autograd.grad(ref["cost"].sum(), z)[0])
        E = admitted if obs else list(range(c, 4)) if c else list(range(4))
        for j in range(3):
            values = {int(e <= j) for e in E}
            known = (obs or c > 0) and not red and len(values) == 1
            assert out["mask"][0, j].item() == known
            assert out["labels"][0, j].item() == (next(iter(values)) if known else 0)
        if red or not (obs or c > 0):
            assert out["cost"].item() == 0
            assert torch.equal(g, torch.zeros_like(g))
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_analytic_gradient_fixed_30_519_mean_and_unconditional_tail(dtype):
    z = torch.linspace(-.8, .7, 520, dtype=dtype)[None].requires_grad_()
    outputs = [known_cdf_timing_loss(z, **evidence(z, L, (300,))) for L in (30, 519)]
    a, b = outputs
    assert torch.equal(a["F"], b["F"][:, :30])
    assert torch.equal(a["mask"], b["mask"][:, :30])
    assert torch.equal(a["labels"], b["labels"][:, :30])
    for L, out in zip((30, 519), outputs):
        torch.testing.assert_close(out["q"].sum(-1), torch.ones(1, dtype=torch.float64), rtol=0, atol=1e-15)
        torch.testing.assert_close(out["q"][:, L:].sum(-1), 1-out["F"][:, -1], rtol=0, atol=1e-15)
        k = torch.arange(520)[None, :]
        j = torch.arange(L)[:, None]
        derivative = out["q"][0, None] * ((k <= j).double()-out["F"][0, :, None])
        analytic = (2/30*(out["F"][0]-out["labels"][0])*out["mask"][0]) @ derivative
        g = torch.autograd.grad(out["cost"].sum(), z, retain_graph=True)[0]
        torch.testing.assert_close(g.double()[0], analytic, rtol=0, atol=1e-9)
    aggregate = (a["cost"]+b["cost"])/2
    independent = sum(((o["F"]-o["labels"]).square()*o["mask"]).sum(-1) for o in outputs)/60
    torch.testing.assert_close(aggregate, independent, rtol=0, atol=1e-15)
    ga = torch.autograd.grad(a["cost"].sum(), z, retain_graph=True)[0]
    gb = torch.autograd.grad(b["cost"].sum(), z, retain_graph=True)[0]
    torch.testing.assert_close(torch.autograd.grad(aggregate.sum(), z)[0], (ga+gb)/2, rtol=0, atol=1e-8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_extrema_and_softmax_underflow_limit(dtype):
    m = torch.finfo(dtype).max
    z = torch.tensor([[-m, m]], dtype=dtype, requires_grad=True)
    if dtype == torch.float64:
        with pytest.raises(ValueError, match="overflow"):
            known_cdf_timing_loss(z, **evidence(z, 1, (0,)))
    else:
        out = known_cdf_timing_loss(z, **evidence(z, 1, (0,)))
        assert torch.isfinite(out["cost"]).all()
    z = torch.tensor([[-1000., -1000., -1000., 0.]], dtype=dtype, requires_grad=True)
    cost = known_cdf_timing_loss(z, **evidence(z, 3))["cost"]
    assert cost.item() > 0
    assert torch.equal(torch.autograd.grad(cost.sum(), z)[0], torch.zeros_like(z))


@pytest.mark.parametrize("override", [
    {"prefix_length": True}, {"prefix_length": 0}, {"prefix_length": 4},
    {"scale_steps": True}, {"scale_steps": 0}, {"scale_steps": float("nan")},
    {"already_red": torch.zeros(1)}, {"already_red": torch.zeros(1, 1, dtype=torch.bool)},
    {"event_allowed_mask": torch.ones(1, 4)},
    {"event_allowed_mask": None}, {"event_observed_mask": torch.ones(1)},
    {"event_allowed_mask": torch.zeros(1, 4, dtype=torch.bool)},
    {"no_entry_prefix": torch.tensor([1.5])}, {"no_entry_prefix": torch.tensor([-1])},
    {"no_entry_prefix": torch.tensor([True])}, {"no_entry_prefix": torch.tensor([4])},
    {"no_entry_prefix": torch.tensor([float("inf")])},
    {"already_red": torch.empty(1, dtype=torch.bool, device="meta")},
])
def test_invalid_evidence(override):
    z = torch.zeros(1, 4)
    with pytest.raises(ValueError):
        known_cdf_timing_loss(z, **{**evidence(z, 3), **override})


@pytest.mark.parametrize("z", [torch.zeros(4), torch.zeros(0, 4), torch.zeros(1, 1),
    torch.zeros(1, 4, dtype=torch.float16), torch.tensor([[float("nan"), 0.]]),
    torch.empty(1, 4, device="meta")])
def test_invalid_logits(z):
    with pytest.raises(ValueError):
        known_cdf_timing_loss(z, prefix_length=1, already_red=torch.tensor([False]))


@pytest.mark.parametrize("path", ["single", "coupled_timing_mixture"])
def test_zero_default_legacy_parity_rng_next_draw_and_gradients(path):
    results = []
    for mode in ("legacy", "default", "zero"):
        model, moments, target, mask, args = model_case(path)
        gen = torch.Generator().manual_seed(411)
        rng = torch.random.get_rng_state().clone()
        if mode == "zero":
            args["event_cdf_weight"] = 0.
        with patch("pdm.models.signal_distribution.known_cdf_timing_loss", side_effect=AssertionError("disabled helper called")):
            terms = (legacy_objective(model, moments, target, mask, 1., generator=gen, **args)
                     if mode == "legacy" else model.objective(moments, target, mask, 1., generator=gen, **args))
        terms["total"].sum().backward()
        results.append((terms, [p.grad for p in model.parameters()], gen.get_state().clone(), model.sample(moments, 1., 8, gen)))
        assert torch.equal(rng, torch.random.get_rng_state())
    for terms, grads, state, paths in results[1:]:
        old = results[0]
        assert terms.keys() == old[0].keys()
        assert all(torch.equal(v, old[0][k]) for k, v in terms.items())
        assert all((a is None and b is None) or torch.equal(a, b) for a, b in zip(grads, old[1]))
        assert torch.equal(state, old[2]) and torch.equal(paths, old[3])


@pytest.mark.parametrize("path", ["single", "coupled_timing_mixture"])
def test_positive_preserves_old_terms_full_marginal_and_upstream_gradients(path):
    model, moments, target, mask, args = model_case(path)
    captured_raw = []
    hook = model.event_head.register_forward_hook(lambda module, inputs, output: captured_raw.append(output))
    moments = model(torch.linspace(-.5, .5, 12, dtype=torch.float64).reshape(2, 3, 2),
                    torch.tensor([.2, 1.2], dtype=torch.float64))
    hook.remove()
    outputs = []
    for weight in (0., 2.):
        gen = torch.Generator().manual_seed(617)
        terms = model.objective(moments, target, mask, 1., generator=gen, event_cdf_weight=weight,
            red_finite_bound_weight=1., red_corridor_width_weight=.2, red_corridor_miss_weight=.3,
            post_factor_smoothness_weight=.01 if path == "single" else 0., **args)
        outputs.append((terms, gen.get_state().clone()))
    base, positive = outputs[0][0], outputs[1][0]
    z = moments["event_logits"]
    expected = sum(known_cdf_timing_loss(z, prefix_length=L, already_red=moments["current_signal"] >= 1,
        event_allowed_mask=args["event_allowed_mask"], event_observed_mask=args["event_observed_mask"])["cost"] for L in (1, 4))/2
    assert torch.equal(positive["event_cdf"], expected)
    assert torch.equal(positive["total"], base["total"]+2*expected)
    assert all(torch.equal(positive[k], v) for k, v in base.items() if k != "total")
    assert torch.equal(outputs[0][1], outputs[1][1])
    assert z.abs().max() < 100  # moderate synthetic logits, no saturation claim
    raw = captured_raw[0]
    raw_gradient = torch.autograd.grad(positive["event_cdf"].sum(), raw, retain_graph=True)[0]
    assert torch.isfinite(raw_gradient).all() and raw_gradient.abs().sum() > 0
    head_gradient = torch.autograd.grad(positive["event_cdf"].sum(), model.event_head.weight, retain_graph=True)[0]
    assert torch.isfinite(head_gradient).all() and head_gradient.abs().sum() > 0
    positive["total"].sum().backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("override", [
    {"event_cdf_weight": True}, {"event_cdf_weight": -1}, {"event_cdf_weight": float("inf")},
    {"event_cdf_scale_steps": True}, {"event_cdf_scale_steps": 0}, {"event_cdf_scale_steps": float("nan")},
    {"red_threshold": 0.}, {"red_threshold": float("inf")},
    {"event_objective_prefix_lengths": [4, 1]}, {"event_objective_prefix_lengths": [True]},
    {"event_allowed_mask": torch.ones(2, 5)}, {"no_entry_prefix": torch.tensor([.5, 0.])},
])
def test_positive_rejects_before_sampler(override):
    model, moments, target, mask, args = model_case()
    args.update({"event_cdf_weight": 2., **override})
    threshold = args.pop("red_threshold", 1.)
    with patch.object(model, "sample", side_effect=AssertionError("sample called")):
        with pytest.raises(ValueError):
            model.objective(moments, target, mask, threshold, **args)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_moderate_finite_differences_and_common_offset(dtype):
    z = torch.tensor([[0., -4., -8., -4.]], dtype=dtype, requires_grad=True)
    args = evidence(z, 3)
    out = known_cdf_timing_loss(z, **args)
    gradient = torch.autograd.grad(out["cost"].sum(), z)[0]
    eps = 1/256 if dtype == torch.float32 else 1e-4
    for k in range(4):
        delta = torch.zeros_like(z)
        delta[0, k] = eps
        numeric = (known_cdf_timing_loss(z.detach()+delta, **args)["cost"] -
                   known_cdf_timing_loss(z.detach()-delta, **args)["cost"])/(2*eps)
        torch.testing.assert_close(numeric, gradient[:, k].double(), rtol=0, atol=1e-7)
    offset = 2**25+16 if dtype == torch.float32 else 2**54+16
    shifted = (z.detach()+offset).requires_grad_()
    other = known_cdf_timing_loss(shifted, **args)
    assert torch.equal(out["cost"], other["cost"])
    assert torch.equal(gradient, torch.autograd.grad(other["cost"].sum(), shifted)[0])


@pytest.mark.parametrize("bad", ["current_nan", "current_rank", "logit_horizon", "logit_batch"])
def test_positive_current_and_saved_shape_reject_before_sampler(bad):
    model, moments, target, mask, args = model_case()
    if bad == "current_nan":
        moments["current_signal"] = torch.tensor([float("nan"), .2])
    elif bad == "current_rank":
        moments["current_signal"] = moments["current_signal"][:, None]
    elif bad == "logit_horizon":
        moments["event_logits"] = moments["event_logits"][:, :-1]
    else:
        moments["event_logits"] = moments["event_logits"][:1]
    with patch.object(model, "sample", side_effect=AssertionError("sample called")):
        with pytest.raises(ValueError):
            model.objective(moments, target, mask, 1., event_cdf_weight=2., **args)
