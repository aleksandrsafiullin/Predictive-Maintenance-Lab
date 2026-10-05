"""Synthetic law/gradient tests. No datasets, checkpoints or fitting."""
import hashlib
import itertools

import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import path_losses, path_score_surrogates


def model(h=4, covariance="separate"):
    return SignalDistribution("gru", 2, h, hidden_size=5, num_layers=1, rank=2,
                              dropout=0, event_distribution="finite_horizon_mixture",
                              phase_covariance=covariance,
                              path_distribution="coupled_timing_mixture").double()


def moments(m, current=(.2, 3.)):
    return m(torch.zeros(len(current), 6, 2, dtype=torch.float64),
             torch.tensor(current, dtype=torch.float64))


@pytest.mark.parametrize("covariance", ["shared", "separate"])
def test_joint_marginal_prefix_shapes_and_support(covariance):
    m = model(covariance=covariance)
    a = moments(m)
    joint = a["joint_event_logits"].exp()
    torch.testing.assert_close(joint.sum(1), a["event_logits"].softmax(-1), rtol=1e-13, atol=1e-15)
    torch.testing.assert_close(joint.sum(-1), a["component_logits"].softmax(-1), rtol=1e-13, atol=1e-15)
    for prefix in range(1, 5):
        torch.testing.assert_close(joint[..., prefix:].sum((1, 2)),
                                   a["event_logits"].softmax(-1)[:, prefix:].sum(-1))
    assert a["mean"].shape == (2, 3, 4)
    p, e, k, score = m.sample(a, 3., 64, torch.Generator().manual_seed(4), return_latent=True)
    assert p.shape == (64, 2, 4) and k.shape == e.shape == score.shape == (64, 2)
    red = p[:, 0] >= 3
    first = torch.where(red.any(-1), red.long().argmax(-1), 4)
    assert torch.equal(first, e[:, 0])
    assert (e[:, 1] == -1).all()
    torch.testing.assert_close(score[:, 1], a["component_logits"][1, k[:, 1]])


@pytest.mark.parametrize("allowed", [[0, 1, 0, 0, 0], [0, 1, 1, 0, 0],
                                     [0, 0, 0, 0, 1], [0, 0, 1, 1, 1], [1]*5])
def test_conditional_responsibilities_live_normalizer(allowed):
    m = model()
    a = moments(m, (.2,))
    mask = torch.tensor([allowed], dtype=torch.bool)
    conditional = m.coupled_latent_logits(a, 3., mask).exp()
    joint = a["joint_event_logits"].exp() * mask[:, None]
    expected = joint / joint.sum((1, 2), keepdim=True)
    torch.testing.assert_close(conditional, expected, rtol=1e-13, atol=1e-15)
    derivative = torch.autograd.grad(conditional.sum(), m.event_head.bias)[0]
    assert derivative.abs().max() < 1e-12
    p, e, k, score = m.sample(a, 3., 32, torch.Generator().manual_seed(9),
                            return_latent=True, entry_allowed_mask=mask)
    assert mask[0, e[:, 0]].all()
    torch.testing.assert_close(score[:, 0], conditional.log()[0, k[:, 0], e[:, 0]])


@pytest.mark.parametrize("allowed", [[0, 1, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 1, 1]])
@pytest.mark.parametrize("term", ["energy", "width", "miss"])
def test_enumerated_joint_scores_equal_exact_expected_gradient(allowed, term):
    # All two-draw configurations are enumerated, no Monte Carlo tolerance.
    m = model(2)
    log_joint = torch.tensor([[[-2., -1., -3.], [-1., -.5, -2.], [-3., -2., -1.]]],
                             dtype=torch.float64, requires_grad=True)
    a = {"joint_event_logits": log_joint, "component_logits": torch.zeros(1, 3, dtype=torch.float64),
         "current_signal": torch.tensor([.2], dtype=torch.float64)}
    lp = m.coupled_latent_logits(a, 3., torch.tensor([allowed], dtype=torch.bool)).flatten()
    ids = torch.where(torch.isfinite(lp))[0].tolist()
    exact = torch.zeros((), dtype=torch.float64)
    estimator = torch.zeros_like(exact)
    for left, right in itertools.product(ids, repeat=2):
        paths = torch.tensor([[[.2 + left, .4 + left / 2]],
                              [[.2 + right, .4 + right / 2]]], dtype=torch.float64)
        target, mask = torch.tensor([[8., 8.]], dtype=torch.float64), torch.ones(1, 2, dtype=torch.bool)
        cost = path_losses(paths, target, mask)[term].sum()
        likelihood = (lp[left] + lp[right]).exp()
        score = lp[torch.tensor([left, right])].reshape(2, 1)
        surrogate = path_score_surrogates(paths, target, mask, score)[term].sum()
        exact = exact + likelihood * cost
        estimator = estimator + likelihood.detach() * surrogate
    actual = torch.autograd.grad(exact, log_joint, retain_graph=True)[0]
    estimated = torch.autograd.grad(estimator, log_joint)[0]
    torch.testing.assert_close(actual, estimated, rtol=1e-10, atol=1e-12)


def test_whole_path_component_and_already_red_score():
    m = model()
    a = moments(m, (3.,))
    # Zero noise and distinct constant post residuals make any lead-wise K
    # redraw detectable. E restrictions must have no effect on already RED.
    a["post_mean"] = torch.arange(3, dtype=torch.float64)[None, :, None].expand(1, 3, 4)
    for name in list(a):
        if name.endswith("diagonal_scale") or name.endswith("factors"):
            a[name] = torch.zeros_like(a[name])
    p, e, k, score = m.sample(a, 3., 20, torch.Generator().manual_seed(8), return_latent=True,
                            entry_allowed_mask=torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool))
    assert torch.equal(p, p[..., :1].expand_as(p))
    assert len(k.unique()) == 3 and (e == -1).all()
    gradient = torch.autograd.grad((score * k.double()).sum(), m.event_head.bias)[0]
    assert gradient[:3].abs().sum() > 0
    assert gradient[3:].abs().sum() == 0


def test_identical_heads_reduce_to_single_paths_for_same_entry_and_innovations(monkeypatch):
    coupled = model()
    a = moments(coupled)
    single = SignalDistribution("gru", 2, 4, hidden_size=5, num_layers=1, rank=2,
                                dropout=0, phase_covariance="separate",
                                event_distribution="finite_horizon_mixture").double()
    old = {}
    for name, value in a.items():
        if name in {"joint_event_logits", "component_logits"}:
            continue
        if name in {"event_logits", "current_signal"}:
            old[name] = value
        else:
            old[name] = value[:, 0]
            a[name] = value[:, :1].expand_as(value)
    entry = torch.tensor([[0, 1, 2, 3, 4, 0], [4, 3, 2, 1, 0, 4]])
    component = torch.tensor([[0, 1, 2, 0, 1, 2], [2, 1, 0, 2, 1, 0]])

    def draw(probabilities, *args, **kwargs):
        return entry if probabilities.shape[-1] == 5 else component * 5 + entry

    monkeypatch.setattr(torch, "multinomial", draw)
    new_paths, new_entries = coupled.sample(a, 3., 6, torch.Generator().manual_seed(65), return_entry=True)
    old_paths, old_entries = single.sample(old, 3., 6, torch.Generator().manual_seed(65), return_entry=True)
    assert torch.equal(new_paths, old_paths) and torch.equal(new_entries, old_entries)


def test_capacity_extreme_finite_and_roundtrip():
    large = SignalDistribution("gru", 33, 519, hidden_size=64, num_layers=1, rank=6,
                               phase_covariance="separate", event_distribution="finite_horizon_mixture",
                               path_distribution="coupled_timing_mixture")
    assert sum(p.numel() for p in large.parameters()) == 2452738
    assert large.path_head[-1].out_features == 37368
    m = model()
    with torch.no_grad():
        m.event_head.weight.zero_()
        m.event_head.bias.copy_(torch.tensor([1000., -1000., 0., -1000., 1000., 0., -1000., 1000., 0., 1000.]))
    a = moments(m)
    assert torch.isfinite(a["joint_event_logits"]).all()
    torch.testing.assert_close(a["joint_event_logits"].exp().sum(1), a["event_logits"].softmax(-1), atol=1e-14, rtol=1e-13)
    clone = model()
    clone.load_state_dict(m.state_dict())
    p = m.sample(a, 3., 8, torch.Generator().manual_seed(1))
    again = clone.sample(moments(clone), 3., 8, torch.Generator().manual_seed(1))
    assert torch.equal(p, again)
    terms = m.objective(a, torch.ones(2, 4, dtype=torch.float64), torch.ones(2, 4, dtype=torch.bool),
                        3., n_samples=8, generator=torch.Generator().manual_seed(5), no_entry_prefix=torch.tensor([4, 0]))
    terms["total"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


def test_production_shape_backward_without_fit_or_optimizer():
    m = SignalDistribution("gru", 33, 519, hidden_size=64, num_layers=1, rank=6,
                           dropout=0, phase_covariance="separate",
                           event_distribution="finite_horizon_mixture",
                           path_distribution="coupled_timing_mixture")
    a = m(torch.zeros(2, 8, 33), torch.tensor([.2, 3.]))
    terms = m.objective(a, torch.ones(2, 519), torch.ones(2, 519, dtype=torch.bool), 3.,
                        n_samples=128, generator=torch.Generator().manual_seed(51),
                        no_entry_prefix=torch.tensor([519, 0]),
                        path_objective_prefix_lengths=(10, 30, 60, 519),
                        event_objective_prefix_lengths=(10, 30, 60, 519))
    terms["total"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())
    assert m.path_head[-1].weight.grad.abs().sum() > 0
    assert m.event_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("evidence", ["singleton", "survival", "already_red", "unknown"])
def test_conditional_phase_objective_keeps_component_gradients(evidence):
    m = model()
    a = moments(m, ((3. if evidence == "already_red" else .2),))
    # Distinct path components make conditional K-score gradients necessary.
    with torch.no_grad():
        bias = m.path_head[-1].bias.reshape(3, 4, -1)
        bias[..., :3] += torch.arange(3, dtype=torch.float64)[:, None, None]
    a = moments(m, ((3. if evidence == "already_red" else .2),))
    kwargs = {}
    if evidence == "singleton":
        kwargs = {"event_allowed_mask": torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.bool),
                  "event_observed_mask": torch.tensor([True])}
    if evidence == "survival":
        kwargs = {"no_entry_prefix": torch.tensor([4])}
    terms = m.objective(a, torch.full((1, 4), 8., dtype=torch.float64),
                        torch.ones(1, 4, dtype=torch.bool), 3., n_samples=32,
                        generator=torch.Generator().manual_seed(54), **kwargs)
    gradient = torch.autograd.grad(terms["phase_energy"].sum(), m.event_head.bias)[0]
    if evidence == "unknown":
        assert terms["phase_energy"].item() == 0 and gradient.abs().sum() == 0
    else:
        assert gradient[:3].abs().sum() > 0


def digest(tensors):
    result = hashlib.sha256()
    for tensor in tensors:
        result.update(tensor.detach().contiguous().numpy().tobytes())
    return result.hexdigest()


@pytest.mark.parametrize("selector", [None, "single"])
def test_default_exact_frozen_prechange_reference(selector):
    # Golden byte digests generated from the frozen pre-coupling source, using
    # the same float64 CPU test configuration. These include all loss gradients.
    with torch.random.fork_rng():
        torch.manual_seed(91)
        kwargs = {} if selector is None else {"path_distribution": selector}
        m = SignalDistribution("gru", 2, 4, hidden_size=5, num_layers=1, rank=2, dropout=0,
                               event_distribution="finite_horizon_mixture", phase_covariance="separate", **kwargs).double()
        assert digest(m.state_dict().values()) == "30f8007edf90d1cfcc3e061b1f764959ded529816028100085fb78bf0c4ff6bb"
        assert digest([torch.random.get_rng_state()]) == "1f7b268d9bd74714511c9ff938b6f31fbe3f2ca0eedd756d0a78636b5f13644a"
        a = m(torch.linspace(-.5, .5, 24, dtype=torch.float64).reshape(2, 6, 2), torch.tensor([.2, 3.], dtype=torch.float64))
        assert digest(a.values()) == "fe59feec966a96eb8eb7cc9e92cecf333357ad2c2f2ba4dd675e9e23a3534f1e"
        p, e = m.sample(a, 3., 8, torch.Generator().manual_seed(12), return_entry=True)
        assert digest([p, e]) == "b5dc3256b8cb62e11746d524992e97b7c8d8c3186a482882eb451a3ac2ebfe44"
        terms = m.objective(a, torch.full((2, 4), 4., dtype=torch.float64), torch.ones(2, 4, dtype=torch.bool),
                            3., n_samples=8, generator=torch.Generator().manual_seed(14), no_entry_prefix=torch.tensor([4, 0]))
        assert digest(terms.values()) == "6bbd3c5595bbcb59e210ac375f87549ce09a3f22da5e59d160c31fb056375e2e"
        terms["total"].sum().backward()
        assert digest(p.grad for p in m.parameters()) == "6aa1e8ffed702eed5530241fd1510659eec9b56962aa5a02a8ff2fb856f5fd92"
