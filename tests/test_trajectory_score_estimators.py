"""Exact-distribution checks for the stochastic categorical gradient estimator."""
import itertools

import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import path_losses, path_score_surrogates


@pytest.mark.parametrize('term', ['energy', 'width', 'miss'])
def test_sample_and_group_baselines_preserve_exact_expected_gradient(term):
    draws = torch.tensor(list(itertools.product([0, 1], repeat=8))).T
    paths = torch.where(draws[..., None].bool(), 5.0, 0.5).double().expand(-1, -1, 3)
    mask = torch.ones(paths.shape[1:], dtype=torch.bool)
    actual = torch.full(mask.shape, 6.0, dtype=torch.float64)
    theta = torch.tensor(-0.7, dtype=torch.float64, requires_grad=True)
    p = theta.sigmoid()
    lp = torch.where(draws.bool(), p.log(), (1 - p).log())
    probability = lp.sum(0).exp()
    values = path_losses(paths, actual, mask)[term]
    exact = torch.autograd.grad((probability * values).sum(), theta, retain_graph=True)[0]
    surrogate = path_score_surrogates(paths, actual, mask, lp)[term]
    estimated = torch.autograd.grad((probability.detach() * surrogate).sum(), theta)[0]
    assert torch.allclose(estimated, exact, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize('term', ['energy', 'width', 'miss'])
def test_other_equipment_target_cannot_change_categorical_gradient(term):
    model = SignalDistribution('gru', 2, 4, hidden_size=8, num_layers=1, rank=2)
    model.eval()
    actual = torch.full((2, 4), 6.0)

    def gradient(target):
        moments = model(torch.zeros(2, 8, 2), torch.tensor([0.6, 3.5]))
        terms = model.objective(moments, target, torch.ones(2, 4, dtype=torch.bool), 3.0,
                                n_samples=16, generator=torch.Generator().manual_seed(4))
        return torch.autograd.grad(terms[term].sum(), moments['event_logits'])[0][0]

    left = gradient(actual)
    actual[1] = 1e6
    right = gradient(actual)
    assert torch.equal(left, right)
    assert left.abs().sum() > 0


def test_two_sample_estimator_remains_defined_and_unknown_rows_have_zero_gradient():
    lp = torch.zeros(2, 2, requires_grad=True)
    mask = torch.tensor([[True, True], [False, False]])
    result = path_score_surrogates(torch.rand(2, 2, 2), torch.ones(2, 2), mask, lp)
    sum(v.sum() for v in result.values()).backward()
    assert torch.isfinite(lp.grad).all()
    assert lp.grad[:, 1].abs().sum() == 0
