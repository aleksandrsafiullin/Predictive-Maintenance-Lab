"""Engineering checks for smooth entry times; no held-out Bearings inputs."""
import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import first_entry_loss


def model_and_moments(horizon=519, batch=1):
    torch.manual_seed(9)
    model = SignalDistribution("external", 1, horizon, hidden_size=8, rank=2,
                               external_feature_size=3, dropout=0)
    with torch.no_grad():
        model.event_head.weight.zero_()
    moments = model(None, torch.full((batch,), .5), torch.zeros(batch, 3))
    return model, moments


def test_dense_mass_matches_analytic_lognormal_cure_and_survival():
    model, moments = model_and_moments()
    probabilities = moments["event_logits"].softmax(-1)[0].double()
    raw = model.event_head.bias.double()
    median = torch.nn.functional.softplus(raw[3:6]) + .05
    scale = torch.nn.functional.softplus(raw[6:9]) + .08
    weights = raw[:3].softmax(-1)
    cure = raw[9].sigmoid()
    for steps in [1, 30, 60, 120, 519]:
        z = (torch.tensor(float(steps)).log() - median.log()) / scale
        expected = (1-cure) * (weights * (.5 * torch.erfc(-z / 2**.5))).sum()
        assert torch.allclose(probabilities[:steps].sum(), expected, atol=2e-7)
        survival_loss = first_entry_loss(moments["event_logits"],
                                         no_entry_prefix=torch.tensor([steps]))
        assert torch.allclose(survival_loss.double(), -(1-expected).log()[None], atol=5e-7)
    assert torch.allclose(probabilities.sum(), torch.tensor(1., dtype=torch.double), atol=2e-7)
    assert probabilities[-1] >= cure
    assert model.event_head.out_features == 10


@pytest.mark.parametrize("term", ["event", "width", "miss"])
def test_event_head_receives_actual_loss_gradients(term):
    model, moments = model_and_moments(120, batch=2)
    target = torch.full((2, 120), 12.)
    mask = torch.ones_like(target, dtype=torch.bool)
    allowed = torch.zeros(2, 121, dtype=torch.bool)
    allowed[:, 29] = True
    terms = model.objective(moments, target, mask, 3., event_allowed_mask=allowed,
                            event_observed_mask=torch.ones(2, dtype=torch.bool),
                            n_samples=24, generator=torch.Generator().manual_seed(83))
    terms[term].sum().backward()
    gradient = model.event_head.bias.grad
    assert torch.isfinite(gradient).all()
    # Mixture, remaining time, scale, and cure are trainable through this loss.
    assert all(gradient[a:b].abs().sum() > 0 for a, b in [(0,3), (3,6), (6,9), (9,10)])
    assert model.event_head.weight.grad.abs().sum() > 0


def test_no_entry_is_explicit_and_finite_horizon_survival_is_not_cure():
    model, _ = model_and_moments()
    with torch.no_grad():
        model.event_head.bias[9] = 80
    cured = model(None, torch.tensor([.5]), torch.zeros(1,3))
    paths, entries = model.sample(cured, 3., 32, return_entry=True)
    assert (entries == 519).all() and (paths < 3.).all()
    with torch.no_grad():
        model.event_head.bias[9] = -80
    finite = model(None, torch.tensor([.5]), torch.zeros(1,3))["event_logits"].softmax(-1)
    assert 0 < finite[0, -1] < .1
    assert torch.isfinite(finite).all()


def test_dense_inference_sampler_agrees_with_bucket_distribution():
    model, moments = model_and_moments()
    with torch.no_grad():
        paths, entries = model.sample(moments, 3., 4000,
                                      torch.Generator().manual_seed(903), return_entry=True)
    crossing = paths >= 3.
    actual_entry = torch.where(crossing.any(-1), crossing.int().argmax(-1), 519)
    assert torch.equal(actual_entry, entries)
    probability = moments["event_logits"].softmax(-1)[0]
    for boundary in [30, 60, 120, 519]:
        empirical = (entries < boundary).float().mean()
        assert abs(empirical - probability[:boundary].sum()) < .035


def test_post_growth_moves_with_entry_but_already_red_stays_current_anchored():
    model, moments = model_and_moments(6, batch=3)
    moments["current_signal"] = torch.tensor([.5, .5, 4.])
    for name in ["mean", "entry_mean", "diagonal_scale", "factors"]:
        moments[name] = torch.zeros_like(moments[name])
    moments["post_mean"] = torch.arange(6.).expand(3, -1)
    moments["event_logits"] = torch.full((3, 7), -torch.inf)
    moments["event_logits"][0, 0] = 0
    moments["event_logits"][1:, 2] = 0
    paths, entries = model.sample(moments, 3., 2, return_entry=True)
    # First post-entry point always has residual zero; subsequent growth shifts.
    assert torch.allclose(paths[:, 0, 1:4], paths[:, 1, 3:6])
    assert torch.allclose(paths[:, 0, 1], torch.tensor(3.))
    assert torch.allclose(paths[:, 1, :2], torch.full((2,2), .5))
    base = 4. + torch.log(-torch.expm1(torch.tensor(-4.)))
    assert torch.allclose(paths[:, 2], torch.nn.functional.softplus(base + torch.arange(6.))[None])
    assert (entries[:, 2] == -1).all()


def test_partial_mask_and_unknown_row_do_not_add_entry_training_evidence():
    model, moments = model_and_moments(120, batch=2)
    moments["event_logits"].retain_grad()
    target = torch.full((2,120), float("nan"))
    target[0,:10] = .5
    mask = torch.zeros_like(target, dtype=torch.bool)
    mask[0,:10] = True
    terms = model.objective(moments, target, mask, 3., no_entry_prefix=torch.tensor([10,0]),
                            n_samples=16, generator=torch.Generator().manual_seed(12))
    assert all(torch.isfinite(value).all() for value in terms.values())
    terms["total"].sum().backward()
    assert moments["event_logits"].grad[1].abs().sum() == 0
    assert moments["event_logits"].grad[0].abs().sum() > 0


def test_extreme_parameters_keep_normalized_distribution_and_finite_gradients():
    model, _ = model_and_moments()
    with torch.no_grad():
        model.event_head.bias[3:6] = torch.tensor([-100., 0., 10000.])
        model.event_head.bias[6:9] = torch.tensor([-100., 0., 50.])
    moments = model(None, torch.tensor([.5]), torch.zeros(1,3))
    logits = moments["event_logits"]
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.softmax(-1).sum(), torch.tensor(1.))
    first_entry_loss(logits, no_entry_prefix=torch.tensor([100])).sum().backward()
    assert torch.isfinite(model.event_head.bias.grad).all()


def test_log_location_parameterization_matches_multiplicative_remaining_times():
    model = SignalDistribution('external', 1, 519, hidden_size=8, rank=2,
                               external_feature_size=3, dropout=0, event_location='log')
    with torch.no_grad():
        model.event_head.weight.zero_()
        model.event_head.bias[3:6] = torch.tensor([4., 60., 2000.]).log()
    moments = model(None, torch.tensor([.5]), torch.zeros(1,3))
    raw = model.event_head.bias.double()
    scale = torch.nn.functional.softplus(raw[6:9]) + .08
    weights, cure = raw[:3].softmax(-1), raw[9].sigmoid()
    probabilities = moments['event_logits'].softmax(-1)[0].double()
    for steps in [1,4,60,120,519]:
        z = (torch.tensor(float(steps)).log() - raw[3:6]) / scale
        expected = (1-cure) * (weights * (.5 * torch.erfc(-z / 2**.5))).sum()
        assert torch.allclose(probabilities[:steps].sum(), expected, atol=2e-7)
    loss = first_entry_loss(moments['event_logits'], no_entry_prefix=torch.tensor([519]))
    loss.sum().backward()
    assert torch.isfinite(model.event_head.bias.grad).all()
    assert model.event_head.bias.grad[3:6].abs().sum() > 0
