"""Synthetic engineering checks; no fits, data access, or forecast-quality claims."""

import inspect

import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import path_losses, path_score_surrogates, simultaneous_band


# Frozen pre-correction arithmetic, including categorical control variates.
def _frozen_simultaneous_band(paths: torch.Tensor, coverage: float = 0.9, target_mask=None):
    """Empirical simultaneous band for paths [samples,batch,horizon].

    A single quantile of the maximum standardized excursion sets every horizon's
    width. Pointwise quantiles alone would not provide simultaneous coverage.
    """
    if not 0 < coverage < 1 or paths.ndim != 3 or paths.shape[0] < 2:
        raise ValueError("Expected >=2 joint samples and coverage in (0,1)")
    center = paths.median(dim=0).values
    scale = paths.std(dim=0, unbiased=False).clamp_min(1e-5)
    excursion_values = (paths - center).abs() / scale
    if target_mask is not None:
        excursion_values = excursion_values * target_mask.bool()[None, ...]
    excursion = excursion_values.amax(dim=-1)
    radius = torch.quantile(excursion, coverage, dim=0)
    return (center - radius[:, None] * scale).clamp_min(0), center + radius[:, None] * scale


def _frozen_path_losses(paths, actual_future, target_mask, *, coverage=0.9, signal_scale=1.0, energy_only=False):
    """Return per-row energy, simultaneous width and worst-path miss in RED-normalized physical units.

    Masking precedes all arithmetic, including NaN handling. Completely unknown
    rows contribute zero. Integration must balance physical units, rather than
    weighting units with more prefixes more heavily.
    """
    mask = target_mask.bool()
    valid = mask.any(dim=-1)
    target = torch.where(mask, actual_future, torch.zeros_like(actual_future))
    if (target < 0).any() or not torch.isfinite(target).all():
        raise ValueError("Observed future signal must be finite and nonnegative")
    scale = torch.as_tensor(signal_scale, dtype=paths.dtype, device=paths.device).reshape(-1, 1)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Frozen signal scale must be positive and finite")
    paths = torch.where(mask[None], paths, torch.zeros_like(paths))
    lp = paths.clamp_min(0) / scale
    lt = target / scale
    count = mask.sum(-1).clamp_min(1)
    # Energy score uses joint distances, preserving dependency across horizons.
    distance = (((lp - lt) ** 2 * mask).sum(-1) / count).clamp_min(1e-12).sqrt()
    # Independent halves avoid O(S^2) memory while retaining an unbiased pair term.
    half = paths.shape[0] // 2
    if half < 1:
        raise ValueError("At least two samples are required")
    pair = (((lp[:half] - lp[half : 2 * half]) ** 2 * mask).sum(-1) / count).clamp_min(1e-12).sqrt()
    energy = distance.mean(0) - 0.5 * pair.mean(0)
    if energy_only:
        return {"energy": energy * valid}
    lower, upper = _frozen_simultaneous_band(paths, coverage, target_mask=mask)
    lo, hi = lower / scale, upper / scale
    width = ((hi - lo) * mask).sum(-1) / count
    miss = (torch.relu(lo - lt) * mask).amax(-1) + (torch.relu(lt - hi) * mask).amax(-1)
    return {"energy": energy * valid, "width": width * valid, "miss": miss * valid}


def _frozen_path_score_surrogates(paths, actual_future, target_mask, sampled_log_prob, *, coverage=0.9, signal_scale=1.0, energy_only=False):
    """Unbiased, within-row categorical gradients for joint path objectives.

    Energy is a mean of sample distances and independent paired distances, so
    each distance uses only its own categorical likelihood score. For a band
    statistic every sample matters: four disjoint groups use the band's cost
    with that whole group removed as an independent control variate. Baselines
    are detached and never borrow another equipment row's target or cost.
    """
    mask = target_mask.bool()
    valid = mask.any(-1)
    count = mask.sum(-1).clamp_min(1)
    scale = torch.as_tensor(signal_scale, dtype=paths.dtype, device=paths.device).reshape(-1, 1)
    with torch.no_grad():
        values = torch.where(mask[None], paths.detach(), torch.zeros_like(paths)) / scale
        target = torch.where(mask, actual_future, torch.zeros_like(actual_future)) / scale
        distance = (((values - target) ** 2).sum(-1) / count).clamp_min(1e-12).sqrt()
        sample_count = len(paths)
        baseline = (distance.sum(0, keepdim=True) - distance) / (sample_count - 1)
        half = sample_count // 2
        pair = (((values[:half] - values[half:2 * half]) ** 2).sum(-1) / count).clamp_min(1e-12).sqrt()
        pair_baseline = (pair.sum(0, keepdim=True) - pair) / (half - 1) if half > 1 else torch.zeros_like(pair)
    energy = ((distance - baseline) * sampled_log_prob).mean(0) - 0.5 * (
        (pair - pair_baseline) * (sampled_log_prob[:half] + sampled_log_prob[half:2 * half])
    ).mean(0)
    if energy_only:
        return {"energy": energy * valid}
    with torch.no_grad():
        full = _frozen_path_losses(paths.detach(), actual_future, mask, coverage=coverage, signal_scale=signal_scale)
    result = {"energy": energy * valid, "width": torch.zeros_like(energy), "miss": torch.zeros_like(energy)}
    if sample_count < 4:
        for name in ("width", "miss"):
            result[name] = full[name] * sampled_log_prob.sum(0) * valid
        return result
    groups = torch.tensor_split(torch.arange(sample_count, device=paths.device), min(4, sample_count // 2))
    for group in groups:
        keep = torch.ones(sample_count, dtype=torch.bool, device=paths.device)
        keep[group] = False
        with torch.no_grad():
            reduced = _frozen_path_losses(paths.detach()[keep], actual_future, mask, coverage=coverage, signal_scale=signal_scale)
        score = sampled_log_prob[group].sum(0)
        for name in ("width", "miss"):
            result[name] = result[name] + (full[name] - reduced[name]) * score * valid
    return result



def _paths():
    return torch.tensor([
        [[1., 5., 2.]], [[2., 5., 3.]], [[3., 5., 4.]], [[4., 5., 5.]],
        [[1., 20., 5.]], [[2., 5., 4.]], [[3., 5., 3.]], [[4., 5., 2.]],
    ], dtype=torch.float64)


def test_paired_masks_and_manual_issued_width_miss():
    paths = _paths()
    target = torch.tensor([[20., 20., 20.]], dtype=paths.dtype)
    masks = [torch.tensor([[True, False, False]]), torch.tensor([[True, True, False]])]
    # The final two sorted max excursions are 2/sqrt(1.25) and
    # 8/sqrt(7); q=.9 interpolates 30% toward the last.
    scale = torch.tensor([1.25, 225.*7./64., 1.25], dtype=paths.dtype).sqrt()
    radius = .7 * 2. / scale[0] + .3 * 8. / torch.tensor(7., dtype=paths.dtype).sqrt()
    center = torch.tensor([[2., 5., 3.]], dtype=paths.dtype)
    lo, hi = (center-radius*scale).clamp_min(0), center+radius*scale
    actual_band = simultaneous_band(paths, .9)
    assert torch.allclose(actual_band[0], lo) and torch.allclose(actual_band[1], hi)
    for mask in masks:
        safe_target = torch.where(mask, target, torch.zeros_like(target))
        actual = path_losses(paths, target, mask, coverage=.9, path_band_geometry="issued_prefix")
        expected_width = ((hi-lo)*mask).sum(-1)/mask.sum(-1)
        expected_miss = (torch.relu(lo-safe_target)*mask).amax(-1)+(torch.relu(safe_target-hi)*mask).amax(-1)
        assert torch.allclose(actual["width"], expected_width)
        assert torch.allclose(actual["miss"], expected_miss)
        legacy = path_losses(paths, target, mask, coverage=.9)
        if not mask[0, 1]:
            assert not torch.equal(legacy["width"], actual["width"])
    # Geometry of issued paths is shared, while legacy geometry changes with evidence.
    bands = [simultaneous_band(torch.where(mask[None], paths, 0.), .9, mask) for mask in masks]
    assert not torch.equal(bands[0][1][:, 0], bands[1][1][:, 0])


@pytest.mark.parametrize("hidden", [float("nan"), float("inf"), -float("inf"), -999.])
def test_hidden_target_invariance_and_unknown_zero_grad(hidden):
    paths = _paths().repeat(1, 2, 1).requires_grad_()
    mask = torch.tensor([[True, False, True], [False, False, False]])
    target = torch.tensor([[20., hidden, 20.], [hidden, hidden, hidden]], dtype=paths.dtype)
    actual = path_losses(paths, target, mask, path_band_geometry="issued_prefix")
    clean = path_losses(paths, torch.where(mask, target, 0.), mask, path_band_geometry="issued_prefix")
    assert all(torch.equal(actual[k], clean[k]) for k in actual)
    assert all(v[1] == 0 for v in actual.values())
    sum(v.sum() for v in actual.values()).backward()
    assert torch.isfinite(paths.grad).all()
    assert torch.equal(paths.grad[:, 1], torch.zeros_like(paths.grad[:, 1]))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf"), -1.])
def test_reject_hidden_bad_generated_paths_only_in_issued_branch(bad):
    paths = _paths()
    paths[0, 0, 1] = bad
    mask = torch.tensor([[True, False, True]])
    target = torch.tensor([[3., float("nan"), 4.]])
    assert all(torch.isfinite(v).all() for v in path_losses(paths, target, mask).values())
    with pytest.raises(ValueError, match="generated paths"):
        path_losses(paths, target, mask, path_band_geometry="issued_prefix")
    with pytest.raises(ValueError, match="generated paths"):
        path_score_surrogates(paths, target, mask, torch.zeros(8, 1), path_band_geometry="issued_prefix")


@pytest.mark.parametrize("complete", [False, True])
def test_frozen_legacy_values_and_gradients_exact(complete):
    mask = torch.tensor([[True, complete, True]])
    target = torch.tensor([[8., 8. if complete else float("nan"), 9.]], dtype=torch.float64)
    results = []
    for fn, extra in [(_frozen_path_losses, {}), (path_losses, {}),
                      (path_losses, {"path_band_geometry": "observed_prefix"})]:
        paths = _paths().requires_grad_()
        terms = fn(paths, target, mask, **extra)
        sum(t.sum() for t in terms.values()).backward()
        results.append((terms, paths.grad))
    for terms, grad in results[1:]:
        assert list(terms) == list(results[0][0])
        assert all(torch.equal(terms[k], results[0][0][k]) for k in terms)
        assert torch.equal(grad, results[0][1])
    for fn, extra in [(_frozen_path_score_surrogates, {}), (path_score_surrogates, {}),
                      (path_score_surrogates, {"path_band_geometry": "observed_prefix"})]:
        score = torch.arange(8., dtype=torch.float64).reshape(8, 1).requires_grad_()
        terms = fn(_paths(), target, mask, score, **extra)
        sum(t.sum() for t in terms.values()).backward()
        if fn == _frozen_path_score_surrogates:
            old_terms, old_grad = terms, score.grad
        else:
            assert all(torch.equal(terms[k], old_terms[k]) for k in terms)
            assert torch.equal(score.grad, old_grad)
    if complete:
        paths = _paths().requires_grad_()
        issued = path_losses(paths, target, mask, path_band_geometry="issued_prefix")
        sum(t.sum() for t in issued.values()).backward()
        assert all(torch.equal(issued[k], results[0][0][k]) for k in issued)
        assert torch.equal(paths.grad, results[0][1])


@pytest.mark.parametrize("invalid", [None, True, 1, [], {}, "issued", "ISSUED_PREFIX", " issued_prefix"])
def test_strict_selector_and_objective_validation_before_sampling(invalid, monkeypatch):
    paths = _paths()
    target, mask = torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="path_band_geometry"):
        path_losses(paths, target, mask, path_band_geometry=invalid)
    with pytest.raises(ValueError, match="path_band_geometry"):
        path_score_surrogates(paths, target, mask, torch.zeros(8, 1), path_band_geometry=invalid)
    model = SignalDistribution("gru", 1, 3, hidden_size=8, num_layers=1, rank=2)
    def fail(*args, **kwargs):
        pytest.fail("Invalid selector reached sampling")
    monkeypatch.setattr(model, "sample", fail)
    with pytest.raises(ValueError, match="path_band_geometry"):
        model.objective({}, target, mask, 3., path_band_geometry=invalid)


def test_surrogate_propagates_selector_to_full_and_leave_group_out(monkeypatch):
    import pdm.trajectory_objectives as objectives
    original, calls = objectives.path_losses, []
    def recording(*args, **kwargs):
        calls.append((len(args[0]), kwargs.get("path_band_geometry")))
        return original(*args, **kwargs)
    monkeypatch.setattr(objectives, "path_losses", recording)
    score = torch.arange(8., dtype=torch.float64).reshape(8, 1).requires_grad_()
    terms = path_score_surrogates(_paths(), torch.tensor([[20., float("inf"), 20.]]),
                                 torch.tensor([[True, False, True]]), score,
                                 path_band_geometry="issued_prefix")
    assert calls == [(8, "issued_prefix")] + [(6, "issued_prefix")] * 4
    sum(t.sum() for t in terms.values()).backward()
    assert torch.isfinite(score.grad).all() and score.grad.abs().sum() > 0


def test_objective_prefix_propagation_rng_and_legacy_equivalence(monkeypatch):
    import pdm.models.signal_distribution as distribution
    torch.manual_seed(123)
    model = SignalDistribution("gru", 1, 3, hidden_size=8, num_layers=1, rank=2).eval()
    moments = model(torch.zeros(2, 4, 1), torch.tensor([.5, .7]))
    target = torch.tensor([[4., float("nan"), 5.], [float("nan")] * 3])
    mask = torch.isfinite(target)
    calls = []
    for name in ("path_losses", "path_score_surrogates"):
        original = getattr(distribution, name)
        def record(*args, _fn=original, _name=name, **kwargs):
            calls.append((_name, args[0].shape[-1], kwargs.get("energy_only", False),
                          kwargs.get("path_band_geometry", "observed_prefix")))
            return _fn(*args, **kwargs)
        monkeypatch.setattr(distribution, name, record)
    outputs, states = [], []
    for extra in ({}, {"path_band_geometry": "observed_prefix"}, {"path_band_geometry": "issued_prefix"}):
        generator = torch.Generator().manual_seed(44)
        outputs.append(model.objective(moments, target, mask, 3., n_samples=8,
                                      generator=generator, path_objective_prefix_lengths=(1, 3), **extra))
        states.append(generator.get_state())
    assert all(torch.equal(states[0], state) for state in states)
    assert all(torch.equal(outputs[0][key], outputs[1][key]) for key in outputs[0])
    last_calls = calls[-7:]
    assert last_calls[:4] == [(name, length, False, "issued_prefix")
                             for length in (1, 3) for name in ("path_losses", "path_score_surrogates")]
    assert all(call[2:] == (True, "observed_prefix") for call in last_calls[4:])
    outputs[2]["total"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.event_head.weight.grad.abs().sum() > 0
    source = inspect.getsource(SignalDistribution.objective)
    assert source.index("validate_path_band_geometry(") < source.index("self.sample(")


@pytest.mark.parametrize("sample_count", [2, 3, 8])
def test_unknown_categorical_rows_zero_and_complete_target_surrogate_equal(sample_count):
    paths = _paths()[:sample_count]
    score = torch.arange(float(sample_count), dtype=paths.dtype).reshape(-1, 1).requires_grad_()
    unknown = path_score_surrogates(paths, torch.full((1, 3), float("nan")),
                                   torch.zeros(1, 3, dtype=torch.bool), score,
                                   path_band_geometry="issued_prefix")
    assert all(torch.equal(v, torch.zeros_like(v)) for v in unknown.values())
    sum(v.sum() for v in unknown.values()).backward()
    assert torch.isfinite(score.grad).all() and torch.equal(score.grad, torch.zeros_like(score.grad))
    target, mask = torch.full((1, 3), 12.), torch.ones(1, 3, dtype=torch.bool)
    old = path_score_surrogates(paths, target, mask, score)
    issued = path_score_surrogates(paths, target, mask, score, path_band_geometry="issued_prefix")
    assert all(torch.equal(old[k], issued[k]) for k in old)
    old_grad = torch.autograd.grad(sum(v.sum() for v in old.values()), score)[0]
    issued_grad = torch.autograd.grad(sum(v.sum() for v in issued.values()), score)[0]
    assert torch.equal(old_grad, issued_grad)
