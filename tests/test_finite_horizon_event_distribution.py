"""Synthetic engineering checks only; no real training or warning-quality claim."""

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from pdm.io_util import sha256_file
from pdm.learned_trajectory import (
    _make_model,
    learned_params,
    load_learned_bundle,
    save_learned_bundle,
)
from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_objectives import first_entry_loss


def make_model(h=12, family="finite_horizon_mixture", location="log"):
    return SignalDistribution(
        "gru",
        2,
        h,
        hidden_size=8,
        num_layers=1,
        rank=2,
        dropout=0.0,
        event_location=location,
        event_distribution=family,
    ).double()


def fixed_logits(model, raw):
    with torch.no_grad():
        model.event_head.weight.zero_()
        model.event_head.bias.copy_(torch.tensor(raw, dtype=torch.float64))
    return model.entry_logits(torch.zeros(1, 8, dtype=torch.float64))


def raw_for(h, medians, scales=(0.3, 0.7, 1.2), weights=(0.2, 0.3, 0.5), s=0.37):
    return (
        [math.log(w) for w in weights]
        + [math.log((m - 0.05) / (h - m)) for m in medians]
        + [math.log(math.expm1(v - 0.08)) for v in scales]
        + [math.log(s / (1 - s))]
    )


def cdf(x, median, sigma):
    return (
        0.0 if x == 0 else 0.5 * math.erfc(-(math.log(x) - math.log(median)) / sigma / math.sqrt(2))
    )


def test_hand_bucket_masses_normalize_each_component_before_mixing():
    h, medians, scales, weights, s = 12, (1.0, 6.0, 11.0), (0.3, 0.7, 1.2), (0.2, 0.3, 0.5), 0.37
    model = make_model(h)
    p = fixed_logits(model, raw_for(h, medians, scales, weights, s)).softmax(-1)[0]
    expected = [
        (1 - s)
        * sum(
            w * (cdf(j, m, v) - cdf(j - 1, m, v)) / cdf(h, m, v)
            for w, m, v in zip(weights, medians, scales)
        )
        for j in range(1, h + 1)
    ] + [s]
    torch.testing.assert_close(
        p, torch.tensor(expected, dtype=torch.float64), rtol=1e-12, atol=1e-14
    )
    # Normalizing the mixture as a whole would change these probabilities.
    wrong = (
        (1 - s)
        * sum(w * cdf(3, m, v) for w, m, v in zip(weights, medians, scales))
        / sum(w * cdf(h, m, v) for w, m, v in zip(weights, medians, scales))
    )
    assert abs(p[:3].sum().item() - wrong) > 0.01
    assert model.event_head.out_features == 10


@pytest.mark.parametrize("h", [1, 3, 519])
@pytest.mark.parametrize("extreme", [False, True])
def test_horizon_bounds_initialization_and_stable_extreme_tails(h, extreme):
    model = make_model(h)
    median = 0.05 + (h - 0.05) * model.event_head.bias[3:6].sigmoid()
    expected = torch.minimum(
        torch.tensor([30.0, 60.0, 120.0], dtype=torch.float64),
        0.05 + (h - 0.05) * torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64),
    )
    torch.testing.assert_close(median, expected, rtol=1e-6, atol=1e-6)
    raw = model.event_head.bias.detach().tolist()
    if extreme:
        raw[:3], raw[3:6], raw[6:9], raw[9] = [1e4, -1e4, 0], [-1e4, 0, 1e4], [-1e4, 0, 1e4], -80
    logits = fixed_logits(model, raw)
    p = logits.softmax(-1)
    assert torch.isfinite(logits).all() and (p > 0).all()
    torch.testing.assert_close(p.sum(-1), torch.ones(1, dtype=torch.float64), rtol=0, atol=1e-14)
    params = model.event_head.bias
    medians = 0.05 + (h - 0.05) * params[3:6].sigmoid()
    assert ((medians >= 0.05) & (medians <= h)).all()
    fh = 0.5 * torch.erfc(
        -(math.log(h) - medians.log()) / (F.softplus(params[6:9]) + 0.08) / math.sqrt(2)
    )
    assert (fh >= 0.5).all()
    first_entry_loss(logits, no_entry_prefix=torch.tensor([h])).sum().backward()
    assert torch.isfinite(model.event_head.bias.grad).all()


def test_horizon_survival_gradient_is_independent_of_timing_and_finite_event_s_gradient():
    model = make_model()
    logits = fixed_logits(model, raw_for(12, (1.0, 6.0, 11.0)))
    survival = first_entry_loss(logits, no_entry_prefix=torch.tensor([12])).sum()
    grad = torch.autograd.grad(survival, model.event_head.bias, retain_graph=True)[0]
    torch.testing.assert_close(grad[:9], torch.zeros(9, dtype=torch.float64), rtol=0, atol=1e-14)
    torch.testing.assert_close(
        grad[9], torch.tensor(-0.63, dtype=torch.float64), rtol=0, atol=1e-14
    )
    allowed = torch.zeros(1, 13, dtype=torch.bool)
    allowed[0, 5] = True
    event = first_entry_loss(
        logits, event_allowed_mask=allowed, event_observed_mask=torch.tensor([True])
    ).sum()
    grad = torch.autograd.grad(event, model.event_head.bias)[0]
    torch.testing.assert_close(grad[9], torch.tensor(0.37, dtype=torch.float64), rtol=0, atol=1e-14)
    assert all(grad[a:b].abs().sum() > 0 for a, b in [(0, 3), (3, 6), (6, 9)])


def test_bracket_partial_censor_prefix_and_unknown_keep_existing_likelihood():
    model = make_model(6)
    logits = fixed_logits(model, raw_for(6, (0.5, 3.0, 5.0))).expand(4, -1)
    p = logits.softmax(-1)
    allowed = torch.zeros(4, 7, dtype=torch.bool)
    allowed[0, 1:4] = True
    evidence = dict(
        event_allowed_mask=allowed,
        event_observed_mask=torch.tensor([True, False, False, False]),
        no_entry_prefix=torch.tensor([0, 2, 6, 0]),
    )
    full = first_entry_loss(logits, **evidence)
    expected = -torch.stack((p[0, 1:4].sum(), p[1, 2:].sum(), p[2, -1], p[3].sum())).log()
    expected[-1] = 0
    torch.testing.assert_close(full, expected, rtol=0, atol=1e-14)
    prefix = first_entry_loss(logits, prefix_length=3, **evidence)
    torch.testing.assert_close(
        prefix[:3],
        -torch.stack((p[0, 1:].sum(), p[1, 2:].sum(), p[2, 3:].sum())).log(),
        rtol=0,
        atol=1e-14,
    )
    assert prefix[-1] == 0
    unknown_grad = torch.autograd.grad(full[-1], model.event_head.bias)[0]
    assert not unknown_grad.any()


def test_samples_have_exact_first_entry_support_and_already_red_is_unrestricted():
    model = make_model(12)
    with torch.no_grad():
        model.event_head.weight.zero_()
    moments = model(
        torch.ones(3, 4, 2, dtype=torch.float64), torch.tensor([0.5, 0.5, 4.0], dtype=torch.float64)
    )
    paths, entries = model.sample(
        moments, 3.0, 2048, torch.Generator().manual_seed(312), return_entry=True
    )
    crossing = paths[:, :2] >= 3
    actual = torch.where(crossing.any(-1), crossing.int().argmax(-1), 12)
    assert torch.equal(actual, entries[:, :2]) and (entries[:, 2] == -1).all()
    assert (paths >= 0).all() and torch.isfinite(paths).all()
    assert set(entries[:, 0].tolist()) == set(range(13))
    probability = moments["event_logits"].softmax(-1)[0]
    assert abs((entries[:, 0] == 12).double().mean().item() - probability[-1].item()) < 0.04


class FrozenLegacy(SignalDistribution):
    """Pre-family recurrent constructor and dense event calculation reference."""

    def __init__(self, h, location):
        nn.Module.__init__(self)
        self.n_horizons, self.rank, self.external_feature_size = h, 2, None
        self.event_location, self.phase_covariance = location, "shared"
        self.path_parameters_per_horizon = 6
        self.encoder = nn.GRU(2, 8, num_layers=1, batch_first=True, dropout=0)
        self.event_head = nn.Linear(8, 10)
        nn.init.normal_(self.event_head.weight, std=0.01)
        nn.init.zeros_(self.event_head.bias)
        with torch.no_grad():
            initial = torch.tensor([30.0, 60.0, 120.0])
            self.event_head.bias[3:6] = initial.log() if location == "log" else initial
            self.event_head.bias[6:9] = torch.log(
                torch.expm1(torch.tensor([0.5, 0.65, 0.9]) - 0.08)
            )
            self.event_head.bias[9] = torch.log(torch.tensor(0.2 / 0.8))
        self.path_head = nn.Sequential(
            nn.Linear(8, 8), nn.SiLU(), nn.Dropout(0), nn.Linear(8, h * 6)
        )
        nn.init.normal_(self.path_head[-1].weight, std=0.01)
        nn.init.zeros_(self.path_head[-1].bias)
        with torch.no_grad():
            self.path_head[-1].bias.reshape(h, 6)[:, 3] = -2.25

    def entry_logits(self, encoded):
        raw = self.event_head(encoded)
        params = raw.double()
        weights = params[:, :3].softmax(-1)
        lm = (
            params[:, 3:6]
            if self.event_location == "log"
            else (F.softplus(params[:, 3:6]) + 0.05).log()
        )
        scales = F.softplus(params[:, 6:9]) + 0.08
        cure = params[:, 9:].sigmoid()
        upper = torch.arange(1, self.n_horizons + 1, device=raw.device, dtype=torch.float64)
        zu = (upper.log()[None, :, None] - lm[:, None]) / scales[:, None]
        zl = torch.cat((torch.full_like(zu[:, :1], -torch.inf), zu[:, :-1]), 1)
        cu = 0.5 * torch.erfc(-zu / 2**0.5)
        cl = 0.5 * torch.erfc(-zl / 2**0.5)
        sl = 0.5 * torch.erfc(zl / 2**0.5)
        su = 0.5 * torch.erfc(zu / 2**0.5)
        bucket = torch.where(zl > 0, sl - su, cu - cl)
        finite = (1 - cure) * (bucket * weights[:, None]).sum(-1)
        no_entry = cure + (1 - cure) * (su[:, -1] * weights).sum(-1, keepdim=True)
        mass = torch.cat((finite, no_entry), -1).clamp_min(1e-30)
        return (mass.log() - mass.sum(-1, keepdim=True).log()).to(raw.dtype)


@pytest.mark.parametrize("location", ["log", "softplus"])
def test_default_legacy_exact_constructor_rng_logits_samples_and_gradients(location):
    torch.manual_seed(913)
    old = FrozenLegacy(12, location)
    old_rng = torch.get_rng_state()
    torch.manual_seed(913)
    new = SignalDistribution(
        "gru", 2, 12, hidden_size=8, num_layers=1, rank=2, dropout=0, event_location=location
    )
    assert torch.equal(torch.get_rng_state(), old_rng)
    assert all(
        torch.equal(a, b) for a, b in zip(old.state_dict().values(), new.state_dict().values())
    )
    x = torch.arange(16.0).reshape(2, 4, 2) / 16
    current = torch.tensor([0.4, 0.5])
    results = []
    for model in (old, new):
        moments = model(x, current)
        paths, entries = model.sample(
            moments, 3.0, 64, torch.Generator().manual_seed(42), return_entry=True
        )
        terms = model.objective(
            moments,
            torch.ones(2, 12),
            torch.ones(2, 12, dtype=torch.bool),
            3.0,
            no_entry_prefix=torch.tensor([6, 12]),
            n_samples=12,
            generator=torch.Generator().manual_seed(23),
        )
        terms["total"].sum().backward()
        results.append((moments, paths, entries, terms))
    assert all(torch.equal(results[0][0][key], results[1][0][key]) for key in results[0][0])
    assert all(torch.equal(results[0][i], results[1][i]) for i in [1, 2])
    assert all(torch.equal(results[0][3][key], results[1][3][key]) for key in results[0][3])
    assert all(torch.equal(a.grad, b.grad) for a, b in zip(old.parameters(), new.parameters()))


def features():
    return pd.DataFrame(
        {
            "unit_id": ["train"] * 7,
            "timestamp_s": np.arange(7) * 60.0,
            "gap_before": [True] + [False] * 6,
        }
    )


@pytest.mark.parametrize("invalid", [None, False, [], {}, "finite", ""])
def test_config_and_constructor_reject_invalid_family(invalid):
    with pytest.raises(ValueError, match="event_distribution"):
        learned_params("gru", {"event_distribution": invalid}, features())
    with pytest.raises(ValueError, match="event distribution"):
        make_model(family=invalid)


@pytest.mark.parametrize("family", ["lognormal_cure", "finite_horizon_mixture"])
def test_config_factory_saved_reload_and_valid_rehash_contradictions(tmp_path, family):
    config = learned_params(
        "gru",
        {
            "event_distribution": family,
            "horizons_s": [60.0, 120.0, 180.0],
            "hidden_size": 16,
            "num_layers": 1,
            "rank": 2,
            "dropout": 0,
        },
        features(),
    )
    model = _make_model("gru", config, 2, None).eval()
    bundle = {
        "model": model,
        "config": config,
        "scaler": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
        "external_scaler": None,
        "selection": {},
        "feature_names": ["a", "b"],
        "encoder": None,
        "trace": [],
    }
    artifacts = save_learned_bundle(bundle, tmp_path)
    run = {"dir": tmp_path, "engine_id": "gru", "params": config, "artifacts": artifacts}
    loaded = load_learned_bundle(run)
    assert loaded["event_distribution"] == family
    expected_location = "bounded_horizon_logistic" if family == "finite_horizon_mixture" else "log"
    assert loaded["effective_event_location"] == expected_location
    x = torch.ones(2, 4, 2)
    current = torch.tensor([0.5, 4.0])
    lhs = model(x, current)
    rhs = loaded["model"](x, current)
    assert all(torch.equal(lhs[key], rhs[key]) for key in lhs)
    for a, b in zip(
        model.sample(lhs, 3.0, 32, torch.Generator().manual_seed(23), return_entry=True),
        loaded["model"].sample(rhs, 3.0, 32, torch.Generator().manual_seed(23), return_entry=True),
    ):
        assert torch.equal(a, b)
    path = tmp_path / "learned_model.json"
    metadata = json.loads(path.read_text())
    for key, value, message in [
        (
            "event_distribution",
            "finite_horizon_mixture" if family == "lognormal_cure" else "lognormal_cure",
            "event distribution",
        ),
        ("effective_event_location", "incorrect", "effective event location"),
    ]:
        path.write_text(json.dumps({**metadata, key: value}))
        with pytest.raises(ValueError, match=message):
            load_learned_bundle(
                {**run, "artifacts": {**artifacts, "learned_model.json": sha256_file(path)}}
            )


def test_old_missing_family_and_optional_metadata_remains_exact_legacy(tmp_path):
    config = learned_params(
        "gru",
        {"horizons_s": [60.0, 120.0], "hidden_size": 16, "num_layers": 1, "rank": 2},
        features(),
    )
    assert config.pop("event_distribution") == "lognormal_cure"
    config.pop("event_location")  # Truly old reconstruction fallback is softplus.
    model = _make_model("gru", config, 2, None)
    assert model.event_distribution == "lognormal_cure" and model.event_location == "softplus"
    bundle = {
        "model": model,
        "config": config,
        "scaler": {},
        "external_scaler": None,
        "selection": {},
        "feature_names": ["a", "b"],
        "encoder": None,
        "trace": [],
    }
    artifacts = save_learned_bundle(bundle, tmp_path)
    path = tmp_path / "learned_model.json"
    metadata = json.loads(path.read_text())
    for key in ["event_distribution", "effective_event_location"]:
        metadata.pop(key)
    path.write_text(json.dumps(metadata))
    artifacts["learned_model.json"] = sha256_file(path)
    loaded = load_learned_bundle(
        {"dir": tmp_path, "engine_id": "gru", "params": config, "artifacts": artifacts}
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(model.state_dict().values(), loaded["model"].state_dict().values())
    )


def test_optional_family_preserves_rng_path_weights_and_ignores_legacy_location_flag():
    torch.manual_seed(71)
    legacy = make_model(100, "lognormal_cure")
    rng = torch.get_rng_state()
    torch.manual_seed(71)
    finite = make_model(100)
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in legacy.state_dict().items():
        if key != "event_head.bias":
            assert torch.equal(value, finite.state_dict()[key])
    for section in [slice(0, 3), slice(6, 10)]:
        assert torch.equal(legacy.event_head.bias[section], finite.event_head.bias[section])
    torch.manual_seed(71)
    softplus_flag = make_model(100, location="softplus")
    assert (
        finite.effective_event_location
        == softplus_flag.effective_event_location
        == "bounded_horizon_logistic"
    )
    assert all(
        torch.equal(a, b)
        for a, b in zip(finite.state_dict().values(), softplus_flag.state_dict().values())
    )


def test_optional_objective_unknown_row_has_zero_event_gradient_and_already_red_no_event():
    model = make_model(6)
    moments = model(
        torch.ones(3, 3, 2, dtype=torch.float64), torch.tensor([0.5, 0.5, 4.0], dtype=torch.float64)
    )
    moments["event_logits"].retain_grad()
    target = torch.full((3, 6), float("nan"), dtype=torch.float64)
    target[0, :2] = 0.6
    mask = torch.isfinite(target)
    terms = model.objective(
        moments,
        target,
        mask,
        3.0,
        no_entry_prefix=torch.tensor([2, 0, 6]),
        n_samples=8,
        generator=torch.Generator().manual_seed(18),
    )
    assert all(torch.isfinite(value).all() for value in terms.values())
    assert terms["event"][1] == terms["event"][2] == 0
    terms["total"].sum().backward()
    assert not moments["event_logits"].grad[1:].any()
    assert moments["event_logits"].grad[0].abs().sum() > 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_finite_family_explicitly_rejects_nonfinite_raw_parameters(bad):
    finite = make_model(3)
    raw = finite.event_head.bias.detach().tolist()
    raw[4] = bad
    with pytest.raises(ValueError, match="event parameters must be finite"):
        fixed_logits(finite, raw)
    # Legacy operation stays unchanged, including its existing nonfinite behavior.
    legacy = make_model(3, "lognormal_cure")
    logits = fixed_logits(legacy, raw)
    reference = FrozenLegacy(3, "log").double()
    torch.testing.assert_close(logits, fixed_logits(reference, raw), rtol=0, atol=0, equal_nan=True)


def finite_saved_run(tmp_path):
    config = learned_params(
        "gru",
        {
            "event_distribution": "finite_horizon_mixture",
            "horizons_s": [60.0, 120.0, 180.0],
            "hidden_size": 16,
            "num_layers": 1,
            "rank": 2,
        },
        features(),
    )
    model = _make_model("gru", config, 2, None)
    bundle = {
        "model": model,
        "config": config,
        "scaler": {},
        "external_scaler": None,
        "selection": {},
        "feature_names": ["a", "b"],
        "encoder": None,
        "trace": [],
    }
    artifacts = save_learned_bundle(bundle, tmp_path)
    return {"dir": tmp_path, "engine_id": "gru", "params": config, "artifacts": artifacts}


@pytest.mark.parametrize(
    "field", ["event_distribution", "effective_event_location", "event_distribution_contract"]
)
@pytest.mark.parametrize("mutation", ["missing", "contradiction"])
def test_finite_required_metadata_rejected_even_after_valid_rehash(tmp_path, field, mutation):
    run = finite_saved_run(tmp_path)
    path = tmp_path / "learned_model.json"
    metadata = json.loads(path.read_text())
    if mutation == "missing":
        metadata.pop(field)
    else:
        metadata[field] = "unknown_family_or_contract"
    path.write_text(json.dumps(metadata))
    run["artifacts"]["learned_model.json"] = sha256_file(path)
    with pytest.raises(ValueError, match=field.replace("_", " ")):
        load_learned_bundle(run)


@pytest.mark.parametrize(
    "field",
    [
        "horizon_steps",
        "horizons_s",
        "median_lower_steps",
        "median_upper_steps",
        "component_truncation",
        "survival_semantics",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "contradiction"])
def test_finite_contract_requires_exact_grid_bounds_truncation_and_semantics(
    tmp_path, field, mutation
):
    run = finite_saved_run(tmp_path)
    path = tmp_path / "learned_model.json"
    metadata = json.loads(path.read_text())
    contract = metadata["event_distribution_contract"]
    assert contract == {
        "horizon_steps": 3,
        "horizons_s": [60.0, 120.0, 180.0],
        "median_lower_steps": 0.05,
        "median_upper_steps": 3,
        "component_truncation": "per component at saved horizon",
        "survival_semantics": "no recorded entry through saved horizon",
    }
    if mutation == "missing":
        contract.pop(field)
    else:
        contract[field] = "contradiction"
    path.write_text(json.dumps(metadata))
    run["artifacts"]["learned_model.json"] = sha256_file(path)
    with pytest.raises(ValueError, match="event distribution contract"):
        load_learned_bundle(run)


@pytest.mark.parametrize(
    "surface", ["manifest", "training_contract.json", "model_input_contract.json"]
)
@pytest.mark.parametrize("mutation", ["missing", "contradiction"])
def test_full_run_surfaces_require_matching_finite_provenance(tmp_path, surface, mutation):
    run = finite_saved_run(tmp_path)
    metadata = json.loads((tmp_path / "learned_model.json").read_text())
    fields = {
        key: metadata[key]
        for key in ["event_distribution", "effective_event_location", "event_distribution_contract"]
    }
    run.update(schema_version="project_signal_forecast_learned_v1", **fields)
    for name in ["training_contract.json", "model_input_contract.json"]:
        path = tmp_path / name
        path.write_text(json.dumps(fields))
        run["artifacts"][name] = sha256_file(path)
    load_learned_bundle(run)  # All full-run surfaces agree.
    if surface == "manifest":
        changed = dict(run)
    else:
        changed = json.loads((tmp_path / surface).read_text())
    if mutation == "missing":
        changed.pop("event_distribution_contract")
    else:
        changed["event_distribution_contract"] = {
            **fields["event_distribution_contract"],
            "survival_semantics": "permanent cure",
        }
    if surface == "manifest":
        run = changed
    else:
        path = tmp_path / surface
        path.write_text(json.dumps(changed))
        run["artifacts"][surface] = sha256_file(path)
    with pytest.raises(ValueError, match="event distribution contract"):
        load_learned_bundle(run)
