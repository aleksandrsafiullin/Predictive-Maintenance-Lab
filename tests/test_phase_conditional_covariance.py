"""Synthetic engineering checks, not evidence of Bearings warning quality."""
import copy
import json

import pandas as pd
import pytest
import torch
from torch.nn import functional as F

from pdm.io_util import sha256_file
from pdm.learned_trajectory import (
    _make_model,
    learned_params,
    load_learned_bundle,
    save_learned_bundle,
)
from pdm.models.signal_distribution import SignalDistribution


def make_model(mode="separate"):
    torch.manual_seed(9)
    return SignalDistribution("external", 1, 6, hidden_size=8, rank=2,
                              external_feature_size=3, dropout=0, phase_covariance=mode)


def legacy_sample(model, moments, generator):
    """Frozen pre-change sampler reference, including draw order and transforms."""
    samples = 12
    entries = torch.multinomial(moments["event_logits"].softmax(-1), samples,
                                replacement=True, generator=generator).T
    mean = moments["mean"][None].expand(samples, -1, -1)
    eps = torch.randn(mean.shape, generator=generator)
    z = torch.randn((*mean.shape[:2], model.rank), generator=generator)
    noise = moments["diagonal_scale"][None] * eps + (moments["factors"][None] * z[..., None, :]).sum(-1)
    threshold = torch.tensor(3.)
    current = moments["current_signal"][None, :, None]
    time = torch.arange(model.n_horizons)[None, None]
    tiny = torch.finfo(mean.dtype).eps
    fraction = (current / threshold).clamp(min=tiny, max=1-tiny)
    below = threshold * torch.sigmoid(torch.logit(fraction) + mean + noise).clamp(max=1-tiny)
    def inverse(value):
        return value + torch.log(-torch.expm1(-value))
    entering = threshold + F.softplus(inverse(threshold * .01) + moments["entry_mean"][None] + noise)
    entering = entering.maximum(torch.nextafter(threshold, torch.full_like(threshold, torch.inf)))
    elapsed = (time - entries[..., None] - 1).clamp(0, model.n_horizons-1)
    post_mean = moments["post_mean"][None].expand(samples, -1, -1).gather(-1, elapsed)
    post = F.softplus(inverse(torch.maximum(current, threshold)) + post_mean + noise)
    paths = torch.where(time < entries[..., None], below,
                        torch.where(time == entries[..., None], entering, post))
    already = current >= threshold
    unrestricted = F.softplus(inverse(current.clamp_min(tiny)) + moments["post_mean"][None] + noise)
    return torch.where(already, unrestricted, paths), torch.where(already[..., 0], -torch.ones_like(entries), entries)


def test_shared_values_rng_and_gradients_match_frozen_legacy():
    model = make_model("shared")
    torch.manual_seed(9)
    default = SignalDistribution("external", 1, 6, hidden_size=8, rank=2,
                                 external_feature_size=3, dropout=0)
    assert all(torch.equal(value, default.state_dict()[name]) for name, value in model.state_dict().items())
    reference = copy.deepcopy(model)
    current, features = torch.tensor([.5, 4.]), torch.ones(2, 3)
    moments = model(None, current, features)
    expected_moments = reference(None, current, features)
    raw = model.path_head(model.encoder(features)).reshape(2, 6, 6)
    assert torch.equal(moments["diagonal_scale"], F.softplus(raw[..., 3]) + 1e-3)
    assert torch.equal(moments["factors"], raw[..., 4:] / 2**.5)
    generator = torch.Generator().manual_seed(83)
    legacy_generator = torch.Generator().manual_seed(83)
    paths, entries = model.sample(moments, 3., 12, generator, return_entry=True)
    expected, expected_entries = legacy_sample(reference, expected_moments, legacy_generator)
    assert torch.equal(paths, expected) and torch.equal(entries, expected_entries)
    assert torch.equal(generator.get_state(), legacy_generator.get_state())
    paths.square().sum().backward()
    expected.square().sum().backward()
    for name, parameter in model.named_parameters():
        gradient = dict(reference.named_parameters())[name].grad
        assert (parameter.grad is None and gradient is None) or torch.equal(parameter.grad, gradient)
    assert model.path_head[-1].out_features == 6 * (4 + 2)


def fixed_moments(model):
    moments = model(None, torch.tensor([.5, .5, 4.]), torch.ones(3, 3))
    # One sampled event with all three phases, no-entry, and an already-RED row.
    moments["event_logits"] = torch.full((3, 7), -torch.inf)
    moments["event_logits"][0, 2] = 0
    moments["event_logits"][1, 6] = 0
    moments["event_logits"][2, 2] = 0
    return moments


@pytest.mark.parametrize("phase", ["pre", "entry", "post"])
@pytest.mark.parametrize("parameter", ["diagonal_scale", "factors"])
def test_separate_noise_changes_only_its_phase_including_already_red(phase, parameter):
    model = make_model()
    moments = fixed_moments(model)
    base, entries = model.sample(moments, 3., 16, torch.Generator().manual_seed(31), return_entry=True)
    changed = dict(moments)
    key = phase + "_" + parameter
    changed[key] = moments[key] + .5
    actual = model.sample(changed, 3., 16, torch.Generator().manual_seed(31))
    time = torch.arange(6)[None, None]
    phase_mask = {"pre": time < entries[..., None], "entry": time == entries[..., None],
                  "post": time > entries[..., None]}[phase]
    assert torch.equal(base[~phase_mask], actual[~phase_mask])
    assert (base[phase_mask] != actual[phase_mask]).any()
    # The already-RED row has entry=-1, so only post covariance can change it.
    assert (not torch.equal(base[:, 2], actual[:, 2])) == (phase == "post")


def test_separate_first_recorded_bucket_support_and_common_low_rank_latent():
    model = make_model()
    moments = fixed_moments(model)
    paths, entries = model.sample(moments, 3., 32, torch.Generator().manual_seed(42), return_entry=True)
    crossing = paths[:, :2] >= 3.
    first = torch.where(crossing.any(-1), crossing.int().argmax(-1), 6)
    assert torch.equal(first, entries[:, :2])
    assert (paths >= 0).all()
    # Identical factor rows with zero diagonal innovations expose the common z.
    for phase in ("pre", "entry", "post"):
        moments[phase + "_diagonal_scale"] = torch.zeros_like(moments[phase + "_diagonal_scale"])
        moments[phase + "_factors"] = torch.ones_like(moments[phase + "_factors"])
    for name in ("mean", "entry_mean", "post_mean"):
        moments[name] = torch.zeros_like(moments[name])
    paths = model.sample(moments, 3., 16, torch.Generator().manual_seed(5))[:, 0]
    def inverse(value):
        return value + torch.log(-torch.expm1(-value))
    pre_z = torch.logit(paths[:, 0] / 3.) - torch.logit(torch.tensor(.5 / 3.))
    entry_z = inverse(paths[:, 2] - 3.) - inverse(torch.tensor(.03))
    post_z = inverse(paths[:, 3]) - inverse(torch.tensor(3.))
    assert torch.allclose(pre_z, entry_z, atol=3e-5, rtol=3e-5)
    assert torch.allclose(pre_z, post_z, atol=3e-5, rtol=3e-5)


def test_observed_phase_objective_has_gradients_to_all_covariance_blocks():
    model = make_model()
    moments = model(None, torch.tensor([.5]), torch.ones(1, 3))
    target = torch.tensor([[.7, .9, 3.5, 5., 4., 6.]])
    allowed = torch.zeros(1, 7, dtype=torch.bool)
    allowed[0, 2] = True
    terms = model.objective(moments, target, torch.ones_like(target, dtype=torch.bool), 3.,
                            event_allowed_mask=allowed, event_observed_mask=torch.ones(1, dtype=torch.bool),
                            n_samples=32, generator=torch.Generator().manual_seed(32))
    # Real observed-phase energy, not a sum of moments or artificial loss.
    terms["phase_energy"].sum().backward()
    gradient = model.path_head[-1].bias.grad.reshape(6, 12)
    for index, phase in enumerate(("pre", "entry", "post")):
        assert torch.isfinite(gradient).all()
        assert gradient[:, 3 + index].abs().sum() > 0, phase
        assert gradient[:, 6 + index * 2:8 + index * 2].abs().sum() > 0, phase


@pytest.mark.parametrize("mode", [None, "shared", "separate"])
def test_config_head_shape_and_verified_exact_reload_without_fit(tmp_path, mode):
    features = pd.DataFrame({"unit_id": ["u"] * 4, "timestamp_s": [0., 60., 120., 180.],
                             "gap_before": [True, False, False, False]})
    params = {"horizons_s": [60., 120., 180.], "hidden_size": 16, "rank": 2, "dropout": 0.}
    if mode is not None:
        params["phase_covariance"] = mode
    config = learned_params("gru", params, features)
    assert config["phase_covariance"] == (mode or "shared")
    # Missing optional field simulates an older saved config, not new defaults.
    if mode is None:
        del config["phase_covariance"]
    model = _make_model("gru", config, 2, None).eval()
    assert model.path_head[-1].out_features == 3 * (12 if mode == "separate" else 6)
    bundle = {"model": model, "config": config, "encoder": None, "external_scaler": None,
              "selection": {}, "feature_names": ["a", "b"], "trace": [], "scaler": {}}
    hashes = save_learned_bundle(bundle, tmp_path)
    metadata = json.loads((tmp_path / "learned_model.json").read_text())
    assert metadata["phase_covariance"] == (mode or "shared")
    assert metadata["path_head_outputs"] == model.path_head[-1].out_features
    if mode is None:
        # An older artifact has neither optional config nor metadata fields.
        del metadata["phase_covariance"], metadata["path_head_outputs"]
        (tmp_path / "learned_model.json").write_text(json.dumps(metadata))
        hashes["learned_model.json"] = sha256_file(tmp_path / "learned_model.json")
    run = {"dir": tmp_path, "params": config, "artifacts": hashes, "engine_id": "gru"}
    loaded = load_learned_bundle(run)
    x, current = torch.ones(1, 2, 2), torch.tensor([.5])
    before = model.sample(model(x, current), 3., 16, torch.Generator().manual_seed(44))
    after = loaded["model"].sample(loaded["model"](x, current), 3., 16, torch.Generator().manual_seed(44))
    assert torch.equal(before, after)
    with (tmp_path / "checkpoint.pt").open("ab") as stream:
        stream.write(b"hash tampering")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_learned_bundle(run)


@pytest.mark.parametrize("value", ["other", None, [], 1])
def test_invalid_phase_covariance_rejected(value):
    features = pd.DataFrame({"unit_id": ["u"] * 2, "timestamp_s": [0., 60.],
                             "gap_before": [True, False]})
    with pytest.raises(ValueError, match="phase_covariance"):
        learned_params("gru", {"phase_covariance": value}, features)
