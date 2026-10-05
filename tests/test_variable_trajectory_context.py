"""Synthetic causal-context checks; no real unit data, fits or Test access."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_data import (
    SENSOR_AVAILABILITY,
    SENSOR_FEATURE_NAMES,
    build_trajectory_frame,
    build_trajectory_prefix,
    causal_features,
    slice_frame,
)

FIXED = {"history_length": 8, "horizons_s": [60.0, 120.0, 180.0, 240.0], "target_tolerance_s": 0.01}
VARIABLE = {**FIXED, "recurrent_context_mode": "variable_causal", "max_history_length": 32}


def synthetic(n=40, entry=13, gaps=(), sensors=False):
    signal = 0.2 + 0.01 * np.arange(n)
    if entry is not None:
        signal[entry:] = 4.0 + 0.01 * np.arange(n - entry)
    gap = np.zeros(n, dtype=bool)
    if n:
        gap[0] = True
    gap[list(gaps)] = True
    features = pd.DataFrame(
        {
            "unit_id": "u",
            "physical_unit_id": "p",
            "timestamp_s": np.arange(n) * 60.0,
            "signal": signal,
            "gap_before": gap,
        }
    )
    schema = {"thresholds": {"mode": "absolute", "direction": "above", "red": 3.0}}
    if sensors:
        for i, name in enumerate(SENSOR_FEATURE_NAMES):
            features[name] = (np.arange(n) + 1.0) * (i + 1.0)
        features["horizontal_rms"] = signal
        features["vertical_rms"] = signal / 2
        schema["trajectory_sensor_features"] = {
            "schema_version": 1,
            "columns": list(SENSOR_FEATURE_NAMES),
            "transform": "log1p",
            "availability": SENSOR_AVAILABILITY,
        }
    return {"features": features, "schema": schema}


def variable_model(kind="gru", **kwargs):
    return SignalDistribution(
        kind,
        3,
        4,
        hidden_size=5,
        num_layers=1,
        rank=2,
        dropout=0,
        recurrent_context_mode="variable_causal",
        min_history_length=8,
        max_history_length=32,
        **kwargs,
    ).double()


@pytest.mark.parametrize("entry", [13, 33])
@pytest.mark.parametrize("gaps", [(), (20,)])
@pytest.mark.parametrize("cap", [None, 5])
def test_same_origins_labels_warnings_caps_and_chronology(entry, gaps, cap):
    data = synthetic(entry=entry, gaps=gaps)
    fixed = build_trajectory_frame(data, ["u"], FIXED, cap)
    variable = build_trajectory_frame(data, ["u"], VARIABLE, cap)
    for name in fixed:
        if name not in {"x", "raw_x"}:
            np.testing.assert_array_equal(variable[name], fixed[name])
    np.testing.assert_array_equal(variable["scaler_x"], fixed["x"])
    if not gaps and cap is None:
        assert len(variable["x"]) == 33
        assert variable["warning_eligible"].sum() == entry - 7
        assert variable["as_of_s"][0] == 7 * 60
    for i, at in enumerate(variable["as_of_s"]):
        end = int(at / 60)
        start = max([0] + [g for g in gaps if g <= end])
        length = min(32, end - start + 1)
        assert variable["history_lengths"][i] == length
        np.testing.assert_array_equal(variable["history_mask"][i], np.arange(32) < length)
        segment = data["features"].iloc[start : end + 1].reset_index(drop=True)
        expected = causal_features(segment)[-length:]
        np.testing.assert_array_equal(variable["x"][i, :length], expected)
        np.testing.assert_array_equal(variable["x"][i, length:], 0)
        np.testing.assert_array_equal(
            variable["raw_x"][i, :length, 0], segment.signal.to_numpy()[-length:].astype(np.float32)
        )
        np.testing.assert_array_equal(variable["raw_x"][i, length:], 0)
        np.testing.assert_array_equal(variable["x"][i, length - 8 : length], fixed["x"][i])


@pytest.mark.parametrize("sensor_mode", ["absolute", "baseline_relative", "combined"])
def test_full_causal_features_initial8_scaler_reference_and_prefix(sensor_mode):
    data = synthetic(n=150, entry=None, gaps=(100,), sensors=True)
    config = {**VARIABLE, "sensor_feature_mode": sensor_mode}
    frame = build_trajectory_frame(data, ["u"], config)
    fixed = build_trajectory_frame(data, ["u"], {**FIXED, "sensor_feature_mode": sensor_mode})
    np.testing.assert_array_equal(frame["scaler_x"], fixed["x"])
    # This is the old per-origin last-eight scaler population, byte-identical.
    weight = np.full(len(frame["x"]), 1 / len(frame["x"]))
    for values in (frame["scaler_x"], fixed["x"]):
        mean = np.einsum("n,ntf->f", weight / 8, values)
        std = np.sqrt(np.einsum("n,ntf->f", weight / 8, (values - mean) ** 2))
        if values is frame["scaler_x"]:
            original = (mean, std)
        else:
            np.testing.assert_array_equal(mean, original[0])
            np.testing.assert_array_equal(std, original[1])
    for at in (7 * 60, 12 * 60, 32 * 60, 99 * 60, 107 * 60, 149 * 60):
        index = frame["as_of_s"].index(at)
        prefix = data["features"].loc[data["features"].timestamp_s <= at]
        issued = build_trajectory_prefix(data, prefix, config)
        for key in ("x", "raw_x", "scaler_x", "history_lengths", "history_mask", "current"):
            np.testing.assert_array_equal(issued[key][0], frame[key][index])
        assert not issued["mask"].any()
        changed = {**data, "features": data["features"].copy()}
        later = changed["features"].timestamp_s > at
        changed["features"].loc[later, list(SENSOR_FEATURE_NAMES)] *= 100
        changed["features"]["signal"] = changed["features"].horizontal_rms
        future_mutated = build_trajectory_frame(changed, ["u"], config)
        np.testing.assert_array_equal(future_mutated["x"][index], frame["x"][index])


def test_empty_short_segments_and_all_new_fields_slice():
    for n in (0, 7):
        frame = build_trajectory_frame(synthetic(n=n, entry=None), ["u"], VARIABLE)
        assert frame["x"].shape == (0, 32, 13)
        assert frame["raw_x"].shape == (0, 32, 1)
        assert frame["scaler_x"].shape == (0, 8, 13)
        assert frame["history_mask"].shape == (0, 32)
        assert frame["history_lengths"].shape == (0,)
        assert frame["history_lengths"].dtype == np.int64
    full = build_trajectory_frame(synthetic(), ["u"], VARIABLE)
    sliced = slice_frame(full, np.asarray([4, 1, 10]))
    for name in ("x", "raw_x", "scaler_x", "history_lengths", "history_mask"):
        np.testing.assert_array_equal(sliced[name], full[name][[4, 1, 10]])
    with pytest.raises(ValueError, match="Insufficient"):
        data = synthetic(n=15, entry=None, gaps=(10,))
        build_trajectory_prefix(data, data["features"], VARIABLE)


@pytest.mark.parametrize("kind", ["gru", "lstm"])
def test_packed_order_matches_each_real_history_and_padding_gradient_invariance(kind):
    torch.manual_seed(10)
    m = variable_model(kind)
    lengths = torch.tensor([13, 32, 8, 17])
    mask = torch.arange(32)[None] < lengths[:, None]
    x = torch.randn(4, 32, 3, dtype=torch.float64)
    current = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float64)
    reference = m(x, current, history_lengths=lengths, history_mask=mask)
    fixed = SignalDistribution(kind, 3, 4, hidden_size=5, num_layers=1, rank=2, dropout=0).double()
    fixed.load_state_dict(m.state_dict())
    for row, length in enumerate(lengths):
        real = fixed(x[row : row + 1, :length], current[row : row + 1])
        for name in reference:
            torch.testing.assert_close(
                reference[name][row : row + 1], real[name], rtol=1e-12, atol=1e-12
            )
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in fixed.parameters())
    results = []
    for pad in (0.0, 1e100, float("nan"), float("inf")):
        mutated = x.clone()
        mutated[~mask] = pad
        mutated.requires_grad_()
        m.zero_grad()
        result = m(mutated, current, history_lengths=lengths, history_mask=mask)
        for name in result:
            assert torch.equal(result[name], reference[name])
        assert torch.equal(
            m.sample(result, 3.0, 8, torch.Generator().manual_seed(81)),
            m.sample(reference, 3.0, 8, torch.Generator().manual_seed(81)),
        )
        result["mean"].sum().backward()
        assert mutated.grad[~mask].abs().sum() == 0
        results.append(
            ([p.grad.clone() for p in m.parameters() if p.grad is not None], mutated.grad.clone())
        )
    for parameter_grads, x_grad in results[1:]:
        for actual, expected in zip(parameter_grads, results[0][0]):
            assert torch.equal(actual, expected)
        assert torch.equal(x_grad, results[0][1])


@pytest.mark.parametrize(
    "fault",
    [
        "missing_lengths",
        "missing_mask",
        "float_lengths",
        "bool_lengths",
        "short",
        "long",
        "batch",
        "mask_dtype",
        "mask_shape",
        "mask_gap",
        "time",
        "nonfinite",
    ],
)
def test_variable_forward_rejects_invalid_or_missing_contract(fault):
    m = variable_model()
    x = torch.zeros(2, 32, 3, dtype=torch.float64)
    lengths = torch.tensor([8, 12])
    mask = torch.arange(32)[None] < lengths[:, None]
    if fault == "missing_lengths":
        lengths = None
    elif fault == "missing_mask":
        mask = None
    elif fault == "float_lengths":
        lengths = lengths.double()
    elif fault == "bool_lengths":
        lengths = lengths.bool()
    elif fault == "short":
        lengths[0] = 7
    elif fault == "long":
        lengths[0] = 33
    elif fault == "batch":
        lengths = lengths[:1]
    elif fault == "mask_dtype":
        mask = mask.long()
    elif fault == "mask_shape":
        mask = mask[:, :30]
    elif fault == "mask_gap":
        mask[0, 2] = False
    elif fault == "time":
        x = x[:, :31]
    elif fault == "nonfinite":
        x[0, 2, 0] = torch.nan
    with pytest.raises(ValueError):
        m(x, torch.ones(2, dtype=torch.float64), history_lengths=lengths, history_mask=mask)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_history_length": None},
        {"max_history_length": None},
        {"min_history_length": 0},
        {"max_history_length": 7},
        {"max_history_length": True},
        {"external_feature_size": 3},
        {"architecture": "external"},
        {"recurrent_context_mode": []},
    ],
)
def test_variable_constructor_rejects_undeclared_or_invalid_context(kwargs):
    options = {
        "architecture": "gru",
        "input_size": 3,
        "n_horizons": 4,
        "recurrent_context_mode": "variable_causal",
        "min_history_length": 8,
        "max_history_length": 32,
    }
    with pytest.raises(ValueError):
        SignalDistribution(**{**options, **kwargs})


def frame_digest(frame):
    checksum = hashlib.sha256()
    for key, value in frame.items():
        checksum.update(key.encode())
        if isinstance(value, np.ndarray):
            checksum.update(str(value.shape).encode())
            checksum.update(str(value.dtype).encode())
            checksum.update(value.tobytes())
        else:
            checksum.update(json.dumps(value).encode())
    return checksum.hexdigest()


@pytest.mark.parametrize("selector", [None, "fixed"])
def test_fixed_builder_frozen_reference(selector):
    config = FIXED if selector is None else {**FIXED, "recurrent_context_mode": selector}
    frame = build_trajectory_frame(
        synthetic(n=40, entry=13, gaps=(20,), sensors=True),
        ["u"],
        {**config, "sensor_feature_mode": "combined"},
        cap=7,
    )
    assert frame_digest(frame) == "de036d32a3b1b3bcd4b1fbce438d148008336ea252d843136181dfc1e5de2339"


def tensor_digest(tensors):
    checksum = hashlib.sha256()
    for tensor in tensors:
        checksum.update(tensor.detach().contiguous().numpy().tobytes())
    return checksum.hexdigest()


@pytest.mark.parametrize("context", [None, "fixed"])
def test_fixed_model_frozen_state_rng_moments_samples_loss_and_gradients(context):
    with torch.random.fork_rng():
        torch.manual_seed(91)
        kwargs = {} if context is None else {"recurrent_context_mode": context}
        m = SignalDistribution(
            "gru",
            2,
            4,
            hidden_size=5,
            num_layers=1,
            rank=2,
            dropout=0,
            event_distribution="finite_horizon_mixture",
            phase_covariance="separate",
            **kwargs,
        ).double()
        assert (
            tensor_digest(m.state_dict().values())
            == "30f8007edf90d1cfcc3e061b1f764959ded529816028100085fb78bf0c4ff6bb"
        )
        assert (
            tensor_digest([torch.random.get_rng_state()])
            == "1f7b268d9bd74714511c9ff938b6f31fbe3f2ca0eedd756d0a78636b5f13644a"
        )
        a = m(
            torch.linspace(-0.5, 0.5, 24, dtype=torch.float64).reshape(2, 6, 2),
            torch.tensor([0.2, 3.0], dtype=torch.float64),
        )
        assert (
            tensor_digest(a.values())
            == "fe59feec966a96eb8eb7cc9e92cecf333357ad2c2f2ba4dd675e9e23a3534f1e"
        )
        p, e = m.sample(a, 3.0, 8, torch.Generator().manual_seed(12), return_entry=True)
        assert (
            tensor_digest([p, e])
            == "b5dc3256b8cb62e11746d524992e97b7c8d8c3186a482882eb451a3ac2ebfe44"
        )
        terms = m.objective(
            a,
            torch.full((2, 4), 4.0, dtype=torch.float64),
            torch.ones(2, 4, dtype=torch.bool),
            3.0,
            n_samples=8,
            generator=torch.Generator().manual_seed(14),
            no_entry_prefix=torch.tensor([4, 0]),
        )
        assert (
            tensor_digest(terms.values())
            == "6bbd3c5595bbcb59e210ac375f87549ce09a3f22da5e59d160c31fb056375e2e"
        )
        terms["total"].sum().backward()
        assert (
            tensor_digest(p.grad for p in m.parameters())
            == "6aa1e8ffed702eed5530241fd1510659eec9b56962aa5a02a8ff2fb856f5fd92"
        )


def test_authoritative_single_model_capacity_unchanged():
    for context in ("fixed", "variable_causal"):
        options = {} if context == "fixed" else {"min_history_length": 8, "max_history_length": 32}
        m = SignalDistribution(
            "gru",
            33,
            519,
            hidden_size=64,
            num_layers=1,
            rank=6,
            phase_covariance="separate",
            event_distribution="finite_horizon_mixture",
            recurrent_context_mode=context,
            **options,
        )
        assert m.path_distribution == "single"
        assert sum(p.numel() for p in m.parameters()) == 833458
