from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.forecasting import (
    continuous_states,
    fit_interval_profile,
    predict_failure_interval,
    score_readout,
    unit_weights,
    weighted_ridge,
)
from pdm.models.reservoir import LeakyESN


def model_fixture():
    return LeakyESN(torch.tensor([[0.3], [-0.2]]), torch.tensor([[0.5, 0.2], [-0.1, 0.3]]),
                    torch.zeros(2), alpha=0.2, state_mode="continuous")


def test_continuous_kernel_equals_incremental_and_remembers_old_history():
    model = model_fixture()
    values = np.linspace(0, 1, 40, dtype=np.float32)[:, None]
    whole = continuous_states(model, values)
    first = model.forward_states(torch.from_numpy(values[:20]))
    second = model.forward_states(torch.from_numpy(values[20:]), x0=first[-1])
    np.testing.assert_allclose(whole[20:], second.numpy(), atol=1e-7)
    changed = values.copy()
    changed[:20] += 10
    assert not np.allclose(whole[20], continuous_states(model, changed)[20])
    np.testing.assert_array_equal(whole[:20], continuous_states(model, values[:20]))


def test_gap_resets_neurons_and_forecast_warmup():
    model = model_fixture()
    values = np.ones((50, 1), dtype=np.float32)
    gaps = np.zeros(50, bool)
    gaps[25] = True
    states = continuous_states(model, values, gaps)
    np.testing.assert_array_equal(states[25:], continuous_states(model, values[25:]))
    result = predict_failure_interval(np.arange(50)*60, np.full(50, 600), {"version": 1}, gap_before=gaps)
    assert result.predicted_rul_s.iloc[:19].isna().all()
    assert result.predicted_rul_s.iloc[25:44].isna().all()
    assert np.isfinite(result.predicted_rul_s.iloc[[19, 44]]).all()


def test_interval_is_causal_and_not_forced_to_narrow():
    profile = {"version": 1, "warmup_measurements": 1, "residual_scale": {"intercept": 5, "slope": 0.5},
               "residual_q_low": -1, "residual_q_high": 2}
    raw = np.r_[np.full(20, 500), np.full(20, 5000)]
    full = predict_failure_interval(np.arange(40)*60, raw, profile)
    prefix = predict_failure_interval(np.arange(20)*60, raw[:20], profile)
    pd.testing.assert_frame_equal(full.iloc[:20], prefix)
    width = full.upper_rul_s - full.lower_rul_s
    assert width.iloc[-1] > width.iloc[19]
    assert (full.lower_rul_s <= full.predicted_rul_s).all()
    assert (full.predicted_rul_s <= full.upper_rul_s).all()


def test_equal_bearing_ridge_ignores_duplicate_windows_in_one_bearing():
    z = np.array([[0, 1], [1, 1], [2, 1], [3, 1]], dtype=float)
    y = np.array([0, 2, 3, 5], dtype=float)
    ids = np.array(["a", "a", "b", "b"])
    original = weighted_ridge(z, y, ids, 0.1)
    pick = [0, 1, 0, 1, 2, 3]
    np.testing.assert_allclose(original, weighted_ridge(z[pick], y[pick], ids[pick], 0.1))
    assert unit_weights(ids)[:2].sum() == unit_weights(ids)[2:].sum()


def test_score_and_runtime_match_after_gap_and_warmup():
    ts = np.arange(60)*60.0
    gaps = np.zeros(60, bool)
    gaps[30] = True
    raw = np.linspace(3000, 1000, 60)
    profile = {"version": 1, "residual_q_low": -2, "residual_q_high": 3}
    expected = predict_failure_interval(ts, raw, profile, gap_before=gaps)
    eligible = expected.predicted_rul_s.notna().to_numpy()
    indices = np.flatnonzero(eligible)
    rows = pd.DataFrame({"unit_id": "a", "timestamp_s": ts[eligible], "target_rul_s": raw[eligible],
                         "gap_before": np.r_[True, np.diff(indices)>1]})
    actual = score_readout(raw[eligible, None], rows, np.ones(1), 1.0, profile)
    np.testing.assert_allclose(actual.predicted_rul_s, expected.predicted_rul_s[eligible])
    np.testing.assert_allclose(actual.lower_rul_s, expected.lower_rul_s[eligible])


def test_calibration_uses_only_declared_units_and_no_probability_guarantee():
    train = pd.DataFrame({"unit_id": ["a"]*3, "target_rul_s": [600, 400, 200], "predicted_rul_s": [700, 500, 300]})
    cal = train.assign(unit_id="b", target_rul_s=[900, 700, 500])
    split = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    profile = fit_interval_profile(train, cal, split)
    assert profile["version"] == 1 and profile["coverage_guarantee"] is False
    assert profile["calibration_ids"] == ["b"]
    assert profile["residual_q_high"] > 0
    with pytest.raises(ValueError, match="split identity"):
        fit_interval_profile(train, cal.assign(unit_id="c"), split)


def test_profile_is_bound_to_checkpoint_preprocessing_graph_and_split(tmp_path):
    from pdm.forecasting import load_interval_profile
    from pdm.io_util import atomic_write_json, sha256_file

    files = {"checkpoint_sha256": "best.pt", "preprocessing_sha256": "preprocessing.json",
             "graph_sha256": "connectome/graph.json", "dataset_fingerprint_sha256": "dataset_fingerprint.json"}
    for relative in files.values():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("artifact")
    profile = {"version": 1, "ready": True, "train_ids": ["a"], "calibration_ids": ["b"], "test_ids": ["c"],
               "source_hashes": {key: sha256_file(tmp_path / name) for key, name in files.items()}}
    atomic_write_json(tmp_path / "interval_profile.json", profile)
    atomic_write_json(tmp_path / "split.json", {"train": ["a"], "validation": ["b"], "test": ["c"]})
    assert load_interval_profile(tmp_path)["version"] == 1
    (tmp_path / "preprocessing.json").write_text("changed")
    with pytest.raises(ValueError, match="preprocessing.json"):
        load_interval_profile(tmp_path)


def test_log_target_inverse_and_model_postprocessing_agree():
    from pdm.forecasting import decode_rul_outputs, encode_rul_targets

    y = np.array([0.0, 60.0, 6000.0, 150000.0])
    encoded = encode_rul_targets(y, 30000, "log1p")
    np.testing.assert_allclose(decode_rul_outputs(encoded, 30000, "log1p"), y)
    model = model_fixture()
    model.time_scale_s = 30000
    model.rul_transform = "log1p"
    value = model._postprocess(torch.tensor(encoded, dtype=torch.float32)[:, None]).numpy()*model.time_scale_s
    np.testing.assert_allclose(value, y, rtol=2e-6, atol=1e-4)
    assert torch.isfinite(model._postprocess(torch.tensor([[1000.0]]))).all()


def test_checkpoint_roundtrip_preserves_log_transform(tmp_path):
    from pdm.io_util import atomic_write_json
    from pdm.models.fly_reservoir import FlyConnectomeReservoir
    from pdm.preprocessing import Preprocessor
    from pdm.train import _save_ckpt, _write_connectome_artifacts, load_trained_model

    model = FlyConnectomeReservoir(np.eye(2), input_size=1, state_mode="continuous", time_scale_s=30000)
    model.rul_transform = "log1p"
    model.rul_reference_s = 60.0
    model.readout.load_ridge_vector(np.array([0.1, 0.2, 0.3, 4.0]))
    prep = Preprocessor(["horizontal_rms"], [], [0], [1], 30000, fill_values={"horizontal_rms": 0}, dataset_id="bearings")
    mcfg = {"architecture": "fly_connectome_reservoir", "history_length": 20,
            "hidden_size": 2, "recurrent_layers": 1, "dropout": 0,
            "reservoir": {"state_mode": "continuous"}}
    split = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    _write_connectome_artifacts(tmp_path, model)
    atomic_write_json(tmp_path / "preprocessing.json", prep.to_dict())
    atomic_write_json(tmp_path / "split.json", split)
    _save_ckpt(tmp_path / "best.pt", model, None, 1, 1, 0, mcfg, prep, split, "bearings", "rul", False)
    loaded, _, meta = load_trained_model(tmp_path)
    assert meta["rul_transform"] == loaded.rul_transform == "log1p"
    assert loaded.rul_reference_s == 60.0
    inputs = torch.ones((1, 20, 1))
    torch.testing.assert_close(loaded.predicted_rul_s(inputs), model.predicted_rul_s(inputs))


def test_scoring_uses_runtime_float32_readout_and_log_inverse():
    from pdm.models.readout import ridge_design_matrix

    model = model_fixture()
    model.rul_transform = "log1p"
    model.time_scale_s = 30000
    weights = np.array([0.1234, -0.6789, 0.5432, 5.678], dtype=np.float32)
    model.readout.load_ridge_vector(weights)
    inputs = torch.linspace(0.1, 1, 25).reshape(-1, 1)
    states = model.forward_states(inputs)
    with torch.no_grad():
        expected = model._postprocess(model.forward_raw(states, inputs))*model.time_scale_s
    rows = pd.DataFrame({"unit_id": "a", "timestamp_s": np.arange(25)*60,
                         "target_rul_s": 1000, "gap_before": False})
    actual = score_readout(ridge_design_matrix(states.numpy(), inputs.numpy()), rows, weights, model.time_scale_s,
                           {"version": 1, "n_nodes": 2, "rul_transform": "log1p"})
    np.testing.assert_array_equal(actual.raw_rul_s, expected.numpy())


def test_short_units_and_missing_gap_flags(monkeypatch):
    from types import SimpleNamespace

    from pdm.forecasting import trajectory_design

    monkeypatch.setattr("pdm.forecasting.apply_preprocessor", lambda prep, selected, ds: selected)
    frames = [pd.DataFrame({"unit_id": uid, "timestamp_s": np.arange(n)*60,
                           "value": 1.0, "gap_before": np.nan}) for uid, n in (("short", 10), ("long", 25))]
    features = pd.concat(frames, ignore_index=True)
    units = pd.DataFrame({"unit_id": ["short", "long"], "event_time_s": [600, 1500]})
    z, rows = trajectory_design(model_fixture(), SimpleNamespace(feature_names=["value"]), features, units, ["short", "long"])
    assert len(z) == len(rows) == 6
    assert set(rows.unit_id) == {"long"}
    with pytest.raises(ValueError, match="strictly increasing"):
        predict_failure_interval([0, np.nan], [100, 100], {"version": 1, "warmup_measurements": 1})


def test_generic_trainer_rejects_continuous_resume_before_config_or_data(monkeypatch, tmp_path):
    from pdm.io_util import dump_yaml
    from pdm.train import run_training

    run = tmp_path / "saved"
    run.mkdir()
    dump_yaml(run / "config.yaml", {"model": {"architecture": "fly_connectome_reservoir", "reservoir": {"state_mode": "continuous"}}})
    monkeypatch.setattr("pdm.train.dataset_runs", lambda dataset: tmp_path)
    monkeypatch.setattr("pdm.train.load_dataset_config", lambda *args: pytest.fail("Read config before rejecting unsafe resume"))
    monkeypatch.setattr("pdm.train.load_processed", lambda *args: pytest.fail("Loaded data before rejecting unsafe resume"))
    with pytest.raises(ValueError, match="scripts/train_brain_forecast.py"):
        run_training("bearings", resume_run_id="saved")


def test_generic_trainer_rejects_new_continuous_config_but_allows_legacy_guard(monkeypatch):
    from pdm.train import _assert_window_training_supported, run_training

    monkeypatch.setattr("pdm.train.load_dataset_config", lambda *args: {"model": {"reservoir": {"state_mode": "continuous"}}})
    monkeypatch.setattr("pdm.train.load_processed", lambda *args: pytest.fail("Loaded data before rejecting unsupported config"))
    with pytest.raises(ValueError, match="generic window trainer"):
        run_training("bearings", architecture="fly_connectome_reservoir")
    _assert_window_training_supported({"architecture": "fly_connectome_reservoir", "reservoir": {"state_mode": "window_reset"}})
