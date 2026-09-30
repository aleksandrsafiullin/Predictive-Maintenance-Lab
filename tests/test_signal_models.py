from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.data.project_prepare import load_snapshot
from pdm.models.signal_recurrent import SignalRecurrent
from pdm.signal_inference import forecast_prefix
from pdm.signal_training import (
    _params,
    _predict,
    _windows,
    available_signal_engines,
    list_project_runs,
    load_signal_run,
    train_signal_run,
)
from tests.project_contract import make_contract_snapshot


@pytest.fixture
def contract(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, snapshot = make_contract_snapshot(root)
    return store, project, snapshot


def test_signal_heads_are_numeric_and_have_finite_gradients():
    for engine in ("gru", "lstm"):
        model = SignalRecurrent(engine, 1, 8, 2)
        result = model(torch.tensor([[[-1.0], [0.0], [1.0]]]))
        assert result.shape == (1, 2)
        result.square().sum().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_residual_recurrent_head_preserves_last_value_and_legacy_head():
    signal = torch.tensor([[[2.0], [3.0], [4.0]]])
    legacy = SignalRecurrent("gru", 1, 8, 3)
    residual = SignalRecurrent("gru", 1, 8, 3, residual_forecast=True)
    residual.load_state_dict(legacy.state_dict())
    assert torch.allclose(residual(signal), legacy(signal) + 4.0)


def test_window_targets_require_one_matching_future_timestamp():
    features = pd.DataFrame({"unit_id": ["A"] * 5,
                             "timestamp_s": [0.0, 1.0, 2.0, 2.0000001, 3.0],
                             "signal": [0.0, 1.0, 2.0, 2.1, 3.0],
                             "gap_before": [True, False, False, False, False]})
    params = {"history_length": 2, "horizons_s": [1.0], "target_tolerance_s": 0.01}
    windows = _windows(features, ["A"], params)
    assert 1.0 not in windows["as_of_s"]  # two possible targets around 2 seconds
    assert windows["as_of_s"] == [2.0, 2.0000001]
    assert windows["y"][:, 0].tolist() == [3.0, 3.0]


def test_quantile_prediction_keeps_signed_domain():
    class NegativeModel:
        def __init__(self, value):
            self.value = value

        def predict(self, x):
            return np.full(len(x), self.value)

    models = [{0.05: NegativeModel(-3.0), 0.5: NegativeModel(-2.0),
               0.95: NegativeModel(-1.0)}]
    frame = {"x": np.asarray([[[-1.0], [-2.0]]], np.float32), "y": np.zeros((1, 1))}
    point, lower, upper = _predict(models, "quantile_boosting", frame, {"mean": 0.0, "std": 1.0})
    assert point[0, 0] == -2.0 and lower[0, 0] == -3.0 and upper[0, 0] == -1.0
    bounded = _predict(models, "quantile_boosting", frame,
                       {"mean": 0.0, "std": 1.0, "output_domain": "nonnegative"})
    assert all(value[0, 0] == 0.0 for value in bounded)


@pytest.mark.parametrize("engine", ["gru", "lstm", "quantile_boosting", "full_cns"])
def test_saved_signal_run_reloads_and_prefix_ignores_future_rows(contract, monkeypatch, engine, request):
    if engine == "full_cns":
        request.getfixturevalue("signal_cns_fixture")
    store, project, snapshot = contract
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    choices = {row["engine_id"]: row for row in available_signal_engines(pid, sid)}
    assert choices[engine]["available"]
    params = {"history_length": 4, "horizons_s": [10.0, 20.0], "epochs": 2,
              "hidden_size": 8, "batch_size": 16, "max_iter": 3, "seed": 7}
    manifest = train_signal_run(pid, sid, engine, params)
    assert manifest["metrics"]["validation"]["known_targets"] > 0
    assert manifest["metrics"]["test"]["known_targets"] > 0
    assert manifest["selection"]["criterion"] == "validation_unit_equal_mae"
    if engine == "quantile_boosting":
        assert manifest["selection"]["selected_max_iter"] in {2, 3}
        assert len(manifest["selection"]["candidates"]) == 2
    assert load_signal_run(pid, manifest["run_id"])["engine_id"] == engine
    assert [r["run_id"] for r in list_project_runs(pid)] == [manifest["run_id"]]
    data = load_snapshot(pid, sid)
    uid = str(data["split"]["test"][0])
    unit = data["features"][data["features"].unit_id == uid].sort_values("timestamp_s")
    as_of = float(unit.timestamp_s.iloc[8])
    original = forecast_prefix(pid, manifest["run_id"], uid, as_of)
    extended = forecast_prefix(pid, manifest["run_id"], uid, as_of, rollout_steps=12)
    assert len(extended["points"]) > len(original["points"])
    assert original["status"] == "available"
    assert original["points"] and all(p["target_time_s"] > as_of for p in original["points"])
    assert original["calibration_status"] == ("unvalidated_pointwise_quantiles" if engine == "quantile_boosting"
                                               else "unavailable")
    assert original["red_entry_corridor"]["status"] == "unavailable"
    assert all(row["timestamp_s"] <= as_of for row in original["observed_prefix"])
    if engine == "quantile_boosting":
        assert all(p["lower"] is not None and p["upper"] is not None for p in original["points"])
    else:
        assert all(p["lower"] is None and p["upper"] is None for p in original["points"])

    # Change only hidden future rows after the saved run has been verified.
    import pdm.signal_inference as inference

    changed = copy.copy(data)
    changed["features"] = data["features"].copy()
    later = (changed["features"].unit_id == uid) & (changed["features"].timestamp_s > as_of)
    changed["features"].loc[later, "signal"] = 100_000.0
    monkeypatch.setattr(inference, "load_snapshot", lambda *_args, **_kwargs: changed)
    replay = inference.forecast_prefix(pid, manifest["run_id"], uid, as_of)
    assert replay["points"] == original["points"]
    assert replay["crossing"] == original["crossing"]
    extended_replay = inference.forecast_prefix(pid, manifest["run_id"], uid, as_of, rollout_steps=12)
    assert extended_replay == extended
    assert any(float(v) < 0 for v in data["features"].signal)


def test_replay_uses_last_observed_sample_for_between_sample_cursor(contract):
    _, project, snapshot = contract
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    run = train_signal_run(pid, sid, "gru", {"history_length": 4, "horizons_s": [10.0],
                                            "epochs": 1, "hidden_size": 8})
    unit_id = str(load_snapshot(pid, sid)["split"]["test"][0])
    forecast = forecast_prefix(pid, run["run_id"], unit_id, 135.0)
    assert forecast["as_of_s"] == 130.0
    assert all(point["target_time_s"] > forecast["as_of_s"] for point in forecast["points"])
    after_end = forecast_prefix(pid, run["run_id"], unit_id, 9999.0)
    assert after_end["as_of_s"] == max(row["timestamp_s"] for row in after_end["observed_prefix"])
    assert all(point["target_time_s"] > after_end["as_of_s"] for point in after_end["points"])


def test_masks_targets_at_gaps_and_unknown_horizons(contract):
    _, project, snapshot = contract
    data = load_snapshot(project["project_id"], snapshot["snapshot_id"])
    train = data["split"]["train"]
    features = data["features"]
    selected = features[features.unit_id.astype(str).isin(train)]
    params = _params("gru", {"history_length": 3, "horizons_s": [10.0, 20.0, 1000.0]}, selected)
    rows = _windows(features, train, params)
    assert not rows["mask"][:, 2].any()
    # Every constructed window stays inside its most recent segment.
    uid = train[0]
    broken = features[features.unit_id == uid].copy()
    broken.loc[broken.index[6], "gap_before"] = True
    isolated = _windows(broken, [uid], {**params, "horizons_s": [10.0]})
    assert len(isolated["x"]) == 10
    assert all(len(window) == 3 for window in isolated["x"])
    assert not np.any(np.isnan(isolated["x"]))


def test_results_already_red_matches_project_zone_label(contract):
    from pdm.project_zones import label_unit

    _, project, snapshot = contract
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    history = 4
    run = train_signal_run(pid, sid, "gru", {"history_length": history, "horizons_s": [10.0],
                                            "epochs": 1, "hidden_size": 8})
    data = load_snapshot(pid, sid)
    checked, red_seen = 0, 0
    for uid in map(str, data["split"]["test"]):
        unit = data["features"][data["features"].unit_id.astype(str) == uid].sort_values("timestamp_s")
        gaps = unit.gap_before.fillna(False).to_numpy(bool)
        for end in range(1, len(unit) + 1):
            last_gap = int(np.flatnonzero(gaps[:end])[-1]) if gaps[:end].any() else 0
            if end - last_gap < history:
                continue
            prefix = unit.iloc[:end]
            forecast = forecast_prefix(pid, run["run_id"], uid, float(prefix.timestamp_s.iloc[-1]))
            assert forecast["points"]
            is_red = label_unit(prefix, run["schema"])["zone"].iloc[-1] == "red"
            assert (forecast["crossing"]["status"] == "already_red") == is_red
            checked += 1
            red_seen += int(is_red)
    assert checked > 0 and red_seen > 0


def test_run_rejects_changed_artifact_and_cross_project_binding(contract, tmp_path, monkeypatch):
    store, project, snapshot = contract
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    run = train_signal_run(pid, sid, "gru", {"history_length": 4, "horizons_s": [10.0],
                                             "epochs": 1, "hidden_size": 8})
    other = store.create("Other", "generic_sensor_csv")
    with pytest.raises((FileNotFoundError, ValueError)):
        load_signal_run(other["project_id"], run["run_id"])
    artifact = store.run_path(pid, run["run_id"]) / run["artifact"]
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash"):
        load_signal_run(pid, run["run_id"])


def test_train_after_move_fits_scaler_on_new_train_units(contract, monkeypatch):
    # Protocol invariant only (epochs=1); not a model-quality claim.
    from pdm.data.project_prepare import move_units

    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_args, **_kwargs: False)
    store, project, snapshot = contract
    pid, old_sid = project["project_id"], snapshot["snapshot_id"]
    params = {"epochs": 1, "history_length": 2, "hidden_size": 4, "horizons_s": [10.0], "seed": 3}
    old_run = train_signal_run(pid, old_sid, "gru", params)
    parent_train = sorted(snapshot["split"]["train"])
    assert old_run["scaler"]["fit_units"] == parent_train

    moved = move_units(pid, [parent_train[0]], "validation", expected_snapshot_id=old_sid, store=store)
    new_split = load_snapshot(pid, moved["snapshot_id"])["split"]
    assert load_signal_run(pid, old_run["run_id"])["snapshot_id"] == old_sid
    with pytest.raises(ValueError, match="Selected run"):
        store.update(pid, selected_run_id=old_run["run_id"])

    run = train_signal_run(pid, moved["snapshot_id"], "gru", params)
    assert run["snapshot_id"] == moved["snapshot_id"]
    assert run["scaler"]["fit_units"] == sorted(new_split["train"])
    assert run["scaler"]["fit_units"] != parent_train
    assert parent_train[0] not in run["scaler"]["fit_units"]


@pytest.mark.parametrize("name", ["manifest.json", "training_contract.json", "model.pt"])
def test_run_rejects_symlinked_children(contract, tmp_path, name):
    store, project, snapshot = contract
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    run = train_signal_run(pid, sid, "gru", {"history_length": 4, "horizons_s": [10.0],
                                            "epochs": 1, "hidden_size": 8})
    path = store.run_path(pid, run["run_id"]) / name
    outside = tmp_path / name
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        load_signal_run(pid, run["run_id"])
