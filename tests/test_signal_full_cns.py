"""Full graph signal adapter: computation, isolation, persistence and failures."""
from __future__ import annotations

import copy

import joblib
import numpy as np
import pytest
import torch

from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json
from pdm.models.reservoir import forward_states
from pdm.models.signal_full_cns import build_signal_full_cns, source_unavailable_reason
from pdm.signal_inference import forecast_prefix
from pdm.signal_training import (
    _predict,
    _windows,
    available_signal_engines,
    load_signal_run,
    train_signal_run,
)
from tests.project_contract import make_contract_snapshot


def test_all_neuron_dynamics_match_reference_and_edges_change_states(signal_cns_fixture):
    model = signal_cns_fixture(42)
    x = np.random.default_rng(3).normal(size=(19, 5, 1)).astype(np.float32)
    actual = model.transform(x)
    expected = []
    for window in x:
        states = forward_states(torch.from_numpy(window), torch.from_numpy(model.input_weights),
                                torch.from_numpy(model.operator.toarray()), torch.from_numpy(model.bias), model.leak)
        expected.append(np.r_[model.pool @ states[-1].numpy(), window[-1, 0]])
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(actual[[12]], model.transform(x[[12]]), rtol=1e-5, atol=1e-7)
    model.operator.data[:] = 0
    assert not np.allclose(model.transform(x)[:, :-1], actual[:, :-1])
    with pytest.raises(InterruptedError, match="cancelled"):
        model.transform(x, should_stop=lambda: True)


def test_missing_source_is_unavailable_without_synthetic_substitution(tmp_path, monkeypatch):
    monkeypatch.setattr("pdm.models.signal_full_cns.default_malemcns_path", lambda: tmp_path / "missing.feather")
    assert "No synthetic substitute" in source_unavailable_reason()
    with pytest.raises(FileNotFoundError, match="official v1.0"):
        build_signal_full_cns(42)
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, snapshot = make_contract_snapshot(root)
    option = next(row for row in available_signal_engines(project["project_id"], snapshot["snapshot_id"])
                  if row["engine_id"] == "full_cns")
    assert option["available"] is False and "No synthetic substitute" in option["reason"]


def test_full_cns_run_replay_parity_test_isolation_and_provenance(tmp_path, monkeypatch, signal_cns_fixture):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, snapshot = make_contract_snapshot(root)
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    params = {"history_length": 4, "horizons_s": [10.0, 20.0]}
    run = train_signal_run(pid, sid, "full_cns", params)
    model = joblib.load(load_signal_run(pid, run["run_id"])["artifact_path"])
    data = load_snapshot(pid, sid)
    test = _windows(data["features"], data["split"]["test"], run["params"])
    expected, _, _ = _predict(model, "full_cns", test, run["scaler"])
    for i in (0, 9, len(test["x"]) - 1):
        replay = forecast_prefix(pid, run["run_id"], test["unit_id"][i], test["as_of_s"][i])
        np.testing.assert_allclose([p["value"] for p in replay["points"]], expected[i], rtol=1e-5, atol=1e-6)
    # Test values cannot change the selected readout or train-only normalization.
    changed = copy.deepcopy(data)
    changed["features"].loc[changed["features"].unit_id.isin(data["split"]["test"]), "signal"] += 200
    monkeypatch.setattr("pdm.signal_training.load_snapshot", lambda *_: changed)
    other = train_signal_run(pid, sid, "full_cns", params)
    other_model = joblib.load(store.run_path(pid, other["run_id"]) / other["artifact"])
    assert other["selection"] == run["selection"]
    assert other["scaler"] == run["scaler"]
    for a, b in zip(model.heads, other_model.heads, strict=True):
        np.testing.assert_array_equal(a.coef_, b.coef_)
    run["connectome"]["graph_hash"] = "tampered"
    atomic_write_json(store.run_path(pid, run["run_id"]) / "manifest.json", run)
    with pytest.raises(ValueError, match="connectome differs"):
        load_signal_run(pid, run["run_id"])


def test_cancelled_full_cns_run_is_not_published(tmp_path, monkeypatch, signal_cns_fixture):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, snapshot = make_contract_snapshot(root)
    with pytest.raises(InterruptedError, match="cancelled"):
        train_signal_run(project["project_id"], snapshot["snapshot_id"], "full_cns",
                         {"history_length": 4}, should_stop=lambda: True)
    assert not list((store.project_path(project["project_id"]) / "runs").glob("*/manifest.json"))
