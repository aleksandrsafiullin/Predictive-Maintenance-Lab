"""Launch/study integration regression checks, not warning-quality evidence."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pdm.cli import spawn_worker
from pdm.io_util import atomic_write_json, read_json
from pdm.project_results_ui import replay_figure
from pdm.project_tasks import TASK_ENGINES
from pdm.signal_inference import forecast_prefix
from pdm.signal_training import ENGINES, _params, load_signal_run
from pdm.worker import job_path, read_status
from tests.project_contract import make_contract_snapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def matrix():
    spec = importlib.util.spec_from_file_location("funnel_matrix_qa", ROOT / "scripts/run_signal_funnel_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def frozen_job(matrix, monkeypatch):
    data = {"features": pd.DataFrame({"unit_id": ["a", "a", "b", "b"],
                                      "timestamp_s": [0., 1., 0., 1.]}),
            "split": {"train": ["a", "b"]}, "fingerprint": {"fixture": "frozen"}}
    config = {"dataset": "fixture", "project_id": "p", "snapshot_id": "s", "engine_id": "gru",
              "snapshot_fingerprint_sha256": matrix.digest(data["fingerprint"]),
              "params": {"forecast_mode": "joint_residual_paths", "horizons_s": [1., 2.]}}
    saved = {**config, "params": _params("gru", config["params"], data["features"]),
             "run_id": "run-fixture", "metrics": {}, "interval_status": "insufficient_calibration"}
    frozen = {"contract_hash": "fixture-contract", "contract": {
        "jobs": [config], "implementation_hashes": {"fixture": "before"}}}
    monkeypatch.setattr(matrix, "load_snapshot", lambda *_args: data)
    monkeypatch.setattr(matrix, "freeze", lambda *_args: frozen)
    monkeypatch.setattr(matrix, "source_hashes", lambda: {"fixture": "before"})
    monkeypatch.setattr(matrix, "load_signal_run", lambda *_args: saved)
    return config, saved


def test_runner_accepts_normalized_params_not_raw_profile(matrix, frozen_job):
    config, saved = frozen_job
    assert saved["params"] != config["params"]
    matrix.verify_run(saved, config)


@pytest.mark.parametrize("field", ["project_id", "snapshot_id", "engine_id", "snapshot_fingerprint_sha256", "params"])
def test_runner_rejects_wrong_saved_binding(matrix, frozen_job, field):
    config, saved = frozen_job
    wrong = copy.deepcopy(saved)
    wrong[field] = {} if field == "params" else "different"
    with pytest.raises(ValueError, match="differ"):
        matrix.verify_run(wrong, config)


def test_failed_job_resume_then_success_uses_latest_terminal_status(matrix, frozen_job, monkeypatch, tmp_path):
    _, saved = frozen_job
    directory = tmp_path / "study"
    directory.mkdir()
    atomic_write_json(directory / "summary.json", {"results": [{"key": "fixture:gru", "status": "failed", "error": "prior failure"}]})
    progress = {}
    launches = []
    def launch(job):
        launches.append(job)
        progress.update(job_id=job["job_id"], status="completed", run_id=saved["run_id"])
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(matrix, "spawn_worker", launch)
    monkeypatch.setattr(matrix, "read_status", lambda: progress)
    assert matrix.run(directory, resume=True, timeout_s=1) == 0
    summary = read_json(directory / "summary.json")
    assert summary["status"] == "completed"
    assert [row["status"] for row in summary["results"]] == ["failed", "completed"]
    assert matrix.run(directory, resume=True, timeout_s=1) == 0
    assert len(launches) == 1
    assert len(read_json(directory / "summary.json")["results"]) == 2


def test_runner_detects_source_change_during_worker_before_accepting_result(matrix, frozen_job, monkeypatch, tmp_path):
    _, saved = frozen_job
    directory = tmp_path / "study"
    directory.mkdir()
    progress = {}
    def launch(job):
        progress.update(job_id=job["job_id"], status="completed", run_id=saved["run_id"])
        monkeypatch.setattr(matrix, "source_hashes", lambda: {"fixture": "after"})
        return SimpleNamespace(poll=lambda: 0)
    monkeypatch.setattr(matrix, "spawn_worker", launch)
    monkeypatch.setattr(matrix, "read_status", lambda: progress)
    with pytest.raises(ValueError, match="changed while worker"):
        matrix.run(directory, False, 1)
    assert read_json(directory / "summary.json")["results"] == []


@pytest.mark.parametrize("engine", ENGINES)
def test_joint_mode_uses_actual_launch_validator_and_saved_worker_job(tmp_path, monkeypatch, engine):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("PDM_WORKER_ROOT", str(tmp_path / "worker"))
    _, project, snapshot = make_contract_snapshot(tmp_path / "projects")
    launches = []
    def launch():
        launches.append(read_json(job_path()))
        return SimpleNamespace(pid=99999999)
    monkeypatch.setattr("pdm.cli._launch_worker_process", launch)
    assert set(TASK_ENGINES["signal_forecast"]) == set(ENGINES)
    params = {"forecast_mode": "joint_residual_paths", "history_length": 2, "horizons_s": [10., 20.]}
    spawn_worker({"kind": "project_train", "task": "signal_forecast", "project_id": project["project_id"],
                  "snapshot_id": snapshot["snapshot_id"], "engine_id": engine, "params": params})
    assert launches[0]["engine_id"] == engine
    assert launches[0]["params"] == params
    assert launches[0]["task"] == "signal_forecast"


def test_real_isolated_worker_joint_gru_and_saved_forecast_reload(tmp_path, monkeypatch):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("PDM_WORKER_ROOT", str(tmp_path / "worker"))
    _, project, snapshot = make_contract_snapshot(tmp_path / "projects")
    params = {"forecast_mode": "joint_residual_paths", "history_length": 2, "horizons_s": [10., 20.],
              "epochs": 1, "hidden_size": 4, "batch_size": 16, "max_windows_per_unit": 5,
              "path_samples": 8, "cv_folds": 3, "seed": 1042}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast", "job_id": "isolated-joint-gru",
                            "project_id": project["project_id"], "snapshot_id": snapshot["snapshot_id"],
                            "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=90) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed", state
    assert state["job_id"] == "isolated-joint-gru"
    run = load_signal_run(project["project_id"], state["run_id"])
    assert run["params"]["forecast_mode"] == "joint_residual_paths"
    assert all((run["dir"] / run["funnel"][name]).is_file() for name in ["artifact", "metadata", "calibration"])
    from pdm.data.project_prepare import load_snapshot
    data = load_snapshot(project["project_id"], snapshot["snapshot_id"])
    uid = str(data["split"]["test"][0])
    time_s = float(data["features"].loc[data["features"].unit_id.astype(str) == uid, "timestamp_s"].iloc[1])
    first = forecast_prefix(project["project_id"], state["run_id"], uid, time_s)
    assert first == forecast_prefix(project["project_id"], state["run_id"], uid, time_s)
    assert first["funnel"]["calibration_status"] == "insufficient_calibration"
    paths = np.asarray(first["sampled_paths"])
    assert paths.shape == (8, 3)
    assert np.all(paths[:, 0] == first["observed_prefix"][-1]["signal"])


def joint_result():
    return {"as_of_s": 10., "observed_prefix": [{"timestamp_s": 0., "signal": 1.},
                                                {"timestamp_s": 10., "signal": 2.}],
            "points": [{"target_time_s": 20., "value": 3., "lower": 1.5, "upper": 4.},
                       {"target_time_s": 30., "value": 4., "lower": 1., "upper": 6.}],
            "funnel": {"calibration_status": "outside_calibration_origin_scope"},
            "red_entry_corridor": {"status": "empirical_conditional", "earliest_s": 10., "latest_s": 30.,
                                   "probability_within_horizon": .5}}


def test_joint_plot_anchors_filled_band_and_accepts_conditional_start_at_now():
    result = joint_result()
    fig = replay_figure(result, {"signal_label": "Vibration", "signal_unit": "g"})
    lower = next(trace for trace in fig.data if trace.name == "Forecast band")
    upper = next(trace for trace in fig.data if trace.name == "Forecast band upper")
    assert lower.fill == "tonexty" and lower.fillcolor
    assert list(lower.x) == list(upper.x) == [10., 20., 30.]
    assert lower.y[0] == upper.y[0] == 2.
    assert any(shape.x0 == 10. and shape.x1 == 30. for shape in fig.layout.shapes)
    assert "Conditional RED window" in [row.text for row in fig.layout.annotations]
    assert not any("quantile" in trace.name.lower() for trace in fig.data)
    assert not any("\u0400" <= c <= "\u04ff" for c in fig.to_json())


def test_joint_context_apptest_english_empirical_probability_and_no_quantile_claim():
    at = AppTest.from_string('''
from pdm.project_results_ui import _show_forecast_context
_show_forecast_context({"as_of_s": 10., "funnel": {"calibration_status": "outside_calibration_origin_scope"},
    "red_entry_corridor": {"status": "empirical_conditional", "earliest_s": 10., "latest_s": 30.,
                           "probability_within_horizon": .5}}, {"signal_unit": "g"})
''').run()
    assert not at.exception
    assert [row.value for row in at.metric][0] == "50.0%"
    visible = " ".join([str(row.value) for row in at.caption] + [str(row.label) for row in at.metric])
    assert "empirical" in visible and "Conditional RED window" in visible
    assert "quantile" not in visible.lower()
    assert not any("\u0400" <= c <= "\u04ff" for c in visible)


def test_default_profile_returns_supported_train_horizons_and_ignores_hidden_partitions():
    from pdm.signal_profiles import funnel_training_profile
    from pdm.signal_training import _windows

    frames = [pd.DataFrame({"unit_id": [uid] * 14, "timestamp_s": np.arange(14, dtype=float),
                            "signal": np.arange(14, dtype=float), "gap_before": [False] * 14})
              for uid in ["train-a", "train-b", "validation", "test"]]
    snapshot = {"features": pd.concat(frames, ignore_index=True),
                "split": {"train": ["train-a", "train-b"], "validation": ["validation"], "test": ["test"]}}
    profile = funnel_training_profile(snapshot, "gru")
    train = snapshot["features"][snapshot["features"].unit_id.isin(snapshot["split"]["train"])]
    config = _params("gru", profile, train)
    windows = _windows(train, snapshot["split"]["train"], config)
    assert len(windows["x"]) > 0
    assert windows["mask"].any(axis=0).all()
    assert max(profile["horizons_s"]) <= 6.
    changed = copy.deepcopy(snapshot)
    hidden = ~changed["features"].unit_id.isin(snapshot["split"]["train"])
    changed["features"].loc[hidden, "timestamp_s"] *= 100000
    changed["features"].loc[hidden, "signal"] = -1e12
    changed["features"].loc[hidden, "gap_before"] = True
    assert funnel_training_profile(changed, "gru") == profile
