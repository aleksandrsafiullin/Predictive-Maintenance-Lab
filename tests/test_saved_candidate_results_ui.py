"""Saved Validation summaries and snapshot-bound Results navigation."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from pdm.data.project_prepare import load_snapshot
from pdm.paths import project_root
from tests.project_contract import make_contract_snapshot


@pytest.mark.parametrize("has_test", [False, True])
def test_saved_learned_results_names_actual_metric_partition(tmp_path, monkeypatch, has_test):
    import pdm.project_results_ui as ui

    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, ref = make_contract_snapshot(root)
    pid, sid = project["project_id"], ref["snapshot_id"]
    snapshot = load_snapshot(pid, sid, store=store)
    validation = {"physical_group_count": 3, "origins": 12, "horizons": []}
    test = {"physical_group_count": 7, "origins": 28, "horizons": []}
    run = {"run_id": "saved-candidate", "project_id": pid, "snapshot_id": sid,
           "task": "signal_forecast", "status": "completed", "engine_id": "gru",
           "params": {"forecast_mode": "learned_joint_trajectories", "horizons_s": [10.]},
           "schema": snapshot["schema"], "funnel": {"mode": "learned_joint_trajectories"},
           "metrics": {"validation": validation, **({"test": test} if has_test else {})}}
    monkeypatch.setattr(ui, "list_project_runs", lambda _pid: [run])
    monkeypatch.setattr(ui, "list_red_entry_runs", lambda _pid: [])
    monkeypatch.setattr(ui, "load_signal_run", lambda *_args: run)
    monkeypatch.setattr(ui, "_play_fragment", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(ui, "cancel_replay_forecast", lambda: None)
    at = AppTest.from_string(
        "from pdm.data.project_prepare import load_snapshot\n"
        "from pdm.project_results_ui import render_results\n"
        f"render_results({pid!r}, load_snapshot({pid!r}), 'saved-candidate', 'light')\n"
    ).run()
    assert not at.exception
    partition = "Test" if has_test else "Validation"
    assert [header.value for header in at.subheader] == [f"{partition} summary"]
    equipment = next(metric for metric in at.metric if metric.label == f"{partition} equipment")
    assert equipment.value == ("7" if has_test else "3")
    assert next(metric for metric in at.metric if metric.label == "Replay origins").value == ("28" if has_test else "12")


@pytest.mark.parametrize("compatible", [True, False])
def test_results_navigation_accepts_completed_active_snapshot_without_selected_run(
    tmp_path, monkeypatch, compatible
):
    import pdm.project_ui as ui

    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, ref = make_contract_snapshot(root)
    pid = project["project_id"]
    store.update(pid, selected_run_id=None)
    row = {"run_id": "available-run", "project_id": pid, "status": "completed",
           "task": "signal_forecast", "snapshot_id": ref["snapshot_id"] if compatible else "older-snapshot"}
    monkeypatch.setattr(ui, "list_project_runs", lambda _pid: [row])
    monkeypatch.setattr(ui, "list_red_entry_runs", lambda _pid: [])
    monkeypatch.setattr(ui, "render_results", lambda *_args: None)
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_args: False)
    monkeypatch.setattr("pdm.project_quality_ui.heavy_job_active", lambda *_args: False)
    at = AppTest.from_file(str(project_root() / "src/pdm/app.py"), default_timeout=15)
    at.session_state["project_id"] = pid
    at.session_state["project_step"] = "Projects"
    at.run()
    assert not at.exception
    results = at.button(key="project_nav:Results")
    assert results.disabled is (not compatible)
    if compatible:
        results.click().run()
        assert not at.exception
        assert at.session_state["project_step"] == "Results"
def test_learned_band_status_uses_saved_uncalibrated_state(monkeypatch):
    from pdm import project_results_ui

    captions = []
    monkeypatch.setattr(project_results_ui.st, 'caption', captions.append)
    project_results_ui._show_forecast_context({
        'funnel': {'mode': 'learned_joint_trajectories'},
        'calibration_status': 'learned_uncalibrated_simultaneous_band',
    }, {})
    assert captions == ['Band: learned uncalibrated simultaneous band']

