from __future__ import annotations

import io

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pdm.paths import project_root
from pdm.project_quality_ui import gap_safe_trace, part_summary
from pdm.project_training_ui import _parse_horizons
from pdm.project_ui import _open_step, _stage_uploads
from pdm.projects import project_store
from tests.project_contract import make_contract_snapshot


def _app() -> AppTest:
    return AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)


def test_create_two_persistent_projects_and_simple_navigation(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    at = _app().run()
    assert not at.exception
    assert [title.value for title in at.title] == ["Projects"]
    assert not any(r.label == "Report view" for r in at.radio)
    assert not any(button.label == "Compare" for button in at.button)
    name = next(widget for widget in at.text_input if widget.label == "Project name")
    name.set_value("Pump A")
    next(button for button in at.button if button.label == "Create project").click()
    at.run()
    assert not at.exception
    assert [title.value for title in at.title] == ["Import data"]
    assert project_store().list()[-1]["name"] == "Pump A"
    assert {widget.label for widget in at.file_uploader} >= {"Training folder"}
    assert {widget.label for widget in at.radio} >= {"Validation", "Testing"}
    next(button for button in at.button if button.label == "Projects").click()
    at.run()
    next(widget for widget in at.text_input if widget.label == "Project name").set_value("Pump B")
    next(button for button in at.button if button.label == "Create project").click()
    at.run()
    assert {row["name"] for row in project_store().list()} >= {"Pump A", "Pump B"}
    restarted = _app().run()
    assert not restarted.exception
    assert any(widget.label == "Project" for widget in restarted.selectbox)


def test_quality_three_sets_and_training_gate(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    store = project_store()
    project = store.create("Motor", "generic_sensor_csv")
    store.update(project["project_id"], active_snapshot_id="snapshot1", state="ready")
    features = pd.DataFrame({"unit_id": ["train1", "train1", "val1", "test1"],
                             "timestamp_s": [0.0, 1.0, 0.0, 0.0],
                             "signal": [-2.0, -1.0, 1.0, 2.0],
                             "gap_before": [False, True, False, False]})
    snapshot = {"project_id": project["project_id"], "snapshot_id": "snapshot1",
                "features": features,
                "split": {"train": ["train1"], "validation": ["val1"], "test": ["test1"]},
                "schema": {"signal_label": "Vibration", "signal_unit": "g"},
                "report": {"by_split": {"train": {"rejected_signal_rows": 1}}}}
    monkeypatch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
    monkeypatch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
    monkeypatch.setattr("pdm.project_quality_ui.available_signal_engines",
                        lambda _pid, _sid: [{"engine_id": "gru", "available": True}])
    at = _app()
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Data Quality"
    at.run()
    assert not at.exception
    assert [tab.label for tab in at.tabs] == ["Training Data", "Validation Data", "Testing Data"]
    assert any("Admitted measurements" in str(c.value) for c in at.caption)
    assert any(button.label == "Continue to Training" for button in at.button)
    assert not any("dataset_version" in str(markdown.value) for markdown in at.markdown)
    monkeypatch.setattr("pdm.project_quality_ui.available_signal_engines",
                        lambda _pid, _sid: [{"engine_id": "gru", "available": False}])
    at.run()
    assert not at.exception
    assert not any(button.label == "Continue to Training" for button in at.button)


def test_saved_snapshot_offers_one_signal_engine_and_horizon_controls(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, _ = make_contract_snapshot(root)
    at = _app()
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Training"
    at.run()
    assert not at.exception
    model = next(widget for widget in at.selectbox if widget.label == "Model")
    assert list(model.options) == ["GRU", "LSTM", "Quantile boosting"]
    assert any(widget.label == "Forecast horizons (seconds)" for widget in at.text_input)
    assert any(button.label == "Train model" for button in at.button)
    assert not any(widget.label == "Report view" for widget in at.radio)


def test_quality_gap_count_excludes_unit_start():
    frame = pd.DataFrame({"unit_id": ["A", "A", "A", "B", "B"],
                          "timestamp_s": [0, 1, 2, 0, 1],
                          "signal": [1, 2, 3, 4, 5],
                          "gap_before": [True, False, True, True, False]})
    summary = part_summary(frame, {"train": ["A", "B"]}, "train")
    assert summary["gaps"] == 1
    assert gap_safe_trace(frame[frame.unit_id == "A"])[0] == [0.0, 1.0, None, 2.0]


def test_replacement_import_locks_training_until_complete(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, _ = make_contract_snapshot(root)
    monkeypatch.setattr("pdm.project_ui.status_for_project",
                        lambda _pid: {"kind": "project_import", "status": "running", "job_id": "job1"})
    at = _app()
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Data Quality"
    at.run()
    assert not at.exception
    assert next(button for button in at.button if button.label == "Training").disabled
    assert next(button for button in at.button if button.label == "Continue to Training").disabled


def test_upload_staging_rejects_unsafe_path_and_removes_partial_copy(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    store = project_store()
    project = store.create("Upload", "generic_sensor_csv")

    class Upload(io.BytesIO):
        def __init__(self, name: str):
            super().__init__(b"unit_id,timestamp_s,signal\nA,0,1\n")
            self.name = name

    with pytest.raises(ValueError, match="unsafe"):
        _stage_uploads(store, project["project_id"], [Upload("a.csv"), Upload("../escape.csv")], "primary")
    assert not list((store.project_path(project["project_id"]) / "uploads").rglob("*.csv"))


def test_horizon_input_rejects_nonfinite_values():
    for text in ("nan, 2", "1, inf", "-1, 2", "2, 1"):
        with pytest.raises(ValueError):
            _parse_horizons(text)


def test_open_prepared_legacy_link_starts_at_quality(monkeypatch):
    monkeypatch.setattr("pdm.project_ui.processed_ready", lambda dataset: dataset == "bearings")
    link = {"storage_mode": "linked_legacy", "state": "created", "active_snapshot_id": None,
            "source_manifest": {"legacy_dataset_id": "bearings"}}
    assert _open_step(link) == "Data Quality"
    link["source_manifest"] = {"legacy_dataset_id": "filters"}
    assert _open_step(link) == "Import data"


def test_failed_worker_launch_cleans_this_attempt_uploads(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    store = project_store()
    project = store.create("Race", "generic_sensor_csv")

    class Upload(io.BytesIO):
        name = "sensor.csv"

    uploaded = Upload(b"unit_id,timestamp_s,signal\nA,0,1\n")
    monkeypatch.setattr("pdm.project_ui._source_widgets",
                        lambda *_args: ({"mode": "uploads", "files": [uploaded], "group": "primary"},
                                        None, None, "auto", "auto"))
    monkeypatch.setattr("pdm.project_ui.worker_alive", lambda: False)
    monkeypatch.setattr("pdm.project_ui.status_for_project", lambda _pid: {"status": "not_ready"})
    monkeypatch.setattr("pdm.project_ui.spawn_worker",
                        lambda _job: (_ for _ in ()).throw(RuntimeError("Another job started")))
    at = _app()
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Import data"
    at.run()
    next(button for button in at.button if button.label == "Import and check data").click()
    at.run()
    assert not at.exception
    assert not list((store.project_path(project["project_id"]) / "uploads").rglob("sensor.csv"))
