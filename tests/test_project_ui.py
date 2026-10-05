from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.errors import AppTestError

from pdm import project_zones, ui_copy
from pdm.data.project_prepare import (
    JOB_ACTIVE_MOVE_ERROR,
    LINKED_LEGACY_MOVE_ERROR,
    SPLIT_LABELS,
    load_snapshot,
    load_zone_limits,
    move_units,
    preview_move,
    preview_swap,
    save_zone_limits,
    swap_units,
)
from pdm.paths import project_root
from pdm.project_quality_ui import gap_safe_trace, limits_key, part_summary
from pdm.project_training_ui import _default_profile, _parse_horizons
from pdm.project_ui import _auto_shares, _open_step, _stage_uploads
from pdm.projects import project_store
from pdm.signal_training import average_training_duration_s
from pdm.zone_limit_proposal import propose_absolute_limits
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
    assert {widget.label for widget in at.selectbox} >= {"Validation data from", "Testing data from"}
    next(button for button in at.button if button.label == "Projects").click()
    at.run()
    next(widget for widget in at.text_input if widget.label == "Project name").set_value("Pump B")
    next(button for button in at.button if button.label == "Create project").click()
    at.run()
    assert {row["name"] for row in project_store().list()} >= {"Pump A", "Pump B"}
    restarted = _app().run()
    assert not restarted.exception
    assert any(widget.label == "Project" for widget in restarted.selectbox)


def _import_app(project_id: str) -> AppTest:
    at = _app()
    at.session_state["project_id"] = project_id
    at.session_state["project_step"] = "Import data"
    return at.run()


def _captions(at: AppTest) -> list[str]:
    return [str(caption.value) for caption in at.caption]


def test_auto_shares_renormalize_over_automatic_groups():
    weights = {"train": 70, "validation": 15, "test": 15}
    assert _auto_shares(weights, "auto", "auto") == pytest.approx({"train": .7, "validation": .15, "test": .15})
    assert _auto_shares(weights, "auto", "folder") == pytest.approx({"train": 70 / 85, "validation": 15 / 85})
    assert _auto_shares(weights, "folder", "folder") == {"train": 1.0}


def test_import_page_three_cards_empty_state(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project = project_store().create("Fresh", "generic_sensor_csv")
    at = _import_app(project["project_id"])
    assert not at.exception
    assert [header.value for header in at.subheader][:3] == ["Training Data", "Validation Data", "Testing Data"]
    assert not any(metric.label == "Units" for metric in at.metric)
    assert sum("No data yet" in str(md.value) for md in at.markdown) == 3
    assert any(widget.label == "Training folder" for widget in at.file_uploader)
    holdouts = [widget for widget in at.selectbox if widget.label in {"Validation data from", "Testing data from"}]
    assert [widget.value for widget in holdouts] == ["Split from training", "Split from training"]
    assert _captions(at).count("Automatically split from training data") == 2
    assert not any(button.label == "View" for button in at.button)


def test_import_cards_show_snapshot_counts_and_view(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, _ = make_contract_snapshot(root)
    snapshot = load_snapshot(project["project_id"], store=store)
    at = _import_app(project["project_id"])
    assert not at.exception
    expected = [part_summary(snapshot["features"], snapshot["split"], part) for part in ("train", "validation", "test")]
    for label, field in (("Units", "units"), ("Admitted rows", "rows"), ("Gaps", "gaps")):
        assert [int(m.value) for m in at.metric if m.label == label] == [summary[field] for summary in expected]
    assert "Counts are the saved snapshot." not in _captions(at)
    at.button(key="import_view:test").click()
    at.run()
    assert not at.exception
    assert at.session_state["project_step"] == "Data Quality"
    assert at.session_state["quality_tab"] == "Testing Data"


def test_import_separate_folder_card_and_renormalized_share(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project = project_store().create("Folders", "generic_sensor_csv")
    at = _import_app(project["project_id"])
    next(widget for widget in at.selectbox if widget.label == "Testing data from").set_value("Separate folder")
    at.run()
    assert not at.exception
    assert any(widget.label == "Testing folder" for widget in at.file_uploader)
    assert "18% of the training pool" in _captions(at)


def test_import_split_settings_keys_drive_card_shares(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project = project_store().create("Weights", "generic_sensor_csv")
    at = _import_app(project["project_id"])
    at.number_input(key="import_weight_validation").set_value(20)
    at.number_input(key="import_weight_train").set_value(65)
    at.run()
    assert not at.exception
    assert "20% of the training pool" in _captions(at)
    assert "65 / 20 / 15 · seed 42" in _captions(at)


def test_split_settings_survive_both_separate_folders(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project = project_store().create("Keep weights", "generic_sensor_csv")
    at = _import_app(project["project_id"])
    at.number_input(key="import_weight_train").set_value(65)
    at.number_input(key="import_weight_validation").set_value(20)
    at.run()
    next(widget for widget in at.selectbox if widget.label == "Validation data from").set_value("Separate folder")
    next(widget for widget in at.selectbox if widget.label == "Testing data from").set_value("Separate folder")
    at.run()
    assert not at.exception
    assert "Both holdouts use separate folders; split settings do not apply." in _captions(at)
    assert at.number_input(key="import_weight_train").disabled
    next(widget for widget in at.selectbox if widget.label == "Testing data from").set_value("Split from training")
    at.run()
    assert not at.exception
    assert "65 / 20 / 15 · seed 42" in _captions(at)


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
    assert any(widget.label.startswith("Inspect ") for widget in at.selectbox)
    assert at.table
    captions = _captions(at)
    assert not any("Admitted measurements for the selected unit" in c for c in captions)
    assert not any(c.startswith("Zones:") or c.startswith("Observed time:") for c in captions)
    assert not any("forecast future" in c or "Recorded experiment endpoints" in c for c in captions)
    assert not any(button.label in {"Move selected units", "Move units"} for button in at.button)
    assert any(button.label == "Continue to Training" for button in at.button)
    ready_copy = (
        "Train, Validation, and Test have admitted measurements. Training uses Train units; "
        "Validation selects the model; Test is held out until evaluation."
    )
    assert ready_copy not in [str(item.value) for item in at.success]
    assert not any("dataset_version" in str(markdown.value) for markdown in at.markdown)
    monkeypatch.setattr("pdm.project_quality_ui.available_signal_engines",
                        lambda _pid, _sid: [{"engine_id": "gru", "available": False}])
    at.run()
    assert not at.exception
    assert not any(button.label == "Continue to Training" for button in at.button)
    assert any("too short" in str(item.value) for item in at.warning)


def _quality_app(project_id: str, tab: str | None = None) -> AppTest:
    at = _app()
    at.session_state["project_id"] = project_id
    at.session_state["project_step"] = "Data Quality"
    if tab:
        at.session_state["quality_tab"] = tab
    return at.run()


def _chart_spec(at: AppTest) -> dict:
    return json.loads(at.get("plotly_chart")[0].proto.spec)


def _chart_trace_names(at: AppTest) -> list[str]:
    return [trace.get("name") for trace in _chart_spec(at)["data"]]


def _chart_limit_ys(at: AppTest) -> set[float]:
    shapes = (_chart_spec(at).get("layout") or {}).get("shapes") or []
    return {round(float(shape["y0"]), 6) for shape in shapes
            if shape.get("y0") is not None and shape.get("y0") == shape.get("y1")}


def _zone_column(at: AppTest) -> list:
    signal_column = _chart_spec(at)["layout"]["yaxis"]["title"]["text"]
    expected_columns = ["Time (s)", signal_column, "Record position", "Zone"]
    sample_tables = [table.value for table in at.table
                     if list(table.value.columns) == expected_columns]
    assert len(sample_tables) == 1, f"Expected one sample table with columns {expected_columns}"
    return list(sample_tables[0]["Zone"])


_ZONE_LABELS = {"green": "Green", "yellow": "Yellow", "red": "Red", "unknown": "Not zoned"}


def _expected_zones(features: pd.DataFrame, unit_id: str, schema: dict) -> list[str]:
    frame = features.loc[features["unit_id"].astype(str) == str(unit_id)]
    return [_ZONE_LABELS[zone] for zone in project_zones.label_unit(frame, schema)["zone"]]


def test_quality_chart_paints_zones_from_snapshot_limits(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, _ = make_contract_snapshot(root)
    snapshot = load_snapshot(project["project_id"], store=store)
    at = _quality_app(project["project_id"])
    assert not at.exception
    assert len(at.get("plotly_chart")) == 1
    selected = sorted(str(uid) for uid in snapshot["split"]["train"])[0]
    features = snapshot["features"]
    labelled = project_zones.label_unit(features[features["unit_id"].astype(str) == selected], snapshot["schema"])
    names = {"green": "Green", "yellow": "Yellow", "red": "Red", "unknown": "Not zoned"}
    expected = [f"{names[zone]} · {count}" for zone in project_zones.ZONES
                if (count := int((labelled["zone"] == zone).sum()))]
    assert [name for name in _chart_trace_names(at) if " · " in str(name)] == expected
    assert _chart_limit_ys(at) == {0.4, 0.8}
    assert _zone_column(at) == [names[zone] for zone in labelled["zone"]]


def _minimal_quality_snapshot(project_id: str, schema: dict) -> dict:
    features = pd.DataFrame({"unit_id": ["train1"] * 6 + ["val1", "test1"],
                             "timestamp_s": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                             "signal": [1.0, 1.0, 1.0, 1.0, 1.4, 3.0, 1.0, 2.0],
                             "gap_before": [False, False, True, False, False, False, False, False]})
    return {"project_id": project_id, "snapshot_id": "snapshot1", "features": features,
            "split": {"train": ["train1"], "validation": ["val1"], "test": ["test1"]},
            "schema": schema, "report": {}}


def _patch_quality_snapshot(monkeypatch, schema: dict) -> str:
    store = project_store()
    project = store.create("Zones", "generic_sensor_csv")
    store.update(project["project_id"], active_snapshot_id="snapshot1", state="ready")
    snapshot = _minimal_quality_snapshot(project["project_id"], schema)
    monkeypatch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
    monkeypatch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
    monkeypatch.setattr("pdm.project_quality_ui.available_signal_engines",
                        lambda _pid, _sid: [{"engine_id": "gru", "available": True}])
    return project["project_id"]


def test_quality_chart_without_limits_is_not_zoned(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project_id = _patch_quality_snapshot(monkeypatch, {"signal_label": "Vibration", "signal_unit": "g"})
    at = _quality_app(project_id)
    assert not at.exception
    assert _chart_trace_names(at) == ["Vibration"]
    assert set(_zone_column(at)) == {"Not zoned"}
    assert not _chart_limit_ys(at)


def test_quality_baseline_rule_caption_uses_schema_values(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    schema = {"signal_label": "Vibration", "signal_unit": "g",
              "thresholds": {"mode": "initial_baseline_multiple", "direction": "above", "baseline_n": 4,
                             "red_ratio": 2.5, "note": "Provisional"}}
    project_id = _patch_quality_snapshot(monkeypatch, schema)
    at = _quality_app(project_id)
    assert not at.exception
    assert _zone_column(at) == ["Not zoned", "Not zoned", "Not zoned", "Green", "Yellow", "Red"]
    assert _chart_limit_ys(at) == {1.25, 2.5}
    assert [name for name in _chart_trace_names(at) if " · " in str(name)] == [
        "Green · 1", "Yellow · 1", "Red · 1", "Not zoned · 3"]


def test_import_page_has_no_limit_widgets(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project = project_store().create("Bearings", "xjtu_bearings")
    at = _import_app(project["project_id"])
    assert not at.exception
    assert not any(r.label == "Red condition" for r in at.radio)
    assert not any(n.label.startswith(("Yellow limit", "Red limit")) for n in at.number_input)
    assert "Yellow and red limits are set on Data Quality after import." not in _captions(at)
    assert not any("max-axis RMS" in c for c in _captions(at))
    assert not any(header.value == "Signal" for header in at.subheader)


def _capture_import(monkeypatch, project_id: str, signal_column: str | None = None) -> dict:
    jobs = []
    monkeypatch.setattr("pdm.project_ui._import_cards",
                        lambda *_args: ({"mode": "folder", "path": "/tmp/source"}, None, None, "auto", "auto"))
    monkeypatch.setattr("pdm.project_ui.worker_alive", lambda: False)
    monkeypatch.setattr("pdm.project_ui.status_for_project", lambda _pid: {"status": "not_ready"})
    monkeypatch.setattr("pdm.project_ui.spawn_worker", jobs.append)
    at = _import_app(project_id)
    if signal_column is not None:
        next(w for w in at.text_input if w.label == "Signal column").set_value(signal_column)
        at.run()
    next(button for button in at.button if button.label == "Import and check data").click()
    at.run()
    assert not at.exception
    assert len(jobs) == 1
    return jobs[0]["source"]


def test_first_hse_import_sends_provisional_limits(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project = project_store().create("First", "hse_filters")
    source = _capture_import(monkeypatch, project["project_id"])
    assert (source["signal_column"], source["signal_label"], source["signal_unit"]) == (
        "differential_pressure", "Differential pressure", "Pa")
    assert source["thresholds"] == {"mode": "absolute", "direction": "above", "yellow": 300.0, "red": 600.0}


def test_first_generic_import_sends_no_limits(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project = project_store().create("First", "generic_sensor_csv")
    assert _capture_import(monkeypatch, project["project_id"])["thresholds"] == {}


def test_first_xjtu_import_sends_baseline_rule(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    project = project_store().create("Bearings", "xjtu_bearings")
    source = _capture_import(monkeypatch, project["project_id"])
    assert (source["signal_column"], source["signal_label"], source["signal_unit"]) == (
        "combined_rms", "Combined max-axis RMS", "g")
    rule = source["thresholds"]
    assert rule["mode"] == "initial_baseline_multiple"
    assert (rule["direction"], rule["baseline_n"], rule["onset_sigma"], rule["onset_ratio"], rule["red_ratio"]) == (
        "above", 5, 3.0, 1.25, 2.0)


def test_reimport_reuses_saved_baseline_rule(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    store = project_store()
    project = store.create("Bearings", "xjtu_bearings")
    pid = project["project_id"]
    store.update(pid, active_snapshot_id="snapshot1", state="ready")
    saved = {"source_kind": "xjtu_bearings", "signal_column": "combined_rms", "signal_unit": "g",
             "thresholds": {"mode": "initial_baseline_multiple", "direction": "above", "baseline_n": 4,
                            "onset_sigma": 2.0, "onset_ratio": 1.5, "red_ratio": 3.0}}
    directory = store.snapshot_path(pid, "snapshot1")
    directory.mkdir(parents=True)
    (directory / "feature_schema.json").write_text(json.dumps(saved))
    rule = _capture_import(monkeypatch, pid)["thresholds"]
    assert rule["mode"] == "initial_baseline_multiple"
    assert (rule["baseline_n"], rule["onset_sigma"], rule["onset_ratio"], rule["red_ratio"]) == (4, 2.0, 1.5, 3.0)


def test_reimport_reuses_replaced_snapshot_limits_not_other_snapshot_edits(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    jobs = []
    monkeypatch.setattr("pdm.project_ui._import_cards",
                        lambda *_args: ({"mode": "folder", "path": "/tmp/source"}, None, None, "auto", "auto"))
    monkeypatch.setattr("pdm.project_ui.worker_alive", lambda: False)
    monkeypatch.setattr("pdm.project_ui.status_for_project", lambda _pid: {"status": "not_ready"})
    monkeypatch.setattr("pdm.project_ui.spawn_worker", jobs.append)
    at = _app()
    at.session_state["project_id"] = pid
    at.session_state["project_step"] = "Import data"
    at.session_state[limits_key(pid, "older-snapshot")] = {"mode": "absolute", "direction": "below",
                                                           "yellow": -5.0, "red": -9.0}
    at.run()
    next(button for button in at.button if button.label == "Import and check data").click()
    at.run()
    assert not at.exception
    source = jobs[0]["source"]
    assert source["thresholds"] == {"mode": "absolute", "direction": "above", "yellow": 0.4, "red": 0.8}
    assert (source["signal_column"], source["signal_label"], source["signal_unit"]) == (
        "vibration", "Signed vibration", "g")


def test_reimport_honors_replaced_snapshot_sidecar(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, ref = make_contract_snapshot(root)
    rule = {"mode": "absolute", "direction": "above", "yellow": 0.2, "red": 0.5}
    save_zone_limits(project["project_id"], rule, expected_snapshot_id=ref["snapshot_id"], store=store)
    assert _capture_import(monkeypatch, project["project_id"])["thresholds"] == rule


def test_generic_reimport_with_new_signal_column_drops_old_limits(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, _ = make_contract_snapshot(root)
    source = _capture_import(monkeypatch, project["project_id"], signal_column="pressure")
    assert source["signal_column"] == "pressure"
    assert source["thresholds"] == {}


def _key(name: str, pid: str, sid: str) -> str:
    return f"quality_limit_{name}:{pid}:{sid}"


def _limit_button(name: str, pid: str, sid: str) -> str:
    return f"quality_limit_{name}:{pid}:{sid}"


def _snapshot_bytes(store, pid: str, sid: str) -> tuple[bytes, bytes]:
    directory = store.snapshot_path(pid, sid)
    return (directory / "feature_schema.json").read_bytes(), (directory / "processed_fingerprint.json").read_bytes()


def test_quality_limit_edit_writes_sidecar_and_recolors_chart(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    before = project_store().get(pid)
    sid = before["active_snapshot_id"]
    original = _snapshot_bytes(store, pid, sid)
    snapshot = load_snapshot(pid, store=store)
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.4)
    at.number_input(key=_key("yellow", pid, sid)).set_value(0.1)
    at.run()
    assert not at.exception
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.1)
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(0.8)
    assert at.radio(key=_key("direction", pid, sid)).value == "above"
    assert _chart_limit_ys(at) == {0.1, 0.8}
    limits = {"mode": "absolute", "direction": "above", "yellow": 0.1, "red": 0.8}
    edited = {**snapshot["schema"], "thresholds": limits}
    selected = sorted(str(uid) for uid in snapshot["split"]["train"])[0]
    features = snapshot["features"]
    assert _zone_column(at) == _expected_zones(features, selected, edited)
    assert load_zone_limits(pid, sid, store=store) is None
    assert not at.button(key=_limit_button("save", pid, sid)).disabled
    at.button(key=_limit_button("save", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert project_store().get(pid) == before
    assert _snapshot_bytes(store, pid, sid) == original
    assert load_zone_limits(pid, sid, store=store) == limits
    assert load_snapshot(pid, store=store)["schema"]["thresholds"]["yellow"] == 0.4
    assert not any("Preview only" in c or "Saved with this data" in c for c in _captions(at))
    assert limits_key(pid, sid) not in at.session_state
    assert at.button(key=_limit_button("save", pid, sid)).disabled
    assert at.button(key=_limit_button("cancel", pid, sid)).disabled
    assert _chart_limit_ys(at) == {0.1, 0.8}
    assert _zone_column(at) == _expected_zones(features, selected, edited)
    reopened = _quality_app(pid, "Training Data")
    assert reopened.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.1)
    assert reopened.number_input(key=_key("red", pid, sid)).value == pytest.approx(0.8)
    assert _chart_limit_ys(reopened) == {0.1, 0.8}
    assert _zone_column(reopened) == _expected_zones(features, selected, edited)


def test_quality_limit_cancel_reverts_preview_without_writing(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    original = _snapshot_bytes(store, pid, sid)
    snapshot = load_snapshot(pid, store=store)
    selected = sorted(str(uid) for uid in snapshot["split"]["train"])[0]
    at = _quality_app(pid, "Training Data")
    before_zones = _zone_column(at)
    before_lines = _chart_limit_ys(at)
    assert before_lines == {0.4, 0.8}
    assert at.button(key=_limit_button("cancel", pid, sid)).disabled
    assert at.button(key=_limit_button("save", pid, sid)).disabled
    at.number_input(key=_key("red", pid, sid)).set_value(1.5)
    at.radio(key=_key("direction", pid, sid)).set_value("above")
    at.run()
    preview = {**snapshot["schema"], "thresholds": {"mode": "absolute", "direction": "above",
                                                    "yellow": 0.4, "red": 1.5}}
    assert _chart_limit_ys(at) == {0.4, 1.5}
    assert _zone_column(at) == _expected_zones(snapshot["features"], selected, preview)
    assert load_zone_limits(pid, sid, store=store) is None
    assert not at.button(key=_limit_button("cancel", pid, sid)).disabled
    at.button(key=_limit_button("cancel", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert _zone_column(at) == before_zones
    assert _chart_limit_ys(at) == before_lines
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(0.8)
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.4)
    assert at.radio(key=_key("direction", pid, sid)).value == "above"
    assert at.button(key=_limit_button("cancel", pid, sid)).disabled
    assert at.button(key=_limit_button("save", pid, sid)).disabled
    assert load_zone_limits(pid, sid, store=store) is None
    assert _snapshot_bytes(store, pid, sid) == original


def test_quality_limit_cancel_restores_sidecar_not_schema(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    sidecar = {"mode": "absolute", "direction": "above", "yellow": 0.2, "red": 0.6}
    save_zone_limits(pid, sidecar, expected_snapshot_id=sid, store=store)
    at = _quality_app(pid, "Training Data")
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.2)
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(0.6)
    assert _chart_limit_ys(at) == {0.2, 0.6}
    before_zones = _zone_column(at)
    at.number_input(key=_key("yellow", pid, sid)).set_value(0.1)
    at.number_input(key=_key("red", pid, sid)).set_value(1.5)
    at.run()
    assert _chart_limit_ys(at) == {0.1, 1.5}
    at.button(key=_limit_button("cancel", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert _chart_limit_ys(at) == {0.2, 0.6}
    assert _zone_column(at) == before_zones
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.2)
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(0.6)
    assert at.button(key=_limit_button("cancel", pid, sid)).disabled
    assert load_zone_limits(pid, sid, store=store) == sidecar


def test_quality_linked_legacy_save_respects_job_and_writes_sidecar_only(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch, active=True)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    registry = json.loads(store.registry_path.read_text())
    registry["projects"][pid]["storage_mode"] = "linked_legacy"
    store.registry_path.write_text(json.dumps(registry))
    original = _snapshot_bytes(store, pid, sid)
    sidecar_path = store.snapshot_path(pid, sid) / "zone_limits.json"
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    assert "A background job is running. Save is available after it finishes." in _captions(at)
    at.number_input(key=_key("red", pid, sid)).set_value(1.5)
    at.run()
    assert _chart_limit_ys(at) == {0.4, 1.5}
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(1.5)
    assert at.button(key=_limit_button("save", pid, sid)).disabled
    assert not sidecar_path.exists()
    _no_job(monkeypatch)
    at.run()
    assert not at.button(key=_limit_button("save", pid, sid)).disabled
    at.button(key=_limit_button("save", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert load_zone_limits(pid, sid, store=store) == {"mode": "absolute", "direction": "above",
                                                       "yellow": 0.4, "red": 1.5}
    assert _snapshot_bytes(store, pid, sid) == original
    assert project_store().get(pid)["storage_mode"] == "linked_legacy"


def test_quality_limit_edit_back_to_committed_rule_is_clean(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    _, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    at = _quality_app(pid, "Training Data")
    at.number_input(key=_key("red", pid, sid)).set_value(1.5)
    at.run()
    at.number_input(key=_key("red", pid, sid)).set_value(0.8)
    at.run()
    assert not at.exception
    assert limits_key(pid, sid) not in at.session_state
    assert at.button(key=_limit_button("cancel", pid, sid)).disabled
    assert at.button(key=_limit_button("save", pid, sid)).disabled


def test_quality_invalid_limits_cancel_reverts_numbers(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    at = _quality_app(pid, "Training Data")
    at.number_input(key=_key("yellow", pid, sid)).set_value(0.9)
    at.run()
    assert at.error
    assert at.button(key=_limit_button("save", pid, sid)).disabled
    assert not at.button(key=_limit_button("cancel", pid, sid)).disabled
    at.button(key=_limit_button("cancel", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.4)
    assert load_zone_limits(pid, sid, store=store) is None


def test_quality_invalid_limits_keep_last_valid_rule_and_do_not_save(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    at = _quality_app(pid, "Training Data")
    at.number_input(key=_key("yellow", pid, sid)).set_value(0.9)
    at.run()
    assert not at.exception
    assert any("yellow limit must be below red" in str(e.value) for e in at.error)
    assert _chart_limit_ys(at) == {0.4, 0.8}
    assert at.radio(key=_key("direction", pid, sid)).value == "above"
    assert load_zone_limits(pid, sid, store=store) is None
    at.number_input(key=_key("yellow", pid, sid)).set_value(0.4)
    at.radio(key=_key("direction", pid, sid)).set_value("below")
    at.run()
    assert not at.exception
    assert any("yellow limit must be above red" in str(e.value) for e in at.error)
    assert _chart_limit_ys(at) == {0.4, 0.8}
    assert at.radio(key=_key("direction", pid, sid)).value == "below"
    assert load_zone_limits(pid, sid, store=store) is None
    at.number_input(key=_key("yellow", pid, sid)).set_value(0.9)
    at.run()
    assert not at.exception and not at.error
    assert at.radio(key=_key("direction", pid, sid)).value == "below"
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.9)
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(0.8)
    assert _chart_limit_ys(at) == {0.9, 0.8}
    snapshot = load_snapshot(pid, store=store)
    selected = sorted(str(uid) for uid in snapshot["split"]["train"])[0]
    below = {**snapshot["schema"], "thresholds": {"mode": "absolute", "direction": "below",
                                                  "yellow": 0.9, "red": 0.8}}
    assert _zone_column(at) == _expected_zones(snapshot["features"], selected, below)
    assert load_zone_limits(pid, sid, store=store) is None


def test_quality_limits_are_preview_while_job_runs(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch, active=True)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    at = _quality_app(pid, "Training Data")
    job_caption = "A background job is running. Save is available after it finishes."
    assert job_caption in _captions(at)
    at.number_input(key=_key("red", pid, sid)).set_value(1.5)
    at.run()
    assert not at.exception
    assert _chart_limit_ys(at) == {0.4, 1.5}
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(1.5)
    assert job_caption in _captions(at)
    assert not any("Preview only" in c or "Saved with this data" in c for c in _captions(at))
    assert load_zone_limits(pid, sid, store=store) is None
    save = at.button(key=_limit_button("save", pid, sid))
    assert save.disabled
    with pytest.raises(AppTestError):
        save.click()
    assert load_zone_limits(pid, sid, store=store) is None


def test_quality_save_failure_keeps_preview_and_shows_error(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    original = _snapshot_bytes(store, pid, sid)
    at = _quality_app(pid, "Training Data")
    at.number_input(key=_key("red", pid, sid)).set_value(1.5)
    at.run()
    _no_job(monkeypatch, active=True)
    at.button(key=_limit_button("save", pid, sid)).click()
    at.run()
    assert not at.exception
    assert any("Press Save after the job finishes." in str(e.value) for e in at.error)
    assert not any("Change a limit again" in str(e.value) for e in at.error)
    assert load_zone_limits(pid, sid, store=store) is None
    assert _snapshot_bytes(store, pid, sid) == original
    assert _chart_limit_ys(at) == {0.4, 1.5}
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(1.5)
    assert at.session_state[limits_key(pid, sid)]["red"] == 1.5
    assert at.button(key=_limit_button("save", pid, sid)).disabled


def test_quality_edit_does_not_follow_to_a_new_snapshot(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch, active=True)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    old_sid = project_store().get(pid)["active_snapshot_id"]
    at = _quality_app(pid, "Training Data")
    at.number_input(key=_key("red", pid, old_sid)).set_value(1.5)
    at.run()
    assert _chart_limit_ys(at) == {0.4, 1.5}
    _no_job(monkeypatch)
    unit = sorted(load_snapshot(pid, store=store)["split"]["train"])[-1]
    new_sid = move_units(pid, [unit], "validation", expected_snapshot_id=old_sid, store=store)["snapshot_id"]
    at.run()
    assert not at.exception
    assert _chart_limit_ys(at) == {0.4, 0.8}
    assert at.number_input(key=_key("yellow", pid, new_sid)).value == pytest.approx(0.4)
    assert at.number_input(key=_key("red", pid, new_sid)).value == pytest.approx(0.8)
    assert not any(c.startswith(("Preview only", "Saved with this data")) for c in _captions(at))


def test_quality_save_with_bound_run_keeps_run_and_results_use_sidecar(monkeypatch, tmp_path):
    from pdm.project_results_ui import replay_figure
    from pdm.signal_inference import forecast_prefix
    from pdm.signal_training import load_signal_run, train_signal_run

    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, ref = make_contract_snapshot(root)
    pid, sid = project["project_id"], ref["snapshot_id"]
    run = train_signal_run(pid, sid, "gru", {"history_length": 4, "horizons_s": [10.0], "epochs": 1,
                                             "hidden_size": 8})
    manifest_before = (store.run_path(pid, run["run_id"]) / "manifest.json").read_bytes()
    record = project_store().get(pid)
    at = _quality_app(pid, "Training Data")
    at.number_input(key=_key("red", pid, sid)).set_value(1.5)
    at.run()
    at.button(key=_limit_button("save", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert not any("Preview only" in c for c in _captions(at))
    assert project_store().get(pid) == record and record["selected_run_id"] == run["run_id"]
    assert (store.run_path(pid, run["run_id"]) / "manifest.json").read_bytes() == manifest_before
    loaded = load_signal_run(pid, run["run_id"])
    assert loaded["schema"]["thresholds"]["red"] == 0.8
    limits = load_zone_limits(pid, sid, store=store)
    unit = str(load_snapshot(pid, store=store)["split"]["test"][0])
    plain = forecast_prefix(pid, run["run_id"], unit, 130.0)
    shown = forecast_prefix(pid, run["run_id"], unit, 130.0, thresholds=limits)
    assert shown["points"] == plain["points"]
    assert (plain["thresholds"]["red"], shown["thresholds"]["red"]) == (0.8, 1.5)
    passed = []
    monkeypatch.setattr("pdm.project_results_ui._play_fragment",
                        lambda *args, **_kwargs: passed.append(args[-1]))
    results = _app()
    results.session_state["project_id"] = pid
    results.session_state["project_step"] = "Results"
    results.run()
    assert not results.exception
    assert passed == [limits]
    figure = replay_figure(shown, load_snapshot(pid, store=store)["schema"])
    lines = {round(float(shape["y0"]), 6) for shape in figure.to_dict()["layout"].get("shapes", [])
             if shape.get("y0") == shape.get("y1")}
    assert 1.5 in lines and 0.8 not in lines


def test_quality_baseline_snapshot_open_does_not_write(monkeypatch, tmp_path):
    from pdm.io_util import sha256_file

    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, ref = make_contract_snapshot(root)
    pid, sid = project["project_id"], ref["snapshot_id"]
    directory = store.snapshot_path(pid, sid)
    schema = json.loads((directory / "feature_schema.json").read_text())
    schema["thresholds"] = {"mode": "initial_baseline_multiple", "direction": "above", "baseline_n": 4}
    (directory / "feature_schema.json").write_text(json.dumps(schema))
    fingerprint = json.loads((directory / "processed_fingerprint.json").read_text())
    fingerprint["file_hashes"]["feature_schema.json"] = sha256_file(directory / "feature_schema.json")
    (directory / "processed_fingerprint.json").write_text(json.dumps(fingerprint))
    original = _snapshot_bytes(store, pid, sid)
    snapshot = load_snapshot(pid, store=store)
    selected = sorted(str(uid) for uid in snapshot["split"]["train"])[0]
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(1.0)
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(2.0)
    assert _zone_column(at) == _expected_zones(snapshot["features"], selected, snapshot["schema"])
    prefix = snapshot["features"].loc[snapshot["features"]["unit_id"].astype(str) == selected].iloc[:4]
    thresholds = project_zones.resolve_thresholds(snapshot["schema"], prefix)
    assert _chart_limit_ys(at) == {round(float(thresholds["yellow"]), 6), round(float(thresholds["red"]), 6)}
    assert _chart_limit_ys(at) != {1.0, 2.0}
    assert at.button(key=_limit_button("cancel", pid, sid)).disabled
    assert at.button(key=_limit_button("save", pid, sid)).disabled
    at.session_state["quality_tab"] = "Validation Data"
    at.run()
    assert not at.exception
    assert not (directory / "zone_limits.json").exists()
    assert _snapshot_bytes(store, pid, sid) == original


def test_quality_baseline_rule_switches_to_absolute_only_after_edit(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    schema = {"signal_label": "Vibration", "signal_unit": "g",
              "thresholds": {"mode": "initial_baseline_multiple", "direction": "above", "baseline_n": 4}}
    project_id = _patch_quality_snapshot(monkeypatch, schema)
    at = _quality_app(project_id)
    assert not at.exception
    baseline_zones = ["Not zoned", "Not zoned", "Not zoned", "Green", "Yellow", "Red"]
    assert _zone_column(at) == baseline_zones
    assert _chart_limit_ys(at) == {1.25, 2.0}
    assert at.number_input(key=_key("yellow", project_id, "snapshot1")).value == 1.0
    at.number_input(key=_key("red", project_id, "snapshot1")).set_value(1.2)
    at.run()
    assert not at.exception
    assert _chart_limit_ys(at) == {1.0, 1.2}
    assert _zone_column(at) == ["Yellow"] * 4 + ["Red", "Red"]
    at.number_input(key=_key("red", project_id, "snapshot1")).set_value(2.0)
    at.run()
    assert _chart_limit_ys(at) == {1.0, 2.0}
    assert _zone_column(at) == ["Yellow"] * 5 + ["Red"]
    assert not at.button(key=_limit_button("cancel", project_id, "snapshot1")).disabled
    at.button(key=_limit_button("cancel", project_id, "snapshot1")).click()
    at.run()
    assert not at.exception
    assert _zone_column(at) == baseline_zones
    assert _chart_limit_ys(at) == {1.25, 2.0}
    assert at.number_input(key=_key("red", project_id, "snapshot1")).value == 2.0
    assert at.button(key=_limit_button("cancel", project_id, "snapshot1")).disabled


def test_quality_invalid_sidecar_is_ignored(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, ref = make_contract_snapshot(root)
    pid, sid = project["project_id"], ref["snapshot_id"]
    sidecar = store.snapshot_path(pid, sid) / "zone_limits.json"
    sidecar.write_text(json.dumps({"mode": "absolute", "direction": "above", "yellow": 3, "red": 1}))
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.4)
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(0.8)
    assert _chart_limit_ys(at) == {0.4, 0.8}
    assert not sidecar.exists()


def test_quality_zone_labels_only_for_open_tab(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, _ = make_contract_snapshot(root)
    snapshot = load_snapshot(project["project_id"], store=store)
    calls = []
    real = project_zones.label_unit

    def spy(frame, schema):
        calls.append(sorted(set(frame["unit_id"].astype(str))))
        return real(frame, schema)

    monkeypatch.setattr("pdm.project_zones.label_unit", spy)
    at = _quality_app(project["project_id"], "Training Data")
    assert not at.exception
    selected = sorted(str(uid) for uid in snapshot["split"]["train"])[0]
    assert calls == [[selected]]
    assert len(at.get("plotly_chart")) == 1


def _no_job(monkeypatch, active: bool = False) -> None:
    monkeypatch.setattr("pdm.project_quality_ui.heavy_job_active", lambda *_a: active)
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_a: active)


def test_training_page_mentions_runs_from_previous_snapshot(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, _ = make_contract_snapshot(root)
    monkeypatch.setattr("pdm.project_training_ui.list_project_runs",
                        lambda _pid: [{"run_id": "old", "snapshot_id": "previous-snapshot"}])
    at = _app()
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Training"
    at.session_state[f"training_task:{project['project_id']}"] = "signal_forecast"
    at.run()
    assert not at.exception
    assert ("1 earlier model run(s) were trained on a previous data snapshot and are not shown. "
            "Train again on this data.") in _captions(at)


def test_saved_snapshot_offers_one_signal_engine_and_horizon_controls(monkeypatch, tmp_path):
    monkeypatch.setattr("pdm.signal_training.source_unavailable_reason", lambda: None)
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, _ = make_contract_snapshot(root)
    at = _app()
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Training"
    at.session_state[f"training_task:{project['project_id']}"] = "signal_forecast"
    at.run()
    assert not at.exception
    model = next(widget for widget in at.selectbox if widget.label == "Model")
    assert list(model.options) == ["GRU", "LSTM", "Quantile boosting", "Fly brain · Full MaleCNS"]
    assert any(widget.label == "Forecast span (minutes)" for widget in at.number_input)
    assert any(button.label == "Train model" for button in at.button)
    assert not any(widget.label == "Report view" for widget in at.radio)
    model.select("full_cns").run()
    assert not at.exception
    assert any("MaleCNS" in block.value for block in at.markdown)
    assert any(widget.label == "Training epochs" for widget in at.number_input)
    assert not any(widget.label in {"Hidden units", "Boosting iterations"}
                   for widget in at.number_input)


def test_model_defaults_cover_direct_horizons_without_looking_at_test():
    rows = [{"unit_id": uid, "timestamp_s": step * i, "signal": float(i), "gap_before": i == 0}
            for uid in ("train1", "train2", "val1", "val2", "test1")
            for i in range(800 if uid in {"train2", "val2"} else 300)
            for step in ([60.0] if uid.startswith(("train", "val")) else [1.0])]
    frame = pd.DataFrame(rows)
    snapshot = {"features": frame, "split": {"train": ["train1", "train2"],
                                             "validation": ["val1", "val2"], "test": ["test1"]},
                "schema": {"source_kind": "xjtu_bearings"}}
    profiles = {engine: _default_profile(snapshot, engine)[0]
                for engine in ("gru", "lstm", "quantile_boosting", "full_cns")}
    assert all(profile["horizons_s"][-1] == 549 * 60 for profile in profiles.values())
    assert all(len(profile["horizons_s"]) <= 24 for profile in profiles.values())
    assert profiles["gru"]["residual_forecast"] and profiles["lstm"]["residual_forecast"]
    assert profiles["gru"]["history_length"] != profiles["lstm"]["history_length"]
    assert profiles["quantile_boosting"]["max_iter"] == 120
    assert profiles["full_cns"]["history_length"] == 20
    changed = frame.copy()
    changed.loc[changed.unit_id == "test1", "timestamp_s"] *= 1000
    assert _default_profile({**snapshot, "features": changed}, "gru")[0] == profiles["gru"]

    filter_frame = frame.loc[frame.unit_id != "test1"].copy()
    filter_frame["timestamp_s"] /= 600
    filter_snapshot = {**snapshot, "features": filter_frame, "schema": {"source_kind": "hse_filters"}}
    profile, coverage = _default_profile(filter_snapshot, "gru")
    assert profile["horizons_s"][-1] == pytest.approx(54.9)
    assert coverage[-1]["validation_units"] == 1


def test_average_length_is_not_claimed_when_no_validation_target_exists():
    frame = pd.DataFrame([{"unit_id": uid, "timestamp_s": float(i * 60),
                           "signal": float(i), "gap_before": i == 0}
                          for uid in ("train1", "val1") for i in range(400)])
    snapshot = {"features": frame, "split": {"train": ["train1"], "validation": ["val1"]},
                "schema": {"source_kind": "xjtu_bearings"}}
    profile, coverage = _default_profile(snapshot, "gru")
    assert profile["horizons_s"][-1] < 399 * 60
    assert all(row["horizon_s"] < 399 * 60 for row in coverage)


def test_average_train_duration_excludes_gaps_and_held_out_test():
    features = pd.DataFrame({"unit_id": ["train"] * 6 + ["test"] * 2,
                             "timestamp_s": [0, 1, 2, 100, 101, 102, 0, 10000],
                             "signal": [1.0] * 8,
                             "gap_before": [True, False, False, True, False, False, True, False]})
    snapshot = {"features": features, "split": {"train": ["train"], "test": ["test"]}}
    assert average_training_duration_s(snapshot) == 4.0


def test_quality_gap_count_excludes_unit_start():
    frame = pd.DataFrame({"unit_id": ["A", "A", "A", "B", "B"],
                          "timestamp_s": [0, 1, 2, 0, 1],
                          "signal": [1, 2, 3, 4, 5],
                          "gap_before": [True, False, True, True, False]})
    summary = part_summary(frame, {"train": ["A", "B"]}, "train")
    assert summary["gaps"] == 1
    assert gap_safe_trace(frame[frame.unit_id == "A"])[0] == [0.0, 1.0, None, 2.0]


def test_full_cns_training_dispatch_and_saved_results(monkeypatch, tmp_path, signal_cns_fixture):
    import os
    from types import SimpleNamespace

    from pdm.io_util import read_json
    from pdm.worker import job_path, run_job, status_for_project

    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    worker_root = tmp_path / "worker"
    worker_root.mkdir()
    monkeypatch.setattr("pdm.worker.worker_dir", lambda: worker_root)
    monkeypatch.setattr("pdm.cli.worker_dir", lambda: worker_root)
    monkeypatch.setattr("pdm.visualization.live.clear_live_activity", lambda: None)
    _no_job(monkeypatch)
    _, project, snapshot = make_contract_snapshot(root)
    queued = []

    def launch():
        queued.append(read_json(job_path()))
        return SimpleNamespace(pid=os.getpid())

    # Exercise real UI -> spawn_worker validation -> queued job -> worker -> Results.
    # Only process creation and the costly biological graph are replaced here.
    monkeypatch.setattr("pdm.cli._launch_worker_process", launch)
    at = _app()
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Training"
    at.session_state[f"training_task:{project['project_id']}"] = "signal_forecast"
    at.run()
    next(widget for widget in at.selectbox if widget.label == "Model").select("full_cns").run()
    next(button for button in at.button if button.label == "Train model").click().run()
    assert not at.exception and len(queued) == 1
    assert queued[0]["engine_id"] == "full_cns"
    assert {"history_length", "horizons_s", "seed", "forecast_mode"} <= set(queued[0]["params"])
    assert queued[0]["params"]["forecast_mode"] == "bounded_trend_corridor"
    assert queued[0]["snapshot_id"] == snapshot["snapshot_id"]
    assert status_for_project(project["project_id"])["status"] == "queued"
    run_job(queued[0])
    assert status_for_project(project["project_id"])["status"] == "completed"
    at.session_state["project_step"] = "Results"
    at.run()
    assert not at.exception and not at.error
    assert any(widget.label == "Connectome source" for widget in at.expander)
    assert any("test-fixture-only" in c for c in _captions(at))


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
    monkeypatch.setattr("pdm.project_ui._import_cards",
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


def _contract_quality(monkeypatch, tmp_path, tab: str = "Training Data"):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _snapshot = make_contract_snapshot(root)
    pid = project["project_id"]
    return store, pid, _quality_app(pid, tab)


def _count_line(counts: dict) -> str:
    return (
        f"Training {counts['train']} · Validation {counts['validation']} · "
        f"Testing {counts['test']} units."
    )


def _raise_membership(exc: BaseException):
    def _fail(*_args, **_kwargs):
        raise exc
    return _fail


def test_quality_move_unit_publishes_new_snapshot(monkeypatch, tmp_path):
    store, pid, at = _contract_quality(monkeypatch, tmp_path)
    assert not at.exception
    before = load_snapshot(pid, store=store)
    sid = before["snapshot_id"]
    unit = at.selectbox(key="quality_unit_train").value
    destination = at.selectbox(key="quality_move_to:train").value
    assert destination == "validation"
    assert len(before["split"]["train"]) >= 2
    preview = preview_move(before, [unit], destination)
    assert preview["problem"] is None
    assert f"After the move: {_count_line(preview['counts'])}" in _captions(at)
    assert ui_copy.QUALITY_MOVE_TEST_OPTIMISM not in _captions(at)
    move_button = at.button(key="quality_move:train")
    replace_button = at.button(key="quality_replace:train")
    assert not move_button.disabled and move_button.proto.type != "primary"
    assert replace_button.proto.type != "primary"
    assert [widget.label for widget in at.selectbox if widget.label == "Move to"] == ["Move to"]
    calls = []

    def track_move(project_id, unit_ids, dest, *, expected_snapshot_id, store=None):
        calls.append((list(unit_ids), dest, expected_snapshot_id))
        return move_units(
            project_id, unit_ids, dest, expected_snapshot_id=expected_snapshot_id, store=store,
        )

    monkeypatch.setattr("pdm.project_quality_ui.move_units", track_move)
    monkeypatch.setattr(
        "pdm.project_quality_ui.swap_units",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("swap during move")),
    )
    at.session_state["quality_move_to:sentinel"] = "keep"
    at.session_state["quality_replace_with:sentinel"] = "keep"
    at.session_state["quality_unit_sentinel"] = "keep"
    at.session_state["quality_limits:sentinel"] = {"yellow": 1.0}
    at.session_state["quality_limit_suggest:p:s"] = "leave"
    at.button(key="quality_move:train").click()
    at.run()
    assert not at.exception
    assert calls == [([unit], destination, sid)]
    loaded = load_snapshot(pid, store=store)
    assert loaded["snapshot_id"] != sid
    assert project_store().get(pid)["active_snapshot_id"] == loaded["snapshot_id"]
    assert unit not in loaded["split"]["train"]
    assert unit in loaded["split"][destination]
    assert load_snapshot(pid, sid, store=store)["split"]["train"] == before["split"]["train"]
    expected = ui_copy.QUALITY_MOVE_DONE.format(unit=unit, destination=SPLIT_LABELS[destination])
    assert [str(item.value) for item in at.success] == [expected]
    for key in ("quality_move_to:sentinel", "quality_replace_with:sentinel", "quality_unit_sentinel"):
        assert key not in at.session_state
    assert at.session_state["quality_limits:sentinel"] == {"yellow": 1.0}
    assert at.session_state["quality_limit_suggest:p:s"] == "leave"
    at.run()
    assert expected not in [str(item.value) for item in at.success]


def test_quality_move_refuses_emptying_a_set(monkeypatch, tmp_path):
    store, pid, at = _contract_quality(monkeypatch, tmp_path, "Testing Data")
    assert not at.exception
    before = load_snapshot(pid, store=store)
    sid = before["snapshot_id"]
    assert len(before["split"]["test"]) == 1
    problem = "Testing Data would have no units. Keep at least one unit in each set."
    assert problem in [str(item.value) for item in at.warning]
    move_to = at.selectbox(key="quality_move_to:test")
    assert move_to.options == ["Training Data", "Validation Data"]
    assert at.button(key="quality_move:test").disabled
    with pytest.raises(AppTestError):
        at.button(key="quality_move:test").click()
    move_to.set_value("validation")
    at.run()
    assert not at.exception
    assert problem in [str(item.value) for item in at.warning]
    assert at.button(key="quality_move:test").disabled
    assert project_store().get(pid)["active_snapshot_id"] == sid
    assert load_snapshot(pid, store=store)["split"] == before["split"]


def test_quality_replace_keeps_counts_in_one_snapshot(monkeypatch, tmp_path):
    store, pid, at = _contract_quality(monkeypatch, tmp_path)
    assert not at.exception
    before = load_snapshot(pid, store=store)
    sid = before["snapshot_id"]
    counts = dict(before["split"]["realized_counts"])
    unit = at.selectbox(key="quality_unit_train").value
    partner = at.selectbox(key="quality_replace_with:train").value
    assert partner in set(map(str, before["split"]["validation"]))
    swap = preview_swap(before, unit, partner)
    assert swap["problem"] is None
    set_a = SPLIT_LABELS[swap["to"][unit]]
    set_b = SPLIT_LABELS[swap["to"][partner]]
    caption = f"{unit} joins {set_a}; {partner} joins {set_b}. {_count_line(swap['counts'])}"
    assert caption in _captions(at)
    assert set_a == "Validation Data" and set_b == "Training Data"
    calls = []

    def track_swap(project_id, unit_a, unit_b, *, expected_snapshot_id, store=None):
        calls.append((unit_a, unit_b, expected_snapshot_id))
        return swap_units(
            project_id, unit_a, unit_b, expected_snapshot_id=expected_snapshot_id, store=store,
        )

    monkeypatch.setattr("pdm.project_quality_ui.swap_units", track_swap)
    monkeypatch.setattr(
        "pdm.project_quality_ui.move_units",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("two moves")),
    )
    at.session_state["quality_move_to:sentinel"] = "keep"
    at.session_state["quality_replace_with:sentinel"] = "keep"
    at.session_state["quality_unit_sentinel"] = "keep"
    at.button(key="quality_replace:train").click()
    at.run()
    assert not at.exception
    assert calls == [(unit, partner, sid)]
    loaded = load_snapshot(pid, store=store)
    assert loaded["snapshot_id"] != sid
    assert loaded["split"]["realized_counts"] == counts
    assert loaded["split"]["train"] == sorted((set(map(str, before["split"]["train"])) - {unit}) | {partner})
    assert loaded["split"]["validation"] == sorted(
        (set(map(str, before["split"]["validation"])) - {partner}) | {unit},
    )
    assert loaded["split"]["test"] == sorted(map(str, before["split"]["test"]))
    assert load_snapshot(pid, sid, store=store)["split"]["realized_counts"] == counts
    expected = ui_copy.QUALITY_REPLACE_DONE.format(unit_a=unit, unit_b=partner)
    assert [str(item.value) for item in at.success] == [expected]
    for key in ("quality_move_to:sentinel", "quality_replace_with:sentinel", "quality_unit_sentinel"):
        assert key not in at.session_state
    at.run()
    assert expected not in [str(item.value) for item in at.success]


def test_quality_membership_disabled_for_linked_legacy_and_active_job(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    calls = {"n": 0}

    def active(*_args, **_kwargs):
        calls["n"] += 1
        return True

    monkeypatch.setattr("pdm.project_quality_ui.heavy_job_active", active)
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_args, **_kwargs: True)
    store, project, _snapshot = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    assert calls["n"] == 1
    assert JOB_ACTIVE_MOVE_ERROR in _captions(at)
    assert ui_copy.QUALITY_MOVE_TEST_OPTIMISM not in _captions(at)
    assert "After the move:" not in "\n".join(_captions(at))
    for key in ("quality_move_to:train", "quality_replace_with:train"):
        assert at.selectbox(key=key).disabled
    for key in ("quality_move:train", "quality_replace:train"):
        button = at.button(key=key)
        assert button.disabled and button.proto.type != "primary"
    with pytest.raises(AppTestError):
        at.button(key="quality_move:train").click()
    assert project_store().get(pid)["active_snapshot_id"] == sid
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.4)
    calls["n"] = 0
    at.run()
    assert calls["n"] == 1

    _no_job(monkeypatch)
    registry = json.loads(store.registry_path.read_text())
    registry["projects"][pid]["storage_mode"] = "linked_legacy"
    store.registry_path.write_text(json.dumps(registry))
    linked = _quality_app(pid, "Training Data")
    assert not linked.exception
    assert [str(item.value) for item in linked.info] == [LINKED_LEGACY_MOVE_ERROR]
    assert not any(widget.label in {"Move to", "Replace with"} for widget in linked.selectbox)
    assert not any(button.label in {"Move unit", "Replace unit"} for button in linked.button)
    assert linked.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.4)
    assert project_store().get(pid)["active_snapshot_id"] == sid


def test_quality_testing_tab_shows_optimism_caption(monkeypatch, tmp_path):
    _store, pid, at = _contract_quality(monkeypatch, tmp_path, "Testing Data")
    assert not at.exception
    snapshot = load_snapshot(pid)
    assert len(snapshot["split"]["test"]) == 1
    assert _captions(at).count(ui_copy.QUALITY_MOVE_TEST_OPTIMISM) == 1
    partner = at.selectbox(key="quality_replace_with:test").value
    assert partner not in set(map(str, snapshot["split"]["test"]))


def test_quality_hides_official_hse_test_units_as_move_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    _no_job(monkeypatch)
    project = project_store().create("HSE view", "hse_filters")
    pid = project["project_id"]
    project_store().update(pid, active_snapshot_id="snapshot1", state="ready")
    ids = ["train_a", "train_b", "val_a", "test_free", "test_official"]
    features = pd.DataFrame({
        "unit_id": ids,
        "timestamp_s": [0.0] * len(ids),
        "signal": [1.0] * len(ids),
        "gap_before": [False] * len(ids),
    })
    units = pd.DataFrame({
        "unit_id": ids,
        "source_group": ["primary", "primary", "primary", "primary", "author_test"],
    })
    snapshot = {
        "project_id": pid,
        "snapshot_id": "snapshot1",
        "features": features,
        "units": units,
        "split": {
            "train": ["train_a", "train_b"],
            "validation": ["val_a"],
            "test": ["test_free", "test_official"],
        },
        "schema": {"signal_label": "Pressure", "signal_unit": "Pa", "source_kind": "hse_filters"},
        "report": {},
    }
    monkeypatch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
    monkeypatch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
    monkeypatch.setattr(
        "pdm.project_quality_ui.available_signal_engines",
        lambda _pid, _sid: [{"engine_id": "gru", "available": True}],
    )
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    labels = at.selectbox(key="quality_replace_with:train").options
    assert "val_a · Validation Data" in labels
    assert "test_free · Testing Data" in labels
    assert "test_official · Testing Data" not in labels
    assert not at.button(key="quality_move:train").disabled

    held = _app()
    held.session_state["project_id"] = pid
    held.session_state["project_step"] = "Data Quality"
    held.session_state["quality_tab"] = "Testing Data"
    held.session_state["quality_unit_test"] = "test_official"
    held.run()
    assert not held.exception
    assert "test_official" in held.selectbox(key="quality_unit_test").options
    assert held.selectbox(key="quality_unit_test").value == "test_official"
    assert held.button(key="quality_move:test").disabled
    assert held.button(key="quality_replace:test").disabled
    assert [str(item.value) for item in held.warning] == ["Official HSE test units stay in Testing Data."]
    assert not any(widget.label == "Replace with" for widget in held.selectbox)
    page = "\n".join([*_captions(held), *[str(item.value) for item in held.warning]])
    assert ui_copy.QUALITY_REPLACE_NONE not in page
    assert _captions(held).count(ui_copy.QUALITY_MOVE_TEST_OPTIMISM) == 1


def test_quality_empty_partner_list_has_no_selectbox(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    _no_job(monkeypatch)
    pid = project_store().create("Empty partners", "generic_sensor_csv")["project_id"]
    project_store().update(pid, active_snapshot_id="snapshot1", state="ready")
    features = pd.DataFrame({
        "unit_id": ["a", "b", "official"],
        "timestamp_s": [0.0, 0.0, 0.0],
        "signal": [1.0, 1.0, 1.0],
        "gap_before": [False, False, False],
    })
    snapshot = {
        "project_id": pid,
        "snapshot_id": "snapshot1",
        "features": features,
        "units": pd.DataFrame({
            "unit_id": ["a", "b", "official"],
            "source_group": ["primary", "primary", "author_test"],
        }),
        "split": {"train": ["a", "b"], "validation": [], "test": ["official"]},
        "schema": {"signal_label": "Vibration", "signal_unit": "g"},
        "report": {},
    }
    monkeypatch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
    monkeypatch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
    monkeypatch.setattr(
        "pdm.project_quality_ui.available_signal_engines",
        lambda _pid, _sid: [{"engine_id": "gru", "available": True}],
    )
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    assert not any(widget.label == "Replace with" for widget in at.selectbox)
    assert ui_copy.QUALITY_REPLACE_NONE in _captions(at)
    assert at.button(key="quality_replace:train").disabled
    assert not at.button(key="quality_move:train").disabled


def test_quality_membership_failure_warns_without_traceback(monkeypatch, tmp_path):
    store, pid, _at = _contract_quality(monkeypatch, tmp_path)
    sid = project_store().get(pid)["active_snapshot_id"]
    split = load_snapshot(pid, store=store)["split"]
    train_ids = sorted(str(uid) for uid in split["train"])
    others = [str(uid) for name in ("validation", "test") for uid in split[name]]
    assert len(train_ids) >= 2 and len(others) >= 2
    for exc in (
        KeyError("membership-key"),
        ValueError("membership-value"),
        RuntimeError("membership-runtime"),
        OSError("membership-os"),
    ):
        monkeypatch.setattr("pdm.project_quality_ui.move_units", _raise_membership(exc))
        at = _quality_app(pid, "Training Data")
        assert not at.exception
        at.selectbox(key="quality_unit_train").set_value(train_ids[1])
        at.selectbox(key="quality_move_to:train").set_value("test")
        at.selectbox(key="quality_replace_with:train").set_value(others[1])
        at.session_state["quality_move_to:sentinel"] = "keep"
        at.session_state["quality_replace_with:sentinel"] = "keep"
        at.session_state["quality_unit_sentinel"] = "keep"
        at.session_state["quality_limits:sentinel"] = {"yellow": 1.0}
        at.button(key="quality_move:train").click()
        at.run()
        assert not at.exception
        assert str(exc) in [str(item.value) for item in at.warning]
        assert not at.success
        assert [title.value for title in at.title] == ["Data Quality"]
        assert project_store().get(pid)["active_snapshot_id"] == sid
        assert load_snapshot(pid, store=store)["snapshot_id"] == sid
        assert at.selectbox(key="quality_unit_train").value == train_ids[1]
        assert at.selectbox(key="quality_move_to:train").value == "test"
        assert at.selectbox(key="quality_replace_with:train").value == others[1]
        assert at.session_state["quality_move_to:sentinel"] == "keep"
        assert at.session_state["quality_replace_with:sentinel"] == "keep"
        assert at.session_state["quality_unit_sentinel"] == "keep"
        assert at.session_state["quality_limits:sentinel"] == {"yellow": 1.0}


def test_quality_session_reset_drops_membership_prefixes():
    def page():
        import streamlit as st

        from pdm.project_ui import _reset_project_session

        if st.button("Arm"):
            st.session_state["quality_move_to:train"] = "test"
            st.session_state["quality_move:train"] = True
            st.session_state["quality_replace_with:v"] = "u"
            st.session_state["quality_replace:v"] = True
            st.session_state["quality_membership_notice"] = ("warning", "x")
            st.session_state["quality_unit_train"] = "unit"
            st.session_state["quality_limit_yellow:p:s"] = 1.0
            st.session_state["quality_limits:p:s"] = {"yellow": 1.0}
            st.session_state["quality_limit_suggest:p:s"] = "leave"
            st.session_state["quality_limit_suggest_note:p:s"] = "note"
        if st.button("Reset"):
            _reset_project_session()

    at = AppTest.from_function(page, default_timeout=15).run()
    next(button for button in at.button if button.label == "Arm").click()
    at.run()
    assert at.session_state["quality_membership_notice"] == ("warning", "x")
    next(button for button in at.button if button.label == "Reset").click()
    at.run()
    for key in (
        "quality_move_to:train", "quality_move:train", "quality_replace_with:v",
        "quality_replace:v", "quality_membership_notice", "quality_unit_train",
    ):
        assert key not in at.session_state
    assert at.session_state["quality_limit_yellow:p:s"] == 1.0
    assert at.session_state["quality_limits:p:s"] == {"yellow": 1.0}
    assert "quality_limit_suggest:p:s" not in at.session_state
    assert "quality_limit_suggest_note:p:s" not in at.session_state


def _texts(elements) -> list[str]:
    return [str(item.value) for item in elements]


def _pair_lines(yellow: float, red: float) -> set[float]:
    return {round(float(yellow), 6), round(float(red), 6)}


def _watch_proposal(monkeypatch) -> list[tuple[list[str], str]]:
    seen: list[tuple[list[str], str]] = []
    real = propose_absolute_limits

    def spy(features, train_ids, direction):
        seen.append(([str(uid) for uid in train_ids], direction))
        return real(features, train_ids, direction)

    monkeypatch.setattr("pdm.project_quality_ui.propose_absolute_limits", spy)
    return seen


def test_quality_suggest_fills_inputs_without_saving(monkeypatch, tmp_path):
    store, pid, at = _contract_quality(monkeypatch, tmp_path)
    assert not at.exception
    snapshot = load_snapshot(pid, store=store)
    sid = snapshot["snapshot_id"]
    original = _snapshot_bytes(store, pid, sid)
    train_ids = [str(uid) for uid in snapshot["split"]["train"]]
    holdout = {str(uid) for name in ("validation", "test") for uid in snapshot["split"][name]}
    proposal = propose_absolute_limits(snapshot["features"], snapshot["split"]["train"], "above")
    assert proposal["ok"] is True and proposal["n"] == 30
    button = at.button(key=_key("suggest", pid, sid))
    assert not button.disabled and button.proto.type != "primary"
    assert (button.help or button.proto.help) == ui_copy.QUALITY_SUGGEST_LIMITS_HELP
    assert ui_copy.QUALITY_SUGGEST_LIMITS_CAPTION in _captions(at)
    assert at.radio(key=_key("direction", pid, sid)).value == "above"
    seen = _watch_proposal(monkeypatch)
    button.click()
    at.run()
    assert not at.exception
    text = ui_copy.QUALITY_SUGGEST_DONE.format(yellow=proposal["yellow"], red=proposal["red"])
    assert _texts(at.success) == [text]
    assert text not in _texts(at.error)
    assert at.session_state[_key("suggest_note", pid, sid)] == text
    assert _key("status", pid, sid) not in at.session_state
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(proposal["yellow"])
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(proposal["red"])
    assert at.radio(key=_key("direction", pid, sid)).value == "above"
    rule = at.session_state[limits_key(pid, sid)]
    assert rule["mode"] == "absolute" and rule["direction"] == "above"
    assert rule["yellow"] == pytest.approx(proposal["yellow"])
    assert rule["red"] == pytest.approx(proposal["red"])
    assert _chart_limit_ys(at) == _pair_lines(proposal["yellow"], proposal["red"])
    assert seen and all(ids == train_ids and direction == "above" and set(ids).isdisjoint(holdout)
                        for ids, direction in seen)
    assert load_zone_limits(pid, sid, store=store) is None
    assert not (store.snapshot_path(pid, sid) / "zone_limits.json").exists()
    assert project_store().get(pid)["active_snapshot_id"] == sid
    assert _snapshot_bytes(store, pid, sid) == original
    assert load_snapshot(pid, store=store)["schema"]["thresholds"]["yellow"] == 0.4
    at.run()
    assert _texts(at.success) == [text]
    at.number_input(key=_key("red", pid, sid)).set_value(1.5)
    at.run()
    assert not at.exception
    assert _key("suggest_note", pid, sid) not in at.session_state
    assert text not in _texts(at.success)
    assert load_zone_limits(pid, sid, store=store) is None


def test_quality_suggest_disabled_when_early_pool_is_short(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    _no_job(monkeypatch)
    store = project_store()
    project = store.create("Short pool", "generic_sensor_csv")
    pid = project["project_id"]
    store.update(pid, active_snapshot_id="snapshot1", state="ready")
    features = pd.DataFrame({
        "unit_id": ["train1", "train1", "val1", "test1"],
        "timestamp_s": [0.0, 1.0, 0.0, 0.0],
        "signal": [-2.0, -1.0, 1.0, 2.0],
        "gap_before": [False, True, False, False],
    })
    snapshot = {
        "project_id": pid, "snapshot_id": "snapshot1", "features": features,
        "split": {"train": ["train1"], "validation": ["val1"], "test": ["test1"]},
        "schema": {"signal_label": "Vibration", "signal_unit": "g"},
        "report": {},
    }
    monkeypatch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
    monkeypatch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
    monkeypatch.setattr(
        "pdm.project_quality_ui.available_signal_engines",
        lambda _pid, _sid: [{"engine_id": "gru", "available": True}],
    )
    at = _quality_app(pid, "Training Data")
    proposal = propose_absolute_limits(features, snapshot["split"]["train"], "above")
    assert not at.exception
    assert proposal["ok"] is False and proposal["n"] == 2
    button = at.button(key=_key("suggest", pid, "snapshot1"))
    assert button.disabled and button.proto.type != "primary"
    assert (button.help or button.proto.help) == ui_copy.QUALITY_SUGGEST_LIMITS_HELP
    assert ui_copy.QUALITY_SUGGEST_LIMITS_CAPTION in _captions(at)
    assert proposal["reason"] in _captions(at)
    with pytest.raises(AppTestError):
        button.click()


def test_quality_suggest_respects_below_direction(monkeypatch, tmp_path):
    store, pid, at = _contract_quality(monkeypatch, tmp_path)
    snapshot = load_snapshot(pid, store=store)
    sid = snapshot["snapshot_id"]
    at.radio(key=_key("direction", pid, sid)).set_value("below")
    at.run()
    assert not at.exception
    assert at.radio(key=_key("direction", pid, sid)).value == "below"
    proposal = propose_absolute_limits(snapshot["features"], snapshot["split"]["train"], "below")
    above = propose_absolute_limits(snapshot["features"], snapshot["split"]["train"], "above")
    assert proposal["ok"] is True
    assert (proposal["yellow"], proposal["red"]) != (above["yellow"], above["red"])
    assert not at.button(key=_key("suggest", pid, sid)).disabled
    at.button(key=_key("suggest", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert at.radio(key=_key("direction", pid, sid)).value == "below"
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(proposal["yellow"])
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(proposal["red"])
    rule = at.session_state[limits_key(pid, sid)]
    assert rule["direction"] == "below"
    assert rule["yellow"] == pytest.approx(proposal["yellow"])
    assert rule["red"] == pytest.approx(proposal["red"])
    text = ui_copy.QUALITY_SUGGEST_DONE.format(yellow=proposal["yellow"], red=proposal["red"])
    assert text in _texts(at.success)
    assert _key("status", pid, sid) not in at.session_state
    assert load_zone_limits(pid, sid, store=store) is None


def test_quality_suggest_cancel_reverts_and_save_writes_sidecar_only(monkeypatch, tmp_path):
    store, pid, at = _contract_quality(monkeypatch, tmp_path)
    snapshot = load_snapshot(pid, store=store)
    sid = snapshot["snapshot_id"]
    original = _snapshot_bytes(store, pid, sid)
    proposal = propose_absolute_limits(snapshot["features"], snapshot["split"]["train"], "above")
    text = ui_copy.QUALITY_SUGGEST_DONE.format(yellow=proposal["yellow"], red=proposal["red"])
    at.button(key=_key("suggest", pid, sid)).click()
    at.run()
    assert not at.exception
    assert _chart_limit_ys(at) == _pair_lines(proposal["yellow"], proposal["red"])
    assert text in _texts(at.success)
    assert load_zone_limits(pid, sid, store=store) is None
    at.button(key=_limit_button("cancel", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(0.4)
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(0.8)
    assert at.radio(key=_key("direction", pid, sid)).value == "above"
    assert _chart_limit_ys(at) == {0.4, 0.8}
    assert _key("suggest_note", pid, sid) not in at.session_state
    assert text not in _texts(at.success)
    assert load_zone_limits(pid, sid, store=store) is None
    assert _snapshot_bytes(store, pid, sid) == original
    assert project_store().get(pid)["active_snapshot_id"] == sid
    at.button(key=_key("suggest", pid, sid)).click()
    at.run()
    assert not at.button(key=_limit_button("save", pid, sid)).disabled
    at.button(key=_limit_button("save", pid, sid)).click()
    at.run()
    assert not at.exception and not at.error
    assert load_zone_limits(pid, sid, store=store) == {
        "mode": "absolute", "direction": "above",
        "yellow": float(proposal["yellow"]), "red": float(proposal["red"]),
    }
    loaded = load_snapshot(pid, store=store)
    assert loaded["snapshot_id"] == sid
    assert loaded["schema"]["thresholds"]["yellow"] == 0.4
    assert loaded["schema"]["thresholds"]["red"] == 0.8
    assert _snapshot_bytes(store, pid, sid) == original
    assert _key("suggest_note", pid, sid) not in at.session_state
    assert text not in _texts(at.success)


def test_quality_suggest_allowed_during_job_and_on_linked_legacy(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))

    def _refuse_save(*_args, **_kwargs):
        raise AssertionError("suggest must not save")

    monkeypatch.setattr("pdm.project_quality_ui.save_zone_limits", _refuse_save)
    _no_job(monkeypatch, active=True)
    store, project, _snapshot = make_contract_snapshot(root)
    pid = project["project_id"]
    sid = project_store().get(pid)["active_snapshot_id"]
    snapshot = load_snapshot(pid, store=store)
    proposal = propose_absolute_limits(snapshot["features"], snapshot["split"]["train"], "above")
    original = _snapshot_bytes(store, pid, sid)
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    suggest = at.button(key=_key("suggest", pid, sid))
    assert not suggest.disabled and suggest.proto.type != "primary"
    assert at.button(key=_limit_button("save", pid, sid)).disabled
    suggest.click()
    at.run()
    assert not at.exception
    assert at.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(proposal["yellow"])
    assert at.number_input(key=_key("red", pid, sid)).value == pytest.approx(proposal["red"])
    assert _chart_limit_ys(at) == _pair_lines(proposal["yellow"], proposal["red"])
    assert at.button(key=_limit_button("save", pid, sid)).disabled
    assert load_zone_limits(pid, sid, store=store) is None
    assert not (store.snapshot_path(pid, sid) / "zone_limits.json").exists()
    assert _snapshot_bytes(store, pid, sid) == original
    assert project_store().get(pid)["active_snapshot_id"] == sid

    _no_job(monkeypatch)
    registry = json.loads(store.registry_path.read_text())
    registry["projects"][pid]["storage_mode"] = "linked_legacy"
    store.registry_path.write_text(json.dumps(registry))
    linked = _quality_app(pid, "Training Data")
    assert not linked.exception
    assert _texts(linked.info) == [LINKED_LEGACY_MOVE_ERROR]
    assert not any(button.label in {"Move unit", "Replace unit"} for button in linked.button)
    legacy = linked.button(key=_key("suggest", pid, sid))
    assert not legacy.disabled and legacy.proto.type != "primary"
    legacy.click()
    linked.run()
    assert not linked.exception
    assert linked.number_input(key=_key("yellow", pid, sid)).value == pytest.approx(proposal["yellow"])
    assert linked.number_input(key=_key("red", pid, sid)).value == pytest.approx(proposal["red"])
    assert load_zone_limits(pid, sid, store=store) is None
    assert project_store().get(pid)["storage_mode"] == "linked_legacy"
    assert project_store().get(pid)["active_snapshot_id"] == sid


def test_quality_suggest_after_move_uses_new_training_ids(monkeypatch, tmp_path):
    store, pid, at = _contract_quality(monkeypatch, tmp_path)
    before = load_snapshot(pid, store=store)
    sid = before["snapshot_id"]
    unit = at.selectbox(key="quality_unit_train").value
    at.button(key="quality_move:train").click()
    at.run()
    assert not at.exception
    loaded = load_snapshot(pid, store=store)
    new_sid = loaded["snapshot_id"]
    assert new_sid != sid
    new_ids = [str(uid) for uid in loaded["split"]["train"]]
    parent_ids = [str(uid) for uid in before["split"]["train"]]
    assert unit not in new_ids and new_ids != parent_ids
    holdout = {str(uid) for name in ("validation", "test") for uid in loaded["split"][name]}
    seen = _watch_proposal(monkeypatch)
    at.button(key=_key("suggest", pid, new_sid)).click()
    at.run()
    assert not at.exception
    assert seen
    for ids, direction in seen:
        assert ids == new_ids and direction == "above"
        assert set(ids).isdisjoint(holdout)
        assert ids != parent_ids
    child = propose_absolute_limits(loaded["features"], loaded["split"]["train"], "above")
    parent = propose_absolute_limits(before["features"], before["split"]["train"], "above")
    assert (child["yellow"], child["red"]) != (parent["yellow"], parent["red"])
    assert at.number_input(key=_key("yellow", pid, new_sid)).value == pytest.approx(child["yellow"])
    assert at.number_input(key=_key("red", pid, new_sid)).value == pytest.approx(child["red"])
    assert load_zone_limits(pid, new_sid, store=store) is None
