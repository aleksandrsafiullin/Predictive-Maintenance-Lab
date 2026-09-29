from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pdm import project_zones
from pdm.data.project_prepare import load_snapshot
from pdm.paths import project_root
from pdm.project_quality_ui import gap_safe_trace, part_summary
from pdm.project_training_ui import _parse_horizons
from pdm.project_ui import _auto_shares, _open_step, _stage_uploads
from pdm.projects import project_store
from pdm.ui_copy import QUALITY_NO_ZONES
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
    assert _captions(at).count("Counts are the saved snapshot.") == 1
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
    assert any("Admitted measurements" in str(c.value) for c in at.caption)
    assert any(button.label == "Continue to Training" for button in at.button)
    assert not any("dataset_version" in str(markdown.value) for markdown in at.markdown)
    monkeypatch.setattr("pdm.project_quality_ui.available_signal_engines",
                        lambda _pid, _sid: [{"engine_id": "gru", "available": False}])
    at.run()
    assert not at.exception
    assert not any(button.label == "Continue to Training" for button in at.button)


def _quality_app(project_id: str, tab: str | None = None) -> AppTest:
    at = _app()
    at.session_state["project_id"] = project_id
    at.session_state["project_step"] = "Data Quality"
    if tab:
        at.session_state["quality_tab"] = tab
    return at.run()


def _chart_trace_names(at: AppTest) -> list[str]:
    return [trace.get("name") for trace in json.loads(at.get("plotly_chart")[0].proto.spec)["data"]]


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
    captions = _captions(at)
    rule = next(c for c in captions if "forecast future" in c)
    assert "not zone classes" in rule and "yellow at ≥ 0.4 g" in rule and "red at ≥ 0.8 g" in rule
    counts = project_zones.zone_counts(features, snapshot["split"]["train"], snapshot["schema"])
    assert (f"Zones: Green {counts['green']} · Yellow {counts['yellow']} · Red {counts['red']} · "
            f"Not zoned {counts['unknown']} rows") in captions
    table = at.table[0].value
    assert list(table["Zone"]) == [names[zone] for zone in labelled["zone"]]


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
    assert QUALITY_NO_ZONES in _captions(at)
    assert not any(c.startswith("Zones:") for c in _captions(at))
    assert _chart_trace_names(at) == ["Vibration"]
    assert set(at.table[0].value["Zone"]) == {"Not zoned"}


def test_quality_baseline_rule_caption_uses_schema_values(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    schema = {"signal_label": "Vibration", "signal_unit": "g",
              "thresholds": {"mode": "initial_baseline_multiple", "direction": "above", "baseline_n": 4,
                             "red_ratio": 2.5, "note": "Provisional"}}
    project_id = _patch_quality_snapshot(monkeypatch, schema)
    at = _quality_app(project_id)
    assert not at.exception
    rule = next(c for c in _captions(at) if "forecast future" in c)
    assert "first 4" in rule and "2.5 × median" in rule and "Provisional" in rule
    zones = list(at.table[0].value["Zone"])
    assert zones[:3] == ["Not zoned"] * 3
    assert zones[3:] == ["Green", "Yellow", "Red"]
    assert "Zones: Green 1 · Yellow 1 · Red 1 · Not zoned 3 rows" in _captions(at)


def test_quality_zone_labels_only_for_open_tab(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, _ = make_contract_snapshot(root)
    snapshot = load_snapshot(project["project_id"], store=store)
    calls = []
    real = project_zones.zone_counts

    def spy(features, unit_ids, schema):
        calls.append(tuple(str(uid) for uid in unit_ids))
        return real(features, unit_ids, schema)

    monkeypatch.setattr("pdm.project_zones.zone_counts", spy)
    at = _quality_app(project["project_id"], "Training Data")
    assert not at.exception
    assert calls == [tuple(str(uid) for uid in snapshot["split"]["train"])]
    assert len(at.get("plotly_chart")) == 1


def _no_job(monkeypatch, active: bool = False) -> None:
    monkeypatch.setattr("pdm.project_quality_ui.heavy_job_active", lambda *_a: active)
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_a: active)


def test_quality_move_unit_publishes_new_snapshot(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    old_id = project_store().get(pid)["active_snapshot_id"]
    unit = sorted(load_snapshot(pid, store=store)["split"]["train"])[-1]
    at = _quality_app(pid, "Training Data")
    assert not at.exception
    assert at.button(key="quality_move:train").disabled
    at.multiselect(key="quality_move_units:train").set_value([unit])
    at.selectbox(key="quality_move_to:train").set_value("Validation Data")
    at.run()
    assert "After the move: Training 5 · Validation 3 · Testing 1 units." in _captions(at)
    at.button(key="quality_move:train").click()
    at.run()
    assert not at.exception
    new_id = project_store().get(pid)["active_snapshot_id"]
    assert new_id != old_id
    assert any("Moved 1 unit(s) to Validation Data. A new data snapshot is active" in str(s.value)
               for s in at.success)
    assert unit in at.multiselect(key="quality_move_units:validation").options
    assert unit not in at.multiselect(key="quality_move_units:train").options
    assert at.multiselect(key="quality_move_units:train").value == []
    assert unit in load_snapshot(pid, new_id, store=store)["split"]["validation"]
    assert unit in load_snapshot(pid, old_id, store=store)["split"]["train"]


def test_quality_move_callback_separates_bad_destination_from_move_errors(monkeypatch):
    from types import SimpleNamespace

    from pdm import project_quality_ui

    fake = SimpleNamespace(session_state={"quality_move_units:train": ["u1"], "quality_move_to:train": "Nowhere"})
    monkeypatch.setattr(project_quality_ui, "st", fake)
    monkeypatch.setattr(project_quality_ui, "move_units",
                        lambda *_a, **_k: pytest.fail("move_units must not run for an unknown set"))
    project_quality_ui._do_move("p1", "s1", "train")
    assert fake.session_state["quality_move_flash"] == ("warning", "Choose the set to move the units to.")

    fake.session_state.pop("quality_move_flash")
    fake.session_state["quality_move_to:train"] = "Validation Data"

    def broken(*_a, **_k):
        raise KeyError("units")

    monkeypatch.setattr(project_quality_ui, "move_units", broken)
    with pytest.raises(KeyError):
        project_quality_ui._do_move("p1", "s1", "train")
    assert "quality_move_flash" not in fake.session_state


def test_quality_move_refuses_emptying_a_set(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch)
    store, project, _ = make_contract_snapshot(root)
    pid = project["project_id"]
    old_id = project_store().get(pid)["active_snapshot_id"]
    units = load_snapshot(pid, store=store)["split"]["validation"]
    at = _quality_app(pid, "Validation Data")
    at.multiselect(key="quality_move_units:validation").set_value(list(units))
    at.run()
    assert not at.exception
    assert any("Validation Data would have no units" in str(w.value) for w in at.warning)
    assert at.button(key="quality_move:validation").disabled
    assert project_store().get(pid)["active_snapshot_id"] == old_id


def test_quality_move_disabled_for_linked_legacy_and_active_job(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _no_job(monkeypatch, active=True)
    _, project, _ = make_contract_snapshot(root)
    at = _quality_app(project["project_id"])
    assert not at.exception
    assert at.button(key="quality_move:train").disabled
    assert at.multiselect(key="quality_move_units:train").disabled
    assert "Wait for the current job to finish before changing sets." in _captions(at)
    monkeypatch.setattr("pdm.project_ui._maybe_wrap_legacy",
                        lambda selected: {**selected, "storage_mode": "linked_legacy"})
    at = _quality_app(project["project_id"])
    assert not at.exception
    assert any("published split of its source dataset" in str(i.value) for i in at.info)
    assert not any(w.label == "Units to move" for w in at.multiselect)
    assert not any(b.label == "Move selected units" for b in at.button)


def test_quality_move_testing_shows_optimism_caption(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    _no_job(monkeypatch)
    project_id = _patch_quality_snapshot(monkeypatch, {"signal_label": "Vibration", "signal_unit": "g"})
    optimism = "Changing Testing units after reviewing results makes later Test scores optimistic."
    at = _quality_app(project_id, "Training Data")
    assert not at.exception
    assert _captions(at).count(optimism) == 1
    at.selectbox(key="quality_move_to:train").set_value("Testing Data")
    at.run()
    assert _captions(at).count(optimism) == 2
    assert not any("official HSE" in c for c in _captions(at))


def test_quality_move_hides_official_hse_test_units(monkeypatch, tmp_path):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    _no_job(monkeypatch)
    store = project_store()
    project = store.create("Filters", "generic_sensor_csv")
    store.update(project["project_id"], active_snapshot_id="snapshot1", state="ready")
    snapshot = _minimal_quality_snapshot(project["project_id"], {"signal_label": "Pressure", "signal_unit": "Pa"})
    snapshot["features"] = pd.concat([snapshot["features"], pd.DataFrame({
        "unit_id": ["test2"], "timestamp_s": [0.0], "signal": [1.0], "gap_before": [False]})], ignore_index=True)
    snapshot["split"]["test"] = ["test1", "test2"]
    snapshot["units"] = pd.DataFrame({"unit_id": ["train1", "val1", "test1", "test2"],
                                      "source_group": ["primary", "primary", "author_test", "primary"]})
    monkeypatch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
    monkeypatch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
    monkeypatch.setattr("pdm.project_quality_ui.available_signal_engines",
                        lambda _pid, _sid: [{"engine_id": "gru", "available": True}])
    at = _quality_app(project["project_id"], "Testing Data")
    assert not at.exception
    assert list(at.multiselect(key="quality_move_units:test").options) == ["test2"]
    assert "1 official HSE test unit(s) are fixed in Testing Data." in _captions(at)


def test_training_page_mentions_runs_from_previous_snapshot(monkeypatch, tmp_path):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, _ = make_contract_snapshot(root)
    monkeypatch.setattr("pdm.project_training_ui.list_project_runs",
                        lambda _pid: [{"run_id": "old", "snapshot_id": "previous-snapshot"}])
    at = _app()
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Training"
    at.run()
    assert not at.exception
    assert ("1 earlier model run(s) were trained on a previous data snapshot and are not shown. "
            "Train again on this data.") in _captions(at)


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
