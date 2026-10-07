"""Display choices survive a fresh session without changing observed data or models."""

from __future__ import annotations

import pytest

from pdm import live_simulator as demo
from pdm.project_snapshot import project_snapshot
from pdm.project_view_preferences import PREFERENCES_FILE, load_preferences, save_preferences
from pdm.projects import project_store
from tests.test_live_monitor import select_machine_row, train
from tests.test_live_monitor import sensor as sensor_fixture
from tests.test_sensor_import_view import app, seal


@pytest.fixture
def sensor(tmp_path, monkeypatch):
    return sensor_fixture.__wrapped__(tmp_path, monkeypatch)


def test_preferences_isolate_projects_views_and_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    store = project_store()
    first = store.create("Benchmark", "generic_sensor_csv")
    second = store.create("Other project", "generic_sensor_csv")
    pid = first["project_id"]
    store.update(pid, active_snapshot_id="snapshot-one")
    save_preferences(pid, "snapshot-one", "results", {"units": {"test": "SYN-00447"}})
    save_preferences(pid, "snapshot-one", "live_monitor", {"demo_units": ["SYN-00446", "SYN-00447", "SYN-00448"]})
    save_preferences(pid, "snapshot-one", "results", {"part": "test"})
    assert load_preferences(pid, "snapshot-one", "results") == {
        "part": "test", "units": {"test": "SYN-00447"},
    }
    assert len(load_preferences(pid, "snapshot-one", "live_monitor")["demo_units"]) == 3
    assert load_preferences(second["project_id"], "snapshot-one", "results") == {}
    path = store.project_path(pid) / PREFERENCES_FILE
    before = path.read_bytes()
    store.update(pid, active_snapshot_id="snapshot-two")
    assert load_preferences(pid, "snapshot-two", "results") == {}
    save_preferences(pid, "snapshot-one", "results", {"part": "train"})
    assert path.read_bytes() == before
    path.write_text("invalid JSON")
    assert load_preferences(pid, "snapshot-two", "results") == {}
    path.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    path.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        save_preferences(pid, "snapshot-two", "results", {"part": "test"})
    assert outside.read_text() == "{}"


def test_results_restore_selected_unit_and_role_in_fresh_session(sensor):
    pid = sensor.project["project_id"]
    record = train(sensor)
    snapshot = project_snapshot(pid)
    sid = snapshot["snapshot_id"]
    before = (seal(snapshot["dir"]), seal(project_store().run_path(pid, record["run_id"])), project_store().get(pid))
    save_preferences(pid, sid, "results", {"part": "test", "units": {"test": "test-1"}})
    page = app(pid, "Results")
    assert not page.exception and not page.error
    assert page.selectbox(key=f"long_unit:{pid}").value == "test-1"
    page.selectbox(key=f"long_unit:{pid}").select("test-0").run()
    reopened = app(pid, "Results")
    assert reopened.selectbox(key=f"long_unit:{pid}").value == "test-0"
    reopened.selectbox(key=f"long_part:{pid}").select("validation").run()
    validation = app(pid, "Results")
    assert not validation.exception and not validation.error
    assert validation.selectbox(key=f"long_part:{pid}").value == "validation"
    assert validation.selectbox(key=f"long_unit:{pid}").value.startswith("validation-")
    validation.selectbox(key=f"long_part:{pid}").select("test").run()
    assert validation.selectbox(key=f"long_unit:{pid}").value == "test-0"
    save_preferences(pid, sid, "results", {"units": {"test": "removed-unit"}})
    fallback = app(pid, "Results")
    assert not fallback.exception and not fallback.error
    assert fallback.selectbox(key=f"long_unit:{pid}").value == "test-0"
    assert (seal(snapshot["dir"]), seal(project_store().run_path(pid, record["run_id"])), project_store().get(pid)) == before


def test_demo_source_units_and_machine_restore_in_fresh_session(sensor):
    pid = sensor.project["project_id"]
    train(sensor)
    sid = project_snapshot(pid)["snapshot_id"]
    save_preferences(pid, sid, "live_monitor", {"source": "Demo feed", "demo_units": ["test-1", "validation-0"]})
    page = app(pid, "Live monitor")
    assert not page.exception and not page.error
    assert page.radio(key=f"live_source:{pid}").value == "Demo feed"
    assert page.multiselect(key=f"live_demo_units:{pid}").value == ["test-1", "validation-0"]
    page.multiselect(key=f"live_demo_units:{pid}").set_value([]).run()
    reopened = app(pid, "Live monitor")
    assert reopened.multiselect(key=f"live_demo_units:{pid}").value == []
    assert next(button for button in reopened.button if button.label == "Start demo feed").disabled
    reopened.multiselect(key=f"live_demo_units:{pid}").set_value(["test-0", "test-1"]).run()
    demo.start_demo(pid, ["test-0", "test-1"])
    demo.advance_demo(pid, 60)
    # An active feed is authoritative even when a saved display choice is stale.
    save_preferences(pid, sid, "live_monitor", {"demo_units": ["validation-0"]})
    running = app(pid, "Live monitor")
    assert not running.exception and not running.error
    assert running.multiselect(key=f"live_demo_units:{pid}").value == ["test-0", "test-1"]
    assert running.multiselect(key=f"live_demo_units:{pid}").disabled
    select_machine_row(running, running.dataframe[0].value.Machine.tolist().index("test-1"), cell=True)
    detail = app(pid, "Live monitor")
    assert detail.session_state[f"live_pick:{pid}"] == "test-1"
    demo.stop_demo(pid)
