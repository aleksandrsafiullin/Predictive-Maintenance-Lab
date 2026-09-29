from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from pdm.projects import ProjectStore, project_store


def test_persistent_independent_projects_and_safe_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    store = project_store()
    first = store.create("Pump A", "generic_sensor_csv")
    second = store.create("Pump B", "generic_sensor_csv")
    assert first["project_id"] != second["project_id"]
    assert store.snapshot_path(first["project_id"], "snap-1") != store.snapshot_path(second["project_id"], "snap-1")
    assert store.run_path(first["project_id"], "run-1") != store.run_path(second["project_id"], "run-1")
    assert [p["name"] for p in ProjectStore().list()] == ["Pump A", "Pump B"]
    with pytest.raises(ValueError, match="already exists"):
        store.create("pump a", "generic_sensor_csv")
    with pytest.raises(ValueError, match="Invalid"):
        store.snapshot_path(first["project_id"], "../escape")
    with pytest.raises(ValueError, match="Immutable"):
        store.update(first["project_id"], source_kind="hse_filters")


def test_symlink_and_archive_project_isolation(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    first = store.create("First", "generic_sensor_csv")
    second = store.create("Second", "generic_sensor_csv")
    source = store.project_path(first["project_id"]) / "source"
    source.rmdir()
    source.symlink_to(store.project_path(second["project_id"]), target_is_directory=True)
    with pytest.raises(ValueError, match="Symlinks"):
        store._owned_path(first["project_id"], "source", "x")
    source.unlink()
    source.mkdir()
    with pytest.raises(RuntimeError, match="queued or live"):
        store.archive(first["project_id"], active_job={"project_id": first["project_id"], "status": "queued"})
    receipt = store.archive(second["project_id"], active_job={"project_id": first["project_id"], "status": "running"})
    assert Path(receipt["archive_path"]).is_dir()
    assert json.loads((Path(receipt["archive_path"]) / "archive_receipt.json").read_text())["project_record"]["name"] == "Second"
    assert store.get(first["project_id"])["name"] == "First"
    with pytest.raises(KeyError):
        store.get(second["project_id"])


def test_run_selection_requires_completed_bound_signal_manifest(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Test", "generic_sensor_csv")
    pid = project["project_id"]
    store.update(pid, active_snapshot_id="snapshot1")
    run_dir = store.run_path(pid, "run1")
    run_dir.mkdir()
    manifest = {"project_id": pid, "snapshot_id": "snapshot1", "run_id": "run1",
                "task": "signal_forecast", "status": "completed"}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))
    assert store.update(pid, selected_run_id="run1")["selected_run_id"] == "run1"
    with pytest.raises(ValueError, match="Selected run"):
        store.update(pid, active_snapshot_id="snapshot2")
    store.update(pid, active_snapshot_id="snapshot2", selected_run_id=None)
    with pytest.raises(ValueError, match="Selected run"):
        store.update(pid, selected_run_id="run1")


def test_legacy_link_archive_preserves_external_files_and_tombstones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    runs = tmp_path / "runs"
    for root in (raw, processed, runs):
        root.mkdir()
    bearing_raw = raw / "bearings"
    bearing_raw.mkdir()
    (bearing_raw / "1.csv").write_text("x\n1\n")
    monkeypatch.setattr("pdm.projects.data_raw", lambda: raw)
    monkeypatch.setattr("pdm.projects.data_processed", lambda: processed)
    monkeypatch.setattr("pdm.projects.runs_root", lambda: runs)
    store = ProjectStore(tmp_path / "projects")
    links = store.register_legacy()
    assert [p["project_id"] for p in links] == ["legacy-bearings"]
    assert store.register_legacy() == links
    assert store.snapshot_path("legacy-bearings", "new") == store.root / "legacy-bearings" / "snapshots" / "new"
    receipt = store.archive("legacy-bearings")
    assert receipt["external_legacy_preserved"] is True
    assert bearing_raw.joinpath("1.csv").exists()
    assert Path(receipt["archive_path"]).is_dir()
    assert store.register_legacy() == []


def test_archive_waits_for_terminal_worker_process_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("Working", "generic_sensor_csv")["project_id"]
    monkeypatch.setattr("pdm.worker.read_status", lambda: {
        "project_id": pid, "job_id": "job-a", "status": "completed",
        "pid": 4321, "updated_at": time.time(),
    })
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)
    monkeypatch.setattr("pdm.worker._pid_exists", lambda _pid: True)
    with pytest.raises(RuntimeError, match="queued or live"):
        store.archive(pid)
    monkeypatch.setattr("pdm.worker._pid_exists", lambda _pid: False)
    assert store.archive(pid)["project_id"] == pid
