from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

from pdm.cli import spawn_worker
from pdm.io_util import atomic_write_json
from pdm.projects import ProjectStore
from pdm.worker import (
    _cleanup_project_uploads,
    _project_stopped,
    read_status,
    request_stop,
    run_job,
    status_for_project,
    status_path,
    stop_path,
)


@pytest.fixture
def worker_store(tmp_path, monkeypatch):
    root = tmp_path / "projects"
    worker_root = tmp_path / "worker"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    monkeypatch.setattr("pdm.worker.worker_dir", lambda: worker_root)
    worker_root.mkdir()
    store = ProjectStore(root)
    one = store.create("One", "generic_sensor_csv")
    two = store.create("Two", "generic_sensor_csv")
    return store, one, two


def test_project_launch_queue_stop_identity_and_archive_isolation(worker_store, monkeypatch):
    store, one, two = worker_store
    monkeypatch.setattr("pdm.cli._launch_worker_process", lambda: SimpleNamespace(pid=os.getpid()))
    job = {"kind": "project_import", "job_id": "job-one", "project_id": one["project_id"],
           "source": {"primary": {"mode": "folder", "path": "/unused"}}}
    spawn_worker(job)
    status = status_for_project(one["project_id"])
    assert status["status"] == "queued" and status["job_id"] == "job-one"
    assert status_for_project(two["project_id"]) == {"status": "not_ready"}
    with pytest.raises(ValueError, match="match"):
        request_stop(expected_job_id="wrong-job")
    request_stop(expected_job_id="job-one")
    assert _project_stopped("job-one") and not _project_stopped("wrong-job")
    with pytest.raises(RuntimeError, match="queued or live"):
        store.archive(one["project_id"])
    assert store.archive(two["project_id"])["project_id"] == two["project_id"]


def test_plain_stop_targets_active_project_job(worker_store):
    _, one, _ = worker_store
    atomic_write_json(status_path(), {"project_id": one["project_id"], "job_id": "active",
                                      "kind": "project_train", "status": "running", "pid": os.getpid(),
                                      "updated_at": time.time()})
    request_stop()
    assert _project_stopped("active")
    assert read_status()["pid"] == os.getpid()


def test_crashed_worker_status_reconciles_and_unblocks_next_launch(worker_store, monkeypatch):
    store, one, _ = worker_store
    atomic_write_json(status_path(), {"project_id": one["project_id"], "job_id": "crashed",
                                      "kind": "project_train", "status": "running", "pid": 99999999,
                                      "updated_at": time.time() - 20})
    assert read_status()["status"] == "failed"
    stop_path().unlink(missing_ok=True)
    monkeypatch.setattr("pdm.cli._launch_worker_process", lambda: SimpleNamespace(pid=os.getpid()))
    spawn_worker({"kind": "project_import", "project_id": one["project_id"],
                  "source": {"primary": {"mode": "folder", "path": "/unused"}}})
    assert status_for_project(one["project_id"])["status"] == "queued"
    with pytest.raises(RuntimeError, match="already running"):
        spawn_worker({"kind": "prepare", "dataset_id": "bearings"})
    with pytest.raises(RuntimeError, match="queued or live"):
        store.archive(one["project_id"])


def test_project_train_worker_terminal_status_carries_identity(worker_store, monkeypatch):
    _, one, _ = worker_store
    import pdm.signal_training as training

    def fake_train(pid, sid, engine, params, *, should_stop, status_cb):
        assert pid == one["project_id"] and sid == "snapshot-a" and engine == "gru"
        assert not should_stop()
        status_cb({"stage": "training", "progress": 0.5, "message": "Epoch 1"})
        return {"run_id": "run-a"}

    monkeypatch.setattr(training, "train_signal_run", fake_train)
    run_job({"kind": "project_train", "project_id": one["project_id"], "job_id": "job-a",
             "snapshot_id": "snapshot-a", "engine_id": "gru", "params": {}})
    state = status_for_project(one["project_id"])
    assert state["status"] == "completed" and state["run_id"] == "run-a"
    assert state["job_id"] == "job-a" and state["project_id"] == one["project_id"]
    assert {"kind", "stage", "progress", "message", "error"}.issubset(state)


def test_upload_cleanup_only_removes_this_projects_browser_stage(worker_store, tmp_path):
    store, one, two = worker_store
    own = store.project_path(one["project_id"]) / "uploads" / ("a" * 32) / "primary" / "sensor.csv"
    other = store.project_path(two["project_id"]) / "uploads" / ("b" * 32) / "primary" / "sensor.csv"
    for path in (own, other):
        path.parent.mkdir(parents=True)
        path.write_text("signal\n1\n", encoding="utf-8")
    external = tmp_path / "external.csv"
    external.write_text("keep", encoding="utf-8")
    job = {"project_id": one["project_id"], "source": {"primary": {"mode": "files", "files": [
        {"path": str(own), "relative_path": "sensor.csv"},
        {"path": str(external), "relative_path": "external.csv"},
    ]}}}
    _cleanup_project_uploads(job)
    assert own.exists() and other.exists() and external.exists()
    job["source"]["primary"]["files"].pop()
    _cleanup_project_uploads(job)
    assert not own.exists() and other.exists() and external.exists()
