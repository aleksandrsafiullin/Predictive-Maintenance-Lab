from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


def test_fly_training_screen_dispatches_complete_population_without_old_controls(monkeypatch):
    import pdm.app as app

    monkeypatch.setattr("pdm.lab_ui.training_overview", lambda *_: True)
    jobs = []
    monkeypatch.setattr(app, "processed_ready", lambda _: True)
    monkeypatch.setattr(app, "load_processed", lambda _: {"report": {}})
    monkeypatch.setattr(app, "list_runs", lambda _: [])
    monkeypatch.setattr(app, "worker_alive", lambda: False)
    monkeypatch.setattr(app, "read_status", lambda: {"status": "idle"})
    monkeypatch.setattr(app, "spawn_worker", jobs.append)
    at = AppTest.from_string('from pdm.app import screen_train\nscreen_train("bearings")', default_timeout=15).run()
    next(s for s in at.selectbox if s.label == "Architecture").set_value("fly_connectome_reservoir")
    at.run()
    assert not at.exception
    assert not at.number_input
    assert not any("Smoke" in r.options for r in at.radio)
    next(b for b in at.button if b.label == "Train full MaleCNS from scratch").click()
    at.run()
    assert not at.exception
    assert jobs == [{"kind": "train_full_cns", "dataset_id": "bearings"}]


@pytest.mark.parametrize("cancel", [False, True])
def test_full_cns_worker_owns_status_and_cancellation(monkeypatch, tmp_path, cancel):
    import pdm.connectome.morphology as morphology
    import pdm.train_full_cns as training
    import pdm.visualization.live as live
    import pdm.worker as worker

    monkeypatch.setattr(worker, "worker_dir", lambda: tmp_path)
    monkeypatch.setattr(live, "clear_live_activity", lambda: None)
    monkeypatch.setattr(morphology, "morphology_directory", lambda: tmp_path)
    (tmp_path / "manifest.json").write_text("{}")
    calls = []

    def train(argv):
        calls.append(argv)
        if cancel:
            raise InterruptedError("cancelled")

    monkeypatch.setattr(training, "main", train)
    worker.run_job({"kind": "train_full_cns", "dataset_id": "bearings"})
    assert calls == [[]]
    assert worker.read_status()["status"] == ("cancelled" if cancel else "completed")
    assert not worker.pid_path().exists()
