from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest


def test_model_report_defaults_to_future_red_without_old_rul_training_launch():
    from pdm.paths import project_root

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.session_state["screen_selection"] = "Model Report"
    at.session_state["report_view"] = "Future-red entry"
    at.run()

    assert not at.exception
    assert any("Future-red entry model report" in str(header.value) for header in at.header)
    assert not any(button.label == "Train full MaleCNS from scratch" for button in at.button)


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
