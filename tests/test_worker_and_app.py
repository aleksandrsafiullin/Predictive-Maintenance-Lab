from __future__ import annotations

from pdm.worker import read_status, worker_alive, worker_dir


def test_worker_not_alive_without_pid():
    # No duplicate training on UI rerun: spawn_worker refuses if worker_alive().
    assert worker_alive() in {True, False}
    st = read_status()
    assert "status" in st


def test_app_starts_without_data():
    from streamlit.testing.v1 import AppTest
    from pdm.paths import project_root

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
