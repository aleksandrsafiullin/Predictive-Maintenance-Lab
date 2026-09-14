"""The operational explorer has one clock and chooses a ready anatomical run."""
from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from pdm.paths import project_root
from pdm.visualization.explorer import EXPLORER_DISCLAIMER, WORKER_BUSY_MESSAGE


@pytest.fixture
def operational_screen(monkeypatch, tmp_path, tiny_bearing_tables):
    import streamlit as st

    from pdm.visualization import simulation_ui

    features, units = tiny_bearing_tables
    unit_ids = units["unit_id"].astype(str).tolist()
    bundle = {
        "features": features, "units": units,
        "split": {"train": unit_ids[:1], "validation": [], "test": unit_ids[1:]},
        "report": {}, "dataset_version": "test", "fingerprint": {"dataset_version": "test"},
        "dir": None,
    }
    rows = []
    for name, ready in [("newer_unprepared", False), ("anatomical_ready", True)]:
        rdir = tmp_path / name
        (rdir / "connectome").mkdir(parents=True)
        (rdir / "connectome" / "provenance.json").write_text(json.dumps({
            "sampling_method": "seeded_bfs_soma_xyz" if ready else "seeded_bfs",
        }))
        if ready:
            (rdir / "interval_profile.json").write_text(json.dumps({"version": 1, "ready": True}))
        rows.append({
            "run_id": name, "has_best": True, "architecture": "fly_connectome_reservoir",
            "graph_mode": "real_connectome", "is_synthetic": False,
        })
    captured = []

    def simulate(dataset_id, rdir, uid, data):
        captured.append((rdir.name, uid))
        st.button("Start test run")

    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: True)
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.replay.bind_replay_to_run", lambda *args: {**bundle, "current_fingerprint": bundle["fingerprint"]})
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds: rows)
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: tmp_path / rid)
    monkeypatch.setattr("pdm.visualization.explorer.load_scene_from_run", lambda p: {
        "graph_mode": "real_connectome", "is_synthetic": False, "nodes": [],
    })
    monkeypatch.setattr(simulation_ui, "render_equipment_simulation", simulate)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)
    monkeypatch.setattr("pdm.worker.read_status", lambda: {})
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=20)
    at.session_state["screen_selection"] = "Neural Activity Explorer"
    return at, captured, unit_ids, rows


def test_operational_explorer_direct_ready_run_and_test_unit(operational_screen):
    at, captured, unit_ids, _ = operational_screen
    at.session_state["_replay_view"] = {"run_id": "newer_unprepared"}
    at.run()
    assert not at.exception
    assert captured == [("anatomical_ready", unit_ids[1])]
    assert not any(r.label == "Mode" for r in at.radio)
    assert [b.label for b in at.button] == ["Start test run"]
    assert EXPLORER_DISCLAIMER in "\n".join(str(c.value) for c in at.caption)


def test_operational_explorer_busy_prevents_simulation(operational_screen, monkeypatch):
    at, captured, _, _ = operational_screen
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: True)
    monkeypatch.setattr("pdm.visualization.live.load_live_activity", lambda: None)
    at.run()
    assert not at.exception
    assert captured == []
    assert WORKER_BUSY_MESSAGE in "\n".join(str(i.value) for i in at.info)


def test_operational_explorer_rejects_nonreservoir(operational_screen):
    at, captured, _, rows = operational_screen
    for row in rows:
        row["architecture"] = "gru"
    at.run()
    assert not at.exception
    assert captured == []
    assert not at.selectbox
    assert any("reservoir" in str(w.value) for w in at.warning)


def test_changing_equipment_pauses_existing_clock(operational_screen):
    at, _, unit_ids, _ = operational_screen
    at.run()
    key = "equipment_sim:previous:unit"
    at.session_state[key] = {"playing": True}
    next(s for s in at.selectbox if s.label == "Unit").set_value(unit_ids[0]).run()
    assert not at.exception
    assert at.session_state[key]["playing"] is False


def test_loaded_data_mismatch_stops_before_inference(operational_screen, monkeypatch):
    from pdm.evaluate import IncompatibleDataError

    at, captured, _, _ = operational_screen

    def mismatch(*args):
        raise IncompatibleDataError(["features_hash"])

    monkeypatch.setattr("pdm.replay.bind_replay_to_run", mismatch)
    at.run()
    assert not at.exception
    assert captured == []
    assert any("features_hash" in str(error.value) for error in at.error)


@pytest.mark.parametrize("profile", [
    {"state_mode": "window", "warmup_measurements": 20},
    {"state_mode": "continuous", "warmup_measurements": 10},
])
def test_interval_profile_must_match_saved_clock(profile):
    from pdm.visualization.simulation_ui import _validate_simulation_profile

    with pytest.raises(ValueError):
        _validate_simulation_profile(profile, 20)
