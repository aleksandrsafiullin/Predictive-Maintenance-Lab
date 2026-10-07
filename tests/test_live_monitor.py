"""Live and Results share saved models, causal histories and bounded Calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm import corridor_calibration as cal
from pdm import live_monitor as lm
from pdm import live_simulator as demo
from pdm import long_forecast_run as runs
from pdm.live_monitor_ui import live_figure
from pdm.long_forecast_data import read_part, source_for
from pdm.project_snapshot import project_snapshot
from pdm.projects import project_store
from tests.project_contract import make_contract_snapshot
from tests.test_sensor_import_view import app, original_import, seal
from tests.test_sensor_import_view import release as sensor_release


@pytest.fixture
def sensor(tmp_path, monkeypatch):
    monkeypatch.setenv("PDM_LIVE_ROOT", str(tmp_path / "live"))
    release = sensor_release.__wrapped__(tmp_path, monkeypatch)
    original_import(release)
    return release


def train(sensor, engine="gru"):
    return runs.train(
        sensor.project["project_id"], engine, overrides=dict(epochs=1, hidden_size=8, threads=1)
    )


def test_csv_reader_aliases_dates_dedup_and_declared_gaps(tmp_path):
    pd.DataFrame(
        dict(
            unit_id=["m", "m", "m", None],
            timestamp_s=[0, 60, 60, 120],
            signal=[0.3, 0.4, 0.5, 0.9],
            gap_before=[True, False, True, False],
        )
    ).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame(dict(unit_id=["n"], timestamp=["2026-10-01T00:00:00Z"], value=[0.2])).to_csv(
        tmp_path / "b.csv", index=False
    )
    (tmp_path / "broken.csv").write_text("")
    rows = lm.read_live_folder(tmp_path)
    assert rows[rows.unit_id.eq("m")].signal.tolist() == [0.3, 0.5]
    assert rows[rows.unit_id.eq("m")].gap_before.tolist() == [True, True]
    assert rows[rows.unit_id.eq("n")].timestamp_s.iloc[0] == pytest.approx(1790812800)
    assert lm.read_live_folder(tmp_path / "missing").empty


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_live_matches_results_all_history_and_calibration(sensor, engine):
    pid = sensor.project["project_id"]
    record = train(sensor, engine)
    calibrated = cal.calibrate(pid, record["run_id"])
    source = source_for(pid)
    frame = read_part(source, "test")
    uid = frame.unit_id.iloc[0]
    prefix = frame.loc[frame.unit_id.eq(uid)].iloc[:100].copy()
    origin = float(prefix.timestamp_s.iloc[-1])
    before = (
        seal(project_snapshot(pid)["dir"]),
        seal(project_store().run_path(pid, record["run_id"])),
        project_store().get(pid),
    )
    model = lm.LiveModel(pid, record["run_id"])
    for use_calibration in (True, False):
        live = model.assess(prefix, use_calibration=use_calibration)
        saved = runs.replay(pid, record["run_id"], uid, origin, use_calibration=use_calibration)
        assert live["status"] == "available"
        assert live["history_observations"] == saved["history_observations"] == 100
        assert live["input_hash"] == saved["input_hash"]
        np.testing.assert_array_equal(
            [[p["lower"], p["value"], p["upper"]] for p in live["points"]], saved["outputs"]
        )
        np.testing.assert_array_equal([p["target_time_s"] for p in live["points"]], saved["times"])
        if use_calibration:
            assert live["calibration"]["calibration_id"] == calibrated["calibration_id"]
            widths = np.diff(saved["outputs"][:, [0, 2]], axis=1).ravel() / saved["outputs"][:, 1]
            assert np.all(widths >= 0.20 - 1e-12) and np.all(widths <= 0.45 + 1e-12)
        else:
            assert live["calibration"] is None
    # A previously unseen live machine is scored with the same received values.
    external = prefix.assign(unit_id="external-pump")
    assert model.assess(external)["points"] == model.assess(prefix)["points"]
    assert (
        seal(project_snapshot(pid)["dir"]),
        seal(project_store().run_path(pid, record["run_id"])),
        project_store().get(pid),
    ) == before
    labels = [trace.name for trace in live_figure(model.assess(prefix), model.schema, "dark").data]
    assert "Trend center" in labels and "Trend corridor" in labels


def test_future_rows_excluded_and_gap_requires_new_continuous_history(sensor):
    pid = sensor.project["project_id"]
    record = train(sensor)
    unit = read_part(source_for(pid), "test").query('unit_id == "test-0"').copy()
    origin = 99 * 60
    reference = runs.forecast_observations(pid, record["run_id"], unit, origin=origin)
    changed = unit.copy()
    changed.loc[changed.timestamp_s.gt(origin), "signal"] = 999
    actual = runs.forecast_observations(pid, record["run_id"], changed, origin=origin)
    np.testing.assert_array_equal(reference["outputs"], actual["outputs"])
    assert reference["input_hash"] == actual["input_hash"]
    broken = unit.copy()
    broken.loc[broken.index[70:], "timestamp_s"] += 120
    model = lm.LiveModel(pid, record["run_id"])
    assert model.assess(broken.iloc[:100])["status"] == "unavailable"
    recovered = model.assess(broken.iloc[:135])
    assert recovered["status"] == "available" and recovered["history_observations"] == 65
    declared = unit.copy()
    declared["gap_before"] = False
    declared.loc[declared.index[70], "gap_before"] = True
    assert model.assess(declared.iloc[:100])["status"] == "unavailable"


def test_demo_progressive_mapped_snapshot_and_isolated_clear(sensor, tmp_path):
    pid = sensor.project["project_id"]
    watched = tmp_path / "real"
    watched.mkdir()
    sensor_file = watched / "sim_pump.csv"
    sensor_file.write_text("real sensor file")
    lm.save_config(pid, dict(folder=str(watched)))
    allowed = demo.demo_units(pid)
    assert "test-0" in allowed and "train-0" not in allowed and "calibration-0" not in allowed
    state = demo.start_demo(pid, ["test-0"])
    assert state["snapshot_id"] == project_snapshot(pid)["snapshot_id"]
    assert lm.read_live_folder(state["folder"]).empty
    state = demo.advance_demo(pid, 30)
    observed = lm.read_live_folder(state["folder"])
    expected = read_part(source_for(pid), "test").query('unit_id == "test-0"').iloc[:30]
    np.testing.assert_array_equal(observed.signal, expected.signal)
    assert observed.timestamp_s.max() == 29 * 60
    demo.stop_demo(pid)
    assert demo.advance_demo(pid, 30)["units"]["test-0"]["written"] == 30
    demo.clear_demo(pid)
    assert not Path(state["folder"]).exists()
    assert sensor_file.read_text() == "real sensor file"
    with pytest.raises(ValueError, match="Testing or Validation"):
        demo.start_demo(pid, ["train-0"])


def test_main_navigation_sensor_page_and_stable_machine_selection(sensor, tmp_path):
    pid = sensor.project["project_id"]
    record = train(sensor)
    folder = tmp_path / "feed"
    folder.mkdir()
    rows = pd.concat(
        [
            pd.DataFrame(dict(unit_id="green", timestamp_s=np.arange(100) * 60, signal=0.3)),
            pd.DataFrame(dict(unit_id="red", timestamp_s=np.arange(100) * 60, signal=0.9)),
        ]
    )
    rows.to_csv(folder / "machines.csv", index=False)
    lm.save_config(pid, dict(folder=str(folder)))
    page = app(pid, "Live monitor")
    assert not page.exception and not page.error
    assert not next(b for b in page.button if b.label == "Live monitor").disabled
    assert next(s for s in page.selectbox if s.label == "Saved model").value == record["run_id"]
    assert {m.label: m.value for m in page.metric}["Machines"] == "2"
    page.selectbox(key=f"live_pick:{pid}").set_value("green").run()
    rows = pd.concat(
        [rows, pd.DataFrame(dict(unit_id="new-red", timestamp_s=np.arange(100) * 60, signal=1.1))]
    )
    rows.to_csv(folder / "machines.csv", index=False)
    page.run()
    assert page.selectbox(key=f"live_pick:{pid}").value == "green"
    assert page.dataframe[0].value.Machine.tolist()[-1] == "green"
    page.selectbox(key=f"live_pick:{pid}").set_value("red").run()
    rows.loc[rows.unit_id.eq("red"), "signal"] = .25
    rows.to_csv(folder / "machines.csv", index=False)
    page.run()
    assert page.selectbox(key=f"live_pick:{pid}").value == "red"
    # Calibration saved after opening the screen invalidates cached interval predictions.
    before = json.loads(page.get("plotly_chart")[0].proto.spec)
    cal.save_settings(
        pid, project_snapshot(pid)["snapshot_id"], dict(min_width=0.25, max_width=0.25)
    )
    cal.calibrate(pid, record["run_id"], dict(min_width=.25, max_width=.25))
    page.run()
    after = json.loads(page.get("plotly_chart")[0].proto.spec)
    assert not page.exception and not page.error
    assert before != after
    assert any("full width 25.0%" in c.value for c in page.caption)


def test_common_screen_and_parity_for_original_snapshot(tmp_path, monkeypatch):
    from pdm.signal_inference import forecast_prefix
    from pdm.signal_training import train_signal_run

    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("PDM_LIVE_ROOT", str(tmp_path / "live"))
    store, project, snapshot = make_contract_snapshot(tmp_path / "projects")
    pid = project["project_id"]
    record = train_signal_run(
        pid,
        snapshot["snapshot_id"],
        "gru",
        dict(history_length=4, horizons_s=[10.0, 20.0], epochs=1, hidden_size=8),
    )
    store.update(pid, selected_run_id=record["run_id"])
    snapshot = project_snapshot(pid)
    model = lm.LiveModel(pid, record["run_id"])
    uid = snapshot["split"]["test"][0]
    prefix = snapshot["features"].query("unit_id == @uid").iloc[:6]
    live = model.assess(prefix)
    saved = forecast_prefix(pid, record["run_id"], uid, float(prefix.timestamp_s.iloc[-1]))
    assert live["points"] == saved["points"]
    folder = tmp_path / "feed"
    folder.mkdir()
    prefix.to_csv(folder / "machine.csv", index=False)
    lm.save_config(pid, dict(folder=str(folder)))
    page = app(pid, "Live monitor")
    assert not page.exception and not page.error
    assert next(m for m in page.metric if m.label == "Machines").value == "1"


def test_demo_controls_work_through_common_screen(sensor):
    pid = sensor.project["project_id"]
    train(sensor)
    page = app(pid, "Live monitor")
    page.radio(key=f"live_source:{pid}").set_value("Demo feed").run()
    assert not page.exception and not page.error
    next(b for b in page.button if b.label == "Start demo feed").click().run()
    assert demo.demo_status(pid)["state"] == "running"
    assert next(m for m in page.metric if m.label == "Machines").value == "3"
    next(b for b in page.button if b.label == "Stop demo feed").click().run()
    assert demo.demo_status(pid)["state"] == "stopped"
    next(b for b in page.button if b.label == "Clear demo data").click().run()
    assert demo.demo_status(pid)["state"] == "idle"


def test_no_model_project_still_has_the_common_screen(sensor):
    page = app(sensor.project["project_id"], "Live monitor")
    assert not page.exception and not page.error
    assert any("No saved model yet" in m.value for m in page.subheader)
    assert not next(b for b in page.button if b.label == "Live monitor").disabled
