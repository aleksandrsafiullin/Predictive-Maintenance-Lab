from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from pdm import live_monitor as lm
from tests.project_contract import make_contract_snapshot


@pytest.fixture
def trained(tmp_path, monkeypatch):
    """Contract fixture project with one small saved GRU and an isolated live root."""
    from pdm.signal_training import train_signal_run

    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    monkeypatch.setenv("PDM_LIVE_ROOT", str(tmp_path / "live"))
    monkeypatch.delenv(lm.TEAMS_ENV, raising=False)
    store, project, snapshot = make_contract_snapshot(root)
    pid = project["project_id"]
    run = train_signal_run(pid, snapshot["snapshot_id"], "gru",
                           {"history_length": 4, "horizons_s": [10.0, 20.0], "epochs": 1, "hidden_size": 8})
    store.update(pid, selected_run_id=run["run_id"])
    from pdm.data.project_prepare import load_snapshot

    return pid, run["run_id"], load_snapshot(pid, snapshot["snapshot_id"])


def _write(folder, name, rows, signal="vibration"):
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["unit_id", "timestamp_s", signal]).to_csv(folder / name, index=False)


def test_read_live_folder_normalises_aliases_dates_and_duplicates(tmp_path):
    folder = tmp_path / "in"
    _write(folder, "a.csv", [["m1", 0, 1.0], ["m1", 10, 2.0], ["m1", 10, 2.5]])
    _write(folder, "b.csv", [["m2", 0, 3.0]], signal="signal")
    pd.DataFrame({"unit_id": ["m3", "m3"], "timestamp": ["2026-10-01T00:00:00Z", "2026-10-01T00:01:00Z"],
                  "value": [4.0, "bad"]}).to_csv(folder / "c.csv", index=False)
    (folder / "notes.txt").write_text("ignored")
    (folder / "broken.csv").write_text("")
    data = lm.read_live_folder(folder, "vibration")
    assert list(data.columns) == ["unit_id", "timestamp_s", "signal"]
    assert data[data.unit_id == "m1"].signal.tolist() == [1.0, 2.5]  # later duplicate wins
    assert data[data.unit_id == "m2"].signal.tolist() == [3.0]
    m3 = data[data.unit_id == "m3"]
    assert len(m3) == 1 and m3.timestamp_s.iloc[0] == pytest.approx(1790812800.0)
    assert lm.read_live_folder(tmp_path / "missing").empty


def test_gaps_break_history_on_clock_jumps():
    unit = pd.DataFrame({"unit_id": "m", "timestamp_s": [0.0, 10.0, 20.0, 50.0, 60.0], "signal": 1.0})
    assert lm.mark_gaps(unit, 10.0).gap_before.tolist() == [True, False, False, True, False]
    assert lm.mark_gaps(unit, None).gap_before.tolist() == [True, False, False, False, False]


def test_escalations_alert_once_per_level_and_rearm_after_green():
    def results(zone):
        return [{"unit_id": "m", "zone": zone, "current": 1.0, "as_of_s": 0.0, "red_in_s": None}]

    state = {}
    seq = ["green", "yellow", "yellow", "red", "red", "yellow", "green", "red", "unknown"]
    fired = [[e["zone"] for e in lm.detect_escalations(results(z), state)] for z in seq]
    assert fired == [[], ["yellow"], [], ["red"], [], [], [], ["red"], []]


def test_live_forecast_matches_results_screen(trained):
    from pdm.signal_inference import forecast_prefix

    pid, run_id, snapshot = trained
    model = lm.LiveModel(pid, run_id)
    uid = snapshot["split"]["test"][0]
    unit = snapshot["features"][snapshot["features"].unit_id == uid].sort_values("timestamp_s")
    for k in (2, 6, len(unit)):
        prefix = unit.iloc[:k][["unit_id", "timestamp_s", "signal"]]
        live = model.assess(prefix)
        reference = forecast_prefix(pid, run_id, uid, float(prefix.timestamp_s.iloc[-1]))
        assert [p["value"] for p in live["points"]] == pytest.approx([p["value"] for p in reference["points"]])
        assert live["crossing"] == reference["crossing"]
        if k < 4:
            assert live["status"] == "unavailable" and "Collecting history" in live["reason"]
    # Adding later rows never changes an earlier forecast.
    early = model.assess(unit.iloc[:6])
    assert [p["target_time_s"] for p in early["points"]] == [unit.timestamp_s.iloc[5] + 10, unit.timestamp_s.iloc[5] + 20]


def test_poll_sends_teams_cards_only_when_enabled(trained):
    pid, run_id, snapshot = trained
    folder = lm.default_folder(pid)
    _write(folder, "m.csv", [["m1", t, v] for t, v in zip(range(0, 100, 10), np.linspace(0.1, 0.9, 10))])
    sent = []
    base = {"run_id": run_id, "folder": str(folder), "teams_webhook_url": "https://example.invalid/hook"}

    cycle = lm.poll_once(pid, "Demo", {**base, "alerts_enabled": False},
                         sender=lambda url, card: sent.append(card) or (True, "ok"))
    assert cycle["results"][0]["zone"] == "red" and not sent
    assert [a["delivery"] for a in lm.recent_alerts(pid)] == ["not sent"]

    lm.reset_alerts(pid)
    lm.poll_once(pid, "Demo", {**base, "alerts_enabled": True},
                 sender=lambda url, card: sent.append((url, card)) or (True, "Delivered (202)"))
    assert len(sent) == 1 and sent[0][0] == base["teams_webhook_url"]
    card = sent[0][1]["attachments"][0]
    assert card["contentType"] == "application/vnd.microsoft.card.adaptive"
    facts = {f["title"]: f["value"] for f in card["content"]["body"][1]["facts"]}
    assert facts["Machine"] == "m1" and facts["Change"] == "green → red"
    assert lm.recent_alerts(pid)[0]["delivery"] == "sent"
    # The same state again does not alert twice.
    lm.poll_once(pid, "Demo", {**base, "alerts_enabled": True}, sender=lambda *a: sent.append(a) or (True, ""))
    assert len(sent) == 1


def test_environment_webhook_wins(monkeypatch):
    monkeypatch.setenv(lm.TEAMS_ENV, "https://env.invalid/hook")
    assert lm.webhook_url({"teams_webhook_url": "https://file.invalid"}) == "https://env.invalid/hook"


def test_post_teams_reports_failures(monkeypatch):
    import requests

    class Response:
        status_code = 400
        text = "Bad payload"

    monkeypatch.setattr(requests, "post", lambda *a, **k: Response())
    assert lm.post_teams("https://x.invalid", {}) == (False, "Teams answered 400: Bad payload")
    assert lm.post_teams("", {})[0] is False


def test_demo_feed_replays_recorded_rows_and_clears_only_its_files(trained, tmp_path):
    from pdm.live_simulator import clear_demo_files, run_simulator, simulator_status

    pid, run_id, snapshot = trained
    folder = tmp_path / "feed"
    folder.mkdir()
    (folder / "real_machine.csv").write_text("unit_id,timestamp_s,vibration\nx,0,1\n")
    units = snapshot["split"]["test"]
    status = run_simulator(pid, folder, units, tick_s=0.0, rows_per_tick=3, signal_column="vibration")
    assert status["state"] == "finished" and simulator_status(pid)["state"] == "finished"
    data = lm.read_live_folder(folder, "vibration")
    for uid in units:
        expected = snapshot["features"][snapshot["features"].unit_id == uid].sort_values("timestamp_s")
        got = data[data.unit_id == uid]
        assert got.signal.tolist() == pytest.approx(expected.signal.tolist())
    assert clear_demo_files(folder) == len(units)
    assert [p.name for p in folder.iterdir()] == ["real_machine.csv"]


def test_live_monitor_screen_renders_fleet(trained):
    from streamlit.testing.v1 import AppTest

    pid, run_id, snapshot = trained
    lm.save_config(pid, {"run_id": run_id})
    folder = lm.default_folder(pid)
    _write(folder, "m.csv", [["m1", t, 0.1] for t in range(0, 100, 10)] + [["m2", t, 0.9] for t in range(0, 30, 10)])
    script = f"""
from pdm.data.project_prepare import load_snapshot
from pdm.live_monitor_ui import render_live_monitor
render_live_monitor({pid!r}, "Demo", load_snapshot({pid!r}), "light")
"""
    at = AppTest.from_string(script, default_timeout=60)
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Machines"] == "2" and metrics["🔴 Fix ASAP"] == "1" and metrics["🟢 Normal"] == "1"
    table = at.dataframe[0].value
    assert table.Machine.tolist() == ["m2", "m1"]  # most urgent first
    picker = at.selectbox(key=f"live_pick:{pid}")
    assert picker.options == ["🟢 m1", "🔴 m2"] and picker.value == "m2"

    def charted():
        figure = json.loads(at.get("plotly_chart")[0].proto.spec)
        return figure["data"][0]["y"][-1]

    assert charted() == pytest.approx(0.9)
    picker.set_value("m1")
    at.run()
    assert charted() == pytest.approx(0.1) and at.selectbox(key=f"live_pick:{pid}").value == "m1"
    # A machine turning urgent reorders the table but keeps the chosen machine charted.
    _write(folder, "m.csv", [["m1", t, 0.1] for t in range(0, 100, 10)] + [["m2", t, 0.9] for t in range(0, 30, 10)]
           + [["m0", t, 0.95] for t in range(0, 30, 10)])
    at.run()
    assert at.dataframe[0].value.Machine.tolist()[-1] == "m1"
    assert at.selectbox(key=f"live_pick:{pid}").value == "m1" and charted() == pytest.approx(0.1)
    assert json.loads(json.dumps(lm.load_config(pid)))["run_id"] == run_id
