from __future__ import annotations

import json

import pandas as pd
from streamlit.testing.v1 import AppTest

from pdm.paths import project_root
from pdm.project_results_ui import (
    advance_play_state,
    first_recorded_red_after_now,
    replay_figure,
    visible_observations,
)
from pdm.ui_theme import TOKENS


def test_play_advances_each_tick_then_stays_paused():
    state = {"cursor": 0, "playing": True}
    assert advance_play_state(state, 5) == {"cursor": 1, "playing": True}
    state = advance_play_state(state, 5)
    state = advance_play_state(state, 5)
    state = advance_play_state(state, 5)
    assert state == {"cursor": 3, "playing": True}
    state["playing"] = False
    assert advance_play_state(state, 5) == {"cursor": 3, "playing": False}
    assert advance_play_state({"cursor": 4, "playing": True}, 5) == {"cursor": 4, "playing": False}


def test_visible_actual_stops_at_cursor_and_gap_breaks_line():
    frame = pd.DataFrame({"unit_id": ["A", "A", "A", "A", "B"],
                          "timestamp_s": [0, 1, 2, 3, 0], "signal": [-2, -1, 0, 1, 99],
                          "gap_before": [False, False, True, False, False]})
    visible = visible_observations(frame, "A", 2)
    assert list(visible["timestamp_s"]) == [0, 1, 2]
    result = {"as_of_s": 2, "observed_prefix": visible.to_dict("records"),
              "points": [{"target_time_s": 3, "value": -0.4}, {"target_time_s": 4, "value": None},
                         {"target_time_s": 5, "value": -2.2}],
              "thresholds": {"status": "available", "mode": "absolute", "direction": "below",
                             "yellow": -1, "red": -2},
              "crossing": {"status": "predicted", "time_s": 5}}
    fig = replay_figure(result, {"signal_label": "Pressure", "signal_unit": "Pa"})
    assert list(fig.data[0].x) == [0, 1, None, 2]
    assert list(fig.data[1].x) == [2, 3, 4, 5]
    assert list(fig.data[1].y) == [0, -0.4, None, -2.2]
    assert all(3 not in trace.x for trace in fig.data[:1])
    assert fig.layout.yaxis.title.text == "Pressure (Pa)"
    assert len(fig.layout.shapes) >= 3  # two alert zones, two levels, and Now
    light = replay_figure(result, {"signal_label": "Pressure", "signal_unit": "Pa"}, theme="light")
    tokens = TOKENS["light"]
    assert light.layout.paper_bgcolor == tokens["chart_bg"]
    assert light.layout.font.color == tokens["chart_text"]
    assert light.layout.xaxis.title.font.color == tokens["chart_text"]
    assert light.layout.yaxis.title.font.color == tokens["chart_text"]
    assert light.layout.shapes[-1].line.color == tokens["series_reference"]
    assert light.data[0].line.color == tokens["series_observed"]
    assert light.data[1].line.color == tokens["series_forecast"]


def test_fragment_play_advances_and_pause_holds():
    at = AppTest.from_file(str(project_root() / "tests" / "project_results_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    assert any('data-theme="light"' in str(markdown.value) for markdown in at.markdown)
    key = "project_play:project1:run1:unit1"
    next(button for button in at.button if button.label == "Play").click()
    at.run()
    assert not at.exception
    assert any('data-theme="light"' in str(markdown.value) for markdown in at.markdown)
    first = at.session_state[key]["cursor"]
    at.run()
    assert not at.exception
    assert any('data-theme="light"' in str(markdown.value) for markdown in at.markdown)
    second = at.session_state[key]["cursor"]
    assert second > first
    next(button for button in at.button if button.label == "Pause").click()
    at.run()
    assert not at.exception
    held = at.session_state[key]["cursor"]
    at.run()
    assert at.session_state[key]["cursor"] == held
    next(button for button in at.button if button.label == "Reset").click()
    at.run()
    assert not at.exception
    assert at.session_state[key]["cursor"] == 0
    assert at.session_state[key]["playing"] is False
    next(slider for slider in at.slider if slider.label == "Observation time").set_value(3)
    at.run()
    assert not at.exception
    assert at.session_state[key]["cursor"] == 3
    next(button for button in at.button if button.label == "Reset").click()
    at.run()
    assert not at.exception
    assert at.session_state[key]["cursor"] == 0


def test_light_theme_marker_survives_many_replay_ticks():
    at = AppTest.from_file(str(project_root() / "tests" / "project_results_harness.py"), default_timeout=15)
    at.run()
    key = "project_play:project1:run1:unit1"
    next(button for button in at.button if button.label == "Play").click()
    for _ in range(12):
        at.run()
        assert not at.exception
        assert any('data-theme="light"' in str(markdown.value) for markdown in at.markdown)
    assert at.session_state[key]["cursor"] >= 12


def test_future_overlay_is_optional_gap_safe_and_does_not_recompute_forecast():
    at = AppTest.from_file(str(project_root() / "tests" / "project_results_harness.py"), default_timeout=15).run()
    next(slider for slider in at.slider if slider.label == "Observation time").set_value(3)
    at.run()
    before = json.loads(at.get("plotly_chart")[0].proto.spec)
    calls = at.session_state["forecast_calls"]
    assert not any("Future actual" in t["name"] for t in before["data"])
    at.toggle[0].set_value(True).run()
    assert not at.exception
    after = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert at.session_state["forecast_calls"] == calls
    assert after["data"][:2] == before["data"][:2]
    future = next(t for t in after["data"] if "Future actual" in t["name"])
    assert future["x"] == list(range(3, 16))
    at.toggle[0].set_value(False).run()
    assert json.loads(at.get("plotly_chart")[0].proto.spec)["data"] == before["data"]
    assert at.session_state["forecast_calls"] == calls
    # Results uses only the saved direct heads; the overlay cannot extend them.
    assert not at.selectbox
    assert at.session_state["forecast_calls"] == calls

    result = {"as_of_s": 1, "observed_prefix": [{"timestamp_s": 1, "signal": 2}],
              "points": [{"target_time_s": 2, "value": 3},
                         {"target_time_s": 3, "value": 4, "kind": "recursive"}]}
    fig = replay_figure(result, {}, future_actual=[{"timestamp_s": 2, "signal": 5, "gap_before": True},
                                                   {"timestamp_s": 3, "signal": 6}])
    by_name = {t.name: t for t in fig.data}
    assert list(by_name["Forecast horizon"].x) == [1, 2]
    assert list(by_name["Recursive forecast"].x) == [2, 3]
    assert list(by_name["Future actual · hidden from model"].x) == [1, None, 2, 3]


def test_actual_red_marker_is_retrospective_and_never_creates_a_corridor():
    result = {"as_of_s": 1, "observed_prefix": [{"timestamp_s": 0, "signal": 0.5},
                                                 {"timestamp_s": 1, "signal": 0.7}],
              "points": [{"target_time_s": 2, "value": 0.9}],
              "thresholds": {"status": "available", "red": 3, "yellow": 1,
                             "direction": "above", "mode": "absolute"}}
    future = [{"timestamp_s": 2, "signal": 2.5},
              {"timestamp_s": 3, "signal": 3.4, "gap_before": True},
              {"timestamp_s": 4, "signal": 4.0}]
    hidden = replay_figure(result, {}, future_actual=None)
    shown = replay_figure(result, {}, future_actual=future)
    assert "First recorded RED sample · hidden from model" not in [trace.name for trace in hidden.data]
    marker = next(trace for trace in shown.data if trace.name == "First recorded RED sample · hidden from model")
    assert list(marker.x) == [3]
    assert list(marker.y) == [3.4]
    assert not any("Predicted RED-entry window" in str(shape) for shape in shown.layout.shapes)
    assert first_recorded_red_after_now(result["observed_prefix"], future, result["thresholds"], 1) == future[1]
    already_red = [{"timestamp_s": 0, "signal": 3.1}, {"timestamp_s": 1, "signal": 0.7}]
    assert first_recorded_red_after_now(already_red, future, result["thresholds"], 1) is None
    baseline = {**result["thresholds"], "baseline_n": 2}
    assert first_recorded_red_after_now(already_red, future, baseline, 1) == future[1]


def test_only_explicit_model_entry_window_is_drawn_and_future_does_not_move_it():
    base = {"as_of_s": 10, "observed_prefix": [{"timestamp_s": 10, "signal": 1}],
            "points": [{"target_time_s": 20, "value": 2, "lower": 1, "upper": 4}],
            "thresholds": {"status": "available", "red": 3, "yellow": 2,
                           "direction": "above", "mode": "absolute"},
            "red_entry_corridor": {"status": "available", "earliest_s": 30, "latest_s": 40}}
    hidden = replay_figure(base, {})
    shown = replay_figure(base, {}, future_actual=[{"timestamp_s": 35, "signal": 3.5}])
    def windows(fig):
        return [(shape.x0, shape.x1) for shape in fig.layout.shapes
                if shape.fillcolor == TOKENS["dark"]["series_band"]]
    assert windows(hidden) == windows(shown) == [(30, 40)]
    invalid = {**base, "red_entry_corridor": {"status": "available", "earliest_s": 40, "latest_s": 30}}
    assert windows(replay_figure(invalid, {})) == []


def test_background_replay_keeps_chart_live_and_play_waits_for_complete_cursor(monkeypatch):
    class ControlledWorker:
        def __init__(self):
            self.key = None
            self.done = False
            self.requests = 0
            self.error = None

        def request(self, key, compute):
            if key != self.key:
                self.key, self.done = key, False
                self.requests += 1
            as_of = key[4]
            return {"done": self.done, "error": self.error, "result": {
                "project_id": key[0], "run_id": key[1], "snapshot_id": key[2], "as_of_s": as_of,
                "points": [{"target_time_s": as_of + 1, "value": as_of + .5}],
                "crossing": {"status": "none_within_horizon" if self.done else "pending", "time_s": None},
            }}

        def cancel(self):
            self.key = None

    worker = ControlledWorker()
    monkeypatch.setattr("pdm.project_results_ui._background_forecasts", lambda: worker)
    at = AppTest.from_file(str(project_root() / "tests" / "project_results_harness.py"), default_timeout=15)
    at.session_state["background_test"] = True
    at.run()
    key = "project_play:project1:run1:unit1"
    assert not at.exception and len(at.get("plotly_chart")) == 1
    next(button for button in at.button if button.label == "Play").click().run()
    for _ in range(3):
        at.run()
        assert at.session_state[key]["cursor"] == 0
        assert len(at.get("plotly_chart")) == 1
        assert any("Calculating the saved model's direct forecast" in str(message.value) for message in at.info)
    assert worker.requests == 1
    assert not any("No red crossing" in str(message.value) for message in at.info)
    next(button for button in at.button if button.label == "Pause").click().run()
    assert not at.session_state[key]["playing"]
    at.slider[0].set_value(3).run()
    assert at.session_state[key]["cursor"] == 3 and worker.requests == 2
    chart = json.loads(at.get("plotly_chart")[0].proto.spec)
    assert chart["data"][1]["x"] == [3, 4]  # no stale cursor-0 prediction
    at.toggle[0].set_value(True).run()
    assert worker.requests == 2
    assert any("Future actual" in trace["name"] for trace in json.loads(at.get("plotly_chart")[0].proto.spec)["data"])
    worker.done = True
    at.run()
    assert not at.exception and not at.get("progress")
    assert at.session_state[key]["cursor"] == 3
    next(button for button in at.button if button.label == "Play").click().run()
    assert at.session_state[key]["cursor"] == 4 and worker.requests == 3
    at.run()
    assert at.session_state[key]["cursor"] == 4  # wait for this forecast too
    next(button for button in at.button if button.label == "Reset").click().run()
    assert at.session_state[key]["cursor"] == 0 and not at.session_state[key]["playing"]
    worker.done, worker.error = True, "Invalid artifact"
    at.run()
    assert at.error and not at.get("progress")
    assert at.session_state[key]["forecast_ready_cursor"] is None
    assert not any("Searching for red entry" in str(c.value) for c in at.caption)
