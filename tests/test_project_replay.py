from __future__ import annotations

import pandas as pd
from streamlit.testing.v1 import AppTest

from pdm.paths import project_root
from pdm.project_results_ui import advance_play_state, replay_figure, visible_observations
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
    assert list(fig.data[1].y) == [-0.4, None, -2.2]
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
