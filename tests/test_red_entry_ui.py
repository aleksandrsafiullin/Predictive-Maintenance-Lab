"""Event controls and causal chart adapter preserve frozen forecast semantics."""

import json

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pdm.project_results_ui import replay_figure
from pdm.red_entry_ui import event_chart_payload, probability_figure
from pdm.ui_theme import tokens


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_probability_figure_uses_explicit_theme_and_preserves_data(theme):
    fig = probability_figure({"probability_by_horizon": [
        {"horizon_s": 60, "probability": 0.125},
        {"horizon_s": 120, "probability": None},
    ]}, theme)
    t = tokens(theme)
    assert fig.layout.paper_bgcolor == fig.layout.plot_bgcolor == t["chart_bg"]
    assert fig.layout.font.color == t["chart_text"]
    assert fig.layout.xaxis.linecolor == fig.layout.yaxis.linecolor == t["chart_axis"]
    assert fig.layout.yaxis.gridcolor == t["chart_grid"]
    assert fig.data[0].line.color == t["series_forecast"]
    assert list(fig.data[0].x) == [60, 120]
    assert list(fig.data[0].y) == [0.125, None]
    assert fig.data[0].connectgaps is False
    assert fig.data[0].name == "First RED probability"
    assert fig.layout.xaxis.title.text == "Horizon from now (s)"
    assert fig.layout.yaxis.title.text == "Probability"
    assert list(fig.layout.yaxis.range) == [0, 1]
    assert fig.layout.height == 280


def test_event_replay_propagates_switched_theme_through_playing_reruns():
    def app():
        from unittest.mock import patch

        import pandas as pd
        import streamlit as st

        from pdm.red_entry_ui import render_event_replay

        class ReadyWorker:
            def request(self, key, callback):
                return {"done": True, "result": {
                    "model_id": "r", "snapshot_id": "s", "issued_at_s": key[-2],
                    "probability_by_horizon": [{"horizon_s": 60, "probability": 0.125}],
                }}

        theme = st.selectbox("Test theme", ["dark", "light"])
        snapshot = {
            "snapshot_id": "s", "schema": {},
            "features": pd.DataFrame({
                "unit_id": ["u"] * 8, "timestamp_s": list(range(8)),
                "signal": [1.0] * 8,
            }),
        }
        with (
            patch("pdm.red_entry_ui.load_zone_limits", return_value=None),
            patch("pdm.project_results_ui._background_forecasts", return_value=ReadyWorker()),
        ):
            render_event_replay("p", "r", "u", snapshot, theme)

    tested = AppTest.from_function(app).run()

    def assert_chart_theme(theme):
        assert not tested.exception
        charts = tested.get("plotly_chart")
        assert len(charts) == 2
        for chart in charts:
            spec = json.loads(chart.proto.spec)
            assert spec["layout"]["paper_bgcolor"] == tokens(theme)["chart_bg"]
            assert spec["layout"]["plot_bgcolor"] == tokens(theme)["chart_bg"]
            assert spec["layout"]["font"]["color"] == tokens(theme)["chart_text"]
            assert chart.proto.theme == ""
        probability = json.loads(charts[0].proto.spec)
        assert probability["data"][0]["line"]["color"] == tokens(theme)["series_forecast"]
        assert probability["data"][0]["y"] == [0.125]
        identity = next(row.value for row in tested.caption
                        if row.value.startswith("Equipment:"))
        assert "Equipment: u" in identity
        assert f"Now: {tested.slider[0].value} s (recording time)" in identity

    assert_chart_theme("dark")
    tested.button[0].click().run()
    tested.selectbox[0].select("light").run()
    assert_chart_theme("light")
    cursor = tested.slider[0].value
    for _ in range(2):
        tested.run()
        assert_chart_theme("light")
        assert tested.slider[0].value > cursor
        cursor = tested.slider[0].value


def test_event_payload_identity_handles_missing_metadata():
    def app():
        from pdm.red_entry_ui import render_event_payload

        render_event_payload({}, "light")

    tested = AppTest.from_function(app).run()
    assert not tested.exception
    assert tested.caption[0].value == (
        "Equipment: N/A · "
        "Now: N/A s (recording time)"
    )


def test_event_slider_timer_acknowledgement_keeps_playing_but_user_seek_pauses():
    from pdm.project_results_ui import advance_play_state
    from pdm.red_entry_ui import _seek_event_cursor

    state = {"cursor": 0, "playing": True, "displayed_cursor": 0, "active_slider": "cursor0"}
    state.update(advance_play_state(state, 12))
    state["displayed_cursor"] = state["cursor"]
    state["active_slider"] = "cursor1"
    _seek_event_cursor(state, 1, "cursor1")
    assert state["playing"] is True
    state.update(advance_play_state(state, 12))
    state["active_slider"] = "cursor2"
    state["displayed_cursor"] = 2
    _seek_event_cursor(state, 1, "cursor1")
    assert state["cursor"] == 2
    assert state["playing"] is True
    _seek_event_cursor(state, 7, "cursor2")
    assert state["cursor"] == 7
    assert state["playing"] is False


def test_event_adapter_keeps_absolute_corridor_and_prefix():
    result = {
        "issued_at_s": 10.0,
        "red_entry_corridor": {"status": "available", "earliest_s": 20.0, "latest_s": 30.0},
    }
    prefix = pd.DataFrame(
        {"timestamp_s": [0.0, 10.0], "signal": [1.0, 2.0], "gap_before": [False, False]}
    )
    payload = event_chart_payload(
        result, prefix, {"status": "available", "red": 3.0, "yellow": 2.5, "direction": "above"}
    )
    fig = replay_figure(payload, {"signal_label": "Signal", "signal_unit": "g"}, "dark")
    actual = fig.data[0]
    assert max(actual.x) == 10.0
    corridor = next(
        shape for shape in fig.layout.shapes if shape.type == "rect" and shape.x0 == 20.0
    )
    assert corridor.x1 == 30.0
    assert payload["points"] == []


def test_unsupported_probabilities_are_not_joined():
    fig = probability_figure(
        {
            "probability_by_horizon": [
                {"horizon_s": 5.0, "probability": 0.2},
                {"horizon_s": 10.0, "probability": None},
            ]
        }
    )
    assert list(fig.data[0].y) == [0.2, None]
    assert fig.data[0].connectgaps is False


def test_training_task_defaults_to_first_red_and_signal_remains_available():
    def app():
        from unittest.mock import patch

        import streamlit as st

        import pdm.project_training_ui as ui
        import pdm.red_entry_ui as event

        with (
            patch.object(
                event,
                "render_event_training",
                lambda project_id, snapshot: st.write("event branch"),
            ),
            patch.object(ui, "_job_status", lambda project_id: None),
            patch.object(
                ui,
                "_render_signal_training",
                lambda project_id, snapshot: st.write("signal branch"),
            ),
            patch.object(ui, "list_project_runs", lambda project_id: []),
            patch("pdm.red_entry_training.list_red_entry_runs", lambda project_id: []),
        ):
            ui.render_training("p", {"snapshot_id": "s"})

    app_test = AppTest.from_function(app).run()
    assert not app_test.exception
    assert app_test.selectbox[0].value == "red_entry"
    assert any(row.value == "event branch" for row in app_test.markdown)
    app_test.selectbox[0].select("signal_forecast").run()
    assert any(row.value == "signal branch" for row in app_test.markdown)
    app_test.selectbox[0].select("legacy_rul").run()
    assert "unavailable" in app_test.info[0].value.lower()


def test_saved_evaluation_preserves_unknowns_and_keeps_raw_manifest_collapsed():
    def app():
        from pdm.red_entry_ui import render_event_evaluation

        render_event_evaluation({
            "contract_hash": "secret-technical-hash",
            "quality_gate_status": "requirements_unset",
            "metrics": {"test": {
                "metrics": {"event_count": 0, "independent_event_units": 0,
                            "reachable_event_count": 0, "alert_time_fraction": None,
                            "operating_exposure_s": None},
                "exposure_basis": "observed_is_running",
                "brier_by_horizon": [{"horizon_s": 60, "brier": None,
                                     "known_origins": 0, "unknown_origins": 8,
                                     "physical_units": 0, "prediction_coverage": 0.0}],
                "metric_scope": {"policy_prediction_coverage": 0.0,
                                 "alert_time": "partial_observed_predictions_only"},
                "quality_gate": {"status": "requirements_unset",
                                 "reason_codes": ["no_independent_holdout", "new_internal_reason"]},
            }},
        })

    tested = AppTest.from_function(app).run()
    assert not tested.exception
    assert [metric.value for metric in tested.metric] == ["0", "0", "0"]
    summary = tested.table[0].value
    assert summary.loc[summary["Metric"] == "Alert time fraction",
                       "Saved value"].iloc[0] == "N/A"
    horizons = tested.table[1].value
    assert pd.isna(horizons["Brier"].iloc[0])
    text = " ".join(row.value for row in tested.caption)
    assert "new_internal_reason" not in text
    details = tested.table[2].value.set_index("Metric")["Saved value"]
    assert details["Operating exposure (s)"] == "N/A"
    assert details["Evaluation gate"] == "Requirements unset"
    assert not tested.warning
    assert not tested.info
    diagnostics = tested.expander[0]
    assert diagnostics.label == "Run diagnostics"
    assert diagnostics.proto.expanded is False
    assert len(diagnostics.get("json")) == 1
    assert "secret-technical-hash" in diagnostics.get("json")[0].value
    assert len(tested.get("json")) == 1


def test_forecast_copy_translates_tokens_without_changing_probabilities():
    def app():
        from pdm.red_entry_ui import render_event_payload

        render_event_payload({
            "unit_id": "equipment-ref", "issued_at_s": 12.5,
            "prediction_status": "available", "risk_status": "at_risk",
            "age_source": "counter",
            "operating_scenario": "continuation_of_observed_regime_assumption",
            "probability_by_horizon": [
                {"horizon_s": 60, "probability": 0.125, "support_status": "supported_exploratory"},
                {"horizon_s": 120, "probability": None, "support_status": "unsupported"},
            ],
            "warning": {"status": "unavailable", "reason_codes": ["policy_not_frozen"]},
            "calibration_status": "insufficient_event_evidence",
            "quality_gate_status": "requirements_unset",
        })

    tested = AppTest.from_function(app).run()
    assert not tested.exception
    text = " ".join(row.value for row in tested.caption)
    assert "Equipment: equipment-ref · Now: 12.5 s (recording time)" in text
    assert "Warning: Unavailable" in text
    assert "First RED corridor: N/A" in text
    assert "policy_not_frozen" not in text
    rows = tested.table[0].value
    assert rows["RED probability"].iloc[0] == 0.125
    assert pd.isna(rows["RED probability"].iloc[1])
    assert list(rows["Data support"]) == ["Exploratory support", "Unsupported"]


def test_legacy_useful_recall_zeros_masked_without_rewriting_saved_metrics():
    from copy import deepcopy
    from unittest.mock import Mock, patch

    import pdm.red_entry_ui as ui

    for reason in ("unset", "unknown_counts", "unknown_events", "scope", "available"):
        evaluation = {
            "metrics": {"event_count": 2, "independent_event_units": 2, "reachable_event_count": 2,
                        "useful_event_recall": 0, "reachable_useful_event_recall": 0,
                        "green_useful_event_recall": 0, "false_alert_episodes": 3},
            "quality_gate": {"unset_requirements": ["minimum_action_lead_s"] if reason == "unset" else []},
        }
        if reason == "unknown_counts":
            evaluation["metrics"]["event_status_counts"] = {"unknown": 1}
        elif reason == "unknown_events":
            evaluation["events"] = [{"status": "unknown"}]
        elif reason == "scope":
            evaluation["metric_scope"] = {"useful_event_recall": {"status": "unavailable"}}
        manifest = {"metrics": {"test": evaluation}}
        original = deepcopy(manifest)
        tables, metric_labels, captions = [], [], []
        column = Mock()
        column.metric.side_effect = lambda label, value: metric_labels.append(label)
        with (patch.object(ui.st, "columns", return_value=[column, column, column]),
              patch.object(ui.st, "selectbox", return_value="test"),
              patch.object(ui.st, "table", side_effect=lambda table, **kwargs: tables.append(table)),
              patch.object(ui.st, "caption", side_effect=captions.append)):
            ui.render_event_evaluation(manifest)
        assert manifest == original
        expected = "0.0%" if reason == "available" else "N/A"
        assert list(tables[0]["Saved value"][:3]) == [expected] * 3
        assert tables[0].loc[tables[0]["Metric"] == "False alert episodes", "Saved value"].iloc[0] == "3"
        assert "Physical units with RED" in metric_labels
        if reason == "unset":
            assert "Events within policy horizon" in metric_labels
