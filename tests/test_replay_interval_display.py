"""Saved interval forecasts remain visible in the evaluation replay screen."""
from __future__ import annotations

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _figure(prediction, show_gt):
    from pdm.app import _replay_demo_figure

    return _replay_demo_figure(
        dataset_id="bearings",
        prefix_meas=pd.DataFrame({"timestamp_s": [60., 120.], "horizontal_rms": [1., 2.]}),
        prefix_pred=prediction, overlay_rul=pd.Series([540., 480.]), show_gt=show_gt,
        h_s=120., replay_time_s=120., warning_time_s=None, pressure_limit_pa=600.,
        event_time_s=600., event_observed=True,
    )


def test_replay_interval_band_is_physical_and_independent_of_ground_truth():
    prediction = pd.DataFrame({
        "timestamp_s": [60., 120.], "raw_rul_s": [500., 200.],
        "predicted_rul_s": [float("nan"), 180.],
        "lower_rul_s": [float("nan"), 120.], "upper_rul_s": [float("nan"), 240.],
    })
    off, on = _figure(prediction, False), _figure(prediction, True)
    off_range = [t for t in off.data if "forecast range" in t.name]
    on_range = [t for t in on.data if "forecast range" in t.name]
    assert len(off_range) == len(on_range) == 2
    np.testing.assert_allclose(off_range[0].y, [np.nan, 2.], equal_nan=True)
    np.testing.assert_allclose(off_range[1].y, [np.nan, 4.], equal_nan=True)
    for a, b in zip(off_range, on_range, strict=True):
        np.testing.assert_allclose(a.y, b.y, equal_nan=True)
        assert a.connectgaps is False
    assert off_range[1].fill == "tonexty"
    latest = next(t for t in off.data if t.name == "latest interval")
    assert list(latest.x) == [2., 2.]
    assert list(latest.y) == [2., 4.]
    assert not any("actual RUL" in t.name for t in off.data)
    assert any("actual RUL" in t.name for t in on.data)


def test_replay_legacy_point_has_no_invented_range():
    prediction = pd.DataFrame({"timestamp_s": [60., 120.], "predicted_rul_s": [200., 180.]})
    fig = _figure(prediction, False)
    assert not any("forecast range" in t.name for t in fig.data)
    assert any(t.name == "neural net RUL (min)" for t in fig.data)


def _replay_page(prediction):
    import pandas as pd
    import streamlit as st

    from pdm.app import _replay_playback_body

    st.session_state["_replay_view"] = {
        "dataset_id": "bearings", "run_id": "fixture", "unit_id": "bearing", "n": 2,
        "h_s": 120.,
        "meas": pd.DataFrame({"timestamp_s": [60., 120.], "horizontal_rms": [1., 2.]}),
        "unit_pred": pd.DataFrame([{"unit_id": "bearing", "timestamp_s": 60., **prediction}]),
        "unit_meta": {"unit_id": "bearing", "event_observed": 1, "event_time_s": 600., "observation_end_s": 600.},
    }
    _replay_playback_body()


def test_replay_header_prefers_failure_and_remaining_ranges():
    at = AppTest.from_function(_replay_page, args=({
        "predicted_rul_s": 180., "lower_rul_s": 120., "upper_rul_s": 240.,
    },), default_timeout=20)
    at.run()
    assert not at.exception
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Failure window (min)"] == "3.0–5.0"
    assert metrics["Remaining life range (min)"] == "2.0–4.0"
    assert "Predicted RUL (min)" not in metrics


def test_replay_header_preserves_legacy_point_and_raw_warmup_is_not_forecast():
    for prediction, expected in [
        ({"predicted_rul_s": 180.}, "3.0"),
        ({"predicted_rul_s": np.nan, "raw_rul_s": 180., "lower_rul_s": np.nan, "upper_rul_s": np.nan}, "—"),
    ]:
        at = AppTest.from_function(_replay_page, args=(prediction,), default_timeout=20)
        at.run()
        assert not at.exception
        metrics = {m.label: m.value for m in at.metric}
        assert metrics["Predicted RUL (min)"] == expected
        assert "Failure window (min)" not in metrics
