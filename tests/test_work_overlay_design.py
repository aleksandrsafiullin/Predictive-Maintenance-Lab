"""Visual layout must preserve physical intervals and legible independent time scales."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.visualization.overlay import build_work_overlay_figure


def _figure(show_gt=False, interval=(120., 36000.)):
    times = np.array([60., 120., 180., 240.])
    return build_work_overlay_figure(
        dataset_id="bearings",
        unit_features=pd.DataFrame({"timestamp_s": times, "horizontal_rms": [1., 1.2, 1.5, 1.8]}),
        now_timestamp_s=180., predicted_rul_s=600., history_length=None,
        show_gt=show_gt, event_time_s=240., event_observed=True,
        prediction_interval_s=interval,
        predicted_rul_by_time=pd.DataFrame({
            "timestamp_s": times[:3], "predicted_rul_s": [np.nan, 660., 600.],
            "lower_rul_s": [np.nan, 180., interval[0]],
            "upper_rul_s": [np.nan, 42000., interval[1]],
        }), actual_rul_s=np.array([180., 120., 60.]),
    )


def test_broad_failure_window_neither_clips_bounds_nor_squashes_observed_history():
    fig = _figure()
    lane = next(t for t in fig.data if t.name == "failure window")
    assert list(lane.x) == [5., 603.]
    assert lane.xaxis == "x2"
    assert fig.layout.xaxis2.range[1] > 603.
    assert fig.layout.xaxis.range[1] < 5.  # recorded vibration, not failure-date horizon
    assert fig.layout.xaxis3.range[1] < 4.  # only elapsed forecast history
    assert fig.layout.xaxis3.matches is None
    upper = next(t for t in fig.data if t.name == "empirical forecast interval")
    np.testing.assert_allclose(upper.y, [np.nan, 700., 600.], equal_nan=True)
    assert upper.connectgaps is False
    assert upper.fill == "tonexty"
    assert not any(s.fillcolor == "rgba(232, 186, 74, 0.18)" for s in fig.layout.shapes)


def test_marker_is_current_measurement_and_ground_truth_remains_separate():
    off, on = _figure(), _figure(True)
    for figure in (off, on):
        marker = next(t for t in figure.data if t.name == "current measurement")
        assert list(marker.x) == [3.]
        assert list(marker.y) == [1.5]
    assert not any(t.name == "recorded endpoint" for t in off.data)
    endpoint = next(t for t in on.data if t.name == "recorded endpoint")
    assert list(endpoint.x) == [4.]
    assert endpoint.xaxis == "x2"
    for name in ("failure window", "predicted RUL", "empirical forecast interval"):
        a, b = (next(t for t in f.data if t.name == name) for f in (off, on))
        np.testing.assert_allclose(a.x, b.x, equal_nan=True)
        np.testing.assert_allclose(a.y, b.y, equal_nan=True)


@pytest.mark.parametrize("interval", [(10., 1.), (-1., 10.), (0., np.inf)])
def test_visual_design_does_not_accept_invalid_interval(interval):
    with pytest.raises(ValueError, match="ordered finite nonnegative"):
        _figure(interval=interval)
