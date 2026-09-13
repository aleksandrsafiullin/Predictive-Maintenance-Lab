from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

OVERLAY_HISTORY_CAPTION = (
    "The faded recording to the right of Now is historical. "
    "The reservoir only sees the highlighted window."
)

_SENSOR_COLUMNS = {
    "bearings": "horizontal_rms",
    "filters": "differential_pressure",
}


def overlay_history_caption() -> str:
    return OVERLAY_HISTORY_CAPTION


def overlay_x_scale(dataset_id: str) -> tuple[float, str]:
    """Bearings plot minutes (s/60). Filters keep internal seconds (Time × 60 caption)."""
    if str(dataset_id) == "filters":
        return 1.0, "Internal time (s; Time × 60, unit unconfirmed)"
    return 60.0, "Operating time (min)"


def predicted_zone_x(
    now_timestamp_s: float,
    predicted_rul_s: float,
    *,
    dataset_id: str,
) -> tuple[float, float, float]:
    """Return (now_x, zone_x1, event_x) in plot units. Does not rescore the model."""
    scale, _ = overlay_x_scale(dataset_id)
    now_x = float(now_timestamp_s) / scale
    event_x = now_x + float(predicted_rul_s) / scale
    return now_x, event_x, event_x


def build_work_overlay_figure(
    *,
    dataset_id: str,
    unit_features: pd.DataFrame | None,
    now_timestamp_s: float | None,
    predicted_rul_s: float | None,
    history_length: int | None,
    show_gt: bool = False,
    event_time_s: float | None = None,
    event_observed: bool = False,
    pressure_limit_pa: float | None = None,
    predicted_rul_by_time: pd.DataFrame | None = None,
    actual_rul_s: np.ndarray | pd.Series | None = None,
) -> go.Figure:
    """Sensor + time-zone overlay. Predicted RUL is never a y-series on the sensor panel.

    ``predicted_rul_s`` is a stored forecast passed in by the caller. This function
    does not look at future rows to rescore. ``show_gt`` only toggles the actual-event
    line (and optional second-row actual RUL), not the predicted zone.
    """
    scale, x_title = overlay_x_scale(dataset_id)
    sensor_col = _SENSOR_COLUMNS.get(str(dataset_id), "horizontal_rms")
    y_title = "RMS" if str(dataset_id) == "bearings" else "Differential pressure (Pa)"

    feat = unit_features.copy() if isinstance(unit_features, pd.DataFrame) else pd.DataFrame()
    if not feat.empty and "timestamp_s" in feat.columns:
        feat = feat.sort_values("timestamp_s")
    has_sensor = (not feat.empty) and sensor_col in feat.columns and "timestamp_s" in feat.columns

    second_row = predicted_rul_by_time is not None and isinstance(predicted_rul_by_time, pd.DataFrame)
    if second_row:
        fig = go.Figure()
        from plotly.subplots import make_subplots

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.12,
            row_heights=[0.62, 0.38],
            subplot_titles=("Sensor", "Remaining useful life"),
        )
        sensor_row = 1
        rul_row = 2
    else:
        fig = go.Figure()
        sensor_row = None
        rul_row = None

    def _add_scatter(**kwargs: Any) -> None:
        if sensor_row is not None:
            fig.add_trace(go.Scatter(**kwargs), row=sensor_row, col=1)
        else:
            fig.add_trace(go.Scatter(**kwargs))

    now_s = float(now_timestamp_s) if _finite(now_timestamp_s) else None
    now_x = None
    zone_x0 = None
    zone_x1 = None
    event_x = None
    stored_pred = float(predicted_rul_s) if _finite(predicted_rul_s) else None

    if has_sensor:
        ts = pd.to_numeric(feat["timestamp_s"], errors="coerce")
        y = pd.to_numeric(feat[sensor_col], errors="coerce")
        x_all = ts / scale
        _add_scatter(
            x=x_all,
            y=y,
            mode="lines",
            name="recorded signal (full history)",
            line=dict(color="rgba(150,158,170,0.42)", width=1.6),
            hovertemplate="%{x:.3g}<br>%{y:.4g}<extra>recorded signal (full history)</extra>",
        )
        if now_s is not None:
            seen = feat.loc[ts <= now_s]
            if not seen.empty:
                _add_scatter(
                    x=pd.to_numeric(seen["timestamp_s"], errors="coerce") / scale,
                    y=pd.to_numeric(seen[sensor_col], errors="coerce"),
                    mode="lines",
                    name="seen by the model",
                    line=dict(color="#7ed0ea", width=2.6),
                    hovertemplate="%{x:.3g}<br>%{y:.4g}<extra>seen by the model</extra>",
                )
            window = _input_window(feat, now_s, history_length)
            if window is not None:
                x0 = float(window["timestamp_s"].iloc[0]) / scale
                x1 = float(window["timestamp_s"].iloc[-1]) / scale
                if x1 <= x0:
                    x1 = x0 + (1.0 / scale)
                _add_vrect(
                    fig,
                    x0=x0,
                    x1=x1,
                    fillcolor="rgba(70, 170, 255, 0.16)",
                    line_width=0,
                    annotation_text="fed into reservoir",
                    annotation_position="top left",
                    row=sensor_row,
                )

    if now_s is not None:
        now_x, zone_x1_pred, event_x_pred = (
            predicted_zone_x(now_s, stored_pred, dataset_id=dataset_id)
            if stored_pred is not None
            else (now_s / scale, None, None)
        )
        _add_vline(fig, x=now_x, line_dash="dot", line_color="#f2f5ff", annotation_text="Now", row=sensor_row)
        if stored_pred is not None:
            zone_x0 = now_x
            zone_x1 = zone_x1_pred
            event_x = event_x_pred
            _add_vrect(
                fig,
                x0=zone_x0,
                x1=zone_x1,
                fillcolor="rgba(232, 186, 74, 0.18)",
                line_width=0,
                annotation_text="predicted failure",
                annotation_position="top right",
                row=sensor_row,
            )
            _add_vline(
                fig,
                x=event_x,
                line_dash="dash",
                line_color="#e8ba4a",
                annotation_text="predicted event",
                row=sensor_row,
            )

    show_event = False
    if _finite(event_time_s):
        et = float(event_time_s)
        if bool(show_gt) or (bool(event_observed) and now_s is not None and et <= now_s):
            show_event = True
            label = "observed 600 Pa" if str(dataset_id) == "filters" else "actual event"
            _add_vline(
                fig,
                x=et / scale,
                line_dash="dash",
                line_color="#e05656",
                annotation_text=label,
                row=sensor_row,
            )

    if str(dataset_id) == "filters" and _finite(pressure_limit_pa):
        if sensor_row is not None:
            fig.add_hline(
                y=float(pressure_limit_pa),
                line_dash="dash",
                annotation_text=f"{float(pressure_limit_pa):.0f} Pa",
                row=sensor_row,
                col=1,
            )
        else:
            fig.add_hline(
                y=float(pressure_limit_pa),
                line_dash="dash",
                annotation_text=f"{float(pressure_limit_pa):.0f} Pa",
            )

    if second_row and predicted_rul_by_time is not None:
        pred = predicted_rul_by_time
        if not pred.empty and "timestamp_s" in pred.columns and "predicted_rul_s" in pred.columns:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_numeric(pred["timestamp_s"], errors="coerce") / scale,
                    y=pd.to_numeric(pred["predicted_rul_s"], errors="coerce") / scale,
                    mode="lines",
                    name="predicted RUL",
                    line=dict(color="#7ed0ea"),
                ),
                row=rul_row,
                col=1,
            )
        if show_gt and actual_rul_s is not None and not pred.empty and "timestamp_s" in pred.columns:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_numeric(pred["timestamp_s"], errors="coerce") / scale,
                    y=np.asarray(actual_rul_s, dtype=float) / scale,
                    mode="lines",
                    name="actual RUL (evaluator overlay)",
                    line=dict(color="#e05656", dash="dash"),
                ),
                row=rul_row,
                col=1,
            )
        rul_title = "Remaining useful life (s, internal)" if str(dataset_id) == "filters" else "Remaining useful life (min)"
        fig.update_yaxes(title_text=rul_title, row=rul_row, col=1)
        fig.update_xaxes(title_text=x_title, row=rul_row, col=1)
        fig.update_yaxes(title_text=y_title, row=sensor_row, col=1)
    else:
        fig.update_xaxes(title_text=x_title)
        fig.update_yaxes(title_text=y_title)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1018",
        plot_bgcolor="#0b1018",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=56, b=48, l=56, r=24),
        meta={
            "predicted_zone_x0": zone_x0,
            "predicted_zone_x1": zone_x1,
            "predicted_event_x": event_x,
            "now_x": now_x,
            "show_gt": bool(show_gt),
            "show_actual_event": bool(show_event),
            "stored_predicted_rul_s": stored_pred,
        },
    )
    return fig


def _input_window(feat: pd.DataFrame, now_s: float, history_length: int | None) -> pd.DataFrame | None:
    ts = pd.to_numeric(feat["timestamp_s"], errors="coerce")
    prefix = feat.loc[ts <= float(now_s)]
    if prefix.empty:
        return None
    n = int(history_length) if history_length else 0
    if n <= 0:
        return prefix
    return prefix.iloc[-n:]


def _add_vrect(fig: go.Figure, *, row: int | None, **kwargs: Any) -> None:
    if row is not None:
        fig.add_vrect(**kwargs, row=row, col=1)
    else:
        fig.add_vrect(**kwargs)


def _add_vline(fig: go.Figure, *, row: int | None, **kwargs: Any) -> None:
    if row is not None:
        fig.add_vline(**kwargs, row=row, col=1)
    else:
        fig.add_vline(**kwargs)


def _finite(val: Any) -> bool:
    if val is None:
        return False
    try:
        if isinstance(val, Mapping):
            return False
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        return False
    try:
        return bool(np.isfinite(float(val)))
    except (TypeError, ValueError):
        return False
