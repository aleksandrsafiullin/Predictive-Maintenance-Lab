from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

OVERLAY_HISTORY_CAPTION = (
    "Cyan shows measurements already received; the muted recording is for comparison. "
    "Future samples are never fed to the model."
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
    prediction_interval_s: tuple[float, float] | None = None,
    event_reference_label: str = "Recorded end",
) -> go.Figure:
    """Sensor + time-zone overlay. Predicted RUL is never a y-series on the sensor panel.

    ``predicted_rul_s`` is a stored forecast passed in by the caller. This function
    does not look at future rows to rescore. ``show_gt`` only toggles the actual-event
    line (and optional second-row actual RUL), not the predicted zone.
    """
    scale, x_title = overlay_x_scale(dataset_id)
    if prediction_interval_s is not None:
        if (len(prediction_interval_s) != 2 or not all(_finite(v) for v in prediction_interval_s)
                or not 0 <= prediction_interval_s[0] <= prediction_interval_s[1]):
            raise ValueError("Failure interval must contain ordered finite nonnegative bounds")
    sensor_col = _SENSOR_COLUMNS.get(str(dataset_id), "horizontal_rms")
    is_bearing = str(dataset_id) == "bearings"
    unit = "min" if is_bearing else "s"
    cyan, amber, coral = "#61d8ee", "#f2be68", "#ff7e83"
    muted, ink = "#8fa3b7", "#edf4fa"
    feat = unit_features.copy() if isinstance(unit_features, pd.DataFrame) else pd.DataFrame()
    if not feat.empty and "timestamp_s" in feat.columns:
        feat = feat.sort_values("timestamp_s")
    has_sensor = not feat.empty and {sensor_col, "timestamp_s"} <= set(feat.columns)
    second_row = isinstance(predicted_rul_by_time, pd.DataFrame)
    from plotly.subplots import make_subplots

    # Failure dates need their own horizontal scale. A broad legitimate interval
    # must not flatten the measured signal or squeeze the observed forecast history.
    fig = make_subplots(
        rows=3 if second_row else 2, cols=1, shared_xaxes=False,
        vertical_spacing=0.17 if second_row else 0.24,
        row_heights=[0.43, 0.12, 0.45] if second_row else [0.78, 0.22],
    )
    now_s = float(now_timestamp_s) if _finite(now_timestamp_s) else None
    now_x = now_s / scale if now_s is not None else None
    zone_x0 = zone_x1 = event_x = None
    stored_pred = float(predicted_rul_s) if _finite(predicted_rul_s) else None
    interval_valid = prediction_interval_s is not None
    show_event = _finite(event_time_s) and (
        bool(show_gt) or (bool(event_observed) and now_s is not None and float(event_time_s) <= now_s)
    )

    def scatter(row: int, **kwargs: Any) -> None:
        fig.add_trace(go.Scatter(**kwargs), row=row, col=1)

    if has_sensor:
        ts = pd.to_numeric(feat["timestamp_s"], errors="coerce")
        y = pd.to_numeric(feat[sensor_col], errors="coerce")
        scatter(1, x=ts / scale, y=y, mode="lines", name="recorded signal (full history)",
                showlegend=False, line=dict(color="rgba(143,163,183,0.32)", width=1.5),
                hovertemplate=f"Recorded · %{{x:.1f}} {unit}<br>%{{y:.3f}}<extra></extra>")
        if now_s is not None:
            seen = feat.loc[ts <= now_s]
            if not seen.empty:
                seen_x = pd.to_numeric(seen["timestamp_s"], errors="coerce") / scale
                seen_y = pd.to_numeric(seen[sensor_col], errors="coerce")
                scatter(1, x=seen_x, y=seen_y, mode="lines", name="seen by the model",
                        showlegend=False, line=dict(color=cyan, width=2.5),
                        fill="tozeroy", fillcolor="rgba(97,216,238,0.055)",
                        hovertemplate=f"Received · %{{x:.1f}} {unit}<br>%{{y:.3f}}<extra></extra>")
                scatter(1, x=[seen_x.iloc[-1]], y=[seen_y.iloc[-1]], mode="markers",
                        name="current measurement", showlegend=False,
                        marker=dict(size=8, color=cyan, line=dict(color="#101925", width=2)),
                        hovertemplate=f"Current · %{{x:.1f}} {unit}<br>%{{y:.3f}}<extra></extra>")
            # Only mark the actual supplied input window, never an invented history.
            window = _input_window(feat, now_s, history_length)
            if window is not None:
                _add_vrect(fig, x0=float(window["timestamp_s"].iloc[0]) / scale,
                           x1=float(window["timestamp_s"].iloc[-1]) / scale,
                           fillcolor="rgba(97,216,238,0.025)", line_width=0, layer="below", row=1)
        sensor_x = np.asarray(ts / scale, dtype=float)
        sensor_x = sensor_x[np.isfinite(sensor_x)]
        if len(sensor_x):
            left, right = float(sensor_x.min()), float(sensor_x.max())
            fig.update_xaxes(range=[left, right + max((right - left) * 0.035, 1 / scale)], row=1, col=1)

    if now_x is not None:
        _add_vline(fig, x=now_x, line_dash="dot", line_color="rgba(97,216,238,0.65)", line_width=1, row=1)
        fig.add_annotation(x=now_x, y=1, xref="x", yref="y domain", text=f"NOW {now_x:.1f}",
                           showarrow=False, yanchor="top", yshift=-2,
                           font=dict(size=10, color=cyan), bgcolor="#101925")
        if stored_pred is not None:
            _, zone_x1, event_x = predicted_zone_x(now_s, stored_pred, dataset_id=dataset_id)
            zone_x0 = now_x
            if interval_valid:
                zone_x0 = now_x + float(prediction_interval_s[0]) / scale
                zone_x1 = now_x + float(prediction_interval_s[1]) / scale
            # A dedicated horizontal interval lane makes the uncertainty readable
            # without painting over the vibration curve. Bounds remain unmodified.
            scatter(2, x=[zone_x0, zone_x1], y=[0.54, 0.54], mode="lines+markers",
                    line=dict(color=amber, width=13), marker=dict(color=amber, size=7, symbol="line-ns"),
                    name="failure window", showlegend=False,
                    hovertemplate=f"Failure window · %{{x:.1f}} {unit}<extra></extra>")
            scatter(2, x=[event_x], y=[0.54], mode="markers", name="forecast center", showlegend=False,
                    marker=dict(size=11, symbol="diamond", color=ink, line=dict(width=2, color="#101925")),
                    hovertemplate=f"Forecast center · %{{x:.1f}} {unit}<extra></extra>")
            fig.add_annotation(x=0, y=1, xref="x2 domain", yref="y2 domain", xanchor="left",
                               text=f"<b>{zone_x0:.1f}–{zone_x1:.1f} {unit}</b>", showarrow=False,
                               font=dict(color=amber, size=12), yshift=11)
        else:
            fig.add_annotation(x=0.5, y=0.55, xref="x2 domain", yref="y2 domain",
                               text="Collecting history", showarrow=False, font=dict(size=12, color=muted))
        _add_vline(fig, x=now_x, line_color=cyan, line_width=1, row=2)

    if show_event:
        et = float(event_time_s) / scale
        scatter(2, x=[et], y=[0.54], mode="markers", name="recorded endpoint", showlegend=False,
                marker=dict(symbol="line-ns", size=27, color=coral, line=dict(width=2, color=coral)),
                hovertemplate=f"{event_reference_label} · %{{x:.1f}} {unit}<extra></extra>")
        # Separate annotation rows prevent Now, center and ground truth collisions.
        fig.add_annotation(x=1, y=1, xref="x2 domain", yref="y2 domain", xanchor="right",
                           text=f"{event_reference_label} {et:.1f}", showarrow=False,
                           font=dict(size=10, color=coral), yshift=11)

    lane_values = [float(v) for v in (now_x, zone_x0, zone_x1, event_x) if _finite(v)]
    if show_event:
        lane_values.append(float(event_time_s) / scale)
    if lane_values:
        lo, hi = min(lane_values), max(lane_values)
        pad = max((hi - lo) * 0.05, 1 / scale)
        fig.update_xaxes(range=[max(0, lo-pad), hi+pad], row=2, col=1)
    fig.update_yaxes(visible=False, range=[0, 1], fixedrange=True, row=2, col=1)
    fig.update_xaxes(showgrid=False, zeroline=False, row=2, col=1)

    if not is_bearing and _finite(pressure_limit_pa):
        fig.add_hline(y=float(pressure_limit_pa), line_dash="dash", line_color=coral,
                      annotation_text=f"{float(pressure_limit_pa):.0f} Pa", row=1, col=1)

    if second_row:
        pred = predicted_rul_by_time
        if not pred.empty and {"timestamp_s", "lower_rul_s", "upper_rul_s"} <= set(pred.columns):
            x = pd.to_numeric(pred["timestamp_s"], errors="coerce") / scale
            lower = pd.to_numeric(pred["lower_rul_s"], errors="coerce") / scale
            upper = pd.to_numeric(pred["upper_rul_s"], errors="coerce") / scale
            custom = np.column_stack([lower, upper])
            scatter(3, x=x, y=lower, mode="lines", line=dict(width=0.8, color="rgba(242,190,104,0.55)"),
                    showlegend=False, hoverinfo="skip", name="interval lower", connectgaps=False)
            scatter(3, x=x, y=upper, mode="lines", line=dict(width=0.8, color="rgba(242,190,104,0.55)"),
                    fill="tonexty", fillcolor="rgba(242,190,104,0.13)", customdata=custom,
                    showlegend=False, name="empirical forecast interval", connectgaps=False,
                    hovertemplate=f"Window · %{{customdata[0]:.1f}}–%{{customdata[1]:.1f}} {unit}<extra></extra>")
        if not pred.empty and {"timestamp_s", "predicted_rul_s"} <= set(pred.columns):
            x = pd.to_numeric(pred["timestamp_s"], errors="coerce") / scale
            forecast_y = pd.to_numeric(pred["predicted_rul_s"], errors="coerce") / scale
            scatter(3, x=x, y=forecast_y, mode="lines", name="predicted RUL", showlegend=False,
                    line=dict(color=amber, width=2), connectgaps=False,
                    hovertemplate=f"Forecast center · %{{y:.1f}} {unit}<extra></extra>")
            valid = np.isfinite(forecast_y)
            if valid.any():
                scatter(3, x=[x.loc[valid].iloc[-1]], y=[forecast_y.loc[valid].iloc[-1]], mode="markers",
                        name="latest forecast", showlegend=False,
                        marker=dict(size=7, color=amber, line=dict(color="#101925", width=1.5)),
                        hovertemplate=f"Latest center · %{{y:.1f}} {unit}<extra></extra>")
        if show_gt and actual_rul_s is not None and not pred.empty and "timestamp_s" in pred.columns:
            scatter(3, x=pd.to_numeric(pred["timestamp_s"], errors="coerce") / scale,
                    y=np.asarray(actual_rul_s, dtype=float) / scale, mode="lines",
                    name="actual RUL (evaluator overlay)", showlegend=False,
                    line=dict(color=coral, width=1.6, dash="dot"), connectgaps=False,
                    hovertemplate=f"Actual remaining · %{{y:.1f}} {unit}<extra></extra>")
        if not pred.empty and "timestamp_s" in pred.columns:
            times = pd.to_numeric(pred["timestamp_s"], errors="coerce").dropna() / scale
            if not times.empty:
                lo, hi = float(times.min()), float(times.max())
                fig.update_xaxes(range=[lo, hi+max((hi-lo)*0.04, 1 / scale)], row=3, col=1)
        has_forecast = "predicted_rul_s" in pred and np.isfinite(
            pd.to_numeric(pred["predicted_rul_s"], errors="coerce")
        ).any()
        if not has_forecast:
            fig.add_annotation(x=0.5, y=0.5, xref="x3 domain", yref="y3 domain",
                               text="Forecast history begins after warm-up", showarrow=False,
                               font=dict(size=12, color=muted))
        fig.update_xaxes(title_text="Measurement time · " + unit if is_bearing else x_title, row=3, col=1)
        fig.update_yaxes(rangemode="tozero", row=3, col=1)

    fig.update_layout(
        template="plotly_dark", height=570, paper_bgcolor="#101925", plot_bgcolor="#101925",
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, sans-serif", color=ink, size=11),
        showlegend=False, hovermode="x", dragmode="pan",
        hoverlabel=dict(bgcolor="#182536", bordercolor="#33465e", font=dict(size=12, color=ink)),
        margin=dict(t=38, b=38, l=43, r=16),
        meta={
            "predicted_zone_x0": zone_x0, "predicted_zone_x1": zone_x1,
            "predicted_event_x": event_x, "now_x": now_x,
            "show_gt": bool(show_gt), "show_actual_event": bool(show_event),
            "stored_predicted_rul_s": stored_pred,
            "interval_lower_s": prediction_interval_s[0] if interval_valid else None,
            "interval_upper_s": prediction_interval_s[1] if interval_valid else None,
        },
    )
    fig.update_xaxes(showline=False, zeroline=False, showgrid=False, tickfont=dict(size=10, color=muted),
                     nticks=6, ticks="", fixedrange=False, title_font=dict(size=10, color=muted))
    fig.update_yaxes(showline=False, zeroline=False, gridcolor="rgba(143,163,183,0.10)",
                     tickfont=dict(size=10, color=muted), nticks=4, ticks="", fixedrange=False)
    titles = ["VIBRATION · RMS" if is_bearing else "DIFFERENTIAL PRESSURE · Pa",
              "FAILURE WINDOW · " + unit.upper(), "REMAINING LIFE · " + unit.upper()]
    for row in range(1, 4 if second_row else 3):
        axis = fig.layout["yaxis" + (str(row) if row > 1 else "")]
        fig.add_annotation(x=0, y=axis.domain[1], xref="paper", yref="paper", xanchor="left",
                           yshift=32 if row == 2 else 22, text=titles[row-1], showarrow=False,
                           font=dict(size=10, color=muted))
    # Compact, fixed-position keys avoid a multi-line legend stealing plot height.
    fig.add_annotation(x=1, y=1, xref="paper", yref="paper", xanchor="right", yshift=22,
                       text=f'<span style="color:{cyan}">━ Received</span>  <span style="color:{muted}">━ Recording</span>',
                       showarrow=False, font=dict(size=10, color=muted))
    if second_row:
        fig.add_annotation(x=1, y=fig.layout.yaxis3.domain[1], xref="paper", yref="paper", xanchor="right", yshift=22,
                           text=f'<span style="color:{amber}">━ {"Range" if prediction_interval_s is not None else "Forecast"}</span>' +
                           (f'  <span style="color:{coral}">┄ Actual</span>' if show_gt else ""),
                           showarrow=False, font=dict(size=10, color=muted))

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
