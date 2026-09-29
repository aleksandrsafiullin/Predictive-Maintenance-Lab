"""Plot styling for the project workflow's signal charts."""
from __future__ import annotations

from collections.abc import Mapping

import plotly.graph_objects as go

from pdm.ui_theme import style_figure, tokens


def style_signal_chart(fig: go.Figure, theme: str) -> go.Figure:
    return style_figure(fig, theme)


def add_threshold_layers(fig: go.Figure, thresholds: Mapping, values: list[float], theme: str) -> None:
    """Yellow/red bands and dashed limit lines shared by Data Quality and Results."""
    if not isinstance(thresholds, Mapping):
        return
    t = tokens(theme)
    available = thresholds.get("status", "available") == "available"
    yellow, red = thresholds.get("yellow"), thresholds.get("red")
    if available and yellow is not None and red is not None:
        low = min([float(yellow), float(red), *values])
        high = max([float(yellow), float(red), *values])
        padding = max((high - low) * 0.12, 0.01)
        yellow_fill, red_fill = t["zone_yellow_fill"], t["zone_red_fill"]
        if thresholds.get("direction", "above") == "below":
            fig.add_hrect(y0=low - padding, y1=float(red), fillcolor=red_fill, line_width=0)
            fig.add_hrect(y0=float(red), y1=float(yellow), fillcolor=yellow_fill, line_width=0)
        else:
            fig.add_hrect(y0=float(yellow), y1=float(red), fillcolor=yellow_fill, line_width=0)
            fig.add_hrect(y0=float(red), y1=high + padding, fillcolor=red_fill, line_width=0)
    for key, color in (("yellow", t["zone_yellow"]), ("red", t["zone_red"])):
        trace = thresholds.get(f"{key}_trace")
        if isinstance(trace, list) and trace:
            fig.add_trace(go.Scatter(x=[r.get("timestamp_s") for r in trace],
                                     y=[r.get("value") for r in trace], mode="lines",
                                     line={"color": color, "width": 1, "dash": "dash", "shape": "hv"},
                                     name=f"{key.title()} limit"))
        elif available and thresholds.get(key) is not None:
            fig.add_hline(y=float(thresholds[key]), line_color=color, line_width=1, line_dash="dash",
                          annotation_text=f"{key.title()} limit")
