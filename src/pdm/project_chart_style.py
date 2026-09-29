"""Plot styling for the project workflow's signal charts."""
from __future__ import annotations

import plotly.graph_objects as go

from pdm.ui_theme import style_figure


def style_signal_chart(fig: go.Figure, theme: str) -> go.Figure:
    return style_figure(fig, theme)
