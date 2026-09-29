"""Single owner of the current UI theme and its design tokens.

``st.session_state["_pdm_theme"]`` is the source of truth. The Appearance widget
(``_pdm_theme_widget``) is input only and ``?theme=`` in the URL is the reload backup.
The store is a plain session key, so it survives runs that ``st.rerun()`` before the
widget is drawn (widget state is discarded on those runs).
"""
from __future__ import annotations

import html
import re

import streamlit as st

THEMES = ("light", "dark")
DEFAULT_THEME = "dark"

_STORE_KEY = "_pdm_theme"
_WIDGET_KEY = "_pdm_theme_widget"
_QUERY_KEY = "theme"

TOKENS: dict[str, dict[str, str | list]] = {
    "light": {
        "bg": "#F5F5F7",
        "sidebar": "#EDEDF0",
        "surface": "#FFFFFF",
        "surface_subtle": "#FAFAFC",
        "input": "#FFFFFF",
        "control": "#FFFFFF",
        "control_hover": "#F0F0F3",
        "text": "#1D1D1F",
        "text_secondary": "#424245",
        "muted": "#636366",
        "border": "#D2D2D7",
        "border_soft": "#E5E5EA",
        "border_strong": "#C7C7CC",
        "accent": "#0071E3",
        "accent_hover": "#0062C4",
        "accent_ink": "#FFFFFF",
        "accent_soft": "#E8F1FC",
        "accent_text": "#0058B0",
        "success": "#1E7B34",
        "warning": "#B35C00",
        "danger": "#D70015",
        "focus_ring": "rgba(0,113,227,0.28)",
        "chart_bg": "#FFFFFF",
        "chart_text": "#636366",
        "chart_grid": "#E5E5EA",
        "chart_axis": "#D2D2D7",
        "series_observed": "#6E6E73",
        "series_forecast": "#0071E3",
        "series_band": "rgba(0,113,227,0.10)",
        "series_band_line": "rgba(0,113,227,0.45)",
        "series_reference": "#1D1D1F",
        "zone_green": "#1E7B34",
        "zone_yellow": "#C28800",
        "zone_red": "#D70015",
        "zone_unknown": "#AEAEB2",
        "zone_yellow_fill": "rgba(194,136,0,0.08)",
        "zone_red_fill": "rgba(215,0,21,0.06)",
        "series_cycle": ["#0071E3", "#6E6E73", "#B35C00", "#1E7B34", "#8944AB", "#D70015"],
        "heatmap_scale": [[0, "#FFFFFF"], [0.5, "#9CC3F0"], [1, "#0071E3"]],
    },
    "dark": {
        "bg": "#0F0F11",
        "sidebar": "#161618",
        "surface": "#1C1C1E",
        "surface_subtle": "#232326",
        "input": "#1C1C1E",
        "control": "#2C2C2E",
        "control_hover": "#3A3A3C",
        "text": "#F2F2F4",
        "text_secondary": "#D1D1D6",
        "muted": "#98989D",
        "border": "#38383A",
        "border_soft": "#2C2C2E",
        "border_strong": "#48484A",
        "accent": "#0A84FF",
        "accent_hover": "#409CFF",
        "accent_ink": "#000000",
        "accent_soft": "#10263D",
        "accent_text": "#64B5FF",
        "success": "#30D158",
        "warning": "#FF9F0A",
        "danger": "#FF453A",
        "focus_ring": "rgba(10,132,255,0.35)",
        "chart_bg": "#1C1C1E",
        "chart_text": "#98989D",
        "chart_grid": "#2C2C2E",
        "chart_axis": "#38383A",
        "series_observed": "#A1A1A6",
        "series_forecast": "#0A84FF",
        "series_band": "rgba(10,132,255,0.16)",
        "series_band_line": "rgba(10,132,255,0.45)",
        "series_reference": "#F2F2F4",
        "zone_green": "#30D158",
        "zone_yellow": "#FFD60A",
        "zone_red": "#FF453A",
        "zone_unknown": "#7C7C80",
        "zone_yellow_fill": "rgba(255,214,10,0.08)",
        "zone_red_fill": "rgba(255,69,58,0.09)",
        "series_cycle": ["#0A84FF", "#A1A1A6", "#FF9F0A", "#30D158", "#BF5AF2", "#FF453A"],
        "heatmap_scale": [[0, "#1C1C1E"], [0.5, "#1F4E80"], [1, "#0A84FF"]],
    },
}

SURFACE_AND_TEXT_KEYS = (
    "bg", "sidebar", "surface", "surface_subtle", "input", "control", "control_hover",
    "text", "text_secondary", "muted", "border", "border_soft", "border_strong",
)
LIGHT_EXCLUSIVE = frozenset(str(TOKENS["light"][key]) for key in SURFACE_AND_TEXT_KEYS)
DARK_EXCLUSIVE = frozenset(str(TOKENS["dark"][key]) for key in SURFACE_AND_TEXT_KEYS)

FONT_STACK = ('-apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Inter", Roboto, '
              '"Helvetica Neue", Arial, sans-serif')
TRANSPARENT = "rgba(0,0,0,0)"


def tokens(theme: str) -> dict[str, str | list]:
    return TOKENS[_valid(theme) or DEFAULT_THEME]


def zone_colors(theme: str) -> dict[str, str]:
    """Health-zone role → line/marker/fill color. ``gray`` is the monitoring alias of ``unknown``."""
    t = tokens(theme)
    return {"green": t["zone_green"], "yellow": t["zone_yellow"], "red": t["zone_red"],
            "unknown": t["zone_unknown"], "gray": t["zone_unknown"]}


def series_cycle(theme: str) -> list[str]:
    return list(tokens(theme)["series_cycle"])


def _legend_shown(fig) -> bool:
    if fig.layout.showlegend is not None:
        return bool(fig.layout.showlegend)
    return sum(trace.showlegend is not False for trace in fig.data) > 1


def style_figure(fig, theme: str, *, height: int | None = None):
    """The only Plotly styler: design-spec §5 Charts on explicit tokens, ``template="none"``."""
    t = tokens(theme)
    top = 48 if _legend_shown(fig) else 16
    if fig.layout.title.text:
        top += 32
    layout = dict(
        template="none", paper_bgcolor=t["chart_bg"], plot_bgcolor=t["chart_bg"],
        font=dict(family=FONT_STACK, size=12, color=t["chart_text"]), colorway=list(t["series_cycle"]),
        title_font=dict(size=13, color=t["chart_text"]),
        margin=dict(l=56, r=16, b=48, t=top),
        legend=dict(orientation="h", x=0, y=1.02, xanchor="left", yanchor="bottom", bgcolor=TRANSPARENT,
                    borderwidth=0, font=dict(size=12, color=t["chart_text"])),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border"],
                        font=dict(family=FONT_STACK, size=12, color=t["text"])),
    )
    if height is not None:
        layout["height"] = height
    fig.update_layout(**layout)
    axis = dict(showline=True, linewidth=1, linecolor=t["chart_axis"], zerolinecolor=t["chart_axis"],
                gridcolor=t["chart_grid"], tickfont=dict(color=t["chart_text"]),
                title_font=dict(color=t["chart_text"]))
    fig.update_xaxes(showgrid=False, **axis)
    fig.update_yaxes(showgrid=True, **axis)
    fig.update_annotations(font_color=t["chart_text"])
    return fig


_LEGACY_ALIASES = (
    ("bg", "bg"), ("sidebar", "sidebar"), ("panel", "surface"),
    ("panel-raised", "surface_subtle"), ("surface", "surface_subtle"), ("input", "input"),
    ("control", "control"), ("control-hover", "control_hover"), ("line", "border"),
    ("line-soft", "border_soft"), ("line-strong", "border_strong"), ("track", "border_strong"),
    ("selected", "accent_soft"), ("text", "text"), ("text-soft", "text_secondary"),
    ("muted-strong", "text_secondary"), ("muted", "muted"), ("accent", "accent"),
    ("accent-soft", "accent_text"), ("accent-ink", "accent_ink"), ("amber", "warning"),
    ("amber-soft", "warning"), ("success", "success"), ("danger", "danger"),
)


def _css_var(key: str) -> str:
    return "--pdm-" + key.replace("_", "-")


def theme_css(theme: str) -> str:
    """One CSS rule with this theme's ``--pdm-*`` tokens and ``--lab-*`` aliases only."""
    theme = _valid(theme) or DEFAULT_THEME
    tokens = TOKENS[theme]
    decls = [f"{_css_var(key)}:{value};" for key, value in tokens.items() if isinstance(value, str)]
    decls += [f"--lab-{alias}:var({_css_var(key)});" for alias, key in _LEGACY_ALIASES]
    decls += [
        "--background-color:var(--pdm-bg);", "--secondary-background-color:var(--pdm-surface);",
        "--text-color:var(--pdm-text);", "--primary-color:var(--pdm-accent);",
        f"--pdm-font:{FONT_STACK};", f"color-scheme:{theme};",
    ]
    return f'body:has(.brain-lab-shell[data-theme="{theme}"]) {{{" ".join(decls)}}}\n'


def page_header(title: str, description: str) -> None:
    """Product-screen title plus its one-sentence page description (design-spec §5 Page header)."""
    st.title(title)
    st.markdown(f'<p class="pdm-page-desc">{html.escape(description)}</p>', unsafe_allow_html=True)


def empty_state(title: str, body: str):
    """Design-spec §5 empty-state card; returns the card so callers can add reasons below the body."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    card = st.container(border=True, key=f"pdm-empty-{slug}")
    with card:
        st.subheader(title, anchor=False)
        st.write(body)
    return card


def _valid(value: object) -> str | None:
    value = str(value).lower().strip() if value is not None else None
    return value if value in THEMES else None


def current_theme() -> str:
    """Resolve the theme: store, then ``?theme=``, then a one-time browser seed, then default."""
    theme = _valid(st.session_state.get(_STORE_KEY))
    if theme is None:
        theme = _valid(st.query_params.get(_QUERY_KEY))
    if theme is None:
        try:
            theme = _valid(st.context.theme.type)
        except (AttributeError, RuntimeError):
            theme = None
    if theme is None:
        theme = DEFAULT_THEME
    if st.session_state.get(_STORE_KEY) != theme:
        st.session_state[_STORE_KEY] = theme
    return theme


def set_theme(theme: str | None) -> None:
    """Store a valid theme and mirror it to the URL; ignore anything else."""
    valid = _valid(theme)
    if valid is None:
        st.session_state[_WIDGET_KEY] = current_theme()
        return
    st.session_state[_STORE_KEY] = valid
    if st.query_params.get(_QUERY_KEY) != valid:
        st.query_params[_QUERY_KEY] = valid


def _on_theme_change() -> None:
    set_theme(st.session_state.get(_WIDGET_KEY))


def render_theme_control(container=None) -> str:
    """Render the only Appearance control and return the resolved theme."""
    container = st.sidebar if container is None else container
    theme = current_theme()
    if st.session_state.get(_WIDGET_KEY) != theme:
        st.session_state[_WIDGET_KEY] = theme
    container.segmented_control(
        "Appearance", options=list(THEMES), format_func=str.title, key=_WIDGET_KEY,
        required=True, on_change=_on_theme_change, label_visibility="collapsed",
    )
    return current_theme()
