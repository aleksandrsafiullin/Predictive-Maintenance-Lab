"""Presentation primitives for the operational fly-brain workspace."""
from pathlib import Path

import streamlit as st

from pdm.ui_theme import DEFAULT_THEME, THEMES, theme_css


def apply_explorer_style(theme: str = "dark"):
    """Mount the shared light or dark workspace theme."""
    theme = str(theme).lower().strip()
    if theme not in THEMES:
        theme = DEFAULT_THEME
    css = theme_css(theme) + Path(__file__).with_name("explorer.css").read_text(encoding="utf-8")
    st.markdown(
        f'<style>{css}</style><span class="brain-lab-shell" data-theme="{theme}" aria-hidden="true"></span>',
        unsafe_allow_html=True,
    )


def render_panel_header(number, title, subtitle):
    from html import escape
    st.markdown(
        '<div class="lab-panel-heading"><div><span class="lab-panel-number">'
        f'{escape(number)}</span><strong>{escape(title)}</strong><span class="lab-panel-subtitle">{escape(subtitle)}</span>'
        '</div></div>', unsafe_allow_html=True,
    )
