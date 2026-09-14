"""Presentation primitives for the operational fly-brain workspace."""
from html import escape
from pathlib import Path

import streamlit as st


def apply_explorer_style():
    """Scope the visual theme to the mounted operational explorer."""
    css = Path(__file__).with_name("explorer.css").read_text()
    st.markdown(f'<style>{css}</style><span class="brain-lab-shell" aria-hidden="true"></span>', unsafe_allow_html=True)


def render_hero(dataset_id, uid, split):
    title = "Bearing intelligence" if dataset_id == "bearings" else "Equipment intelligence"
    st.markdown(
        '<header class="lab-hero"><div><div class="lab-eyebrow">CONNECTOME INTELLIGENCE / RECORDED EXPERIMENT</div>'
        f'<h1>{title}<span class="lab-title-dot">.</span></h1>'
        '<p>Recorded vibration → neural state → failure window</p></div>'
        f'<div class="lab-specimen"><span class="lab-specimen-label">TEST SPECIMEN</span><strong>{escape(str(uid))}</strong>'
        f'<span>{escape(str(split))} split · recorded measurements</span></div></header>',
        unsafe_allow_html=True,
    )


def render_metrics(*, now, interval, history, node_count, playing, unit_label, scale, history_length, progress=0):
    status = "RUNNING" if playing else "PAUSED"
    window = "Collecting history" if interval is None else f"{(now + interval[0]) / scale:.1f}–{(now + interval[1]) / scale:.1f}"
    remaining = "—" if interval is None else f"{interval[0] / scale:.1f}–{interval[1] / scale:.1f}"
    suffix = "" if interval is None else unit_label
    cards = [
        ("MEASUREMENT CLOCK", f"{now / scale:.1f}", unit_label, f"<span class='lab-status {'is-running' if playing else ''}'>{status}</span>", "cyan"),
        ("PREDICTED FAILURE WINDOW", window, suffix, "Operating time · empirical interval", "amber"),
        ("REMAINING USEFUL LIFE", remaining, suffix, "Range from the current measurement", "amber"),
        ("NEURAL MEMORY", f"{history:,}", "samples", f"{node_count:,} computing neurons · " + ("continuous history" if history >= history_length else f"warmup {history}/{history_length}"), "neutral"),
    ]
    progress = min(100.0, max(0.0, float(progress)))
    markup = f'<style>.st-key-lab_controls{{--lab-progress:{progress:.4f}%;}}</style><div class="lab-metrics">'
    for label, value, unit, detail, tone in cards:
        value_class = " lab-value-warmup" if value == "Collecting history" else ""
        markup += (
            f'<section class="lab-metric lab-tone-{tone}"><div class="lab-metric-label">{label}</div>'
            f'<div class="lab-metric-value{value_class}">{escape(value).replace("–", "<wbr>–")}<small>{escape(unit)}</small></div>'
            f'<div class="lab-metric-detail">{detail}</div></section>'
        )
    st.markdown(markup + '</div>', unsafe_allow_html=True)


def render_panel_header(number, title, subtitle, badge):
    st.markdown(
        '<div class="lab-panel-heading"><div><span class="lab-panel-number">'
        f'{escape(number)}</span><strong>{escape(title)}</strong><span class="lab-panel-subtitle">{escape(subtitle)}</span>'
        f'</div><span class="lab-panel-badge">{escape(badge)}</span></div>', unsafe_allow_html=True,
    )
