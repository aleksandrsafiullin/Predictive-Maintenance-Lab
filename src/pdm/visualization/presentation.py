"""Presentation primitives for the operational fly-brain workspace."""
from html import escape
from pathlib import Path

import streamlit as st


def apply_explorer_style():
    """Mount the shared laboratory visual theme."""
    css = Path(__file__).with_name("explorer.css").read_text()
    st.markdown(f'<style>{css}</style><span class="brain-lab-shell" aria-hidden="true"></span>', unsafe_allow_html=True)


def render_hero(dataset_id, uid, split):
    title = "Bearing intelligence" if dataset_id == "bearings" else "Equipment intelligence"
    st.markdown(
        '<header class="lab-hero"><div><div class="lab-eyebrow">MODEL REPORT / RECORDED EXPERIMENT</div>'
        f'<h1>{title}<span class="lab-title-dot">.</span></h1>'
        '<p>Observed measurements → model state → failure forecast</p></div>'
        f'<div class="lab-specimen"><span class="lab-specimen-label">TEST SPECIMEN</span><strong>{escape(str(uid))}</strong>'
        f'<span>{escape(str(split))} split · recorded measurements</span></div></header>',
        unsafe_allow_html=True,
    )


def render_metrics(*, now, interval, history, node_count, playing, unit_label, scale, history_length, progress=0, predicted_rul_s=None, continuous=True, interval_method=None):
    status = "RUNNING" if playing else "PAUSED"
    window = "Collecting history" if interval is None else f"{(now + interval[0]) / scale:.1f}–{(now + interval[1]) / scale:.1f}"
    remaining = "—" if predicted_rul_s is None else f"{predicted_rul_s / scale:.1f}"
    if interval is None and predicted_rul_s is not None:
        window = f"{(now + predicted_rul_s) / scale:.1f}"
        remaining = f"{predicted_rul_s / scale:.1f}"
    suffix = unit_label if interval is not None or predicted_rul_s is not None else ""
    interval_note = interval_method or "Point forecast"
    remaining_note = (f"Point forecast · range {interval[0] / scale:.1f}–{interval[1] / scale:.1f} {unit_label}"
                      if interval is not None else "Remaining time from the current measurement")
    cards = [
        ("MEASUREMENT CLOCK", f"{now / scale:.1f}", unit_label, f"<span class='lab-status {'is-running' if playing else ''}'>{status}</span>", "cyan"),
        ("PREDICTED FAILURE WINDOW", window, suffix, interval_note, "amber"),
        ("REMAINING USEFUL LIFE", remaining, suffix, remaining_note, "amber"),
        ("NEURAL MEMORY", f"{history:,}", "samples", f"{node_count:,} computing neurons · " + (("continuous history" if continuous else f"{history_length}-measurement window") if history >= history_length else f"warmup {history}/{history_length}"), "neutral"),
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
