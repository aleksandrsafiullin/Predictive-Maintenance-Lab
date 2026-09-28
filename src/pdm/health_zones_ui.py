"""Model Report → Health zones: green / yellow / red status for a bearing over time."""
from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm.health_zones import (
    ZONE_ACTIONS,
    ZONE_LABELS,
    ZONES,
    list_zone_runs,
    load_zone_run,
    predict_unit,
    rule_baseline,
)

ZONE_COLORS = {"green": "#2ecc71", "yellow": "#f1c40f", "red": "#e74c3c"}
METHODS = {
    "Signal rule (recommended)": "rule",
    "GRU classifier (experimental)": "gru",
}


@st.cache_resource(show_spinner=False)
def _features(dataset_version: str):
    from pdm.data.prepare import load_processed

    data = load_processed("bearings", dataset_version)
    if data.get("dataset_version") != dataset_version:
        raise ValueError(f"Processed snapshot does not match zone run: {dataset_version}")
    return data["features"], data["split"], data.get("dataset_version")


@st.cache_resource(show_spinner=False)
def _run(rdir: str):
    return load_zone_run(Path(rdir))


@st.cache_data(show_spinner=False)
def _predictions(rdir: str, method: str, uid: str, dataset_version: str) -> pd.DataFrame:
    features, _, _ = _features(dataset_version)
    model, scaler, meta, metrics = _run(rdir)
    cfg = meta["config"]
    unit = features[features["unit_id"].astype(str) == uid]
    if method == "rule":
        return rule_baseline(features, [uid], cfg, float(metrics["rule_baseline_red_ratio"]))
    return predict_unit(model, scaler, cfg, unit)


def _segments(t_min: np.ndarray, zone: np.ndarray):
    """Contiguous (start, end, zone) spans; each point covers until the next one."""
    if len(t_min) == 0:
        return []
    edges = np.append(t_min, t_min[-1] + (t_min[-1] - t_min[-2] if len(t_min) > 1 else 1.0))
    out, start = [], 0
    for i in range(1, len(zone) + 1):
        if i == len(zone) or zone[i] != zone[start]:
            out.append((edges[start], edges[i], int(zone[start])))
            start = i
    return out


def _status_card(zone: int, now_min: float, recording_end_min: float | None, ratio: float, probs: dict | None) -> None:
    name = ZONES[zone]
    color = ZONE_COLORS[name]
    extra = ""
    if probs:
        extra = " · ".join(f"{ZONES[k].capitalize()} {100 * p:.0f}%" for k, p in enumerate(probs))
        extra = f"<div style='opacity:.8;font-size:.85rem;margin-top:.35rem'>GRU scores (uncalibrated): {escape(extra)}</div>"
    truth = ""
    if recording_end_min is not None:
        truth = (f"<div style='opacity:.8;font-size:.85rem'>Recorded experiment end: "
                 f"{recording_end_min:.0f} min ({recording_end_min - now_min:.0f} min from this point)</div>")
    st.markdown(
        f"""<div style="border-radius:14px;padding:18px 22px;margin:6px 0 14px 0;
                    background:{color}22;border:2px solid {color};display:flex;gap:22px;align-items:center">
              <div style="width:64px;height:64px;border-radius:50%;background:{color};
                          box-shadow:0 0 24px {color};flex:none"></div>
              <div>
                <div style="font-size:.8rem;letter-spacing:.12em;opacity:.8">BEARING STATUS AT {now_min:.0f} MIN</div>
                <div style="font-size:1.9rem;font-weight:700;color:{color}">{name.upper()} · {escape(ZONE_LABELS[name])}</div>
                <div style="font-size:1rem">{escape(ZONE_ACTIONS[name])}</div>
                <div style="opacity:.8;font-size:.85rem;margin-top:.35rem">
                  Vibration RMS: {ratio:.2f}× this bearing's initial baseline</div>
                {extra}{truth}
              </div>
            </div>""",
        unsafe_allow_html=True,
    )


def _chart(pred: pd.DataFrame, index: int, show_truth: bool) -> go.Figure:
    t_min = (pred["timestamp_s"].to_numpy(float) - float(pred["timestamp_s"].iloc[0])) / 60.0
    rms = pred["combined_rms"].to_numpy(float)
    fig = go.Figure()
    for x0, x1, z in _segments(t_min[: index + 1], pred["zone"].to_numpy()[: index + 1]):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=ZONE_COLORS[ZONES[z]], opacity=0.28, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=t_min[: index + 1], y=rms[: index + 1], name="Vibration RMS (max of axes)",
                             line=dict(color="#e8eef5", width=2)))
    if show_truth:
        fig.add_trace(go.Scatter(x=t_min[index:], y=rms[index:], name="Recording (future)",
                                 line=dict(color="#7f8c9a", width=1, dash="dot")))
        ymax = float(np.nanmax(rms)) * 1.08
        for x0, x1, z in _segments(t_min, pred["true_zone"].to_numpy()):
            fig.add_shape(type="rect", x0=x0, x1=x1, y0=ymax * 0.97, y1=ymax, fillcolor=ZONE_COLORS[ZONES[z]],
                          line_width=0, opacity=0.9)
        fig.add_annotation(x=t_min[0], y=ymax, text="Sensor-rule reference zones", showarrow=False, xanchor="left",
                           yanchor="bottom", font=dict(size=11))
    fig.add_vline(x=t_min[index], line=dict(color="#5bd3f5", width=2, dash="dash"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Operating time, min", yaxis_title="RMS, g",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def _metrics_table(metrics: dict) -> pd.DataFrame:
    rows = []
    for split in ("validation", "test"):
        for key, label in (("rule_baseline", "Signal rule"), ("model", "GRU classifier")):
            m = metrics.get(split, {}).get(key)
            if not m:
                continue
            rec = m["unit_balanced_recall"]
            for uid, u in m["per_unit"].items():
                rows.append({
                    "split": split, "method": label, "bearing": uid, "recorded duration, min": round(u["life_min"]),
                    "first red, min before recording end": u["first_red_min_before_endpoint"],
                    "min green while signal reference red": u["minutes_green_while_red"],
                    "signal-rule agreement": round(u["balanced_accuracy"], 3),
                })
            rows.append({
                "split": split, "method": label, "bearing": "ALL (bearing-balanced)", "recorded duration, min": None,
                "first red, min before recording end": m["median_first_red_min_before_endpoint"],
                "min green while signal reference red": None,
                "signal-rule agreement": round(m["unit_balanced_balanced_accuracy"], 3),
                "recall green/yellow/red": "/".join("—" if rec[z] is None else f"{rec[z]:.2f}" for z in ZONES),
            })
    return pd.DataFrame(rows)


def screen_health_zones(dataset_id: str) -> None:
    st.subheader("Health zones")
    if dataset_id != "bearings":
        st.info("Health zones are available for bearings (run-to-failure vibration data). Select Bearings in the sidebar.")
        return
    runs = list_zone_runs("bearings")
    if not runs:
        st.info("No health-zone model yet. Train one from a terminal:")
        st.code(".venv/bin/python -m pdm zones-train", language="bash")
        return
    legacy = [run for run in runs if not run.get("compatible", False)]
    runs = [run for run in runs if run.get("compatible", False)]
    if legacy and not runs:
        st.warning("Saved zone runs use an obsolete or incomplete zone definition and cannot be replayed under the current sensor-zone policy. Retrain the zone model:")
        st.code(".venv/bin/python -m pdm zones-train", language="bash")
        return
    if legacy:
        st.info(f"{len(legacy)} older or incomplete zone run(s) are hidden. Retrain them with the current sensor-zone policy to replay.")
    st.caption("🟢 Baseline-level vibration · 🟡 Persistent rise above baseline · 🔴 High vibration RMS. "
               "Zones are sensor states separate from remaining life. Signal thresholds are provisional; "
               "the experiment-end proxy is used only for retrospective timing summaries. "
               "Weak labels and 15 bearings do not establish industrial operating limits.")
    c1, c2, c3 = st.columns([2, 2, 2])
    run = c1.selectbox("Zone model", [r["run_id"] for r in runs], key="hz_run")
    rdir = next(r["dir"] for r in runs if r["run_id"] == run)
    try:
        _, _, meta, metrics = _run(rdir)
    except Exception as exc:
        st.error(f"This zone run is incomplete or damaged and cannot be loaded. Retrain it with the current sensor-zone policy: {exc}")
        st.code(".venv/bin/python -m pdm zones-train", language="bash")
        return
    version = meta.get("dataset_version")
    if not version:
        st.error("This zone run has no saved dataset version. Retrain it before replaying predictions.")
        return
    try:
        features, split, _ = _features(version)
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"The processed dataset snapshot for this zone run is unavailable: {exc}")
        return
    if any(split.get(s) != meta.get("split", {}).get(s) for s in ("train", "validation", "test")):
        st.error("The saved zone run split does not match its processed dataset snapshot.")
        return
    split_of = {u: s for s in ("train", "validation", "test") for u in split.get(s, [])}
    method = METHODS[c2.radio("Zone engine", list(METHODS), key="hz_method")]
    units = split.get("test", []) + split.get("validation", []) + split.get("train", [])
    uid = c3.selectbox("Bearing", units, key="hz_unit", format_func=lambda u: f"{u} ({split_of.get(u, '?')})")
    if split_of.get(uid) == "train":
        st.warning("This bearing was used to train the model — its zones are optimistic. Use validation/test bearings to judge quality.")

    pred = _predictions(rdir, method, str(uid), version)
    n = len(pred)
    play_key, cursor_key = f"hz_play:{uid}", f"hz_cursor:{rdir}:{uid}"
    st.session_state.setdefault(play_key, False)
    st.session_state.setdefault(cursor_key, min(n - 1, 5))
    step = max(1, n // 120)

    @st.fragment(run_every=0.6 if st.session_state[play_key] else None)
    def timeline():
        b1, b2, b3, show = st.columns([1, 1, 1, 2])
        if b1.button("▶ Play", key=f"hz_start:{uid}", type="primary", width="stretch"):
            if st.session_state[cursor_key] >= n - 1:
                st.session_state[cursor_key] = 0
            st.session_state[play_key] = True
            st.rerun()
        if b2.button("⏸ Pause", key=f"hz_pause:{uid}", width="stretch"):
            st.session_state[play_key] = False
            st.rerun()
        if b3.button("⟲ Reset", key=f"hz_reset:{uid}", width="stretch"):
            st.session_state[play_key] = False
            st.session_state[cursor_key] = 0
            st.rerun()
        show_truth = show.checkbox("Show sensor-rule reference", value=True, key="hz_truth")
        if st.session_state[play_key]:
            st.session_state[cursor_key] = min(n - 1, st.session_state[cursor_key] + step)
            if st.session_state[cursor_key] >= n - 1:
                st.session_state[play_key] = False
        index = st.slider("Measurement (minutes since start)", 0, n - 1, key=cursor_key)
        row = pred.iloc[index]
        base = float(np.median(pred["combined_rms"].to_numpy()[:int(meta["config"].get("baseline_n", 5))]))
        probs = [row["p_green"], row["p_yellow"], row["p_red"]] if "p_red" in pred.columns else None
        t0 = float(pred["timestamp_s"].iloc[0])
        recording_end = (float(pred["timestamp_s"].iloc[-1]) - t0) / 60.0 if show_truth else None
        _status_card(int(row["zone"]), (float(row["timestamp_s"]) - t0) / 60.0, recording_end,
                     float(row["combined_rms"]) / base, probs)
        st.plotly_chart(_chart(pred, index, show_truth), width="stretch", key=f"hz_chart:{uid}")

    timeline()
    with st.expander("How good is it? Validation and test results"):
        cfg = meta["config"]
        st.markdown(
            f"- **Signal reference zones**: green below the persistent-rise threshold; yellow after max-axis RMS "
            f"exceeds {cfg['onset_ratio']:.2f}× baseline (and {cfg['onset_sigma']:.0f} SD above baseline) for "
            f"{cfg['onset_persist']} consecutive measurements; red at {cfg['red_ratio']:.2f}× baseline RMS.\n"
            "- These are weak vibration-derived reference labels, not independent expert diagnoses.\n"
            f"- **Endpoint timing** = minutes from first red to the end of this recorded experiment; "
            "the endpoint is a proxy, not a confirmed failure timestamp. It does not define zone labels.\n"
            f"- **Yellow threshold baseline** uses the bearing's first {cfg['baseline_n']} measurements; "
            f"its threshold also uses {cfg['onset_sigma']:.0f} SD above baseline.\n"
            f"- **GRU classifier**: {meta['config']['hidden_size']}-unit GRU over the last {cfg['history']} measurements, "
            f"epoch {metrics.get('best_epoch')} selected on validation; zones smoothed "
            f"(escalate after {cfg['escalate_n']}, de-escalate after {cfg['deescalate_n']} consistent predictions).\n"
            "- GRU scores measure agreement with the weak labels, not independently validated fault detection. "
            "The table reports zone agreement and endpoint timing separately."
        )
        st.dataframe(_metrics_table(metrics), hide_index=True, width="stretch")
