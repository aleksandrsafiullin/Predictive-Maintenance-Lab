"""Full-horizon trend training and honest saved-run replay in project pages."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import plotly.graph_objects as go
import streamlit as st

from pdm import stable_forecast as stable
from pdm.io_util import read_json
from pdm.long_forecast import PROTOCOL
from pdm.long_forecast_data import MIN_HISTORY, read_part, segments, source_for
from pdm.long_forecast_run import (
    alive,
    launch,
    list_runs,
    load_run,
    read_status,
    replay,
    request_stop,
)
from pdm.projects import project_store
from pdm.ui_theme import style_figure, tokens

LABELS = {
    "robust_trend": "Robust trend",
    "trend_boosting": "Trend + Boosting correction",
    "trend_gru": "Trend + GRU correction",
}
LABELS.update(stable.LABELS)
LABELS["all"] = "GRU + LSTM + MaleCNS"


def enabled(project):
    if project.get("source_kind", "").startswith("synthetic_"):
        return True
    key = "long_forecast_mode:" + project["project_id"]
    if key in st.session_state:
        return st.session_state[key]
    rid = project.get("selected_run_id")
    if rid:
        if read_json(project_store().run_path(project["project_id"], rid) / "manifest.json").get(
            "protocol"
        ) in {PROTOCOL, *stable.PROTOCOLS}:
            return True
    mapped = (
        (project.get("source_manifest") or {})
        .get("snapshots", {})
        .get(project.get("active_snapshot_id"))
    )
    if mapped and project.get("source_kind") != "hse_filters":
        from pathlib import Path

        return (Path(mapped) / "snapshot.json").is_file()
    return False


def _label(run):
    cfg = run.get("config", {})
    name = LABELS.get(run.get("engine_id"), run.get("engine_id", "Model").upper())
    time = (
        datetime.fromisoformat(run["created_at"])
        .astimezone(ZoneInfo("Asia/Almaty"))
        .strftime("%d %b, %H:%M:%S")
    )
    family = " · observed acceleration" if cfg.get("baseline_family") == "acceleration_v2" else ""
    archived = "Archive · " if run.get("protocol") != stable.PROTOCOL else ""
    width = cfg.get("width", cfg.get("width_budget", 0.3)) * 50
    return f"{archived}{name}{family} · training ±{width:g}% · seed {cfg.get('seed', '?')} · {time}"


@st.fragment(run_every=2)
def _progress(project_id):
    state = read_status(project_id)
    if alive(state):
        engine = (state.get("progress") or {}).get("engine", state.get("engine"))
        st.info(f"Training · {LABELS.get(engine, engine)}")
        if state.get("progress"):
            progress = state["progress"]
            if "epoch" in progress:
                st.caption(
                    f"Epoch {progress['epoch']} · weakest Validation quarter {progress['worst_quarter_containment']:.1%}"
                )
            elif progress.get("stage") == "reservoir":
                st.caption(
                    f"Preparing observed histories · {progress['role'].title()} {progress['done']}/{progress['total']}"
                )
            else:
                st.caption(progress.get("message", "Saving completed model"))
        if st.button("Stop training", key="long_stop:" + project_id):
            request_stop(project_id)
    elif state.get("status") == "completed":
        st.success(
            f"Saved {len(state.get('runs', [])) or 1} model(s). Open Results to replay them."
        )
        if st.button("Open Results", key="long_results:" + project_id):
            st.session_state.pop("long_model:" + project_id, None)
            st.session_state["project_step"] = "Results"
            st.rerun()
    elif state.get("status") == "failed":
        st.error(state.get("error", "Training failed"))
    elif state.get("status") == "stopped":
        st.info(
            "Training stopped. Completed models remain available; unfinished models were not saved."
        )
    elif state.get("status") in {"queued", "running", "stopping"}:
        st.warning("Training process ended before publishing a completed model.")


def training(project, source):
    st.title("Training")
    st.caption("Training task · Trend corridor")
    train_frame = read_part(source, "train")
    support = stable.supported_horizon(
        train_frame, read_part(source, "validation"), source["cadence_s"]
    )
    horizon = support["horizon"]
    st.info(
        f"Forecast · {horizon} observations · {horizon * source['cadence_s'] / 3600:g} hours. History: all available continuous measurements; minimum 60 to start."
    )
    st.caption(
        f"Last forecast point is observed in {support['train']['support_at_last_lead']}/{support['train']['total_units']} Train units and {support['validation']['support_at_last_lead']}/{support['validation']['total_units']} Validation units. This is data support, not an accuracy guarantee."
    )
    st.caption(
        "GRU, LSTM and MaleCNS. Smooth observed-trend parameters; ±22.5% corridor, 50% wider than ±15%. Selection prioritizes the weakest horizon quarter. Train and Validation are used; Test remains separate."
    )
    with st.form("long_training:" + project["project_id"]):
        engine = st.selectbox(
            "Train",
            ["all", *stable.ENGINES],
            format_func=lambda e: "All three · GRU, LSTM, MaleCNS" if e == "all" else LABELS[e],
        )
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1, value=21, step=1)
        epochs = st.number_input("Maximum epochs", min_value=1, max_value=500, value=80, step=1)
        patience = st.number_input(
            "Epochs without improvement", min_value=1, max_value=200, value=20, step=1
        )
        st.caption(
            "Width ±22.5% · batch 64 · learning rate 0.001. GRU/LSTM: 128 hidden units, 60 aggregated history blocks. MaleCNS: fixed reservoir of all 166,700 neurons, trained readout, 16 history blocks. Recent 60/120/240 and full-history descriptors accompany each encoder."
        )
        submitted = st.form_submit_button(
            "Start training", disabled=alive(read_status(project["project_id"]))
        )
    if submitted:
        try:
            launch(
                project["project_id"],
                engine,
                int(seed),
                dict(epochs=int(epochs), patience=int(patience)),
            )
            st.rerun()
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    _progress(project["project_id"])


def results(project, source, theme="dark"):
    st.title("Results")
    runs = [
        row
        for row in list_runs(project["project_id"])
        if row.get("snapshot_id") == source["snapshot_id"]
    ]
    if not runs:
        st.info("Train a model first.")
        return
    ids = [row["run_id"] for row in runs]
    selected = project.get("selected_run_id")
    rid = st.selectbox(
        "Saved model",
        ids,
        index=ids.index(selected) if selected in ids else 0,
        format_func=lambda value: _label(runs[ids.index(value)]),
        key="long_model:" + project["project_id"],
    )
    frozen = load_run(project["project_id"], rid)
    manifest, _ = frozen
    cfg = manifest["config"]
    h = cfg.get("horizon", cfg.get("max_horizon"))
    width = cfg.get("width", cfg.get("width_budget", 0.3)) * 50
    history = (
        "all available continuous history"
        if manifest.get("protocol") in stable.PROTOCOLS
        else f"saved {cfg.get('history_capacity', cfg.get('history_length', 60))}-observation history policy"
    )
    st.caption(
        f"Issued horizon · {h} measurements · {h * source['cadence_s'] / 3600:g} hours · saved training corridor ±{width:g}% · {history}"
    )
    if manifest.get("protocol") != stable.PROTOCOL:
        st.warning(
            "Archive model: this is the original saved forecast. Train GRU, LSTM and MaleCNS to use the updated history, horizon, width and stability rules."
            if manifest.get("protocol") not in stable.PROTOCOLS
            else "Earlier forecast version: uses its saved parameters and formula. Current GRU, LSTM and MaleCNS are available in Training."
        )
    part = st.selectbox(
        "Data set",
        ["test", "validation", "train"],
        format_func=str.title,
        key="long_part:" + project["project_id"],
    )
    frame = read_part(source, part)
    options = sorted(frame.unit_id.unique())
    if not options:
        st.info("This role has no observations.")
        return
    uid = st.selectbox("Unit", options, key="long_unit:" + project["project_id"])
    unit_frame = frame[frame.unit_id.eq(uid)]
    origins = [
        float(t)
        for _, segment in segments(unit_frame, source["cadence_s"])
        for t in segment.timestamp_s.iloc[MIN_HISTORY - 1 :]
    ]
    if not origins:
        st.info("No continuous 60-observation history.")
        return
    origin = st.select_slider(
        "Measurements received · Now",
        origins,
        value=origins[len(origins) // 3],
        key=f"long_now:{project['project_id']}:{part}:{uid}",
    )
    show_future = st.checkbox("Show future actual · hidden from model", value=True)
    from pdm.corridor_calibration import load_active
    active_calibration = load_active(project["project_id"], rid, manifest)
    use_calibration = active_calibration is not None
    if active_calibration is not None:
        use_calibration = st.checkbox("Use saved Calibration corridor", value=True,
                                      key=f"long_calibration:{project['project_id']}:{rid}")
    with st.spinner("Predicting the complete saved horizon"):
        forecast = replay(project["project_id"], rid, uid, origin, part, frozen=frozen,
                          use_calibration=use_calibration)
    if forecast.get("calibration"):
        cal = forecast["calibration"]
        st.caption(f"Calibration corridor · full width {100 * cal['width']:.1f}% (±{50 * cal['width']:.1f}%) · limits {100 * cal['settings']['min_width']:g}–{100 * cal['settings']['max_width']:g}% · empirical Calibration coverage {100 * cal['point_coverage']:.1f}%")
    st.caption(
        f"Measurements used: {forecast['history_observations']} · future actual is hidden from the model. Forecast quality remains exploratory."
    )
    fig = go.Figure()
    chart_tokens = tokens(theme)
    for _, segment in segments(unit_frame, source["cadence_s"]):
        past = segment[segment.timestamp_s.le(origin)]
        future = segment[segment.timestamp_s.gt(origin)]
        if len(past):
            fig.add_trace(
                go.Scatter(
                    x=past.timestamp_s,
                    y=past.signal,
                    mode="lines+markers",
                    marker_size=3,
                    line=dict(color=chart_tokens["series_reference"], width=1),
                    name="Measurements received",
                    showlegend=not any(t.name == "Measurements received" for t in fig.data),
                )
            )
        if show_future and len(future):
            fig.add_trace(
                go.Scatter(
                    x=future.timestamp_s,
                    y=future.signal,
                    mode="lines",
                    line=dict(color=chart_tokens["series_future_actual"], width=1),
                    name="Future actual · hidden from model",
                    showlegend=not any(t.name.startswith("Future actual") for t in fig.data),
                )
            )
    out, times = forecast["outputs"], forecast["times"]
    fig.add_trace(
        go.Scatter(x=times, y=out[:, 0], line_width=0, showlegend=False, hoverinfo="skip")
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=out[:, 2],
            line=dict(color="#168bff", width=0.5),
            fill="tonexty",
            fillcolor="rgba(22,139,255,.18)",
            name="Trend corridor",
        )
    )
    fig.add_trace(
        go.Scatter(x=times, y=out[:, 1], line=dict(color="#168bff", width=3), name="Trend center")
    )
    for name, color in (("yellow", "#e6c125"), ("red", "#ff4638")):
        limit = (manifest.get("source") or {}).get(name, cfg.get(name, source.get(name)))
        if limit is not None:
            fig.add_hline(
                y=limit,
                line=dict(color=color, dash="dash"),
                annotation_text=name.title() + " limit",
            )
    fig.add_vline(x=origin, line=dict(color="#cccccc", dash="dot"))
    fig.update_layout(
        height=440,
        margin=dict(l=30, r=30, t=40, b=40),
        xaxis_title="Time (s)",
        yaxis_title=f"{source['label']} ({source['unit']})",
        legend=dict(orientation="h", y=1.15),
    )
    style_figure(fig, theme)
    fig.update_layout(title_text="")
    st.plotly_chart(fig, width="stretch", key="long_plot")


def render_step(project, step, theme="dark"):
    try:
        source = source_for(project["project_id"])
        if source["kind"] == "project" and step in {"Training", "Results"}:
            if st.button(
                "Open previous saved models"
                if step == "Results"
                else "Use previous training workflow"
            ):
                st.session_state["long_forecast_mode:" + project["project_id"]] = False
                st.rerun()
        if step == "Training":
            training(project, source)
        elif step == "Results":
            results(project, source, theme)
        elif step == "Data Quality":
            from pdm.project_data_profile import quality_provider
            from pdm.project_quality_ui import render_quality

            provider = quality_provider(project)
            ready = render_quality(provider.view, theme, provider=provider) if provider.snapshot else render_quality(provider.view, theme)
            if ready and st.button("Continue to Training"):
                st.session_state["project_step"] = "Training"
                st.rerun()
        else:
            st.title(step)
            st.caption("Saved observed data · immutable snapshot")
            for role in ("train", "validation", "test"):
                part = read_part(source, role)
                st.write(role.title(), f"{part.unit_id.nunique()} units · {len(part)} measurements")
            if st.button("Continue to Training"):
                st.session_state["project_step"] = "Training"
                st.rerun()
    except (ValueError, OSError, KeyError) as exc:
        st.error(f"Full-horizon workflow: {exc}")
