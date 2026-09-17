"""Condition-first views inside the existing four-screen Streamlit application."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm.cli import spawn_worker
from pdm.data.prepare import load_processed
from pdm.io_util import read_json, sha256_file
from pdm.monitoring.bundle import bundle_root, load_bundle
from pdm.monitoring.contracts import unit_verification
from pdm.monitoring.journal import FEEDBACK, append_feedback
from pdm.monitoring.study import freeze_study, plan_study
from pdm.paths import project_root, runs_root
from pdm.worker import read_status, request_stop, worker_alive

COLORS = {"green": "#5bc98d", "yellow": "#efbf54", "red": "#f27580", "gray": "#9aa8b8"}
LABELS = {"green": "Normal", "yellow": "Attention required", "red": "Urgent action required", "gray": "Assessment unavailable"}


def bundles_for(dataset_id):
    result = []
    for path in sorted((runs_root() / "monitoring_bundles").glob("*/monitoring_bundle.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        payload = read_json(path)
        if payload["profile"]["dataset_id"] == dataset_id:
            result.append(payload)
    return result


def quality_details(dataset_id):
    with st.expander("Monitoring units, endpoints and origin identity"):
        st.json(unit_verification(dataset_id))
        data = load_processed(dataset_id)
        cols = [c for c in ("unit_id", "origin_unit_id", "author_split", "event_source", "endpoint_definition") if c in data["units"]]
        st.dataframe(data["units"][cols], hide_index=True, width="stretch")
        st.caption("Runtime checks run again before inference. Invalid and stale observations remain visible; hard limits require a usable signal channel.")
        for bundle in bundles_for(dataset_id)[:1]:
            st.json({"reference_status": bundle["reference"]["status"],
                     "selected_segments": bundle["reference"]["selected_segments"],
                     "unavailable_reasons": bundle["reference"]["rejected"]}, expanded=False)


def training_controls(dataset_id):
    with st.expander("Condition and sensor forecast study"):
        config = project_root() / "configs" / f"condition_{dataset_id}.yaml"
        st.caption("Task: event forecast + sensor forecast + monitoring policy. Train groups select candidates; validation is stopping/diagnostics. No test tuning.")
        try:
            plan = plan_study(config)
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            return
        st.write(f"{plan['planned_fits']} fitting jobs for this dataset; paired profiles plan 44 total (limit 48). Each horizon/quantile fit is counted.")
        st.dataframe(pd.DataFrame(plan["fit_jobs"])[["id", "kind", "status"]], hide_index=True, width="stretch")
        studies = [p.parent for p in sorted((runs_root() / "condition_studies").glob("*/study_manifest.json"), reverse=True)
                   if read_json(p)["config"]["dataset_id"] == dataset_id]
        selected = st.selectbox("Condition study", ["New study", *[p.name for p in studies]])
        left, right = st.columns(2)
        if left.button("Run / resume condition study", disabled=worker_alive()):
            spawn_worker({"kind": "condition_study", "config": str(config), "study_id": None if selected == "New study" else selected})
            st.rerun()
        if right.button("Stop condition study", disabled=not worker_alive()):
            request_stop()
        if selected != "New study":
            manifest = read_json(runs_root() / "condition_studies" / selected / "study_manifest.json")
            st.dataframe(pd.DataFrame(manifest["fit_jobs"]), hide_index=True, width="stretch")
            if st.button("Freeze monitoring bundle", disabled=worker_alive()):
                try:
                    bundle = freeze_study(selected)
                    st.success("Frozen bundle " + bundle["bundle_id"])
                except (ValueError, OSError) as exc:
                    st.error(str(exc))
        status = read_status()
        if status.get("kind") == "condition_study":
            st.caption(status.get("message", status.get("status", "")))


@st.cache_data(show_spinner=False)
def _read_observations(path, digest):
    del digest
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def screen_condition_report(dataset_id):
    st.header("Condition & Forecast")
    st.caption("Bearing condition from vibration" if dataset_id == "bearings" else "Filter condition from differential pressure")
    st.caption("Real model states and causal forecasts from recorded measurements — model-state details remain in Experimental / Model activity.")
    bundles = bundles_for(dataset_id)
    if not bundles:
        st.info("Assessment unavailable: no frozen monitoring bundle. Existing event models remain available in Diagnostics / Model activity. A RUL-only model does not produce a sensor forecast.")
        return
    selected = st.selectbox("Monitoring bundle", [b["bundle_id"] for b in bundles])
    try:
        bundle, _, _ = load_bundle(selected, load_models=False)
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return
    root = bundle_root(selected)
    evaluations = [read_json(p) for p in sorted((root / "evaluations").glob("*/evaluation.json"), reverse=True)]
    if not evaluations:
        st.info("No saved monitoring replay. Run a full-history evaluation through the worker.")
        if st.button("Evaluate monitoring on validation", disabled=worker_alive()):
            spawn_worker({"kind": "monitor_evaluate", "bundle_id": selected, "split_name": "validation"})
            st.rerun()
        return
    ids = {e["eval_id"]: e for e in evaluations}
    eid = st.selectbox("Monitoring evaluation", list(ids), format_func=lambda e: ids[e]["split"] + " · " + e)
    evaluation = ids[eid]
    directory = root / "evaluations" / eid
    for name, sha in evaluation["artifact_hashes"].items():
        # Feedback intentionally evolves append-only; issued predictions never do.
        if name != "feedback.jsonl" and sha256_file(directory / name) != sha:
            st.error("Saved monitoring artifact changed: " + name)
            return
    path = directory / "observations.jsonl"
    observations = _read_observations(str(path), sha256_file(path))
    uid = st.selectbox("Monitored unit", sorted({r["unit_id"] for r in observations}))
    rows = [r for r in observations if r["unit_id"] == uid]
    data = load_processed(dataset_id, bundle["dataset_version"])
    measurements = data["features"].loc[lambda f: f.unit_id == uid].sort_values("timestamp_s")
    key = "condition:" + selected + ":" + eid + ":" + uid
    state = st.session_state.setdefault(key, {"step": 0, "playing": False})
    fragment = st.fragment(run_every=.75 if state["playing"] else None)
    fragment(lambda: _render_replay(rows, measurements, bundle, directory, key))()
    with st.expander("Evaluation and exact saved exports"):
        st.caption(evaluation["evaluation_status"] + " · no operational promotion")
        st.json(read_json(directory / "monitoring_results.json"))
        for name in ("event_results.csv", "signal_results.csv", "per_unit_results.csv"):
            if (directory / name).exists():
                st.write(name)
                st.dataframe(pd.read_csv(directory / name), hide_index=True, width="stretch")
        for name in ("signal_forecasts.parquet", "state_history.parquet", "alert_episodes.parquet", "observations.jsonl", "evaluation.json"):
            st.download_button("Download " + name, (directory / name).read_bytes(), file_name=name, key=key+name)
        st.json({"event_source": bundle["event_model_run_id"], "sensor_source": bundle["sensor_model_run_id"],
                 "profile": bundle["profile"], "reference": bundle["reference"]["status"]}, expanded=False)


def _render_replay(rows, measurements, bundle, directory, key):
    state = st.session_state[key]
    n = len(rows)
    def move(step=None, playing=None):
        if step is not None:
            state["step"] = max(0, min(n - 1, step))
            st.session_state[key + ":seek"] = state["step"]
        if playing is not None:
            state["playing"] = playing
    buttons = st.columns(4)
    actions = [buttons[i].button(label, key=key+label.lower()) for i, label in enumerate(("Start", "Pause", "Next", "Reset"))]
    if any(actions):
        if actions[0]:
            move(playing=True)
        elif actions[1]:
            move(playing=False)
        elif actions[2]:
            move(step=state["step"] + 1, playing=False)
        else:
            move(step=0, playing=False)
        # Recreate the fragment timer on Start/Pause; idle views do not rerun.
        st.rerun(scope="app")
    speed = st.select_slider("Playback speed", [1, 5, 20], key=key+"speed")
    if state["playing"]:
        move(min(n - 1, state["step"] + speed), state["step"] + speed < n - 1)
    st.session_state.setdefault(key+":seek", state["step"])
    step = st.slider("Measurement", 0, max(n - 1, 1), key=key+":seek")
    if step != state["step"]:
        state.update(step=step, playing=False)
        st.rerun(scope="app")
    st.caption("Playback: " + ("playing" if state["playing"] else "paused"))
    result = rows[min(step, n - 1)]
    condition, profile = result["condition"], bundle["profile"]
    zone, now = condition["display_zone"], result["as_of"]
    st.subheader(LABELS[zone])
    st.caption(bundle["state_policy"]["laboratory_notice"])
    st.write(condition["action_text"])
    for reason in condition["reason_codes"][:3]:
        st.write("• " + reason.replace("_", " "))
    c1, c2, c3 = st.columns(3)
    c1.metric("Observation", result["data_quality_status"])
    c2.metric("Regime", result["model_applicability"].replace("_", " "))
    c3.metric("Forecast", result["forecast_status"])
    history = result["history"]
    duration = "not available" if history["duration"] is None else f"{history['duration']:g} " + ("seconds" if profile["time_basis"] == "physical_seconds" else "dataset time units")
    st.caption(f"History: {history['used_measurements']} measurements · duration {duration} · {history['mode']}. Reference: {condition['reference_status']}.")
    if condition["critical_latch"]:
        st.warning("An open critical alert remains active until the saved recovery rule is satisfied.")
    show_gt = st.checkbox("Show ground truth (evaluation overlay only)", value=False, key=key+"gt")
    show_old = st.checkbox("Show previously issued forecasts", value=False, key=key+"old")
    signal, unit = profile["signal_name"], profile["signal_unit"]
    scale = 60. if profile["time_basis"] == "physical_seconds" else 1.
    time_label = "minutes" if scale == 60. else "dataset internal time"
    fig = go.Figure()
    observed = measurements.loc[measurements.timestamp_s <= now]
    fig.add_trace(go.Scatter(x=observed.timestamp_s / scale, y=observed[signal], mode="lines", name="Observed " + signal))
    if show_gt:
        future = measurements.loc[measurements.timestamp_s > now]
        fig.add_trace(go.Scatter(x=future.timestamp_s / scale, y=future[signal], mode="lines", name="Future fact · evaluator only", line=dict(dash="dot", color="#aab0bb")))
    if show_old:
        for prior in rows[max(0, step - 20):step:5]:
            forecasts = pd.DataFrame(prior["signal_forecasts"])
            if len(forecasts):
                fig.add_trace(go.Scatter(x=forecasts.target_time / scale, y=forecasts.point, mode="lines", opacity=.2, showlegend=False))
    forecasts = pd.DataFrame(result["signal_forecasts"])
    if len(forecasts) and forecasts.point.notna().any():
        fig.add_trace(go.Scatter(x=forecasts.target_time / scale, y=forecasts.point, mode="lines+markers", name="Sensor forecast", line_color="#61d8ee"))
        if forecasts.lower.notna().any():
            fig.add_trace(go.Scatter(x=forecasts.target_time / scale, y=forecasts.upper, mode="lines", line_width=0, showlegend=False))
            fig.add_trace(go.Scatter(x=forecasts.target_time / scale, y=forecasts.lower, mode="lines", line_width=0, fill="tonexty", name="Pointwise 5–95% · unvalidated"))
    else:
        st.info("Sensor forecast unavailable: " + (str(forecasts.reason.iloc[0]) if len(forecasts) else "no trained sensor model"))
    ref = result["normality"].get("residuals", {}).get(signal)
    if ref:
        fig.add_hline(y=ref["expected"], line_dash="dot", annotation_text="Provisional regime reference")
        enter = bundle["state_policy"]["warning_enter"]
        if enter is not None:
            fig.add_hline(y=ref["expected"] + enter * ref["scale"], line_dash="dash", line_color=COLORS["yellow"], annotation_text="Statistical deviation boundary")
    limit = bundle["state_policy"].get("critical_limit")
    if limit:
        fig.add_hline(y=limit["value"], line_color=COLORS["red"], annotation_text=limit["verification_status"] + " limit")
    fig.add_vline(x=now / scale, line_dash="dot", annotation_text="Now")
    crossing = result["crossing"]
    if crossing["time"] is not None:
        fig.add_vline(x=crossing["time"] / scale, line_color=COLORS["yellow"], annotation_text="Median crossing")
    fig.update_layout(height=460, margin=dict(l=20,r=20,t=40,b=35), xaxis_title=time_label, yaxis_title=f"{signal} ({unit})", legend=dict(orientation="h"))
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Issued at {now / scale:g} {time_label}. Hold current observed conditions. Lines between discrete horizons are visual interpolation. " + crossing["reason"])
    history_rows = rows[:step+1]
    ribbon = go.Figure(go.Scatter(x=[r["as_of"] / scale for r in history_rows], y=[1]*len(history_rows),
                mode="markers", marker=dict(symbol="square", size=10, color=[COLORS[r["condition"]["display_zone"]] for r in history_rows]),
                text=[LABELS[r["condition"]["display_zone"]] for r in history_rows], hovertemplate="%{text}<extra></extra>"))
    ribbon.update_layout(height=100, margin=dict(l=20,r=20,t=10,b=20), yaxis_visible=False, xaxis_title=time_label)
    st.plotly_chart(ribbon, width="stretch")
    with st.expander("Diagnostics · event forecast and action timing"):
        event = result["event_forecast"]
        st.caption("Point RUL and research probabilities do not validate operational timing. Endpoints may differ from the sensor limit.")
        st.json(event)
        st.json(condition["timing"])
        event_fig = go.Figure(go.Scatter(x=[r["as_of"] / scale for r in history_rows], y=[r["event_forecast"]["point"] for r in history_rows], name="Diagnostic point RUL"))
        event_fig.update_layout(height=240, yaxis_title=profile["time_basis"], xaxis_title=time_label)
        st.plotly_chart(event_fig, width="stretch")
    with st.expander("Alert episodes and inspection feedback"):
        alerts = pd.read_parquet(directory / "alert_episodes.parquet")
        if len(alerts):
            alerts = alerts.loc[alerts.unit_id.eq(result["unit_id"]) & alerts.confirmed_at.le(now)].copy()
            alerts.loc[alerts.escalated_at > now, "level"] = "yellow"
            # Do not expose later resolution/escalation while replaying the past.
            for col in ("resolved_at", "escalated_at"):
                alerts.loc[alerts[col] > now, col] = None
            feedback_path = directory / "feedback.jsonl"
            feedback = [json.loads(line) for line in feedback_path.read_text().splitlines()] if feedback_path.exists() else []
            visible_feedback = [f for f in feedback if f["unit_id"] == result["unit_id"] and f["at"] <= now]
            latest = {f["episode_id"]: f["status"] for f in sorted(visible_feedback, key=lambda f: f["at"])}
            acknowledged = {f["episode_id"] for f in visible_feedback if f["status"] == "acknowledged"}
            alerts["review_status"] = alerts.episode_id.map(latest).fillna("unreviewed")
            alerts["acknowledged"] = alerts.episode_id.isin(acknowledged)
            st.dataframe(alerts.drop(columns=["updates"], errors="ignore"), width="stretch", hide_index=True)
        eid = condition["alert_episode_id"]
        if eid:
            with st.form(key+"feedback"):
                status = st.selectbox("Review status", sorted(FEEDBACK))
                note = st.text_input("Inspection result / action / confirmed cause")
                if st.form_submit_button("Save feedback"):
                    append_feedback(directory / "feedback.jsonl", {"episode_id": eid, "unit_id": result["unit_id"],
                        "at": now, "status": status, "source": "local_operator", "verification_status": "unverified", "note": note})
                    st.success("Feedback appended. Issued forecasts and state policy are unchanged.")


def monitoring_comparison(dataset_id):
    task = st.radio("Comparison task", ["Event forecast", "Sensor forecast", "End-to-end monitoring"], horizontal=True)
    if task == "Event forecast":
        return False
    for bundle in bundles_for(dataset_id):
        for path in sorted((bundle_root(bundle["bundle_id"]) / "evaluations").glob("*/evaluation.json")):
            evaluation = read_json(path)
            st.subheader(f"{evaluation['split']} · {bundle['bundle_id']}")
            st.caption(bundle["profile"]["event_definition_id"] + " · " + bundle["profile"]["time_basis"] + " · " + evaluation["evaluation_status"])
            target = path.parent / ("signal_results.csv" if task == "Sensor forecast" else "per_unit_results.csv")
            if target.exists():
                st.dataframe(pd.read_csv(target), width="stretch", hide_index=True)
                st.download_button("Download " + task, target.read_bytes(), target.name, key=str(target))
    return True
