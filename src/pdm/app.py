from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pdm.alerts import (
    MISSING_FROZEN_POLICY_MESSAGE,
    alert_policy_hash,
    build_alert_policy,
    load_alert_policy,
    save_alert_policy,
)
from pdm.architectures import is_reservoir
from pdm.cli import spawn_worker
from pdm.config import load_dataset_config, model_defaults
from pdm.connectome.provenance import SYNTHETIC_DISCLAIMER
from pdm.data.filters import filter_time_scale_meta
from pdm.data.prepare import load_processed, processed_ready
from pdm.device import resolve_device
from pdm.evaluate import (
    IncompatibleDataError,
    default_horizon_s,
    load_evaluation_alert_policy,
    load_run_pressure_limit_pa,
)
from pdm.experiments import (
    baseline_coverage_callout,
    empty_evaluation_artifacts,
    evaluation_metric_cards,
    evaluation_notes,
    evaluations_for_mode,
    legacy_evaluation_artifacts,
    list_evaluations,
    list_runs,
    mask_unit_ids,
    metrics_by_unit_display_frame,
    near_event_zone_frame,
    read_evaluation_metrics,
    read_metrics_by_unit,
    resolve_evaluation_artifacts,
    run_dir,
)
from pdm.paths import dataset_raw
from pdm.replay import (
    END_OF_OBSERVED_DATA,
    active_warning_episode_time_s,
    alert_status_at_time,
    bind_replay_to_run,
    evaluator_overlay_rul,
    filter_replay_alert_log,
    replay_end_state,
    replay_session_key,
    rescore_replay_alerts,
    slice_predictions_to_replay_time,
    split_label_for_unit,
    visible_replay_slice,
)
from pdm.train import (
    load_saved_train_settings,
    next_max_windows_on_mode_change,
    selection_metric_spec,
    training_mode_label,
)
from pdm.visualization.comparison import (
    BIOLOGICAL_SECTION,
    SMOKE_NOTE,
    SYNTHETIC_SECTION,
    build_comparison_table,
    comparison_sections,
)
from pdm.visualization.component import neural_activity_explorer
from pdm.visualization.explorer import (
    EXPLORER_DISCLAIMER,
    LIVE_CONTEXT_CAP,
    RESERVOIR_REQUIRED_MESSAGE,
    WORKER_BUSY_MESSAGE,
    build_ui_explorer_payload,
    explorer_anatomy_captions,
    explorer_overlay_clock,
    fallback_positions,
    is_active_train_live,
    load_scene_from_run,
    soma_join_allowed,
    subset_scene,
    synthetic_banner_required,
)
from pdm.visualization.live import load_live_activity
from pdm.visualization.simulation_ui import render_equipment_simulation
from pdm.worker import read_status, request_stop, worker_alive

st.set_page_config(page_title="Predictive Maintenance Lab", layout="wide")
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

LABELS = {"bearings": "Bearings", "filters": "Filters"}
REPLAY_MODE_VALIDATION = "Validation"
REPLAY_MODE_TEST = "Test"
REPLAY_MODE_RESEARCH = "Research"
REPLAY_MODES = (REPLAY_MODE_VALIDATION, REPLAY_MODE_TEST, REPLAY_MODE_RESEARCH)
REPLAY_PLAY_INTERVAL_S = 0.4
_REPLAY_VIEW_KEY = "_replay_view"
_REPLAY_SESSION_KEYS = (
    "replay_cache",
    "replay_alert_episodes",
    "replay_alert_steps",
    "replay_alert_cache_key",
    "replay_unit",
    _REPLAY_VIEW_KEY,
)


def _pause_neural_runs() -> None:
    """Changing views or equipment never silently resumes an old test clock."""
    for key, value in st.session_state.items():
        if str(key).startswith("equipment_sim:") and isinstance(value, dict) and "playing" in value:
            value["playing"] = False


def _dataset() -> str:
    choice = st.sidebar.radio("Dataset", ["Bearings", "Filters"], horizontal=True, on_change=_pause_neural_runs)
    return "bearings" if choice == "Bearings" else "filters"


def _device_box() -> None:
    if st.session_state.get("screen_selection") in {"Neural Activity Explorer", "Model Report"}:
        st.sidebar.caption("Replay device: **CPU**")
        return
    info = resolve_device("auto")
    st.sidebar.caption(f"Compute device: **{info.name}**")
    if info.fallback_reason:
        st.sidebar.warning(info.fallback_reason)


def _data_state(dataset_id: str) -> str:
    raw = dataset_raw(dataset_id)
    has_raw = raw.exists() and any(raw.rglob("*"))
    if processed_ready(dataset_id):
        return "ready"
    if has_raw:
        return "not_ready"
    return "not_ready"


def _status_chip(*, compact: bool = False) -> dict:
    st_ = read_status()
    alive = worker_alive()
    st.sidebar.write("Worker:", "running" if alive else "idle")
    details = {k: st_.get(k) for k in ("status", "dataset_id", "run_id", "epoch", "message", "error") if k in st_ or st_.get(k)}
    if compact:
        with st.sidebar.expander("Job details", expanded=bool(st_.get("error"))):
            st.json(details)
    else:
        st.sidebar.json(details)
    if st_.get("status") in {"cancelled", "stopped"}:
        st.sidebar.caption(f"Job interrupted ({st_.get('status')})")
    return st_


def main() -> None:
    from pdm.lab_ui import matrix_controls, screen_comparison
    from pdm.visualization.presentation import apply_explorer_style

    aliases = {"Data": "Data Quality", "Train": "Training", "Neural Activity Explorer": "Model Report", "Test & Replay": "Model Report"}
    old = st.session_state.get("screen_selection")
    if old in aliases:
        st.session_state["screen_selection"] = aliases[old]
        if old == "Test & Replay":
            st.session_state["report_view"] = "Evaluation settings"
    if "screen_selection" not in st.session_state and st.query_params.get("view") == "brain":
        st.session_state["screen_selection"] = "Model Report"
    dataset_id = _dataset()
    _device_box()
    page = st.sidebar.radio("Screen", ["Data Quality", "Training", "Model Report", "Compare Models"],
                            key="screen_selection", on_change=_pause_neural_runs)
    apply_explorer_style()
    _watch_worker_lifecycle()
    _status_chip(compact=True)
    if page == "Data Quality":
        screen_data(dataset_id)
    elif page == "Training":
        matrix_controls()
        screen_train(dataset_id)
    elif page == "Model Report":
        view = st.radio("Report view", ["Model replay", "Evaluation settings"], horizontal=True, key="report_view", on_change=_pause_neural_runs)
        if view == "Evaluation settings":
            screen_replay(dataset_id)
        else:
            screen_explorer(dataset_id)
    elif page == "Compare Models":
        screen_comparison(dataset_id)


def _filters_time_warning(report: dict | None = None) -> str:
    note = (report or {}).get("time_unit_note")
    if note:
        return str(note)
    return str(filter_time_scale_meta(load_dataset_config("filters"))["time_unit_note"])


def _render_filters_time_warning():
    st.caption("Filter time scale is unverified. Time × 60 is an internal comparison unit, not confirmed wall-clock seconds.")
    with st.expander("Time-scale assumption and source details"):
        st.warning(_filters_time_warning())


def _filters_scale_unverified(report: dict | None = None) -> bool:
    if report and "time_scale_verified" in report:
        return not bool(report["time_scale_verified"])
    return not bool(load_dataset_config("filters").get("time_scale_verified", False))


def _finite_number(val) -> bool:
    try:
        if val is None or pd.isna(val):
            return False
    except (TypeError, ValueError):
        return False
    try:
        return math.isfinite(float(val))
    except (TypeError, ValueError):
        return False


def _record_kind(row: pd.Series, dataset_id: str) -> str:
    ev = 0
    if "event_observed" in row.index and _finite_number(row.get("event_observed")):
        ev = int(row.get("event_observed") or 0)
    has_official = _finite_number(row.get("official_rul_at_prefix_end_s"))
    if dataset_id == "filters":
        if ev:
            return "Observed 600 Pa event"
        if has_official:
            return "Test prefix (official RUL, evaluation only)"
        return "Right-censored (sensor end, no 600 Pa)"
    if ev:
        return "Last recorded sample (endpoint approx.)"
    return "Right-censored (sensor end)"


def _units_display_frame(dataset_id: str, units: pd.DataFrame, split: dict) -> pd.DataFrame:
    show = units.copy()
    show["split"] = show["unit_id"].map(_split_map(split))
    if "origin_unit_id" not in show.columns:
        if "author_data_no" in show.columns:
            show["origin_unit_id"] = show["author_data_no"]
        else:
            show["origin_unit_id"] = show["unit_id"]
    if "author_split" in show.columns:
        show["origin"] = (
            show["author_split"].astype(str) + ":" + show["origin_unit_id"].astype(str)
        )
    else:
        show["origin"] = show["origin_unit_id"].astype(str)
    show["record_kind"] = [_record_kind(row, dataset_id) for _, row in show.iterrows()]
    preferred = [
        "unit_id",
        "split",
        "origin",
        "origin_unit_id",
        "author_split",
        "record_kind",
        "regime_id",
        "event_observed",
        "event_time_s",
        "observation_end_s",
        "official_rul_at_prefix_end_s",
        "official_rul_at_prefix_end_original",
        "n_measurements",
        "endpoint_definition",
        "event_source",
    ]
    ordered = [c for c in preferred if c in show.columns]
    rest = [c for c in show.columns if c not in ordered]
    return show[ordered + rest]


def _window_counts_frame(report: dict) -> pd.DataFrame | None:
    by_split = ((report.get("window_counts") or {}).get("by_split")) or {}
    if not by_split:
        return None
    rows = []
    for part in ("train", "validation", "test"):
        rec = by_split.get(part) or {}
        rows.append(
            {
                "split": part,
                "units": rec.get("n_units"),
                "eligible windows": rec.get("eligible_windows"),
                "excluded (gap)": rec.get("excluded_gap"),
                "excluded (post-event)": rec.get("excluded_post_event"),
                "excluded (insufficient length)": rec.get("excluded_insufficient_length"),
            }
        )
    return pd.DataFrame(rows)


def _regimes_frame(report: dict) -> pd.DataFrame | None:
    rows = report.get("regimes") or []
    if not rows:
        return None
    return pd.DataFrame(rows)


def screen_data(dataset_id: str) -> None:
    st.header(f"Data Quality — {LABELS[dataset_id]}")
    if dataset_id == "filters" and _filters_scale_unverified():
        _render_filters_time_warning()
    state = _data_state(dataset_id)
    st.write("Data state:", state)
    raw = dataset_raw(dataset_id)
    with st.expander("Prepare source data", expanded=state != "ready"):
        st.caption(f"Raw directory: `{raw}`")
        local = st.text_input("Local archive or folder path (optional, for multi-GB data)")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Download / locate data", disabled=worker_alive()):
                spawn_worker({"kind": "download", "dataset_id": dataset_id, "local_path": local or None})
                st.rerun()
        with c2:
            if st.button("Inspect & prepare", disabled=worker_alive()):
                spawn_worker({"kind": "prepare", "dataset_id": dataset_id})
                st.rerun()
    if worker_alive():
        st.info("A background job is running. This page refreshes while it works.")
        _auto_refresh()
    ws = read_status()
    if ws.get("status") == "failed":
        st.error(ws.get("error") or "Job failed")
        if ws.get("traceback"):
            st.code(ws["traceback"])
    if not processed_ready(dataset_id):
        st.warning("No prepared dataset yet. Download/locate data, then Inspect & prepare.")
        return
    bundle = load_processed(dataset_id)
    units = bundle["units"]
    features = bundle["features"]
    split = bundle["split"]
    report = bundle["report"] or {}
    from pdm.lab_ui import quality_overview

    quality_overview(bundle)
    fingerprint = bundle.get("fingerprint") or {}
    st.subheader("Prepared snapshot")
    ver = report.get("dataset_version") or bundle.get("dataset_version") or fingerprint.get("dataset_version") or "—"
    proto = report.get("split_protocol") or fingerprint.get("split_protocol") or split.get("protocol") or "—"
    shash = report.get("split_hash") or fingerprint.get("split_hash") or "—"
    st.markdown(
        f"**dataset_version:** `{ver}`  \n"
        f"**split_protocol:** `{proto}`  \n"
        f"**split_hash:** `{shash}`"
    )
    if dataset_id == "filters":
        st.caption(
            "Filters reuse origin_unit_id (HSE Data_No) 1–50 in Train and Test files. "
            "The units table shows author_split:origin so those IDs are not collapsed."
        )
    else:
        st.caption("origin_unit_id matches the bearing folder / unit_id.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train units", split.get("n_train"))
    c2.metric("Validation units", split.get("n_validation"))
    c3.metric("Test units", split.get("n_test"))
    c4.metric("Measurements", int(len(features)))
    ev = report.get("events_vs_censoring") or {}
    n_obs = ev.get("n_event_observed")
    if n_obs is None and "event_observed" in units.columns:
        n_obs = int(units["event_observed"].sum())
    n_cens = ev.get("n_right_censored")
    n_official = ev.get("n_official_rul_known_not_observed")
    n_sensor = ev.get("n_sensor_end_without_event_or_official_rul")
    if dataset_id == "filters":
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Observed 600 Pa events", n_obs if n_obs is not None else "—")
        e2.metric("Right-censored", n_cens if n_cens is not None else "—")
        e3.metric("Official RUL known (eval only)", n_official if n_official is not None else "—")
        e4.metric("Sensor end only (no event / RUL)", n_sensor if n_sensor is not None else "—")
    else:
        e1, e2 = st.columns(2)
        e1.metric("Observed events", n_obs if n_obs is not None else "—")
        e2.metric("Right-censored", n_cens if n_cens is not None else "—")
    if ev.get("note"):
        st.caption(ev["note"])
    obs = report.get("observation_time_s") or {}
    span = obs.get("total_span_s")
    if span is not None:
        if dataset_id == "bearings":
            st.metric("Total observation span", f"{float(span) / 60.0:.1f} min")
        else:
            st.metric("Total observation span", f"{float(span):.1f} s")
            st.caption("Observation span is internal seconds (CSV Time × 60; not wall-clock minutes).")
    regimes = _regimes_frame(report)
    if regimes is not None:
        st.subheader("Regimes")
        st.dataframe(regimes, width="stretch", hide_index=True)
    st.subheader("Windows by split")
    wc = report.get("window_counts") or {}
    hist_len = wc.get("history_length") or report.get("history_length")
    counts_tbl = _window_counts_frame(report)
    if counts_tbl is None:
        st.info("This snapshot has no cached window counts. Re-run Inspect & prepare.")
    else:
        st.dataframe(counts_tbl, width="stretch", hide_index=True)
        excluded = wc.get("excluded") or {}
        st.caption(
            f"Cached counts for history_length={hist_len} from prepare (dataset config). "
            "Eligible windows match training windowing at that length. "
            "Post-event is t ≥ event_time (observed) or t ≥ observation_end (censored). "
            "This screen reads data_report.json and does not rebuild windows."
        )
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Eligible windows", wc.get("eligible"))
        m2.metric("Excluded: gap", excluded.get("gap"))
        m3.metric("Excluded: post-event", excluded.get("post_event"))
        m4.metric("Excluded: short history", excluded.get("insufficient_length"))
    st.subheader("Units")
    show = _units_display_frame(dataset_id, units, split)
    st.dataframe(show, width="stretch", hide_index=True)
    uid = st.selectbox("Unit for sensor plot", show["unit_id"].tolist())
    g = features[features["unit_id"] == uid].sort_values("timestamp_s")
    fig = go.Figure()
    if dataset_id == "bearings" and "horizontal_rms" in g.columns:
        fig.add_trace(go.Scatter(x=g["timestamp_s"] / 60.0, y=g["horizontal_rms"], name="horizontal RMS"))
        fig.update_xaxes(title_text="Operating time (min)")
        fig.update_yaxes(title_text="RMS (g, original units)")
    else:
        fig.add_trace(go.Scatter(x=g["timestamp_s"], y=g["differential_pressure"], name="Δp"))
        fig.add_hline(y=600.0, line_dash="dash", annotation_text="600 Pa")
        fig.update_xaxes(title_text="Internal time (s; Time × 60, unit unconfirmed)")
        fig.update_yaxes(title_text="Differential pressure (Pa)")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Target and quality")
    st.write(report.get("sensor_time_note", ""))
    phys = report.get("history_length_physical") or {}
    st.caption(phys.get("note") or "")
    if dataset_id == "filters" and report.get("time_unit_note"):
        st.caption(
            "Exports keep original CSV Time/RUL as time_original and "
            "official_rul_at_prefix_end_original. Metrics use an _s suffix on the "
            "internal Time × 60 scale — not wall-clock minutes. "
            "Official test RUL is evaluation-only; sensor observations stop at "
            "observation_end_s and that is not a 600 Pa event."
        )
    if report.get("issues_and_decisions"):
        for issue in report["issues_and_decisions"]:
            st.warning(issue)
    with st.expander("data_report.json (debug)"):
        st.json(report)


def _mode_badge(smoke: bool, *, kind: str = "Training") -> None:
    label = training_mode_label(smoke)
    st.badge(label, color="orange" if smoke else "green")
    st.markdown(f"{kind} mode: **{label}**")


def _fmt_windows_cap(cap) -> str:
    if cap in (None, "", "None"):
        return "all"
    try:
        n = int(cap)
    except (TypeError, ValueError):
        return "all"
    return "all" if n <= 0 else str(n)


def _windows_used_caption(rec: dict) -> str:
    wpu = rec.get("train_windows_per_unit") or {}
    cap = _fmt_windows_cap(rec.get("max_windows_per_unit"))
    n_tr = rec.get("n_train_windows")
    n_va = rec.get("n_val_windows")
    bits = [f"Train windows: {n_tr if n_tr is not None else '—'}", f"val windows: {n_va if n_va is not None else '—'}"]
    if wpu:
        mean = wpu.get("mean")
        mean_s = f"{float(mean):.1f}" if _finite_number(mean) else "—"
        bits.append(
            f"windows/unit used min {wpu.get('min', '—')} / max {wpu.get('max', '—')} / mean {mean_s} (cap {cap})"
        )
    else:
        bits.append(f"windows/unit cap {cap}")
    return " · ".join(bits)


def screen_train(dataset_id: str) -> None:
    st.header(f"Training — {LABELS[dataset_id]}")
    if not processed_ready(dataset_id):
        st.warning("Prepare data on the Data screen first. There is no dummy training path.")
        return
    from pdm.lab_ui import training_overview

    if not training_overview(load_processed(dataset_id), load_dataset_config(dataset_id)):
        return
    arch = st.selectbox(
        "Architecture", ["gru", "lstm", "fly_connectome_reservoir", "random_reservoir"], index=0,
    )
    scope = "1,000-node subgraph"
    if dataset_id == "bearings" and arch == "fly_connectome_reservoir":
        scope = st.radio("Connectome scope", ["Full MaleCNS", "1,000-node subgraph"], horizontal=True)
    if dataset_id == "bearings" and arch == "fly_connectome_reservoir" and scope == "Full MaleCNS":
        st.subheader("Train the complete MaleCNS")
        st.write("All classified neurons and their directed connections participate in every measurement. "
                 "A new forecast readout is fitted from scratch using whole-bearing cross-validation, "
                 "with separate interval calibration.")
        st.caption("Full chronological histories · 20-measurement warmup · CPU sparse computation. "
                   "The population is determined by the source annotations; it is not a neuron-count setting.")
        start, stop = st.columns(2)
        if start.button("Train full MaleCNS from scratch", disabled=worker_alive(), type="primary"):
            from pdm.training_protocol import protocol

            spawn_worker({"kind": "train_full_cns", "dataset_id": "bearings", "training_protocol": protocol("bearings")})
            st.rerun()
        if stop.button("Stop", disabled=not worker_alive()):
            request_stop()
            st.rerun()
        ws = read_status()
        if worker_alive():
            st.info(ws.get("message") or "Training the full connectome. This can take several minutes.")
            _auto_refresh()
        elif ws.get("status") == "failed":
            st.error(ws.get("error") or ws.get("message") or "Job failed")
        elif ws.get("status") == "completed":
            st.success("Training completed. Open Model Report to run the saved model.")
        elif ws.get("status") == "cancelled":
            st.info("Training cancelled.")
        return
    graph_mode = None
    if is_reservoir(arch):
        graph_mode = st.selectbox("Graph source", ["real_connectome", "synthetic_fixture"])
    cfg = load_dataset_config(dataset_id)
    mcfg = model_defaults(cfg)
    bundle = load_processed(dataset_id)
    phys = (bundle.get("report") or {}).get("history_length_physical") or {}
    sel_spec = selection_metric_spec(dataset_id)
    smoke_cap = int(mcfg.get("smoke_max_windows_per_unit", 32))
    smoke_epochs = int(mcfg.get("smoke_max_epochs", 5))
    mode_key = f"train_mode_{dataset_id}"
    cap_key = f"max_windows_widget_{dataset_id}"
    prev_key = f"_prev_train_smoke_{dataset_id}"
    if mode_key not in st.session_state:
        st.session_state[mode_key] = "Full"
    if cap_key not in st.session_state:
        st.session_state[cap_key] = 0
    mode = st.radio("Training mode", ["Smoke", "Full"], horizontal=True, key=mode_key)
    smoke = mode == "Smoke"
    optimization = st.selectbox("Optimization protocol", ["Adaptive v2", "Diagnostic 100 epochs", "Legacy"], disabled=smoke)
    use_v2 = not smoke and optimization != "Legacy"
    feature_recipe = "base_v1"
    sampling = "unit_replacement"
    near_weight = 0.0
    if use_v2:
        st.caption("Training v2 uses CPU for reproducible continuation and float64 survival calculations.")
        feature_recipe = st.selectbox("Feature recipe", ["base_v1", "degradation_v1"])
        sampling = st.selectbox("Window sampling", ["unit_replacement", "full_pass"])
        if dataset_id == "bearings" and sampling == "full_pass":
            near_weight = .5 if st.checkbox("Give half the training weight to the final 30 minutes") else 0.0
        sel_spec = {"label": "near_30m_mae_s" if dataset_id == "bearings" else "survival_nll (internal seconds)"}
    _mode_badge(smoke)
    if smoke:
        st.warning("Smoke test — not a quality benchmark")
        st.caption(
            f"Smoke caps epochs at {smoke_epochs} and windows/unit at {smoke_cap} "
            "unless you set max windows/unit to 0 (all)."
        )
    else:
        st.caption("All eligible windows enter the sampling pool when max windows/unit is 0; the selected sampler determines which windows are visited.")
        st.caption("Diagnostic mode runs exactly 100 epochs without early stopping; the best checkpoint still uses validation."
                   if use_v2 and optimization == "Diagnostic 100 epochs" else "Early stopping uses the validation metric.")
    st.caption(
        f"Epoch selection metric: **{sel_spec['label']}** "
        "(unit-equal on a fixed window mask). Do not compare NLL and MAE as one accuracy."
    )
    next_cap = next_max_windows_on_mode_change(
        prev_smoke=st.session_state.get(prev_key),
        smoke=smoke,
        current_cap=int(st.session_state.get(cap_key) or 0),
        smoke_cap=smoke_cap,
    )
    if int(st.session_state.get(cap_key) or 0) != next_cap:
        st.session_state[cap_key] = next_cap
    st.session_state[prev_key] = smoke
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=100 if use_v2 else int(mcfg["max_epochs"]), disabled=use_v2)
    hist = st.number_input(
        "History length (measurements)", min_value=2, max_value=128, value=int(mcfg["history_length"]), disabled=use_v2
    )
    st.caption(phys.get("note") or f"{hist} measurements of history")
    max_w = st.number_input("Max windows / unit (0 = all)", min_value=0, key=cap_key, disabled=use_v2)
    history_ready = True
    if int(hist) != int(mcfg["history_length"]):
        st.caption("Admission counts for the selected history length")
        history_ready = training_overview(bundle, cfg, int(hist))
    with st.expander("Advanced"):
        learning_rate = st.number_input("Learning rate", min_value=0.00001, value=float(mcfg["learning_rate"]), format="%.5f", disabled=not use_v2 or optimization == "Diagnostic 100 epochs")
        st.number_input("Hidden size", value=int(mcfg["hidden_size"]), disabled=True)
        st.number_input("Batch size", value=int(mcfg["batch_size"]), disabled=True)
    runs = [r for r in list_runs(dataset_id) if r.get("has_last")]
    resume_opt = ["(new experiment)"] + [r["run_id"] for r in runs]
    resume = st.selectbox("Resume last checkpoint", resume_opt)
    resume_id = None if resume == "(new experiment)" else resume
    resume_saved = load_saved_train_settings(run_dir(dataset_id, resume_id), resume_id) if resume_id else None
    if resume_saved is not None:
        st.info(
            f"Resume uses saved **{training_mode_label(resume_saved.get('smoke'))}** "
            f"(epochs={resume_saved.get('max_epochs', '—')}, "
            f"windows/unit={_fmt_windows_cap(resume_saved.get('max_windows_per_unit'))}). "
            "Form Smoke/Full, epochs, and windows cap are ignored for this job."
        )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start training", disabled=worker_alive() or not history_ready):
            job = {
                "kind": "train",
                "dataset_id": dataset_id,
                "architecture": arch,
                "graph_mode": graph_mode,
                "n_nodes": 1000 if is_reservoir(arch) else None,
                "readout": ("ridge" if dataset_id == "bearings" else "gradient") if is_reservoir(arch) else None,
                "max_epochs": int(epochs),
                "history_length": int(hist),
                "smoke": bool(smoke),
                "resume_run_id": resume_id,
                "max_windows_per_unit": int(max_w),
            }
            if use_v2 and not resume_id:
                from pdm.training_protocol import protocol

                job["training_protocol"] = protocol(dataset_id, mode="diagnostic" if optimization == "Diagnostic 100 epochs" else "adaptive",
                            learning_rate=float(learning_rate), sampling=sampling, near_weight=near_weight, feature_recipe=feature_recipe)
            if resume_saved is not None:
                job["smoke"] = bool(resume_saved.get("smoke", False))
                mw_saved = resume_saved.get("max_windows_per_unit")
                job["max_windows_per_unit"] = 0 if mw_saved is None else int(mw_saved)
                if resume_saved.get("max_epochs") is not None:
                    job["max_epochs"] = int(resume_saved["max_epochs"])
            spawn_worker(job)
            st.rerun()
    with c2:
        if st.button("Stop", disabled=not worker_alive()):
            request_stop()
            st.rerun()
    ws = read_status()
    if ws.get("status") in {"training", "preparing"} or worker_alive():
        live_mode = ws.get("mode") or training_mode_label(ws.get("smoke"), ws.get("run_id") or "")
        st.markdown(f"Live run: **{live_mode}**")
        if ws.get("epoch") is not None:
            st.write(f"Epoch {ws['epoch']} / {ws.get('max_epochs', '—')}")
            st.write(
                f"train loss {ws.get('train_loss', '—')}  val loss {ws.get('val_loss', '—')}  "
                f"train metric {ws.get('train_metric', '—')}  "
                f"{ws.get('selection_metric_label') or 'val metric'} {ws.get('val_metric', '—')}  "
                f"best epoch {ws.get('best_epoch', '—')}"
            )
            st.caption(_windows_used_caption(ws))
        else:
            st.caption(f"{ws.get('status', 'running').title()} · {ws.get('stage') or ws.get('kind', '')}")
        st.caption(ws.get("message") or "")
        _render_train_live_activity(dataset_id, ws)
        _auto_refresh()
    if ws.get("status") == "failed":
        st.error(ws.get("error") or ws.get("message") or "Job failed")
    if ws.get("status") in {"cancelled", "stopped"}:
        st.info(f"Job interrupted ({ws.get('status')})")
    st.subheader("Experiments")
    table = list_runs(dataset_id)
    if not table:
        st.write("No runs yet.")
        return
    df = pd.DataFrame(table)
    df["mode"] = [training_mode_label(row.get("smoke"), str(row.get("run_id") or "")) for row in table]
    cols = [
        c
        for c in [
            "run_id",
            "mode",
            "architecture",
            "status",
            "best_metric",
            "best_epoch",
            "n_train_windows",
            "n_val_windows",
            "updated_at",
            "has_best",
        ]
        if c in df.columns
    ]
    st.dataframe(df[cols] if cols else df, width="stretch", hide_index=True)
    pick = st.selectbox("Open saved run (does not retrain)", [r["run_id"] for r in table])
    rec = next((r for r in table if r.get("run_id") == pick), {})
    rdir = run_dir(dataset_id, pick)
    run_status = dict(rec)
    status_path = rdir / "status.json"
    if status_path.exists():
        try:
            run_status.update(json.loads(status_path.read_text(encoding="utf-8")))
        except Exception:
            pass
    vm: dict = {}
    vm_path = rdir / "validation_metrics.json"
    if vm_path.exists():
        try:
            vm = json.loads(vm_path.read_text(encoding="utf-8"))
            run_status.update({k: vm[k] for k in vm if k not in {"last", "last_train"}})
        except Exception:
            vm = {}
    run_smoke = bool(run_status.get("smoke")) if "smoke" in run_status else str(pick).startswith("smoke_")
    _mode_badge(run_smoke, kind="Run")
    best_epoch = run_status.get("best_epoch") or vm.get("best_epoch")
    best_metric = run_status.get("best_metric") if run_status.get("best_metric") is not None else vm.get("best_metric")
    sel_label = run_status.get("selection_metric_label") or vm.get("selection_metric_label") or sel_spec["label"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best epoch", best_epoch if best_epoch not in (None, 0) else "—")
    if _finite_number(best_metric):
        m2.metric(sel_label, f"{float(best_metric):.4f}")
    else:
        m2.metric(sel_label, "—")
    m3.metric(
        "Train windows",
        run_status.get("n_train_windows") if run_status.get("n_train_windows") is not None else "—",
    )
    m4.metric("Val windows", run_status.get("n_val_windows") if run_status.get("n_val_windows") is not None else "—")
    st.caption(_windows_used_caption(run_status))
    last_train = (vm.get("last_train") or {}) if vm else {}
    last_val = (vm.get("last") or {}) if vm else {}
    if last_train or last_val:
        st.caption(
            "Train/val diagnostics are unit-equal on the fixed window mask for this run "
            "(train uses windows kept after the per-unit cap; val uses all validation windows)."
        )
        d1, d2 = st.columns(2)
        d1.write(
            f"Train mask {vm.get('selection_metric_name') or sel_spec.get('name', sel_label)}: "
            f"{last_train.get('selection_metric', run_status.get('train_metric', '—'))}"
        )
        d2.write(
            f"Val mask {sel_label}: "
            f"{last_val.get('selection_metric', run_status.get('val_metric', best_metric if best_metric is not None else '—'))}"
        )
    hist_csv = rdir / "training_history.csv"
    if hist_csv.exists():
        hdf = pd.read_csv(hist_csv)
        fig = go.Figure()
        if "train_loss" in hdf.columns:
            fig.add_trace(go.Scatter(x=hdf["epoch"], y=hdf["train_loss"], name="train loss"))
        if "val_loss" in hdf.columns:
            fig.add_trace(go.Scatter(x=hdf["epoch"], y=hdf["val_loss"], name="val loss"))
        if fig.data:
            if _finite_number(best_epoch):
                fig.add_vline(x=int(best_epoch), line_dash="dash", annotation_text=f"best epoch {int(best_epoch)}")
            fig.update_xaxes(title="Epoch")
            fig.update_yaxes(title="Loss (not comparable across loss types)")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Closed-form readout fitting has no gradient loss curve.")
        if "val_metric" in hdf.columns or "train_metric" in hdf.columns:
            fig_m = go.Figure()
            if "train_metric" in hdf.columns:
                fig_m.add_trace(
                    go.Scatter(x=hdf["epoch"], y=hdf["train_metric"], name="train (unit-equal, fixed mask)")
                )
            if "val_metric" in hdf.columns:
                fig_m.add_trace(
                    go.Scatter(x=hdf["epoch"], y=hdf["val_metric"], name="val (unit-equal, fixed mask)")
                )
            if _finite_number(best_epoch):
                fig_m.add_vline(x=int(best_epoch), line_dash="dash", annotation_text=f"best epoch {int(best_epoch)}")
            fig_m.update_xaxes(title="Epoch")
            fig_m.update_yaxes(title=sel_label)
            st.plotly_chart(fig_m, width="stretch")
        st.caption(
            "MAE is in physical seconds when defined; NLL is not MAE. "
            "Do not treat different losses as one accuracy."
        )
    if vm:
        with st.expander("validation_metrics.json (debug)"):
            st.json(vm)


def _explorer_mode_label(mode: str) -> str:
    return {
        "Overview": "Overview — full window",
        "Equipment replay": "Equipment replay — run the model step by step",
        "Inside prediction window": "Inside prediction window — what the model saw",
        "Alert inspection": "Alert inspection — at a stored warning",
    }.get(mode, mode)


def _reservoir_live_activity(ws: dict | None = None) -> dict | None:
    live = load_live_activity()
    if not live:
        return None
    if str(live.get("status") or "") == "not_reservoir":
        return None
    arch = str(live.get("architecture") or (ws or {}).get("architecture") or "")
    if not is_reservoir(arch):
        return None
    if live.get("states") is None:
        return None
    return live


def _active_train_live(ws: dict | None = None) -> dict | None:
    rec = ws if ws is not None else {}
    live = _reservoir_live_activity(rec)
    if not is_active_train_live(
        worker_alive=worker_alive(),
        worker_kind=rec.get("kind"),
        live=live,
    ):
        return None
    return live


def _run_n_model(rec: dict | None) -> int | None:
    """ESN width from the run row. Never ``len(display nodes)`` after a live cap."""
    rec = rec or {}
    raw = rec.get("n_nodes")
    if raw is None:
        return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _emit_explorer_anatomy_captions(
    payload: dict,
    *,
    is_synthetic: bool,
    schema_error: str | None,
) -> None:
    for line in explorer_anatomy_captions(
        payload, is_synthetic=is_synthetic, schema_error=schema_error
    ):
        st.caption(line)


def _render_train_live_activity(
    dataset_id: str,  # UI dataset unused; live scene uses live dataset_id + run_id
    ws: dict,
    *,
    component_key: str = "train_live_activity",
) -> None:
    """Compact reservoir viewer. GRU/LSTM and leftover files skip the widget."""
    live = _active_train_live(ws)
    if live is None:
        return
    st.caption("Live training — train-split window only")
    probe = str(live.get("unit_id") or "").strip()
    if probe:
        st.caption(f"Probe unit `{probe}` (train-split window). Not a living fly recording.")
    else:
        st.caption("Training activity from a train-split window. Not a living fly recording.")
    run_id = str(live.get("run_id") or "")
    ds = str(live.get("dataset_id") or "")
    scene: dict = {"nodes": [], "edges": [], "positions": {}}
    order = [str(n) for n in (live.get("node_order") or [])]
    rdir = None
    if run_id and ds:
        try:
            rdir = run_dir(ds, run_id)
            scene = load_scene_from_run(rdir, node_order=order or None)
        except Exception:  # noqa: BLE001
            scene = {"nodes": order, "edges": [], "positions": fallback_positions(order)}
    if order:
        scene = subset_scene(scene, order)
    if not scene.get("nodes"):
        scene["nodes"] = order
    if not scene.get("positions") and scene.get("nodes"):
        scene["positions"] = fallback_positions(scene["nodes"])
    is_synthetic = synthetic_banner_required(scene, scene)
    join_soma = soma_join_allowed(scene, scene)
    payload, schema_error = build_ui_explorer_payload(
        nodes=scene.get("nodes") or order,
        edges=scene.get("edges") or [],
        positions=scene.get("positions") or {},
        states=live.get("states"),
        inputs=live.get("inputs"),
        predicted_rul_s=live.get("predicted_rul_s"),
        frame_map=live.get("frame_map") or [],
        flags={
            "mode": "Overview",
            "phase": "training",
            "compact": True,
            "is_synthetic": bool(scene.get("is_synthetic")) or is_synthetic,
            "graph_mode": str(scene.get("graph_mode") or ""),
            "architecture": str(live.get("architecture") or ""),
            "downsampled": bool(live.get("downsampled")),
            "stored_predicted_rul_s": live.get("predicted_rul_s"),
        },
        run_dir=rdir,
        n_model=None,
        join_soma=join_soma,
        context_cap=LIVE_CONTEXT_CAP,
    )
    _emit_explorer_anatomy_captions(payload, is_synthetic=is_synthetic, schema_error=schema_error)
    neural_activity_explorer(**payload, key=component_key)


def _render_explorer_intro() -> None:
    st.markdown(
        "Run recorded equipment measurements through a **fly connectome reservoir**. "
        "Its computational units process the observed history to estimate **remaining useful life** "
        "and a possible failure-time range. The brain activity and forecast share one test-run clock."
    )
    st.markdown(
        "Choose a model and equipment unit, then start the test run. "
        "Inspect sensor input, recurrent activity and the forecast beside the recorded outcome."
    )


def _explorer_now_s(
    mode: str,
    trace: dict | None,
    replay_step: int,
    alert_ts: float | None,
    view: dict | None,
) -> float | None:
    return explorer_overlay_clock(
        trace,
        mode=mode,
        replay_step=replay_step,
        alert_ts=alert_ts,
        view=view,
    )


def _explorer_status_label(trace: dict | None, alert_rows: list[dict], now_s: float | None) -> str:
    rec = trace or {}
    status = str(rec.get("status") or rec.get("valid_history_reason") or "")
    if "Collecting history" in status:
        return "Collecting history"
    if now_s is not None:
        for row in alert_rows or []:
            ts = row.get("timestamp_s")
            kind = str(row.get("type") or row.get("alert_status") or "").lower()
            if _finite_number(ts) and float(ts) <= float(now_s) and "warn" in kind:
                return "warning"
    if rec.get("predicted_rul_s") is not None:
        return "estimating"
    return "Collecting history"


def _payload_from_scene_trace(
    *,
    scene: dict,
    trace: dict | None,
    flags: dict,
    predicted_rul_s=None,
    run_dir=None,
    n_model: int | None = None,
    join_soma: bool = True,
    context_cap: int | None = None,
) -> tuple[dict, str | None]:
    nodes = list(scene.get("nodes") or [])
    states = (trace or {}).get("states")
    return build_ui_explorer_payload(
        nodes=nodes,
        edges=scene.get("edges") or [],
        positions=scene.get("positions") or {},
        states=states,
        inputs=(trace or {}).get("inputs"),
        predicted_rul_s=predicted_rul_s if predicted_rul_s is not None else (trace or {}).get("predicted_rul_s"),
        frame_map=(trace or {}).get("frame_map") or [],
        flags=flags,
        run_dir=run_dir,
        n_model=n_model,
        join_soma=join_soma,
        context_cap=context_cap,
    )


def _render_architecture_comparison(dataset_id: str) -> None:
    """One canonical comparison table. Synthetic rows never share the real section."""
    st.markdown("**Architecture comparison**")
    try:
        table = build_comparison_table(dataset_id)
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Comparison table unavailable ({exc}).")
        return
    real, synth = comparison_sections(table)
    if real.empty and synth.empty:
        st.caption("No finished runs to compare on this dataset.")
        return
    if not real.empty:
        st.markdown(f"**{BIOLOGICAL_SECTION}**")
        st.caption(
            "Same dataset split / evaluation mask. Synthetic fixture graphs are excluded. "
            "Never mix bearings and filters checkpoints."
        )
        if bool(real["smoke"].any()) if "smoke" in real.columns else False:
            st.caption(SMOKE_NOTE)
        st.dataframe(real, width="stretch", hide_index=True)
    st.markdown(f"**{SYNTHETIC_SECTION}**")
    if synth.empty:
        st.caption("No synthetic-fixture runs on this dataset.")
    else:
        st.caption("These rows are not a biological connectome and are not averaged with real runs.")
        if bool(synth["smoke"].any()) if "smoke" in synth.columns else False:
            st.caption(SMOKE_NOTE)
        st.dataframe(synth, width="stretch", hide_index=True)


def _neural_test_run_ready(dataset_id: str, rec: dict) -> bool:
    """Prefer a model prepared for complete anatomy and calibrated replay."""
    if str(rec.get("graph_mode") or "") != "real_connectome" or rec.get("is_synthetic"):
        return False
    rdir = run_dir(dataset_id, str(rec["run_id"]))
    try:
        provenance = json.loads((rdir / "connectome" / "provenance.json").read_text())
        profile = json.loads((rdir / "interval_profile.json").read_text())
    except (OSError, ValueError, TypeError):
        return False
    return (
        isinstance(provenance, dict)
        and provenance.get("sampling_method") == "all_classified_neurons"
        and isinstance(profile, dict)
        and profile.get("version") == 1
        and profile.get("ready") is True
    )


def _explorer_run_label(run_id: str) -> str:
    parts = str(run_id).rsplit("_", 3)
    if len(parts) == 4:
        try:
            stamp = datetime.strptime(parts[1] + parts[2], "%Y%m%d%H%M%S")
            return f"{stamp:%d %b · %H:%M} · {parts[3]}"
        except ValueError:
            pass
    return str(run_id)


def screen_explorer(dataset_id: str) -> None:
    from pdm.visualization.presentation import apply_explorer_style

    apply_explorer_style()
    with st.sidebar.expander("About model reports"):
        st.caption("Real model states and causal forecasts from recorded measurements. Future outcomes are evaluation overlays only.")

    table = [r for r in list_runs(dataset_id) if r.get("has_best") or r.get("has_last")]
    reservoir_rows = table
    if not reservoir_rows:
        st.warning("No saved model. Open Training to create a full experiment.")
        if table:
            st.info(RESERVOIR_REQUIRED_MESSAGE)
        return

    # Prefer the completed study's validation leader, while preserving an
    # explicit report link or the user's current selection.
    ready_rows = [r for r in reservoir_rows if _neural_test_run_ready(dataset_id, r)]
    real_rows = [r for r in reservoir_rows if str(r.get("graph_mode") or "") == "real_connectome"
                 and not r.get("is_synthetic")]
    preferred_rows = ready_rows or real_rows or reservoir_rows
    from pdm.io_util import read_json
    from pdm.lab_ui import model_label
    from pdm.paths import runs_root

    for path in sorted((runs_root() / "training_studies").glob("*/manifest.json"), reverse=True):
        study = read_json(path)
        if study.get("status") != "completed":
            continue
        scores = {task["run_id"]: task["primary_score"] for key, task in study.get("tasks", {}).items()
                  if ":main:" in key and task.get("dataset_id") == dataset_id and task.get("primary_score") is not None}
        candidates = [row for row in reservoir_rows if row["run_id"] in scores]
        if candidates:
            preferred_rows = sorted(candidates, key=lambda row: scores[row["run_id"]])
            break
    view = st.session_state.get(_REPLAY_VIEW_KEY) or {}
    explicit_run = st.session_state.pop("_report_open_run", None)
    preferred_run = str(explicit_run or view.get("run_id") or "")
    preferred_ids = [str(r["run_id"]) for r in preferred_rows]
    if preferred_run not in ([str(r["run_id"]) for r in table] if explicit_run else preferred_ids):
        preferred_run = preferred_ids[0]
    run_ids = [str(r["run_id"]) for r in reservoir_rows]
    st.sidebar.markdown("#### Experiment setup")
    selection_key = "report_run:" + dataset_id
    if explicit_run in run_ids or st.session_state.get(selection_key) not in run_ids:
        st.session_state[selection_key] = preferred_run
    run_id = st.sidebar.selectbox("Run", run_ids, key=selection_key,
                                 format_func=lambda rid: model_label(next(r for r in table if r["run_id"] == rid)), on_change=_pause_neural_runs)
    rec = next(r for r in reservoir_rows if str(r["run_id"]) == str(run_id))
    rdir = run_dir(dataset_id, run_id)
    if is_reservoir(rec.get("architecture")):
        with st.sidebar.expander("Connectome interpretation"):
            st.caption(EXPLORER_DISCLAIMER)
            _render_explorer_intro()
    scene = load_scene_from_run(rdir) if is_reservoir(rec.get("architecture")) else {}
    if synthetic_banner_required(rec, scene):
        st.warning(SYNTHETIC_DISCLAIMER)

    ws = read_status()
    if worker_alive():
        _pause_neural_runs()
        st.info(WORKER_BUSY_MESSAGE)
        if _active_train_live(ws) is not None:
            _render_train_live_activity(dataset_id, ws, component_key="explorer_train_live")
        _auto_refresh()
        from pdm.lab_ui import report_evaluations

        report_evaluations(dataset_id, run_id)
        return

    try:
        bound = bind_replay_to_run(dataset_id, run_id)
    except IncompatibleDataError as exc:
        _pause_neural_runs()
        st.error(str(exc))
        from pdm.lab_ui import report_evaluations

        report_evaluations(dataset_id, run_id)
        return
    except (OSError, ValueError) as exc:
        _pause_neural_runs()
        st.error(f"Incompatible data: run snapshot is incomplete ({exc})")
        from pdm.lab_ui import report_evaluations

        report_evaluations(dataset_id, run_id)
        return
    bundle = {
        **bound,
        "fingerprint": bound["current_fingerprint"],
        "dataset_version": bound["current_fingerprint"].get("dataset_version"),
    }
    unit_ids = [str(u) for u in bundle["units"]["unit_id"].tolist()]
    if not unit_ids:
        st.warning("No equipment units are available in the prepared data.")
        return
    test_ids = [str(u) for u in (bundle.get("split") or {}).get("test", [])]
    preferred_unit = str(view.get("unit_id") or st.session_state.get("replay_unit") or "")
    if preferred_unit not in unit_ids:
        preferred_unit = next((u for u in test_ids if u in unit_ids), unit_ids[0])
    uid = st.sidebar.selectbox("Unit", unit_ids, index=unit_ids.index(preferred_unit), on_change=_pause_neural_runs)

    render_equipment_simulation(dataset_id, rdir, str(uid), bundle)
    from pdm.lab_ui import report_evaluations

    report_evaluations(dataset_id, run_id)


def _replay_pressure_limit_pa(rdir, cfg: dict) -> float:
    """Snapshot when present; else live YAML (subtask 04 fallback)."""
    return load_run_pressure_limit_pa(rdir, cfg.get("pressure_limit_pa"))


def _freeze_checkpoint_hash(bound: dict, rdir) -> str | None:
    fp = bound.get("run_fingerprint") if isinstance(bound.get("run_fingerprint"), dict) else {}
    stored = fp.get("checkpoint_hash")
    if stored:
        return str(stored)
    ckpt = rdir / "best.pt"
    if ckpt.exists():
        from pdm.io_util import checkpoint_hash

        return checkpoint_hash(ckpt)
    return None


def _mode_unit_ids(split: dict, replay_mode: str) -> list[str]:
    key = "validation" if replay_mode == REPLAY_MODE_VALIDATION else "test"
    return [str(u) for u in (split.get(key) or [])]


def _evaluate_scope_noun(replay_mode: str) -> str:
    if replay_mode == REPLAY_MODE_VALIDATION:
        return "validation set"
    if replay_mode == REPLAY_MODE_RESEARCH:
        return "test set (not a blind benchmark)"
    return "test set"


def _overall_heading(replay_mode: str) -> str:
    if replay_mode == REPLAY_MODE_VALIDATION:
        return "**Overall validation**"
    if replay_mode == REPLAY_MODE_RESEARCH:
        return "**Overall test (not a blind benchmark)**"
    return "**Overall test**"


def _test_mode_replay_policy(frozen: dict | None, stored_policy: dict | None) -> dict | None:
    """Test Play/rescore uses freeze file / eval policy, never leftover widgets."""
    if isinstance(frozen, dict) and frozen.get("H_trigger") is not None:
        return frozen
    if isinstance(stored_policy, dict) and stored_policy.get("H_trigger") is not None:
        return stored_policy
    return None


def _evaluate_job(
    *,
    dataset_id: str,
    run_id: str,
    replay_mode: str,
    h_s: float,
    k: int,
    lead_s: float,
    max_useful: float | None,
) -> dict:
    job: dict = {"kind": "evaluate", "dataset_id": dataset_id, "run_id": run_id}
    if replay_mode == REPLAY_MODE_TEST:
        job["split_name"] = "test"
        job["policy_mode"] = "frozen"
        return job
    job["policy_mode"] = "research"
    job["split_name"] = "validation" if replay_mode == REPLAY_MODE_VALIDATION else "test"
    job["H_trigger"] = h_s
    job["warning_horizon_s"] = h_s
    job["confirmation_count"] = int(k)
    job["minimum_action_lead_time"] = lead_s
    job["max_useful_horizon_s"] = max_useful
    return job


def _open_neural_test_run() -> None:
    st.session_state["screen_selection"] = "Model Report"
    st.session_state["report_view"] = "Model replay"


def screen_replay(dataset_id: str) -> None:
    st.button("Open model replay", on_click=_open_neural_test_run)
    st.header(f"Test & Replay — {LABELS[dataset_id]}")
    st.info("Historical replay — not a live equipment connection")
    if dataset_id == "filters" and _filters_scale_unverified():
        _render_filters_time_warning()
    if not processed_ready(dataset_id):
        st.warning("Prepare data first.")
        return
    table = [r for r in list_runs(dataset_id) if r.get("has_best") or r.get("has_last")]
    if not table:
        st.warning("No saved model. Train first.")
        return
    run_id = st.selectbox("Saved model", [r["run_id"] for r in table])
    rdir = run_dir(dataset_id, run_id)
    try:
        bound = bind_replay_to_run(dataset_id, run_id)
    except IncompatibleDataError as exc:
        st.error(str(exc))
        return
    except FileNotFoundError as exc:
        st.error(f"Incompatible data: run snapshot is incomplete ({exc})")
        return
    features = bound["features"]
    units = bound["units"]
    split = bound["split"]
    replay_mode = st.radio("Replay mode", REPLAY_MODES, index=0, horizontal=True)
    mode_units = _mode_unit_ids(split, replay_mode)
    uid = st.selectbox(
        "Unit",
        mode_units,
        key=f"replay_unit_{dataset_id}_{run_id}_{replay_mode}",
    )
    cfg = load_dataset_config(dataset_id)
    train_units = units[units["unit_id"].isin(split["train"])]
    alerts_cfg = dict(cfg.get("alerts") or {})
    frozen = load_alert_policy(rdir)
    hk_enabled = replay_mode != REPLAY_MODE_TEST
    freeze_enabled = replay_mode == REPLAY_MODE_VALIDATION
    default_h = (
        float(frozen["H_trigger"])
        if frozen
        else default_horizon_s(train_units, float(alerts_cfg.get("horizon_fraction_of_median_train", 0.1)))
    )
    unit_h = st.selectbox(
        "Horizon unit",
        ["seconds", "minutes", "hours"],
        index=0 if dataset_id == "filters" else 1,
        help=(
            "For filters, seconds are internal Time × 60 values. Choosing minutes "
            "only divides by 60 — not calibrated wall-clock."
            if dataset_id == "filters"
            else "Horizon display unit. Bearings fragment interval is 60 s."
        ),
    )
    factor = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[unit_h]
    h_disp = st.number_input(
        "H_trigger (warning horizon)",
        value=float(default_h / factor),
        min_value=0.0,
        key=f"h_trigger_{dataset_id}_{run_id}",
        disabled=not hk_enabled,
        help="Trigger when predicted RUL ≤ H. Alias of warning_horizon_s. Changing H does not retrain.",
    )
    h_s = float(h_disp) * factor
    default_lead = (
        float(frozen["minimum_action_lead_time"])
        if frozen
        else float(h_s) * float(alerts_cfg.get("minimum_action_lead_time_fraction", 0.5))
    )
    lead_disp = st.number_input(
        "Minimum action lead time",
        value=float(default_lead / factor),
        min_value=0.0,
        key=f"min_lead_{dataset_id}_{run_id}",
        disabled=not hk_enabled,
        help="timely iff confirmed lead_time ≥ this value and the alert is before the event.",
    )
    lead_s = float(lead_disp) * factor
    frozen_max = frozen.get("max_useful_horizon_s") if frozen else alerts_cfg.get("max_useful_horizon_s")
    max_disp = st.number_input(
        "Max useful horizon (optional, 0 = none)",
        value=float((frozen_max or 0.0) / factor),
        min_value=0.0,
        key=f"max_useful_{dataset_id}_{run_id}",
        disabled=not hk_enabled,
        help="Alerts with lead_time above this are too_early. 0 disables the cap.",
    )
    max_s = float(max_disp) * factor
    max_useful = max_s if max_s > 0.0 else None
    default_k = int(frozen["confirmation_count"]) if frozen else int(alerts_cfg.get("confirmation_count", 3))
    k = st.number_input(
        "Confirmation count K",
        min_value=1,
        max_value=20,
        value=default_k,
        key=f"confirm_k_{dataset_id}_{run_id}",
        disabled=not hk_enabled,
    )
    delay = max(int(k) - 1, 0)
    st.caption(
        f"Alert confirms after {k} consecutive predictions with RUL ≤ H_trigger. "
        f"Confirmed time is the K-th step ({delay} extra measurement steps vs the first trigger). "
        "lead_time = event_time − confirmed_alert_time."
    )
    if dataset_id == "filters":
        st.caption(
            "H_trigger is a threshold on internal seconds (Time × 60), not an accuracy "
            "promise and not a calibrated “minutes before failure” clock. Changing H "
            "does not retrain the network."
        )
    else:
        st.caption(
            "H_trigger is a trigger threshold, not an accuracy promise. "
            "Changing H does not retrain the network."
        )
    if replay_mode == REPLAY_MODE_RESEARCH:
        st.caption("Research — not a blind benchmark.")
    if frozen:
        st.caption(
            f"Frozen run policy (`alert_policy.json`): H_trigger={frozen['H_trigger']:.4g} s, "
            f"minimum_action_lead_time={frozen['minimum_action_lead_time']:.4g} s, "
            f"K={frozen['confirmation_count']}. Widget changes are research-only until you freeze again."
        )
        if replay_mode == REPLAY_MODE_TEST:
            st.caption("Test evaluation uses the frozen validation-selected policy.")
    else:
        st.caption("No frozen policy yet. Tune on validation, then Freeze.")
        if replay_mode == REPLAY_MODE_TEST:
            st.caption(MISSING_FROZEN_POLICY_MESSAGE)
    if st.button("Freeze alert policy", disabled=not freeze_enabled):
        if replay_mode != REPLAY_MODE_VALIDATION:
            st.error("Freeze only from Validation.")
        else:
            save_alert_policy(
                rdir,
                build_alert_policy(
                    H_trigger=h_s,
                    minimum_action_lead_time=lead_s,
                    confirmation_count=int(k),
                    reset_factor=float(alerts_cfg.get("reset_factor", 1.2)),
                    max_useful_horizon_s=max_useful,
                    source="validation_ui",
                    split="validation",
                    unit_ids=[str(u) for u in (split.get("validation") or [])],
                    checkpoint_hash=_freeze_checkpoint_hash(bound, rdir),
                ),
                overwrite=True,
            )
            st.success("Saved frozen policy to alert_policy.json")
            st.rerun()
    eval_label = (
        "Evaluate validation set" if replay_mode == REPLAY_MODE_VALIDATION else "Evaluate test set"
    )
    if st.button(eval_label, disabled=worker_alive()):
        if replay_mode == REPLAY_MODE_TEST and frozen is None:
            st.error(MISSING_FROZEN_POLICY_MESSAGE)
        else:
            spawn_worker(
                _evaluate_job(
                    dataset_id=dataset_id,
                    run_id=run_id,
                    replay_mode=replay_mode,
                    h_s=h_s,
                    k=int(k),
                    lead_s=lead_s,
                    max_useful=max_useful,
                )
            )
            st.rerun()
    eval_running = worker_alive() and read_status().get("kind") in {"evaluate", "replay_predict"}
    if eval_running:
        st.info("Evaluating…")
        _auto_refresh()

    mode_evals = evaluations_for_mode(list_evaluations(rdir), replay_mode)
    eval_id = None
    if mode_evals:
        eval_id = st.selectbox(
            "Evaluation",
            [e["eval_id"] for e in mode_evals],
            key=f"eval_pick_{dataset_id}_{run_id}_{replay_mode}",
        )
        st.caption(
            "Each Evaluate writes a new `evaluations/<eval_id>/` and never overwrites a prior report. "
            "predictions.csv does not depend on H/K."
        )
    elif replay_mode == REPLAY_MODE_RESEARCH and (rdir / "predictions.csv").exists():
        st.caption(
            "Legacy run-root `predictions.csv` / `test_metrics.json` (read-only). "
            "New evaluations write under `evaluations/<eval_id>/`."
        )

    if eval_id:
        artifacts = resolve_evaluation_artifacts(rdir, eval_id)
    elif replay_mode == REPLAY_MODE_RESEARCH and not mode_evals:
        artifacts = legacy_evaluation_artifacts(rdir)
    else:
        artifacts = empty_evaluation_artifacts()
    table_unit_ids = mask_unit_ids(artifacts.get("evaluate_mask"), mode_units)
    stored_policy = load_evaluation_alert_policy(
        artifacts.get("eval_dir"),
        config_path=artifacts.get("evaluation_config"),
    )
    widget_policy = build_alert_policy(
        H_trigger=h_s,
        minimum_action_lead_time=lead_s,
        confirmation_count=int(k),
        reset_factor=float(alerts_cfg.get("reset_factor", 1.2)),
        max_useful_horizon_s=max_useful,
        source="replay_ui",
        fill_max_useful_from_cfg=False,
    )
    if replay_mode == REPLAY_MODE_TEST:
        replay_policy = _test_mode_replay_policy(frozen, stored_policy)
    else:
        replay_policy = widget_policy
    test_policy_missing = replay_mode == REPLAY_MODE_TEST and replay_policy is None
    if replay_policy is not None:
        replay_h_s: float | None = float(replay_policy["H_trigger"])
        policy_hash = str(replay_policy.get("policy_hash") or alert_policy_hash(replay_policy))
    else:
        replay_h_s = None
        policy_hash = "test-no-frozen-policy"
    split_label = split_label_for_unit(split, uid)
    _render_replay_header(
        run_id=run_id,
        split_label=split_label,
        eval_id=eval_id,
        h_s=replay_h_s,
        dataset_id=dataset_id,
        has_predictions=artifacts.get("predictions") is not None,
    )
    session_key = replay_session_key(dataset_id, run_id, uid, eval_id, policy_hash)
    _sync_replay_session(session_key)

    meas = features[features["unit_id"] == uid].sort_values("timestamp_s").reset_index(drop=True)
    n = len(meas)
    if n <= 0:
        st.warning("Selected unit has no measurements.")
        _render_evaluation_panel(dataset_id, artifacts, table_unit_ids, replay_mode=replay_mode)
        return
    preds_path = artifacts.get("predictions")
    has_preds = preds_path is not None
    frozen_alerts_path = artifacts.get("alerts")
    frozen_alerts = pd.read_csv(frozen_alerts_path) if frozen_alerts_path is not None else pd.DataFrame()
    stored_hash = None
    if stored_policy:
        stored_hash = stored_policy.get("policy_hash") or alert_policy_hash(stored_policy)
    policy_stale = (
        replay_mode != REPLAY_MODE_TEST
        and bool(stored_hash)
        and str(stored_hash) != str(policy_hash)
    )

    hist_csv = rdir / "training_history.csv"
    st.subheader("1. Training history")
    if hist_csv.exists():
        hdf = pd.read_csv(hist_csv)
        fig = go.Figure()
        for column in ("train_loss", "val_loss"):
            if column in hdf and hdf[column].notna().any():
                fig.add_trace(go.Scatter(x=hdf["epoch"], y=hdf[column], name=column.replace("_", " ")))
        if fig.data:
            fig.update_xaxes(title="Epoch")
            fig.update_yaxes(title="Loss")
            st.plotly_chart(fig, width="stretch")
        else:
            st.caption("Closed-form readout fitting has no gradient loss curve.")
            st.dataframe(hdf, hide_index=True, width="stretch")
    else:
        st.write("No training history file.")

    matching_eval = bool(eval_id) or bool(artifacts.get("legacy") and has_preds)
    uid_in_mask = str(uid) in set(table_unit_ids)
    play_enabled = bool(matching_eval and has_preds and uid_in_mask and not test_policy_missing)
    if not play_enabled:
        if test_policy_missing:
            st.error(
                f"{MISSING_FROZEN_POLICY_MESSAGE}. "
                "Test replay does not use widget H/K."
            )
        elif matching_eval and has_preds and not uid_in_mask:
            st.error(
                "Play is blocked: selected unit is not in this evaluation's unit mask. "
                "This screen does not run model inference on the Streamlit request thread."
            )
        else:
            st.error(
                "Play is blocked until Evaluate writes `evaluations/<eval_id>/predictions.csv`. "
                "This screen does not run model inference on the Streamlit request thread. "
                "Queue Evaluate (background worker) — not an unbounded inline Predictor."
            )
        if eval_running:
            st.info("Evaluate is running in the worker. Replay unlocks when predictions.csv is written.")
        _render_replay_controls(enabled=False, n=max(n, 1))
        _render_evaluation_panel(dataset_id, artifacts, table_unit_ids, replay_mode=replay_mode)
        return

    preds = pd.read_csv(preds_path)
    unit_pred = preds[preds["unit_id"] == uid].sort_values("timestamp_s").reset_index(drop=True)
    cache_key = (str(eval_id or ""), policy_hash, str(uid))
    pressure_limit_pa = _replay_pressure_limit_pa(rdir, cfg)
    block_alert_status = False
    if st.session_state.get("replay_alert_cache_key") != cache_key:
        try:
            episodes, steps = rescore_replay_alerts(
                unit_pred,
                replay_policy,
                run_id=run_id,
                measurements=meas,
                pressure_limit_pa=pressure_limit_pa,
            )
            st.session_state.replay_alert_cache_key = cache_key
            st.session_state.replay_alert_episodes = episodes
            st.session_state.replay_alert_steps = steps
        except Exception as exc:  # noqa: BLE001
            block_alert_status = True
            st.warning(
                "Could not rescore alerts for the current H/K. "
                f"Alert status is blocked to avoid a stale result ({exc})."
            )
            # Empty episodes — never fall back to frozen_alerts while status is blocked.
            st.session_state.pop("replay_alert_cache_key", None)
            st.session_state.replay_alert_episodes = pd.DataFrame()
            st.session_state.replay_alert_steps = pd.DataFrame()
    episodes = st.session_state.get("replay_alert_episodes", pd.DataFrame())
    steps = st.session_state.get("replay_alert_steps", pd.DataFrame())
    rescored = st.session_state.get("replay_alert_cache_key") == cache_key
    if policy_stale and not rescored:
        block_alert_status = True
        st.warning(
            "Alert policy changed (H/K). Frozen alerts.csv does not match the current widgets. "
            "Status is blocked until alerts are rescored from predictions.csv."
        )
    elif policy_stale:
        st.info(
            "Replay alerts are rescored in memory from frozen predictions.csv using the current H/K widgets. "
            "Evaluation tables and alerts.csv below still use the eval policy "
            f"`{(str(stored_hash)[:12] + '…') if stored_hash else 'unknown'}`."
        )
    unit_hit = units[units["unit_id"].astype(str) == str(uid)]
    unit_meta = unit_hit.iloc[0].to_dict() if len(unit_hit) else {}
    metrics = read_evaluation_metrics(artifacts)
    unit_table = metrics_by_unit_display_frame(
        read_metrics_by_unit(artifacts),
        dataset_id=dataset_id,
        unit_ids=table_unit_ids,
    )
    st.session_state[_REPLAY_VIEW_KEY] = {
        "dataset_id": dataset_id,
        "run_id": run_id,
        "unit_id": uid,
        "eval_id": eval_id,
        "split_label": split_label,
        "meas": meas,
        "unit_pred": unit_pred,
        "episodes": episodes,
        "steps": steps,
        "h_s": replay_h_s,
        "n": n,
        "block_alert_status": block_alert_status,
        "unit_meta": unit_meta,
        "pressure_limit_pa": pressure_limit_pa,
        "unit_table": unit_table,
        "overall_cards": evaluation_metric_cards(metrics, dataset_id),
        "replay_mode": replay_mode,
    }
    _mount_replay_playback(playing=bool(st.session_state.get("playing")))
    _render_evaluation_panel(dataset_id, artifacts, table_unit_ids, replay_mode=replay_mode)
    if frozen_alerts_path is not None:
        with st.expander("Retrospective eval alert log (all units)"):
            st.caption("Frozen `alerts.csv` from Evaluate. Not filtered to replay time; not the live H/K rescore.")
            st.dataframe(frozen_alerts, width="stretch", hide_index=True)


def _render_evaluation_panel(
    dataset_id: str, artifacts: dict, unit_ids: list, *, replay_mode: str
) -> None:
    st.subheader("Evaluation summary")
    eval_id = artifacts.get("eval_id")
    if eval_id:
        st.markdown(f"**eval_id:** `{eval_id}`")
        st.caption(f"Immutable report under `evaluations/{eval_id}/`.")
    elif artifacts.get("legacy"):
        st.caption("Legacy `test_metrics.json` (read-only). New evaluations write under `evaluations/<eval_id>/`.")

    metrics = read_evaluation_metrics(artifacts)
    if metrics is None:
        st.info(f"No metrics.json yet. Run Evaluate {_evaluate_scope_noun(replay_mode)}.")
        _eval_artifact_downloads(artifacts, dataset_id)
        return
    if not metrics.get("primary_metric"):
        st.caption("RUL metric tables are not populated yet (stub `metrics.json`).")

    cards = evaluation_metric_cards(metrics, dataset_id)
    if cards:
        for start in range(0, len(cards), 4):
            chunk = cards[start : start + 4]
            cols = st.columns(len(chunk))
            for col, card in zip(cols, chunk):
                col.metric(card["label"], card["value"])

    for note in evaluation_notes(metrics, dataset_id):
        st.caption(note)

    callout = baseline_coverage_callout(metrics)
    if callout:
        st.info(callout)

    zones = near_event_zone_frame(metrics)
    if zones is not None:
        st.caption("Near-event MAE uses frozen config zones, not thresholds fit on test.")
        st.dataframe(zones, width="stretch", hide_index=True)

    st.subheader("Per-unit metrics")
    table = metrics_by_unit_display_frame(
        read_metrics_by_unit(artifacts),
        dataset_id=dataset_id,
        unit_ids=[str(u) for u in (unit_ids or [])],
    )
    st.dataframe(table, width="stretch", hide_index=True)
    if dataset_id == "bearings":
        st.caption(
            "All units in the selected evaluation mask are listed as rows. "
            "XJTU-SY holds out 3 bearings (instance 5 in each regime)."
        )
    else:
        st.caption(
            "Prefix-end predicted RUL vs official RUL at prefix end. Official RUL is an "
            "evaluation label, not a sensor-observed 600 Pa event."
        )

    with st.expander("metrics.json (debug)"):
        st.json(metrics)
    _eval_artifact_downloads(artifacts, dataset_id)


def _eval_artifact_downloads(artifacts: dict, dataset_id: str) -> None:
    pred = artifacts.get("predictions")
    alerts = artifacts.get("alerts")
    if pred is None and alerts is None:
        return
    eval_key = artifacts.get("eval_id") or "legacy"
    if dataset_id == "filters":
        st.caption(
            "CSV columns use an _s suffix on the internal Time × 60 scale and keep "
            "time_original when present. That is not calibrated wall-clock minutes. "
            "Files are read from the evaluation directory."
        )
    else:
        st.caption("CSV downloads are the files in the evaluation directory, not a live rerun.")
    if pred is not None:
        st.download_button(
            "Download predictions CSV",
            data=pred.read_bytes(),
            file_name="predictions.csv",
            mime="text/csv",
            key=f"dl_predictions_{eval_key}",
        )
    if alerts is not None:
        st.download_button(
            "Download alerts CSV",
            data=alerts.read_bytes(),
            file_name="alerts.csv",
            mime="text/csv",
            key=f"dl_alerts_{eval_key}",
        )


def _format_h_trigger_header(h_s: float | None, dataset_id: str) -> str:
    if h_s is None or not _finite_number(h_s):
        return "unavailable"
    if dataset_id == "filters":
        return f"{h_s:.4g} s (internal)"
    return f"{h_s:.4g} s ({h_s / 60.0:.3g} min)"


def _format_replay_clock(t_s: float, dataset_id: str) -> str:
    if dataset_id == "filters":
        return f"{t_s:.1f} s (internal)"
    return f"{t_s / 60.0:.2f} min"


def _render_replay_header(
    *,
    run_id: str,
    split_label: str,
    eval_id: str | None,
    h_s: float | None,
    dataset_id: str,
    has_predictions: bool = False,
) -> None:
    bits = [
        f"**model:** `{run_id}`",
        f"**split:** `{split_label}`",
        f"**H_trigger:** `{_format_h_trigger_header(h_s, dataset_id)}`",
    ]
    if eval_id:
        bits.insert(1, f"**eval_id:** `{eval_id}`")
    st.markdown(" · ".join(bits))
    if eval_id:
        st.caption("Historical replay of a frozen evaluation. Predictions are not recomputed on this page.")
    elif has_predictions:
        st.caption(
            "No `evaluations/<eval_id>/` yet — using legacy run-root `predictions.csv`. "
            "Play is unlocked. Predictions are not recomputed on this page."
        )
    else:
        st.caption("No `evaluations/<eval_id>/` yet — Play stays blocked until Evaluate writes predictions.csv.")


def _sync_replay_session(key: tuple[str, str, str, str, str]) -> None:
    if st.session_state.get("replay_session_key") == key:
        st.session_state.setdefault("replay_step", 0)
        st.session_state.setdefault("playing", False)
        return
    st.session_state.replay_session_key = key
    st.session_state.replay_step = 0
    st.session_state.playing = False
    for name in _REPLAY_SESSION_KEYS:
        st.session_state.pop(name, None)


def _render_replay_controls(*, enabled: bool, n: int) -> tuple[int, bool]:
    speed = st.slider("Replay speed (steps per refresh)", 1, 20, 1, key="replay_speed", disabled=not enabled)
    show_gt = st.checkbox("Show ground truth", value=False, key="replay_show_gt", disabled=not enabled)
    st.caption(
        "Ground truth is an evaluator overlay only. It does not change predictions, "
        "baselines, or alerts. Official test RUL is an evaluation annotation, not a sensor fact."
    )
    b1, b2, b3, b4 = st.columns(4)
    play = b1.button("Play", disabled=not enabled)
    pause = b2.button("Pause", disabled=not enabled)
    step = b3.button("Step", disabled=not enabled)
    reset = b4.button("Reset", disabled=not enabled)
    if not enabled:
        return int(speed), bool(show_gt)
    if play:
        st.session_state.playing = True
        st.rerun()
    if pause:
        st.session_state.playing = False
        st.rerun()
    if step:
        st.session_state.replay_step = min(n - 1, int(st.session_state.get("replay_step", 0)) + 1)
        st.session_state.playing = False
        st.rerun()
    if reset:
        st.session_state.replay_step = 0
        st.session_state.playing = False
        st.session_state.pop("replay_cache", None)
        st.rerun()
    return int(speed), bool(show_gt)


def _mount_replay_playback(*, playing: bool) -> None:
    interval = REPLAY_PLAY_INTERVAL_S if playing else None

    @st.fragment(run_every=interval, key="replay_playback")
    def _replay_playback() -> None:
        _replay_playback_body()

    _replay_playback()


def _current_replay_interval(current) -> tuple[float, float] | None:
    """Only completed finite forecasts can supply an interval; raw warmup cannot."""
    if current is None or not _finite_number(current.get("predicted_rul_s")):
        return None
    lower, upper = current.get("lower_rul_s"), current.get("upper_rul_s")
    if not _finite_number(lower) or not _finite_number(upper):
        return None
    return (float(lower), float(upper)) if 0 <= float(lower) <= float(upper) else None


def _replay_demo_figure(
    *,
    dataset_id: str,
    prefix_meas: pd.DataFrame,
    prefix_pred: pd.DataFrame,
    overlay_rul: pd.Series | None,
    show_gt: bool,
    h_s: float,
    replay_time_s: float,
    warning_time_s: float | None,
    pressure_limit_pa: float,
    event_time_s: float | None,
    event_observed: bool,
) -> go.Figure:
    """Sensor and RUL on separate y-axes. Pressure is never plotted on the RUL axis."""
    filter_scale = dataset_id == "filters"
    x_scale = 1.0 if filter_scale else 60.0
    x_title = (
        "Internal time (s; Time × 60, unit unconfirmed)" if filter_scale else "Operating time (min)"
    )
    rul_title = "Remaining useful life (s, internal)" if filter_scale else "Remaining useful life (min)"
    sensor_title = "Sensor"
    rul_panel = "RUL"
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(sensor_title, rul_panel),
        row_heights=[0.45, 0.55],
    )
    g = prefix_meas
    if dataset_id == "bearings" and not g.empty and "horizontal_rms" in g.columns:
        fig.add_trace(
            go.Scatter(
                x=g["timestamp_s"] / x_scale,
                y=g["horizontal_rms"],
                name="horizontal RMS",
                mode="lines",
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="RMS", row=1, col=1)
    elif not g.empty and "differential_pressure" in g.columns:
        fig.add_trace(
            go.Scatter(
                x=g["timestamp_s"] / x_scale,
                y=g["differential_pressure"],
                name="Δp (Pa)",
                mode="lines",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(
            y=float(pressure_limit_pa),
            line_dash="dash",
            annotation_text=f"{pressure_limit_pa:.0f} Pa",
            row=1,
            col=1,
        )
        fig.update_yaxes(title_text="Differential pressure (Pa)", row=1, col=1)
    else:
        fig.update_yaxes(title_text="Sensor", row=1, col=1)

    rul_unit = "s" if filter_scale else "min"
    if len(prefix_pred) and "timestamp_s" in prefix_pred.columns:
        x = prefix_pred["timestamp_s"] / x_scale
        if {"lower_rul_s", "upper_rul_s"} <= set(prefix_pred.columns):
            intervals = prefix_pred.apply(_current_replay_interval, axis=1)
            lower = intervals.map(lambda bounds: bounds[0] / x_scale if bounds is not None else float("nan"))
            upper = intervals.map(lambda bounds: bounds[1] / x_scale if bounds is not None else float("nan"))
            if lower.notna().any():
                fig.add_trace(go.Scatter(
                    x=x, y=lower, mode="lines", line=dict(width=0),
                    name="forecast range lower", showlegend=False, hoverinfo="skip",
                    legendgroup="forecast_range", connectgaps=False,
                ), row=2, col=1)
                fig.add_trace(go.Scatter(
                    x=x, y=upper, mode="lines", line=dict(width=0),
                    fill="tonexty", fillcolor="rgba(61, 190, 225, 0.22)",
                    name=f"forecast range ({rul_unit})", legendgroup="forecast_range",
                    connectgaps=False,
                ), row=2, col=1)
                latest_bounds = _current_replay_interval(prefix_pred.iloc[-1])
                if latest_bounds is not None:
                    latest_x = float(prefix_pred.iloc[-1]["timestamp_s"]) / x_scale
                    fig.add_trace(go.Scatter(
                        x=[latest_x, latest_x], y=[bound / x_scale for bound in latest_bounds],
                        mode="lines+markers", line=dict(color="#3dbee1", width=3),
                        marker=dict(size=4), name="latest interval", showlegend=False,
                    ), row=2, col=1)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=prefix_pred["predicted_rul_s"] / x_scale,
                name=f"neural net RUL ({rul_unit})",
                mode="lines+markers",
                marker=dict(size=4),
            ),
            row=2,
            col=1,
        )
        if "baseline_rul_s" in prefix_pred.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=prefix_pred["baseline_rul_s"] / x_scale,
                    name=f"baseline RUL ({rul_unit})",
                    mode="lines",
                ),
                row=2,
                col=1,
            )
        if show_gt and overlay_rul is not None and overlay_rul.notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=overlay_rul.to_numpy(dtype=float) / x_scale,
                    name=f"actual RUL ({rul_unit}, evaluator overlay)",
                    line=dict(dash="dash"),
                    mode="lines",
                ),
                row=2,
                col=1,
            )
    fig.add_hline(y=h_s / x_scale, line_dash="dot", annotation_text="H_trigger", row=2, col=1)
    fig.update_yaxes(title_text=rul_title, row=2, col=1)
    fig.update_xaxes(title_text=x_title, row=2, col=1)

    if math.isfinite(replay_time_s):
        x_now = replay_time_s / x_scale
        fig.add_vline(x=x_now, line_dash="dot", annotation_text="now", row=1, col=1)
        fig.add_vline(x=x_now, line_dash="dot", row=2, col=1)
    if warning_time_s is not None and math.isfinite(warning_time_s):
        x_w = warning_time_s / x_scale
        fig.add_vline(x=x_w, line_dash="dash", line_color="orange", annotation_text="warning", row=1, col=1)
        fig.add_vline(x=x_w, line_dash="dash", line_color="orange", row=2, col=1)
    # Observed event only — never official_rul_overlay as a sensor crossing.
    if (
        event_observed
        and event_time_s is not None
        and math.isfinite(event_time_s)
        and math.isfinite(replay_time_s)
        and event_time_s <= replay_time_s
    ):
        x_e = event_time_s / x_scale
        label = "observed 600 Pa" if dataset_id == "filters" else "event"
        fig.add_vline(x=x_e, line_dash="dash", line_color="crimson", annotation_text=label, row=1, col=1)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), margin=dict(t=60))
    return fig


def _official_rul_caption(official_rul_s: float | None, *, show_gt: bool, at_end: bool) -> str | None:
    if official_rul_s is None:
        return None
    if not show_gt and not at_end:
        return None
    return (
        f"Official test RUL at prefix end: {official_rul_s:.1f} s "
        "(evaluation annotation, not a sensor-observed 600 Pa event and not future Δp)."
    )


def _render_end_of_unit_card(
    *,
    uid: str,
    end_state: dict,
    unit_table: pd.DataFrame | None,
    overall_cards: list[dict],
    dataset_id: str,
    replay_mode: str,
) -> None:
    st.subheader("End of unit")
    msg = end_state.get("message")
    if msg == END_OF_OBSERVED_DATA:
        st.info(END_OF_OBSERVED_DATA)
        bits = ["The file ended without an observed event."]
        if not end_state.get("sensor_limit_reached"):
            bits.append("This is not a 600 Pa threshold crossing.")
        bits.append("No future pressure samples are invented.")
        st.caption(" ".join(bits))
    elif msg:
        st.info(msg)
    note = _official_rul_caption(end_state.get("official_rul_s"), show_gt=True, at_end=True)
    if note:
        st.caption(note)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**This unit** (`{uid}`)")
        if unit_table is not None and not getattr(unit_table, "empty", True) and "Unit" in unit_table.columns:
            row = unit_table[unit_table["Unit"].astype(str) == str(uid)]
            if row.empty:
                st.caption("No per-unit evaluation row for this unit.")
            else:
                st.dataframe(row, width="stretch", hide_index=True)
        else:
            st.caption(f"No evaluation table yet. Run Evaluate {_evaluate_scope_noun(replay_mode)}.")
    with right:
        st.markdown(_overall_heading(replay_mode))
        cards = list(overall_cards or [])
        if not cards:
            if replay_mode == REPLAY_MODE_VALIDATION:
                st.caption("No overall validation metrics yet.")
            elif replay_mode == REPLAY_MODE_RESEARCH:
                st.caption("No overall test metrics yet (not a blind benchmark).")
            else:
                st.caption("No overall test metrics yet.")
        else:
            for start in range(0, min(len(cards), 4), 2):
                chunk = cards[start : start + 2]
                cols = st.columns(len(chunk))
                for col, card in zip(cols, chunk):
                    col.metric(card["label"], card["value"])
    table_word = (
        "overall validation table"
        if replay_mode == REPLAY_MODE_VALIDATION
        else (
            "overall test table (not a blind benchmark)"
            if replay_mode == REPLAY_MODE_RESEARCH
            else "overall test table"
        )
    )
    if dataset_id == "filters":
        st.caption(
            f"Unit row vs {table_word} from the frozen evaluation. "
            "Official RUL in that table is an evaluation label, not a sensor fact. "
            "This card does not change replay predictions."
        )
    else:
        st.caption(f"Unit row vs {table_word} from the frozen evaluation. Overlay only — predictions unchanged.")


def _replay_playback_body() -> None:
    view = st.session_state.get(_REPLAY_VIEW_KEY) or {}
    meas: pd.DataFrame = view.get("meas", pd.DataFrame())
    unit_pred: pd.DataFrame = view.get("unit_pred", pd.DataFrame())
    episodes: pd.DataFrame = view.get("episodes", pd.DataFrame())
    steps: pd.DataFrame = view.get("steps", pd.DataFrame())
    dataset_id = str(view.get("dataset_id") or "")
    run_id = str(view.get("run_id") or "")
    uid = str(view.get("unit_id") or "")
    eval_id = view.get("eval_id")
    h_s = float(view.get("h_s") or 0.0)
    n = int(view.get("n") or len(meas) or 1)
    block_status = bool(view.get("block_alert_status"))
    unit_meta = view.get("unit_meta") or {}
    pressure_limit_pa = float(view.get("pressure_limit_pa") or 600.0)
    unit_table = view.get("unit_table")
    overall_cards = view.get("overall_cards") or []
    n = max(n, 1)
    speed, show_gt = _render_replay_controls(enabled=True, n=n)
    if st.session_state.get("playing"):
        nxt = min(n - 1, int(st.session_state.get("replay_step", 0)) + int(speed))
        st.session_state.replay_step = nxt
        if nxt >= n - 1:
            st.session_state.playing = False
            st.rerun()
    step = max(0, min(n - 1, int(st.session_state.get("replay_step", 0))))
    st.session_state.replay_step = step
    prefix_meas = meas.iloc[: step + 1] if len(meas) else meas
    replay_time = float(prefix_meas.iloc[-1]["timestamp_s"]) if len(prefix_meas) else float("-inf")
    prefix_pred = slice_predictions_to_replay_time(unit_pred, replay_time)
    current = prefix_pred.iloc[-1] if len(prefix_pred) else None
    live_alerts = filter_replay_alert_log(episodes, unit_id=uid, replay_time_s=replay_time)
    at_last = step >= n - 1
    end_state = replay_end_state(
        dataset_id=dataset_id,
        unit_meta=unit_meta,
        replay_time_s=replay_time if math.isfinite(replay_time) else 0.0,
        prefix_measurements=prefix_meas,
        pressure_limit_pa=pressure_limit_pa,
        at_last_sample=at_last,
    )
    overlay = pd.Series(dtype="float64")
    if len(prefix_pred) and "timestamp_s" in prefix_pred.columns:
        overlay = pd.Series(
            evaluator_overlay_rul(prefix_pred["timestamp_s"], unit_meta, dataset_id=dataset_id),
            index=prefix_pred.index,
            dtype="float64",
        )
    warn_t = (
        None
        if block_status
        else active_warning_episode_time_s(
            live_alerts,
            unit_id=uid,
            replay_time_s=replay_time,
            steps=steps,
        )
    )
    event_t = None
    raw_et = unit_meta.get("event_time_s") if isinstance(unit_meta, dict) else None
    if _finite_number(raw_et):
        event_t = float(raw_et)

    st.subheader("2–3. Sensor and RUL")
    fig = _replay_demo_figure(
        dataset_id=dataset_id,
        prefix_meas=prefix_meas,
        prefix_pred=prefix_pred,
        overlay_rul=overlay,
        show_gt=show_gt,
        h_s=h_s,
        replay_time_s=replay_time,
        warning_time_s=warn_t,
        pressure_limit_pa=pressure_limit_pa,
        event_time_s=event_t,
        event_observed=bool(end_state.get("event_observed")),
    )
    st.plotly_chart(fig, width="stretch")
    mid_note = _official_rul_caption(
        end_state.get("official_rul_s"),
        show_gt=show_gt,
        at_end=bool(end_state.get("at_observation_end")),
    )
    if mid_note and not end_state.get("at_observation_end"):
        st.caption(mid_note)
    if show_gt and (overlay.empty or overlay.isna().all()) and not end_state.get("event_observed"):
        st.caption(
            "No actual-RUL overlay: this unit has no observed event. "
            "Official RUL, if present, stays an evaluation annotation."
        )

    filter_scale = dataset_id == "filters"
    if current is not None or len(prefix_meas):
        c1, c2, c3, c4 = st.columns(4)
        age_s = replay_time if math.isfinite(replay_time) else 0.0
        scale = 1.0 if filter_scale else 60.0
        display_unit = "s, internal" if filter_scale else "min"
        rul = current.get("predicted_rul_s") if current is not None else None
        interval = _current_replay_interval(current)
        c1.metric(f"Operating age ({display_unit})", f"{age_s / scale:.1f}")
        if interval is not None:
            # Anchor the absolute failure date to the measurement that generated
            # this forecast, even if a later observation has no saved prediction.
            forecast_at = float(current["timestamp_s"])
            low, high = interval
            c2.metric(f"Failure window ({display_unit})", f"{(forecast_at + low) / scale:.1f}–{(forecast_at + high) / scale:.1f}")
            c4.metric(f"Remaining life range ({display_unit})", f"{low / scale:.1f}–{high / scale:.1f}")
            st.caption(
                f"Forecast center: {float(rul) / scale:.1f} {display_unit} remaining. "
                "Empirical forecast range; not a guaranteed failure deadline."
            )
            if forecast_at != age_s:
                st.caption(f"Latest saved forecast was made at {_format_replay_clock(forecast_at, dataset_id)}.")
        else:
            c2.metric(f"Predicted RUL ({display_unit})", "—" if not _finite_number(rul) else f"{float(rul) / scale:.1f}")
            c4.metric("Model", run_id)
        c3.metric("Alert", _alert_status_for_row(current, live_alerts, uid, steps=steps, blocked=block_status))
    if warn_t is not None:
        st.caption(
            f"Warning episode time: {_format_replay_clock(warn_t, dataset_id)} "
            "(confirmed at the K-th step)."
        )
    st.caption("Replay speed does not change physical time, targets, or alert conditions.")
    if end_state.get("at_observation_end"):
        _render_end_of_unit_card(
            uid=uid,
            end_state=end_state,
            unit_table=unit_table,
            overall_cards=overall_cards,
            dataset_id=dataset_id,
            replay_mode=str(view.get("replay_mode") or REPLAY_MODE_TEST),
        )
    st.subheader("Alert log")
    st.caption("Selected unit only, timestamps ≤ current replay time. Full eval log is in the expander below.")
    st.dataframe(live_alerts, width="stretch", hide_index=True)
    visible = visible_replay_slice(prefix_pred, live_alerts)
    if not visible.empty:
        st.download_button(
            "Download visible replay slice CSV",
            data=visible.to_csv(index=False).encode("utf-8"),
            file_name=f"replay_slice_{uid}.csv",
            mime="text/csv",
            key=f"dl_visible_replay_{run_id}_{uid}_{eval_id or 'legacy'}",
        )
        st.caption(
            "Visible slice is issued forecasts and alerts with timestamp ≤ current replay time. "
            "Ground-truth overlay is omitted from this file."
        )


def _alert_status_for_row(
    current,
    alerts: pd.DataFrame,
    uid: str,
    *,
    steps: pd.DataFrame | None = None,
    blocked: bool = False,
) -> str:
    """Prefer rescored step status. Block stale/inconsistent status when requested."""
    if blocked:
        return "—"
    if current is None:
        return "—"
    if steps is not None and not getattr(steps, "empty", True) and "timestamp_s" in getattr(current, "index", []):
        return alert_status_at_time(steps, unit_id=uid, timestamp_s=float(current["timestamp_s"]))
    if "alert_status" in current.index and pd.notna(current.get("alert_status")):
        return str(current.get("alert_status"))
    if alerts is None or len(alerts) == 0:
        return "—"
    unit_alerts = alerts[alerts["unit_id"] == uid] if "unit_id" in alerts.columns else alerts
    if unit_alerts.empty or "timestamp_s" not in unit_alerts.columns:
        return "—"
    t = float(current["timestamp_s"])
    opened = unit_alerts[unit_alerts["timestamp_s"] <= t]
    return "Warning" if len(opened) else "No horizon alert"


def _split_map(split: dict) -> dict:
    m = {}
    for part in ("train", "validation", "test"):
        for u in split.get(part) or []:
            m[u] = part
    return m


@st.fragment(run_every=3.0)
def _watch_worker_lifecycle() -> None:
    """Detect CLI starts as well as completion, even from an idle screen."""
    alive = worker_alive()
    previous = st.session_state.get("_worker_watch_alive", alive)
    st.session_state["_worker_watch_alive"] = alive
    if previous != alive:
        st.rerun(scope="app")


@st.fragment(run_every=2.0)
def _worker_poll_fragment() -> None:
    ws = read_status()
    alive = worker_alive()
    previous = st.session_state.get("_worker_poll_alive", alive)
    st.session_state["_worker_poll_alive"] = alive
    if previous != alive:
        st.rerun(scope="app")
    st.caption(f"Worker status {ws.get('status', '—')} @ {time.strftime('%H:%M:%S')}")


def _auto_refresh(seconds: float = 2.0) -> None:
    try:
        _worker_poll_fragment()
    except Exception:
        time.sleep(min(max(seconds, 0.05), 2.0))
        st.rerun()


if __name__ == "__main__":
    main()
