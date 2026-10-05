"""Project controls for frozen first-RED event models."""

from __future__ import annotations

import json
import uuid
from functools import partial

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm.cli import spawn_worker
from pdm.data.project_prepare import load_zone_limits
from pdm.red_entry_inference import forecast_red_entry_prefix
from pdm.red_entry_protocol import load_protocol
from pdm.red_entry_training import available_red_entry_engines
from pdm.ui_theme import style_figure, tokens
from pdm.worker import read_status, worker_alive

ENGINE_LABELS = {
    "gru": "GRU",
    "lstm": "LSTM",
    "hazard_boosting": "Hazard boosting",
    "full_cns": "Full MaleCNS",
    "kaplan_meier": "Kaplan–Meier (known age)",
    "always_no_entry": "No RED entry baseline",
    "trend_to_red": "Trend to RED baseline",
}
MODE_LABELS = {
    "age_context": "Operating age and conditions",
    "sensor_only": "Sensors only",
    "hybrid": "Sensors, operating age and conditions",
}
STATUS_LABELS = {
    "available": "Available",
    "unavailable": "Unavailable",
    "insufficient_context": "Insufficient data",
    "unsupported_regime": "Unsupported regime",
    "already_red": "Already RED",
    "post_event": "After first RED",
    "at_risk": "Before first RED",
    "unknown_event_history": "Unknown event history",
    "requirements_unset": "Requirements unset",
    "exploratory": "Exploratory",
    "insufficient_event_evidence": "Insufficient independent events",
    "not_passed": "Not passed",
    "passed": "Passed",
    "fitted": "Fitted",
    "not_validated": "Not validated",
    "validated": "Validated",
    "active": "Active",
    "inactive": "Inactive",
    "supported_exploratory": "Exploratory support",
    "unsupported": "Unsupported",
    "rule_unavailable": "RED rule unavailable",
    "selected_development": "Selected on development data",
    "continuation_of_observed_regime_assumption": "Observed regime continuation",
    "policy_not_frozen": "Warning policy not frozen",
    "exploratory_frozen_policy": "Frozen exploratory policy",
    "unsupported_policy_horizon": "Unsupported warning horizon",
    "operational_requirements_unset": "Requirements unset",
    "no_independent_holdout": "No independent holdout",
    "partial_policy_prediction_coverage": "Partial prediction coverage",
    "model_or_policy_not_frozen": "Model or policy not frozen",
    "interval_coverage_not_validated": "Interval coverage not validated",
    "interval_calibration_support_not_validated": "Interval calibration not validated",
    "insufficient_interval_event_evidence": "Insufficient interval events",
    "regime_coverage_not_validated": "Regime coverage not validated",
    "max_alert_time_fraction_not_met": "Alert time budget exceeded",
    "max_false_alert_episodes_per_100_operating_hours_not_met": "False alert budget exceeded",
}


def readable_status(value) -> str:
    """Keep unfamiliar internal tokens in diagnostics, without inferring their meaning."""
    if value is None or value == "":
        return "N/A"
    return STATUS_LABELS.get(str(value), "Unknown")


def _number(value, *, percent=False) -> str:
    if value is None or not pd.notna(value):
        return "N/A"
    return f"{float(value):.1%}" if percent else f"{float(value):g}"


def render_event_evaluation(manifest: dict) -> None:
    """Show saved evidence only; no re-evaluation or assumptions about missing metrics."""
    st.subheader("Saved event evaluation")
    evaluations = manifest.get("metrics") or {}
    parts = [part for part in ("test", "validation", "holdout") if evaluations.get(part)]
    if not parts:
        st.info("No saved event evaluation.")
    else:
        labels = {"test": "Historical Test", "validation": "Development Validation", "holdout": "Holdout"}
        part = st.selectbox("Evaluation split", parts, format_func=labels.get)
        evaluation = evaluations[part]
        metrics = evaluation.get("metrics") or {}
        gate = evaluation.get("quality_gate") or {}
        lead_unset = "minimum_action_lead_s" in (gate.get("unset_requirements") or [])
        statuses = metrics.get("event_status_counts") or {}
        events = evaluation.get("events") or []
        lead_unset = lead_unset or bool(statuses.get("action_lead_requirement_unset")) or any(
            event.get("status") == "action_lead_requirement_unset" for event in events)
        unknown_events = bool(statuses.get("unknown")) or any(event.get("status") == "unknown" for event in events)
        scope = evaluation.get("metric_scope") or {}
        recall_scope = scope.get("useful_event_recall") or {}
        recall_unavailable = lead_unset or unknown_events or recall_scope.get("status") == "unavailable"
        c1, c2, c3 = st.columns(3)
        c1.metric("Observed RED events", _number(metrics.get("event_count")))
        c2.metric("Physical units with RED", _number(metrics.get("independent_event_units")))
        reachability_label = "Events within policy horizon" if lead_unset else "Reachable warning events"
        c3.metric(reachability_label, _number(metrics.get("reachable_event_count")))
        summary = [
            ("Useful event recall", "useful_event_recall", True),
            ("Reachable useful event recall", "reachable_useful_event_recall", True),
            ("Green-zone useful event recall", "green_useful_event_recall", True),
            ("Confirmed alert episodes", "confirmed_alert_episodes", False),
            ("False alert episodes", "false_alert_episodes", False),
            ("Unknown alert episodes", "unknown_alert_episodes", False),
            ("False alerts per 100 operating hours", "false_alert_episodes_per_100_operating_hours", False),
            ("Alert time fraction", "alert_time_fraction", True),
        ]
        st.table(pd.DataFrame([
            {"Metric": label, "Saved value": _number(None if recall_unavailable and key.endswith("useful_event_recall") else metrics.get(key), percent=percent)}
            for label, key, percent in summary
        ]), hide_index=True)
        rows = evaluation.get("brier_by_horizon") or []
        if rows:
            st.table(pd.DataFrame([{
                "Horizon (s)": row.get("horizon_s"),
                "Brier": row.get("brier"),
                "Known origins": row.get("known_origins"),
                "Excluded origins": row.get("unknown_origins"),
                "Equipment": row.get("physical_units"),
                "Prediction coverage": row.get("prediction_coverage"),
            } for row in rows]), hide_index=True)
        st.table(pd.DataFrame([
            {"Metric": "Policy prediction coverage", "Saved value": _number(scope.get("policy_prediction_coverage"), percent=True)},
            {"Metric": "Operating exposure (s)", "Saved value": _number(metrics.get("operating_exposure_s"))},
            {"Metric": "Evaluation gate", "Saved value": readable_status(gate.get("status"))},
        ] + ([{"Metric": "Alert time scope", "Saved value": "Partial predictions"}]
             if scope.get("alert_time") == "partial_observed_predictions_only" else [])), hide_index=True)
    with st.expander("Run diagnostics", expanded=False):
        st.json(manifest)


def render_event_training(project_id: str, snapshot: dict) -> None:
    protocol = load_protocol()
    capabilities = available_red_entry_engines(project_id, snapshot["snapshot_id"])
    engines = [row["engine_id"] for row in capabilities if row.get("available")]
    if not engines:
        st.warning("No RED entry models available for these data.")
        return
    defaults = protocol["training"]
    recurrent = defaults["recurrent"]
    grid = protocol["horizons_s"].get(snapshot["schema"].get("source_kind"))
    with st.form(f"red_entry_train:{project_id}:{snapshot['snapshot_id']}"):
        engine = st.selectbox("Event model", engines, format_func=lambda x: ENGINE_LABELS[x])
        mode = st.selectbox("Inputs", list(MODE_LABELS), format_func=lambda x: MODE_LABELS[x])
        raw_grid = st.text_input("Probability horizons (s, comma-separated)", value=", ".join(map(str, grid or [])))
        c1, c2 = st.columns(2)
        history = c1.number_input("History length", 1, 256, recurrent["history_length"])
        epochs = c2.number_input("Max GRU/LSTM epochs", 1, 500, recurrent["max_epochs"])
        seed = st.number_input("Random seed", 0, 2**31 - 1, defaults["seed"])
        busy = worker_alive() or read_status().get("status") in {
            "queued", "running", "training", "preparing", "stopping",
        }
        submitted = st.form_submit_button("Train first RED model", type="primary", disabled=busy)
    if submitted:
        from pdm.project_training_ui import _parse_horizons

        try:
            params = dict(
                input_mode=mode,
                horizons_s=_parse_horizons(raw_grid),
                history_length=int(history),
                epochs=int(epochs),
                seed=int(seed),
            )
            spawn_worker(
                dict(
                    kind="project_train",
                    task="red_entry",
                    job_id=uuid.uuid4().hex,
                    project_id=project_id,
                    snapshot_id=snapshot["snapshot_id"],
                    engine_id=engine,
                    params=params,
                )
            )
            st.success("Training queued.")
            st.rerun()
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Training failed to start: {exc}")


def event_chart_payload(result: dict, prefix: pd.DataFrame, thresholds: dict) -> dict:
    """Adapt event result to auxiliary signal chart; corridor times are absolute."""
    columns = [name for name in ("timestamp_s", "signal", "gap_before") if name in prefix]
    return {
        "as_of_s": result["issued_at_s"],
        "observed_prefix": prefix[columns].to_dict("records"),
        "points": [],
        "thresholds": thresholds,
        "red_entry_corridor": result.get("red_entry_corridor", {}),
    }


def probability_figure(result: dict, theme: str = "dark") -> go.Figure:
    rows = result.get("probability_by_horizon") or []
    fig = go.Figure(
        go.Scatter(
            x=[row["horizon_s"] for row in rows],
            y=[row.get("probability") for row in rows],
            mode="lines+markers",
            connectgaps=False,
            name="First RED probability",
            line={"color": tokens(theme)["series_forecast"]},
        )
    )
    fig.update_layout(
        xaxis_title="Horizon from now (s)",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        height=280,
    )
    return style_figure(fig, theme)


def render_event_payload(result: dict, theme: str = "dark", *, unit_id: str | None = None) -> None:
    displayed_unit = result.get("unit_id") or unit_id or "N/A"
    st.caption(
        f"Equipment: {displayed_unit} · "
        f"Now: {_number(result.get('issued_at_s'))} s (recording time)"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric(
        "Prediction",
        readable_status(result.get("prediction_status")),
    )
    c2.metric(
        "Event state",
        readable_status(result.get("risk_status")),
    )
    c3.metric(
        "Model age source",
        {
            "counter": "Counter",
            "laboratory_proxy": "Laboratory proxy",
            "running_clock": "Running clock",
            "unknown": "Unknown",
        }.get(result.get("age_source"), "N/A"),
    )
    st.plotly_chart(probability_figure(result, theme), width="stretch", theme=None)
    rows = result.get("probability_by_horizon") or []
    if rows:
        st.table(
            pd.DataFrame([{
                "horizon_s": row.get("horizon_s"),
                "probability": row.get("probability"),
                "support_status": readable_status(row.get("support_status")),
            } for row in rows]).rename(
                columns={
                    "horizon_s": "Horizon (s)",
                    "probability": "RED probability",
                    "support_status": "Data support",
                }
            ),
            hide_index=True,
        )
    warning = result.get("warning") or {}
    st.caption(f"Warning: {readable_status(warning.get('status'))}")
    quantiles = result.get("time_to_red_quantiles_s") or {}
    if quantiles:
        st.table(pd.DataFrame([
            {"Quantile": {"q05": "5%", "q50": "50%", "q95": "95%"}.get(key, key),
             "Grid time to RED (s)": "Not reached" if value is None else _number(value)}
            for key, value in quantiles.items()
        ]), hide_index=True)
    corridor = result.get("red_entry_corridor") or {}
    st.caption(
        f"First RED corridor: {corridor.get('earliest_s')}–{corridor.get('latest_s')} s (recording time)"
        if corridor.get("status") == "available" else "First RED corridor: N/A"
    )


def _seek_event_cursor(state: dict, cursor: int, slider_key: str) -> None:
    """Ignore late slider events from an earlier timer cursor."""
    if slider_key != state.get("active_slider"):
        return
    if cursor == state.get("displayed_cursor", state["cursor"]):
        return
    state.update(cursor=cursor, playing=False)


@st.fragment(run_every=0.6)
def render_event_replay(
    project_id: str, run_id: str, unit_id: str, snapshot: dict, theme="dark"
) -> None:
    from pdm.project_results_ui import (
        _background_forecasts,
        advance_play_state,
        replay_figure,
        visible_observations,
    )
    from pdm.project_zones import resolve_thresholds
    from pdm.visualization.presentation import apply_explorer_style

    apply_explorer_style(theme)
    one = (
        snapshot["features"]
        .loc[snapshot["features"].unit_id.astype(str) == unit_id]
        .sort_values("timestamp_s")
    )
    clocks = sorted(one.timestamp_s.dropna().astype(float).unique())
    if not clocks:
        st.warning("No measurements to replay.")
        return
    limits = load_zone_limits(project_id, snapshot["snapshot_id"])
    rule_key = json.dumps(limits or snapshot["schema"].get("thresholds"), sort_keys=True)
    key = f"project_play:event:{project_id}:{run_id}:{unit_id}:{rule_key}"
    state = st.session_state.setdefault(key, {"cursor": 0, "playing": False})

    def pause():
        state["playing"] = False

    def toggle_play():
        state["playing"] = not state["playing"]

    def reset():
        state.update(cursor=0, playing=False)

    show_future = st.toggle(
        "Show future actuals (review)",
        key=f"event_future:{key}",
        on_change=pause,
    )
    c1, c2, c3 = st.columns([1, 1, 5])
    c1.button("Pause" if state["playing"] else "Play", key=f"event_play:{key}", on_click=toggle_play)
    c2.button("Reset", key=f"event_reset:{key}", on_click=reset)
    if state.get("ready") == state["cursor"]:
        state.update(advance_play_state(state, len(clocks)))
    slider = f"play_slider:{key}:{state['cursor']}"

    def seek():
        if slider in st.session_state:
            _seek_event_cursor(state, int(st.session_state[slider]), slider)

    state["active_slider"] = slider
    state["displayed_cursor"] = state["cursor"]
    if len(clocks) > 1:
        c3.slider("Observed measurements", 0, len(clocks) - 1, value=state["cursor"], key=slider, on_change=seek)
    issued = clocks[state["cursor"]]
    forecast_key = (
        "red_entry",
        project_id,
        snapshot["snapshot_id"],
        run_id,
        unit_id,
        issued,
        rule_key,
    )
    worker = _background_forecasts()
    st.session_state["project_forecast_worker"] = worker
    job = worker.request(
        forecast_key,
        partial(forecast_red_entry_prefix, project_id, run_id, unit_id, issued, thresholds=limits),
    )
    if job.get("error"):
        state["playing"] = False
        st.error(f"Event forecast failed: {job['error']}")
        return
    result = job.get("result") if job.get("done") else None
    if (
        not result
        or result.get("model_id") != run_id
        or result.get("snapshot_id") != snapshot["snapshot_id"]
        or result.get("issued_at_s") != issued
    ):
        st.info("Computing forecast…")
        return
    state["ready"] = state["cursor"]
    prefix = visible_observations(snapshot["features"], unit_id, issued)
    schema = {**snapshot["schema"], **({"thresholds": limits} if limits else {})}
    payload = event_chart_payload(result, prefix, resolve_thresholds(schema, prefix))
    render_event_payload(result, theme, unit_id=unit_id)
    latest = prefix.iloc[-1]
    age = latest.get("operating_age_s")
    known_age = str(latest.get("operating_age_known", False)).lower() in {"true", "1", "1.0"}
    st.caption(
        f"Operating age: {float(age):g} s"
        if known_age and pd.notna(age)
        else "Operating age: N/A"
    )
    regime = {
        name: latest[name]
        for name in (
            "rpm",
            "load_kn",
            "flow_rate",
            "dust_feed",
            "dust",
            "temperature",
            "is_running",
        )
        if name in latest and pd.notna(latest[name])
    }
    regime_labels = {
        "rpm": "Speed (rpm)", "load_kn": "Load (kN)",
        "flow_rate": "Flow rate", "dust_feed": "Dust feed", "dust": "Dust",
        "temperature": "Temperature", "is_running": "Running state",
    }
    st.caption(
        "Conditions: "
        + (", ".join(f"{regime_labels[name]}: {value}" for name, value in regime.items()) or "N/A")
    )
    future = one.loc[one.timestamp_s > issued] if show_future else None
    st.plotly_chart(
        replay_figure(payload, schema, theme, future_actual=future), width="stretch", theme=None
    )
    with st.expander("Diagnostics"):
        st.json({key: value for key, value in result.items() if key not in ("observed_prefix",)})
