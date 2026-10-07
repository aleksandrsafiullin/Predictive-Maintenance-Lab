"""One Live monitor screen for all project snapshot formats."""

from __future__ import annotations

import json
from functools import partial

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm import live_simulator as demo
from pdm import stable_forecast as stable
from pdm.corridor_calibration import load_active
from pdm.data.project_prepare import read_zone_limits
from pdm.io_util import read_json
from pdm.live_monitor import LiveModel, available_runs, load_config, read_live_folder, save_config
from pdm.long_forecast_ui import _label
from pdm.project_chart_style import add_threshold_layers, style_signal_chart
from pdm.project_results_ui import _run_label
from pdm.project_snapshot import project_snapshot
from pdm.projects import project_store
from pdm.ui_theme import empty_state, page_header, tokens

ZONE_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴", "unknown": "⚪"}
ZONE_TEXT = {
    "green": "Green zone",
    "yellow": "Yellow zone",
    "red": "Red zone",
    "unknown": "No limits",
}
REFRESH_CHOICES = [5, 10, 30, 60, 300]


@st.cache_resource(show_spinner=False, max_entries=8)
def _live_model(project_id, run_id, root, identity):
    return LiveModel(project_id, run_id)


@st.cache_data(show_spinner=False, max_entries=256, ttl=3600)
def _score_machine(_model, unit, identity, use_calibration, calibration_key):
    return _model.assess(unit, use_calibration=use_calibration)


def _duration(seconds):
    if seconds is None:
        return "—"
    return f"{seconds / 60:.0f} min" if seconds < 5400 else f"{seconds / 3600:.1f} h"


def _forecast_text(result):
    if result["zone"] == "red":
        return "Already red"
    if result["status"] != "available":
        return result.get("reason") or "No forecast"
    if result["crossing"]["status"] == "predicted":
        return f"Red in ~{_duration(result['red_in_s'])}"
    if result["crossing"]["status"] == "none_within_horizon":
        return "No red within forecast range"
    return "No saved red limit"


def _select_machine(project_id, table_key, unit_ids):
    """Map a click using the row order shown when its callback was registered."""
    selection = st.session_state[table_key]["selection"]
    # A cell click selects its machine; the row marker also remains usable.
    rows = [selection["cells"][0][0]] if selection.get("cells") else selection["rows"]
    if rows:
        st.session_state[f"live_pick:{project_id}"] = unit_ids[rows[0]]


def live_figure(result, schema, theme):
    colors = tokens(theme)
    fig = go.Figure()
    x, y = [], []
    for row in result["observed_prefix"]:
        if row["gap_before"] and x:
            x.append(None)
            y.append(None)
        x.append(row["timestamp_s"])
        y.append(row["signal"])
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name="Measurements received",
            line=dict(color=colors["series_observed"], width=1),
            marker_size=3,
        )
    )
    points = result["points"]
    times = [row["target_time_s"] for row in points]
    if points and all(
        row.get("lower") is not None and row.get("upper") is not None for row in points
    ):
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[row["lower"] for row in points],
                line_width=0,
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[row["upper"] for row in points],
                fill="tonexty",
                fillcolor=colors["series_band"],
                line=dict(color=colors["series_band_line"], width=0.5),
                name="Trend corridor",
            )
        )
    if points:
        fig.add_trace(
            go.Scatter(
                x=times,
                y=[row["value"] for row in points],
                line=dict(color=colors["series_forecast"], width=3),
                name="Trend center",
            )
        )
    values = [row["signal"] for row in result["observed_prefix"]]
    values += [
        row[key]
        for row in points
        for key in ("value", "lower", "upper")
        if row.get(key) is not None
    ]
    add_threshold_layers(fig, result["thresholds"], values, theme)
    fig.add_vline(
        x=result["as_of_s"],
        line_dash="dot",
        line_color=colors["series_reference"],
        annotation_text="Now",
    )
    fig.update_layout(
        height=440,
        margin=dict(l=25, r=25, t=35, b=40),
        xaxis_title="Time (s)",
        yaxis_title=f"{schema.get('signal_label', 'Signal')} ({schema.get('signal_unit', '')})",
        legend=dict(orientation="h", y=1.15),
        hovermode="x unified",
    )
    return style_signal_chart(fig, theme)


def _demo_controls(project_id):
    state = demo.demo_status(project_id)
    running = state["state"] == "running"
    with st.expander("Demo feed · recorded measurements", expanded=True):
        options = demo.demo_units(project_id)
        c1, c2 = st.columns([3, 1])
        units = c1.multiselect(
            "Machines to replay",
            options,
            default=options[:3],
            disabled=running,
            key=f"live_demo_units:{project_id}",
        )
        speed = c2.select_slider(
            "Measurements per refresh",
            [1, 5, 10, 30, 60],
            value=10,
            key=f"live_demo_speed:{project_id}",
        )
        st.caption(
            "Testing and Validation measurements arrive progressively in a separate demo folder. The feed advances while this screen is open."
        )
        c1, c2, c3 = st.columns(3)
        if c1.button(
            "Start demo feed", disabled=running or not units, type="primary", width="stretch"
        ):
            demo.start_demo(project_id, units)
            st.rerun()
        if c2.button("Stop demo feed", disabled=not running, width="stretch"):
            demo.stop_demo(project_id)
            st.rerun()
        if c3.button(
            "Clear demo data", disabled=running or not state.get("folder"), width="stretch"
        ):
            demo.clear_demo(project_id)
            st.rerun()
    state = demo.advance_demo(project_id, speed)
    if state.get("units"):
        written = sum(row["written"] for row in state["units"].values())
        total = sum(row["total"] or 0 for row in state["units"].values())
        st.caption(
            f"Demo feed {state['state']} · {written} of {total or '…'} measurements received"
        )
    if state.get("reason"):
        st.info(state["reason"])
    return state.get("folder")


def render_live_monitor(project_id, project_name, snapshot=None, theme="dark"):
    snapshot = snapshot or project_snapshot(project_id)
    page_header(
        "Live monitor", "Current measurements and the saved model's forecast for each machine."
    )
    runs = available_runs(project_id, snapshot["snapshot_id"])
    if not runs:
        empty_state(
            "No saved model yet",
            "Train a signal model for the current data, then open Live monitor.",
        )
        return
    config = load_config(project_id)
    ids = [row["run_id"] for row in runs]
    selected = config.get("run_id") or project_store().get(project_id).get("selected_run_id")
    labels = {row["run_id"]: _label(row) if row.get("config") else _run_label(row) for row in runs}
    c1, c2 = st.columns([3, 1])
    run_id = c1.selectbox(
        "Saved model",
        ids,
        index=ids.index(selected) if selected in ids else 0,
        format_func=labels.get,
        key=f"live_run:{project_id}",
    )
    refresh = c2.selectbox(
        "Refresh every",
        REFRESH_CHOICES,
        index=REFRESH_CHOICES.index(config["refresh_s"])
        if config["refresh_s"] in REFRESH_CHOICES
        else 1,
        format_func=lambda s: f"{s} s" if s < 60 else f"{s // 60} min",
        key=f"live_refresh:{project_id}",
    )
    mode = st.radio(
        "Measurements from",
        ["Live folder", "Demo feed"],
        horizontal=True,
        key=f"live_source:{project_id}",
    )
    folder = config["folder"]
    if mode == "Live folder":
        folder = st.text_input("Watched folder", value=folder, key=f"live_folder:{project_id}")
        signal = snapshot["schema"].get("signal_column") or "signal"
        st.caption(
            f"CSV columns: unit_id, timestamp_s (or timestamp as a date-time), {signal} (or signal). Files are read on the computer running this app."
        )
    changes = {
        key: value
        for key, value in dict(run_id=run_id, refresh_s=refresh, folder=folder.strip()).items()
        if config.get(key) != value
    }
    if changes:
        config = save_config(project_id, changes)
    row = runs[ids.index(run_id)]
    calibrated = row.get("protocol") in stable.PROTOCOLS
    use_calibration = (
        st.checkbox(
            "Use saved Calibration corridor",
            value=True,
            key=f"live_calibration:{project_id}:{run_id}",
        )
        if calibrated
        else False
    )
    if row.get("protocol") and row["protocol"] != stable.PROTOCOL:
        st.caption("Archive model · uses its saved forecast parameters and history policy.")
    period = 2 if mode == "Demo feed" and demo.demo_status(project_id)["state"] == "running" else refresh
    _fleet(
        project_id,
        run_id,
        config,
        mode,
        theme,
        use_calibration,
        period,
    )


def _fleet(project_id, run_id, config, mode, theme, use_calibration, run_every):
    @st.fragment(run_every=run_every)
    def body():
        try:
            snapshot = project_snapshot(project_id)
            record = read_json(project_store().run_path(project_id, run_id) / "manifest.json")
            identity = json.dumps(
                dict(
                    root=str(project_store().root),
                    snapshot_id=snapshot["snapshot_id"],
                    run=record,
                    limits=read_zone_limits(snapshot["dir"]),
                ),
                sort_keys=True,
            )
            model = _live_model(project_id, run_id, str(project_store().root), identity)
            calibration = (
                load_active(project_id, run_id, model.run)
                if model.run.get("protocol") in stable.PROTOCOLS
                else None
            )
            calibration_key = json.dumps(calibration, sort_keys=True)
            folder = _demo_controls(project_id) if mode == "Demo feed" else config["folder"]
            if not folder:
                empty_state(
                    "No measurements yet", "Start the demo feed to receive recorded measurements."
                )
                return
            data = read_live_folder(folder, model.signal_column)
            if data.empty:
                empty_state(
                    "No measurements yet",
                    "Put sensor CSV files in the watched folder, or select Demo feed.",
                )
                return
            results = [
                _score_machine(model, unit, identity, use_calibration, calibration_key)
                for _, unit in data.groupby("unit_id", sort=True)
            ]
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Live monitor could not refresh: {exc}")
            return
        results.sort(
            key=lambda r: (
                -{"unknown": -1, "green": 0, "yellow": 1, "red": 2}[r["zone"]],
                r["red_in_s"] if r["red_in_s"] is not None else float("inf"),
                r["unit_id"],
            )
        )
        columns = st.columns(4)
        columns[0].metric("Machines", len(results))
        for i, zone in enumerate(("green", "yellow", "red"), 1):
            columns[i].metric(
                f"{ZONE_ICON[zone]} {ZONE_TEXT[zone]}", sum(r["zone"] == zone for r in results)
            )
        by_id = {r["unit_id"]: r for r in results}
        unit_ids = tuple(by_id)
        key = f"live_pick:{project_id}"
        if st.session_state.get(key) not in by_id:
            st.session_state[key] = unit_ids[0]
        table_key = f"live_table:{project_id}"
        # Keep the highlighted row with the machine when status sorting changes.
        st.session_state[table_key] = {
            "selection": {"rows": [unit_ids.index(st.session_state[key])]}
        }
        st.caption("Click a machine row to show its forecast.")
        st.dataframe(
            pd.DataFrame(
                [
                    dict(
                        Status=f"{ZONE_ICON[r['zone']]} {ZONE_TEXT[r['zone']]}",
                        Machine=r["unit_id"],
                        **{
                            f"Latest ({model.schema.get('signal_unit', '')})": round(
                                r["current"], 4
                            )
                        },
                        Forecast=_forecast_text(r),
                        Measurements=r["n_rows"],
                        **{"History used": r["history_observations"]},
                    )
                    for r in results
                ]
            ),
            hide_index=True,
            width="stretch",
            key=table_key,
            on_select=partial(_select_machine, project_id, table_key, unit_ids),
            selection_mode=["single-row-required", "single-cell"],
        )
        picked = st.session_state[key]
        result = by_id[picked]
        st.subheader(f"Machine detail · {ZONE_ICON[result['zone']]} {picked}")
        st.plotly_chart(
            live_figure(result, model.schema, theme),
            width="stretch",
            theme=None,
            key=f"live_chart:{project_id}",
        )
        st.caption(
            f"{_forecast_text(result)} · {result['history_observations']} observations used for this forecast."
        )
        if result.get("calibration"):
            cal = result["calibration"]
            st.caption(
                f"Calibration corridor · full width {100 * cal['width']:.1f}% · limits {100 * cal['settings']['min_width']:g}–{100 * cal['settings']['max_width']:g}%"
            )
        elif calibrated_model(model) and use_calibration:
            st.caption(
                "Saved training corridor · no Calibration result has been applied to this model."
            )

    body()


def calibrated_model(model):
    return model.run.get("protocol") in stable.PROTOCOLS
