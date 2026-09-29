"""One causal signal chart for a saved project signal run."""
from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm.data.project_prepare import load_zone_limits
from pdm.project_chart_style import add_threshold_layers, style_signal_chart
from pdm.projects import project_store
from pdm.signal_inference import forecast_prefix
from pdm.signal_training import list_project_runs, load_signal_run
from pdm.ui_theme import empty_state, page_header, tokens
from pdm.visualization.presentation import apply_explorer_style


def visible_observations(features: pd.DataFrame, unit_id: str, as_of_s: float) -> pd.DataFrame:
    """Only rows at the current playback time may reach the actual trace."""
    one = features.loc[features["unit_id"].astype(str) == str(unit_id)]
    return one.loc[pd.to_numeric(one["timestamp_s"], errors="coerce") <= as_of_s].sort_values("timestamp_s")


def _records(value) -> list[dict]:
    if isinstance(value, pd.DataFrame):
        return value.to_dict("records")
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def replay_figure(result: dict, schema: dict, theme: str = "dark") -> go.Figure:
    label = str(schema.get("signal_label") or schema.get("signal_column") or "Signal")
    unit = str(schema.get("signal_unit") or "")
    t = tokens(theme)
    actual_color, forecast_color = t["series_observed"], t["series_forecast"]
    red_color = t["zone_red"]
    observed = _records(result.get("observed_prefix"))
    all_points = _records(result.get("points"))
    points = [row for row in all_points if row.get("value") is not None]
    fig = go.Figure()
    observed_x, observed_y = [], []
    for row in observed:
        if row.get("gap_before") and observed_x:
            observed_x.append(None)
            observed_y.append(None)
        observed_x.append(row.get("timestamp_s"))
        observed_y.append(row.get("signal"))
    fig.add_trace(go.Scatter(x=observed_x, y=observed_y,
                             mode="lines+markers", name="Measurements received", line={"color": actual_color, "width": 2},
                             marker={"size": 5}))
    if points:
        fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in all_points], y=[r.get("value") for r in all_points],
                                 mode="lines+markers", name="Forecast horizon", line={"color": forecast_color, "width": 2},
                                 marker={"size": 6}, connectgaps=False))
        if any(row.get("lower") is not None and row.get("upper") is not None for row in points):
            fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in all_points],
                                     y=[r.get("upper") for r in all_points], mode="lines",
                                     line={"color": t["series_band_line"], "width": 1},
                                     name="Pointwise upper quantile", connectgaps=False))
            fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in all_points],
                                     y=[r.get("lower") for r in all_points], mode="lines", fill="tonexty",
                                     fillcolor=t["series_band"],
                                     line={"color": t["series_band_line"], "width": 1},
                                     name="Pointwise lower quantile", connectgaps=False))
    thresholds = result.get("thresholds") or schema.get("thresholds") or {}
    values = [float(v) for v in observed_y if v is not None] + [float(row["value"]) for row in points]
    add_threshold_layers(fig, thresholds, values, theme)
    crossing = result.get("crossing") or {}
    if crossing.get("time_s") is not None:
        point = next((r for r in points if float(r["target_time_s"]) == float(crossing["time_s"])), None)
        if point:
            fig.add_trace(go.Scatter(x=[point["target_time_s"]], y=[point["value"]], mode="markers",
                                     marker={"symbol": "x", "size": 12, "color": red_color, "line": {"width": 2}},
                                     name="Expected red entry"))
    as_of = result.get("as_of_s")
    if as_of is not None:
        fig.add_vline(x=float(as_of), line_dash="dot", line_width=1, line_color=t["series_reference"],
                      annotation_text="Now", annotation_position="top")
    fig.update_layout(height=440, margin={"l": 20, "r": 20, "t": 20, "b": 35},
                      xaxis_title="Time (s)", yaxis_title=f"{label} ({unit})", legend={"orientation": "h"},
                      hovermode="x unified")
    return style_signal_chart(fig, theme)


def _run_id(row: dict) -> str | None:
    return row.get("run_id") or row.get("id")


def _run_label(row: dict) -> str:
    engine_id = str(row.get("engine_id") or "model")
    engine = {"gru": "GRU", "lstm": "LSTM", "quantile_boosting": "Quantile boosting"}.get(
        engine_id, engine_id.replace("_", " ").title())
    created = str(row.get("created_at") or "")
    try:
        date = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%b %d, %H:%M")
    except ValueError:
        date = "Saved run"
    return f"{engine} · {date} · {str(_run_id(row))[-8:]}"


def advance_play_state(state: dict, length: int) -> dict:
    """Advance one timer tick; keep pause/reset stable at a finite cursor."""
    next_state = {"cursor": min(max(int(state.get("cursor", 0)), 0), max(length - 1, 0)),
                  "playing": bool(state.get("playing"))}
    if next_state["playing"] and next_state["cursor"] < length - 1:
        next_state["cursor"] += 1
    elif next_state["cursor"] >= length - 1:
        next_state["playing"] = False
    return next_state


def _show_forecast_context(result: dict, schema: dict, interval_status: str | None = None) -> None:
    points = _records(result.get("points"))
    supported = [row for row in points if row.get("value") is not None]
    if supported:
        furthest = max(float(row["target_time_s"]) for row in supported)
        st.caption(f"Forecast horizon through {furthest:g} s. Only saved model points are forecasts; gaps have no estimate.")
    else:
        st.info(str(result.get("reason") or "A forecast is unavailable at this point. Collect more continuous history."))
    crossing = result.get("crossing") or {}
    when = crossing.get("time_s")
    if crossing.get("status") == "already_red":
        st.warning("The latest measurement is already in the red zone.")
    elif when is not None:
        st.warning(f"Expected red entry at {float(when):g} s, the first supported forecast point across the red limit.")
    elif crossing.get("status") in {"none", "no_crossing", "not_predicted", "none_within_horizon"}:
        st.info("No red crossing is predicted within the shown forecast horizon.")
    else:
        st.caption("Red crossing time is unavailable for this prefix or threshold rule.")
    calibration = interval_status or result.get("calibration_status") or schema.get("calibration_status")
    if calibration == "unvalidated_pointwise_quantiles":
        st.caption("Blue band: pointwise 5–95% model quantiles. Its real-world coverage has not been calibrated or verified.")
    elif calibration:
        st.caption(f"Forecast interval status: {calibration.replace('_', ' ')}.")


@st.fragment(run_every=0.6)
def _play_fragment(project_id: str, run_id: str, unit_id: str, snapshot: dict,
                   interval_status: str | None = None, theme: str = "dark",
                   zone_limits: dict | None = None) -> None:
    # Streamlit replaces fragment content on each timer tick. Keep the theme
    # marker inside that content so the body/sidebar CSS remains selected.
    apply_explorer_style(theme)
    features = snapshot["features"]
    schema = snapshot["schema"]
    one = features.loc[features["unit_id"].astype(str) == str(unit_id)].sort_values("timestamp_s")
    clocks = sorted({float(t) for t in pd.to_numeric(one["timestamp_s"], errors="coerce").dropna()})
    if not clocks:
        st.warning("This unit has no admitted measurements.")
        return
    key = f"project_play:{project_id}:{run_id}:{unit_id}"
    state = st.session_state.setdefault(key, {"cursor": 0, "playing": False})
    slider_key = f"play_slider:{key}"

    def set_cursor() -> None:
        state["cursor"] = int(st.session_state[slider_key])
        state["playing"] = False

    c1, c2, c3 = st.columns([1, 1, 5])
    if c1.button("Pause" if state["playing"] else "Play", key=f"play_toggle:{key}", type="primary"):
        state["playing"] = not state["playing"]
    if c2.button("Reset", key=f"play_reset:{key}"):
        state.update(cursor=0, playing=False)
    else:
        state.update(advance_play_state(state, len(clocks)))
    st.session_state[slider_key] = state["cursor"]
    c3.slider("Observation time", min_value=0, max_value=len(clocks) - 1,
              format="sample %d", key=slider_key, on_change=set_cursor)
    as_of_s = clocks[state["cursor"]]
    time_label = f"{as_of_s / 60:g} min" if schema.get("source_kind") == "xjtu_bearings" else f"{as_of_s:g} s"
    st.caption(f"Measurements received through {time_label} · sample {state['cursor'] + 1} of {len(clocks)}")
    forecast_key = (project_id, run_id, snapshot["snapshot_id"], unit_id, as_of_s,
                    json.dumps(zone_limits, sort_keys=True))
    saved = st.session_state.get("project_last_forecast")
    if isinstance(saved, dict) and saved.get("key") == forecast_key:
        result = saved["result"]
    else:
        try:
            result = forecast_prefix(project_id, run_id, unit_id, as_of_s, thresholds=zone_limits)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Saved forecast cannot be opened: {exc}")
            return
        st.session_state["project_last_forecast"] = {"key": forecast_key, "result": result}
    if result.get("project_id") != project_id or result.get("run_id") != run_id or result.get("snapshot_id") != snapshot["snapshot_id"]:
        st.error("Forecast binding does not match this project, run, and snapshot.")
        return
    # The chart's observed line is derived independently from the cursor. A bad
    # forecast payload cannot reveal later measurements on the actual trace.
    result = dict(result)
    result["observed_prefix"] = visible_observations(features, unit_id, as_of_s)[["timestamp_s", "signal", "gap_before"]].to_dict("records")
    st.plotly_chart(replay_figure(result, schema, theme), width="stretch", theme=None)
    _show_forecast_context(result, schema, interval_status)


def render_results(project_id: str, snapshot: dict, selected_run_id: str | None = None,
                   theme: str = "dark") -> None:
    page_header("Results", "Replay a saved model on a held-out Test unit, one measurement at a time.")
    runs = []
    for row in list_project_runs(project_id):
        run_id = _run_id(row)
        if not run_id or row.get("snapshot_id") != snapshot["snapshot_id"]:
            continue
        try:
            load_signal_run(project_id, run_id)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
        runs.append(row)
    if not runs:
        empty_state("No saved model yet", "Train a signal model to see its saved forecast here.")
        return
    ids = [_run_id(row) for row in runs]
    initial = selected_run_id if selected_run_id in ids else ids[0]
    run_column, unit_column = st.columns(2)
    run_id = run_column.selectbox("Saved model run", ids, index=ids.index(initial),
                          format_func=lambda value: next((_run_label(row) for row in runs if _run_id(row) == value), str(value)),
                          key=f"result_run:{project_id}:{snapshot['snapshot_id']}")
    try:
        manifest = load_signal_run(project_id, run_id)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error(f"Saved model run cannot be verified: {exc}")
        return
    if manifest.get("project_id") != project_id or manifest.get("snapshot_id") != snapshot["snapshot_id"]:
        st.error("This saved model belongs to another project or data snapshot.")
        return
    if run_id != selected_run_id:
        project_store().update(project_id, selected_run_id=run_id)
    units = sorted(str(uid) for uid in snapshot["split"].get("test") or [])
    if not units:
        st.warning("This snapshot has no held-out Test unit for replay.")
        return
    unit_id = unit_column.selectbox("Test unit", units, key=f"result_unit:{project_id}:{run_id}")
    pair = (project_id, run_id, unit_id)
    if st.session_state.get("result_active_pair") != pair:
        for key in list(st.session_state):
            if str(key).startswith(("project_play:", "play_slider:")):
                st.session_state.pop(key, None)
        st.session_state.pop("project_last_forecast", None)
        st.session_state["result_active_pair"] = pair
    engine = manifest.get("engine_id") or manifest.get("engine") or "Saved model"
    st.caption(f"{engine} · held-out Test unit · signal forecast in {snapshot['schema'].get('signal_unit', 'native units')}")
    _play_fragment(project_id, run_id, unit_id, snapshot, manifest.get("interval_status"), theme,
                   load_zone_limits(project_id, snapshot["snapshot_id"]))
    metrics = (manifest.get("metrics") or {}).get("test") or {}
    if metrics:
        st.subheader("Held-out Test summary")
        c1, c2 = st.columns(2)
        c1.metric("Known future measurements", metrics.get("known_targets", "—"))
        error = metrics.get("mae")
        c2.metric(f"Mean absolute error ({snapshot['schema'].get('signal_unit', '')})",
                  "—" if error is None else f"{float(error):.3g}")
        horizons = metrics.get("by_horizon") or []
        if horizons:
            rows = [{"Horizon (s)": row.get("horizon_s"),
                     "Known measurements": row.get("known_targets"),
                     "Test units": row.get("independent_units"),
                     f"Mean absolute error ({snapshot['schema'].get('signal_unit', '')})": row.get("mae")}
                    for row in horizons]
            st.table(pd.DataFrame(rows), hide_index=True, border="horizontal")
