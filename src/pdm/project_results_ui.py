"""One causal signal chart for a saved project signal run."""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from datetime import datetime
from functools import partial

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm.data.project_prepare import load_zone_limits
from pdm.forecast_worker import ForecastWorker
from pdm.project_chart_style import add_threshold_layers, style_signal_chart
from pdm.project_zones import is_beyond, resolve_thresholds
from pdm.projects import project_store
from pdm.red_entry_training import list_red_entry_runs, load_red_entry_run
from pdm.signal_inference import forecast_prefix
from pdm.signal_training import list_project_runs, load_signal_run
from pdm.ui_theme import empty_state, page_header, tokens
from pdm.visualization.presentation import apply_explorer_style


@st.cache_resource(scope="session", show_spinner=False, on_release=lambda worker: worker.close())
def _background_forecasts() -> ForecastWorker:
    return ForecastWorker()


def cancel_replay_forecast() -> None:
    worker = st.session_state.pop("project_forecast_worker", None)
    if worker is not None:
        worker.cancel()


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


def first_recorded_red_after_now(observed: list[dict], future: list[dict],
                                 thresholds: dict, as_of_s: float | None) -> dict | None:
    """Retrospective display only; never feed this outcome to forecast_prefix."""
    if as_of_s is None or thresholds.get("status") != "available" or thresholds.get("red") is None:
        return None
    rows = sorted([*_records(observed), *_records(future)], key=lambda row: float(row["timestamp_s"]))
    baseline_n = int(thresholds.get("baseline_n") or 1)
    for index, row in enumerate(rows):
        if index < baseline_n - 1:
            continue
        try:
            value = float(row["signal"])
            time_s = float(row["timestamp_s"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(time_s) or not math.isfinite(value):
            continue
        if bool(is_beyond([value], thresholds["red"], thresholds["direction"])[0]):
            return row if time_s > float(as_of_s) else None
    return None


def _red_entry_corridor(result: dict) -> tuple[float, float] | None:
    """Accept only an explicit model-issued interval, never infer one from Test data."""
    entry = result.get("red_entry_corridor") or {}
    empirical = entry.get("status") in {"empirical_conditional", "learned", "derived"}
    if entry.get("status") != "available" and not empirical:
        return None
    try:
        start, end = float(entry["earliest_s"]), float(entry["latest_s"])
        now = float(result["as_of_s"])
    except (KeyError, TypeError, ValueError):
        return None
    valid = now <= start < end if empirical else now < start <= end
    return (start, end) if all(map(math.isfinite, (start, end, now))) and valid else None


def replay_figure(result: dict, schema: dict, theme: str = "dark", *,
                  future_actual=None) -> go.Figure:
    label = str(schema.get("signal_label") or schema.get("signal_column") or "Signal")
    unit = str(schema.get("signal_unit") or "")
    t = tokens(theme)
    actual_color, forecast_color = t["series_observed"], t["series_forecast"]
    red_color = t["zone_red"]
    observed = _records(result.get("observed_prefix"))
    as_of = result.get("as_of_s")
    if as_of is not None:
        observed = [r for r in observed if float(r["timestamp_s"]) <= float(as_of)]
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
    anchor = ([{"target_time_s": observed[-1]["timestamp_s"], "value": observed[-1]["signal"],
                "lower": observed[-1]["signal"], "upper": observed[-1]["signal"]}]
              if observed else [])
    if points:
        direct = anchor + [r for r in all_points if r.get("kind") not in {"recursive", "anchor"}]
        recursive = [r for r in all_points if r.get("kind") == "recursive"]
        if (result.get("funnel") or {}).get("mode") not in {"learned_joint_trajectories", "bounded_trend_corridor"}:
            fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in direct], y=[r.get("value") for r in direct],
                                 mode="lines+markers", name="Forecast horizon", line={"color": forecast_color, "width": 2},
                                 marker={"size": 6}, connectgaps=False))
        if recursive:
            continuation = direct[-1:] + recursive
            fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in continuation],
                                     y=[r.get("value") for r in continuation], mode="lines+markers",
                                     name="Recursive forecast", line={"color": forecast_color, "width": 2, "dash": "dash"},
                                     marker={"size": 4}, connectgaps=False))
        if any(row.get("lower") is not None and row.get("upper") is not None for row in points):
            joint = bool(result.get("funnel"))
            bounded = (result.get("funnel") or {}).get("mode") == "bounded_trend_corridor"
            fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in direct],
                                     y=[r.get("upper") for r in direct], mode="lines",
                                     line={"color": t["series_band_line"], "width": 1},
                                     name="Trend corridor upper" if bounded else "Forecast band upper" if joint else "Pointwise upper quantile", connectgaps=False))
            fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in direct],
                                     y=[r.get("lower") for r in direct], mode="lines", fill="tonexty",
                                     fillcolor=t["series_band"],
                                     line={"color": t["series_band_line"], "width": 1},
                                     name="Trend corridor" if bounded else "Forecast band" if joint else "Pointwise lower quantile", connectgaps=False))
    future = sorted([r for r in _records(future_actual)
                     if as_of is not None and float(r["timestamp_s"]) > float(as_of)],
                    key=lambda r: float(r["timestamp_s"]))
    if future:
        future_x = [observed[-1]["timestamp_s"]] if observed else []
        future_y = [observed[-1]["signal"]] if observed else []
        for row in future:
            if row.get("gap_before") and future_x:
                future_x.append(None)
                future_y.append(None)
            future_x.append(row["timestamp_s"])
            future_y.append(row["signal"])
        fig.add_trace(go.Scatter(x=future_x, y=future_y, mode="lines",
                                 name="Future actual · hidden from model", opacity=0.55,
                                 line={"color": actual_color, "width": 2}, connectgaps=False))
    thresholds = result.get("thresholds") or schema.get("thresholds") or {}
    values = ([float(v) for v in observed_y if v is not None] + [float(row["value"]) for row in points]
              + [float(row[key]) for row in points for key in ("lower", "upper") if row.get(key) is not None]
              + [float(row["signal"]) for row in future if row.get("signal") is not None])
    add_threshold_layers(fig, thresholds, values, theme)
    corridor = _red_entry_corridor(result)
    if corridor:
        fig.add_vrect(x0=corridor[0], x1=corridor[1], fillcolor=t["series_band"],
                      line_width=0, annotation_text="RED window" if (result.get("funnel") or {}).get("mode") in {"learned_joint_trajectories", "bounded_trend_corridor"} else "Conditional RED window" if result.get("funnel") else "Predicted RED-entry window",
                      annotation_position="top left")
    actual_red = first_recorded_red_after_now(observed, future, thresholds, as_of) if future else None
    if actual_red is not None:
        fig.add_trace(go.Scatter(x=[actual_red["timestamp_s"]], y=[actual_red["signal"]],
                                 mode="markers", name="First recorded RED sample · hidden from model",
                                 marker={"symbol": "circle-open", "size": 12, "color": red_color,
                                         "line": {"width": 2}}))
    crossing = result.get("crossing") or {}
    if crossing.get("time_s") is not None:
        point = next((r for r in points if float(r["target_time_s"]) == float(crossing["time_s"])), None)
        if point:
            fig.add_trace(go.Scatter(x=[point["target_time_s"]], y=[point["value"]], mode="markers",
                                     marker={"symbol": "x", "size": 12, "color": red_color, "line": {"width": 2}},
                                     name="Forecast point in red zone"))
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
    engine = {"gru": "GRU", "lstm": "LSTM", "quantile_boosting": "Quantile boosting",
              "full_cns": "Fly brain · Full MaleCNS"}.get(
        engine_id, engine_id.replace("_", " ").title())
    created = str(row.get("created_at") or "")
    try:
        date = datetime.fromisoformat(created.replace("Z", "+00:00")).strftime("%b %d, %H:%M")
    except ValueError:
        date = "Saved run"
    task = ("First RED entry" if row.get("task") == "red_entry" else
            "Trend corridor" if (row.get("params") or {}).get("forecast_mode") == "bounded_trend_corridor" else "Signal")
    return f"{task} · {engine} · {date}"


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
    if result.get("corridor_contract") and result.get("status") != "available":
        st.info(result.get("reason") or "The trend corridor is unavailable at this observation time.")
        return
    if (result.get("funnel") or {}).get("mode") == "bounded_trend_corridor":
        bounds = result.get("corridor_contract") or result["funnel"]
        target = bounds.get("target_relative_width", .20)
        maximum = bounds.get("maximum_relative_width", .30)
        st.caption(f"Trend corridor · ±{50*target:g}% of the predicted level, up to ±{50*maximum:g}%. "
                   "Containment is uncalibrated; misses remain forecast errors.")
        entry = result.get("red_entry_corridor") or {}
        window = _red_entry_corridor(result)
        if window:
            st.metric("First RED entry · from corridor",
                      f"{_time_label(window[0]-result['as_of_s'], schema)} – "
                      f"{_time_label(window[1]-result['as_of_s'], schema)} ahead")
            st.caption("This window follows from the two boundary crossings, conditional on the signal staying inside the corridor.")
        elif entry.get("status") == "open":
            st.info(f"RED entry is possible from {_time_label(entry['earliest_s']-result['as_of_s'], schema)} ahead. "
                    "The lower-risk boundary does not cross within this span; the end of the window is open.")
        elif entry.get("status") == "already_red":
            st.warning("The latest measurement is already in the red zone.")
        elif entry.get("status") == "previously_red":
            st.caption("A RED entry has already been recorded in the received history.")
        else:
            st.info("No RED entry is predicted within the shown corridor horizon.")
        return
    if result.get("funnel"):
        funnel = result["funnel"]
        status = str(funnel.get("calibration_status") or result.get("calibration_status")
                     or funnel.get("status") or "unavailable")
        st.caption(f"Band: {status.replace('_', ' ')}")
        entry = result.get("red_entry_corridor") or {}
        probability = entry.get("probability_within_horizon")
        if probability is not None:
            c1, c2 = st.columns(2)
            learned = funnel.get("mode") == "learned_joint_trajectories"
            c1.metric("RED within horizon" if learned else "RED within horizon · empirical", f"{float(probability):.1%}")
            corridor = _red_entry_corridor(result)
            c2.metric("RED window" if learned else "Conditional RED window · empirical", "Open" if corridor is None and learned else "N/A" if corridor is None else
                      f"{_time_label(corridor[0] - float(result['as_of_s']), schema)} – "
                      f"{_time_label(corridor[1] - float(result['as_of_s']), schema)}")
        return
    points = _records(result.get("points"))
    supported = [row for row in points if row.get("value") is not None]
    if supported:
        furthest = max(float(row["target_time_s"]) for row in supported)
        st.caption(f"Forecast through {_time_label(furthest, schema)} · "
                   f"{_time_label(furthest - float(result['as_of_s']), schema)} ahead of Now.")
    crossing = result.get("crossing") or {}
    when = crossing.get("time_s")
    if crossing.get("status") == "already_red":
        st.warning("The latest measurement is already in the red zone.")
    elif when is not None:
        st.warning(f"First saved forecast point in the red zone: "
                   f"{_time_label(float(when) - float(result['as_of_s']), schema)} ahead "
                   f"(at {_time_label(float(when), schema)}). This is not a validated alert time.")
    elif crossing.get("status") in {"none", "no_crossing", "not_predicted", "none_within_horizon"}:
        st.info("No red crossing is predicted within the shown forecast horizon.")
    elif crossing.get("status") == "pending":
        st.caption("Searching for red entry; the forecast is still being calculated.")
    rollout = result.get("rollout") or {}
    if rollout.get("status") in {"unavailable", "stopped"}:
        st.warning(str(rollout["reason"]))
    if any(row.get("kind") == "recursive" for row in points):
        st.caption("Dashed blue: recursive forecast. The saved model receives its own predicted values after the direct horizon. "
                   "Accuracy over this longer horizon has not been evaluated; uncertainty can grow at each step.")
        if rollout.get("interpolated_inputs"):
            st.caption("Intermediate inputs between saved model horizons are interpolated at the Training sample interval.")
    calibration = interval_status or result.get("calibration_status") or schema.get("calibration_status")
    if calibration == "unvalidated_pointwise_quantiles":
        st.caption("Blue band: pointwise 5–95% model quantiles. Its real-world coverage has not been calibrated or verified.")
    elif calibration:
        st.caption(f"Forecast interval status: {calibration.replace('_', ' ')}.")
    corridor = _red_entry_corridor(result)
    if corridor:
        st.caption(f"Model RED-entry window: {_time_label(corridor[0] - float(result['as_of_s']), schema)} "
                   f"to {_time_label(corridor[1] - float(result['as_of_s']), schema)} ahead "
                   f"(width {_time_label(corridor[1] - corridor[0], schema)}). "
                   "Its coverage must be checked on independent units.")
    elif crossing.get("status") != "pending":
        st.caption("No model-issued RED-entry time window is available for this saved run. "
                   "Pointwise signal bands do not establish an event-time window.")


def _time_label(seconds: float, schema: dict) -> str:
    return f"{seconds / 60:g} min" if schema.get("source_kind") == "xjtu_bearings" else f"{seconds:g} s"


def _lead_label(seconds: float, schema: dict) -> str:
    return f"{seconds / 60:.1f} min" if schema.get("source_kind") == "xjtu_bearings" else f"{seconds:.1f} s"


def forecast_span_options(horizons_s: list[float]) -> tuple[list[float], int]:
    """Bound span choices to saved support, defaulting to 30 minutes."""
    supported = max(float(h) for h in horizons_s)
    first = min(float(h) for h in horizons_s)
    options = sorted({max(first, min(minutes * 60.0, supported))
                      for minutes in (10, 20, 30, 60, 120)} | {supported})
    return options, options.index(max(first, min(30 * 60.0, supported)))


def replay_forecast_key(project_id: str, run_id: str, snapshot_id: str, unit_id: str,
                        as_of_s: float, zone_limits: dict | None,
                        prediction_horizon_s: float | None = None) -> tuple:
    return (project_id, run_id, snapshot_id, unit_id, as_of_s,
            json.dumps(zone_limits, sort_keys=True), prediction_horizon_s)


@st.fragment(run_every=0.6)
def _play_fragment(project_id: str, run_id: str, unit_id: str, snapshot: dict,
                   interval_status: str | None = None, theme: str = "dark",
                   zone_limits: dict | None = None, *, background: bool = False,
                   learned_horizons_s: list[float] | None = None) -> None:
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
        # An already queued widget event can outlive a model/unit switch.
        if slider_key not in st.session_state:
            return
        state["cursor"] = int(st.session_state[slider_key])
        state["playing"] = False

    def pause() -> None:
        state["playing"] = False

    prediction_horizon_s = None
    if learned_horizons_s:
        spans, default_index = forecast_span_options(learned_horizons_s)
        supported = max(spans)
        prediction_horizon_s = st.selectbox(
            "Forecast span", spans, index=default_index,
            format_func=lambda span: (f"Full supported horizon ({span / 60:g} min)"
                                      if span == supported else f"{span / 60:g} min"),
            key=f"forecast_span:{project_id}:{run_id}", on_change=pause)
    request_options = {"prediction_horizon_s": prediction_horizon_s} if prediction_horizon_s is not None else {}
    show_future = st.toggle("Show future actual measurements", value=False,
                            key=f"future_actual:{project_id}:{run_id}", on_change=pause,
                            help="Show recorded Test measurements after Now in gray, for comparison only. "
                                 "The model never receives these future measurements.")
    c1, c2, c3 = st.columns([1, 1, 5])
    if c1.button("Pause" if state["playing"] else "Play", key=f"play_toggle:{key}", type="primary"):
        state["playing"] = not state["playing"]
    if c2.button("Reset", key=f"play_reset:{key}"):
        state.update(cursor=0, playing=False)
    elif not background or state.get("forecast_ready_cursor") == state["cursor"]:
        # Heavy models must finish and display this cursor before Play advances.
        state.update(advance_play_state(state, len(clocks)))
    st.session_state[slider_key] = state["cursor"]
    c3.slider("Observation time", min_value=0, max_value=len(clocks) - 1,
              format="sample %d", key=slider_key, on_change=set_cursor)
    as_of_s = clocks[state["cursor"]]
    time_label = f"{as_of_s / 60:g} min" if schema.get("source_kind") == "xjtu_bearings" else f"{as_of_s:g} s"
    st.caption(f"Measurements received through {time_label} · sample {state['cursor'] + 1} of {len(clocks)}")
    forecast_key = replay_forecast_key(project_id, run_id, snapshot["snapshot_id"], unit_id,
                                       as_of_s, zone_limits, prediction_horizon_s)
    saved = st.session_state.get("project_last_forecast")
    pending = failed = False
    if isinstance(saved, dict) and saved.get("key") == forecast_key:
        cancel_replay_forecast()
        result = saved["result"]
    elif background:
        worker = _background_forecasts()
        st.session_state["project_forecast_worker"] = worker
        job = worker.request(forecast_key, partial(
            forecast_prefix, project_id, run_id, unit_id, as_of_s,
            thresholds=dict(zone_limits) if zone_limits else None, **request_options))
        pending = not job["done"]
        if job["error"]:
            failed = True
            state["playing"] = False
            st.error(f"Saved forecast cannot be opened: {job['error']}")
        prefix = visible_observations(features, unit_id, as_of_s)
        result = job["result"] or {
            "project_id": project_id, "run_id": run_id, "snapshot_id": snapshot["snapshot_id"],
            "as_of_s": as_of_s, "points": [], "reason": "Calculating the first forecast points…",
            "thresholds": resolve_thresholds({**schema, **({"thresholds": zone_limits} if zone_limits else {})}, prefix),
            "crossing": {"status": "pending", "time_s": None}}
        if job["done"] and not job["error"] and job["result"] is not None:
            st.session_state["project_last_forecast"] = {"key": forecast_key, "result": result}
        if pending:
            st.info("Calculating the saved model's direct forecast…")
    else:
        cancel_replay_forecast()
        try:
            with st.spinner("Calculating forecast…"):
                result = forecast_prefix(project_id, run_id, unit_id, as_of_s, thresholds=zone_limits,
                                         **request_options)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Saved forecast cannot be opened: {exc}")
            return
        st.session_state["project_last_forecast"] = {"key": forecast_key, "result": result}
    state["forecast_ready_cursor"] = None if pending or failed else state["cursor"]
    if result.get("project_id") != project_id or result.get("run_id") != run_id or result.get("snapshot_id") != snapshot["snapshot_id"]:
        st.error("Forecast binding does not match this project, run, and snapshot.")
        return
    # The chart's observed line is derived independently from the cursor. A bad
    # forecast payload cannot reveal later measurements on the actual trace.
    result = dict(result)
    result["observed_prefix"] = visible_observations(features, unit_id, as_of_s)[["timestamp_s", "signal", "gap_before"]].to_dict("records")
    future = one.loc[one["timestamp_s"] > as_of_s] if show_future else None
    if future is not None and prediction_horizon_s is not None:
        displayed_span = float((result.get("funnel") or {}).get("issued_horizon_s") or prediction_horizon_s)
        issued = result.get("as_of_s")
        future = future.loc[future["timestamp_s"] <= (float(issued) if issued is not None else as_of_s) + displayed_span]
    st.plotly_chart(replay_figure(result, schema, theme, future_actual=future), width="stretch", theme=None)
    if not failed:
        _show_forecast_context(result, schema, interval_status)


def render_results(project_id: str, snapshot: dict, selected_run_id: str | None = None,
                   theme: str = "dark") -> None:
    page_header("Results", "")
    runs = []
    for row in sorted(list_project_runs(project_id) + list_red_entry_runs(project_id), key=lambda row: str(row.get("created_at", "")), reverse=True):
        run_id = _run_id(row)
        if not run_id or row.get("snapshot_id") != snapshot["snapshot_id"]:
            continue
        if row.get("task") != "red_entry":
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
        selected_row = next(row for row in runs if _run_id(row) == run_id)
        manifest = (load_red_entry_run if selected_row.get("task") == "red_entry" else load_signal_run)(project_id, run_id)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error(f"Saved run cannot be verified: {exc}")
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
        cancel_replay_forecast()
        for key in list(st.session_state):
            if str(key).startswith(("project_play:", "play_slider:")):
                st.session_state.pop(key, None)
        st.session_state.pop("project_last_forecast", None)
        st.session_state["result_active_pair"] = pair
    if manifest.get("task") == "red_entry":
        from pdm.red_entry_ui import render_event_evaluation, render_event_replay
        render_event_replay(project_id, run_id, unit_id, snapshot, theme)
        render_event_evaluation(manifest)
        return
    if (manifest.get("engine_id") or manifest.get("engine")) == "full_cns":
        provenance = manifest["connectome"]
        with st.expander("Connectome source"):
            st.markdown("[Original MaleCNS data · HHMI Janelia / Cambridge / MRC LMB / Google Research](https://male-cns.janelia.org/download/)")
            st.caption(f"Graph SHA-256: {provenance['graph_hash']}")
    corridor = manifest.get("params", {}).get("forecast_mode") == "bounded_trend_corridor"
    learned = manifest.get("params", {}).get("forecast_mode") == "learned_joint_trajectories"
    if not corridor:
        st.warning("This saved model uses the previous wide-band objective. Train a new Trend corridor model "
                   "to obtain the ±10% to ±15% corridor; its saved predictions are preserved.")
    replay_limits = (manifest["schema"].get("thresholds") if learned else
                     load_zone_limits(project_id, snapshot["snapshot_id"]))
    _play_fragment(project_id, run_id, unit_id, snapshot, manifest.get("interval_status"), theme,
                   replay_limits, background=manifest.get("engine_id") == "full_cns",
                   learned_horizons_s=manifest["params"]["horizons_s"] if learned or corridor else None)
    metrics = (manifest.get("metrics") or {}).get("test") or {}
    metric_partition = "Test"
    if (learned or corridor) and not metrics:
        metrics = (manifest.get("metrics") or {}).get("validation") or {}
        metric_partition = "Validation"
    if metrics:
        st.subheader("Validation summary" if metric_partition == "Validation" else "Test summary · exploratory" if corridor else
                     "Test summary" if manifest.get("funnel") else "Held-out Test summary")
        if corridor:
            c1, c2 = st.columns(2)
            coverage, width = metrics.get("point_coverage"), metrics.get("mean_relative_width")
            c1.metric("Measured containment", "—" if coverage is None else f"{coverage:.1%}")
            c2.metric("Mean corridor half-width", "—" if width is None else f"±{width/2:.1%}")
            st.table(pd.DataFrame([{"Horizon (s)": row["horizon_s"],
                         "Whole-path containment": "Unknown" if row["whole_path_coverage"] is None else f"{row['whole_path_coverage']:.1%}",
                         "Equipment with complete follow-up": row["complete_physical_groups"],
                         "Complete paths": row["complete_origins"]} for row in metrics.get("horizons", [])]))
            st.caption("Whole-path containment uses complete recorded follow-up only. "
                       "Historical splits have been reused; prediction quality remains exploratory.")
            if coverage is not None and coverage < manifest["params"]["nominal_coverage"]:
                st.warning("This model misses the containment target. Its bounds remain narrow; "
                           "the run has not demonstrated reliable corridor prediction.")
            return
        if learned:
            c1,c2=st.columns(2)
            c1.metric(f"{metric_partition} equipment",metrics.get("physical_group_count",0))
            c2.metric("Replay origins",metrics.get("origins",0))
            st.table(pd.DataFrame([{"Horizon (min)":row["horizon_s"]/60,
                          "Whole-path coverage":row["whole_path_coverage"],
                          "Mean band width (g)":row["mean_width_g"],
                          "Complete equipment":row["complete_physical_groups"],
                          "Useful RED warnings":row["events_with_useful_warning"]}
                          for row in metrics.get("horizons",[])]))
            return
        c1, c2 = st.columns(2)
        c1.metric("Known future measurements", metrics.get("known_targets", "—"))
        error = metrics.get("mae")
        c2.metric(f"Mean absolute error ({snapshot['schema'].get('signal_unit', '')})",
                  "—" if error is None else f"{float(error):.3g}")
        funnel_metrics = metrics.get("funnel") or {}
        if funnel_metrics:
            c1, c2 = st.columns(2)
            coverage = funnel_metrics.get("whole_path_coverage")
            c1.metric("Whole-path coverage · empirical", "N/A" if coverage is None else f"{float(coverage):.1%}")
            c2.metric("Complete-path Test equipment", funnel_metrics.get("complete_physical_group_count", 0))
        horizons = metrics.get("by_horizon") or []
        if horizons:
            rows = [{"Horizon (s)": row.get("horizon_s"),
                     "Known measurements": row.get("known_targets"),
                     "Test units": row.get("independent_units"),
                     f"Model MAE ({snapshot['schema'].get('signal_unit', '')})": row.get("mae"),
                     f"Last-value MAE ({snapshot['schema'].get('signal_unit', '')})": row.get("persistence_mae")}
                    for row in horizons]
            st.table(pd.DataFrame(rows), hide_index=True, border="horizontal")
