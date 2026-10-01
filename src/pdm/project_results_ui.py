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
from pdm.signal_inference import forecast_prefix
from pdm.signal_training import average_training_duration_s, list_project_runs, load_signal_run
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
    if entry.get("status") != "available":
        return None
    try:
        start, end = float(entry["earliest_s"]), float(entry["latest_s"])
        now = float(result["as_of_s"])
    except (KeyError, TypeError, ValueError):
        return None
    return (start, end) if all(map(math.isfinite, (start, end, now))) and now < start <= end else None


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
        direct = anchor + [r for r in all_points if r.get("kind") != "recursive"]
        recursive = [r for r in all_points if r.get("kind") == "recursive"]
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
            fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in direct],
                                     y=[r.get("upper") for r in direct], mode="lines",
                                     line={"color": t["series_band_line"], "width": 1},
                                     name="Pointwise upper quantile", connectgaps=False))
            fig.add_trace(go.Scatter(x=[r["target_time_s"] for r in direct],
                                     y=[r.get("lower") for r in direct], mode="lines", fill="tonexty",
                                     fillcolor=t["series_band"],
                                     line={"color": t["series_band_line"], "width": 1},
                                     name="Pointwise lower quantile", connectgaps=False))
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
              + [float(row["signal"]) for row in future if row.get("signal") is not None])
    add_threshold_layers(fig, thresholds, values, theme)
    corridor = _red_entry_corridor(result)
    if corridor:
        fig.add_vrect(x0=corridor[0], x1=corridor[1], fillcolor=t["series_band"],
                      line_width=0, annotation_text="Predicted RED-entry window",
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
        st.caption(f"Forecast through {_time_label(furthest, schema)} · "
                   f"{_time_label(furthest - float(result['as_of_s']), schema)} ahead of Now.")
    else:
        st.info(str(result.get("reason") or "A forecast is unavailable at this point. Collect more continuous history."))
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
    else:
        st.caption("Red crossing time is unavailable for this prefix or threshold rule.")
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


@st.fragment(run_every=0.6)
def _play_fragment(project_id: str, run_id: str, unit_id: str, snapshot: dict,
                   interval_status: str | None = None, theme: str = "dark",
                   zone_limits: dict | None = None, *, background: bool = False) -> None:
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
    forecast_key = (project_id, run_id, snapshot["snapshot_id"], unit_id, as_of_s,
                    json.dumps(zone_limits, sort_keys=True))
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
            thresholds=dict(zone_limits) if zone_limits else None))
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
                result = forecast_prefix(project_id, run_id, unit_id, as_of_s, thresholds=zone_limits)
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
    st.plotly_chart(replay_figure(result, schema, theme, future_actual=future), width="stretch", theme=None)
    if show_future:
        st.caption("Gray after Now: recorded future measurements for comparison only; hidden from the model.")
        if future.empty:
            st.caption("No recorded measurements remain after Now.")
        else:
            actual_red = first_recorded_red_after_now(
                result["observed_prefix"], future.to_dict("records"), result.get("thresholds") or {}, as_of_s)
            if actual_red is not None:
                st.caption(f"First recorded RED sample: {_time_label(float(actual_red['timestamp_s']), schema)}. "
                           "The threshold crossing may have occurred between samples; this outcome is shown only for review.")
                corridor = _red_entry_corridor(result)
                if corridor:
                    inside = corridor[0] <= float(actual_red["timestamp_s"]) <= corridor[1]
                    st.caption("The first recorded RED sample is "
                               f"{'inside' if inside else 'outside'} the model-issued window.")
                lead_target = average_training_duration_s(snapshot) / 3 if snapshot.get("split") else 0
                first_sample_s = float(one["timestamp_s"].iloc[0])
                if lead_target > 0 and float(actual_red["timestamp_s"]) - first_sample_s < lead_target:
                    st.caption(f"A {_lead_label(lead_target, schema)} warning lead (one third of mean Train "
                               "history) is impossible for this unit: its first recorded RED sample "
                               "occurs sooner than that after the first measurement.")
    if not failed:
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
        cancel_replay_forecast()
        for key in list(st.session_state):
            if str(key).startswith(("project_play:", "play_slider:")):
                st.session_state.pop(key, None)
        st.session_state.pop("project_last_forecast", None)
        st.session_state["result_active_pair"] = pair
    engine = manifest.get("engine_id") or manifest.get("engine") or "Saved model"
    if engine == "full_cns":
        engine = "Fly brain · Full MaleCNS"
        provenance = manifest["connectome"]
        st.caption(f"{provenance['dataset']} · {provenance['n_nodes']:,} neurons · "
                   f"{provenance['n_edges']:,} directed connections · {provenance['n_synapses']:,} synaptic contacts.")
        with st.expander("Connectome source"):
            st.markdown("[Original MaleCNS data · HHMI Janelia / Cambridge / MRC LMB / Google Research](https://male-cns.janelia.org/download/)")
            st.caption(f"Graph SHA-256: {provenance['graph_hash']}")
    trained = manifest["params"]["horizons_s"]
    st.caption(f"{engine} · held-out Test unit · direct model forecast through "
               f"{_time_label(max(trained), snapshot['schema'])} ahead · "
               f"signal in {snapshot['schema'].get('signal_unit', 'native units')}")
    mean_train_duration = average_training_duration_s(snapshot)
    if mean_train_duration > 0:
        st.caption(f"Proposed RED-warning lead: {_lead_label(mean_train_duration / 3, snapshot['schema'])} "
                   "(one third of mean Train history).")
    if max(trained) + 1e-6 < mean_train_duration:
        average_label = (f"{mean_train_duration / 60:.1f} min"
                         if snapshot["schema"].get("source_kind") == "xjtu_bearings"
                         else f"{mean_train_duration:.1f} s")
        st.info(f"This run stops at {_time_label(max(trained), snapshot['schema'])} ahead, shorter than the "
                f"average Train history of {average_label}. "
                "Retrain with the longer direct horizons to extend this model's forecast.")
    validation_rows = ((manifest.get("metrics") or {}).get("validation") or {}).get("by_horizon") or []
    weaker = [row for row in validation_rows if row.get("mae") is not None
              and row.get("persistence_mae") is not None and row["mae"] >= row["persistence_mae"]]
    if weaker:
        st.warning(f"This model did not beat the last-value baseline on {len(weaker)} of "
                   f"{len(validation_rows)} Validation horizons. Its red-entry forecast is not reliable "
                   "without better independent validation.")
    _play_fragment(project_id, run_id, unit_id, snapshot, manifest.get("interval_status"), theme,
                   load_zone_limits(project_id, snapshot["snapshot_id"]),
                   background=manifest.get("engine_id") == "full_cns")
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
                     f"Model MAE ({snapshot['schema'].get('signal_unit', '')})": row.get("mae"),
                     f"Last-value MAE ({snapshot['schema'].get('signal_unit', '')})": row.get("persistence_mae")}
                    for row in horizons]
            st.table(pd.DataFrame(rows), hide_index=True, border="horizontal")
