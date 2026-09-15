"""One server clock drives inference, neural activity and equipment charts."""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from pdm.replay import split_label_for_unit
from pdm.visualization.component import neural_activity_explorer
from pdm.visualization.explorer import (
    WORKER_BUSY_MESSAGE,
    _soma_cache_key,
    build_ui_explorer_payload,
    load_scene_from_run,
    soma_join_allowed,
)
from pdm.visualization.overlay import build_work_overlay_figure
from pdm.visualization.presentation import render_hero, render_metrics, render_panel_header
from pdm.visualization.simulation import neuron_details, simulate_step
from pdm.worker import worker_alive


@st.cache_resource(show_spinner=False, max_entries=2)
def _simulation_model(path: str, checkpoint_stamp: int):
    from pdm.train import load_trained_model

    return load_trained_model(Path(path), device="cpu", which="best")


@st.cache_data(show_spinner=False, max_entries=8)
def _simulation_scene(path, checkpoint_stamp, soma_stamp, context=True):
    if (Path(path) / "connectome/recurrent.npz").is_file():
        from pdm.visualization.full_scene import full_cns_scene

        return full_cns_scene(path)
    scene = load_scene_from_run(Path(path))
    return build_ui_explorer_payload(
        nodes=scene["nodes"], edges=scene["edges"], positions=scene["positions"],
        states=np.zeros((0, len(scene["nodes"]))), frame_map=[],
        flags={"graph_mode": scene["graph_mode"], "is_synthetic": scene["is_synthetic"]},
        run_dir=Path(path), join_soma=soma_join_allowed(scene, scene),
        context_cap=40_000,
    )


@st.cache_data(show_spinner=False)
def _interval_profile(path, checkpoint_stamp, profile_stamp):
    from pdm.forecasting import load_interval_profile

    return load_interval_profile(Path(path))


def _validate_simulation_profile(profile, history_length):
    """The calibrated clock must describe the loaded model's state and warmup."""
    if profile.get("state_mode") != "continuous":
        raise ValueError("Forecast interval profile does not use continuous neuron state")
    if int(profile.get("warmup_measurements", 0)) != int(history_length):
        raise ValueError("Forecast interval warmup does not match the saved model")


def _pause(key):
    st.session_state[key]["playing"] = False


def render_equipment_simulation(dataset_id, rdir, uid, bundle):
    key = f"equipment_sim:{rdir}:{uid}"
    scheduled = bool(st.session_state.get(key, {}).get("playing"))
    # No timer during initial model loading or while paused. Otherwise timer
    # reruns can interrupt a slow first render before the chart is complete.
    fragment = st.fragment(run_every=1.0 if scheduled else None)(_render_equipment_simulation)
    fragment(dataset_id, rdir, uid, bundle, scheduled)


def _render_equipment_simulation(dataset_id, rdir, uid, bundle, scheduled):
    """Each timer tick calls the real model once; no autonomous browser playback."""
    key = f"equipment_sim:{rdir}:{uid}"
    cursor_key = key + ":cursor"
    if key not in st.session_state:
        st.session_state[key] = {"playing": False, "predictions": {}, "trace": None, "at": None}
        st.session_state[cursor_key] = 0
    session = st.session_state[key]
    feat = bundle["features"]
    feat = feat[feat["unit_id"].astype(str) == str(uid)].sort_values("timestamp_s").reset_index(drop=True)
    if feat.empty:
        session["playing"] = False
        if scheduled:
            st.rerun()
        st.info("No measurements for this unit.")
        return
    # Data may be replaced while this browser keeps a previous slider position.
    # Reset before constructing the widget or indexing the new table.
    version = bundle.get("dataset_version")
    if session.get("dataset_version") != version:
        session.update(playing=False, predictions={}, forecast_rows={}, trace=None, at=None, dataset_version=version)
        st.session_state[cursor_key] = 0
    cursor = int(st.session_state.get(cursor_key, 0))
    if cursor < 0 or cursor >= len(feat):
        st.session_state[cursor_key] = max(0, min(cursor, len(feat) - 1))
        session.update(playing=False, predictions={}, forecast_rows={}, trace=None, at=None)
    render_hero(dataset_id, uid, split_label_for_unit(bundle.get("split"), uid))
    with st.container(key="lab_controls"):
        with st.container(key="lab_transport"):
            cols = st.columns(4)
            start = cols[0].button("Start test run", key=key + ":start", type="primary", width="stretch")
            pause = cols[1].button("Pause test run", key=key + ":pause", width="stretch")
            step = cols[2].button("Next measurement", key=key + ":next", width="stretch")
            reset = cols[3].button("Reset test run", key=key + ":reset", width="stretch")
        busy = worker_alive()
        if busy:
            session["playing"] = False
            if scheduled:
                st.rerun()
            st.info(WORKER_BUSY_MESSAGE)
            return
        if reset:
            session.update(playing=False, predictions={}, forecast_rows={}, trace=None, at=None)
            st.session_state[cursor_key] = 0
        elif pause:
            session["playing"] = False
        elif start:
            session["playing"] = True
        elif step or session["playing"]:
            st.session_state[cursor_key] = min(int(st.session_state[cursor_key]) + 1, len(feat) - 1)
        timeline, evaluator = st.columns([4, 1])
        with timeline:
            if len(feat) > 1:
                index = st.slider(
                    "Measurement", 0, len(feat) - 1, key=cursor_key,
                    on_change=_pause, args=(key,), label_visibility="collapsed",
                )
            else:
                index = 0
        if index == len(feat) - 1:
            session["playing"] = False
        if session["playing"] != scheduled:
            st.rerun()
        now = float(feat.iloc[index]["timestamp_s"])
        with evaluator:
            show_gt = st.checkbox("Show ground truth", value=True, key=key + ":gt", on_change=_pause, args=(key,))
    checkpoint = Path(rdir) / "best.pt"
    if not checkpoint.is_file():
        checkpoint = Path(rdir) / "last.pt"
    try:
        stamp = checkpoint.stat().st_mtime_ns
        model, prep, meta = _simulation_model(str(rdir), stamp)
        hist_len = int(meta["history_length"])
        continuous = getattr(model, "state_mode", None) == "continuous"
        recurrent = hasattr(model, "encoder")
        if recurrent:
            payload, error = {"flags": {}, "positions": {}}, None
        else:
            payload, error = _simulation_scene(str(rdir), stamp, _soma_cache_key(None))
            if error:
                st.caption(error)
            if not model.is_synthetic and not payload["flags"].get("full_cns") and int(payload["flags"].get("n_with_soma") or 0) != model.n_nodes:
                st.caption("This model cannot show every computing neuron in anatomical coordinates. A schematic network layout is used where coordinates are unavailable.")
        profile_path = Path(rdir) / "interval_profile.json"
        profile = None
        if continuous and profile_path.is_file():
            try:
                profile = _interval_profile(str(rdir), stamp, profile_path.stat().st_mtime_ns)
                _validate_simulation_profile(profile, hist_len)
            except ValueError as exc:
                profile = None
                st.warning(f"Interval unavailable: {exc}. Showing the raw point forecast.")
        # checkpoint also discards cached predictions and states.
        version = bundle.get("dataset_version")
        if session.get("checkpoint") != stamp or session.get("dataset_version") != version:
            session.update(predictions={}, forecast_rows={}, trace=None, at=None, checkpoint=stamp, dataset_version=version)
        if session["at"] != index:
            session["predictions"] = {t: p for t, p in session["predictions"].items() if t <= now}
            session["trace"] = simulate_step(feat, uid, now, model, prep, hist_len, previous_trace=session["trace"])
            session["at"] = index
        trace = session["trace"]
        pred = trace["predicted_rul_s"]
        if pred is not None:
            session["predictions"][now] = pred
        if profile is not None:
            from pdm.forecasting import predict_failure_interval

            points = predict_failure_interval(trace["timestamps_s"], trace["raw_rul_s"], profile)
            latest = points.iloc[-1]
            pred = float(latest["predicted_rul_s"]) if pd.notna(latest["predicted_rul_s"]) else None
            interval = (float(latest["lower_rul_s"]), float(latest["upper_rul_s"])) if pred is not None else None
        elif not continuous:
            from pdm.visualization.simulation import window_forecast_history

            points = window_forecast_history(feat.iloc[:index + 1], model, prep, hist_len,
                                             cached=session.get("forecast_rows"))
            session["forecast_rows"] = {float(r["timestamp_s"]): r for r in points.to_dict("records")}
            latest = points.iloc[-1]
            pred = float(latest.predicted_rul_s) if pd.notna(latest.predicted_rul_s) else None
            interval = ((float(latest.lower_rul_s), float(latest.upper_rul_s))
                        if pred is not None and pd.notna(latest.get("lower_rul_s")) else None)
        else:
            points = pd.DataFrame([
                {"timestamp_s": t, "predicted_rul_s": p} for t, p in sorted(session["predictions"].items())
            ], columns=["timestamp_s", "predicted_rul_s"])
            interval = ((float(trace["lower_rul_s"]), float(trace["upper_rul_s"]))
                        if pred is not None and "lower_rul_s" in trace else None)
    except Exception as exc:  # noqa: BLE001
        session["playing"] = False
        if scheduled:
            st.rerun()
        st.error(f"Test run could not advance: {exc}")
        return

    predicted = trace["status"] == "predicted"
    activity = "State"
    values = np.zeros((0, model.n_nodes), dtype=np.float32)
    has_activity = len(trace["states"]) > 0
    if has_activity:
        values = trace["states"][-1:]
    payload.update(
        nodes=list(model.node_order), states=values.tolist(),
        inputs=trace["inputs"][-1:].tolist() if has_activity else [],
        predicted_rul_s=pred,
        frame_map=[{"timestamp_s": now, "frame_index": index}] if has_activity else [],
    )
    if payload["flags"].get("full_cns") and has_activity:
        payload["states_b64"] = base64.b64encode(np.asarray(values[-1], dtype="<f4").tobytes()).decode()
        payload["states"] = []
    payload["flags"].update(
        mode="Equipment replay", phase="simulation", synchronized=True,
        signal_label=activity, now_timestamp_s=now, history_length=hist_len,
        activity_scale=1.0 if activity == "State" else None,
        failure_window_s=[now + v for v in interval] if interval is not None else None,
        time_scale=60 if dataset_id == "bearings" else 1, time_unit="min" if dataset_id == "bearings" else "s",
        continuous_history=continuous,
    )
    visible = len(payload["positions"])
    unit = bundle["units"].loc[bundle["units"]["unit_id"].astype(str) == str(uid)].iloc[0]
    observed = bool(unit.get("event_observed", False))
    event = float(unit["event_time_s"]) if observed and pd.notna(unit.get("event_time_s")) else None
    actual = np.maximum(event - points["timestamp_s"].to_numpy(), 0) if event is not None else None
    scale = 60.0 if dataset_id == "bearings" else 1.0
    unit_label = "min" if dataset_id == "bearings" else "s, internal"
    actual_now = max(event - now, 0) if event is not None else None
    event_reference_label = "Recorded end"
    if dataset_id == "filters" and pd.notna(unit.get("official_rul_at_prefix_end_s")):
        event_reference_label = "Official RUL end"
        event = float(unit["observation_end_s"]) + float(unit["official_rul_at_prefix_end_s"])
        actual = np.maximum(event - points["timestamp_s"].to_numpy(), 0)
        actual_now = max(event - now, 0)
    render_metrics(
        now=now, interval=interval, history=trace["n_history"] if continuous else min(trace["n_history"], hist_len), node_count=model.n_nodes,
        playing=session["playing"], unit_label=unit_label, scale=scale, history_length=hist_len,
        progress=100 * index / max(len(feat) - 1, 1), predicted_rul_s=pred,
        continuous=continuous, interval_method=("Empirical calibrated interval" if profile else trace.get("interval_method")),
    )
    if not predicted:
        st.caption(f"{trace['status']} · {trace['n_history']} / {hist_len} measurements before the first forecast")
    elif interval is None:
        st.caption(f"Point forecast: {pred/scale:.2f} {unit_label} remaining · no calibrated interval available.")
    else:
        st.caption("Empirical forecast interval · calibration coverage is not an independent test" if profile else trace.get("interval_method", ""))
    left, right = st.columns([1, 1.35], gap="medium")
    with left, st.container(key="lab_brain_panel"):
        if recurrent:
            from pdm.visualization.recurrent_trace import render_recurrent_trace

            render_panel_header("01", model.architecture.upper() + " · recurrent network", "Actual model activity", "SYNCHRONIZED")
            render_recurrent_trace(trace, model, prep)
        else:
            full_cns = bool(payload["flags"].get("full_cns"))
            random_control = model.architecture == "random_reservoir"
            render_panel_header("01", "MaleCNS · whole connectome" if full_cns else ("Random reservoir" if random_control else "The computing brain"),
                                "Engineered control" if random_control else "MaleCNS v1.0", f"{model.n_nodes:,} computing neurons")
            if random_control:
                payload["context_positions"] = {}
                payload["hull_polyline"] = []
                st.caption("Rewired computational connections and actual node states. Source neuron positions are inherited from the matched fly model; the connections are not biological.")
            if full_cns:
                st.caption(f"{model.provenance['n_edges']:,} directed connections · {model.provenance['n_synapses']:,} synapses")
                color, arbors = st.columns([1.3, 1])
                with color:
                    payload["flags"]["color_mode"] = st.radio("Color", ["Activity", "Cell classes"],
                                                              horizontal=True, key=key + ":color", label_visibility="collapsed")
                with arbors:
                    payload["flags"]["show_morphology"] = st.checkbox("Neuron arbors", value=True, key=key + ":arbors")
                    payload["flags"]["show_connections"] = st.checkbox("Connection sample", value=False, key=key + ":edges")
                history = trace.get("activity_history")
                if history is not None:
                    payload["activity_history"] = history.tolist()
                    payload["activity_history_timestamps_s"] = trace["activity_history_timestamps_s"].tolist()
                    payload["activity_history_ids"] = [model.node_order[i] for i in trace["activity_history_ids"]]
            payload["flags"]["compact"] = True
            neural_activity_explorer(**payload, key=key + ":brain")
            if full_cns:
                morph = payload.get("morphology", {})
                st.caption(f"{visible:,} curated soma/to-soma positions · {model.n_nodes - visible:,} neurons compute without a curated point. "
                           f"Arbors: {morph.get('n_neurons', 0)} reconstructed neurons. "
                           "Connection sample: up to 12,000 real pairs, drawn as straight links; not axon paths.")
                st.caption("MaleCNS data: FlyEM / HHMI Janelia, Cambridge / MRC LMB, Google Research · CC-BY 4.0. "
                           "Computational states, not biological recordings or spikes.")
            else:
                st.markdown('<div class="lab-panel-foot">Color and intensity show the current state.</div>', unsafe_allow_html=True)
    with right, st.container(key="lab_forecast_panel"):
        render_panel_header("02", "Failure forecast", "Evidence → failure window", "SYNCHRONIZED")
        fig = build_work_overlay_figure(
            dataset_id=dataset_id, unit_features=feat, now_timestamp_s=now,
            predicted_rul_s=pred, history_length=trace["n_history"] if continuous else hist_len, show_gt=show_gt,
            event_time_s=event, event_observed=observed,
            predicted_rul_by_time=points, actual_rul_s=actual,
            event_reference_label=event_reference_label,
            prediction_interval_s=interval,
        )
        fig.update_layout(height=570, uirevision=key)
        st.plotly_chart(fig, width="stretch", theme=None, key=key + ":chart")
        truth = (
            f"Actual remaining life: {actual_now/scale:.1f} {unit_label} · {event_reference_label.lower()}: {event/scale:.1f} {unit_label}"
            if show_gt and actual_now is not None else "Ground truth hidden · forecast uses observed measurements only"
        )
        st.markdown(f'<div class="lab-panel-foot">{truth}</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="lab-explanation"><span><b>One measurement clock.</b> The model receives observed history; '
        'future recording and actual life are evaluation overlays only.</span>'
        '<span><b>Forecast range.</b> The interval type is shown above when available. '
        'Recorded endpoints and actual RUL are evaluation information.</span></div>', unsafe_allow_html=True,
    )
    if index == len(feat) - 1:
        st.caption("End of recorded measurements.")
    if recurrent or not has_activity:
        return
    with st.expander("Inspect a computing neuron", expanded=False):
        if hasattr(model, "pool_index"):
            node = st.text_input("MaleCNS body ID", value=model.node_order[0], key=key + ":neuron", on_change=_pause, args=(key,)).strip()
            if node not in model.node_order:
                st.info("Enter a classified MaleCNS body ID from this model.")
                return
        else:
            node = st.selectbox("Reservoir body ID", list(model.node_order), key=key + ":neuron", on_change=_pause, args=(key,))
        i = model.node_order.index(node)
        proc = trace["processing"]
        inp, recurrent, bias, prev = (
            float(proc["input_drive"][-1, i]), float(proc["recurrent_drive"][-1, i]),
            float(proc["bias"][i]), float(proc["previous_state"][-1, i]),
        )
        state = float(trace["states"][-1, i])
        leak = proc["leak"]
        st.code(
            f"sensor input = Σ W_in × u = {inp:.6f}\n"
            f"neighbor input = Σ W_res × previous state = {recurrent:.6f}\n"
            f"bias = {bias:.6f}; previous state = {prev:.6f}; leak = {leak:g}\n"
            f"new state = (1 − {leak:g}) × {prev:.6f} + {leak:g} × "
            f"tanh({inp:.6f} + {recurrent:.6f} + {bias:.6f}) = {state:.6f}"
        )
        st.caption(
            "Inputs are normalized bearing features. Sensor projection and trained readout are "
            "engineering additions; recurrent connections come from the selected graph. "
            + ("State carries forward through the observed history." if continuous else "Each prediction resets at the start of its history window.")
        )
        sensors, neighbors = neuron_details(trace, model, prep, i)
        a, b = st.columns(2)
        with a:
            st.caption("Sensor → neuron: every input product")
            st.dataframe(sensors, hide_index=True, width="stretch")
        with b:
            st.caption("Connected neurons → neuron: signed recurrent products")
            st.dataframe(neighbors, hide_index=True, width="stretch")
        c = trace["contributions"]
        if continuous and predicted:
            st.caption(f"Raw neural readout: {trace['predicted_rul_s']/scale:.2f} {unit_label} remaining. Forecast center after causal history filtering: {pred/scale:.2f} {unit_label}.")
        st.caption(
            f"Linear readout (before output transform): bias {float(c['intercept'][0]):.6f} "
            f"+ sensor terms {float(c['input'][-1, :, 0].sum()):.6f} "
            f"+ neuron terms {float(c['neuron'][-1, :, 0].sum()):.6f} "
            f"= {float(c['raw'][-1, 0]):.6f}. "
            f"Selected neuron contribution: {float(c['neuron'][-1, i, 0]):.6f}."
        )
