"""Single-engine signal training controls and project-specific job status."""
from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from pdm import ui_copy
from pdm.cli import spawn_worker
from pdm.models.signal_full_cns import ENGINE_LABEL
from pdm.signal_training import (
    _segments,
    available_signal_engines,
    average_training_duration_s,
    list_project_runs,
)
from pdm.ui_theme import empty_state, page_header
from pdm.worker import read_status, request_stop, status_for_project, worker_alive

ENGINE_LABELS = {"gru": "GRU", "lstm": "LSTM", "quantile_boosting": "Quantile boosting", "full_cns": ENGINE_LABEL}


def _training_step(snapshot: dict) -> float:
    frame = snapshot["features"]
    train = {str(uid) for uid in snapshot["split"].get("train") or []}
    samples = []
    for _, group in frame.loc[frame["unit_id"].astype(str).isin(train)].groupby("unit_id"):
        times = pd.to_numeric(group["timestamp_s"], errors="coerce").dropna().sort_values().to_numpy(float)
        intervals = np.diff(times)
        samples.extend(intervals[np.isfinite(intervals) & (intervals > 0)].tolist())
    step = float(np.median(samples)) if samples else 1.0
    return step


def _time_for_training(seconds: float, schema: dict) -> str:
    return f"{seconds / 60:.1f} min" if schema.get("source_kind") == "xjtu_bearings" else f"{seconds:.1f} s"


def _default_profile(snapshot: dict, engine_id: str) -> tuple[dict, list[dict]]:
    """Choose editable starting settings using Train/Validation clocks only."""
    profiles = {
        "gru": {"history_length": 24, "hidden_size": 64, "batch_size": 64,
                "epochs": 80, "learning_rate": 0.0007, "residual_forecast": True},
        "lstm": {"history_length": 32, "hidden_size": 64, "batch_size": 64,
                 "epochs": 80, "learning_rate": 0.0007, "residual_forecast": True},
        "quantile_boosting": {"history_length": 24, "max_iter": 120},
        "full_cns": {"history_length": 20},
    }
    profile = dict(profiles[engine_id])
    source_kind = snapshot["schema"].get("source_kind")
    if source_kind == "xjtu_bearings":
        multipliers = [1, 2, 3, 5, 10, 15, 20, 30, 45, 60, 75, 90, 120, 150, 180]
    elif source_kind == "hse_filters":
        multipliers = [1, 2, 3, 5, 10, 20, 30, 60, 100, 150, 200, 300]
    else:
        multipliers = [1, 2, 3, 5, 10, 15, 20, 30]
    step = _training_step(snapshot)
    features, split = snapshot["features"], snapshot["split"]
    segments = {part: [(str(uid), segment.timestamp_s.to_numpy(float))
                       for uid in split[part] for segment in _segments(features, str(uid))]
                for part in ("train", "validation")}
    mean_duration = average_training_duration_s(snapshot)
    terminal_steps = max(1, math.ceil(mean_duration / step - 1e-9))
    if terminal_steps > multipliers[-1]:
        extensions = np.geomspace(multipliers[-1], terminal_steps, num=7)[1:]
        multipliers = sorted(set(multipliers + [max(1, round(float(x))) for x in extensions]
                                 + [terminal_steps]))
    longest = [max((len(ts) for unit, ts in segments[part] if unit == str(uid)), default=0)
               for part in segments for uid in split[part]]
    if longest:
        profile["history_length"] = min(profile["history_length"], max(2, min(longest) - 1))
    history = profile["history_length"]
    tolerance = max(0.000001, min(step * 0.01, 0.01))
    coverage = []
    for multiplier in multipliers:
        horizon = step * multiplier
        row = {"horizon_s": horizon}
        for part in ("train", "validation"):
            units = set()
            targets = 0
            for uid, ts in segments[part]:
                origins = ts[history - 1:]
                if not len(origins):
                    continue
                desired = origins + horizon
                left = np.searchsorted(ts, desired - tolerance, side="left")
                right = np.searchsorted(ts, desired + tolerance, side="right")
                eligible = (right - left == 1) & (left < len(ts))
                candidates = np.flatnonzero(eligible)
                count = int(np.count_nonzero(ts[left[candidates]] > origins[candidates]))
                if count:
                    units.add(uid)
                    targets += count
            row[f"{part}_units"] = len(units)
            row[f"{part}_targets"] = targets
        if row["train_units"] and row["validation_units"]:
            coverage.append(row)
    profile["horizons_s"] = [row["horizon_s"] for row in coverage] or [step]
    return profile, coverage


def _parse_horizons(raw: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("Enter positive horizon times in seconds, separated by commas.") from exc
    if not values or any(not math.isfinite(value) or value <= 0 for value in values) or values != sorted(set(values)):
        raise ValueError("Horizons must be unique positive seconds in increasing order.")
    return values


@st.fragment(run_every=1.5)
def _job_status(project_id: str) -> None:
    status = status_for_project(project_id)
    if status.get("kind") != "project_train":
        return
    state = str(status.get("status") or "not_ready")
    if state == "not_ready":
        return
    label = status.get("message") or "Working on the selected model"
    if state in {"queued", "running", "training", "preparing", "stopping"}:
        st.info(str(label))
        value = status.get("progress")
        if isinstance(value, (float, int)) and 0 <= value <= 1:
            st.progress(float(value))
        if st.button("Stop job", key=f"stop_job:{project_id}", disabled=state == "stopping",
                     help=ui_copy.TRAIN_STOP_HELP):
            try:
                request_stop(expected_job_id=status["job_id"])
                st.info("Stop requested. The job will finish its current safe step.")
            except (KeyError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
    elif state == "completed":
        st.success("Job completed. Its saved model is ready to open.")
        if st.session_state.get(f"last_terminal:{project_id}") != status.get("job_id"):
            st.session_state[f"last_terminal:{project_id}"] = status.get("job_id")
            st.rerun(scope="app")
    elif state in {"failed", "cancelled", "stopped"}:
        st.error(str(status.get("error") or status.get("message") or f"Job {state}."))


def render_training(project_id: str, snapshot: dict) -> None:
    page_header("Training", "Choose one signal model and its settings. Training uses Train units, Validation selects the saved model, and Test remains held out until evaluation.")
    _job_status(project_id)
    runs = list_project_runs(project_id)
    stale = sum(1 for row in runs if row.get("snapshot_id") != snapshot["snapshot_id"])
    if stale:
        st.caption(ui_copy.TRAIN_STALE_RUNS_CAPTION.format(n=stale))
    capabilities = available_signal_engines(project_id, snapshot["snapshot_id"])
    for entry in capabilities:
        if entry["engine_id"] == "full_cns" and not entry.get("available"):
            st.info(f"{ENGINE_LABEL}: {entry['reason']}")
    available = [entry for entry in capabilities if entry.get("available")]
    if not available:
        with empty_state("No eligible model", "No signal engine is eligible for this snapshot. Check the Data Quality findings and continuous history length."):
            for entry in capabilities:
                if entry.get("reason"):
                    st.caption(f"{entry.get('label', entry.get('engine_id'))}: {entry['reason']}")
        return
    ids = [str(entry["engine_id"]) for entry in available]
    selected = st.selectbox("Model", ids, format_func=lambda key: ENGINE_LABELS.get(key, key),
                            help=ui_copy.TRAIN_MODEL_HELP)
    st.caption(ui_copy.TRAIN_MODEL_CAPTION)
    st.caption("Settings below are data-aware starting points, not proven optimal parameters. "
               "Use Validation and the last-value baseline to judge quality; keep Test out of tuning.")
    recurrent = selected in {"gru", "lstm"}
    full_cns = selected == "full_cns"
    defaults, coverage = _default_profile(snapshot, selected)
    profile_key = f"{selected}:{snapshot['snapshot_id']}"
    mean_duration = average_training_duration_s(snapshot)
    step = _training_step(snapshot)
    required_horizon = max(1, math.ceil(mean_duration / step - 1e-9)) * step
    st.caption(f"Average observed Train history per unit: {_time_for_training(mean_duration, snapshot['schema'])}. "
               f"The proposed direct forecast reaches {_time_for_training(max(defaults['horizons_s']), snapshot['schema'])}.")
    if mean_duration > 0:
        st.caption(f"Proposed useful RED-warning lead: {_time_for_training(mean_duration / 3, snapshot['schema'])} "
                   "(one third of mean Train history). This is an evaluation target, not a trained alert guarantee.")
    if max(defaults["horizons_s"]) + 1e-6 < required_horizon:
        st.warning("This snapshot has no known Train and Validation targets at the average Train history length. "
                   "A validated direct forecast that far is not possible until longer continuous histories "
                   "are available in both splits.")
    if full_cns:
        st.info(ui_copy.TRAIN_FULL_CNS_CAPTION)
        st.markdown("[MaleCNS v1.0 · public connection map and annotations](https://male-cns.janelia.org/download/)")
    with st.form("signal_train_form"):
        with st.container(border=True, key="pdm-train-window"):
            st.subheader("Data window", anchor=False)
            c1, c2 = st.columns(2)
            history = c1.number_input("History samples", min_value=2, max_value=256,
                                      value=defaults["history_length"], step=1, key=f"history:{profile_key}",
                                      help=ui_copy.TRAIN_HISTORY_HELP)
            horizons = c2.text_input("Forecast horizons (seconds)",
                                     value=", ".join(f"{value:g}" for value in defaults["horizons_s"]),
                                     key=f"horizons:{profile_key}",
                                     help=ui_copy.TRAIN_HORIZONS_HELP)
        with st.container(border=True, key="pdm-train-size"):
            st.subheader("Model size", anchor=False)
            if recurrent:
                c1, c2 = st.columns(2)
                hidden = c1.number_input("Hidden units", min_value=4, max_value=512,
                                         value=defaults["hidden_size"], step=4, key=f"hidden:{profile_key}",
                                         help=ui_copy.TRAIN_HIDDEN_HELP)
                batch = c2.number_input("Batch size", min_value=1, max_value=1024,
                                        value=defaults["batch_size"], step=1, key=f"batch:{profile_key}",
                                        help=ui_copy.TRAIN_BATCH_HELP)
                residual = st.checkbox("Predict change from last measurement", value=True,
                                       key=f"residual:{profile_key}",
                                       help="The model learns a change at each horizon and adds it to the latest "
                                            "observed signal. This gives a stable last-value starting point.")
            elif full_cns:
                st.caption("All classified MaleCNS neurons and all connections between them participate in every state update. The graph size is fixed by the source data.")
            else:
                max_iter = st.number_input("Boosting iterations", min_value=10, max_value=1000,
                                           value=defaults["max_iter"], step=10,
                                           key=f"max_iter:{profile_key}", help=ui_copy.TRAIN_BOOSTING_ITER_HELP)
        with st.container(border=True, key="pdm-train-repeat"):
            st.subheader("Repeatability", anchor=False)
            c1, c2 = st.columns(2)
            seed = c1.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                                   key=f"seed:{profile_key}",
                                   help=ui_copy.TRAIN_SEED_HELP)
            if recurrent:
                epochs = c2.number_input("Training epochs", min_value=1, max_value=500,
                                         value=defaults["epochs"], step=1, key=f"epochs:{profile_key}",
                                         help=ui_copy.TRAIN_EPOCHS_HELP)
                learning_rate = st.number_input("Learning rate", min_value=0.00001, max_value=0.1,
                                                value=defaults["learning_rate"], step=0.0001,
                                                format="%.5f", key=f"learning_rate:{profile_key}")
            elif full_cns:
                st.caption("Only the signal readout is fitted. Validation selects ridge regularization; the original connectivity stays fixed.")
            else:
                st.caption(ui_copy.TRAIN_BOOSTING_NO_EPOCHS_CAPTION)
        params: dict = {"history_length": int(history), "seed": int(seed)}
        if recurrent:
            params["epochs"] = int(epochs)
            params["hidden_size"] = int(hidden)
            params["batch_size"] = int(batch)
            params["learning_rate"] = float(learning_rate)
            params["residual_forecast"] = bool(residual)
        elif not full_cns:
            params["max_iter"] = int(max_iter)
        global_busy = worker_alive() or read_status().get("status") in {"queued", "running", "training", "preparing", "stopping"}
        submitted = st.form_submit_button("Train model", type="primary", disabled=global_busy,
                                          help=ui_copy.TRAIN_SUBMIT_HELP)
    if global_busy:
        st.info("A job is already active. Training can start when it finishes.")
    if coverage:
        weak = [row for row in coverage if row["validation_units"] < 2]
        if weak:
            first = weak[0]["horizon_s"]
            st.warning(f"Some direct horizons starting at {first:g} s have targets from only one Validation unit. "
                       "These long-range accuracy estimates are exploratory.")
        with st.expander("Default horizon coverage · Train and Validation only"):
            st.dataframe(pd.DataFrame(coverage).rename(columns={
                "horizon_s": "Horizon (s)", "train_units": "Train units",
                "train_targets": "Train targets", "validation_units": "Validation units",
                "validation_targets": "Validation targets"}), hide_index=True)
    if submitted:
        try:
            params["horizons_s"] = _parse_horizons(horizons)
            status = status_for_project(project_id)
            if status.get("status") in {"queued", "running", "training", "preparing", "stopping"}:
                raise RuntimeError("This project already has a running job.")
            spawn_worker({"kind": "project_train", "job_id": uuid.uuid4().hex, "project_id": project_id,
                          "snapshot_id": snapshot["snapshot_id"], "engine_id": selected, "params": params})
            st.success("Training queued. Progress appears above.")
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Training could not start: {exc}")
    compatible = [row for row in runs if row.get("snapshot_id") == snapshot["snapshot_id"]]
    if compatible:
        st.success(f"{len(compatible)} saved signal run{'s' if len(compatible) != 1 else ''} for this data snapshot.")
        latest = compatible[0]
        metrics = latest.get("metrics") or {}
        validation = metrics.get("validation") or {}
        test = metrics.get("test") or {}
        if validation or test:
            st.caption(f"Latest saved model: {ENGINE_LABELS.get(latest.get('engine_id'), 'Signal model')}. Mean absolute error is in {snapshot['schema'].get('signal_unit', 'signal units')}.")
            c1, c2 = st.columns(2)
            val_mae = validation.get("mae")
            test_mae = test.get("mae")
            c1.metric("Validation mean absolute error", "—" if val_mae is None else f"{float(val_mae):.3g}",
                      help=ui_copy.TRAIN_VALIDATION_MAE_HELP)
            c2.metric("Held-out Test mean absolute error", "—" if test_mae is None else f"{float(test_mae):.3g}",
                      help=ui_copy.TRAIN_TEST_MAE_HELP)
            st.caption(f"Known future measurements: Validation {validation.get('known_targets', 0)} · Test {test.get('known_targets', 0)}.")
            by_horizon = validation.get("by_horizon") or []
            if by_horizon:
                st.caption("Validation by horizon. The last-value baseline repeats the most recent measurement; "
                           "lower MAE is better. Long horizons with one unit are exploratory.")
                st.dataframe(pd.DataFrame([{
                    "Horizon (s)": row.get("horizon_s"),
                    "Validation units": row.get("independent_units"),
                    "Known targets": row.get("known_targets"),
                    "Model MAE": row.get("mae"),
                    "Last-value MAE": row.get("persistence_mae"),
                } for row in by_horizon]), hide_index=True)
                beaten = [row for row in by_horizon if row.get("mae") is not None
                          and row.get("persistence_mae") is not None
                          and row["mae"] >= row["persistence_mae"]]
                if beaten:
                    st.warning(f"On {len(beaten)} of {len(by_horizon)} Validation horizons, the saved model "
                               "does not beat repeating the last measurement. Do not treat those forecasts as reliable warnings.")
