"""Single-engine signal training controls and project-specific job status."""
from __future__ import annotations

import math
import uuid

import numpy as np
import pandas as pd
import streamlit as st

from pdm import ui_copy
from pdm.cli import spawn_worker
from pdm.signal_training import available_signal_engines, list_project_runs
from pdm.ui_theme import empty_state, page_header
from pdm.worker import read_status, request_stop, status_for_project, worker_alive

ENGINE_LABELS = {"gru": "GRU", "lstm": "LSTM", "quantile_boosting": "Quantile boosting"}


def _default_horizons(snapshot: dict) -> str:
    frame = snapshot["features"]
    train = {str(uid) for uid in snapshot["split"].get("train") or []}
    samples = []
    for _, group in frame.loc[frame["unit_id"].astype(str).isin(train)].groupby("unit_id"):
        times = pd.to_numeric(group["timestamp_s"], errors="coerce").dropna().sort_values().to_numpy(float)
        intervals = np.diff(times)
        samples.extend(intervals[np.isfinite(intervals) & (intervals > 0)].tolist())
    step = float(np.median(samples)) if samples else 1.0
    return ", ".join(f"{step * multiplier:g}" for multiplier in (1, 2, 3))


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
    recurrent = selected in {"gru", "lstm"}
    with st.form("signal_train_form"):
        with st.container(border=True, key="pdm-train-window"):
            st.subheader("Data window", anchor=False)
            c1, c2 = st.columns(2)
            history = c1.number_input("History samples", min_value=2, max_value=256, value=8, step=1,
                                      help=ui_copy.TRAIN_HISTORY_HELP)
            horizons = c2.text_input("Forecast horizons (seconds)", value=_default_horizons(snapshot),
                                     help=ui_copy.TRAIN_HORIZONS_HELP)
        with st.container(border=True, key="pdm-train-size"):
            st.subheader("Model size", anchor=False)
            if recurrent:
                c1, c2 = st.columns(2)
                hidden = c1.number_input("Hidden units", min_value=4, max_value=512, value=32, step=4,
                                         help=ui_copy.TRAIN_HIDDEN_HELP)
                batch = c2.number_input("Batch size", min_value=1, max_value=1024, value=32, step=1,
                                        help=ui_copy.TRAIN_BATCH_HELP)
            else:
                max_iter = st.number_input("Boosting iterations", min_value=10, max_value=1000, value=40,
                                           step=10, help=ui_copy.TRAIN_BOOSTING_ITER_HELP)
        with st.container(border=True, key="pdm-train-repeat"):
            st.subheader("Repeatability", anchor=False)
            c1, c2 = st.columns(2)
            seed = c1.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=42, step=1,
                                   help=ui_copy.TRAIN_SEED_HELP)
            if recurrent:
                epochs = c2.number_input("Training epochs", min_value=1, max_value=500, value=12, step=1,
                                         help=ui_copy.TRAIN_EPOCHS_HELP)
            else:
                st.caption(ui_copy.TRAIN_BOOSTING_NO_EPOCHS_CAPTION)
        params: dict = {"history_length": int(history), "seed": int(seed)}
        if recurrent:
            params["epochs"] = int(epochs)
            params["hidden_size"] = int(hidden)
            params["batch_size"] = int(batch)
        else:
            params["max_iter"] = int(max_iter)
        global_busy = worker_alive() or read_status().get("status") in {"queued", "running", "training", "preparing", "stopping"}
        submitted = st.form_submit_button("Train model", type="primary", disabled=global_busy,
                                          help=ui_copy.TRAIN_SUBMIT_HELP)
    if global_busy:
        st.info("A job is already active. Training can start when it finishes.")
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
