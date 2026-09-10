from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm.cli import spawn_worker
from pdm.config import load_dataset_config, model_defaults
from pdm.data.prepare import load_processed, processed_ready
from pdm.device import resolve_device
from pdm.evaluate import default_horizon_s
from pdm.experiments import list_runs, run_dir
from pdm.paths import dataset_raw, project_root
from pdm.worker import read_status, request_stop, worker_alive

st.set_page_config(page_title="Predictive Maintenance Lab", layout="wide")
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

LABELS = {"bearings": "Bearings", "filters": "Filters"}


def _dataset() -> str:
    choice = st.sidebar.radio("Dataset", ["Bearings", "Filters"], horizontal=True)
    return "bearings" if choice == "Bearings" else "filters"


def _device_box() -> None:
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


def _status_chip() -> dict:
    st_ = read_status()
    alive = worker_alive()
    st.sidebar.write("Worker:", "running" if alive else "idle")
    st.sidebar.json({k: st_.get(k) for k in ("status", "dataset_id", "run_id", "epoch", "message", "error") if k in st_ or st_.get(k)})
    return st_


def main() -> None:
    st.title("Predictive Maintenance Lab")
    st.info("Historical replay — not a live equipment connection")
    dataset_id = _dataset()
    _device_box()
    page = st.sidebar.radio("Screen", ["Data", "Train", "Test & Replay"], index=0)
    _status_chip()
    if page == "Data":
        screen_data(dataset_id)
    elif page == "Train":
        screen_train(dataset_id)
    else:
        screen_replay(dataset_id)


def screen_data(dataset_id: str) -> None:
    st.header(f"Data — {LABELS[dataset_id]}")
    state = _data_state(dataset_id)
    st.write("Data state:", state)
    raw = dataset_raw(dataset_id)
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
    report = bundle["report"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train units", split.get("n_train"))
    c2.metric("Validation units", split.get("n_validation"))
    c3.metric("Test units", split.get("n_test"))
    c4.metric("Measurements", int(len(features)))
    st.metric("Observed events", int(units["event_observed"].sum()) if "event_observed" in units.columns else "—")
    st.subheader("Units")
    show = units.copy()
    show["split"] = show["unit_id"].map(_split_map(split))
    st.dataframe(show, width="stretch", hide_index=True)
    uid = st.selectbox("Unit for sensor plot", show["unit_id"].tolist())
    g = features[features["unit_id"] == uid].sort_values("timestamp_s")
    fig = go.Figure()
    if dataset_id == "bearings" and "horizontal_rms" in g.columns:
        fig.add_trace(go.Scatter(x=g["timestamp_s"] / 60.0, y=g["horizontal_rms"], name="horizontal RMS"))
        fig.update_xaxes(title_text="Operating time (min)")
        fig.update_yaxes(title_text="RMS (g, original units)")
    else:
        fig.add_trace(go.Scatter(x=g["timestamp_s"] / 60.0, y=g["differential_pressure"], name="Δp"))
        fig.add_hline(y=600.0, line_dash="dash", annotation_text="600 Pa")
        fig.update_xaxes(title_text="Operating time (min)")
        fig.update_yaxes(title_text="Differential pressure (Pa)")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Target and quality")
    st.write(report.get("sensor_time_note", ""))
    phys = report.get("history_length_physical") or {}
    st.caption(phys.get("note") or "")
    if report.get("issues_and_decisions"):
        for issue in report["issues_and_decisions"]:
            st.warning(issue)
    with st.expander("data_report.json"):
        st.json(report)


def screen_train(dataset_id: str) -> None:
    st.header(f"Train — {LABELS[dataset_id]}")
    if not processed_ready(dataset_id):
        st.warning("Prepare data on the Data screen first. There is no dummy training path.")
        return
    cfg = load_dataset_config(dataset_id)
    mcfg = model_defaults(cfg)
    bundle = load_processed(dataset_id)
    phys = (bundle.get("report") or {}).get("history_length_physical") or {}
    arch = st.selectbox("Architecture", ["gru", "lstm"], index=0)
    epochs = st.number_input("Epochs", min_value=1, max_value=200, value=int(mcfg["max_epochs"]))
    hist = st.number_input("History length (measurements)", min_value=2, max_value=128, value=int(mcfg["history_length"]))
    st.caption(phys.get("note") or f"{hist} measurements of history")
    with st.expander("Advanced"):
        smoke = st.checkbox("Smoke test — not a quality benchmark", value=True)
        st.number_input("Learning rate", value=float(mcfg["learning_rate"]), format="%.5f", disabled=True)
        st.number_input("Hidden size", value=int(mcfg["hidden_size"]), disabled=True)
        st.number_input("Batch size", value=int(mcfg["batch_size"]), disabled=True)
        max_w = st.number_input("Max windows / unit (0 = all)", min_value=0, value=48)
    runs = [r for r in list_runs(dataset_id) if r.get("has_last")]
    resume_opt = ["(new experiment)"] + [r["run_id"] for r in runs]
    resume = st.selectbox("Resume last checkpoint", resume_opt)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start training", disabled=worker_alive()):
            job = {
                "kind": "train",
                "dataset_id": dataset_id,
                "architecture": arch,
                "max_epochs": int(epochs),
                "history_length": int(hist),
                "smoke": bool(smoke),
                "resume_run_id": None if resume == "(new experiment)" else resume,
                "max_windows_per_unit": int(max_w) if max_w else None,
            }
            spawn_worker(job)
            st.rerun()
    with c2:
        if st.button("Stop", disabled=not worker_alive()):
            request_stop()
            st.rerun()
    ws = read_status()
    if ws.get("status") in {"training", "preparing"} or worker_alive():
        st.write(f"Epoch {ws.get('epoch', '—')} / {ws.get('max_epochs', '—')}")
        st.write(f"train loss {ws.get('train_loss')}  val loss {ws.get('val_loss')}  best epoch {ws.get('best_epoch')}")
        st.caption(ws.get("message") or "")
        _auto_refresh()
    if ws.get("status") == "failed":
        st.error(ws.get("error"))
    st.subheader("Experiments")
    table = list_runs(dataset_id)
    if not table:
        st.write("No runs yet.")
        return
    df = pd.DataFrame(table)
    cols = [c for c in ["run_id", "architecture", "status", "best_metric", "best_epoch", "smoke", "updated_at", "has_best"] if c in df.columns]
    st.dataframe(df[cols] if cols else df, width="stretch", hide_index=True)
    pick = st.selectbox("Open saved run (does not retrain)", [r["run_id"] for r in table])
    rdir = run_dir(dataset_id, pick)
    hist_csv = rdir / "training_history.csv"
    if hist_csv.exists():
        hdf = pd.read_csv(hist_csv)
        fig = go.Figure()
        if "train_loss" in hdf.columns:
            fig.add_trace(go.Scatter(x=hdf["epoch"], y=hdf["train_loss"], name="train loss"))
        if "val_loss" in hdf.columns:
            fig.add_trace(go.Scatter(x=hdf["epoch"], y=hdf["val_loss"], name="val loss"))
        fig.update_xaxes(title="Epoch")
        fig.update_yaxes(title="Loss (not comparable across loss types)")
        st.plotly_chart(fig, width="stretch")
        st.caption("MAE is shown in physical units when defined; do not treat different losses as one accuracy.")
    if (rdir / "validation_metrics.json").exists():
        st.json(json.loads((rdir / "validation_metrics.json").read_text()))


def screen_replay(dataset_id: str) -> None:
    st.header(f"Test & Replay — {LABELS[dataset_id]}")
    st.info("Historical replay — not a live equipment connection")
    if not processed_ready(dataset_id):
        st.warning("Prepare data first.")
        return
    table = [r for r in list_runs(dataset_id) if r.get("has_best") or r.get("has_last")]
    if not table:
        st.warning("No saved model. Train first.")
        return
    bundle = load_processed(dataset_id)
    features = bundle["features"]
    units = bundle["units"]
    split = bundle["split"]
    run_id = st.selectbox("Saved model", [r["run_id"] for r in table])
    test_units = split.get("test") or []
    uid = st.selectbox("Test unit", test_units)
    rdir = run_dir(dataset_id, run_id)
    cfg = load_dataset_config(dataset_id)
    train_units = units[units["unit_id"].isin(split["train"])]
    default_h = default_horizon_s(train_units, float(cfg.get("alerts", {}).get("horizon_fraction_of_median_train", 0.1)))
    unit_h = st.selectbox("Horizon unit", ["seconds", "minutes", "hours"], index=1)
    factor = {"seconds": 1.0, "minutes": 60.0, "hours": 3600.0}[unit_h]
    h_disp = st.number_input("Warning horizon H", value=float(default_h / factor))
    h_s = float(h_disp) * factor
    k = st.number_input("Confirmation count K", min_value=1, max_value=20, value=3)
    st.caption(
        f"Alert confirms after {k} consecutive predictions with RUL ≤ H. "
        f"That adds up to {int(k) - 1} extra measurement steps of delay vs the first trigger."
    )
    st.caption("H is a trigger threshold, not an accuracy promise. Changing H does not retrain the network.")
    if st.button("Evaluate test set", disabled=worker_alive()):
        spawn_worker(
            {
                "kind": "evaluate",
                "dataset_id": dataset_id,
                "run_id": run_id,
                "warning_horizon_s": h_s,
                "confirmation_count": int(k),
            }
        )
        st.rerun()
    if worker_alive() and read_status().get("kind") == "evaluate":
        st.info("Evaluating…")
        _auto_refresh()

    meas = features[features["unit_id"] == uid].sort_values("timestamp_s").reset_index(drop=True)
    n = len(meas)
    if "replay_step" not in st.session_state or st.session_state.get("replay_unit") != uid:
        st.session_state.replay_step = 0
        st.session_state.replay_unit = uid
        st.session_state.playing = False
    speed = st.slider("Replay speed (steps per refresh)", 1, 20, 1)
    show_gt = st.checkbox("Show ground truth", value=False)
    st.caption("Ground-truth toggle does not change predictions.")
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Play"):
        st.session_state.playing = True
    if b2.button("Pause"):
        st.session_state.playing = False
    if b3.button("Step"):
        st.session_state.replay_step = min(n - 1, st.session_state.replay_step + 1)
        st.session_state.playing = False
    if b4.button("Reset"):
        st.session_state.replay_step = 0
        st.session_state.playing = False
        st.session_state.pop("replay_cache", None)
    if st.session_state.playing:
        st.session_state.replay_step = min(n - 1, st.session_state.replay_step + speed)
        if st.session_state.replay_step >= n - 1:
            st.session_state.playing = False
        _auto_refresh(0.4)

    preds_path = rdir / "predictions.csv"
    if not preds_path.exists():
        st.info("Run Evaluate test set to compute predictions, alerts, and CSV export for all test units.")
        # still allow on-demand prefix prediction for the selected unit
        _live_prefix_charts(dataset_id, run_id, uid, meas, st.session_state.replay_step, show_gt, units, h_s)
        return

    preds = pd.read_csv(preds_path)
    unit_pred = preds[preds["unit_id"] == uid].sort_values("timestamp_s")
    step = st.session_state.replay_step
    prefix_pred = unit_pred.iloc[: step + 1] if len(unit_pred) else unit_pred
    current = prefix_pred.iloc[-1] if len(prefix_pred) else None

    hist_csv = rdir / "training_history.csv"
    st.subheader("1. Training history")
    if hist_csv.exists():
        hdf = pd.read_csv(hist_csv)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hdf["epoch"], y=hdf["train_loss"], name="train loss"))
        fig.add_trace(go.Scatter(x=hdf["epoch"], y=hdf["val_loss"], name="val loss"))
        fig.update_xaxes(title="Epoch")
        fig.update_yaxes(title="Loss")
        st.plotly_chart(fig, width="stretch")
    else:
        st.write("No training history file.")

    st.subheader("2. Predicted vs actual RUL")
    fig = go.Figure()
    x = prefix_pred["timestamp_s"] / 60.0
    fig.add_trace(go.Scatter(x=x, y=prefix_pred["predicted_rul_s"] / 60.0, name="neural net RUL (min)"))
    if "baseline_rul_s" in prefix_pred.columns:
        fig.add_trace(go.Scatter(x=x, y=prefix_pred["baseline_rul_s"] / 60.0, name="baseline RUL (min)"))
    if show_gt and "actual_rul_s" in prefix_pred.columns:
        fig.add_trace(go.Scatter(x=x, y=prefix_pred["actual_rul_s"] / 60.0, name="actual RUL (min)", line=dict(dash="dash")))
    fig.add_hline(y=h_s / 60.0, line_dash="dot", annotation_text="H")
    if current is not None:
        fig.add_vline(x=float(current["timestamp_s"]) / 60.0, line_dash="dot")
    fig.update_xaxes(title="Operating time (min)")
    fig.update_yaxes(title="Remaining useful life (min)")
    st.plotly_chart(fig, width="stretch")

    st.subheader("3. Sensor / feature trend")
    g = meas.iloc[: step + 1]
    fig2 = go.Figure()
    if dataset_id == "bearings":
        fig2.add_trace(go.Scatter(x=g["timestamp_s"] / 60.0, y=g["horizontal_rms"], name="horizontal RMS"))
        fig2.update_yaxes(title="RMS")
    else:
        fig2.add_trace(go.Scatter(x=g["timestamp_s"] / 60.0, y=g["differential_pressure"], name="Δp (Pa)"))
        fig2.add_hline(y=600.0, line_dash="dash", annotation_text="600 Pa")
        fig2.update_yaxes(title="Differential pressure (Pa)")
    fig2.update_xaxes(title="Operating time (min)")
    st.plotly_chart(fig2, width="stretch")

    if current is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Operating age (min)", f"{float(current['timestamp_s']) / 60.0:.1f}")
        rul = current.get("predicted_rul_s")
        c2.metric("Predicted RUL (min)", "—" if pd.isna(rul) else f"{float(rul) / 60.0:.1f}")
        c3.metric("Alert", current.get("alert_status", "—"))
        c4.metric("Model", run_id)
    st.caption("Replay speed does not change physical time, targets, or alert conditions.")

    if (rdir / "test_metrics.json").exists():
        st.subheader("Quality by unit")
        metrics = json.loads((rdir / "test_metrics.json").read_text())
        st.json(metrics)
    alerts_path = rdir / "alerts.csv"
    if alerts_path.exists():
        st.subheader("Alert log")
        st.dataframe(pd.read_csv(alerts_path), width="stretch", hide_index=True)
    st.download_button("Download predictions CSV", data=preds.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    if alerts_path.exists():
        st.download_button("Download alerts CSV", data=alerts_path.read_text(), file_name="alerts.csv", mime="text/csv")


def _live_prefix_charts(dataset_id, run_id, uid, meas, step, show_gt, units, h_s) -> None:
    st.caption("Showing the selected prefix only. Full test-set tables appear after Evaluate test set.")
    g = meas.iloc[: step + 1]
    fig = go.Figure()
    if dataset_id == "bearings" and "horizontal_rms" in g.columns:
        fig.add_trace(go.Scatter(x=g["timestamp_s"] / 60.0, y=g["horizontal_rms"], name="RMS"))
    elif "differential_pressure" in g.columns:
        fig.add_trace(go.Scatter(x=g["timestamp_s"] / 60.0, y=g["differential_pressure"], name="Δp"))
        fig.add_hline(y=600)
    st.plotly_chart(fig, width="stretch")


def _split_map(split: dict) -> dict:
    m = {}
    for part in ("train", "validation", "test"):
        for u in split.get(part) or []:
            m[u] = part
    return m


@st.fragment(run_every=2.0)
def _worker_poll_fragment() -> None:
    ws = read_status()
    st.caption(f"Worker status {ws.get('status', '—')} @ {time.strftime('%H:%M:%S')}")


def _auto_refresh(seconds: float = 2.0) -> None:
    try:
        _worker_poll_fragment()
    except Exception:
        time.sleep(min(max(seconds, 0.05), 2.0))
        st.rerun()


if __name__ == "__main__":
    main()
