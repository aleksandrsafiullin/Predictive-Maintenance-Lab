"""Shared experiment views for the four-page laboratory."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from pdm.benchmark import compare_evaluations, evaluation_choices, load_evaluation, save_comparison
from pdm.cli import spawn_worker
from pdm.data.quality import training_admission
from pdm.experiments import list_evaluations, run_dir
from pdm.io_util import read_json
from pdm.paths import runs_root
from pdm.worker import read_status, request_stop, worker_alive

COLORS = ["#61d8ee", "#f2be68", "#b59cf8", "#82daa6", "#fc8caa", "#a6b8ff"]
MODEL_NAMES = {"gru": "GRU", "lstm": "LSTM", "fly_connectome_reservoir": "Fly reservoir", "random_reservoir": "Random reservoir"}


def model_label(run):
    nodes = run.get("n_nodes")
    name = "Full MaleCNS" if nodes == 166700 else MODEL_NAMES.get(run.get("architecture"), run.get("architecture", "Model"))
    size = f"{nodes or '—'} h / {nodes or '—'} c" if run.get("architecture") == "lstm" else f"{nodes or '—'} states"
    return f"{name} · {size} · {run['run_id'][-6:]}"


def style_figure(fig, height=500):
    fig.update_layout(height=height, paper_bgcolor="#101925", plot_bgcolor="#101925", font_color="#edf4fa",
                      margin=dict(l=30, r=20, t=50, b=35), legend=dict(orientation="h", y=1.12))
    fig.update_xaxes(gridcolor="#263447")
    fig.update_yaxes(gridcolor="#263447")
    return fig


def quality_overview(bundle):
    from pdm.monitoring.ui import quality_details

    quality_details(bundle["split"]["dataset_id"])
    q = bundle.get("report", {}).get("quality")
    if not q:
        st.warning("This snapshot predates Data Quality admission. Prepare a new version before training.")
        return
    cols = st.columns(4)
    for col, label, value in zip(cols, ["Admitted measurements", "Excluded measurements", "Needs attention", "Admitted units"],
                                 [q["admitted_measurements"], q["excluded_measurements"], q["attention_measurements"], q["admitted_units"]], strict=True):
        col.metric(label, f"{value:,}")
    st.caption("Attention records remain in training: unusual signals can contain degradation evidence. Raw source files are preserved.")
    if bundle["split"]["dataset_id"] == "filters":
        st.info("A valid series without failure teaches a lower bound: the filter kept operating for the remaining observed time under its measured pressure, flow and dust conditions. The replacement time is not labelled as a failure.")
    with st.expander("Admission decisions and source records"):
        path = Path(bundle["dir"]) / "quality_records.parquet"
        audit = pd.read_parquet(path)
        status = st.multiselect("Record status", ["admitted", "attention", "excluded"], default=["attention", "excluded"])
        selection = audit[audit.status.isin(status)]
        st.dataframe(selection, hide_index=True, width="stretch")
        if q["excluded_units"]:
            st.json(q["excluded_units"])
        st.download_button("Download admission audit", audit.to_csv(index=False), "data-quality.csv", "text/csv")
        st.caption(f"Policy: {q['policy']['version']} · dataset {bundle['dataset_version']}")


def training_overview(bundle, cfg, history_length=None):
    try:
        _, counts = training_admission(bundle, cfg, int(history_length or cfg["model"]["history_length"]))
        st.dataframe(pd.DataFrame(counts), hide_index=True, width="stretch")
    except (ValueError, OSError) as exc:
        st.error(str(exc))
        return False
    return True


def matrix_controls():
    st.fragment(run_every=3 if worker_alive() else None)(_training_study_controls)()
    st.fragment(run_every=3 if worker_alive() else None)(_matrix_controls)()


def _training_study_controls():
    with st.expander("Training improvement study", expanded=True):
        st.write("Test 100 epochs, adaptive learning rates, complete window coverage and degradation features. "
                 "Screen GRU recipes, validate by equipment groups, then train all architectures and confirm finalists.")
        a, b = st.columns(2)
        if a.button("Start improvement study", disabled=worker_alive(), type="primary"):
            spawn_worker({"kind": "training_study"})
            st.rerun()
        paths = sorted((runs_root() / "training_studies").glob("*/manifest.json"), reverse=True)
        if not paths:
            return
        manifest = read_json(paths[0])
        if b.button("Resume improvement study", disabled=worker_alive() or manifest.get("status") == "completed"):
            spawn_worker({"kind": "training_study", "study_id": manifest["study_id"]})
            st.rerun()
        status = read_status()
        tasks = list(manifest.get("tasks", {}).values())
        st.caption(f"Study {manifest['study_id']} · {manifest.get('status')} · "
                   f"{sum(t.get('status') == 'completed' for t in tasks)} completed runs. Later stages are selected from validation.")
        if status.get("study_id") == manifest["study_id"]:
            if status.get("unit_total"):
                st.progress(status["unit_index"] / status["unit_total"],
                            text=f"Evaluating {status['unit_id']} · {status['unit_index']} / {status['unit_total']} objects")
                st.caption(f"{status.get('stage', '')} · {status.get('run_id', '')}")
            elif status.get("stage"):
                st.write(status.get("message") or status["stage"])
            if status.get("epoch"):
                st.progress(min(status["epoch"] / status.get("max_epochs", 100), 1.0),
                            text=f"Epoch {status['epoch']} / {status.get('max_epochs', 100)} · best {status.get('best_epoch', '—')}")
                st.caption(f"Learning rate: {status.get('learning_rate', '—')} · stop reason: {status.get('stop_reason', 'running')}")
                st.caption(f"Features: {status.get('feature_recipe', '—')} · unique windows this epoch: "
                           f"{status.get('n_unique_sampled_windows', '—')} / {status.get('n_eligible_windows', '—')}")
            if status.get("run_id") and status.get("dataset_id"):
                history = run_dir(status["dataset_id"], status["run_id"]) / "training_history.csv"
                if history.exists():
                    curve = pd.read_csv(history)
                    if {"epoch", "train_metric", "val_metric"}.issubset(curve) and len(curve) > 1:
                        st.plotly_chart(style_figure(px.line(curve, x="epoch", y=["train_metric", "val_metric"],
                                                            title=status.get("selection_metric_name", "Selection metric")), 260),
                                        width="stretch", theme=None, key="study_live_metric")
        if manifest.get("error") and manifest.get("status") == "failed":
            st.error(manifest["error"])
        if worker_alive() and status.get("study_id") == manifest["study_id"]:
            log = paths[0].parent / "study.log"
            if log.exists():
                with st.expander("Current computation log"):
                    st.code("\n".join(log.read_text()[-4000:].splitlines()[-10:]), language="text")
        if tasks:
            table = pd.DataFrame([{"stage": key, **task} for key, task in manifest["tasks"].items()])
            st.dataframe(table[[c for c in ("stage", "dataset_id", "architecture", "seed", "status", "primary_score", "best_epoch", "run_id") if c in table]], hide_index=True, width="stretch")
        if worker_alive() and st.button("Stop improvement study"):
            request_stop()
            st.rerun()
        active_study = worker_alive() and status.get("study_id") == manifest["study_id"]
        if active_study:
            st.caption("Study exports become available when the worker finishes or pauses.")
        else:
            for name in ("results.csv", "results.json", "baselines.csv", "screening.csv", "grouped_validation.csv",
                         "epoch_diagnostics.csv", "seed_dispersion.csv", "report.md", "manifest.json"):
                path = paths[0].parent / name
                if path.exists():
                    st.download_button("Download " + name, path.read_bytes(), name, key="study:" + name, on_click="ignore")


def _matrix_controls():
    with st.expander("Historical training protocol · comparison study", expanded=read_status().get("kind") == "train_matrix"):
        st.write("Train and evaluate 9 full models across bearings and filters, plus the paired filter censoring study (14 experiments total). Jobs run sequentially.")
        a, b = st.columns(2)
        if a.button("Run full training matrix", disabled=worker_alive(), type="primary"):
            spawn_worker({"kind": "train_matrix", "datasets": ["bearings", "filters"], "include_ablation": True})
            st.rerun()
        manifests = sorted((runs_root() / "batches").glob("*/manifest.json"), reverse=True)
        if manifests:
            manifest = read_json(manifests[0])
            unfinished = manifest.get("status") != "completed"
            if b.button("Resume unfinished matrix", disabled=worker_alive() or not unfinished):
                spawn_worker({"kind": "train_matrix", "batch_id": manifest["batch_id"]})
                st.rerun()
            tasks = manifest["tasks"]
            done = sum(t.get("status") == "completed" for t in tasks)
            st.progress(done / max(len(tasks), 1), text=f"{done} / {len(tasks)} completed · {manifest.get('status')}")
            status = read_status()
            if status.get("batch_id") == manifest["batch_id"] and worker_alive():
                st.caption(f"{status.get('message', '')} · {status.get('stage', status.get('status', ''))}")
                if status.get("epoch"):
                    st.write(f"Epoch {status['epoch']} / {status.get('max_epochs', 30)} · validation {status.get('val_metric', '—')}")
                if status.get("unit_index"):
                    st.write(f"Evaluating unit {status['unit_index']} / {status['unit_total']} · {status['unit_id']}")
                log = manifests[0].parent / "batch.log"
                if log.exists():
                    with st.expander("Current computation log"):
                        st.code("\n".join(log.read_text()[-6000:].splitlines()[-18:]), language="text")
            st.dataframe(pd.DataFrame(tasks)[[c for c in ("dataset_id", "architecture", "seed", "events_only", "status", "run_id", "error") if any(c in t for t in tasks)]], hide_index=True, width="stretch")
            if worker_alive() and st.button("Stop matrix"):
                request_stop()
                st.rerun()
            ablation = manifests[0].parent / "censoring_ablation.csv"
            if ablation.exists():
                st.markdown("**Censoring study · GRU · fixed validation cohort**")
                frame = pd.read_csv(ablation)
                st.dataframe(frame, hide_index=True, width="stretch")
                failures = sorted(frame.n_observed_events.dropna().astype(int).unique())
                st.caption(f"Lower NLL is better. Observed validation failures per run: {', '.join(map(str, failures)) or 'unavailable'}. Results describe these GRU experiments and this fixed cohort.")
                paired = frame.pivot(index="seed", columns="events_only", values="survival_nll").dropna()
                if False in paired and True in paired and len(paired):
                    difference = paired[False] - paired[True]
                    st.write(f"Paired seeds completed: {len(paired)} / 3. Mean NLL difference (all valid − failures only): {difference.mean():+.4f}. Negative values favour retaining censored series.")
                st.download_button("Download censoring study", frame.to_csv(index=False), "censoring-ablation.csv", "text/csv")


def report_evaluations(dataset_id, run_id):
    root = run_dir(dataset_id, run_id)
    st.subheader("Forecast quality")
    evaluations = list_evaluations(root)
    if not evaluations:
        legacy = [p for p in (root / "test_metrics.json", root / "validation_metrics.json") if p.exists()]
        if legacy:
            current = any(read_json(path).get("training_protocol", {}).get("version") == "training_v2" for path in legacy)
            st.caption("Checkpoint selection diagnostics · full replay evaluation has not been saved yet." if current
                       else "Historical metrics · original protocol; excluded from the new quality ranking.")
            for path in legacy:
                st.json(read_json(path), expanded=False)
                st.download_button("Download " + path.name, path.read_bytes(), run_id + "-" + path.name, "application/json")
        else:
            st.info("No saved evaluation for this run. Use Evaluation settings to create one.")
    else:
        options = {e["eval_id"]: e for e in evaluations}
        eid = st.selectbox("Report evaluation", list(options),
                           format_func=lambda x: f"{options[x]['evaluate_mask']['split']} · {x}", key="report_eval:" + run_id)
        try:
            ev = load_evaluation(dataset_id, run_id, eid)
            result = compare_evaluations([ev])
            row = result["table"].iloc[0]
            st.caption("Validation · used for model selection." if result["split"] == "validation"
                       else "Test · previously inspected holdout · exploratory results; choose models using validation.")
            metric = "30-min MAE" if dataset_id == "bearings" else "Survival NLL" if result["split"] == "validation" else "Prefix-end MAE"
            cols = st.columns(3)
            score = row.primary_score
            cols[0].metric(metric, "—" if pd.isna(score) else f"{score:.3f}", help="MAE is in internal seconds; NLL is dimensionless.")
            cols[1].metric("Prediction coverage", f"{row.prediction_coverage:.0%}")
            cols[2].metric("Evaluation units", int(row.units))
            if row.reason:
                st.caption(row.reason)
            interval_label = "unavailable" if row.interval_origin == "unavailable" else f"{row.interval_origin} · {row.interval_evaluation}"
            st.caption(f"Interval: {interval_label}. Observed failures: {row.observed_events}.")
            if pd.notna(row.get("warning_goal_met")):
                warning_cols = st.columns(2)
                warning_cols[0].metric("Useful warning episodes", "—" if pd.isna(row.useful_precision) else f"{row.useful_precision:.0%}")
                warning_cols[1].metric("Timely failure warnings", "—" if pd.isna(row.timely_recall) else f"{row.timely_recall:.0%}")
                if not row.warning_goal_met:
                    st.warning("Warning usefulness target not met. A low RUL error alone does not qualify this model for use.")
                st.caption(str(row.get("evidence_status", "")))
                with st.expander("Warning outcomes across the full recorded history"):
                    st.json(ev["metrics"].get("alerts", {}).get("useful", {}))
            if dataset_id == "filters" and result["split"] == "test":
                st.caption(f"Official prefix-end RUL labels: {int(row.units)}. These are evaluation references; the observed prefixes do not contain those failures.")
            st.dataframe(result["table"][["all_history_mae_s", "overestimation_s", "interval_coverage", "interval_width_s", "alerts_timely", "alerts_late", "alerts_miss"]], hide_index=True, width="stretch")
            st.dataframe(result["per_unit"], hide_index=True, width="stretch")
            st.download_button("Download report predictions", ev["predictions"].to_csv(index=False), f"{run_id}-predictions.csv", "text/csv")
            st.download_button("Download evaluation JSON", json.dumps(ev["metrics"], indent=2), f"{eid}.json", "application/json")
        except (ValueError, KeyError, OSError) as exc:
            st.caption(f"Historical evaluation: {exc}. Its original metrics remain available.")
            saved_metrics = root / "evaluations" / eid / "metrics.json"
            if saved_metrics.exists():
                st.json(read_json(saved_metrics), expanded=False)
                st.download_button("Download original evaluation JSON", saved_metrics.read_bytes(), f"{eid}.json", "application/json")
    history = root / "training_history.csv"
    if history.exists():
        with st.expander("Training history and experiment details"):
            frame = pd.read_csv(history)
            status = read_json(root / "status.json") if (root / "status.json").exists() else {}
            prep = read_json(root / "preprocessing.json") if (root / "preprocessing.json").exists() else {}
            st.caption(f"Completed epoch: {status.get('epoch', len(frame))} · best epoch: {status.get('best_epoch', '—')} · "
                       f"stop reason: {status.get('stop_reason', 'legacy protocol')} · "
                       f"features: {prep.get('feature_recipe', 'base_v1')} ({len(prep.get('feature_names', []))})")
            if prep.get("feature_recipe") == "degradation_v1":
                st.caption("Model window: 20 measurements. Features use 5/20-point slopes and causal segment history for initial levels or accumulated dust. Gaps reset that history.")
            if "learning_rate" in frame:
                st.plotly_chart(style_figure(px.line(frame, x="epoch", y="learning_rate"), 220), width="stretch", theme=None)
            cols = [c for c in ("train_metric", "val_metric") if c in frame and frame[c].notna().any()]
            if len(frame) > 1 and cols:
                st.plotly_chart(style_figure(px.line(frame, x="epoch", y=cols, color_discrete_sequence=COLORS), 300), width="stretch", theme=None)
            loss_cols = [c for c in ("train_loss", "val_loss") if c in frame and frame[c].notna().any()]
            if len(frame) > 1 and loss_cols:
                st.plotly_chart(style_figure(px.line(frame, x="epoch", y=loss_cols, title="Optimization loss", color_discrete_sequence=COLORS), 240), width="stretch", theme=None)
            st.dataframe(frame, hide_index=True, width="stretch")
            if not cols:
                st.caption("Closed-form readout fitting has no gradient loss curve.")
            if (root / "dataset_fingerprint.json").exists():
                st.json(read_json(root / "dataset_fingerprint.json"), expanded=False)


def _open_report(run_id):
    st.session_state["screen_selection"] = "Model Report"
    st.session_state["report_view"] = "Model replay"
    st.session_state["_report_open_run"] = run_id


def _restore_comparison(dataset_id):
    path = st.session_state.get("saved_comparison:" + dataset_id)
    if not path:
        return
    doc = read_json(Path(path))
    st.session_state["comparison_split:" + dataset_id] = doc["split"]
    st.session_state["comparison_runs:" + dataset_id + ":" + doc["split"]] = [s["run_id"] for s in doc["selections"]]
    for item in doc["selections"]:
        st.session_state[f"compare_eval:{doc['split']}:{item['run_id']}"] = item["eval_id"]


def screen_comparison(dataset_id):
    from pdm.monitoring.ui import monitoring_comparison

    st.header("Compare models")
    if monitoring_comparison(dataset_id):
        return
    st.caption("One cohort. One measurement clock. Explicit saved evaluations.")
    saved = {str(p): read_json(p) for p in sorted((runs_root() / "comparisons").glob("*/comparison.json"), reverse=True)}
    saved = {p: d for p, d in saved.items() if d.get("dataset_id") == dataset_id}
    if saved:
        st.selectbox("Saved comparison", [None, *saved], format_func=lambda p: "New selection" if p is None else f"{saved[p]['split']} · {len(saved[p]['selections'])} models · {Path(p).parent.name}",
                     key="saved_comparison:" + dataset_id, on_change=_restore_comparison, args=(dataset_id,))
    split = st.radio("Comparison split", ["validation", "test"], horizontal=True, key="comparison_split:" + dataset_id)
    st.info("Choose models using validation. Test is a previously inspected holdout; these results are exploratory." if split == "test" else "Validation selects candidates. Interval coverage on calibration units is labelled in each model report.")
    if dataset_id == "filters":
        st.caption("Filter time scale is unverified. MAE uses the existing internal Time × 60 convention.")
    choices = evaluation_choices(dataset_id, split)
    include_diagnostics = st.checkbox("Include historical, smoke and synthetic runs", value=False)
    if not include_diagnostics:
        choices = [r for r in choices if r.get("quality_policy_hash") and not r.get("smoke")
                   and not r.get("is_synthetic") and r.get("graph_mode") != "synthetic_fixture"]
    if not choices:
        st.info("No admitted full evaluations for this split yet. Run the full training matrix. Historical results remain available in Model Report and through the diagnostic-runs switch above.")
        return
    runs = {r["run_id"]: r for r in reversed(choices)}
    ordered = list(reversed(runs))
    # Prefer a completed matrix over mixing recent ablation and main runs.
    defaults, default_evaluations = [], {}
    for path in sorted((runs_root() / "training_studies").glob("*/manifest.json"), reverse=True):
        study = read_json(path)
        if study.get("status") != "completed":
            continue
        tasks = [t for key, t in study.get("tasks", {}).items() if ":main:" in key and t.get("dataset_id") == dataset_id]
        defaults = [t["run_id"] for t in tasks if t["run_id"] in runs and t.get(split + "_evaluation")]
        default_evaluations = {t["run_id"]: t.get(split + "_evaluation") for t in tasks if t["run_id"] in defaults}
        if defaults:
            break
    for manifest in sorted((runs_root() / "batches").glob("*/manifest.json"), reverse=True):
        if defaults:
            break
        tasks = read_json(manifest).get("tasks", [])
        defaults = [t["run_id"] for t in tasks if t.get("role") == "main" and t.get("dataset_id") == dataset_id and t.get("run_id") in runs]
        if defaults:
            default_evaluations = {t["run_id"]: t.get(f"{split}_eval_id") for t in tasks if t.get("run_id") in defaults}
            break
    if not defaults:
        newest = runs[ordered[0]].get("dataset_version")
        defaults = [rid for rid in ordered if runs[rid].get("dataset_version") == newest and not runs[rid].get("smoke")][:5]
    ids = st.multiselect("Models to compare", ordered, default=defaults,
                        format_func=lambda rid: model_label(runs[rid]), key="comparison_runs:" + dataset_id + ":" + split)
    selected = []
    with st.expander("Selected evaluation artifacts"):
        for rid in ids:
            items = [e for e in choices if e["run_id"] == rid]
            eids = [e["eval_id"] for e in items]
            preferred = default_evaluations.get(rid)
            eid = st.selectbox(rid, eids, index=eids.index(preferred) if preferred in eids else 0, key=f"compare_eval:{split}:{rid}")
            selected.append((rid, eid))
    if not selected:
        return
    try:
        result = compare_evaluations([load_evaluation(dataset_id, rid, eid) for rid, eid in selected])
    except (ValueError, OSError, KeyError) as exc:
        st.warning(str(exc))
        return
    table = result["table"]
    metric = "MAE in the final 30 minutes" if dataset_id == "bearings" else "Survival NLL on all validation units" if split == "validation" else "MAE at official prefix endpoints"
    st.markdown(f"**Primary metric: {metric} · lower is better**")
    st.caption("Every equipment unit has equal weight. Missing forecasts reduce coverage and prevent ranking. All MAE columns use internal seconds.")
    leading_columns = ["rank", "model", "nodes", "state_mode", "feature_recipe", "primary_score", "prediction_coverage", "units", "observed_events", "warning_goal_met", "reason"]
    display = table[leading_columns].copy()
    display["model"] = table.run_id.map(lambda rid: model_label(runs[rid]))
    st.dataframe(display, hide_index=True, width="stretch", column_config={
        "rank": st.column_config.NumberColumn("Rank", format="%.0f"),
        "model": "Model", "nodes": "State size", "state_mode": "Memory mode",
        "primary_score": st.column_config.NumberColumn("Primary score", format="%.3f"),
        "prediction_coverage": st.column_config.NumberColumn("Coverage (0–1)", format="%.3f"),
        "units": "Units", "observed_events": "Observed failures", "reason": "Ranking exclusion",
    })
    with st.expander("Errors, intervals and warning quality"):
        details = table.drop(columns=["alerts", "eval_id", "run_id", "rank_eligible"]).copy()
        details["model"] = display["model"]
        st.dataframe(details, hide_index=True, width="stretch")
    eligible = table[table.rank_eligible]
    if len(eligible) >= 2:
        best = eligible[eligible["rank"] == eligible["rank"].min()]
        st.success(("Leading validation model: " if split == "validation" else "Lowest error on this reused test cohort: ") + ", ".join(model_label(runs[rid]) for rid in best.run_id))
        if best.warning_goal_met.notna().any() and not best.warning_goal_met.fillna(False).all():
            st.warning("The RUL leader does not meet the warning usefulness target. Ranking is not a readiness decision.")
    if not result["alert_policies_match"]:
        st.warning("Alert thresholds differ; alert outcomes are not ranked together.")
    st.caption("Alert columns describe each saved evaluation's complete history. Interval results on calibration units are marked separately; Weibull ranges describe the fitted distribution and have no empirical calibration guarantee.")
    pred = result["predictions"]
    uid = st.selectbox("Comparison unit", sorted(pred.unit_id.unique()))
    unit = pred[pred.unit_id == uid]
    clock = st.slider("Measurement clock", float(unit.timestamp_s.min()), float(unit.timestamp_s.max()), float(unit.timestamp_s.max())) if unit.timestamp_s.nunique() > 1 else float(unit.timestamp_s.iloc[0])
    visible = unit[unit.timestamp_s <= clock]
    scale = 60 if dataset_id == "bearings" else 1
    truth = visible.drop_duplicates("timestamp_s")
    has_truth = truth.actual_rul_s.notna().any()
    fig = make_subplots(rows=2 if has_truth else 1, cols=1, shared_xaxes=True,
                        subplot_titles=("Remaining useful life", "Absolute prediction error") if has_truth else ("Remaining useful life",),
                        vertical_spacing=0.17)
    if has_truth:
        fig.add_trace(go.Scatter(x=truth.timestamp_s / scale, y=truth.actual_rul_s / scale, name="Actual", line=dict(color="#ff7e83", dash="dash")), row=1, col=1)
    elif "outcome_duration_s" in truth:
        st.caption("This filter has no registered failure. Remaining observed operation is a lower bound on RUL; point error is unavailable. Its survival likelihood still contributes to validation NLL.")
        fig.add_trace(go.Scatter(x=truth.timestamp_s / scale, y=truth.outcome_duration_s / scale,
                                name="Observed RUL lower bound", line=dict(color="#ff7e83", dash="dash")), row=1, col=1)
    for i, (rid, g) in enumerate(visible.groupby("run_id", sort=False)):
        label = model_label(runs[rid])
        color = COLORS[i % len(COLORS)]
        fig.add_trace(go.Scatter(x=g.timestamp_s / scale, y=g.predicted_rul_s / scale, name=label, line_color=color), row=1, col=1)
        if has_truth:
            fig.add_trace(go.Scatter(x=g.timestamp_s / scale, y=g.absolute_error_s / scale, name=label, line_color=color, showlegend=False), row=2, col=1)
    fig.update_xaxes(title_text="Time (min)" if scale == 60 else "Time (internal seconds)", row=2 if has_truth else 1, col=1)
    style_figure(fig, 740 if has_truth else 540)
    # Up to six wrapped legend lines on a narrow screen. Anchor above the
    # subplot title so labels never cover the measurements.
    fig.update_layout(margin=dict(l=35, r=15, t=175, b=45),
                      legend=dict(orientation="h", y=1.13, yanchor="bottom", font_size=11))
    st.plotly_chart(fig, width="stretch", theme=None)
    st.dataframe(result["per_unit"], hide_index=True, width="stretch")
    open_id = st.selectbox("Open model report", ids, format_func=lambda rid: f"{runs[rid].get('architecture')} · {rid[-6:]}")
    st.button("Open selected report", on_click=_open_report, args=(open_id,))
    a, b, c = st.columns(3)
    a.download_button("Download comparison CSV", table.to_csv(index=False), "model-comparison.csv", "text/csv")
    b.download_button("Download predictions CSV", pred.to_csv(index=False), "comparison-predictions.csv", "text/csv")
    doc = {k: v for k, v in result.items() if k not in {"table", "per_unit", "predictions"}}
    doc["table"] = json.loads(table.to_json(orient="records"))
    c.download_button("Download comparison JSON", json.dumps(doc, indent=2), "comparison.json", "application/json")
    st.download_button("Download predictions JSON", pred.to_json(orient="records"), "comparison-predictions.json", "application/json")
    if st.button("Save this comparison"):
        st.success(f"Saved: {save_comparison(result)}")
    st.caption("Synthetic fixtures and smoke runs can be inspected but cannot lead the quality ranking. Random reservoir is an engineered control; only its matched 1,000-node fly run isolates topology differences.")
