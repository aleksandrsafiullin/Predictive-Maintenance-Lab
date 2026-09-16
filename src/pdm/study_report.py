"""Durable study results, including unsuccessful experiments and simple controls."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from pdm.benchmark import compare_evaluations, load_evaluation
from pdm.data.prepare import load_processed
from pdm.experiments import run_dir
from pdm.io_util import atomic_write_json, read_json
from pdm.study_baselines import evaluate_baseline
from pdm.study_metrics import useful_warning_metrics


def markdown_table(frame):
    def cell(value):
        if pd.isna(value):
            return "—"
        return (f"{value:.4f}" if isinstance(value, float) else str(value)).replace("|", "\\|").replace("\n", " ")

    return "\n".join(["| " + " | ".join(frame.columns) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |",
                      *["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]])


def score_frame(frame, dataset_id, split):
    ready = frame[~frame.prediction_status.eq("Collecting history")].copy()
    if dataset_id == "bearings":
        ready = ready[ready.actual_rul_s > 0]
    elif split == "validation":
        ready = ready[ready.outcome_duration_s > 0]
    valid = ready.predicted_rul_s.notna() & np.isfinite(ready.predicted_rul_s)
    if dataset_id == "bearings":
        scored = ready[ready.actual_rul_s.between(0, 1800, inclusive="right")]
        score = abs(scored.predicted_rul_s - scored.actual_rul_s).groupby(scored.unit_id).mean().mean()
        primary_valid = np.isfinite(scored.predicted_rul_s)
    elif split == "validation":
        complete = "survival_nll" in ready and np.isfinite(ready.survival_nll).all()
        score = ready.groupby("unit_id").survival_nll.mean().mean() if complete else None
        primary_valid = np.isfinite(ready.survival_nll) if "survival_nll" in ready else pd.Series(False, index=ready.index)
    else:
        scored = ready.sort_values("timestamp_s").groupby("unit_id").tail(1)
        score = abs(scored.predicted_rul_s - scored.actual_rul_s).mean()
        primary_valid = np.isfinite(scored.predicted_rul_s)
    error = abs(ready.predicted_rul_s - ready.actual_rul_s)
    event_mae = error.groupby(ready.unit_id).mean().mean()
    return {"primary_score": float(score) if score is not None and np.isfinite(score) else None,
            "prediction_coverage": float(valid.mean()), "units": int(ready.unit_id.nunique()),
            "primary_points_available": int(primary_valid.sum()), "primary_points_expected": len(primary_valid),
            "primary_coverage": float(primary_valid.mean()),
            "event_mae_s": float(event_mae) if np.isfinite(event_mae) else None}


def counterfactual_stopping(history, metric_column, limit=30, patience=5):
    best, bad, selected = np.inf, 0, None
    for row in history.iloc[:limit].to_dict("records"):
        if row[metric_column] < best - 1e-8:
            best, bad, selected = row[metric_column], 0, row
        else:
            bad += 1
        if bad >= patience:
            break
    return {"stop_epoch": int(row["epoch"]), "best_epoch": int(selected["epoch"]),
            "primary_score": float(selected["val_metric"])}


def build_study_report(root, manifest):
    from pdm.training_protocol import protocol_artifact

    rows = []
    tasks = manifest["tasks"]
    if manifest.get("error"):
        manifest.setdefault("resolved_errors", []).append(manifest.pop("error"))
    for task in tasks.values():
        if task.get("status") == "completed" and task.get("error"):
            task.setdefault("resolved_errors", []).append(task.pop("error"))
        artifact = run_dir(task["dataset_id"], task["run_id"]) / "training_protocol.json"
        previous = read_json(artifact) if artifact.exists() else {}
        atomic_write_json(artifact, protocol_artifact(task["recipe"], task["dataset_id"], task["architecture"], task["seed"],
                                                    training_identity=previous.get("training_identity")))
    for key, task in tasks.items():
        if ":main:" not in key and ":confirmation:" not in key:
            continue
        for part in ("validation", "test"):
            if not task.get(part + "_evaluation"):
                continue
            ev = load_evaluation(task["dataset_id"], task["run_id"], task[part + "_evaluation"])
            row = compare_evaluations([ev])["table"].iloc[0].to_dict()
            rows.append({**row, "dataset_id": task["dataset_id"], "split": part,
                         "architecture": task["architecture"], "seed": task["seed"], "role": "new",
                         "recipe": task["recipe"]["feature_recipe"], "best_epoch": task.get("best_epoch")})
    for task in manifest["reference_runs"]:
        for part in ("validation", "test"):
            ev = load_evaluation(task["dataset_id"], task["run_id"], task[part + "_evaluation"])
            row = compare_evaluations([ev])["table"].iloc[0].to_dict()
            rows.append({**row, "dataset_id": task["dataset_id"], "split": part, "architecture": task["architecture"],
                         "seed": 42, "role": "reference", "recipe": "base_v1"})
    baseline_rows = []
    for ds in manifest["datasets"]:
        model = read_json(root / "baselines" / ds / "model.json")
        task = tasks[f"{ds}:main:gru:42"]
        for part in ("validation", "test"):
            ev = load_evaluation(ds, task["run_id"], task[part + "_evaluation"])
            reference = ev["predictions"]
            path = root / "baselines" / ds / (part + "_predictions.csv")
            frame = evaluate_baseline(model, reference, path)
            useful, episodes = useful_warning_metrics(frame, load_processed(ds, manifest["datasets"][ds])["units"], manifest["warning_settings"][ds])
            episodes.to_csv(path.with_name(part + "_warnings.csv"), index=False)
            baseline_rows.append({"dataset_id": ds, "split": part, "architecture": model["name"],
                                  "optimization_converged": model.get("optimization", {}).get("success", True),
                                  **score_frame(frame, ds, part), **useful})
            simple = reference.copy()
            simple["predicted_rul_s"] = simple.baseline_rul_s
            simple["raw_rul_s"] = simple.predicted_rul_s
            simple["absolute_error_s"] = abs(simple.predicted_rul_s - simple.actual_rul_s)
            simple["source_reference_run_id"] = task["run_id"]
            simple["run_id"] = "baseline_age_by_regime" if ds == "bearings" else "baseline_pressure_trend"
            simple["model"] = "age_by_regime" if ds == "bearings" else "pressure_trend"
            unavailable = ~np.isfinite(simple.predicted_rul_s) & ~simple.prediction_status.eq("Collecting history")
            simple.loc[unavailable, "prediction_status"] = "Unavailable"
            simple = simple.drop(columns=["survival_nll", "weibull_scale_s", "weibull_shape",
                                           "lower_rul_s", "upper_rul_s", "interval_method"], errors="ignore")
            simple.to_csv(path.with_name(part + "_simple_predictions.csv"), index=False)
            simple_useful, simple_episodes = useful_warning_metrics(simple, load_processed(ds, manifest["datasets"][ds])["units"],
                                                                    manifest["warning_settings"][ds])
            simple_episodes.to_csv(path.with_name(part + "_simple_warnings.csv"), index=False)
            baseline_rows.append({"dataset_id": ds, "split": part,
                                  "architecture": "age_by_regime" if ds == "bearings" else "pressure_trend",
                                  "optimization_converged": True,
                                  **score_frame(simple, ds, part), **simple_useful})
    result = pd.DataFrame(rows)
    baseline_table = pd.DataFrame(baseline_rows)
    for i, row in result.iterrows():
        baseline = baseline_table[(baseline_table.dataset_id == row.dataset_id) & baseline_table.split.eq(row.split)
                                  & baseline_table.prediction_coverage.eq(1) & baseline_table.optimization_converged]
        if baseline.empty:
            result.loc[i, "baseline_goal_met"] = False
            continue
        error_field = "event_mae_s" if row.dataset_id == "filters" and row.split == "validation" else "primary_score"
        usable = baseline.dropna(subset=[error_field])
        if usable.empty:
            result.loc[i, "baseline_goal_met"] = False
            continue
        control = usable.sort_values(error_field).iloc[0]
        actual = row.all_history_mae_s if error_field == "event_mae_s" else row.primary_score
        improvement = 1 - actual / control[error_field] if control[error_field] > 0 else np.nan
        result.loc[i, "baseline_error_s"] = control[error_field]
        result.loc[i, "baseline_error_name"] = control.architecture
        result.loc[i, "error_improvement_fraction"] = improvement
        nll_ok = True
        if row.dataset_id == "filters" and row.split == "validation":
            nll_control = baseline.primary_score.dropna().min()
            difference = row.primary_score - nll_control
            result.loc[i, "nll_difference"] = difference
            nll_ok = pd.notna(difference) and difference < 0
        result.loc[i, "baseline_goal_met"] = bool(row.prediction_coverage == 1 and improvement >= .2 and nll_ok)
        if row.dataset_id == "bearings":
            result.loc[i, "laboratory_goals_met"] = bool(row.primary_score <= 600 and pd.notna(row.warning_goal_met) and bool(row.warning_goal_met) and row.prediction_coverage == 1)
    result.to_csv(root / "results.csv", index=False)
    baseline_table.to_csv(root / "baselines.csv", index=False)
    atomic_write_json(root / "results.json", json.loads(result.to_json(orient="records")))
    screening = []
    for key, task in tasks.items():
        if ":main:" in key or ":confirmation:" in key or ":cv:" in key:
            continue
        screening.append({"stage": key, "dataset_id": task["dataset_id"], "run_id": task["run_id"],
                          "parent": task.get("parent"), "primary_score": task.get("primary_score"),
                          "best_epoch": task.get("best_epoch"), **task["recipe"]})
    screening = pd.DataFrame(screening)
    scores_by_id = {t["run_id"]: t.get("primary_score") for t in tasks.values()}
    screening["parent_primary_score"] = screening.parent.map(scores_by_id)
    screening["change_from_parent"] = screening.primary_score - screening.parent_primary_score
    screening.to_csv(root / "screening.csv", index=False)
    lines = [f"# Training improvement study · {manifest['study_id']}", "",
             "Real data, sequential worker, fixed equipment split and 20-measurement model windows. "
             "Test was evaluated only after freezing candidates and warning policies; the holdout was inspected in earlier work.",
             "", "## Did additional epochs help?", ""]
    epoch_rows = []
    for ds, item in manifest.get("epoch_experiment", {}).items():
        history = pd.read_csv(run_dir(ds, item["run_id"]) / "training_history.csv")
        original = counterfactual_stopping(history, "val_all_history_mae_s" if ds == "bearings" else "val_metric")
        aligned = counterfactual_stopping(history, "val_metric")
        item.update(former_metric_rule=original, aligned_metric_rule=aligned)
        epoch_rows.append({"dataset_id": ds, "former_checkpoint_primary": original["primary_score"],
                           "aligned_30_epoch_primary": aligned["primary_score"],
                           "best_100_epoch_primary": item["best_new_metric_through_100"],
                           "additional_epochs_difference": item["best_new_metric_through_100"] - aligned["primary_score"]})
        lines.append(f"- **{ds}:** former metric and 30/5 rule: stop {original['stop_epoch']}, checkpoint {original['best_epoch']}, "
                     f"primary score {original['primary_score']:.4f}. Aligned primary metric with the same 30/5 limit: "
                     f"stop {aligned['stop_epoch']}, checkpoint {aligned['best_epoch']}, score {aligned['primary_score']:.4f}. "
                     f"Continuation to 100: score {item['best_new_metric_through_100']:.4f}, checkpoint {item['best_epoch']}. "
                     "This separates the stopping-metric correction from the extra-epoch benefit on one trajectory.")
    pd.DataFrame(epoch_rows).to_csv(root / "epoch_diagnostics.csv", index=False)
    lines += ["", "## Sequential screening", "", markdown_table(screening[["stage", "primary_score", "change_from_parent", "best_epoch", "feature_recipe", "sampling"]]),
              "", "Each stage retains its parent run. The two best pilot recipes were checked using train-only equipment folds; preprocessing was fitted separately in each fold."]
    cv_rows = []
    for ds, selection in manifest["selections"].items():
        for candidate in selection["candidates"]:
            cv_rows.append({"dataset_id": ds, "features": candidate["recipe"]["feature_recipe"],
                            "mode": candidate["recipe"]["mode"], "sampling": candidate["recipe"]["sampling"],
                            "mean_score": candidate["primary_score"],
                            "fold_scores": ", ".join(f"{v:.4f}" for v in candidate["fold_scores"]),
                            "selected": candidate["source_run"] == selection["chosen"]["source_run"]})
    pd.DataFrame(cv_rows).to_csv(root / "grouped_validation.csv", index=False)
    lines += ["", "## Grouped train validation", "", markdown_table(pd.DataFrame(cv_rows)),
              "", f"Superseded implementation attempts retained in the manifest: {len(manifest.get('superseded_tasks', []))}. "
              "Only the corrected recipes contribute to these tables. Saved reference weights were re-evaluated before the main architecture matrix.",
              "", "## Main and confirmation results", ""]
    show = ["dataset_id", "split", "architecture", "seed", "role", "primary_score", "prediction_coverage", "useful_precision", "timely_recall", "warning_goal_met"]
    lines.append(markdown_table(result[show]))
    lines += ["", "Bearing scores are seconds of error in the final 30 minutes. Filter validation uses NLL in internal seconds; filter test uses official prefix-end MAE. "
              "Independent failures remain few: 3 validation bearings, 1 validation filter failure. Seeds do not increase this count.",
              "", "## Simple controls", "", markdown_table(baseline_table[["dataset_id", "split", "architecture", "primary_score", "event_mae_s", "prediction_coverage", "primary_points_available", "primary_points_expected"]]),
              "", "A partial-coverage trend control cannot be treated as a full-cohort winner. Compare its errors only on explicitly matched points. "
              "Undefined warning precision and censored unknown outcomes are not successes.",
              "", "## Laboratory targets", "", "Bearing target: MAE ≤600 seconds, timely recall ≥0.90, useful episode precision ≥0.80. "
              "Point forecasts and warnings must both meet their targets; even then this small dataset does not establish deployment readiness."]
    for ds in manifest["datasets"]:
        candidates = result[(result.dataset_id == ds) & result.split.eq("validation") & result.role.eq("new") & result.seed.eq(42) & result.rank_eligible]
        if candidates.empty:
            lines += ["", f"- **{ds}:** no validation candidate has complete, eligible predictions; no leader is declared."]
            continue
        best = candidates.sort_values("primary_score").iloc[0]
        lines += ["", f"- **{ds} validation leader:** {best.architecture}, primary score {best.primary_score:.4f}; "
                  f"warning goal met: {best.warning_goal_met}. No test-based re-selection was performed."]
    new = result[result.role.eq("new")]
    dispersion = new.groupby(["dataset_id", "split", "architecture"]).primary_score.agg(["count", "mean", "std", "min", "max"]).reset_index()
    dispersion = dispersion[dispersion["count"] > 1]
    dispersion.to_csv(root / "seed_dispersion.csv", index=False)
    lines += ["", "## Initialization stability", "", markdown_table(dispersion),
              "", "These repetitions measure initialization sensitivity on the same objects, not additional failure evidence.",
              "", "## Improvement against full-coverage controls", "",
              markdown_table(result[["dataset_id", "split", "architecture", "seed", "role", "error_improvement_fraction", "nll_difference", "baseline_goal_met"]]),
              "", "The error target is a reduction of at least 20%. Filter validation additionally requires a lower NLL; "
              "NLL differences are absolute, never percentages. Filter validation error describes the one observed failure. "
              "The adaptive pilot changes scheduling and stopping together, so it cannot isolate their individual causal effects."]
    (root / "report.md").write_text("\n".join(lines) + "\n")
