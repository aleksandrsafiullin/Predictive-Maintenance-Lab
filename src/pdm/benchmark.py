"""Explicit evaluation selection and common-cohort model comparisons."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.experiments import list_evaluations, list_runs, run_dir
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.paths import runs_root

COMPARISON_VERSION = "common_cohort_v1"
IDENTITY_FIELDS = ("dataset_id", "dataset_version", "features_hash", "units_hash", "split_hash",
                   "quality_policy_hash", "quality_records_hash", "metrics_version")


def add_survival_scores(predictions, units):
    """Join outcomes only AFTER inference; score censored and observed durations."""
    result = predictions.copy()
    if not {"weibull_scale_s", "weibull_shape"}.issubset(result.columns):
        return result
    meta = units.set_index("unit_id")
    observed = result.unit_id.map(meta.event_observed).to_numpy(float)
    end = result.unit_id.map(meta.observation_end_s).to_numpy(float)
    event = result.unit_id.map(meta.event_time_s).to_numpy(float)
    duration = np.where(observed == 1, event, end) - result.timestamp_s.to_numpy(float)
    scale = result.weibull_scale_s.to_numpy(float)
    shape = result.weibull_shape.to_numpy(float)
    valid = (duration > 0) & np.isfinite(scale) & (scale > 0) & np.isfinite(shape) & (shape > 0)
    nll = np.full(len(result), np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        log_ratio = np.log(duration[valid] / scale[valid])
        survival = np.exp(np.minimum(shape[valid] * log_ratio, 50))
        # Density in fixed internal seconds, comparable across train time scales.
        nll[valid] = survival - observed[valid] * (np.log(shape[valid]) - np.log(scale[valid]) + (shape[valid] - 1) * log_ratio)
    result["survival_nll"] = nll
    result["outcome_observed"] = observed
    result["outcome_duration_s"] = duration
    return result


def evaluation_choices(dataset_id, split_name):
    choices = []
    for run in list_runs(dataset_id):
        for ev in list_evaluations(run_dir(dataset_id, run["run_id"])):
            if ev["evaluate_mask"]["split"] == split_name and ev["has_predictions"] and ev["has_metrics"]:
                choices.append({**run, **ev, "evaluation_path": ev["path"]})
    return choices


def load_evaluation(dataset_id, run_id, eval_id):
    root = run_dir(dataset_id, run_id)
    if Path(eval_id).name != eval_id or Path(run_id).name != run_id:
        raise ValueError("Invalid experiment identifier")
    path = root / "evaluations" / eval_id
    config = read_json(path / "evaluation_config.json")
    if config.get("run_id") != run_id or config.get("eval_id") != eval_id:
        raise ValueError("Evaluation identity does not match its directory")
    if config.get("predictions_sha256") and sha256_file(path / "predictions.csv") != config["predictions_sha256"]:
        raise ValueError("Saved predictions have changed since evaluation")
    profile_path = root / "interval_profile.json"
    if config.get("interval_profile_sha256") and profile_path.exists() and sha256_file(profile_path) == config["interval_profile_sha256"]:
        profile = read_json(profile_path)
        config.setdefault("interval_calibration_ids", profile.get("calibration_ids", []))
    run = next((r for r in list_runs(dataset_id) if r["run_id"] == run_id), {})
    return {"run": run, "config": config, "metrics": read_json(path / "metrics.json"),
            "predictions": pd.read_csv(path / "predictions.csv"), "path": path}


def _equal_unit(frame, values):
    if frame.empty:
        return None
    series = pd.Series(np.asarray(values, dtype=float), index=frame.index)
    return _number(series.groupby(frame.unit_id).mean().mean())


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else None


def compare_evaluations(evaluations):
    if not evaluations:
        raise ValueError("Select at least one saved evaluation")
    configs = [e["config"] for e in evaluations]
    reference = configs[0]
    split_name = reference["evaluate_mask"]["split"]
    units = sorted(reference["evaluate_mask"].get("unit_ids") or [])
    if not units:
        raise ValueError("Evaluation has no explicit unit cohort")
    for cfg in configs:
        mismatches = [key for key in IDENTITY_FIELDS if cfg.get(key) != reference.get(key)]
        if cfg.get("evaluate_mask", {}).get("split") != split_name or sorted(cfg.get("evaluate_mask", {}).get("unit_ids") or []) != units:
            mismatches.append("evaluation cohort")
        if mismatches:
            raise ValueError("Incompatible evaluations: " + ", ".join(mismatches))
    if len({c["run_id"] for c in configs}) != len(configs):
        raise ValueError("Choose one evaluation per run")
    dataset_id = reference["dataset_id"]
    keys = ["unit_id", "timestamp_s"]
    frames, eligible_sets, excluded_sets = [], [], []
    for ev in evaluations:
        frame = ev["predictions"].copy()
        if frame.duplicated(keys).any() or set(frame.unit_id) != set(units):
            raise ValueError("Evaluation contains duplicate timestamps or an incomplete unit cohort")
        frame = frame.set_index(keys, drop=False)
        reason = frame.get("valid_history_reason", pd.Series("", index=frame.index)).fillna("")
        eligible = ~reason.isin(["insufficient_length", "gap_in_window"])
        if dataset_id == "filters" and split_name == "validation":
            eligible &= frame.get("outcome_duration_s", pd.Series(np.nan, index=frame.index)) > 0
        elif dataset_id == "bearings":
            eligible &= frame.actual_rul_s > 0
        elif dataset_id == "filters":
            # Official labels are attached at all prefix points; primary is last.
            eligible &= np.isfinite(frame.actual_rul_s)
        eligible_sets.append(set(frame.index[eligible]))
        excluded_sets.append(set(frame.index[~eligible]))
        frames.append(frame)
    # A missing CSV row is missing coverage, not permission to remove a hard
    # measurement from every other model's score. Only explicit history/outcome
    # ineligibility reduces the common scoring clock.
    common = set.union(*eligible_sets) - set.union(*excluded_sets)
    if not common or {k[0] for k in common} != set(units):
        raise ValueError("No shared eligible prediction history for every evaluation unit")
    index = pd.MultiIndex.from_tuples(sorted(common), names=keys)
    # Missing predictions never remove a difficult point from the reference mask.
    truth_ref = pd.Series(np.nan, index=index)
    for frame in frames:
        actual = frame.reindex(index).get("actual_rul_s", pd.Series(np.nan, index=index))
        overlap = truth_ref.notna() & actual.notna()
        if not np.allclose(truth_ref[overlap], actual[overlap]):
            raise ValueError("Actual RUL differs between evaluations")
        truth_ref = truth_ref.combine_first(actual)
    expected_ends = pd.concat([f[["unit_id", "timestamp_s"]].reset_index(drop=True) for f in frames]).groupby("unit_id").timestamp_s.max()
    rows, per_unit, charts = [], [], []
    for ev, frame in zip(evaluations, frames, strict=True):
        part = frame.reindex(index).drop(columns=keys).reset_index()
        present = part.set_index(keys).index.isin(frame.index)
        if not np.allclose(part.loc[present].get("actual_rul_s", np.nan), truth_ref.to_numpy()[present], equal_nan=True):
            raise ValueError("Actual RUL differs between evaluations")
        part["actual_rul_s"] = truth_ref.to_numpy()
        finite = np.isfinite(part.predicted_rul_s.to_numpy(float)) & (part.predicted_rul_s >= 0)
        coverage = float(finite.mean())
        good = part.loc[finite].copy()
        error = good.predicted_rul_s - good.get("actual_rul_s", np.nan)
        good["absolute_error_s"] = error.abs()
        near = good[(good.get("actual_rul_s", pd.Series(np.nan, index=good.index)) > 0) & (good.get("actual_rul_s", pd.Series(np.nan, index=good.index)) <= 1800)]
        mae = _equal_unit(good, error.abs())
        near_mae = _equal_unit(near, near.absolute_error_s)
        score_survival = dataset_id == "filters" and split_name == "validation"
        nll = _equal_unit(good, good.survival_nll) if score_survival and "survival_nll" in good else None
        ends = good.sort_values("timestamp_s").groupby("unit_id").tail(1)
        # Last eligible point must be the actual final recorded prefix point.
        end_ok = all(uid in set(ends.unit_id) and float(ends.loc[ends.unit_id == uid, "timestamp_s"].iloc[0]) == float(expected_ends[uid]) for uid in units)
        prefix_mae = _equal_unit(ends, ends.absolute_error_s) if end_ok else None
        primary = near_mae if dataset_id == "bearings" else nll if split_name == "validation" else prefix_mae
        run = ev["run"]
        reasons = []
        if coverage < 1:
            reasons.append("incomplete predictions")
        if not reference.get("quality_policy_hash"):
            reasons.append("legacy data quality policy")
        if run.get("smoke"):
            reasons.append("smoke")
        if run.get("is_synthetic") or run.get("graph_mode") == "synthetic_fixture":
            reasons.append("synthetic fixture")
        if primary is None:
            reasons.append("primary metric unavailable")
        if dataset_id == "bearings" and near.unit_id.nunique() != len(units):
            reasons.append("incomplete near-event cohort")
        if dataset_id == "filters" and split_name == "validation" and ("survival_nll" not in part or not np.isfinite(part.survival_nll).all()):
            reasons.append("incomplete survival likelihood")
        interval = good.dropna(subset=["lower_rul_s", "upper_rul_s", "actual_rul_s"]) if {"lower_rul_s", "upper_rul_s", "actual_rul_s"}.issubset(good) else good.iloc[:0]
        alert = ev["metrics"].get("alerts") or {}
        methods = sorted(set(good.get("interval_method", pd.Series(dtype=str)).dropna().astype(str)))
        calibration_ids = ev["config"].get("interval_calibration_ids") or []
        interval_origin = ("empirical calibration" if ev["config"].get("interval_profile_sha256")
                           else "; ".join(methods) or "unavailable")
        row = {"run_id": ev["config"]["run_id"], "eval_id": ev["config"]["eval_id"],
               "model": run.get("architecture"), "nodes": run.get("n_nodes"),
               "state_mode": run.get("state_mode") or "window_reset", "history_length": run.get("history_length", 20),
               "primary_score": primary, "near_30m_mae_s": near_mae, "all_history_mae_s": mae,
               "overestimation_s": _equal_unit(good, np.maximum(error, 0)), "survival_nll": nll,
               "prefix_end_mae_s": prefix_mae, "prediction_coverage": coverage,
               "common_points": len(part), "units": len(units), "rank_eligible": not reasons,
               "observed_events": (int(frame.drop_duplicates("unit_id").outcome_observed.eq(1).sum())
                                   if "outcome_observed" in frame else ev["metrics"].get("n_observed_events", alert.get("n_units_with_event"))),
               "interval_origin": interval_origin, "interval_scored_points": len(interval),
               "interval_evaluation": ("on calibration units" if set(units) & set(calibration_ids)
                                        else "outside calibration units" if calibration_ids else "uncalibrated distribution" if methods else "unavailable"),
               "reason": "; ".join(reasons), "interval_coverage": _equal_unit(interval, (interval.actual_rul_s >= interval.lower_rul_s) & (interval.actual_rul_s <= interval.upper_rul_s)) if len(interval) else None,
               "interval_width_s": _equal_unit(interval, interval.upper_rul_s - interval.lower_rul_s) if len(interval) else None,
               "alerts": json.dumps(alert, sort_keys=True)}
        row.update({"alerts_" + k: alert.get(k) for k in ("timely", "late", "miss", "too_early", "mean_lead_time_s", "fraction_time_in_warning")})
        row["timely_warning_units"] = alert.get("n_units_timely")
        row["scored_warning_units"] = alert.get("n_units_scored")
        rows.append(row)
        for uid in units:
            group = good[good.unit_id == uid]
            expected = part[part.unit_id == uid]
            end = group[group.timestamp_s == expected_ends[uid]]
            per_unit.append({"run_id": row["run_id"], "unit_id": uid, "mae_s": _number(group.absolute_error_s.mean()),
                             "near_30m_mae_s": _number(group.loc[group.actual_rul_s.between(0, 1800, inclusive="right"), "absolute_error_s"].mean()),
                             "prefix_end_mae_s": _number(end.absolute_error_s.mean()) if dataset_id == "filters" and split_name == "test" else None,
                             "survival_nll": _number(group.survival_nll.mean()) if score_survival and "survival_nll" in group else None,
                             "points": len(group), "eligible_points": len(expected), "prediction_coverage": len(group) / len(expected)})
        part["run_id"] = row["run_id"]
        part["absolute_error_s"] = (part.predicted_rul_s - part.get("actual_rul_s", np.nan)).abs()
        charts.append(part)
    table = pd.DataFrame(rows)
    table["rank"] = table.primary_score.where(table.rank_eligible).rank(method="min")
    table = table.sort_values(["rank", "run_id"], na_position="last").reset_index(drop=True)
    # Alert outcomes are comparable only with the same frozen threshold settings.
    policy_keys = ("H_trigger", "confirmation_count", "minimum_action_lead_time", "reset_factor", "max_useful_horizon_s")
    policies = [tuple((c.get("alert_policy") or {}).get(k) for k in policy_keys) for c in configs]
    return {"version": COMPARISON_VERSION, "dataset_id": dataset_id, "split": split_name,
            "table": table, "per_unit": pd.DataFrame(per_unit), "predictions": pd.concat(charts, ignore_index=True),
            "selections": [{"run_id": c["run_id"], "eval_id": c["eval_id"]} for c in configs],
            "alert_policies_match": len(set(policies)) == 1,
            "evaluation_status": "validation_selection" if split_name == "validation" else "exploratory_reused_holdout"}


def save_comparison(result, *, directory=None):
    root = Path(directory) if directory else runs_root() / "comparisons" / uuid.uuid4().hex[:12]
    root.mkdir(parents=True, exist_ok=False)
    for key in ("table", "per_unit", "predictions"):
        result[key].to_csv(root / f"{key}.csv", index=False)
    doc = {k: v for k, v in result.items() if k not in {"table", "per_unit", "predictions"}}
    doc["table"] = json.loads(result["table"].to_json(orient="records"))
    atomic_write_json(root / "comparison.json", doc)
    return root
