"""Evaluator-only outcome joins; preserve the clock and every refusal."""
from __future__ import annotations

import json
import time
import uuid

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_pinball_loss
from threadpoolctl import threadpool_limits

from pdm.data.prepare import load_processed
from pdm.io_util import atomic_write_json, sha256_file
from pdm.losses import weibull_nll_seconds
from pdm.monitoring.bundle import bundle_root, code_identity, load_bundle, runtime_args
from pdm.monitoring.runtime import replay_monitoring
from pdm.monitoring.signal_forecast import supervised_targets


def signal_metrics(forecasts, labels, metadata, horizons):
    per_unit = []
    for j, h in enumerate(horizons):
        truth = metadata[["unit_id", "issued_at"]].copy()
        truth["actual"] = labels[:, j]
        issued = forecasts[forecasts.horizon.eq(h)]
        joined = truth.merge(issued, on=["unit_id", "issued_at"], how="left", validate="one_to_one")
        for uid, group in joined.groupby("unit_id"):
            target = np.isfinite(group.actual)
            valid = target & np.isfinite(pd.to_numeric(group.point, errors="coerce"))
            measured = group.loc[valid]
            error = measured.point - measured.actual
            interval = measured.lower.notna() & measured.upper.notna()
            band = measured.loc[interval]
            pinball = {}
            for q, col in ((.05, "lower"), (.5, "point"), (.95, "upper")):
                supported = measured[col].notna()
                pinball[f"pinball_{q}"] = float(mean_pinball_loss(measured.loc[supported, "actual"], measured.loc[supported, col], alpha=q)) if supported.any() else None
            per_unit.append({"unit_id": uid, "horizon": h, "expected_targets": int(target.sum()),
                 "available_predictions": int(valid.sum()), "prediction_coverage": float(valid.sum() / target.sum()) if target.any() else None,
                 "mae": float(error.abs().mean()) if len(error) else None,
                 "rmse": float(np.sqrt((error ** 2).mean())) if len(error) else None,
                 "interval_coverage": float(((band.actual >= band.lower) & (band.actual <= band.upper)).mean()) if len(band) else None,
                 "interval_width": float((band.upper - band.lower).mean()) if len(band) else None,
                 **pinball})
    frame = pd.DataFrame(per_unit)
    summaries = []
    for h, g in frame.groupby("horizon"):
        total, available = int(g.expected_targets.sum()), int(g.available_predictions.sum())
        summaries.append({"horizon": h, "independent_units": int(g.unit_id.nunique()),
            "expected_targets": total, "available_predictions": available,
            "prediction_coverage": available / total if total else None,
            "unit_balanced_mae": float(g.mae.mean()) if g.mae.notna().any() else None,
            "pooled_mae": float((g.mae.fillna(0) * g.available_predictions).sum() / available) if available else None,
            "unit_balanced_rmse": float(g.rmse.mean()) if g.rmse.notna().any() else None,
            "interval_coverage": float(g.interval_coverage.mean()) if g.interval_coverage.notna().any() else None,
            "interval_width": float(g.interval_width.mean()) if g.interval_width.notna().any() else None,
            **{col: float(g[col].mean()) if g[col].notna().any() else None for col in ("pinball_0.05", "pinball_0.5", "pinball_0.95")}})
    return pd.DataFrame(summaries), frame


def score_episodes(episodes, units, *, minimum_lead=None, maximum_horizon=None):
    outcomes = []
    matched = set()
    for episode in sorted(episodes, key=lambda e: e["confirmed_at"]):
        uid = episode["unit_id"]
        meta = units.set_index("unit_id").loc[uid]
        category, lead = "unknown_diagnostic_outcome", None
        if episode["kind"] == "prognostic":
            end, t = float(meta.observation_end_s), episode["confirmed_at"]
            if "planned" in str(meta.get("event_source", "")) or "preventive" in str(meta.get("event_source", "")):
                category = "unknown_informative_intervention"
            elif meta.event_observed and np.isfinite(meta.event_time_s):
                lead = float(meta.event_time_s - t)
                if uid in matched:
                    category = "repeated"
                elif minimum_lead is None:
                    category = "timeliness_not_configured"
                elif lead < minimum_lead:
                    category = "late"
                elif maximum_horizon is not None and lead > maximum_horizon:
                    category = "early"
                else:
                    category = "timely"
                    matched.add(uid)
            elif end - t >= float(episode.get("issued_horizon") or float("inf")):
                category = "false_with_sufficient_followup"
            else:
                category = "unknown_insufficient_followup"
        outcomes.append({"episode_id": episode["episode_id"], "unit_id": uid, "category": category, "lead_time": lead})
    n_events = int(units.event_observed.sum())
    return outcomes, {"observed_events": n_events, "timely_events": len(matched),
        "timely_recall": len(matched) / n_events if n_events and minimum_lead is not None else None,
        "unmatched_events": n_events - len(matched), "timeliness_status": "not_configured" if minimum_lead is None else "configured"}


@threadpool_limits.wrap(limits=1)
def evaluate_monitoring(bundle_id, split_name, *, should_stop=None, log=None):
    torch.set_num_threads(1)
    if split_name not in {"validation", "test"}:
        raise ValueError("Monitoring evaluation is validation or frozen exploratory test")
    payload, predictor, sensor = load_bundle(bundle_id)
    profile = payload["profile"]
    data = load_processed(profile["dataset_id"], payload["dataset_version"])
    ids = data["split"][split_name]
    features = data["features"][data["features"].unit_id.isin(ids)]
    units = data["units"][data["units"].unit_id.isin(ids)]
    # Clock fixed before asking either model, including warmup and unknown states.
    clock = features[["unit_id", "timestamp_s"]].copy()
    args = runtime_args(payload, predictor, sensor)
    records, forecasts, episodes, event_rows, unit_rows = [], [], [], [], []
    start = time.monotonic()
    eval_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]
    root = bundle_root(bundle_id) / "evaluations" / eval_id
    root.mkdir(parents=True)
    clock.to_parquet(root / "evaluation_clock.parquet", index=False)
    for uid, group in features.groupby("unit_id", sort=True):
        if should_stop and should_stop():
            atomic_write_json(root / "status.json", {"status": "cancelled", "evaluated_units": len(unit_rows)})
            raise InterruptedError("Stopped monitoring evaluation; incomplete artifact retained")
        rows, state = replay_monitoring(group.sort_values("timestamp_s"), should_stop=should_stop, **args)
        meta = units.set_index("unit_id").loc[uid]
        local = []
        for result in rows:
            condition = result["condition"]
            flat = {"unit_id": uid, "as_of": result["as_of"], "bundle_id": bundle_id,
                    "segment_id": result["segment_id"], "time_basis": result["time_basis"],
                    "health_state": condition["health_state"], "display_zone": condition["display_zone"],
                    "data_quality_status": result["data_quality_status"], "model_applicability": result["model_applicability"],
                    "forecast_status": result["forecast_status"], "critical_latch": condition["critical_latch"],
                    "reason_codes": ";".join(condition["reason_codes"]), "alert_episode_id": condition["alert_episode_id"],
                    "used_measurements": result["history"]["used_measurements"],
                    "available_measurements": result["history"]["available_measurements"],
                    "history_duration": result["history"]["duration"], "planning_margin": condition["timing"]["planning_margin"]}
            records.append(flat)
            local.append(flat)
            forecasts.extend(result["signal_forecasts"])
            event = result["event_forecast"]
            t = result["as_of"]
            actual = float(meta.event_time_s - t) if meta.event_observed and np.isfinite(meta.event_time_s) and t < meta.event_time_s else None
            distribution = event.get("distribution") or {}
            event_rows.append({"unit_id": uid, "as_of": t, "point": event["point"], "actual_remaining": actual,
                    "duration": max(0., float(meta.observation_end_s) - t), "event_observed": int(meta.event_observed),
                    "scale": distribution.get("scale"), "shape": distribution.get("shape"),
                    "status": event["status"], "reason": event["reason"], "event_definition_id": event["event_definition_id"],
                    "time_basis": profile["time_basis"], "available_measurements": result["history"]["available_measurements"]})
        if state:
            episodes.extend(state["episodes"])
        table = pd.DataFrame(local)
        unit_rows.append({"unit_id": uid, "measurements": len(table),
            "assessment_availability": float(table.health_state.ne("unknown").mean()),
            "signal_availability": float(table.forecast_status.ne("unavailable").mean()),
            "unknown_fraction": float(table.health_state.eq("unknown").mean()),
            "alarm_fraction": float(table.display_zone.isin(["yellow", "red"]).mean()),
            "observed_event": int(meta.event_observed),
            "endpoint": profile["event_definition_id"]})
        with (root / "observations.jsonl").open("a") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
        if log:
            log(f"Monitoring {uid}: {len(rows)} issued assessments")
    states = pd.DataFrame(records)
    signals = pd.DataFrame(forecasts)
    states.to_parquet(root / "state_history.parquet", index=False)
    signals.to_parquet(root / "signal_forecasts.parquet", index=False)
    pd.DataFrame(episodes, columns=list(episodes[0]) if episodes else ["episode_id", "unit_id", "confirmed_at", "kind"]).to_parquet(root / "alert_episodes.parquet", index=False)
    pd.DataFrame(unit_rows).to_csv(root / "per_unit_results.csv", index=False)
    event_table = pd.DataFrame(event_rows)
    event_table["survival_nll"] = np.nan
    admitted = event_table.scale.notna() & event_table["shape"].notna() & event_table.duration.gt(0)
    if admitted.any():
        subset = event_table.loc[admitted]
        event_table.loc[admitted, "survival_nll"] = weibull_nll_seconds(
            torch.tensor(subset.duration.to_numpy()), torch.tensor(subset.event_observed.to_numpy()),
            torch.tensor(subset.scale.to_numpy()), torch.tensor(subset["shape"].to_numpy())).numpy()
    event_table.to_csv(root / "event_forecasts.csv", index=False)
    event_metrics, event_per_unit = [], []
    for mode in ("end_to_end", "common_clock_60"):
        selected = event_table if mode == "end_to_end" else event_table[event_table.available_measurements.ge(60)]
        valid = selected.point.notna() & selected.actual_remaining.notna()
        scored = selected.loc[valid].copy()
        scored["absolute_error"] = (scored.point - scored.actual_remaining).abs()
        for uid, unit in selected.groupby("unit_id"):
            observed = unit.point.notna() & unit.actual_remaining.notna()
            event_per_unit.append({"unit_id": uid, "clock": mode, "expected": len(unit),
                "available": int(unit.point.notna().sum()), "mae_observed": float((unit.loc[observed, "point"] - unit.loc[observed, "actual_remaining"]).abs().mean()) if observed.any() else None,
                "survival_nll": float(unit.survival_nll.mean()) if unit.survival_nll.notna().any() else None})
        event_metrics.append({"clock": mode, "expected": len(selected), "available": int(selected.point.notna().sum()),
            "coverage": float(selected.point.notna().mean()) if len(selected) else None,
            "unit_balanced_mae": float(scored.groupby("unit_id").absolute_error.mean().mean()) if len(scored) else None,
            "unit_balanced_survival_nll": float(selected.groupby("unit_id").survival_nll.mean().mean()) if selected.survival_nll.notna().any() else None,
            "endpoint": profile["event_definition_id"], "time_basis": profile["time_basis"]})
    pd.DataFrame(event_metrics).to_csv(root / "event_results.csv", index=False)
    pd.DataFrame(event_per_unit).to_csv(root / "event_per_unit_results.csv", index=False)
    if sensor is not None:
        cfg = sensor["config"]
        _, labels, metadata = supervised_targets(features, profile, cfg["horizons"], tolerance=cfg["tolerance"],
                             history_min=20, history_max=60)
        summary, per_unit = signal_metrics(signals, labels, metadata, cfg["horizons"])
        summary.to_csv(root / "signal_results.csv", index=False)
        per_unit.to_csv(root / "signal_per_unit.csv", index=False)
    episode_outcomes, event_summary = score_episodes(episodes, units)
    pd.DataFrame(episode_outcomes).to_csv(root / "episode_outcomes.csv", index=False)
    durations = states.groupby("unit_id").as_of.diff().fillna(0).clip(lower=0)
    # Time in a state uses the PREVIOUS issued state over each inter-measurement interval.
    time_states = states.groupby("unit_id").display_zone.shift().fillna("gray")
    monitoring = {"expected_measurements": len(clock), "recorded_assessments": len(states),
            "independent_units": len(ids), "time_basis": profile["time_basis"],
            "state_durations": durations.groupby(time_states).sum().to_dict(),
            "episodes": len(episodes), "diagnostic_episodes": sum(e["kind"] == "diagnostic" for e in episodes),
            "unknown_fraction": float(states.health_state.eq("unknown").mean()),
            "assessment_availability": float(states.health_state.ne("unknown").mean()),
            **event_summary}
    atomic_write_json(root / "monitoring_results.json", monitoring)
    (root / "feedback.jsonl").touch()
    result = {"eval_id": eval_id, "bundle_id": bundle_id, "split": split_name,
              "execution_source_worktree_hash": code_identity(),
              "bundle_source_worktree_hash": payload["source_worktree_hash"],
              "status": "completed", "evaluation_status": "exploratory_reused_holdout" if split_name == "test" else "validation_diagnostics",
              "processed_measurements": len(clock), "elapsed_seconds": time.monotonic() - start,
              "artifact_hashes": {p.name: sha256_file(p) for p in root.iterdir() if p.is_file()},
              "directory": str(root)}
    atomic_write_json(root / "evaluation.json", result)
    return result
