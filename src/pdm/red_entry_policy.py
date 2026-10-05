"""Physical-time confirmation, hysteresis and cooldown for RED alerts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.red_entry_protocol import validate_horizons


def validate_policy(policy, horizons_s):
    grid = validate_horizons(horizons_s)
    out = {"horizon_s": grid[-1], "threshold": 0.5, "release_threshold": 0.4,
           "confirmation_s": 0.0, "cooldown_s": 0.0, "max_gap_s": None, **(policy or {})}
    if out["horizon_s"] not in grid:
        raise ValueError("Policy horizon must be a supported horizon")
    if not 0 <= out["release_threshold"] <= out["threshold"] <= 1:
        raise ValueError("Policy thresholds require release <= activation in [0,1]")
    for key in ("confirmation_s", "cooldown_s"):
        if not np.isfinite(out[key]) or out[key] < 0:
            raise ValueError("Policy durations must be finite nonnegative seconds")
    if out["max_gap_s"] is not None and (not np.isfinite(out["max_gap_s"]) or out["max_gap_s"] <= 0):
        raise ValueError("Maximum gap must be positive seconds")
    return out


def reliable_origin(row):
    def known_true(name, default=True):
        value = row.get(name, default)
        return pd.notna(value) and value in (True, 1)
    return known_true("at_risk") and row.get("risk_status", "at_risk") == "at_risk" \
        and known_true("quality_ok") and known_true("context_known")


def alert_episodes(origins, probabilities, horizons_s, *, policy=None):
    """Each confirmed episode carries its confirming origin's immutable horizon.

    Unobserved gaps and unknown context terminate pending and confirmed alerts;
    elapsed gaps never count towards confirmation or alert duration.
    """
    frame = pd.DataFrame(origins).reset_index(drop=True)
    if not {"unit_id", "timestamp_s"}.issubset(frame):
        raise ValueError("Policy origins need unit identity and timestamp")
    if frame.unit_id.isna().any() or not np.isfinite(frame.timestamp_s.to_numpy(float)).all():
        raise ValueError("Policy origins require known units and finite times")
    p = np.asarray(probabilities, float)
    grid = validate_horizons(horizons_s)
    cfg = validate_policy(policy, grid)
    if p.shape != (len(frame), len(grid)) or np.isinf(p).any() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must align and be in [0,1] or unknown NaN")
    column = grid.index(cfg["horizon_s"])
    episodes = []
    keys = ["unit_id", "episode_id"] if "episode_id" in frame else ["unit_id"]
    for _, group in frame.groupby(keys, sort=True):
        group = group.sort_values("timestamp_s")
        if group.timestamp_s.duplicated().any():
            raise ValueError("Duplicate policy timestamps")
        active, pending, previous, cooldown_end = None, None, None, -np.inf
        for index, row in group.iterrows():
            t = float(row.timestamp_s)
            gap = (pd.notna(row.get("gap_before")) and bool(row.get("gap_before"))) or (previous is not None and
                    cfg["max_gap_s"] is not None and t - previous > cfg["max_gap_s"])
            reliable = reliable_origin(row) and np.isfinite(p[index, column])
            if gap or not reliable:
                if active is not None:
                    active.update(end_s=previous, termination="gap" if gap else "unknown")
                    cooldown_end = t + cfg["cooldown_s"]
                active, pending = None, None
                previous = t
                if not reliable:
                    continue
            probability = p[index, column]
            if active is not None:
                if probability < cfg["release_threshold"]:
                    active.update(end_s=t, termination="released")
                    active, pending = None, None
                    cooldown_end = t + cfg["cooldown_s"]
                else:
                    active["end_s"] = t
            elif t >= cooldown_end:
                if probability >= cfg["threshold"]:
                    pending = t if pending is None else pending
                    if t - pending >= cfg["confirmation_s"]:
                        active = {"unit_id": str(row.unit_id),
                                  "physical_unit_id": str(row.get("physical_unit_id", row.unit_id)),
                                  "episode_id": str(row.get("episode_id", row.unit_id)),
                                  "start_s": t, "pending_start_s": pending, "end_s": t,
                                  "origin_index": int(index), "horizon_s": cfg["horizon_s"],
                                  "zone": row.get("zone", "unknown"), "termination": "observation_end"}
                        episodes.append(active)
                        pending = None
                else:
                    pending = None
            previous = t
    return episodes


def select_alert_policy(origins, events, probabilities, horizons_s, *, requirements=None,
                        selection_part="validation", candidate_policies=None,
                        laboratory_elapsed_assumption=False):
    """Select a preregistered policy on development equipment only.

    The return value is a JSON-ready frozen policy artifact. Both operational
    budgets constrain candidates; useful recall never justifies exceeding one.
    Admission on independent holdout remains a separate evaluation step.
    """
    from pdm.red_entry_evaluation import evaluate_red_entry
    from pdm.red_entry_protocol import canonical_json_hash, load_protocol

    frame, event_frame = pd.DataFrame(origins), pd.DataFrame(events)
    if selection_part != "validation":
        raise ValueError("Alert policy selection is restricted to development validation")
    for table in (frame, event_frame):
        if len(table) and ("split" not in table or set(table.split.astype(str)) != {"validation"}):
            raise ValueError("Policy selection requires explicit validation-only origins and events")
    grid = validate_horizons(horizons_s)
    default = validate_policy({"threshold": 0.5}, grid)
    configs = candidate_policies if candidate_policies is not None else [
        {"threshold": threshold, "release_threshold": max(0.0, threshold-0.1),
         "confirmation_s": confirmation, "cooldown_s": cooldown}
        for threshold in (0.25, 0.5, 0.75)
        for confirmation in (0.0, grid[0])
        for cooldown in (0.0, grid[0])]
    configs = [validate_policy(config, grid) for config in configs]
    if not configs:
        raise ValueError("Policy candidate list must be preregistered and nonempty")
    required = load_protocol()["operational_requirements"]
    required.update(requirements or {})
    artifact = {"version": "red_entry_alert_policy_selection_v1", "selection_part": selection_part,
                "candidate_hash": canonical_json_hash(configs), "preregistered_candidates": configs,
                "status": "requirements_unset", "config": default, "frozen": True,
                "selection_quality_claim": False, "candidate_metrics": [],
                "reason": "explicit_exploratory_default_no_budget_optimization"}
    if any(value is None for value in required.values()):
        return artifact
    evaluations = []
    feasible = []
    for config in configs:
        report = evaluate_red_entry(frame, event_frame, probabilities, grid, policy=config,
                                    requirements=required,
                                    laboratory_elapsed_assumption=laboratory_elapsed_assumption)
        metrics = report["metrics"]
        rate = metrics["false_alert_episodes_per_100_operating_hours"]
        fraction = metrics["alert_time_fraction"]
        recall = metrics["useful_event_recall"]
        admissible = (report["metric_scope"]["policy_prediction_coverage"] == 1.0 and rate is not None and fraction is not None and recall is not None and
                      rate <= required["max_false_alert_episodes_per_100_operating_hours"] and
                      fraction <= required["max_alert_time_fraction"])
        evaluations.append({"config": config, "metrics": metrics, "budget_feasible": admissible})
        if admissible:
            feasible.append((recall, -rate, -fraction, -len(evaluations), config))
    artifact["candidate_metrics"] = evaluations
    if not feasible:
        artifact.update(status="not_passed", reason="no_candidate_meets_both_alert_budgets")
        return artifact
    selected = max(feasible, key=lambda item: item[:4])[-1]
    artifact.update(status="selected_development", config=selected,
                    reason="maximum_useful_recall_subject_to_both_budgets",
                    selection_quality_claim=False)
    return artifact
