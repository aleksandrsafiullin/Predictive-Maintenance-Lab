"""RED event/alert evaluation with explicit censoring and evidence boundaries.

Brier is a unit-equal known-outcome fallback, NOT IPCW survival Brier. Unknown
follow-up is excluded and support is reported. Informative maintenance censoring
is not repaired by this exclusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.red_entry_policy import alert_episodes, reliable_origin, validate_policy
from pdm.red_entry_protocol import load_protocol, validate_horizons


def _outcome(row, horizon):
    if not reliable_origin(row):
        return None
    duration = float(row.get("followup_duration_s", 0))
    event = bool(row.get("event_observed", False))
    if event and duration <= horizon:
        return 1
    if duration >= horizon:
        return 0
    return None


def _exposure(frame, episodes, cfg, lab):
    per_unit = {}
    keys = ["unit_id", "episode_id"] if "episode_id" in frame else ["unit_id"]
    for _, group in frame.groupby(keys):
        group = group.sort_values("timestamp_s")
        for i in range(1, len(group)):
            left, right = group.iloc[i-1], group.iloc[i]
            uid = str(left.get("physical_unit_id", left.unit_id))
            totals = per_unit.setdefault(uid, {"operating_s": 0.0, "alert_s": 0.0})
            dt = float(right.timestamp_s-left.timestamp_s)
            if dt <= 0 or (pd.notna(right.get("gap_before")) and bool(right.get("gap_before"))) or (
                cfg["max_gap_s"] is not None and dt > cfg["max_gap_s"]):
                continue
            if not reliable_origin(left.drop(labels=["context_known"], errors="ignore")) or not (reliable_origin(right.drop(labels=["context_known"], errors="ignore")) or
                                                  right.get("risk_status") == "event_observed"):
                continue
            running = (pd.notna(left.get("is_running")) and pd.notna(right.get("is_running"))
                       and bool(left.get("is_running")) and bool(right.get("is_running")))
            explicit_stop = any(pd.notna(row.get("is_running")) and not bool(row.get("is_running"))
                                for row in (left, right))
            if not running and (not lab or explicit_stop):
                continue
            totals["operating_s"] += dt
            for episode in episodes:
                if episode["unit_id"] != str(left.unit_id) or episode["episode_id"] != str(
                        left.get("episode_id", left.unit_id)):
                    continue
                totals["alert_s"] += max(0.0, min(float(right.timestamp_s), episode["end_s"]) -
                                         max(float(left.timestamp_s), episode["start_s"]))
    return per_unit


def evaluate_red_entry(origins, events, probabilities, horizons_s, *, policy=None,
                       requirements=None, provenance=None, model_frozen=False,
                       policy_frozen=False, laboratory_elapsed_assumption=False, validation_evidence=None):
    frame, event_frame = pd.DataFrame(origins).reset_index(drop=True), pd.DataFrame(events)
    grid, p = validate_horizons(horizons_s), np.asarray(probabilities, float)
    cfg = validate_policy(policy, grid)
    episodes = alert_episodes(frame, p, grid, policy=cfg)
    if (np.diff(p, axis=1) < -1e-12).any():
        raise ValueError("RED cumulative probabilities must be monotone")
    required = load_protocol()["operational_requirements"]
    required.update(requirements or {})
    lead_min = required.get("minimum_action_lead_s")
    # Only confirmed inactive histories leave the potential policy-origin scope.
    # Uncertain quality, rules or event history remain coverage obligations.
    inactive = frame.get("risk_status", pd.Series("at_risk", index=frame.index)).isin(
        ["event_observed", "post_event", "already_red"]).to_numpy()
    eligible = ~inactive
    reliable = np.array([reliable_origin(row) for _, row in frame.iterrows()], dtype=bool)
    qualified = np.isfinite(p) & (eligible & reliable)[:, None]
    eligible_count = int(eligible.sum())
    excluded_count = int(inactive.sum())
    policy_column = grid.index(cfg["horizon_s"])
    policy_available_count = int(qualified[:, policy_column].sum())
    policy_coverage = policy_available_count / eligible_count if eligible_count else 0.0
    event_results = []
    for _, event in event_frame.iterrows():
        if not bool(event.get("first_event_verified", False)) or pd.isna(event.get("first_red_timestamp_s")):
            continue
        t = float(event.first_red_timestamp_s)
        episode_id = str(event.get("episode_id", event.unit_id))
        unit_rows = frame[(frame.unit_id.astype(str) == str(event.unit_id)) &
                          (frame.get("episode_id", frame.unit_id).astype(str) == episode_id)]
        candidates = [a for a in episodes if a["unit_id"] == str(event.unit_id)
                      and a["episode_id"] == episode_id and a["start_s"] < t]
        reachable = any(reliable_origin(row) and 0 < t-float(row.timestamp_s) <= cfg["horizon_s"]
                        and (lead_min is None or t-float(row.timestamp_s) >= lead_min)
                        for _, row in unit_rows.iterrows())
        first = candidates[0] if candidates else None
        lead = t-first["start_s"] if first else None
        unknown_prediction = any(
            eligible[index]
            and 0 < t-float(row.timestamp_s) <= cfg["horizon_s"]
            and not qualified[index, policy_column]
            for index, row in unit_rows.iterrows())
        if unknown_prediction:
            status = "unknown"
        elif not reachable:
            status = "unreachable"
        elif first is None:
            status = "missed"
        elif lead > first["horizon_s"]:
            status = "early"
        elif lead_min is None:
            status = "action_lead_requirement_unset"
        elif lead < lead_min:
            status = "late"
        else:
            status = "useful"
        green_reachable = any(reliable_origin(row) and str(row.get("zone")).lower() == "green"
                              and 0 < t-float(row.timestamp_s) <= cfg["horizon_s"]
                              and (lead_min is None or t-float(row.timestamp_s) >= lead_min)
                              for _, row in unit_rows.iterrows())
        event_results.append({"unit_id": str(event.unit_id),
                              "physical_unit_id": str(event.get("physical_unit_id", event.unit_id)),
                              "episode_id": episode_id, "status": status, "reachable": reachable,
                              "lead_s": lead, "confirmed_episode_count": len(candidates),
                              "green_reachable": green_reachable,
                              "green_useful": status == "useful" and str(first["zone"]).lower() == "green"})
    for episode in episodes:
        outcome = _outcome(frame.iloc[episode["origin_index"]], episode["horizon_s"])
        episode["horizon_outcome"] = "unknown" if outcome is None else "event" if outcome else "false"
    exposure = _exposure(frame, episodes, cfg, laboratory_elapsed_assumption)
    exposure_available = laboratory_elapsed_assumption or (
        "is_running" in frame and frame.is_running.notna().any())
    operating = sum(v["operating_s"] for v in exposure.values()) if exposure_available else None
    alert = sum(v["alert_s"] for v in exposure.values())
    false = sum(a["horizon_outcome"] == "false" for a in episodes)
    useful = sum(e["status"] == "useful" for e in event_results)
    reachable_count = sum(e["reachable"] for e in event_results)
    green_count = sum(e["green_reachable"] for e in event_results)
    unknown_event_count = sum(e["status"] == "unknown" for e in event_results)
    recall_reasons = []
    if lead_min is None:
        recall_reasons.append("minimum_action_lead_s_unset")
    if unknown_event_count:
        recall_reasons.append("unknown_event_outcomes")
    if not event_results:
        recall_reasons.append("no_verified_events")
    # These are complete-event recalls, never a score on only evaluable events.
    recall_available = not recall_reasons
    known_event_count = len(event_results) - unknown_event_count if lead_min is not None else 0
    brier = []
    for column, horizon in enumerate(grid):
        unit_losses, support, unknown = {}, 0, 0
        for index, row in frame.iterrows():
            if inactive[index]:
                continue
            outcome = _outcome(row, horizon)
            if outcome is None or not np.isfinite(p[index, column]):
                unknown += 1
                continue
            uid = str(row.get("physical_unit_id", row.unit_id))
            unit_losses.setdefault(uid, []).append(float((p[index, column]-outcome)**2))
            support += 1
        means = {uid: float(np.mean(loss)) for uid, loss in unit_losses.items()}
        brier.append({"horizon_s": horizon, "brier": float(np.mean(list(means.values()))) if means else None,
                      "known_origins": support, "unknown_origins": unknown,
                      "physical_units": len(means), "per_unit": means,
                      "prediction_coverage": int(qualified[:, column].sum()) / eligible_count if eligible_count else 0.0,
                      "prediction_eligible_origin_count": eligible_count,
                      "prediction_available_origin_count": int(qualified[:, column].sum()),
                      "known_inactive_excluded_origin_count": excluded_count})
    green_episodes = [a for a in episodes if str(a["zone"]).lower() == "green"]
    green_false = sum(a["horizon_outcome"] == "false" for a in green_episodes)
    green_metrics = {"confirmed_alert_episodes": len(green_episodes), "false_alert_episodes": green_false,
                     "unknown_alert_episodes": sum(a["horizon_outcome"] == "unknown" for a in green_episodes),
                     "false_alert_episodes_per_100_operating_hours": green_false*360000/operating if operating else None,
                     "exposure_denominator": "all_reliable_operating_exposure",
                     "lead_s": [e["lead_s"] for e in event_results if e["green_useful"]]}
    metrics = {"event_count": len(event_results), "independent_event_units": len({
        e["physical_unit_id"] for e in event_results}),
        "useful_event_recall": useful/len(event_results) if recall_available else None,
        "reachable_event_count": reachable_count,
        "reachable_useful_event_recall": useful/reachable_count if recall_available and reachable_count else None,
        "green_reachable_event_count": green_count,
        "green_useful_event_recall": sum(e["green_useful"] for e in event_results)/green_count if recall_available and green_count else None,
        "false_alert_episodes": false, "unknown_alert_episodes": sum(a["horizon_outcome"] == "unknown" for a in episodes),
        "confirmed_alert_episodes": len(episodes), "operating_exposure_s": operating,
        "alert_operating_s": alert, "alert_time_fraction": alert/operating if operating else None,
        "false_alert_episodes_per_100_operating_hours": false*360000/operating if operating else None,
        "event_status_counts": {status: sum(e["status"] == status for e in event_results)
                                for status in ("useful", "early", "late", "missed", "unreachable", "unknown",
                                               "action_lead_requirement_unset")}}
    reasons = []
    if not eligible_count:
        reasons.append("no_eligible_policy_origins")
    if policy_coverage < 1.0:
        reasons.append("partial_policy_prediction_coverage")
    unset = [key for key, value in required.items() if value is None]
    if unset:
        reasons.append("operational_requirements_unset")
    provenance = provenance or {}
    holdout_ids = set(map(str, provenance.get("physical_ids_by_part", {}).get("holdout", [])))
    evaluated_ids = set(frame.get("physical_unit_id", frame.unit_id).astype(str))
    if (not provenance.get("independent_holdout_available") or
            not provenance.get("physical_identity_verified") or not evaluated_ids or
            not evaluated_ids.issubset(holdout_ids) or
            evaluated_ids.intersection(map(str, provenance.get("exposed_physical_ids", [])))):
        reasons.append("no_independent_holdout")
    if not model_frozen or not policy_frozen:
        reasons.append("model_or_policy_not_frozen")
    minimum = required.get("minimum_independent_event_evidence")
    if not event_results or minimum is None or metrics["independent_event_units"] < minimum:
        reasons.append("insufficient_event_evidence")
    for requirement, metric in (("max_alert_time_fraction", "alert_time_fraction"),
            ("max_false_alert_episodes_per_100_operating_hours", "false_alert_episodes_per_100_operating_hours")):
        if required.get(requirement) is not None and (metrics[metric] is None or metrics[metric] > required[requirement]):
            reasons.append(requirement + "_not_met")
    evidence = validation_evidence or {}
    coverage, width = evidence.get("interval_coverage"), evidence.get("interval_maximum_width_s")
    interval_min = required.get("interval_coverage_evidence_requirement")
    if (not evidence.get("interval_support_validated") or interval_min is None or
            evidence.get("interval_independent_event_units", 0) < interval_min or
            required.get("nominal_interval_coverage") is None or coverage is None or
            not np.isfinite(coverage) or coverage < required["nominal_interval_coverage"] or
            required.get("maximum_interval_width_s") is None or width is None or
            not np.isfinite(width) or width < 0 or width > required["maximum_interval_width_s"]):
        reasons.append("interval_coverage_not_validated")
    regimes = required.get("required_regime_coverage")
    observed_regimes = evidence.get("observed_regimes", [])
    if (not evidence.get("regime_support_validated") or not regimes or
            not isinstance(regimes, (list, tuple)) or not set(regimes).issubset(observed_regimes)):
        reasons.append("regime_coverage_not_validated")
    passed = not reasons
    return {"version": "red_entry_evaluation_v1", "metrics": metrics, "events": event_results,
            "alert_episodes": episodes, "brier_by_horizon": brier, "per_unit_exposure": exposure,
            "green_metrics": green_metrics,
            "interval_metrics": {"coverage": coverage, "width_s": width,
                                 "independent_event_units": evidence.get("interval_independent_event_units", 0),
                                 "status": "validated" if "interval_coverage_not_validated" not in reasons else "not_validated"},
            "quality_gate": {"status": "passed" if passed else "requirements_unset" if unset else "not_passed",
                             "can_pass": passed, "reason_codes": reasons, "unset_requirements": unset},
            "policy": cfg, "exposure_basis": "explicit_laboratory_elapsed_assumption" if laboratory_elapsed_assumption else "observed_is_running",
            "metric_scope": {"brier": "unit_equal_known_outcome_only_not_IPCW",
                             "useful_event_recall": {
                                 "status": "available" if recall_available else "unavailable",
                                 "reason_codes": recall_reasons,
                                 "basis": "complete_verified_event_outcomes_only",
                                 "applies_to": ["useful_event_recall", "reachable_useful_event_recall",
                                                "green_useful_event_recall"],
                                 "verified_event_count": len(event_results),
                                 "known_usefulness_event_count": known_event_count,
                                 "unknown_event_count": unknown_event_count,
                                 "known_usefulness_coverage": known_event_count/len(event_results) if event_results else None,
                                 "denominators": {"all_events": len(event_results),
                                                  "reachable_events": reachable_count,
                                                  "green_reachable_events": green_count}},
                             "reachability": "policy_horizon_history_only" if lead_min is None else "action_lead_and_policy_horizon",
                             "policy_prediction_coverage": policy_coverage,
                             "policy_prediction_eligible_origin_count": eligible_count,
                             "policy_prediction_available_origin_count": policy_available_count,
                             "policy_prediction_known_inactive_excluded_origin_count": excluded_count,
                             "prediction_coverage_basis": "finite_and_reliable_on_potential_policy_origins; confirmed_inactive_excluded; uncertain_origins_retained",
                             "common_origin_prediction_count": int(qualified.all(axis=1).sum()),
                             "alert_time": "complete" if policy_coverage == 1 else "partial_observed_predictions_only",
                             "censoring": "incomplete_horizons_excluded; informative_maintenance_censoring_unresolved",
                             "uncertainty": "per_physical_unit_distributions; no_row_bootstrap"}}
