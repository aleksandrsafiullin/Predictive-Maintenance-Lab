"""Serializable deterministic state machine. No outcomes, wall clock or fitting."""
from __future__ import annotations

from copy import deepcopy

from pdm.monitoring.policy import ACTIONS, action_timing
from pdm.training_protocol import fingerprint


def initial_state(unit_id, bundle_id):
    return {"unit_id": str(unit_id), "bundle_id": bundle_id, "last_as_of": None,
            "last_measurement_at": None, "segment_id": None, "health_state": "unknown",
            "display_zone": "gray", "critical_latch": False, "active_episode": None,
            "episodes": [], "warning_count": 0, "warning_since": None,
            "recovery_count": 0, "recovery_since": None, "limit_since": None,
            "last_resolved_at": None, "reason_codes": [], "last_result": None}


def _reset_confirmation(state):
    state.update(warning_count=0, warning_since=None, recovery_count=0, recovery_since=None, limit_since=None)


def update_state(previous, condition, event, profile, policy, as_of):
    state = deepcopy(previous)
    if state["last_as_of"] is not None and as_of < state["last_as_of"]:
        raise ValueError("Seek requires prefix replay, not mutation of future state")
    if state["last_as_of"] == as_of:
        return state  # Repeated UI render/seek cannot create another issued alert.
    quality = condition["quality"]
    normality = condition["normality"]
    segment = condition["segment_id"]
    changed = segment != state["segment_id"]
    fresh = quality["last_measurement_at"] != state["last_measurement_at"]
    if changed:
        _reset_confirmation(state)
    state.update(segment_id=segment, last_as_of=float(as_of), last_measurement_at=quality["last_measurement_at"])
    good = quality["data_quality_status"] in {"valid", "degraded"}
    reasons = list(quality["reason_codes"] + normality["reason_codes"])
    timing = action_timing(event, profile, policy, normality["model_applicability"], quality["data_quality_status"])
    limit = policy.get("critical_limit")
    hard = False
    measured_at = quality["last_measurement_at"] if quality["last_measurement_at"] is not None else as_of
    if limit and quality.get("critical_channel_usable"):
        row = condition["measurement"]
        applicable = (limit["unit"] == profile["signal_unit"] and limit["signal_name"] == profile["signal_name"]
                      and all(row.get(k, profile.get(k)) == v for k, v in limit["applicability"].items()))
        value = row.get(limit["signal_name"])
        hit = applicable and value is not None and (value >= limit["value"] if limit["direction"] == "above" else value <= limit["value"])
        if hit:
            if state["limit_since"] is None:
                state["limit_since"] = measured_at
            hard = limit["aggregation_rule"] == "instantaneous" or measured_at - state["limit_since"] >= limit["duration"]
        else:
            state["limit_since"] = None
    score = normality["score"]
    enter, leave = policy["warning_enter"], policy["warning_exit"]
    eligible = timing["prognostic_escalation_eligible"]
    # A validated forecast can warn before any anomaly; readiness alone is insufficient.
    risk_warning = eligible and event.get("risk_band") in {"elevated", "urgent"}
    deviation = good and score is not None and enter is not None and score >= enter
    if fresh:
        if deviation or risk_warning:
            if state["warning_since"] is None:
                state["warning_since"] = measured_at
            state["warning_count"] += 1
            state.update(recovery_count=0, recovery_since=None)
        else:
            state.update(warning_count=0, warning_since=None)
    confirmed = (state["warning_count"] >= policy["confirmation_count"] and
                 state["warning_since"] is not None and measured_at - state["warning_since"] >= policy["confirmation_duration"])
    recovering = good and normality["model_applicability"] == "in_domain" and score is not None and leave is not None and score < leave and not risk_warning and not hard
    if fresh:
        if recovering:
            if state["recovery_since"] is None:
                state["recovery_since"] = measured_at
            state["recovery_count"] += 1
        else:
            state.update(recovery_count=0, recovery_since=None)
    recovered = (state["recovery_count"] >= policy["recovery_count"] and
                 state["recovery_since"] is not None and measured_at - state["recovery_since"] >= policy["recovery_duration"])
    active = state["active_episode"]
    if recovered and active:
        active["resolved_at"] = float(as_of)
        active["updates"].append({"as_of": as_of, "type": "resolved_by_sustained_recovery"})
        state["episodes"] = [active if e["episode_id"] == active["episode_id"] else e for e in state["episodes"]]
        state.update(active_episode=None, critical_latch=False, last_resolved_at=float(as_of))
        active = None
    if hard:
        health, zone, action = "critical", "red", "configured_limit"
        reasons.append("observed_applicable_critical_limit")
        state["critical_latch"] = True
    elif quality["data_quality_status"] in {"invalid", "stale", "insufficient_history"}:
        health, zone, action = "unknown", "gray", "restore_observation"
        _reset_confirmation(state)
    elif normality["model_applicability"] != "in_domain" or score is None or enter is None:
        health, zone, action = "unknown", "gray", "assessment_unavailable"
        if enter is None:
            reasons.append("warning_threshold_not_fitted")
    elif confirmed and eligible and timing["planning_margin"] <= 0:
        health, zone, action = "critical", "red", "urgent_review"
        reasons.append("eligible_prognostic_urgency")
        state["critical_latch"] = True
    elif state["critical_latch"]:
        health, zone, action = "critical", "red", "configured_limit"
        reasons.append("open_critical_alert_awaiting_recovery")
    elif confirmed or active:
        health, zone, action = "deviation", "yellow", "prepare_inspection" if risk_warning else "inspect_recent_deviation"
        reasons.append("persistent_regime_adjusted_deviation" if not risk_warning else "eligible_forecast_risk")
    else:
        health, zone, action = "normal", "green", "continue_monitoring"
        reasons.append("no_persistent_deviations_detected")
    if zone in {"yellow", "red"}:
        cooldown = state["last_resolved_at"] is not None and as_of - state["last_resolved_at"] < policy["cooldown"]
        if not active and (hard or not cooldown):
            eid = fingerprint([state["bundle_id"], state["unit_id"], as_of])[:20]
            active = {"episode_id": eid, "unit_id": state["unit_id"], "bundle_id": state["bundle_id"],
                      "first_deviation_at": state["warning_since"] or as_of, "confirmed_at": float(as_of),
                      "level": zone, "reason": reasons[-1], "escalated_at": float(as_of) if zone == "red" else None,
                      "resolved_at": None, "acknowledged": False,
                      "kind": "prognostic" if risk_warning else "diagnostic",
                      "event_definition_id": event.get("event_definition_id") if risk_warning else None,
                      "issued_horizon": event.get("validated_horizon") if risk_warning else None,
                      "policy_version": policy["version"], "updates": []}
            state["episodes"].append(active)
            state["active_episode"] = active
        elif active and zone == "red" and active["level"] != "red":
            active.update(level="red", escalated_at=float(as_of))
            active["updates"].append({"as_of": float(as_of), "type": "escalation"})
    if state["active_episode"]:
        eid = state["active_episode"]["episode_id"]
        state["episodes"] = [state["active_episode"] if e["episode_id"] == eid else e for e in state["episodes"]]
    state.update(health_state=health, display_zone=zone, reason_codes=reasons)
    state["last_result"] = {"health_state": health, "display_zone": zone, "reason_codes": reasons,
        "event_status": "observed_limit" if hard else "not_observed", "urgency": action,
        "action_code": action, "action_text": ACTIONS[action], "timing": timing,
        "critical_latch": state["critical_latch"],
        "alert_episode_id": state["active_episode"]["episode_id"] if state["active_episode"] else None}
    return state
