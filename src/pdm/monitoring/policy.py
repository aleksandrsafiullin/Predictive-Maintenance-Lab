"""Explicit laboratory policy, eligible risk and compatible action arithmetic."""
from __future__ import annotations

import numpy as np


def default_policy(profile):
    interval = profile["nominal_interval"]
    return {"version": "condition_policy_v1", "warning_enter": None, "warning_exit": None,
            "confirmation_count": 3, "confirmation_duration": 2 * interval,
            "recovery_count": 5, "recovery_duration": 4 * interval,
            "cooldown": 5 * interval, "time_basis": profile["time_basis"],
            "critical_limit": ({"signal_name": "differential_pressure", "value": 600., "unit": "Pa",
                "direction": "above", "aggregation_rule": "instantaneous", "applicability": {"dataset_id": "filters"},
                "source": "configs/filters.yaml:pressure_limit_pa", "verification_status": "laboratory_definition",
                "policy_version": "condition_policy_v1"} if profile["dataset_id"] == "filters" else None),
            "prognostic_methods": [], "max_interval_width": None,
            "action_profile": {"required_action_lead_time": None, "safety_buffer": None,
                "time_basis": "physical_seconds", "verification_status": "not_configured",
                "action": "inspection", "source": None},
            "laboratory_notice": "Laboratory demonstration — not an equipment operating instruction"}


def validate_policy(policy, profile):
    if policy["time_basis"] != profile["time_basis"]:
        raise ValueError("State policy time basis mismatch")
    enter, leave = policy.get("warning_enter"), policy.get("warning_exit")
    if (enter is None) != (leave is None) or (enter is not None and not 0 <= leave < enter):
        raise ValueError("Invalid hysteresis thresholds")
    for name in ("confirmation_count", "recovery_count"):
        if policy[name] < 1:
            raise ValueError("Invalid policy count")
    for name in ("confirmation_duration", "recovery_duration", "cooldown"):
        if not np.isfinite(policy[name]) or policy[name] < 0:
            raise ValueError("Invalid policy duration")
    limit = policy.get("critical_limit")
    if limit:
        required = {"signal_name", "value", "unit", "direction", "aggregation_rule", "applicability", "source", "verification_status", "policy_version"}
        if required - set(limit) or not limit["source"] or not np.isfinite(limit["value"]):
            raise ValueError("Critical limit requires provenance and applicability")
        if limit["verification_status"] not in {"engineering_verified", "laboratory_definition", "experimental_demo"}:
            raise ValueError("Unverified critical limit")
        if limit["direction"] not in {"above", "below"} or limit["aggregation_rule"] not in {"instantaneous", "sustained"}:
            raise ValueError("Unknown limit convention")
        if limit["aggregation_rule"] == "sustained" and limit.get("duration", 0) <= 0:
            raise ValueError("Sustained limit requires a positive duration")


def action_timing(event, profile, policy, applicability, quality):
    result = {"planning_margin": None, "required_time": None,
              "conservative_time_to_event": None, "prognostic_escalation_eligible": False,
              "reason": "forecast_not_validated_for_operational_timing"}
    action = policy["action_profile"]
    if action["time_basis"] != profile["time_basis"] or not profile["time_scale_verified"]:
        result["reason"] = "incompatible_or_unverified_time_basis"
        return result
    if action["required_action_lead_time"] is None or action["safety_buffer"] is None:
        result["reason"] = "action_lead_time_not_configured"
        return result
    if action["verification_status"] not in {"engineering_verified", "laboratory_definition", "experimental_demo"}:
        return result
    interval = event.get("interval")
    if (not interval or event.get("event_definition_id") != profile["event_definition_id"]
            or event.get("time_basis") != profile["time_basis"] or applicability != "in_domain"
            or quality != "valid" or not event.get("timing_validated")
            or interval.get("method") not in policy["prognostic_methods"]):
        return result
    lo, hi = interval.get("lower"), interval.get("upper")
    if lo is None or hi is None or not np.isfinite([lo, hi]).all() or not 0 <= lo <= hi:
        return result
    max_width = policy.get("max_interval_width")
    if max_width is None or hi - lo > max_width:
        result["reason"] = "forecast_uncertainty_too_high"
        return result
    required = action["required_action_lead_time"] + action["safety_buffer"]
    if required < 0 or required > event.get("validated_horizon", 0):
        result["reason"] = "action_time_outside_validated_horizon"
        return result
    result.update(planning_margin=float(lo - required), required_time=float(required),
                  conservative_time_to_event=float(lo), prognostic_escalation_eligible=True,
                  method="lower_bound_of_named_interval", reason="compatible_laboratory_planning_indicator")
    return result


ACTIONS = {
    "continue_monitoring": "Continue monitoring the supported signals.",
    "inspect_recent_deviation": "Inspect the cause and review the recent trend.",
    "prepare_inspection": "Prepare the planned inspection or standby arrangement.",
    "urgent_review": "Urgent review required. Available lead time may be shorter than the preparation time.",
    "configured_limit": "A configured limit has been reached. Review the applicable equipment procedure.",
    "restore_observation": "Restore valid measurements. The last open alert remains active.",
    "assessment_unavailable": "Review observation quality and operating regime before using this assessment.",
}
