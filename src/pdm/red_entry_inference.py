"""Frozen-run strictly causal RED-entry replay; no future target lookup."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.project_zones import resolve_thresholds, zone_labels
from pdm.red_entry_features import build_windows, transform_prefix
from pdm.red_entry_protocol import red_rule_identity
from pdm.red_entry_training import calibrated_hazard, load_red_entry_run, predict_hazard


def _risk(prefix, schema):
    cycle = next((c for c in ("component_cycle_id", "cycle_id") if c in prefix), None)
    start = 0
    for i in range(1, len(prefix)):
        replaced = prefix.iloc[i].get("component_replaced", False)
        if (cycle and str(prefix[cycle].iloc[i]) != str(prefix[cycle].iloc[i - 1])) or (
            pd.notna(replaced) and str(replaced).lower() in ("true", "1", "1.0")
        ):
            start = i
    episode = prefix.iloc[start:].reset_index(drop=True)
    uncertain = False
    ever_red = False
    current = "unknown"
    for i, row in episode.iterrows():

        def flag(name, default=True):
            v = row.get(name, default)
            return pd.notna(v) and str(v).lower() in ("true", "1", "1.0")

        good = np.isfinite(float(row.signal)) and flag("quality_ok") and flag("usable")
        if "quality_status" in row and pd.notna(row.quality_status):
            good &= str(row.quality_status).lower() in ("good", "ok", "usable", "valid")
        if i and flag("gap_before", False):
            uncertain = True
        if not good:
            uncertain = True
        rule = resolve_thresholds(schema, episode.iloc[: i + 1])
        current = str(zone_labels([row.signal], rule)[0]) if good else "unknown"
        ever_red |= current == "red"
    return current, ever_red, uncertain, rule["status"] == "available"


def forecast_red_entry_prefix(
    project_id, run_id, unit_id, as_of_s, *, thresholds=None, should_stop=None, progress_cb=None
):
    if should_stop and should_stop():
        raise InterruptedError("RED-entry inference cancelled")
    if not np.isfinite(as_of_s):
        raise ValueError("as_of_s must be finite")
    run = load_red_entry_run(project_id, run_id)
    contract = run["contract"]
    schema = run["effective_schema"]
    if (
        thresholds is not None
        and red_rule_identity(schema, effective_thresholds=thresholds)["red_rule_hash"]
        != contract["red_rule"]["red_rule_hash"]
    ):
        raise ValueError("Inference RED rule changed")
    data = run["snapshot"]
    grid = contract["horizons_s"]
    prefix = (
        data["features"]
        .loc[
            (data["features"].unit_id.astype(str) == str(unit_id))
            & (data["features"].timestamp_s <= float(as_of_s))
        ]
        .sort_values("timestamp_s", kind="stable")
        .copy()
    )
    result = dict(
        issued_at_s=float(as_of_s),
        task_version=contract["task_version"],
        snapshot_id=run["snapshot_id"],
        model_id=run_id,
        red_rule_version=contract["red_rule"]["red_rule_version"],
        input_mode=contract["input_mode"],
        prediction_status="insufficient_context",
        probability_by_horizon=[
            {"horizon_s": h, "probability": None, "support_status": "unavailable"} for h in grid
        ],
        time_to_red_quantiles_s={"q05": None, "q50": None, "q95": None},
        red_entry_corridor={
            "status": "unavailable",
            "earliest_s": None,
            "latest_s": None,
            "nominal_coverage": None,
            "calibration_status": run["calibration_status"],
            "reason": "insufficient_event_evidence",
        },
        warning={
            "status": "unavailable",
            "policy_version": None,
            "probability_horizon_s": None,
            "threshold": None,
            "reason_codes": ["policy_not_frozen"],
        },
        age_source="unknown",
        input_quality={"status": "unavailable", "real_history_length": 0},
        calibration_status=run["calibration_status"],
        quality_gate_status=run["quality_gate_status"],
        already_red=False,
        risk_status="insufficient_context",
        operating_scenario=contract["scenario"],
    )
    if prefix.empty:
        return result
    current, ever_red, uncertain, rule_available = _risk(prefix, schema)
    transformed = transform_prefix(prefix, schema, run["feature_state"])
    age_available = bool(transformed["age_available"][-1])
    result["age_source"] = (
        str(prefix.iloc[-1].get("operating_age_source", "unknown")) if age_available else "unknown"
    )
    sensor_available = any(
        spec["role"] == "sensor"
        and pd.notna(prefix.iloc[-1].get(name))
        and (
            name + "_known" not in prefix
            or str(prefix.iloc[-1].get(name + "_known")).lower() in ("true", "1", "1.0")
        )
        for name, spec in run["feature_state"]["source_specs"].items()
    )
    effective_branch = (
        "hybrid"
        if age_available and sensor_available
        else "sensor_only"
        if sensor_available
        else "age_context"
        if age_available
        else "context_only"
    )
    result["input_quality"] = {
        "status": "available" if transformed["availability"][-1] else "unavailable",
        "real_history_length": min(len(prefix), run["params"]["history_length"]),
        "age_available": age_available,
        "effective_input_branch": effective_branch,
    }
    result["already_red"] = current == "red"
    if ever_red:
        result.update(
            prediction_status="unavailable",
            risk_status="already_red" if current == "red" else "post_event",
        )
        result["warning"]["reason_codes"] = [result["risk_status"]]
        return result
    if uncertain or not rule_available:
        result.update(
            prediction_status="unavailable",
            risk_status="unknown_event_history" if uncertain else "rule_unavailable",
        )
        return result
    if not transformed["supported_regime"][-1]:
        result["prediction_status"] = "unsupported_regime"
        return result
    if not transformed["availability"][-1]:
        return result
    origin = prefix[["unit_id", "timestamp_s"]].reset_index(drop=True)
    causal_data = {**data, "features": prefix}
    batch = build_windows(
        causal_data, origin, run["feature_state"], run["params"]["history_length"], schema=schema
    )
    batch["prefixes"] = [prefix.iloc[:i+1].copy() for i in range(len(prefix))]
    batch["schema"] = schema
    hazards = predict_hazard(run["model"], batch, should_stop=should_stop, status_cb=progress_cb)
    hazards = calibrated_hazard(hazards, contract.get("calibration", {}))
    hazard = hazards[-1]
    if np.any(np.isfinite(hazard) & ((hazard < 0) | (hazard > 1))):
        raise ValueError("Invalid model hazard")
    probability = 1 - np.cumprod(1 - hazard)
    supported = np.asarray(run["support_by_horizon"], bool) & np.isfinite(probability)
    result.update(
        prediction_status="available" if supported.any() else "unavailable", risk_status="at_risk"
    )
    result["probability_by_horizon"] = [
        {
            "horizon_s": h,
            "probability": float(p) if support else None,
            "support_status": "supported_exploratory" if support else "unsupported",
        }
        for h, p, support in zip(grid, probability, supported)
    ]
    for label, q in [("q05", 0.05), ("q50", 0.5), ("q95", 0.95)]:
        reached = np.flatnonzero(supported & (probability >= q))
        if len(reached):
            result["time_to_red_quantiles_s"][label] = float(grid[reached[0]])
    result["quantile_status"] = "calibrated_grid_boundary" if run["calibration_status"] == "fitted" else "raw_uncalibrated_grid_boundary"
    from pdm.red_entry_calibration import event_corridor
    from pdm.red_entry_policy import alert_episodes
    policy = contract.get("policy", {})
    calibration = contract.get("calibration", {})
    requirements = contract["protocol"]["operational_requirements"]
    if supported.all():
        result["red_entry_corridor"] = event_corridor(probability, grid, float(as_of_s),
            calibration=calibration, requirements=requirements)
    if policy.get("frozen"):
        rows = prefix.reset_index(drop=True).copy()
        eligibility, zones, episode_ids = [], [], []
        episode = 0
        for i in range(len(rows)):
            _cancelled = should_stop and should_stop()
            if _cancelled:
                raise InterruptedError("RED-entry inference cancelled")
            row = rows.iloc[i]
            cycle = next((c for c in ("component_cycle_id", "cycle_id") if c in rows), None)
            if i and ((cycle and str(row[cycle]) != str(rows.iloc[i-1][cycle])) or
                      str(row.get("component_replaced", False)).lower() in ("true", "1", "1.0")):
                episode += 1
            zone, seen, uncertain_past, available_rule = _risk(rows.iloc[:i+1], schema)
            eligibility.append(not seen and not uncertain_past and available_rule)
            zones.append(zone)
            episode_ids.append(str(unit_id) + ":" + str(episode))
        rows["at_risk"] = eligibility
        rows["risk_status"] = ["at_risk" if v else "unavailable" for v in eligibility]
        rows["context_known"] = batch["availability"] & batch["supported_regime"]
        rows["zone"], rows["episode_id"] = zones, episode_ids
        probabilities = 1 - np.cumprod(1 - hazards, axis=1)
        probabilities[:, ~np.asarray(run["support_by_horizon"], bool)] = np.nan
        episodes = alert_episodes(rows, probabilities, grid, policy=policy["config"])
        latest = float(rows.timestamp_s.iloc[-1])
        active = next((e for e in reversed(episodes) if e["end_s"] == latest and
                       e["termination"] == "observation_end"), None)
        policy_column = grid.index(policy["config"]["horizon_s"])
        policy_supported = bool(np.isfinite(probabilities[-1, policy_column]))
        result["warning"] = {"status": "active" if active else "inactive" if policy_supported else "unavailable",
            "policy_version": policy["version"], "probability_horizon_s": policy["config"]["horizon_s"],
            "threshold": policy["config"]["threshold"], "confirmed_at_s": active["start_s"] if active else None,
            "operational_quality_claim": False,
            "reason_codes": ["exploratory_frozen_policy", policy["status"]] if policy_supported else ["unsupported_policy_horizon"]}

    result["input_quality"]["real_history_length"] = int(batch["lengths"][-1])
    if progress_cb:
        progress_cb({"status": "completed", "issued_at_s": float(as_of_s)})
    return result
