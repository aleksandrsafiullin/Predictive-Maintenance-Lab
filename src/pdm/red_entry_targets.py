"""Project-scoped v2 first-recorded-RED survival targets, separate from v1 labels.

Each row remains an origin, including warmup, uncertain histories and post-event
rows. Target masks encode what follow-up actually proves; zero placeholders in
masked bins are not negative evidence. Measurement brackets are not physical
crossing times. A gap never starts a fresh event episode.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from pdm.project_zones import resolve_thresholds, zone_labels
from pdm.red_entry_protocol import canonical_json_hash, red_rule_identity, validate_horizons

TARGET_VERSION = "project_red_entry_survival_targets_v2"
REGIME_FIELDS = ("operating_regime", "regime_id", "is_running", "rpm", "load_kn", "flow_rate",
                 "dust_feed", "dust", "temperature")
ENDPOINT_FIELDS = ("confirmed_failure", "confirmed_failure_timestamp_s", "emergency_stop",
                   "emergency_stop_timestamp_s", "planned_maintenance", "maintenance_timestamp_s",
                   "component_replaced", "replacement_timestamp_s")


def _flag(row: pd.Series, name: str, default: bool = False) -> bool:
    value = row.get(name, default)
    if pd.isna(value):
        return default
    if isinstance(value, str):
        if value.strip().lower() in {"true", "1"}:
            return True
        if value.strip().lower() in {"false", "0"}:
            return False
        raise ValueError(f"Invalid boolean {name}")
    if value not in (True, False, 0, 1):
        raise ValueError(f"Invalid boolean {name}")
    return bool(value)



def _age(row: pd.Series | None) -> tuple[float | None, str]:
    if row is None:
        return None, "unknown"
    source = str(row.get("operating_age_source", "unknown"))
    value = row.get("operating_age_s", np.nan)
    if source not in {"counter", "laboratory_proxy", "running_clock"} or not _flag(row, "operating_age_known"):
        return None, "unknown"
    if pd.isna(value) or not np.isfinite(float(value)) or float(value) < 0:
        return None, "unknown"
    return float(value), source


def _regime(row: pd.Series | None) -> dict:
    if row is None:
        return {}
    return {key: row[key] for key in REGIME_FIELDS if key in row and pd.notna(row[key])}


def _json_records(frame: pd.DataFrame) -> list[dict]:
    # Pandas JSON maps missing numeric values to JSON null, preserving column order.
    import json
    return json.loads(frame.to_json(orient="records", double_precision=15))


def build_red_entry_targets(snapshot: dict, horizons_s: list[float], *, schema=None) -> dict:
    """Return origin/event tables and aligned [N,K] conditional-hazard targets.

    ``schema`` is the explicit effective project schema (e.g. threshold sidecar
    overrides). No display state is read implicitly. ``gap_before`` is admitted
    import evidence; cadence is never estimated using future measurements.
    Quality admits finite signals unless quality_ok/usable explicitly rejects a
    row, or quality_status is outside good/ok/usable/valid. Explicit cycle IDs
    and component_replaced begin episodes; ordinary service does not.
    """
    grid = validate_horizons(horizons_s)
    saved = snapshot["schema"]
    effective = copy.deepcopy(saved if schema is None else schema)
    rule = red_rule_identity(saved, effective_thresholds=(effective.get("thresholds") or {})
                             if schema is not None else None)
    frame = snapshot["features"].copy()
    if not {"unit_id", "timestamp_s", "signal"}.issubset(frame.columns):
        raise ValueError("Targets require unit_id, timestamp_s and signal")
    if frame.unit_id.isna().any():
        raise ValueError("Missing unit identity")
    frame["unit_id"] = frame.unit_id.astype(str)
    frame["timestamp_s"] = pd.to_numeric(frame.timestamp_s, errors="coerce")
    if not np.isfinite(frame.timestamp_s.to_numpy(float)).all():
        raise ValueError("Targets require finite timestamps")
    frame["signal"] = pd.to_numeric(frame.signal, errors="coerce")
    # The quality report may cover a full snapshot while callers deliberately
    # pass only development features. Never read endpoint evidence for held-out
    # units into development targets or their identity.
    included_units = set(frame.unit_id)
    auxiliary_endpoints = [copy.deepcopy(record) for record in
                           snapshot.get("report", {}).get("quality", {}).get("endpoint_records", [])
                           if str(record.get("unit_id")) in included_units]
    assigned_endpoints = set()
    ownership = {str(uid): part for part in ("train", "validation", "test", "holdout")
                 for uid in snapshot["split"].get(part, [])}
    if sum(len(snapshot["split"].get(p, [])) for p in ("train", "validation", "test", "holdout")) != len(ownership):
        raise ValueError("Duplicate split assignment")
    units = snapshot.get("units", pd.DataFrame())
    physical_map = (dict(zip(units.unit_id.astype(str), units.physical_unit_id.astype(str)))
                    if {"unit_id", "physical_unit_id"}.issubset(units.columns) else {})
    origins, events, targets, masks = [], [], [], []
    for uid, unit in frame.groupby("unit_id", sort=True):
        if uid not in ownership:
            raise ValueError(f"Unit has no split: {uid}")
        unit = unit.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        if unit.timestamp_s.duplicated().any():
            raise ValueError(f"Duplicate timestamps for unit {uid}")
        cycle_key = next((key for key in ("cycle_id", "component_cycle_id") if key in unit), None)
        if cycle_key and unit[cycle_key].isna().any():
            raise ValueError("Missing explicit cycle identity")
        segments, segment_start, replacement_number = [], 0, 0
        for i in range(1, len(unit)):
            changed = cycle_key and str(unit[cycle_key].iloc[i]) != str(unit[cycle_key].iloc[i - 1])
            if changed or _flag(unit.iloc[i], "component_replaced"):
                segments.append((segment_start, i, replacement_number))
                segment_start, replacement_number = i, replacement_number + 1
        segments.append((segment_start, len(unit), replacement_number))
        for start, end, number in segments:
            group = unit.iloc[start:end].reset_index(drop=True)
            cycle = str(group[cycle_key].iloc[0]) if cycle_key else str(number)
            # Replacement can explicitly reopen an episode under an unchanged imported ID.
            episode_id = f"{uid}:{cycle}:{number}"
            times = group.timestamp_s.to_numpy(float)
            zones, limits, usable, breaks, statuses = [], [], [], [], []
            history_unknown, observed_red = False, False
            for i, row in group.iterrows():
                thresholds = resolve_thresholds(effective, group.iloc[:i + 1])
                available = thresholds["status"] == "available"
                quality = np.isfinite(row.signal) and _flag(row, "quality_ok", True) and _flag(row, "usable", True)
                if "quality_status" in group and pd.notna(row.quality_status):
                    quality = quality and str(row.quality_status).lower() in {"good", "ok", "usable", "valid"}
                gap = i > 0 and _flag(row, "gap_before")
                zone = str(zone_labels([row.signal], thresholds)[0]) if quality else "unknown"
                if gap or not quality:
                    history_unknown = True
                if observed_red:
                    status = "post_event"
                elif not quality:
                    status = "quality_unavailable"
                elif not available:
                    status = "rule_unavailable"
                elif history_unknown:
                    status = "unknown_event_history"
                elif zone == "red":
                    status = "event_observed"
                else:
                    status = "at_risk"
                if zone == "red":
                    observed_red = True
                zones.append(zone)
                limits.append(thresholds.get("red") if available else None)
                usable.append(bool(quality and available))
                breaks.append("gap" if gap else "quality" if not quality else "rule_unavailable" if not available else None)
                statuses.append(status)
            red_indices = [i for i, zone in enumerate(zones) if zone == "red"]
            first_red = red_indices[0] if red_indices else None
            # Backward propagation gives each origin its next reliable endpoint
            # in linear time, rather than scanning the remaining file per row.
            ends = np.arange(len(group))
            future_events = [None] * len(group)
            reasons = ["observation_end" if end == len(unit) else "new_cycle"] * len(group)
            for i in range(len(group) - 2, -1, -1):
                j = i + 1
                if breaks[j] or not usable[j]:
                    reasons[i] = breaks[j] or "quality"
                elif zones[j] == "red":
                    ends[i], future_events[i], reasons[i] = j, j, "red_event"
                else:
                    ends[i], future_events[i], reasons[i] = ends[j], future_events[j], reasons[j]
            for i, t in enumerate(times):
                at_risk = statuses[i] == "at_risk"
                followup_end = int(ends[i]) if at_risk else i
                event_index = future_events[i] if at_risk else None
                reason = reasons[i]
                duration = float(times[followup_end] - t) if at_risk else 0.0
                event_time = float(times[event_index]) if event_index is not None else None
                y, mask = np.zeros(len(grid), dtype=np.int8), np.zeros(len(grid), dtype=bool)
                if at_risk:
                    if event_index is not None:
                        event_bin = int(np.searchsorted(grid, duration, side="left"))
                        mask[:min(event_bin + 1, len(grid))] = True
                        if event_bin < len(grid):
                            y[event_bin] = 1
                    else:
                        mask[:] = np.asarray(grid) <= duration
                physical = str(group.physical_unit_id.iloc[i]) if "physical_unit_id" in group else physical_map.get(uid, uid)
                origins.append({"unit_id": uid, "physical_unit_id": physical, "cycle_id": cycle,
                                "episode_id": episode_id, "timestamp_s": float(t), "split": ownership[uid],
                                "zone": zones[i], "at_risk": at_risk, "risk_status": statuses[i],
                                "followup_duration_s": duration, "event_observed": event_index is not None,
                                "event_time_s": event_time, "red_limit": limits[i],
                                "censor_reason": reason if at_risk and event_index is None else None})
                targets.append(y)
                masks.append(mask)
            bracket_left = (float(times[first_red - 1]) if first_red is not None and first_red > 0
                            and usable[first_red - 1] and not breaks[first_red] and zones[first_red - 1] != "red" else None)
            reliable_end = None
            censor_reason = "observation_end" if end == len(unit) else "new_cycle"
            for i in range(len(group)):
                if statuses[i] == "rule_unavailable" and breaks[i] == "rule_unavailable":
                    continue
                if breaks[i]:
                    censor_reason = breaks[i]
                    break
                reliable_end = float(times[i])
                if zones[i] == "red":
                    censor_reason = "red_event"
                    break
            endpoint_records = [{"timestamp_s": float(row.timestamp_s), **{
                name: row[name] for name in ENDPOINT_FIELDS if name in group and pd.notna(row[name])}}
                for _, row in group.iterrows() if any(name in group and pd.notna(row[name]) for name in ENDPOINT_FIELDS)]
            for index, record in enumerate(auxiliary_endpoints):
                upper = record.get("episode_end_timestamp_s")
                if (str(record["unit_id"]) == uid
                        and str(record["component_cycle_id"]) == cycle
                        and float(record["episode_start_timestamp_s"]) <= times[0]
                        and (upper is None or times[0] < float(upper))):
                    endpoint_records.append(record)
                    assigned_endpoints.add(index)
            endpoint_records.sort(key=lambda record: float(record["timestamp_s"]))
            verified = first_red is not None and statuses[first_red] == "event_observed"
            event_row = group.iloc[first_red] if verified else None
            censor_row = (group.loc[group.timestamp_s == reliable_end].iloc[0]
                          if reliable_end is not None and censor_reason != "red_event" else None)
            start_age, start_source = _age(group.iloc[0])
            event_age, event_source = _age(event_row)
            censor_age, censor_source = _age(censor_row)
            events.append({"unit_id": uid, "physical_unit_id": origins[-1]["physical_unit_id"],
                           "cycle_id": cycle, "episode_id": episode_id, "split": ownership[uid],
                           "first_red_timestamp_s": float(times[first_red]) if first_red is not None else None,
                           "first_event_verified": verified,
                           "observation_start_s": float(times[0]),
                           "age_at_start_s": start_age, "age_at_start_source": start_source,
                           "age_at_event_s": event_age, "age_at_event_source": event_source,
                           "age_at_censor_s": censor_age, "age_at_censor_source": censor_source,
                           "regime_metadata": {"at_start": _regime(group.iloc[0]),
                                               "at_event": _regime(event_row), "at_censor": _regime(censor_row)},
                           "transition_left_s": bracket_left,
                           "transition_right_s": float(times[first_red]) if first_red is not None else None,
                           "transition_semantics": "last_usable_nonred_to_first_recorded_red_measurements",
                           "observation_end_s": float(times[-1]), "reliable_followup_end_s": reliable_end,
                           "censor_reason": censor_reason, "endpoint_records": endpoint_records,
                           "red_rule_version": rule["red_rule_version"], "red_rule_hash": rule["red_rule_hash"]})
    origin_frame = pd.DataFrame(origins, columns=["unit_id", "physical_unit_id", "cycle_id", "episode_id",
                                                 "timestamp_s", "split", "zone", "at_risk", "risk_status",
                                                 "followup_duration_s", "event_observed", "event_time_s",
                                                 "red_limit", "censor_reason"])
    event_frame = pd.DataFrame(events, columns=["unit_id", "physical_unit_id", "cycle_id", "episode_id",
                                               "split", "first_red_timestamp_s", "first_event_verified",
                                               "transition_left_s", "transition_right_s", "transition_semantics",
                                               "observation_end_s", "reliable_followup_end_s", "censor_reason",
                                               "endpoint_records", "red_rule_version", "red_rule_hash", "observation_start_s",
                                               "age_at_start_s", "age_at_start_source", "age_at_event_s",
                                               "age_at_event_source", "age_at_censor_s", "age_at_censor_source",
                                               "regime_metadata"])
    y = np.asarray(targets, dtype=np.int8).reshape(-1, len(grid))
    mask = np.asarray(masks, dtype=bool).reshape(-1, len(grid))
    identity = {"target_version": TARGET_VERSION, "snapshot_id": snapshot.get("snapshot_id"),
                "snapshot_schema_hash": canonical_json_hash(saved), "red_rule": rule,
                "horizons_s": grid, "horizon_unit": "seconds",
                "quality_policy": "finite_signal_and_explicit_quality_v2",
                "gap_policy": "admitted_gap_before_no_implicit_episode_reset",
                "source_features_hash": canonical_json_hash(_json_records(frame)),
                "split_hash": canonical_json_hash(snapshot["split"])}
    unassigned_endpoints = [record for index, record in enumerate(auxiliary_endpoints)
                            if index not in assigned_endpoints]
    target_content = {**identity, "origins": _json_records(origin_frame),
                      "events": _json_records(event_frame),
                      "hazard_targets": y.tolist(), "hazard_mask": mask.tolist()}
    if unassigned_endpoints:
        target_content["unassigned_endpoint_records"] = unassigned_endpoints
    identity["target_hash"] = canonical_json_hash(target_content)
    return {"origins": origin_frame, "events": event_frame, "hazard_targets": y,
            "hazard_mask": mask, "horizons_s": grid, "identity": identity,
            "target_hash": identity["target_hash"], "unassigned_endpoint_records": unassigned_endpoints}
