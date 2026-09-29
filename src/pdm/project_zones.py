"""Shared project zone rule: saved snapshot thresholds -> per-row green/yellow/red/unknown."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

ZONES = ("green", "yellow", "red", "unknown")
LABEL_COLUMNS = ["unit_id", "timestamp_s", "signal", "gap_before", "zone", "yellow_limit", "red_limit"]


def resolve_thresholds(schema: Mapping, prefix) -> dict:
    rule = dict(schema.get("thresholds") or {})
    mode = rule.get("mode", "absolute")
    direction = rule.get("direction", "above")
    if direction not in {"above", "below"}:
        return {"mode": mode, "direction": direction, "yellow": None, "red": None,
                "status": "unavailable", "reason": "Unsupported threshold direction"}
    if mode == "absolute":
        try:
            red = float(rule["red"])
        except (KeyError, TypeError, ValueError):
            red = None
        try:
            yellow = float(rule["yellow"]) if rule.get("yellow") is not None else None
        except (TypeError, ValueError):
            yellow = None
        if red is None or not np.isfinite(red) or (yellow is not None and not np.isfinite(yellow)):
            return {"mode": mode, "direction": direction, "yellow": None, "red": None,
                    "status": "unavailable", "reason": "No valid saved signal thresholds"}
        return {"mode": mode, "direction": direction, "yellow": yellow, "red": red,
                "status": "available", "reason": None}
    if mode == "initial_baseline_multiple":
        n = int(rule.get("baseline_n", 5))
        if len(prefix) < n:
            return {"mode": mode, "direction": direction, "yellow": None, "red": None,
                    "status": "unavailable", "reason": "Waiting for the initial causal baseline"}
        first = prefix.iloc[:n].signal.to_numpy(float)
        if first.size == 0 or not np.isfinite(first).all():
            return {"mode": mode, "direction": direction, "yellow": None, "red": None,
                    "status": "unavailable", "reason": "Initial baseline contains missing signal values"}
        baseline = float(np.median(first))
        yellow = max(baseline + float(rule.get("onset_sigma", 3.0)) * float(np.std(first)),
                     float(rule.get("onset_ratio", 1.25)) * baseline)
        red = float(rule.get("red_ratio", 2.0)) * baseline
        return {"mode": mode, "direction": direction, "yellow": yellow, "red": red,
                "status": "available", "reason": None, "baseline": baseline,
                "baseline_n": n}
    return {"mode": mode, "direction": direction, "yellow": None, "red": None,
            "status": "unavailable", "reason": "Unsupported saved threshold rule"}


def is_beyond(values, limit, direction: str) -> np.ndarray:
    """Inclusive limit test; NaN values never count as beyond."""
    arr = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        hit = arr >= float(limit) if direction == "above" else arr <= float(limit)
    return np.asarray(hit & np.isfinite(arr), dtype=bool)


def zone_labels(values, thresholds: Mapping) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    labels = np.full(arr.shape, "unknown", dtype=object)
    if thresholds.get("status") != "available":
        return labels
    direction = thresholds["direction"]
    labels[:] = "green"
    if thresholds.get("yellow") is not None:
        labels[is_beyond(arr, thresholds["yellow"], direction)] = "yellow"
    labels[is_beyond(arr, thresholds["red"], direction)] = "red"
    labels[~np.isfinite(arr)] = "unknown"
    return labels


def label_unit(frame: pd.DataFrame, schema: Mapping) -> pd.DataFrame:
    """Label one unit's admitted rows. Gaps never change labels."""
    unit = frame.sort_values("timestamp_s", kind="mergesort").reset_index(drop=True)
    gap = (unit["gap_before"].fillna(False).astype(bool) if "gap_before" in unit
           else pd.Series(False, index=unit.index))
    out = pd.DataFrame({"unit_id": unit["unit_id"].astype(str), "timestamp_s": unit["timestamp_s"].astype(float),
                        "signal": unit["signal"].astype(float), "gap_before": gap.to_numpy(bool)})
    rule = dict(schema.get("thresholds") or {})
    if rule.get("mode") == "initial_baseline_multiple":
        n = int(rule.get("baseline_n", 5))
        thresholds = resolve_thresholds(schema, unit.iloc[:n])
        zoned = np.arange(len(unit)) >= n - 1
    else:
        thresholds = resolve_thresholds(schema, unit)
        zoned = np.ones(len(unit), dtype=bool)
    zones = zone_labels(out["signal"].to_numpy(float), thresholds)
    zones[~zoned] = "unknown"
    known = zones != "unknown"
    yellow = np.nan if thresholds.get("yellow") is None else float(thresholds["yellow"])
    red = np.nan if thresholds.get("red") is None else float(thresholds["red"])
    out["zone"] = zones.astype(str)
    out["yellow_limit"] = np.where(known, yellow, np.nan)
    out["red_limit"] = np.where(known, red, np.nan)
    return out[LABEL_COLUMNS]


def zone_counts(features: pd.DataFrame, unit_ids: Iterable[str], schema: Mapping) -> dict[str, int]:
    counts = dict.fromkeys(ZONES, 0)
    ids = features["unit_id"].astype(str)
    for uid in unit_ids:
        unit = features[ids == str(uid)]
        if unit.empty:
            continue
        values = label_unit(unit, schema)["zone"].value_counts()
        for zone in ZONES:
            counts[zone] += int(values.get(zone, 0))
    return counts


def has_valid_rule(schema: Mapping) -> bool:
    """Schema-only check: can the saved rule zone measurements at all?"""
    rule = dict(schema.get("thresholds") or {})
    if not rule or rule.get("direction", "above") not in {"above", "below"}:
        return False
    mode = rule.get("mode", "absolute")
    if mode == "absolute":
        return resolve_thresholds(schema, pd.DataFrame({"signal": []}))["status"] == "available"
    if mode == "initial_baseline_multiple":
        try:
            n = int(rule.get("baseline_n", 5))
            params = [float(rule.get(key, default)) for key, default in
                      (("onset_sigma", 3.0), ("onset_ratio", 1.25), ("red_ratio", 2.0))]
        except (TypeError, ValueError):
            return False
        return n >= 1 and all(np.isfinite(params))
    return False


def describe_rule(schema: Mapping) -> str:
    rule = dict(schema.get("thresholds") or {})
    mode = rule.get("mode", "absolute")
    direction = rule.get("direction", "above")
    invalid = "No valid yellow/red limits are saved with this data, so measurements are not zoned."
    if not has_valid_rule(schema):
        return invalid
    if mode == "absolute":
        thresholds = resolve_thresholds(schema, pd.DataFrame({"signal": []}))
        if thresholds["status"] != "available":
            return invalid
        op = "≥" if direction == "above" else "≤"
        unit = str(schema.get("signal_unit") or "").strip()
        suffix = f" {unit}" if unit else ""
        parts = []
        if thresholds["yellow"] is not None:
            parts.append(f"yellow at {op} {thresholds['yellow']:g}{suffix}")
        parts.append(f"red at {op} {thresholds['red']:g}{suffix}")
        text = (f"Zones use the limits saved with this data: {', '.join(parts)} "
                "(instantaneous, per measurement).")
    elif mode == "initial_baseline_multiple":
        try:
            n = int(rule.get("baseline_n", 5))
            sigma = float(rule.get("onset_sigma", 3.0))
            ratio = float(rule.get("onset_ratio", 1.25))
            red_ratio = float(rule.get("red_ratio", 2.0))
        except (TypeError, ValueError):
            return invalid
        text = (f"Zones use each unit's initial baseline: median of its first {n} measurements; "
                f"yellow = max(median + {sigma:g} × SD, {ratio:g} × median), red = {red_ratio:g} × median. "
                f"The first {n - 1} measurements are not zoned.")
    else:
        return invalid
    note = str(rule.get("note") or "").strip()
    if note:
        text = f"{text} {note if note.endswith('.') else note + '.'}"
    return text
