"""Train-only, age-conditioned RED-entry baselines with explicit support limits.

Kaplan–Meier uses supplied known component ages and delayed observation entry.
Recorded RED after uncertain history is censored at the reliable endpoint, never
promoted to a lifetime event. Sensor trend outputs are deterministic references,
not calibrated probabilities. This module leaves historical v1 models untouched.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping

import numpy as np
import pandas as pd

from pdm.project_zones import resolve_thresholds
from pdm.red_entry_protocol import canonical_json_hash, validate_horizons

VERSION = "red_entry_age_km_baseline_v1"
KNOWN_AGE_SOURCES = {"counter", "laboratory_proxy", "running_clock"}
REGIME_FIELDS = ("operating_regime", "regime_id", "is_running", "rpm", "load_kn", "flow_rate",
                 "dust_feed", "dust", "temperature")


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _regime(value) -> dict:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"operating_regime": value}
    if not isinstance(value, Mapping):
        raise ValueError("Regime must be a label or mapping")
    result = {}
    for name, item in value.items():
        if item is None or (not isinstance(item, (dict, list)) and pd.isna(item)):
            continue
        if isinstance(item, np.generic):
            item = item.item()
        result[str(name)] = item
    canonical_json_hash(result)
    return result


def _episode_regime(row, observed):
    metadata = row.get("regime_metadata")
    if isinstance(metadata, Mapping):
        start = _regime(metadata.get("at_start"))
        stop = _regime(metadata.get("at_event" if observed else "at_censor"))
        # Exact observed regime groups; do not assume an unmodelled transition
        # has the same lifetime distribution as its initial regime.
        if start != stop:
            return None
        return start
    direct = row.get("regime", row.get("operating_regime"))
    if direct is not None and not pd.isna(direct):
        return _regime(direct)
    return _regime({name: row[name] for name in REGIME_FIELDS if name in row and pd.notna(row[name])})


def _support_intervals(episodes):
    merged = []
    for left, right in sorted((e["entry_age_s"], e["exit_age_s"]) for e in episodes):
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return merged


def fit_age_baseline(events: pd.DataFrame, *, train_unit_ids, min_group_units=2) -> dict:
    """Fit weighted delayed-entry KM only on explicitly selected Train histories.

    Every physical unit has total episode weight one across accepted cycles.
    Censoring ties remain in the risk set for an event at the same age. The
    estimator is descriptive; unsupported groups and tails never return zero.
    """
    if isinstance(min_group_units, bool) or not isinstance(min_group_units, int) or min_group_units < 1:
        raise ValueError("min_group_units must be a positive integer")
    ids = sorted(set(map(str, train_unit_ids)))
    if "unit_id" not in events:
        raise ValueError("Events require unit_id")
    selected = events[events.unit_id.astype(str).isin(ids)]
    if "split" in selected and not selected.split.astype(str).eq("train").all():
        raise ValueError("Age baseline may fit Train rows only")
    exclusions, accepted = Counter(), []
    for _, row in selected.iterrows():
        observed = row.get("first_event_verified", False)
        observed = bool(observed) if pd.notna(observed) else False
        end_role = "event" if observed else "censor"
        start, stop = _number(row.get("age_at_start_s")), _number(row.get(f"age_at_{end_role}_s"))
        source = str(row.get("age_at_start_source", "unknown"))
        stop_source = str(row.get(f"age_at_{end_role}_source", "unknown"))
        if start is None or stop is None or source not in KNOWN_AGE_SOURCES or stop_source != source:
            exclusions["unknown_or_inconsistent_age"] += 1
            continue
        if start < 0 or stop <= start:
            exclusions["no_positive_observed_age_interval"] += 1
            continue
        regime = _episode_regime(row, observed)
        if regime is None:
            exclusions["regime_changed"] += 1
            continue
        physical = row.get("physical_unit_id", row.unit_id)
        if pd.isna(physical):
            raise ValueError("Missing physical unit identity")
        accepted.append({"unit_id": str(row.unit_id), "physical_unit_id": str(physical),
                         "entry_age_s": start, "exit_age_s": stop, "event_observed": observed,
                         "age_source": source, "regime": regime})
    cycle_counts = Counter(e["physical_unit_id"] for e in accepted)
    grouped = {}
    for episode in accepted:
        episode["weight"] = 1.0 / cycle_counts[episode["physical_unit_id"]]
        key = canonical_json_hash({"age_source": episode["age_source"], "regime": episode["regime"]})
        grouped.setdefault(key, []).append(episode)
    groups = {}
    for key, episodes in sorted(grouped.items()):
        physical_ids = sorted({e["physical_unit_id"] for e in episodes})
        survival, curve = 1.0, []
        for age in sorted({e["exit_age_s"] for e in episodes if e["event_observed"]}):
            risk = sum(e["weight"] for e in episodes if e["entry_age_s"] <= age <= e["exit_age_s"])
            failures = sum(e["weight"] for e in episodes if e["exit_age_s"] == age and e["event_observed"])
            survival *= max(0.0, 1.0 - failures / risk)
            curve.append({"age_s": age, "survival": survival, "risk_weight": risk, "event_weight": failures})
        groups[key] = {"age_source": episodes[0]["age_source"], "regime": episodes[0]["regime"],
                       "physical_unit_ids": physical_ids, "physical_unit_count": len(physical_ids),
                       "episode_count": len(episodes), "observed_event_count": sum(e["event_observed"] for e in episodes),
                       "supported": len(physical_ids) >= min_group_units, "curve": curve,
                       "support_intervals_s": _support_intervals(episodes), "episodes": episodes}
    state = {"version": VERSION, "fit_part": "train", "train_unit_ids": ids,
             "min_group_units": min_group_units, "weighting": "one_total_weight_per_physical_unit",
             "age_policy": "explicit_known_age_only_no_timestamp_imputation",
             "groups": groups, "excluded_counts": dict(sorted(exclusions.items())),
             "selected_episode_count": len(selected), "accepted_episode_count": len(accepted)}
    state["state_hash"] = canonical_json_hash(state)
    return state


def _prediction(grid, values, *, status, reason=None, **metadata):
    hazard, previous = [], 0.0
    for value in values:
        hazard.append(None if value is None or previous is None else 0.0 if previous >= 1 else
                      float(np.clip((value - previous) / (1 - previous), 0, 1)))
        previous = value
    return {"horizons_s": grid, "probabilities": values,
            "probability_by_horizon": {str(h): value for h, value in zip(grid, values)},
            "hazard": hazard, "status": status, "reason": reason, **metadata}


def predict_age_baseline(state: dict, age_s, horizons_s, *, regime=None, age_source=None) -> dict:
    """Predict 1-S(age+h)/S(age) only within continuously observed age support."""
    grid = validate_horizons(horizons_s)
    if state.get("version") != VERSION:
        raise ValueError("Unsupported age baseline state")
    null = [None] * len(grid)
    age = _number(age_s)
    if age is None or age < 0:
        return _prediction(grid, null, status="unsupported", reason="unknown_age")
    requested_regime = _regime(regime)
    candidates = [(key, group) for key, group in state["groups"].items()
                  if group["regime"] == requested_regime and
                  (age_source is None or group["age_source"] == age_source)]
    if len(candidates) != 1:
        return _prediction(grid, null, status="unsupported", reason="unknown_or_ambiguous_age_source_or_regime")
    key, group = candidates[0]
    if not group["supported"]:
        return _prediction(grid, null, status="unsupported", reason="insufficient_physical_units", group_key=key)
    interval = next((bounds for bounds in group["support_intervals_s"] if bounds[0] <= age < bounds[1]), None)
    if interval is None:
        return _prediction(grid, null, status="unsupported", reason="age_outside_observed_support", group_key=key)
    def survival(t):
        return next((point["survival"] for point in reversed(group["curve"]) if point["age_s"] <= t), 1.0)
    current = survival(age)
    if current <= 0:
        return _prediction(grid, null, status="unsupported", reason="no_surviving_risk_set", group_key=key)
    values = [float(np.clip(1.0 - survival(age + h) / current, 0, 1)) if age + h <= interval[1] else None for h in grid]
    return _prediction(grid, values, status="supported" if all(v is not None for v in values) else "partial_support",
                       reason=None if all(v is not None for v in values) else "horizon_beyond_observed_support",
                       group_key=key, age_source=group["age_source"], age_s=age,
                       support_interval_s=interval)


def always_no_entry(horizons_s) -> dict:
    """Explicit constant reference; availability/risk masks remain the caller's job."""
    grid = validate_horizons(horizons_s)
    return _prediction(grid, [0.0] * len(grid), status="reference", baseline="always_no_entry")


def _current_cycle(prefix: pd.DataFrame) -> pd.DataFrame:
    frame = prefix.copy().reset_index(drop=True)
    if not {"timestamp_s", "signal"}.issubset(frame):
        raise ValueError("Prefix requires timestamp_s and signal")
    times = pd.to_numeric(frame.timestamp_s, errors="coerce").to_numpy(float)
    if not np.isfinite(times).all() or (np.diff(times) <= 0).any():
        raise ValueError("Prefix timestamps must be finite and strictly increasing")
    cycle = next((name for name in ("cycle_id", "component_cycle_id") if name in frame), None)
    if cycle and len(frame):
        ids = frame[cycle].astype(str).to_numpy()
        changes = np.flatnonzero(ids[1:] != ids[:-1]) + 1
        if len(changes):
            frame = frame.iloc[int(changes[-1]):].reset_index(drop=True)
    elif "component_replaced" in frame and len(frame):
        replacement = frame.component_replaced.fillna(False).eq(True).to_numpy()
        indices = np.flatnonzero(replacement)
        if len(indices):
            frame = frame.iloc[int(indices[-1]):].reset_index(drop=True)
    return frame


def trend_to_red(prefix: pd.DataFrame, schema: dict, horizons_s) -> dict:
    """Causal least-squares extrapolation of the last eight usable measurements.

    Unknown first-event history, unknown relative rule and current/post RED do
    not yield negative event scores. The output is a deterministic reference.
    """
    grid = validate_horizons(horizons_s)
    null = [None] * len(grid)
    frame = _current_cycle(prefix)
    if frame.empty:
        return _prediction(grid, null, status="unsupported", reason="empty_prefix")
    values = pd.to_numeric(frame.signal, errors="coerce").to_numpy(float)
    quality = np.isfinite(values)
    for name in ("quality_ok", "usable"):
        if name in frame:
            quality &= frame[name].fillna(False).eq(True).to_numpy()
    if "quality_status" in frame:
        quality &= frame.quality_status.fillna("good").astype(str).str.lower().isin(["good", "ok", "usable", "valid"]).to_numpy()
    gaps = frame.get("gap_before", pd.Series(False, index=frame.index)).fillna(False).eq(True).to_numpy()
    if not quality.all() or gaps[1:].any():
        return _prediction(grid, null, status="unsupported", reason="unknown_first_event_history")
    thresholds = resolve_thresholds(schema, frame)
    if thresholds["status"] != "available":
        return _prediction(grid, null, status="unsupported", reason="rule_unavailable")
    n = int((schema.get("thresholds") or {}).get("baseline_n", 1)) if thresholds["mode"] == "initial_baseline_multiple" else 1
    limit = float(thresholds["red"])
    direction = thresholds["direction"]
    relevant = values[n - 1:]
    if ((relevant >= limit) if direction == "above" else (relevant <= limit)).any():
        return _prediction(grid, null, status="unsupported", reason="event_or_post_event")
    if len(frame) < 2:
        return _prediction(grid, null, status="unsupported", reason="insufficient_trend_history")
    recent = frame.iloc[-8:]
    x = recent.timestamp_s.to_numpy(float)
    x = x - x[-1]
    signal = recent.signal.to_numpy(float)
    slope = float(np.dot(x - x.mean(), signal - signal.mean()) / np.dot(x - x.mean(), x - x.mean()))
    toward = slope > 0 if direction == "above" else slope < 0
    crossing = (limit - values[-1]) / slope if toward else None
    predictions = [float(crossing is not None and 0 < crossing <= h) for h in grid]
    return _prediction(grid, predictions, status="reference", baseline="trend_to_red",
                       crossing_delay_s=crossing, slope_per_s=slope, red_limit=limit,
                       calibration="deterministic_uncalibrated")


def last_value_signal(prefix: pd.DataFrame, horizons_s) -> dict:
    """Auxiliary numeric forecast; never interpreted as RED-entry probability."""
    grid = validate_horizons(horizons_s)
    frame = _current_cycle(prefix)
    value = _number(frame.signal.iloc[-1]) if len(frame) else None
    if len(frame):
        for name in ("quality_ok", "usable"):
            if name in frame and not bool(frame[name].fillna(False).eq(True).iloc[-1]):
                value = None
        if "quality_status" in frame and str(frame.quality_status.iloc[-1]).lower() not in {"good", "ok", "usable", "valid"}:
            value = None
    return {"baseline": "last_value_signal", "task": "signal_forecast", "horizons_s": grid,
            "values": [value] * len(grid), "signal_by_horizon": {str(h): value for h in grid},
            "status": "supported" if value is not None else "unsupported"}


BASELINE_ENGINES = ("kaplan_meier", "always_no_entry", "trend_to_red")


def fit_baseline_model(engine_id, train, validation, config, *, events=None,
                       should_stop=None, status_cb=None):
    """Adapt references to the v2 training engine contract; validation never fits KM."""
    if engine_id not in BASELINE_ENGINES:
        raise ValueError(f"Unsupported baseline engine: {engine_id}")
    if should_stop is not None and should_stop():
        raise InterruptedError("Baseline fit cancelled")
    grid = validate_horizons(config["horizons_s"])
    model = {"version": "red_entry_baseline_adapter_v1", "engine_id": engine_id, "horizons_s": grid}
    origins = train["origins"]
    ids = config.get("train_unit_ids", sorted(set(origins.unit_id.astype(str))))
    if engine_id == "kaplan_meier":
        event_frame = events if events is not None else config.get("events")
        if not isinstance(event_frame, pd.DataFrame):
            raise ValueError("Age KM fit requires original event table")
        model["state"] = fit_age_baseline(event_frame, train_unit_ids=ids,
                                           min_group_units=config.get("min_group_units", 2))
    if engine_id == "trend_to_red":
        effective_schema = config.get("schema", train.get("schema"))
        if effective_schema is None:
            raise ValueError("Trend baseline requires explicit effective schema")
        model["schema"] = effective_schema
    if status_cb is not None:
        status_cb({"phase": "baseline_fit", "engine_id": engine_id, "completed": True})
    selection = {"selection": "fixed_reference_no_validation_fitting", "fit_part": "train",
                 "train_unit_ids": sorted(set(map(str, ids)))}
    canonical_json_hash(model)
    return model, selection


def _batch_prefix(batch, config, index, origin):
    supplied = batch.get("prefixes")
    if supplied is not None:
        if len(supplied) != len(batch["origins"]):
            raise ValueError("Prefixes must align with origins")
        prefix = supplied[index]
        if len(prefix) and (prefix.timestamp_s > origin.timestamp_s).any():
            raise ValueError("Prefix contains future measurements")
        return prefix
    features = batch.get("features", config.get("features"))
    if not isinstance(features, pd.DataFrame):
        return None
    return features[(features.unit_id.astype(str) == str(origin.unit_id)) &
                    (features.timestamp_s <= float(origin.timestamp_s))].sort_values("timestamp_s", kind="stable")


def predict_baseline_hazard(model, batch, config=None):
    """Return [N,K] hazards, with unsupported predictions represented by NaN.

    ``np.isfinite(result)`` is the prediction support mask. Target masks are
    separate and must also be applied by evaluation. Numerical x tensors cannot
    reconstruct the original age/rule/prefix semantics for these references.
    """
    config = config or {}
    grid = validate_horizons(model["horizons_s"])
    origins = batch["origins"]
    result = np.full((len(origins), len(grid)), np.nan, dtype=float)
    for index, (_, origin) in enumerate(origins.iterrows()):
        if "at_risk" in origin and not bool(origin.at_risk):
            continue
        engine = model["engine_id"]
        if engine == "always_no_entry":
            prediction = always_no_entry(grid)
        else:
            prefix = _batch_prefix(batch, config, index, origin)
            if engine == "trend_to_red":
                if prefix is None:
                    continue
                prediction = trend_to_red(prefix, model["schema"], grid)
            elif engine == "kaplan_meier":
                row = prefix.iloc[-1] if prefix is not None and len(prefix) else origin
                known = row.get("operating_age_known", False)
                if pd.isna(known) or not bool(known):
                    continue
                regime = {name: row[name] for name in REGIME_FIELDS if name in row and pd.notna(row[name])}
                prediction = predict_age_baseline(model["state"], row.get("operating_age_s"), grid,
                                                    regime=regime, age_source=row.get("operating_age_source", "unknown"))
            else:
                raise ValueError(f"Unsupported baseline engine: {engine}")
        result[index] = [np.nan if value is None else value for value in prediction["hazard"]]
    return result
