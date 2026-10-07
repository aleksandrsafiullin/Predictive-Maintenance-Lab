"""Causal timestamp-aligned windows and explicitly retrospective reference cases."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .contract import (
    canonical_hash,
    config_hash,
    default_config,
    is_dense_v2,
    is_reference_balanced,
    reference_origin_policy,
    threshold_crossed,
    unit_balanced_weights,
)


def causal_anchor_content_hash(history, targets, mask) -> str:
    """Exact physical content binding, ignoring unavailable target payloads."""
    supported = np.asarray(mask, dtype=bool)
    arrays = (np.asarray(history, dtype="<f8"), supported.astype("u1"),
              np.asarray(np.where(supported, targets, 0.), dtype="<f8"))
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(canonical_hash([array.dtype.str, list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _supplied_barriers(group: pd.DataFrame) -> np.ndarray:
    """Retain decoder/snapshot barriers even when retained timestamps align."""
    barriers = np.zeros(len(group), dtype=bool)
    if "gap_before" in group:
        barriers[1:] |= group["gap_before"].to_numpy(dtype=bool)[1:]
    if "segment_id" in group:
        segments = group["segment_id"].to_numpy()
        barriers[1:] |= segments[1:] != segments[:-1]
    # The first supplied row starts the observation stream, as in the old protocol.
    return barriers


def _segments(group: pd.DataFrame, config: dict) -> list[pd.DataFrame]:
    """Invalid rows are continuity barriers; a 120s interval is never a 60s step."""
    times = group["timestamp_s"].to_numpy(dtype=float)
    values = group[config["target"]].to_numpy(dtype=float)
    valid = np.isfinite(times) & np.isfinite(values) & (values > 0)
    if "valid" in group:
        valid &= group["valid"].to_numpy(dtype=bool)
    barriers = _supplied_barriers(group)
    finite = times[np.isfinite(times)]
    if (np.diff(finite) <= 0).any():
        raise ValueError("Duplicate or out-of-order timestamps cannot form windows")
    starts, result = None, []
    for i in range(len(group)):
        continuous = i > 0 and not barriers[i] and valid[i - 1] and valid[i] and abs(
            times[i] - times[i - 1] - config["cadence_s"]
        ) <= config["clock_tolerance_s"]
        if starts is not None and (not valid[i] or not continuous):
            result.append(group.iloc[starts:i].reset_index(drop=True))
            starts = None
        if valid[i] and starts is None:
            starts = i
    if starts is not None:
        result.append(group.iloc[starts:].reset_index(drop=True))
    return result


def _aligned(times: np.ndarray, index: int, length: int, config: dict) -> bool:
    offsets = times[index - length + 1:index + 1] - times[index]
    expected = config["cadence_s"] * np.arange(-length + 1, 1)
    return bool(np.all(np.abs(offsets - expected) <= config["clock_tolerance_s"]))


def _targets(segment: pd.DataFrame, index: int, config: dict) -> tuple[np.ndarray, np.ndarray]:
    horizon = config["max_horizon"]
    target = np.full(horizon, np.nan, dtype=float)
    mask = np.zeros(horizon, dtype=bool)
    times = segment["timestamp_s"].to_numpy(dtype=float)
    values = segment[config["target"]].to_numpy(dtype=float)
    supported = min(horizon, len(segment) - index - 1)
    offsets = times[index + 1:index + 1 + supported] - times[index]
    mismatches = np.flatnonzero(np.abs(offsets - config["cadence_s"] * np.arange(1, supported + 1))
                               > config["clock_tolerance_s"])
    if len(mismatches):
        supported = int(mismatches[0])
    target[:supported] = values[index + 1:index + 1 + supported]
    mask[:supported] = True
    return target, mask


def _rng(uid: str, purpose: str) -> np.random.Generator:
    # Intentionally identical to the supplied benchmark.py for evaluation-anchor.
    seed = int.from_bytes(
        hashlib.sha256(f"baseline-v1:{uid}:{purpose}".encode()).digest()[:8], "little"
    )
    return np.random.default_rng(seed)


def _prefix_status(group: pd.DataFrame, config: dict) -> list[dict]:
    """One past-event state per row; later rows cannot affect any earlier state."""
    times = group["timestamp_s"].to_numpy(dtype=float)
    values = group[config["target"]].to_numpy(dtype=float)
    valid = np.isfinite(times) & np.isfinite(values) & (values > 0)
    if "valid" in group:
        valid &= group["valid"].to_numpy(dtype=bool)
    barriers = _supplied_barriers(group)
    result, unknown, first_observed, first_unknown = [], False, None, False
    for index in range(len(group)):
        if not valid[index] or barriers[index] or (index > 0 and (
            not valid[index - 1] or abs(times[index] - times[index - 1] - config["cadence_s"])
            > config["clock_tolerance_s"]
        )):
            unknown = True
        if valid[index] and threshold_crossed(values[index], config["red"], config.get("threshold_direction", "above")) and first_observed is None:
            first_observed, first_unknown = float(times[index]), unknown
        event_unknown = unknown if first_observed is None else first_unknown
        result.append({
            "first_event_status": "already_observed" if first_observed is not None else
            "first_event_unknown" if event_unknown else "not_observed",
            "already_red": first_observed is not None, "first_event_unknown": event_unknown,
            "history_has_gaps": unknown,
            "first_red_s": first_observed if not first_unknown else None,
            "first_observed_red_s": first_observed,
            "current_value": float(values[index]) if valid[index] else None,
        })
    return result


def build_windows(frame: pd.DataFrame, config: dict, *, mode="train", origin_manifest=None) -> dict:
    config = default_config(**config)
    dense_v2 = is_dense_v2(config)
    balanced_train = mode == "train" and is_reference_balanced(config)
    if mode not in ("train", "validation", "reference", "rolling"):
        raise ValueError("Unsupported window protocol")
    observed_reference = mode == "reference" and config.get("observed_profile") is True and not dense_v2
    v2_reference = mode == "reference" and dense_v2
    physical_reference_policy = "observed-v1:sha256-physical-anchor:from-baseline-unit-anchors"
    split = frame.attrs.get("split")
    if mode in ("train", "validation") and split is not None and split != mode:
        raise ValueError("Training/Validation windows may only read their explicit assigned split")
    if frame.attrs.get("external") and mode in ("train", "validation"):
        raise ValueError("External evaluation data cannot form training windows")
    if not set(config["schema"]).issubset(frame.columns):
        raise ValueError("Missing observed signal schema")
    source_config = frame.attrs.get("config")
    if source_config and any(source_config[k] != config[k] for k in (
        "schema", "target", "unit", "cadence_s", "positive_domain", "clock_tolerance_s"
    )):
        raise ValueError("Window schema/physical cadence is incompatible with its source snapshot")
    binding = {key: frame.attrs[key] for key in (
        "snapshot_id", "dataset_hash", "release_id", "suite", "profile", "evaluation_status",
    ) if key in frame.attrs}
    release = binding.get("release_id", "unregistered")
    origin_contract = {key: config[key] for key in (
        "target", "unit", "schema", "cadence_s", "clock_tolerance_s", "positive_domain",
        "common_history_length", "max_horizon",
    )}
    if dense_v2:
        origin_contract["horizon_protocol"] = "dense-v2"
    # Old Above manifests stay exact. New event rules bind the causal past status.
    if config.get("threshold_direction", "above") == "below":
        origin_contract.update(threshold_direction="below", red=config["red"], yellow=config["yellow"])
    if "zone_rule_hash" in config:
        origin_contract.update(zone_rule_hash=config["zone_rule_hash"],
                               threshold_direction=config.get("threshold_direction", "above"),
                               red=config["red"], yellow=config["yellow"])
    if observed_reference:
        origin_contract["reference_physical_origin_policy"] = physical_reference_policy
    policy = reference_origin_policy(config) if mode == "reference" else (
        f"causal-{mode}:stride-{config['origin_stride']}:common-history-60"
        + (f":max-{config['max_origins_per_unit']}:seed-{config['seed']}" if mode == "train" else "")
    )
    if dense_v2 and mode != "reference":
        policy = "dense-v2:" + policy + (":reserve-earliest-per-segment" if mode == "train" else "")
    supplied = {}
    if origin_manifest is not None:
        unsigned = {k: v for k, v in origin_manifest.items() if k != "origin_hash"}
        if origin_manifest.get("origin_hash") != canonical_hash(unsigned):
            raise ValueError("Origin manifest integrity mismatch")
        for key in (
            "snapshot_id", "dataset_hash", "release_id", "suite", "profile", "evaluation_status",
        ):
            if origin_manifest.get(key) != binding.get(key):
                raise ValueError(f"Origin manifest source provenance mismatch: {key}")
        if origin_manifest.get("origin_contract") != origin_contract:
            raise ValueError("Origin manifest physical/window contract mismatch")
        if (
            origin_manifest.get("origin_policy") != policy
            or origin_manifest.get("common_history_length") != 60
            or origin_manifest.get("cadence_s") != config["cadence_s"]
            or origin_manifest.get("max_horizon") != config["max_horizon"]
        ):
            raise ValueError("Origin manifest protocol mismatch")
        supplied = {(r["unit_id"], float(r["origin_s"])): r for r in origin_manifest["records"]}
        if len(supplied) != len(origin_manifest["records"]):
            raise ValueError("Origin manifest has duplicate origins")
    xs, ys, masks, units, physical_units, origins, records = [], [], [], [], [], [], []
    past_status = []
    exclusions, common_count, shorter_count = [], 0, 0
    history_availability_by_unit = []
    seen_supplied = set()
    reserved_records = []
    causal_unit_anchors = []
    for uid, group in frame.groupby("unit_id", sort=True):
        uid = str(uid)
        statuses = _prefix_status(group, config)
        positions = {float(t): i for i, t in enumerate(group["timestamp_s"]) if np.isfinite(t)}
        physical = str(group["physical_unit_id"].iloc[0]) if "physical_unit_id" in group else release + ":" + uid
        if "physical_unit_id" in group and group["physical_unit_id"].nunique() != 1:
            raise ValueError("One unit cannot have multiple physical identities")
        candidates = []
        reserved = []
        causal_anchor = None
        unit_available_origins = 0
        for segment in _segments(group, config):
            segment_first = len(candidates)
            times = segment["timestamp_s"].to_numpy(dtype=float)
            configured_available = sum(_aligned(times, i, config["history_length"], config)
                                       for i in range(config["history_length"] - 1, len(segment)))
            shorter_count += configured_available
            unit_available_origins += configured_available
            for index in range(59, len(segment)):
                if not _aligned(times, index, 60, config):
                    continue
                common_count += 1
                if balanced_train and causal_anchor is None:
                    history = segment[config["target"]].iloc[index-config["history_length"]+1:index+1].to_numpy(dtype=float)
                    targets, supported = _targets(segment,index,config)
                    causal_anchor = {"unit_id":uid,"physical_unit_id":physical,"origin_s":float(times[index]),
                                     "content_hash":causal_anchor_content_hash(history,targets,supported)}
                    causal_unit_anchors.append(causal_anchor)
                if mode == "reference":
                    if not dense_v2 and (index + 60 >= len(segment) or not _targets(segment, index, config)[1].all()):
                        continue
                elif (index - 59) % config["origin_stride"]:
                    continue
                if mode in ("train", "validation") and not _targets(segment, index, config)[1].any():
                    continue
                candidates.append((segment, index))
            if dense_v2 and mode == "train" and len(candidates) > segment_first:
                reserved.append(segment_first)
        history_availability_by_unit.append({"unit_id": uid, "physical_unit_id": physical,
                                             "recorded_origins": len(group),
                                             "available_origins": unit_available_origins,
                                             "refused_origins": len(group) - unit_available_origins,
                                             "availability_rule": "configured_continuous_history_before_stride_or_future_support"})
        if v2_reference and candidates:
            chosen = [candidates[0]]
        elif dense_v2 and mode == "train":
            budget = config["max_origins_per_unit"]
            if len(reserved) > budget:
                raise ValueError("Train earliest segment reservations exceed max_origins_per_unit")
            remaining = np.setdiff1d(np.arange(len(candidates)), reserved)
            count = min(budget - len(reserved), len(remaining))
            picks = _rng(uid, f"dense-v2-train-origins:{config['seed']}").choice(remaining, count, replace=False)
            chosen = [candidates[int(i)] for i in sorted(set(reserved) | set(picks.tolist()))]
            reserved_records.extend({"unit_id": uid, "physical_unit_id": physical,
                                     "origin_s": float(candidates[i][0]["timestamp_s"].iloc[candidates[i][1]])}
                                    for i in reserved)
        elif origin_manifest is not None and not observed_reference:
            chosen = [(segment, index) for segment, index in candidates
                      if (uid, float(segment["timestamp_s"].iloc[index])) in supplied]
            if mode == "reference" and candidates:
                expected_segment, expected_index = candidates[int(
                    _rng(uid, "evaluation-anchor").integers(len(candidates))
                )]
                expected_origin = float(expected_segment["timestamp_s"].iloc[expected_index])
                if len(chosen) != 1 or float(chosen[0][0]["timestamp_s"].iloc[chosen[0][1]]) != expected_origin:
                    raise ValueError("Reference origin differs from original SHA256 anchor policy")
        elif mode == "reference" and candidates:
            chosen = [candidates[int(_rng(uid, "evaluation-anchor").integers(len(candidates)))]]
        elif mode == "train" and len(candidates) > config["max_origins_per_unit"]:
            picks = _rng(uid, f"shared-train-origins:{config['seed']}").choice(
                len(candidates), config["max_origins_per_unit"], replace=False
            )
            chosen = [candidates[int(i)] for i in sorted(picks)]
        else:
            chosen = candidates
        if not chosen:
            exclusions.append({"unit_id": uid, "physical_unit_id": physical,
                               "reason": "no_causal_common_history" if v2_reference else
                               "no_complete_reference_path" if mode == "reference" else "insufficient_history_or_target"})
        if mode == "reference" and len(chosen) > 1:
            raise ValueError("Reference population requires exactly one origin per physical unit")
        for segment, index in chosen:
            origin = float(segment["timestamp_s"].iloc[index])
            seen_supplied.add((uid, origin))
            if origin_manifest is not None and not (observed_reference or dense_v2) and supplied[(uid, origin)]["physical_unit_id"] != physical:
                raise ValueError("Origin manifest physical identity mismatch")
            x = segment[config["target"]].iloc[index - config["history_length"] + 1:index + 1].to_numpy(dtype=float)
            y, mask = _targets(segment, index, config)
            xs.append(x)
            ys.append(y)
            masks.append(mask)
            units.append(uid)
            physical_units.append(physical)
            origins.append(origin)
            records.append({"unit_id": uid, "physical_unit_id": physical, "origin_s": origin})
            past_status.append(statuses[positions[origin]])
    reference_selection = None
    if v2_reference:
        candidate_count = len(records)
        by_physical = {}
        for index, physical in enumerate(physical_units):
            by_physical.setdefault(physical, []).append(index)
        # The hash input is the canonical JSON pair [physical_id, unit_id].
        selected = sorted(min(indices, key=lambda i: (canonical_hash([physical_units[i], units[i]]), units[i]))
                          for _, indices in sorted(by_physical.items()))
        xs, ys, masks, units, physical_units, origins, records, past_status = (
            [values[index] for index in selected]
            for values in (xs, ys, masks, units, physical_units, origins, records, past_status)
        )
        reference_selection = {"policy": reference_origin_policy(config),
                               "population": "one_reference_origin_per_independent_physical_equipment",
                               "selection_information": "earliest_causal_common_history;sha256_canonical_json_[physical_id,unit_id];no_future_masks_or_event_labels",
                               "source_units": int(frame["unit_id"].nunique()),
                               "candidate_unit_origins": candidate_count,
                               "selected_physical_units": len(by_physical)}
        seen_supplied = {(record["unit_id"], record["origin_s"]) for record in records}
    if observed_reference:
        candidate_count = len(records)
        by_physical = {}
        for index, physical in enumerate(physical_units):
            by_physical.setdefault(physical, []).append(index)
        selected = sorted(indices[int(_rng(physical, "observed-physical-reference-anchor").integers(len(indices)))]
                          for physical, indices in sorted(by_physical.items()))
        xs, ys, masks, units, physical_units, origins, records, past_status = (
            [values[index] for index in selected]
            for values in (xs, ys, masks, units, physical_units, origins, records, past_status)
        )
        reference_selection = {"policy": physical_reference_policy,
                               "population": "one_reference_origin_per_independent_physical_equipment",
                               "selection_information": "physical_id_and_complete_unit_anchor_candidates;no_forecast_error_or_event_labels",
                               "source_units": int(frame["unit_id"].nunique()),
                               "candidate_unit_origins": candidate_count,
                               "selected_physical_units": len(by_physical)}
        seen_supplied = {(record["unit_id"], record["origin_s"]) for record in records}
        if origin_manifest is not None:
            if origin_manifest.get("reference_selection") != reference_selection or origin_manifest.get("records") != records:
                raise ValueError("Observed physical reference origin selection mismatch")
    if origin_manifest is not None and seen_supplied != set(supplied):
        raise ValueError("Origin manifest contains unavailable/noncausal histories")
    if dense_v2 and origin_manifest is not None and (origin_manifest.get("records") != records
            or origin_manifest.get("reference_selection") != reference_selection):
        raise ValueError("Dense-v2 deterministic origin selection mismatch")
    x = np.asarray(xs, dtype=float).reshape(-1, config["history_length"])
    y = np.asarray(ys, dtype=float).reshape(-1, config["max_horizon"])
    mask = np.asarray(masks, dtype=bool).reshape(-1, config["max_horizon"])
    units = np.asarray(units, dtype=str)
    physical_units = np.asarray(physical_units, dtype=str)
    manifest = {
        "origin_policy": policy, "common_history_length": 60,
        "cadence_s": config["cadence_s"], "max_horizon": config["max_horizon"], "records": records,
        "origin_contract": origin_contract,
        **binding,
    }
    if reference_selection is not None:
        manifest["reference_selection"] = reference_selection
    manifest["origin_hash"] = canonical_hash(manifest)
    result = {
        "x": x, "y": y, "mask": mask, "units": units, "physical_units": physical_units,
        "origin_s": np.asarray(origins, dtype=float),
        "past_status": past_status,
        "weights": unit_balanced_weights(mask, physical_units),
        "admission": {
            "total_units": int(frame["unit_id"].nunique()), "accepted_units": len(np.unique(physical_units)),
            "origins": len(x), "known_targets": int(mask.sum()), "complete_paths": int(mask.all(axis=1).sum()),
            "partial_paths": int((mask.any(axis=1) & ~mask.all(axis=1)).sum()),
            "empty_paths": int((~mask.any(axis=1)).sum()), "excluded_units": exclusions,
            "common_history_available_origins": common_count,
            "configured_history_available_origins": shorter_count,
            "additional_short_history_origins": shorter_count - common_count,
            "history_availability_by_unit": history_availability_by_unit,
        }, "origin_manifest": manifest, "origin_policy": policy,
        "config": config, "config_hash": config_hash(config), "mode": mode, **binding,
    }
    if split is not None:
        result["split"] = split
    if observed_reference or v2_reference:
        result["admission"]["source_units"] = result["admission"]["total_units"]
        result["admission"]["total_units"] = int(frame["physical_unit_id"].nunique()) if "physical_unit_id" in frame else int(frame["unit_id"].nunique())
    if dense_v2:
        result["admission"]["reserved_earliest_origins"] = reserved_records
        result["admission"]["reserved_earliest_origin_count"] = len(reserved_records)
    if balanced_train:
        by_physical = {}
        for index,record in enumerate(causal_unit_anchors):
            by_physical.setdefault(record["physical_unit_id"],[]).append(index)
        selected = sorted(min(indices,key=lambda i:(canonical_hash([
            causal_unit_anchors[i]["physical_unit_id"],causal_unit_anchors[i]["unit_id"]]),
            causal_unit_anchors[i]["unit_id"])) for indices in by_physical.values())
        evidence = {"origin_policy":reference_origin_policy(config),"config_hash":config_hash(config),
                    "rolling_origin_hash":manifest["origin_hash"],"candidate_unit_origins":len(causal_unit_anchors),
                    "records":[causal_unit_anchors[i] for i in selected]}
        evidence["evidence_hash"] = canonical_hash(evidence)
        result["admission"]["reference_train_evidence"] = evidence
    return result


def observed_history(frame: pd.DataFrame, unit_id: str, origin_s: float, config: dict) -> dict:
    """Inspect only the observed prefix; future availability never gates inference."""
    config = default_config(**config)
    group = frame.loc[frame["unit_id"].astype(str) == str(unit_id)]
    # Time locates Now; positional prefix keeps preceding invalid timestamps as barriers.
    matches = np.flatnonzero(np.abs(group["timestamp_s"].to_numpy(dtype=float) - origin_s)
                             <= config["clock_tolerance_s"])
    binding = {key: frame.attrs[key] for key in (
        "snapshot_id", "dataset_hash", "release_id", "suite", "profile", "split",
    ) if key in frame.attrs}
    result = {
        "status": "unknown_origin", "unit_id": str(unit_id), "origin_s": float(origin_s),
        "physical_unit_id": binding.get("release_id", "unregistered") + ":" + str(unit_id),
        "config": config, "config_hash": config_hash(config), **binding,
    }
    if len(matches) != 1:
        return result
    past = group.iloc[:int(matches[0]) + 1].copy()
    exact = past.iloc[-1:]
    if "physical_unit_id" in past:
        result["physical_unit_id"] = str(exact["physical_unit_id"].iloc[0])
    result["past_status"] = _prefix_status(past, config)[-1]
    if result["past_status"]["current_value"] is None:
        result["status"] = "invalid_observation"
        return result
    segments = _segments(past, config)
    segment = segments[-1]
    length = config["history_length"]
    if len(segment) < length or not _aligned(segment["timestamp_s"].to_numpy(dtype=float), len(segment) - 1, length, config):
        result["status"] = "insufficient_history"
        return result
    x = segment[config["target"]].iloc[-length:].to_numpy(dtype=float)
    result.update(status="available", x=x, history=x.copy(), history_timestamps=segment["timestamp_s"].iloc[-length:].to_numpy(dtype=float))
    return result
