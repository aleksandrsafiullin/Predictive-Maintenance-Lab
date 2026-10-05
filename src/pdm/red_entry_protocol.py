"""Versioned, inert RED-entry contracts; no changes to historical models or labels.

Persist the returned contract alongside a future run. Project display limits must
be passed explicitly: they live outside immutable snapshot schema and can change
without changing snapshot_id. This module does not fit targets or models.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "configs/red_entry_protocol.yaml"
PARTS = ("train", "validation", "test", "holdout")
TASKS = ("signal_forecast", "red_entry", "legacy_rul")
INPUT_MODES = ("age_context", "sensor_only", "hybrid")


def canonical_json_hash(value: Any) -> str:
    """Full SHA-256 of canonical UTF-8 JSON; reject NaN and unsupported objects."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_horizons(horizons: Sequence[float]) -> list[float]:
    if isinstance(horizons, (str, bytes)) or not horizons:
        raise ValueError("Horizons must be a nonempty ordered grid in seconds")
    values = []
    for item in horizons:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError("Horizons must be finite numeric seconds")
        value = float(item)
        if not math.isfinite(value) or value <= 0 or (values and value <= values[-1]):
            raise ValueError("Horizons must be positive, finite and strictly increasing")
        values.append(value)
    return values


def validate_protocol(protocol: Mapping) -> dict:
    out = copy.deepcopy(dict(protocol))
    if out.get("protocol_version") != "red_entry_protocol_v1":
        raise ValueError("Unsupported RED-entry protocol version")
    if set(out.get("tasks", {})) != set(TASKS):
        raise ValueError("Protocol must explicitly separate all three tasks")
    for task in TASKS:
        if not out["tasks"][task].get("version") or not out["tasks"][task].get("target"):
            raise ValueError("Every task needs a version and target")
    if set(out.get("input_modes", {})) != set(INPUT_MODES):
        raise ValueError("Protocol needs all three explicit input modes")
    expected = {"age_context": {"known_age", "operating_context", "quality"},
                "sensor_only": {"sensor", "quality"},
                "hybrid": {"sensor", "known_age", "operating_context", "quality"}}
    if any(set(out["input_modes"][name]) != roles for name, roles in expected.items()):
        raise ValueError("Input mode roles must preserve the age/sensor ablation contract")
    for grid in out["horizons_s"].values():
        if grid is not None:
            validate_horizons(grid)
    training = out["training"]
    if training.get("transform_fit_part") != "train" or any(
        training.get(key) != "validation" for key in
        ("model_selection_part", "calibration_part", "alert_policy_selection_part")
    ):
        raise ValueError("Transforms fit Train; selection, calibration and policy use development")
    if training.get("frozen_before_model_comparison") is not True:
        raise ValueError("Development/training defaults must be frozen before comparison")
    canonical_json_hash(out)
    return out


def load_protocol(path: str | Path | None = None) -> dict:
    with Path(path or DEFAULT_PROTOCOL_PATH).open(encoding="utf-8") as stream:
        return validate_protocol(yaml.safe_load(stream))


def red_rule_identity(schema: Mapping, *, effective_thresholds: Mapping | None = None) -> dict:
    """Bind RED semantics to the effective rule, not merely the imported snapshot.

    Yellow/display text is preserved for audit but excluded from RED event hash.
    Relative baseline quality semantics describe existing project_zones v1, not a
    stronger quality filter. A future quality policy needs a new rule version.
    """
    saved = copy.deepcopy(dict(schema.get("thresholds") or {}))
    rule = copy.deepcopy(dict(effective_thresholds if effective_thresholds is not None else saved))
    mode, direction = rule.get("mode", "absolute"), rule.get("direction", "above")
    if direction not in {"above", "below"}:
        raise ValueError("Unsupported RED direction")
    semantic = {"version": "project_red_rule_v1", "signal_column": schema.get("signal_column"),
                "signal_unit": schema.get("signal_unit"), "mode": mode, "direction": direction,
                "comparison": "inclusive", "confirmation": "instantaneous_measurement",
                "admissible_operating_regimes": rule.get("admissible_operating_regimes", "all"),
                "baseline_quality_policy": "finite_initial_admitted_signal_rows_v1"}
    if mode == "absolute":
        raw = rule.get("red")
        if isinstance(raw, bool) or raw is None or not math.isfinite(float(raw)):
            raise ValueError("Absolute RED needs a finite limit")
        semantic["red"] = float(raw)
        semantic["baseline_quality_policy"] = "not_applicable"
    elif mode == "initial_baseline_multiple":
        n, ratio = rule.get("baseline_n", 5), rule.get("red_ratio", 2.0)
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("Relative RED needs positive integer baseline_n")
        if isinstance(ratio, bool) or not math.isfinite(float(ratio)) or float(ratio) <= 0:
            raise ValueError("Relative RED needs a finite positive red_ratio")
        semantic.update(baseline_n=n, red_ratio=float(ratio), baseline_statistic="median",
                        baseline_availability="after_first_n_admitted_measurements")
    else:
        raise ValueError("Unsupported or missing RED rule")
    return {"red_rule_version": semantic["version"], "red_rule_hash": canonical_json_hash(semantic),
            "semantics": semantic, "saved_thresholds": saved, "effective_thresholds": rule,
            "threshold_source": "project_override" if effective_thresholds is not None else "snapshot_schema"}


def validate_physical_unit_split(units: pd.DataFrame, split: Mapping) -> dict:
    """Reject equipment leakage even when cycle/unit IDs differ; do not guess IDs.

    unit_id fallback allows v1 inspection but explicitly cannot prove equipment
    independence. All rows of a unit must agree on its physical identity.
    """
    if "unit_id" not in units or units["unit_id"].isna().any():
        raise ValueError("Units require nonmissing unit_id")
    identity = next((name for name in ("physical_unit_id", "physical_equipment_id", "equipment_id")
                     if name in units), "unit_id")
    if units[identity].isna().any() or units[identity].astype(str).str.strip().eq("").any():
        raise ValueError("Physical identity cannot be missing")
    groups = units.assign(_unit=units.unit_id.astype(str), _physical=units[identity].astype(str))
    if (groups.groupby("_unit")["_physical"].nunique() != 1).any():
        raise ValueError("One unit_id refers to inconsistent physical equipment")
    by_unit = dict(zip(groups._unit, groups._physical))
    assignment, physical_parts, normalized = {}, {}, {}
    for part in PARTS:
        ids = [str(uid) for uid in split.get(part, [])]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate unit_id in {part}")
        normalized[part] = sorted(ids)
        for uid in ids:
            if uid not in by_unit:
                raise ValueError(f"Unknown split unit: {uid}")
            if uid in assignment:
                raise ValueError(f"Unit split leakage: {uid}")
            assignment[uid] = part
            physical = by_unit[uid]
            if physical in physical_parts and physical_parts[physical] != part:
                raise ValueError(f"Physical equipment split leakage: {physical}")
            physical_parts[physical] = part
    if set(assignment) != set(by_unit):
        raise ValueError("Every unit must have exactly one split assignment")
    return {"unit_ids_by_part": normalized, "physical_ids_by_part": {
        part: sorted({by_unit[uid] for uid in normalized[part]}) for part in PARTS},
        "physical_identity_column": identity, "physical_identity_verified": identity != "unit_id",
        "split_hash": canonical_json_hash({"assignments": normalized, "physical_mapping": by_unit})}


def freeze_split_provenance(units: pd.DataFrame, split: Mapping, *,
                            outcomes_inspected_parts: Sequence[str] = ("test",),
                            used_for_selection_parts: Sequence[str] = ("validation",),
                            previous: Mapping | None = None) -> dict:
    """Accumulate physical exposure: reshuffling an explored unit cannot clean it."""
    report = validate_physical_unit_split(units, split)
    for part in (*outcomes_inspected_parts, *used_for_selection_parts):
        if part not in PARTS:
            raise ValueError(f"Unknown provenance split part: {part}")
    exposed = set((previous or {}).get("exposed_physical_ids", []))
    exposure_parts = {"train", *outcomes_inspected_parts, *used_for_selection_parts}
    for part in exposure_parts:
        exposed.update(report["physical_ids_by_part"][part])
    holdout = report["physical_ids_by_part"]["holdout"]
    independent = bool(holdout) and not exposed.intersection(holdout) and report["physical_identity_verified"]
    report.update(provenance_version="physical_split_provenance_v1",
                  source_protocol=str(split.get("protocol") or "unspecified"),
                  outcomes_inspected_parts=sorted(set(outcomes_inspected_parts)),
                  used_for_selection_parts=sorted(set(used_for_selection_parts)),
                  exposed_physical_ids=sorted(exposed), independent_holdout_available=independent,
                  historical_test_status="explored" if "test" in outcomes_inspected_parts else "uninspected",
                  previous_provenance_hash=canonical_json_hash(previous) if previous else None)
    report["provenance_hash"] = canonical_json_hash(report)
    return report


def admission_status(protocol: Mapping, provenance: Mapping) -> dict:
    """Stage A cannot award quality: no evaluation evidence exists in this module."""
    unset = [name for name, value in protocol["operational_requirements"].items() if value is None]
    reasons = []
    if unset:
        reasons.append("operational_requirements_unset")
    if not provenance.get("independent_holdout_available"):
        reasons.append("no_independent_holdout")
    reasons.append("quality_evaluation_not_performed")
    return {"status": "requirements_unset" if unset else "not_passed",
            "can_pass": False, "reason_codes": reasons, "unset_requirements": unset}


def build_run_contract(snapshot: Mapping, *, task: str = "red_entry", input_mode: str = "hybrid",
                       effective_thresholds: Mapping | None = None, protocol: Mapping | None = None,
                       split_provenance: Mapping | None = None,
                       horizons_s: Sequence[float] | None = None) -> dict:
    """Build an auditable future run header; never reinterpret existing runs."""
    config = validate_protocol(protocol) if protocol is not None else load_protocol()
    if task not in TASKS or input_mode not in INPUT_MODES:
        raise ValueError("Unsupported task or input mode")
    schema = copy.deepcopy(dict(snapshot["schema"]))
    fresh = validate_physical_unit_split(snapshot["units"], snapshot["split"])
    provenance = copy.deepcopy(dict(split_provenance)) if split_provenance else freeze_split_provenance(
        snapshot["units"], snapshot["split"])
    if provenance.get("split_hash") != fresh["split_hash"]:
        raise ValueError("Split provenance does not match snapshot physical assignments")
    grid = horizons_s if horizons_s is not None else config["horizons_s"].get(schema.get("source_kind"))
    if grid is None:
        raise ValueError("Source needs an explicit horizon grid in seconds")
    out = {"contract_version": "red_entry_run_contract_v1", "task": task,
           "task_version": config["tasks"][task]["version"], "target": config["tasks"][task]["target"],
           "snapshot_id": snapshot["snapshot_id"], "snapshot_schema_hash": canonical_json_hash(schema),
           "protocol_hash": canonical_json_hash(config), "protocol": config,
           "input_mode": input_mode, "allowed_input_roles": config["input_modes"][input_mode],
           "horizons_s": validate_horizons(grid), "horizon_unit": "seconds",
           "split_provenance": provenance, "quality_gate": admission_status(config, provenance)}
    if task == "red_entry":
        out["red_rule"] = red_rule_identity(schema, effective_thresholds=effective_thresholds)
        out["event"] = config["event"]
    else:
        out["red_rule"] = None
    out["contract_hash"] = canonical_json_hash(out)
    return out


def assert_run_compatible(contract: Mapping, snapshot: Mapping, *,
                          effective_thresholds: Mapping | None = None) -> None:
    """Reject inference under changed snapshot, schema, split or effective RED."""
    if contract.get("snapshot_id") != snapshot["snapshot_id"]:
        raise ValueError("Run snapshot does not match inference snapshot")
    if contract.get("snapshot_schema_hash") != canonical_json_hash(snapshot["schema"]):
        raise ValueError("Run schema does not match inference snapshot")
    split = validate_physical_unit_split(snapshot["units"], snapshot["split"])
    if contract["split_provenance"]["split_hash"] != split["split_hash"]:
        raise ValueError("Run physical split does not match inference snapshot")
    if contract["task"] == "red_entry":
        effective = red_rule_identity(snapshot["schema"], effective_thresholds=effective_thresholds)
        if contract["red_rule"]["red_rule_hash"] != effective["red_rule_hash"]:
            raise ValueError("Effective RED rule changed; event run requires new targets/evaluation")
