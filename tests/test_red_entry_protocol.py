import copy

import pandas as pd
import pytest

from pdm.red_entry_protocol import (
    admission_status,
    assert_run_compatible,
    build_run_contract,
    canonical_json_hash,
    freeze_split_provenance,
    load_protocol,
    red_rule_identity,
    validate_horizons,
    validate_physical_unit_split,
    validate_protocol,
)


def snapshot():
    return {"snapshot_id": "v1-snapshot",
            "schema": {"source_kind": "xjtu_bearings", "signal_column": "rms", "signal_unit": "g",
                       "thresholds": {"mode": "absolute", "direction": "above", "red": 3, "yellow": 2}},
            "units": pd.DataFrame({"unit_id": ["a", "b", "c", "d"],
                                   "physical_unit_id": ["A", "B", "C", "D"]}),
            "split": {"train": ["a"], "validation": ["b"], "test": ["c"], "holdout": ["d"]}}


def test_canonical_hash_order_unicode_and_no_invalid_json():
    assert canonical_json_hash({"b": [1, "я"], "a": 2}) == canonical_json_hash({"a": 2, "b": [1, "я"]})
    assert canonical_json_hash([1, 2]) != canonical_json_hash([2, 1])
    with pytest.raises(ValueError):
        canonical_json_hash({"value": float("nan")})


@pytest.mark.parametrize("grid", [[], [0, 1], [2, 1], [1, 1], [1, float("inf")], [True], ["1"]])
def test_bad_horizons_rejected(grid):
    with pytest.raises(ValueError):
        validate_horizons(grid)


def test_explicit_config_separates_tasks_and_freezes_development():
    config = load_protocol()
    assert len({task["target"] for task in config["tasks"].values()}) == 3
    assert all(value is None for value in config["operational_requirements"].values())
    config["training"]["calibration_part"] = "test"
    with pytest.raises(ValueError, match="development"):
        validate_protocol(config)


@pytest.mark.parametrize("role", ["known_age", "operating_context"])
def test_sensor_only_cannot_acquire_age_or_context_role(role):
    config = load_protocol()
    assert set(config["input_modes"]["sensor_only"]) == {"sensor", "quality"}
    config["input_modes"]["sensor_only"].append(role)
    with pytest.raises(ValueError, match="ablation"):
        validate_protocol(config)


def test_effective_rule_identity_not_snapshot_id_tracks_event_semantics():
    data = snapshot()
    saved = red_rule_identity(data["schema"])
    override = {"mode": "absolute", "direction": "above", "red": 4, "yellow": 2}
    effective = red_rule_identity(data["schema"], effective_thresholds=override)
    assert effective["red_rule_hash"] != saved["red_rule_hash"]
    assert effective["saved_thresholds"]["red"] == 3
    assert effective["threshold_source"] == "project_override"
    override["yellow"] = 1
    assert red_rule_identity(data["schema"], effective_thresholds=override)["red_rule_hash"] == effective["red_rule_hash"]
    assert data["schema"]["thresholds"]["red"] == 3


def test_relative_rule_defaults_and_units_have_identity():
    schema = snapshot()["schema"]
    schema["thresholds"] = {"mode": "initial_baseline_multiple"}
    default = red_rule_identity(schema)
    schema["thresholds"].update(baseline_n=5, red_ratio=2.0, direction="above")
    assert red_rule_identity(schema)["red_rule_hash"] == default["red_rule_hash"]
    assert default["semantics"]["baseline_availability"] == "after_first_n_admitted_measurements"
    schema["signal_unit"] = "m/s2"
    assert red_rule_identity(schema)["red_rule_hash"] != default["red_rule_hash"]


@pytest.mark.parametrize("edit", ["cross_cycle", "duplicate", "missing", "unknown"])
def test_split_validation_rejects_physical_leak_and_bad_coverage(edit):
    data = snapshot()
    if edit == "cross_cycle":
        data["units"].loc[1, "physical_unit_id"] = "A"
    elif edit == "duplicate":
        data["split"]["train"].append("a")
    elif edit == "missing":
        data["split"]["test"] = []
    else:
        data["split"]["test"] = ["missing"]
    with pytest.raises(ValueError):
        validate_physical_unit_split(data["units"], data["split"])


def test_v1_unit_proxy_not_verified_holdout():
    data = snapshot()
    data["units"] = data["units"].drop(columns="physical_unit_id")
    provenance = freeze_split_provenance(data["units"], data["split"])
    assert not provenance["physical_identity_verified"]
    assert not provenance["independent_holdout_available"]


def test_explored_test_cannot_become_independent_by_swapping():
    data = snapshot()
    first = freeze_split_provenance(data["units"], data["split"])
    assert first["historical_test_status"] == "explored"
    assert first["independent_holdout_available"]
    data["split"]["test"], data["split"]["holdout"] = ["d"], ["c"]
    second = freeze_split_provenance(data["units"], data["split"], previous=first)
    assert not second["independent_holdout_available"]
    assert second["previous_provenance_hash"] == canonical_json_hash(first)


def test_stage_a_never_passes_without_downstream_quality_evidence():
    data = snapshot()
    data["split"]["test"].extend(data["split"].pop("holdout"))
    provenance = freeze_split_provenance(data["units"], data["split"])
    config = load_protocol()
    status = admission_status(config, provenance)
    assert status["status"] == "requirements_unset" and not status["can_pass"]
    assert "no_independent_holdout" in status["reason_codes"]
    config["operational_requirements"] = {name: 1 for name in config["operational_requirements"]}
    assert admission_status(config, provenance)["status"] == "not_passed"


def test_contract_frozen_override_compatibility_and_task_scope():
    data = snapshot()
    override = {"mode": "absolute", "direction": "above", "red": 4, "yellow": 2}
    contract = build_run_contract(data, effective_thresholds=override)
    assert_run_compatible(contract, data, effective_thresholds=override)
    with pytest.raises(ValueError, match="Effective RED"):
        assert_run_compatible(contract, data)
    override["red"] = 5
    assert contract["red_rule"]["effective_thresholds"]["red"] == 4
    changed = copy.deepcopy(data)
    changed["snapshot_id"] = "new"
    with pytest.raises(ValueError, match="snapshot"):
        assert_run_compatible(contract, changed)
    assert build_run_contract(data, task="signal_forecast")["red_rule"] is None
    assert build_run_contract(data, task="legacy_rul")["target"] == "historical_endpoint_remaining_time"


def test_generic_requires_explicit_horizons_and_stale_split_provenance_rejected():
    data = snapshot()
    data["schema"]["source_kind"] = "generic_sensor_csv"
    with pytest.raises(ValueError, match="explicit horizon"):
        build_run_contract(data)
    assert build_run_contract(data, horizons_s=[10, 20])["horizons_s"] == [10.0, 20.0]
    provenance = freeze_split_provenance(data["units"], data["split"])
    data["split"]["train"], data["split"]["validation"] = ["b"], ["a"]
    with pytest.raises(ValueError, match="provenance"):
        build_run_contract(data, horizons_s=[10], split_provenance=provenance)
