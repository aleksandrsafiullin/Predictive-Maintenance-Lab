"""Rejected telemetry preserves target-only endpoint evidence and snapshot binding."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from pdm.data.generic_csv import read_generic_csv
from pdm.data.project_import import import_project
from pdm.data.project_prepare import load_snapshot, prepare_project
from pdm.projects import ProjectStore
from pdm.red_entry_context import context_schema
from pdm.red_entry_features import fit_feature_state, transform_prefix
from pdm.red_entry_targets import ENDPOINT_FIELDS, build_red_entry_targets


def read(tmp_path, signals, **columns):
    source = pd.DataFrame({"unit_id": ["u"] * len(signals), "timestamp_s": np.arange(len(signals)),
                           "sensor": signals, **columns})
    path = tmp_path / "sensor.csv"
    source.to_csv(path, index=False)
    frame, units, quality = read_generic_csv(
        {"primary": [{"path": path, "relative_path": path.name}]}, "sensor",
        context_mapping={name: name for name in columns})
    schema = context_schema(frame, {"signal_unit": "Pa", "input_columns": ["signal"],
                                   "thresholds": {"mode": "absolute", "red": 10}})
    return {"features": frame, "units": units, "schema": schema,
            "split": {"train": ["u"], "validation": [], "test": []},
            "report": {"quality": quality}}


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_rejected_endpoint_is_retained_without_red_or_feature_leak(tmp_path, bad):
    source = read(tmp_path, [1, bad, 2, 3],
                  confirmed_failure=[None, True, None, True],
                  confirmed_failure_timestamp_s=[None, 100, None, 200])
    assert np.isfinite(source["features"].signal).all()
    assert source["features"].timestamp_s.tolist() == [0, 2, 3]
    assert source["report"]["quality"]["endpoint_records"][0]["confirmed_failure_timestamp_s"] == 100
    result = build_red_entry_targets(source, [1, 2])
    event = result["events"].iloc[0]
    assert [record["timestamp_s"] for record in event.endpoint_records] == [1, 3]
    assert pd.isna(event.first_red_timestamp_s)
    assert result["hazard_targets"].sum() == 0
    assert result["origins"].risk_status.tolist() == ["at_risk", "unknown_event_history", "unknown_event_history"]
    baseline = copy.deepcopy(source)
    baseline["report"]["quality"].pop("endpoint_records")
    old = build_red_entry_targets(baseline, [1, 2])
    pd.testing.assert_frame_equal(result["origins"], old["origins"])
    np.testing.assert_array_equal(result["hazard_targets"], old["hazard_targets"])
    np.testing.assert_array_equal(result["hazard_mask"], old["hazard_mask"])
    assert result["target_hash"] != old["target_hash"]
    for mode in ("sensor_only", "age_context", "hybrid"):
        state = fit_feature_state(source, mode)
        assert not set(ENDPOINT_FIELDS) & set(state["source_specs"])
        np.testing.assert_array_equal(transform_prefix(source["features"], source["schema"], state)["x"],
                                      transform_prefix(baseline["features"], baseline["schema"], state)["x"])


def test_skipped_replacement_with_same_cycle_id_starts_explicit_episode(tmp_path):
    source = read(tmp_path, [1, 10, np.nan, 1, 10], component_cycle_id=["same"] * 5,
                  component_replaced=[False, False, True, False, False],
                  replacement_timestamp_s=[None, None, 2, None, None])
    assert source["features"].component_replaced.tolist() == [False, False, True, False]
    result = build_red_entry_targets(source, [1, 2])
    assert len(result["events"]) == 2
    assert not any(record.get("replacement_timestamp_s") == 2 for record in result["events"].iloc[0].endpoint_records)
    assert any(record.get("replacement_timestamp_s") == 2 for record in result["events"].iloc[1].endpoint_records)
    assert result["events"].first_red_timestamp_s.tolist() == [1, 4]
    assert result["origins"].iloc[2].at_risk


def test_logs_before_after_and_entirely_outside_admitted_episode(tmp_path):
    source = read(tmp_path, [np.nan, 1, 2, np.nan, np.nan, 3, 4, np.nan],
                  component_cycle_id=["a", "a", "a", "b", "b", "c", "c", "c"],
                  maintenance_timestamp_s=[0, None, None, 3, None, None, None, 7])
    result = build_red_entry_targets(source, [1, 2])
    assert len(result["events"]) == 2
    assert result["events"].iloc[0].endpoint_records[0]["timestamp_s"] == 0
    assert result["events"].iloc[1].endpoint_records[0]["timestamp_s"] == 7
    assert result["unassigned_endpoint_records"][0]["component_cycle_id"] == "b"
    assert result["unassigned_endpoint_records"][0]["timestamp_s"] == 3
    assert result["hazard_targets"].sum() == 0
    assert result["events"].observation_end_s.tolist() == [2, 6]


def test_development_targets_exclude_test_logs_even_when_full_report_is_present(tmp_path):
    source = read(tmp_path, [1, np.nan, 2], confirmed_failure_timestamp_s=[None, 5, None])
    held_out = source["features"].copy()
    held_out["unit_id"] = "test_only"
    source["split"]["test"] = ["test_only"]
    test_record = copy.deepcopy(source["report"]["quality"]["endpoint_records"][0])
    test_record.update(unit_id="test_only", confirmed_failure_timestamp_s=999)
    source["report"]["quality"]["endpoint_records"].append(test_record)
    development = build_red_entry_targets(source, [1, 2])
    without_test = copy.deepcopy(source)
    without_test["report"]["quality"]["endpoint_records"].pop()
    assert development["target_hash"] == build_red_entry_targets(without_test, [1, 2])["target_hash"]
    assert development["unassigned_endpoint_records"] == []
    source["features"] = pd.concat([source["features"], held_out], ignore_index=True)
    full = build_red_entry_targets(source, [1, 2])
    assert full["events"].set_index("unit_id").loc["test_only"].endpoint_records[0]["confirmed_failure_timestamp_s"] == 999


def test_snapshot_reload_hash_binding_and_old_snapshot_compatibility(tmp_path):
    folder = tmp_path / "input"
    folder.mkdir()
    pd.DataFrame({"unit_id": uid, "timestamp_s": t, "sensor": np.nan if t == 1 else index + t,
                  "failure": 50 + index if t == 1 else None}
                 for index, uid in enumerate(["a", "b", "c"])
                 for t in range(4)).to_csv(folder / "sensor.csv", index=False)
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("Endpoints", "generic_sensor_csv")["project_id"]
    manifest = import_project(pid, {"primary": {"mode": "folder", "path": str(folder)},
        "signal_column": "sensor", "signal_unit": "Pa",
        "context_mapping": {"confirmed_failure_timestamp_s": "failure"},
        "thresholds": {"mode": "absolute", "direction": "above", "yellow": 5, "red": 10}}, store=store)
    prepare_project(pid, manifest["manifest_id"], store=store)
    source = load_snapshot(pid, store=store)
    assert len(source["report"]["quality"]["endpoint_records"]) == 3
    assert np.isfinite(source["features"].signal).all()
    result = build_red_entry_targets(source, [1, 2])
    assert sum(len(records) for records in result["events"].endpoint_records) == 3
    assert result["target_hash"] == build_red_entry_targets(load_snapshot(pid, store=store), [1, 2])["target_hash"]
    old = copy.deepcopy(source)
    old["report"]["quality"].pop("endpoint_records")
    absent = copy.deepcopy(old)
    absent.pop("report")
    assert build_red_entry_targets(old, [1, 2])["target_hash"] == build_red_entry_targets(absent, [1, 2])["target_hash"]
    report_path = source["dir"] / "data_report.json"
    report = json.loads(report_path.read_text())
    report["quality"]["endpoint_records"][0]["confirmed_failure_timestamp_s"] = 999
    report_path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="hash mismatch: data_report.json"):
        load_snapshot(pid, store=store)
