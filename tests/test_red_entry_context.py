"""Physical age, context preservation and grouped snapshot contracts."""
import numpy as np
import pandas as pd
import pytest

from pdm.data.generic_csv import read_generic_csv
from pdm.data.project_prepare import _canonicalize_adapted, allocate_project_split
from pdm.red_entry_context import add_context, assert_physical_split, context_schema
from pdm.red_entry_features import fit_feature_state, transform_prefix
from pdm.red_entry_targets import ENDPOINT_FIELDS, build_red_entry_targets


def history(**extra):
    return pd.DataFrame({"unit_id": ["u"] * 4, "timestamp_s": [0., 10., 20., 100.],
                         "signal": [1., 2., 3., 4.], "gap_before": [True, False, False, True], **extra})


def test_unknown_age_is_not_known_zero():
    frame = add_context(history())
    assert frame.operating_age_s.isna().all()
    assert not frame.operating_age_known.any()
    assert frame.observed_elapsed_s.tolist() == [0, 10, 20, 100]


def test_running_clock_stops_and_gap_loses_certainty_without_reset():
    frame = add_context(history(operating_age_s=[40., np.nan, np.nan, np.nan],
                                is_running=[False, True, True, True]), age_source="running_clock")
    assert frame.operating_age_s.iloc[:3].tolist() == [40., 40., 50.]
    assert np.isnan(frame.operating_age_s.iloc[-1])
    assert not frame.operating_age_known.iloc[-1]


def test_counter_survives_gap_and_replacement_is_explicit():
    frame = add_context(history(operating_age_s=[100., 110., 120., 190.]), age_source="counter")
    assert frame.operating_age_s.iloc[-1] == 190
    with pytest.raises(ValueError, match="replacement"):
        add_context(history(operating_age_s=[100., 110., 0., 10.]), age_source="counter")
    reset = add_context(history(operating_age_s=[100., 110., np.nan, 10.],
                                component_replaced=[False, False, True, False]), age_source="counter")
    assert reset.operating_age_s.iloc[2] == 0
    assert reset.component_cycle_id.tolist() == ["0", "0", "1", "1"]


def test_context_is_prefix_causal():
    original = history(operating_age_s=[10., 20., 30., 90.], rpm=[100, 200, 300, 900])
    changed = original.copy()
    changed.loc[3, ["operating_age_s", "timestamp_s", "rpm"]] = [900, 1000, 9999]
    pd.testing.assert_frame_equal(add_context(original, age_source="counter").iloc[:3],
                                  add_context(changed, age_source="counter").iloc[:3])


def test_generic_optional_context_and_minimal_compatibility(tmp_path):
    path = tmp_path / "sensor.csv"
    pd.DataFrame({"unit_id": ["u", "u", "u"], "timestamp_s": [0, 10, 20],
                  "pressure": [1, 2, 3], "hours": [3600, 3610, 3620],
                  "load": [12, np.nan, 13]}).to_csv(path, index=False)
    files = {"primary": [{"path": str(path), "relative_path": path.name}]}
    bare, _, _ = read_generic_csv(files, "pressure")
    assert bare.operating_age_s.isna().all()
    rich, units, _ = read_generic_csv(files, "pressure", context_mapping={"operating_age_s": "hours", "load_kn": "load"})
    assert rich.operating_age_s.tolist() == [3600, 3610, 3620]
    assert rich.operating_age_known.all()
    assert rich.load_kn.isna().sum() == 1
    assert units.physical_unit_id.tolist() == ["u"]
    schema = context_schema(rich, {"signal_unit": "Pa", "input_columns": ["signal"]})
    assert schema["input_columns"] == ["signal"]
    assert schema["schema_version"] == 2
    assert schema["columns"]["load_kn"]["unit"] == "kN"
    with pytest.raises(ValueError, match="Unsupported context"):
        read_generic_csv(files, "pressure", context_mapping={"event_time_s": "hours"})


def test_adapter_preserves_modes_and_lab_proxy():
    source = history(operating_age_s=[0, 10, 20, 100], rpm=[2100]*4, load_kn=[12]*4,
                     flow_rate=[1]*4, dust_feed=[2]*4, dust=["A"]*4)
    units = pd.DataFrame({"unit_id": ["u"]})
    frame, _, _ = _canonicalize_adapted(source, units, signal=source.signal)
    assert {"rpm", "load_kn", "flow_rate", "dust_feed", "dust"}.issubset(frame)
    assert frame.operating_age_source.eq("laboratory_proxy").all()
    assert frame.operating_age_s.iloc[-1] == 100


def test_cycles_share_physical_split():
    units = pd.DataFrame({"unit_id": ["a1", "a2", "b", "c"],
                          "origin_unit_id": ["a1", "a2", "b", "c"],
                          "physical_unit_id": ["a", "a", "b", "c"], "source_group": ["primary"]*4})
    split = allocate_project_split(units, {})
    assert_physical_split(units, split)
    assert next(name for name in ("train", "validation", "test") if "a1" in split[name]) == next(name for name in ("train", "validation", "test") if "a2" in split[name])
    with pytest.raises(ValueError, match="cycles"):
        assert_physical_split(units, {"train": ["a1"], "validation": ["a2", "b"], "test": ["c"]})


def test_adapter_to_allocation_preserves_shared_equipment_identity():
    records = [{"unit_id": uid, "physical_unit_id": physical, "timestamp_s": t,
                "signal": float(index * 10 + t), "gap_before": t == 0}
               for index, (uid, physical) in enumerate([( "a1", "a"), ("a2", "a"), ("b", "b"), ("c", "c")])
               for t in [0, 1]]
    frame = pd.DataFrame(records)
    units = frame[["unit_id", "physical_unit_id"]].drop_duplicates()
    canonical, admitted, _ = _canonicalize_adapted(frame, units, signal=frame.signal)
    assert admitted.physical_unit_id.tolist() == ["a", "a", "b", "c"]
    split = allocate_project_split(admitted, {"seed": 0})
    assert_physical_split(admitted, split)
    assert canonical.groupby("unit_id").physical_unit_id.first().to_dict() == admitted.set_index("unit_id").physical_unit_id.to_dict()
    conflicting = units.copy()
    conflicting.loc[conflicting.unit_id == "a1", "physical_unit_id"] = "other"
    with pytest.raises(ValueError, match="disagree"):
        _canonicalize_adapted(frame, conflicting, signal=frame.signal)


def test_replacement_on_rejected_signal_row_is_preserved(tmp_path):
    path = tmp_path / "cycles.csv"
    pd.DataFrame({"unit_id": ["u"]*4, "timestamp_s": [0, 1, 2, 3],
                  "sensor": [1, 2, np.nan, 3], "age": [100, 101, 0, 1],
                  "replace": [False, False, True, False]}).to_csv(path, index=False)
    frame, _, _ = read_generic_csv({"primary": [{"path": path, "relative_path": path.name}]},
                                   "sensor", context_mapping={"operating_age_s": "age", "component_replaced": "replace"})
    assert frame.component_cycle_id.tolist() == ["0", "0", "1"]
    assert frame.operating_age_s.tolist() == [100, 101, 1]
    assert frame.gap_before.tolist() == [True, False, True]


def test_age_units_are_normalized_and_schema_roles_explicit(tmp_path):
    path = tmp_path / "age.csv"
    pd.DataFrame({"unit_id": ["u", "u"], "timestamp_s": [0, 1],
                  "signal": [1, 2], "hours": [2, 3], "rpm": [100, 101]}).to_csv(path, index=False)
    mapping = {"operating_age_s": "hours", "rpm": "rpm"}
    frame, _, _ = read_generic_csv({"primary": [{"path": path, "relative_path": path.name}]},
                                   "signal", context_mapping=mapping,
                                   context_units={"operating_age_s": "h"})
    assert frame.operating_age_s.tolist() == [7200, 10800]
    schema = context_schema(frame, {"input_columns": ["signal"], "signal_unit": "g",
                                    "context_mapping": mapping, "context_units": {"operating_age_s": "h"}})
    assert schema["columns"]["operating_age_s"]["unit"] == "s"
    assert schema["columns"]["operating_age_s"]["role"] == "known_age"
    assert schema["columns"]["rpm"]["role"] == "operating_context"
    assert schema["columns"]["observed_elapsed_s"]["role"] == "metadata"
    assert "observed_elapsed_s" not in schema["red_entry_input_columns"]


def test_generic_supplied_endpoints_survive_schema_and_targets_without_feature_leak(tmp_path):
    path = tmp_path / "events.csv"
    source = pd.DataFrame({"unit_id": ["u"] * 4, "timestamp_s": [0, 1, 2, 3],
                           "sensor": [1, 2, 3, 4]})
    for name in ENDPOINT_FIELDS:
        source["source_" + name] = ([False, False, False, True]
                                   if not name.endswith("_timestamp_s") else [np.nan, np.nan, np.nan, 3.])
    # No replacement episode here: its flag is supplied explicitly as false.
    source["source_component_replaced"] = False
    source.to_csv(path, index=False)
    mapping = {name: "source_" + name for name in ENDPOINT_FIELDS}
    frame, units, _ = read_generic_csv({"primary": [{"path": path, "relative_path": path.name}]},
                                       "sensor", context_mapping=mapping)
    schema = context_schema(frame, {"signal_unit": "Pa", "context_mapping": mapping,
                                    "input_columns": ["signal", *ENDPOINT_FIELDS],
                                    "thresholds": {"mode": "absolute", "direction": "above", "yellow": 5, "red": 10}})
    assert schema["input_columns"] == ["signal"]
    for name in ENDPOINT_FIELDS:
        assert schema["columns"][name]["role"] == "target_only"
        assert schema["columns"][name]["provenance"]["source_column"] == mapping[name]
        assert name not in schema["red_entry_input_columns"]
        if name.endswith("_timestamp_s"):
            assert schema["columns"][name]["unit"] == "s"
    snapshot = {"features": frame, "units": units, "schema": schema,
                "split": {"train": ["u"], "validation": [], "test": []}}
    targets = build_red_entry_targets(snapshot, [1, 2])
    event = targets["events"].iloc[0]
    assert event.endpoint_records[-1]["confirmed_failure"]
    assert event.endpoint_records[-1]["emergency_stop"]
    assert event.endpoint_records[-1]["planned_maintenance"]
    assert event.endpoint_records[-1]["replacement_timestamp_s"] == 3.
    assert pd.isna(event.first_red_timestamp_s)
    assert not event.first_event_verified
    assert targets["hazard_targets"].sum() == 0
    for mode in ("sensor_only", "age_context", "hybrid"):
        state = fit_feature_state(snapshot, mode)
        assert not set(ENDPOINT_FIELDS) & set(state["source_specs"])
        changed = frame.copy()
        changed.loc[3, "confirmed_failure"] = False
        changed.loc[3, "confirmed_failure_timestamp_s"] = 1000.
        np.testing.assert_array_equal(transform_prefix(frame, schema, state)["x"],
                                      transform_prefix(changed, schema, state)["x"])


@pytest.mark.parametrize("role,value,units,match", [
    ("confirmed_failure", "yes", {}, "Boolean"),
    ("emergency_stop", "unknown", {}, "Boolean"),
    ("planned_maintenance", 2, {}, "Boolean"),
    ("confirmed_failure_timestamp_s", "tomorrow", {}, "finite timestamps"),
    ("emergency_stop_timestamp_s", np.inf, {}, "finite timestamps"),
    ("maintenance_timestamp_s", 1, {"maintenance_timestamp_s": "h"}, "same clock"),
    ("replacement_timestamp_s", 1, {"replacement_timestamp_s": "min"}, "same clock"),
])
def test_generic_endpoint_values_and_seconds_are_validated(tmp_path, role, value, units, match):
    path = tmp_path / "invalid_event.csv"
    pd.DataFrame({"unit_id": ["u"] * 2, "timestamp_s": [0, 1],
                  "signal": [1, 2], "event": [np.nan, value]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        read_generic_csv({"primary": [{"path": path, "relative_path": path.name}]},
                         "signal", context_mapping={role: "event"}, context_units=units)
