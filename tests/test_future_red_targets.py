import hashlib

import pandas as pd
import pytest

from pdm.future_red_targets import DEFAULT_HORIZONS_S, _targets, load_future_red_targets
from pdm.io_util import atomic_write_json, sha256_file


def _labels(zones, times=None, *, dataset="bearings"):
    times = times or list(range(0, len(zones) * 10, 10))
    common = {"unit_id": ["u1"] * len(zones), "split": ["test"] * len(zones), "timestamp_s": times}
    if dataset == "bearings":
        codes = {"green": 0, "yellow": 1, "red": 2}
        common["true_zone"] = [codes.get(zone, -1) for zone in zones]
        common["zone_name"] = zones
    else:
        common["display_zone"] = zones
    return pd.DataFrame(common)


def test_future_red_targets_label_entry_and_mask_censoring():
    result, counts = _targets(_labels(["green", "yellow", "red", "yellow", "green"]), "bearings", 20)
    assert result.target.iloc[:2].tolist() == [1, 1]
    assert result.target.iloc[2:].isna().all()
    assert result.target_status.tolist() == ["positive", "positive", "out_of_risk_set", "out_of_risk_set", "out_of_risk_set"]
    assert counts["first_red_timestamp_s_by_unit"] == {"u1": 20.0}
    assert counts["known_positive_count"] == 2


def test_bearing_saved_numeric_code_and_zone_name_schema():
    labels = _labels(["green", "red", "yellow", "red"])
    result, counts = _targets(labels, "bearings", 10)
    assert result.first_red_timestamp_s.eq(10).all()
    assert result.target.iloc[0] == 1
    assert counts["known_positive_count"] == 1


def test_bearing_code_name_disagreement_is_rejected():
    labels = _labels(["green", "red"])
    labels.loc[1, "true_zone"] = 1
    with pytest.raises(ValueError, match="codes and saved zone names disagree"):
        _targets(labels, "bearings", 10)


def test_future_red_filter_comparator_uses_saved_display_red_and_known_negative():
    result, counts = _targets(_labels(["green", "yellow", "green", "red", "red"], dataset="filters"), "filters", 20)
    assert result.target.iloc[0] == 0
    assert result.target.iloc[1] == 1
    assert result.first_red_timestamp_s.iloc[0] == 30
    assert counts["known_negative_count"] == 1


def test_future_red_targets_mask_unknown_quality_and_large_gaps():
    labels = _labels(["green", "unknown", "red", "green"], [0, 10, 20, 30])
    result, _ = _targets(labels, "bearings", 20)
    assert result.target.iloc[0] is pd.NA
    gap = _labels(["green", "yellow", "green", "green"], [0, 10, 40, 50])
    gapped, _ = _targets(gap, "bearings", 20)
    assert gapped.target_status.iloc[0] == "masked"


def test_load_future_red_targets_verifies_file_hash(tmp_path):
    path = tmp_path / "targets.csv"
    pd.DataFrame({"unit_id": ["u"], "split": ["test"], "timestamp_s": [0],
                  "first_red_timestamp_s": [None], "at_risk": [True], "target": [0],
                  "target_known": [True], "target_status": ["negative"]}).to_csv(path, index=False)
    source = tmp_path / "source"
    source.mkdir()
    labels_path = source / "labels.csv"
    labels_path.write_text("unit_id\nu\n", encoding="utf-8")
    source_manifest = {"artifact_id": "zones", "labels_file": "labels.csv"}
    atomic_write_json(source / "manifest.json", source_manifest)
    manifest = {"schema_version": "future_sensor_red_entry_targets_v1", "targets_file": path.name,
                "targets_sha256": sha256_file(path), "row_count": 1,
                "source_zone_label_directory": str(source),
                "source_zone_label_manifest_sha256": sha256_file(source / "manifest.json"),
                "source_zone_label_file_sha256": sha256_file(labels_path),
                "source_label_artifact_id": "zones", "zone_policy": {},
                "zone_policy_sha256": hashlib.sha256(b"{}").hexdigest()}
    atomic_write_json(tmp_path / "manifest.json", manifest)
    frame, loaded = load_future_red_targets(tmp_path)
    assert len(frame) == 1 and loaded["targets_sha256"] == manifest["targets_sha256"]
    path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="file hash"):
        load_future_red_targets(tmp_path)


def test_horizon_defaults_are_fixed_per_dataset():
    assert DEFAULT_HORIZONS_S == {"bearings": 1800.0, "filters": 20.0}
