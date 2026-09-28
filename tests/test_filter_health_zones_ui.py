from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from pdm.filter_health_zones_ui import _read_current_filter_zone_labels
from pdm.monitoring.filter_zones import FILTER_ZONE_POLICY


def _data():
    return {
        "dataset_version": "filters-v7",
        "fingerprint": {"features_hash": "features-sha", "quality_records_hash": "quality-sha"},
        "split": {"train": ["filter-a", "filter-c"], "validation": ["filter-b"], "test": []},
    }


def _write_artifact(root, data):
    artifact = root / "artifact-1"
    artifact.mkdir(parents=True)
    labels = pd.DataFrame([
        {"unit_id": "filter-a", "split": "train", "timestamp_s": 0.,
         "differential_pressure_pa": 299., "flow_rate_recorded": 10., "dust_feed_recorded": 2.,
         "sensor_zone": "below_provisional_pressure_band", "display_zone": "green", "zone_reason": "below",
         "quality_status": "admitted"},
        {"unit_id": "filter-a", "split": "train", "timestamp_s": 1.,
         "differential_pressure_pa": 300., "flow_rate_recorded": 10., "dust_feed_recorded": 2.,
         "sensor_zone": "pressure_warning_band", "display_zone": "yellow", "zone_reason": "warning",
         "quality_status": "attention"},
        {"unit_id": "filter-b", "split": "validation", "timestamp_s": 0.,
         "differential_pressure_pa": 610., "flow_rate_recorded": 0., "dust_feed_recorded": 0.,
         "sensor_zone": "configured_600_pa_limit", "display_zone": "red", "zone_reason": "limit",
         "quality_status": "attention"},
        {"unit_id": "filter-c", "split": "train", "timestamp_s": 0.,
         "differential_pressure_pa": float("nan"), "flow_rate_recorded": 10., "dust_feed_recorded": 2.,
         "sensor_zone": "unknown", "display_zone": "gray", "zone_reason": "missing_pressure",
         "quality_status": "attention"},
    ])
    labels_path = artifact / "labels.csv"
    labels.to_csv(labels_path, index=False)
    fingerprint_sha = hashlib.sha256(json.dumps(data["fingerprint"], sort_keys=True).encode("utf-8")).hexdigest()
    manifest = {
        "dataset_id": "filters", "dataset_version": data["dataset_version"],
        "dataset_fingerprint_sha256": fingerprint_sha, "created_at": "2026-09-28T00:00:00Z",
        "artifact_id": "filter-labels-v1", "labels_file": "labels.csv",
        "labels_sha256": hashlib.sha256(labels_path.read_bytes()).hexdigest(), "row_count": len(labels),
        "policy": {"version": FILTER_ZONE_POLICY["version"], "yellow_limit_pa": 300., "red_limit_pa": 600.},
        "class_counts_by_split": {"train": {"green": 1, "yellow": 1, "red": 0, "unknown": 0}},
    }
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return artifact, manifest


def test_filter_label_replay_loads_matching_versioned_artifact_and_maps_unknown(tmp_path):
    data = _data()
    artifact, _ = _write_artifact(tmp_path, data)
    labels, manifest, loaded_dir = _read_current_filter_zone_labels(data, tmp_path)

    assert loaded_dir == artifact
    assert manifest["artifact_id"] == "filter-labels-v1"
    assert labels["zone"].tolist() == ["green", "yellow", "red", "unknown"]
    assert labels["split"].tolist() == ["train", "train", "validation", "train"]


def test_filter_label_replay_rejects_a_label_for_a_different_prepared_fingerprint(tmp_path):
    data = _data()
    _write_artifact(tmp_path, data)
    changed = {**data, "fingerprint": {**data["fingerprint"], "features_hash": "new-features-sha"}}

    with pytest.raises(FileNotFoundError, match="matches the latest prepared snapshot"):
        _read_current_filter_zone_labels(changed, tmp_path)


def test_filter_label_replay_rejects_tampered_labels_and_split_drift(tmp_path):
    data = _data()
    artifact, _ = _write_artifact(tmp_path, data)
    labels_path = artifact / "labels.csv"
    labels_path.write_text(labels_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="No intact filter sensor-zone"):
        _read_current_filter_zone_labels(data, tmp_path)

    labels_path.write_text(
        labels_path.read_text(encoding="utf-8").replace("train,0.0", "validation,0.0"), encoding="utf-8")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["labels_sha256"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="split does not match"):
        _read_current_filter_zone_labels(data, tmp_path)


def test_filter_health_zones_is_accessible_without_a_model_bundle(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    from pdm.data import prepare
    from pdm.monitoring import ui as monitoring_ui
    from pdm.paths import project_root

    data = _data()
    artifact_root = tmp_path / "_zones" / "filters" / "label_artifacts"
    _write_artifact(artifact_root, data)
    monkeypatch.setattr(prepare, "load_processed", lambda dataset_id, version=None: data)
    monkeypatch.setattr("pdm.filter_health_zones_ui.runs_root", lambda: tmp_path)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda dataset_id: [])
    monkeypatch.setattr(monitoring_ui, "bundles_for", lambda dataset_id: [])
    # Select the standalone replay view explicitly; Future-red entry is the report default.
    app_test = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=30)
    app_test.session_state["screen_selection"] = "Model Report"
    app_test.run()
    next(r for r in app_test.sidebar.radio if r.label == "Dataset").set_value("Filters")
    app_test.run()
    report_view = next(r for r in app_test.radio if r.label == "Report view")
    report_view.set_value("Health zones")
    app_test.run()

    assert not app_test.exception
    assert [r.label for r in app_test.sidebar.radio if r.label == "Screen"]
    assert any("Health zones · filters" in str(h.value) for h in app_test.subheader)
    assert any("Configured laboratory pressure limit reached" in str(m.value) for m in app_test.markdown)
    assert report_view.value == "Health zones"

    next(r for r in app_test.sidebar.radio if r.label == "Dataset").set_value("Bearings")
    app_test.session_state["screen_selection"] = "Model Report"
    app_test.run()
    assert not app_test.exception
    report_view = next(r for r in app_test.radio if r.label == "Report view")
    assert report_view.value == "Future-red entry"
