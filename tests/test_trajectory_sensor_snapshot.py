"""Causality, source admission, and immutable snapshot engineering checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.data.project_import import import_project
from pdm.data.project_prepare import SNAPSHOT_FILES, load_snapshot, prepare_project
from pdm.io_util import atomic_write_json, sha256_file
from pdm.projects import ProjectStore
from pdm.signal_training import _validate_snapshot
from pdm.trajectory_data import (
    FEATURE_NAMES,
    SENSOR_AVAILABILITY,
    SENSOR_FEATURE_NAMES,
    build_trajectory_frame,
    build_trajectory_prefix,
)

spec = importlib.util.spec_from_file_location(
    "sensor_preparation", Path(__file__).resolve().parents[1] / "scripts/prepare_bearings_trajectory_snapshot.py"
)
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)
CONFIG = {"history_length": 2, "horizons_s": [60., 120.], "target_tolerance_s": .01}


def enriched_data():
    features = pd.DataFrame({"unit_id": "u", "timestamp_s": np.arange(5) * 60.,
                             "signal": [.2, .3, .4, .5, .6], "gap_before": [True, False, False, False, False]})
    for i, column in enumerate(SENSOR_FEATURE_NAMES):
        features[column] = np.arange(5) + i + 1.
    features["horizontal_rms"] = features.signal
    features["vertical_rms"] = features.signal / 2
    return {"features": features, "schema": {
        "thresholds": {"mode": "absolute", "direction": "above", "red": 3.},
        "trajectory_sensor_features": {"schema_version": 1, "columns": list(SENSOR_FEATURE_NAMES),
                                       "transform": "log1p", "availability": SENSOR_AVAILABILITY},
    }}


def test_enriched_inputs_are_explicit_local_and_prefix_invariant():
    data = enriched_data()
    original = build_trajectory_frame(data, ["u"], CONFIG)
    assert original["x"].shape == (4, 2, 33)
    assert original["feature_names"] == FEATURE_NAMES + ["log1p_" + c for c in SENSOR_FEATURE_NAMES]
    np.testing.assert_allclose(original["x"][0, :, 13:],
                               np.log1p(data["features"].loc[:1, list(SENSOR_FEATURE_NAMES)]), rtol=1e-6)
    changed = enriched_data()
    changed["features"].loc[2:, "horizontal_band_0"] = 999999.
    changed["features"]["event_time_s"] = 0.
    changed["features"]["observation_end_s"] = 10e8
    changed["features"]["life_fraction"] = .999
    changed["features"]["RUL"] = -999.
    np.testing.assert_array_equal(original["x"][0], build_trajectory_frame(changed, ["u"], CONFIG)["x"][0])
    prefix = build_trajectory_prefix(data, data["features"].iloc[:2], CONFIG)
    np.testing.assert_array_equal(prefix["x"][0], original["x"][0])
    assert not prefix["mask"].any()
    baseline = {**data, "schema": {"thresholds": data["schema"]["thresholds"]}}
    assert build_trajectory_frame(baseline, ["u"], CONFIG)["x"].shape[-1] == 13


@pytest.mark.parametrize("fault", ["missing", "nonfinite", "negative", "rms", "declaration"])
def test_invalid_declared_sensors_fail_loudly_on_replay(fault):
    data = enriched_data()
    if fault == "missing":
        data["features"] = data["features"].drop(columns="horizontal_band_0")
    elif fault == "nonfinite":
        data["features"].loc[1, "horizontal_band_0"] = np.nan
    elif fault == "negative":
        data["features"].loc[1, "horizontal_band_0"] = -1
    elif fault == "rms":
        data["features"].loc[1, "horizontal_rms"] += .1
    else:
        data["schema"]["trajectory_sensor_features"]["columns"].append("event_time_s")
    with pytest.raises(ValueError):
        build_trajectory_prefix(data, data["features"].iloc[:2], CONFIG)


def archive_for(parent):
    source = parent[["unit_id", "timestamp_s"]].copy()
    source["file_index"] = source.timestamp_s / 60
    source["n_samples"] = 32768
    source["sample_count_ok"] = True
    source["n_nan"] = 0
    for name in SENSOR_FEATURE_NAMES:
        source[name] = parent.signal / 2
    source["horizontal_rms"] = parent.signal
    for name in ("operating_age_s", "rpm", "load_kn"):
        if name in parent:
            source[name] = parent[name]
    source["event_time_s"] = -1000
    source["gap_before"] = False
    return source


@pytest.mark.parametrize("fault", ["duplicate", "extra", "missing", "quality", "time", "rms"])
def test_bad_archive_admission_rejected(fault):
    parent = enriched_data()["features"].drop(columns=list(SENSOR_FEATURE_NAMES))
    source = archive_for(parent)
    if fault == "duplicate":
        source = pd.concat([source, source.iloc[:1]])
    elif fault == "extra":
        extra = source.iloc[:1].copy()
        extra["unit_id"] = "other"
        source = pd.concat([source, extra])
    elif fault == "missing":
        source = source.iloc[:-1]
    elif fault == "quality":
        source.loc[0, "n_nan"] = 1
    elif fault == "time":
        source.loc[0, "file_index"] = 99
    else:
        source.loc[0, "horizontal_rms"] += .1
    with pytest.raises(ValueError):
        preparation.align_sensors(parent, source)


def test_snapshot_round_trip_immutability_and_archive_independence(tmp_path):
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Synthetic sensor storage QA", "generic_sensor_csv")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pd.DataFrame([{"unit_id": f"u{u}", "timestamp_s": t * 60., "rms": .2 + .1 * t + .01 * u}
                  for u in range(6) for t in range(4)]).to_csv(source_dir / "rms.csv", index=False)
    manifest = import_project(project["project_id"], {
        "primary": {"mode": "folder", "path": str(source_dir)}, "signal_column": "rms",
        "signal_unit": "g", "thresholds": {"mode": "absolute", "direction": "above", "yellow": 1., "red": 3.},
    }, store=store)
    saved = prepare_project(project["project_id"], manifest["manifest_id"], store=store)
    parent = load_snapshot(project["project_id"], saved["snapshot_id"], store=store)
    original_project = store.get(project["project_id"])
    original_hashes = {n: sha256_file(parent["dir"] / n) for n in (*SNAPSHOT_FILES, "processed_fingerprint.json")}
    archive = tmp_path / "archive_version"
    archive.mkdir()
    archive_for(parent["features"]).iloc[::-1].to_parquet(archive / "features.parquet", index=False)
    atomic_write_json(archive / "feature_schema.json", {})
    atomic_write_json(archive / "processed_fingerprint.json", {"features_hash": sha256_file(archive / "features.parquet")})
    audit = tmp_path / "audit.json"
    atomic_write_json(audit, {"allowlist": list(SENSOR_FEATURE_NAMES), "files": [
        {"path": str(p), "sha256": sha256_file(p)} for p in
        [*(archive / n for n in ("features.parquet", "feature_schema.json", "processed_fingerprint.json")),
         *(parent["dir"] / n for n in original_hashes)]],
    })
    enriched = preparation.prepare_sensor_snapshot(project["project_id"], parent["snapshot_id"], archive, audit, store=store)
    assert enriched["snapshot_id"] != parent["snapshot_id"]
    assert store.get(project["project_id"]) == original_project
    assert original_hashes == {n: sha256_file(parent["dir"] / n) for n in original_hashes}
    for name in ("split.json", "units.parquet"):
        assert sha256_file(enriched["dir"] / name) == original_hashes[name]
    pd.testing.assert_frame_equal(enriched["features"].loc[:, parent["features"].columns], parent["features"])
    assert "event_time_s" not in enriched["features"]
    _validate_snapshot(enriched)
    before = set(parent["dir"].parent.iterdir())
    (archive / "feature_schema.json").write_text('{"tampered": true}')
    with pytest.raises(ValueError, match="Archive hash differs from audit"):
        preparation.prepare_sensor_snapshot(project["project_id"], parent["snapshot_id"], archive, audit, store=store)
    assert set(parent["dir"].parent.iterdir()) == before
    (archive / "features.parquet").unlink()
    reloaded = load_snapshot(project["project_id"], enriched["snapshot_id"], store=store)
    frame = build_trajectory_frame(reloaded, reloaded["split"]["train"], CONFIG)
    assert frame["x"].shape[-1] == 33
    # Prepared artifact tampering is caught by the standard loader.
    with (enriched["dir"] / "features.parquet").open("ab") as f:
        f.write(b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(project["project_id"], enriched["snapshot_id"], store=store)
