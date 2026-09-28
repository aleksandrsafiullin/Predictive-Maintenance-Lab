from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

import pdm.health_zones as hz


def _unit(rms, uid="Bearing9_9"):
    """Labelled synthetic bearing: one measurement per minute, given RMS curve."""
    n = len(rms)
    row = {f"{c}_{s}": 1.0 for c in hz.CHANNELS for s in hz.SIGNALS}
    df = pd.DataFrame([row] * n)
    df["horizontal_rms"] = rms
    df["vertical_rms"] = np.asarray(rms) * 0.8
    df["unit_id"] = uid
    df["timestamp_s"] = np.arange(n, dtype=float) * 60.0
    df["rpm"], df["load_kn"] = 2100.0, 12.0
    return df


def _degrading(n=120, start=60, seed=0):
    rng = np.random.default_rng(seed)
    rms = 1.0 + 0.01 * rng.standard_normal(n)
    rms[start:] += np.linspace(0.5, 6.0, n - start)
    return rms


def _snapshot_fingerprint(tmp_path, split):
    from pdm.io_util import atomic_write_json, sha256_file
    from pdm.splits import split_hash

    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    filenames = ("features.parquet", "units.parquet", "split.json", "feature_schema.json", "quality_records.parquet")
    atomic_write_json(processed / "split.json", split)
    for name in filenames:
        path = processed / name
        if name != "split.json" and not path.exists():
            path.write_bytes(f"fixture:{name}".encode())
    fingerprint = {
        "dataset_id": "bearings", "dataset_version": "fixture-v1", "split_hash": split_hash(split),
        "features_hash": sha256_file(processed / "features.parquet"),
        "units_hash": sha256_file(processed / "units.parquet"),
        "split_json_hash": sha256_file(processed / "split.json"),
        "feature_schema_hash": sha256_file(processed / "feature_schema.json"),
        "quality_records_hash": sha256_file(processed / "quality_records.parquet"),
    }
    atomic_write_json(processed / "processed_fingerprint.json", fingerprint)
    return processed, fingerprint


def test_labels_are_distinct_signal_bands_not_recording_end_windows():
    lab = hz.label_unit(_unit(_degrading()), hz.DEFAULT_CONFIG)
    z = lab["true_zone"].to_numpy()
    assert (z[:60] == 0).all()
    assert 1 in z and 2 in z
    first_yellow, first_red = np.flatnonzero(z == 1)[0], np.flatnonzero(z == 2)[0]
    assert first_yellow >= 60 and first_red > first_yellow
    assert (z[first_yellow:first_red] == 1).all() and (z[first_red:] == 2).all()
    assert (z >= 2).any()
    # Changing only the endpoint-based timing setting cannot move a signal zone.
    np.testing.assert_array_equal(z, hz.label_unit(_unit(_degrading()), hz.DEFAULT_CONFIG)["true_zone"])


def test_train_ratio_is_configured_signal_boundary_not_outcome_tuned():
    assert hz.fit_rule_red_ratio(pd.DataFrame(), [], {**hz.DEFAULT_CONFIG, "red_ratio": 2.4}) == 2.4


@pytest.mark.parametrize("red_ratio", [np.nan, np.inf, 1.25, 1.0])
def test_export_and_training_reject_invalid_red_threshold_before_data_load(monkeypatch, red_ratio):
    import pdm.data.prepare as prepare

    monkeypatch.setattr(prepare, "load_processed", lambda dataset_id: pytest.fail("invalid policy loaded data"))
    with pytest.raises(ValueError, match="red_ratio must be finite and strictly greater than onset_ratio"):
        hz.export_zone_labels(config={"red_ratio": red_ratio})
    with pytest.raises(ValueError, match="red_ratio must be finite and strictly greater than onset_ratio"):
        hz.train_zone_model(config={"red_ratio": red_ratio})


@pytest.mark.parametrize("command", ["zones-train", "zones-labels"])
@pytest.mark.parametrize("red_ratio", ["nan", "inf", "1.25"])
def test_cli_rejects_invalid_bearing_red_threshold(command, red_ratio, capsys):
    from pdm.cli import main

    with pytest.raises(SystemExit) as exc:
        main([command, "--red-ratio", red_ratio])
    assert exc.value.code == 2
    assert "red_ratio must be finite and strictly greater than onset_ratio" in capsys.readouterr().err


def test_versioned_label_export_covers_split_units_and_excludes_endpoint(monkeypatch, tmp_path, tiny_bearing_tables):
    import pdm.data.prepare as prepare

    features, _ = tiny_bearing_tables
    unit_ids = sorted(features.unit_id.astype(str).unique())
    split = {"train": unit_ids[:3], "validation": unit_ids[3:4], "test": unit_ids[4:]}
    processed, fingerprint = _snapshot_fingerprint(tmp_path, split)
    data = {"features": features, "split": split, "dataset_version": "fixture-v1",
            "fingerprint": fingerprint, "dir": processed}
    monkeypatch.setattr(prepare, "load_processed", lambda dataset_id: data)
    monkeypatch.setattr(hz, "zones_root", lambda dataset_id="bearings": tmp_path)

    result = hz.export_zone_labels()
    labels_path = tmp_path / "label_artifacts" / result["artifact_id"] / "labels.csv"
    manifest_path = labels_path.with_name("manifest.json")
    labels = pd.read_csv(labels_path)
    manifest = json.loads(manifest_path.read_text())
    assert labels.unit_id.nunique() == len(unit_ids)
    assert set(labels.columns) == {"unit_id", "split", "timestamp_s", "combined_rms", "baseline_rms", "true_zone", "zone_name"}
    assert manifest["rul_or_endpoint_used_for_classification"] is False
    assert manifest["dataset_version"] == "fixture-v1"
    assert manifest["dataset_fingerprint_sha256"] == hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    assert manifest["labels_sha256"] == hashlib.sha256(labels_path.read_bytes()).hexdigest()
    assert manifest["row_count"] == len(labels)


@pytest.mark.parametrize("damage", ["features_file", "quality_records_file", "saved_fingerprint", "loaded_split"])
def test_label_export_fails_if_snapshot_no_longer_matches_fingerprint(monkeypatch, tmp_path, tiny_bearing_tables, damage):
    import pdm.data.prepare as prepare
    from pdm.io_util import atomic_write_json

    features, _ = tiny_bearing_tables
    unit_ids = sorted(features.unit_id.astype(str).unique())
    split = {"train": unit_ids[:3], "validation": unit_ids[3:4], "test": unit_ids[4:]}
    processed, fingerprint = _snapshot_fingerprint(tmp_path, split)
    data = {"features": features, "split": split, "dataset_version": "fixture-v1",
            "fingerprint": fingerprint, "dir": processed}
    if damage == "features_file":
        (processed / "features.parquet").write_bytes(b"mutated feature file")
        expected = "does not match saved fingerprint: features.parquet"
    elif damage == "quality_records_file":
        (processed / "quality_records.parquet").write_bytes(b"mutated quality records")
        expected = "does not match saved fingerprint: quality_records.parquet"
    elif damage == "saved_fingerprint":
        atomic_write_json(processed / "processed_fingerprint.json", {**fingerprint, "features_hash": "0" * 64})
        expected = "differs from processed_fingerprint.json"
    else:
        data["split"] = {**split, "train": split["train"][:-1]}
        expected = "Loaded bearing split does not match"
    monkeypatch.setattr(prepare, "load_processed", lambda dataset_id: data)
    monkeypatch.setattr(hz, "zones_root", lambda dataset_id="bearings": tmp_path / "runs")

    with pytest.raises(ValueError, match=expected):
        hz.export_zone_labels()
    assert not (tmp_path / "runs").exists()


def test_loaded_run_rejects_red_threshold_not_above_yellow_threshold(tmp_path):
    from pdm.io_util import atomic_write_json

    run_dir = tmp_path / "bad-threshold"
    run_dir.mkdir()
    atomic_write_json(run_dir / "meta.json", {
        "zone_definition": hz.ZONE_DEFINITION_VERSION,
        "config": {**hz.DEFAULT_CONFIG, "red_ratio": hz.DEFAULT_CONFIG["onset_ratio"]},
    })
    assert not hz._zone_run_compatible(hz.read_json(run_dir / "meta.json"))
    with pytest.raises(ValueError, match="incomplete zone metadata"):
        hz.load_zone_run(run_dir)


def test_signal_zone_labels_for_a_prefix_do_not_depend_on_future_rows():
    unit = _unit(_degrading())
    full = hz.label_unit(unit, hz.DEFAULT_CONFIG)
    for cut in (10, 60, 90):
        prefix = hz.label_unit(unit.iloc[:cut], hz.DEFAULT_CONFIG)
        np.testing.assert_array_equal(prefix["true_zone"], full["true_zone"].iloc[:cut])


def test_exported_baseline_is_causal_while_warming_up():
    unit = _unit([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    ref = hz.signal_zone_reference(unit, hz.DEFAULT_CONFIG)
    np.testing.assert_allclose(ref["baseline_rms"], [1.0, 1.5, 2.0, 2.5, 3.0, 3.0])


def test_small_drift_is_not_onset():
    rms = 1.0 + np.linspace(0, 0.1, 200)  # +10 % drift stays below the 1.25x ratio
    assert hz.onset_index(rms, hz.DEFAULT_CONFIG) is None


def test_smoothing_needs_confirmation_and_is_sticky():
    raw = np.array([0, 2, 0, 1, 1, 1, 0, 1, 2, 2, 0, 0])
    out = hz.smooth_zones(raw, escalate_n=2, deescalate_n=3)
    assert out.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]


def test_features_are_causal():
    unit = hz.label_unit(_unit(_degrading()), hz.DEFAULT_CONFIG)
    full = hz.unit_features(unit, hz.DEFAULT_CONFIG)
    for k in (3, 30, 70, 100):
        prefix = hz.unit_features(unit.iloc[:k], hz.DEFAULT_CONFIG)
        np.testing.assert_allclose(prefix, full[:k])
    assert full.shape[1] == len(hz.feature_names(hz.DEFAULT_CONFIG))


def test_prediction_at_t_ignores_future_rows():
    cfg = hz.DEFAULT_CONFIG
    torch.manual_seed(0)
    model = hz.ZoneGRU(len(hz.feature_names(cfg)), 8)
    scaler = hz.Scaler(np.zeros(len(hz.feature_names(cfg))), np.ones(len(hz.feature_names(cfg))))
    rms = _degrading()
    full = hz.predict_unit(model, scaler, cfg, _unit(rms))
    cut = hz.predict_unit(model, scaler, cfg, _unit(rms[:70]))
    np.testing.assert_allclose(cut["p_red"].to_numpy(), full["p_red"].to_numpy()[:70], rtol=1e-5, atol=1e-6)


def test_rule_baseline_tracks_sensor_bands_without_latching_old_alerts():
    rms = np.ones(80)
    rms[20:30] = 1.4
    rms[40:45] = 3.2
    feats = _unit(rms)
    pred = hz.rule_baseline(feats, ["Bearing9_9"], hz.DEFAULT_CONFIG, red_ratio=3.0)
    z = pred["zone"].to_numpy()
    assert (z[:20] == 0).all()
    assert (z[24:30] == 1).all()
    assert (z[30:40] == 0).all()
    assert (z[40:45] == 2).all()
    assert (z[45:] == 0).all()


def test_metrics_report_first_red_and_misses():
    lab = hz.label_unit(_unit(_degrading()), hz.DEFAULT_CONFIG)
    perfect = lab.assign(zone=lab["true_zone"])
    m = hz.zone_metrics(perfect)
    assert m["unit_balanced_balanced_accuracy"] == pytest.approx(1.0)
    assert m["per_unit"]["Bearing9_9"]["first_red_min_before_endpoint"] is not None
    assert "red_too_early" not in m["per_unit"]["Bearing9_9"]
    assert "median_first_red_min_before_failure" not in m
    never = lab.assign(zone=0)
    m = hz.zone_metrics(never)
    assert m["units_red_raised"] == 0
    assert m["per_unit"]["Bearing9_9"]["minutes_green_while_red"] == int((lab["true_zone"] == 2).sum())


def test_health_zones_view_without_model(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import pdm.health_zones_ui as ui
    from pdm.paths import project_root

    monkeypatch.setattr(ui, "list_zone_runs", lambda dataset_id="bearings": [])
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=30)
    at.session_state["screen_selection"] = "Model Report"
    at.session_state["report_view"] = "Health zones"
    at.run()
    assert not at.exception
    assert any("pdm zones-train" in str(c.value) for c in at.code)


def test_health_zones_load_the_run_snapshot(monkeypatch):
    import pdm.data.prepare as prepare
    import pdm.health_zones_ui as ui

    calls = []

    def load_processed(dataset_id, dataset_version=None):
        calls.append((dataset_id, dataset_version))
        return {"features": pd.DataFrame(), "split": {"train": []}, "dataset_version": dataset_version}

    monkeypatch.setattr(prepare, "load_processed", load_processed)
    ui._features.clear()
    try:
        _, _, version = ui._features("saved-zone-version")
        assert version == "saved-zone-version"
        assert calls == [("bearings", "saved-zone-version")]
    finally:
        ui._features.clear()


def test_health_zones_reject_a_mismatched_snapshot(monkeypatch):
    import pdm.data.prepare as prepare
    import pdm.health_zones_ui as ui

    monkeypatch.setattr(prepare, "load_processed", lambda dataset_id, version: {
        "features": pd.DataFrame(), "split": {}, "dataset_version": "different-version",
    })
    ui._features.clear()
    try:
        with pytest.raises(ValueError, match="does not match zone run"):
            ui._features("saved-zone-version")
    finally:
        ui._features.clear()


def test_legacy_zone_runs_are_listed_as_incompatible_and_rejected(monkeypatch, tmp_path):
    from pdm.io_util import atomic_write_json

    monkeypatch.setattr(hz, "zones_root", lambda dataset_id="bearings": tmp_path)
    old = tmp_path / "old"
    current = tmp_path / "current"
    incomplete = tmp_path / "incomplete"
    malformed = tmp_path / "malformed"
    for directory in (old, current, incomplete, malformed):
        directory.mkdir()
        (directory / "model.pt").write_bytes(b"model placeholder")
    old_meta = {"config": {"red_minutes": 30.0}}
    atomic_write_json(old / "meta.json", old_meta)
    atomic_write_json(current / "meta.json", {
        "zone_definition": hz.ZONE_DEFINITION_VERSION,
        "config": {**hz.DEFAULT_CONFIG, "hidden_size": 8, "dropout": 0.0},
        "dataset_version": "snapshot-v1", "zones": list(hz.ZONES),
        "split": {"train": [], "validation": [], "test": []},
        "feature_names": ["x"], "scaler": {"mean": [0.0], "std": [1.0]},
    })
    atomic_write_json(incomplete / "meta.json", {
        "zone_definition": hz.ZONE_DEFINITION_VERSION,
        "config": {"red_ratio": 2.0},
    })
    (malformed / "meta.json").write_text("[]")

    listed = {run["run_id"]: run for run in hz.list_zone_runs()}
    assert listed["old"]["compatible"] is False
    assert listed["current"]["compatible"] is True
    assert listed["incomplete"]["compatible"] is False
    assert listed["malformed"]["compatible"] is False
    with pytest.raises(ValueError, match="older zone definition"):
        hz.load_zone_run(old)
    with pytest.raises(ValueError, match="incomplete zone metadata"):
        hz.load_zone_run(incomplete)
    with pytest.raises(ValueError, match="obsolete"):
        hz.predict_unit(None, None, old_meta["config"], _unit([1.0, 1.0]))


def test_health_zones_view_explains_legacy_runs_need_retraining(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import pdm.health_zones_ui as ui
    from pdm.paths import project_root

    monkeypatch.setattr(ui, "list_zone_runs", lambda dataset_id="bearings": [
        {"run_id": "old-run", "dir": "/tmp/old-run", "compatible": False},
    ])
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=30)
    at.session_state["screen_selection"] = "Model Report"
    at.session_state["report_view"] = "Health zones"
    at.run()
    assert not at.exception
    assert any("obsolete or incomplete zone definition" in str(w.value) for w in at.warning)
    assert any("zones-train" in str(c.value) for c in at.code)


def test_health_zones_view_reports_missing_run_snapshot(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import pdm.health_zones_ui as ui
    from pdm.paths import project_root

    monkeypatch.setattr(ui, "list_zone_runs", lambda dataset_id="bearings": [
        {"run_id": "saved-run", "dir": "/tmp/saved-run", "compatible": True},
    ])
    monkeypatch.setattr(ui, "_run", lambda rdir: (
        None, None, {"dataset_version": "missing-version"}, {},
    ))
    monkeypatch.setattr(ui, "_features", lambda version: (_ for _ in ()).throw(
        FileNotFoundError("missing-version")))
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=30)
    at.session_state["screen_selection"] = "Model Report"
    at.session_state["report_view"] = "Health zones"
    at.run()
    assert not at.exception
    assert any("snapshot for this zone run is unavailable" in str(e.value) for e in at.error)
