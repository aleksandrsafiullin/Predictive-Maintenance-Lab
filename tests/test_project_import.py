from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from pdm.data.project_import import import_project
from pdm.data.project_prepare import load_snapshot, prepare_project
from pdm.projects import ProjectStore
from tests.project_contract import make_contract_snapshot


def _write_generic(root: Path, units: list[str], *, offset: float = 0.0) -> Path:
    root.mkdir(parents=True)
    pd.DataFrame(
        {"unit_id": unit, "timestamp_s": float(t), "vibration": offset + i + t / 10}
        for i, unit in enumerate(units) for t in range(8)
    ).to_csv(root / "signals.csv", index=False)
    return root


def _spec(primary: Path, validation: Path | None = None, test: Path | None = None) -> dict:
    return {
        "primary": {"mode": "folder", "path": str(primary)},
        "validation_mode": "folder" if validation else "auto",
        "validation": {"mode": "folder", "path": str(validation)} if validation else None,
        "test_mode": "folder" if test else "auto",
        "test": {"mode": "folder", "path": str(test)} if test else None,
        "signal_column": "vibration", "signal_label": "Signed vibration", "signal_unit": "g",
        "thresholds": {"mode": "absolute", "direction": "above", "yellow": 4, "red": 8},
        "seed": 3,
    }


@pytest.mark.parametrize("manual_validation,manual_test", [(False, False), (True, False), (False, True), (True, True)])
def test_independent_manual_and_auto_whole_unit_splits(
    tmp_path: Path, manual_validation: bool, manual_test: bool
):
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("Project", "generic_sensor_csv")["project_id"]
    primary = _write_generic(tmp_path / "primary", [f"p{i}" for i in range(6)], offset=-5)
    validation = _write_generic(tmp_path / "validation", ["v0"], offset=10) if manual_validation else None
    test = _write_generic(tmp_path / "test", ["t0"], offset=20) if manual_test else None
    manifest = import_project(pid, _spec(primary, validation, test), store=store)
    snapshot = prepare_project(pid, manifest["manifest_id"], store=store)
    loaded = load_snapshot(pid, store=store)
    split = loaded["split"]
    assert all(split[group] for group in ("train", "validation", "test"))
    assert not (set(split["train"]) & set(split["validation"]) | set(split["train"]) & set(split["test"]))
    assert ("v0" in split["validation"]) == manual_validation
    assert ("t0" in split["test"]) == manual_test
    assert loaded["schema"]["input_columns"] == ["signal"]
    assert loaded["schema"]["output_domain"] == "real"
    assert snapshot["fingerprint"]["source_digest"] == manifest["source_digest"]
    assert loaded["report"]["by_split"]["train"]["units"] == len(split["train"])
    assert loaded["report"]["quality"]["gap_boundaries"] == 0


def test_too_few_units_duplicate_content_and_failed_import_rollback(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("Project", "generic_sensor_csv")["project_id"]
    good = _write_generic(tmp_path / "good", [f"u{i}" for i in range(5)])
    first = import_project(pid, _spec(good), store=store)
    prepared = prepare_project(pid, first["manifest_id"], store=store)
    old = store.get(pid)
    small = _write_generic(tmp_path / "small", ["only"])
    with pytest.raises(ValueError, match="at least"):
        import_project(pid, _spec(small), store=store)
    assert store.get(pid) == old
    assert not list((store.project_path(pid) / "source").glob(".import-*"))
    duplicate = _write_generic(tmp_path / "duplicate", ["renamed"])
    # Match u0's physical samples despite a different unit ID and file name.
    frame = pd.read_csv(good / "signals.csv").query("unit_id == 'u0'").copy()
    frame["unit_id"] = "renamed"
    frame.to_csv(duplicate / "signals.csv", index=False)
    with pytest.raises(ValueError, match="Duplicate physical history"):
        import_project(pid, _spec(good, duplicate), store=store)
    assert store.get(pid)["active_snapshot_id"] == prepared["snapshot_id"]


def test_signed_signal_rejected_row_gap_is_causal(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("Project", "generic_sensor_csv")["project_id"]
    root = _write_generic(tmp_path / "source", [f"u{i}" for i in range(5)], offset=-10)
    source = pd.read_csv(root / "signals.csv")
    source.loc[(source.unit_id == "u0") & (source.timestamp_s == 3), "vibration"] = float("nan")
    source.to_csv(root / "signals.csv", index=False)
    manifest = import_project(pid, _spec(root), store=store)
    prepare_project(pid, manifest["manifest_id"], store=store)
    loaded = load_snapshot(pid, store=store)
    one = loaded["features"].query("unit_id == 'u0'")
    assert one["signal"].min() < 0
    assert one.loc[one.timestamp_s == 4, "gap_before"].item()
    group = next(name for name in ("train", "validation", "test") if "u0" in loaded["split"][name])
    assert loaded["report"]["by_split"][group]["rejected_signal_rows"] == 1

    earlier = one.loc[one.timestamp_s <= 5, "gap_before"].tolist()
    source.loc[(source.unit_id == "u0") & (source.timestamp_s >= 6), "timestamp_s"] += 1000
    source.to_csv(root / "signals.csv", index=False)
    second = import_project(pid, _spec(root), store=store)
    prepare_project(pid, second["manifest_id"], store=store)
    revised = load_snapshot(pid, store=store)["features"].query("unit_id == 'u0'")
    assert revised.loc[revised.timestamp_s <= 5, "gap_before"].tolist() == earlier


def test_snapshot_hash_and_cancellation_preserve_previous_activation(tmp_path: Path):
    store, project, snapshot = make_contract_snapshot(tmp_path / "projects")
    pid = project["project_id"]
    assert store.get(pid)["active_snapshot_id"] == snapshot["snapshot_id"]
    with pytest.raises(InterruptedError, match="cancelled"):
        prepare_project(pid, store.get(pid)["source_manifest"]["manifest_id"],
                        store=store, should_stop=lambda: True)
    assert store.get(pid)["active_snapshot_id"] == snapshot["snapshot_id"]
    (snapshot["dir"] / "split.json").write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_snapshot(pid, store=store)


def test_two_same_kind_projects_keep_distinct_source_and_snapshots(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    source_a = _write_generic(tmp_path / "a", [f"a{i}" for i in range(4)], offset=-10)
    source_b = _write_generic(tmp_path / "b", [f"b{i}" for i in range(4)], offset=10)
    ids = [store.create(name, "generic_sensor_csv")["project_id"] for name in ("A", "B")]
    manifests = [import_project(pid, _spec(source), store=store)
                 for pid, source in zip(ids, (source_a, source_b), strict=True)]
    snapshots = [prepare_project(pid, man["manifest_id"], store=store)
                 for pid, man in zip(ids, manifests, strict=True)]
    assert manifests[0]["source_digest"] != manifests[1]["source_digest"]
    assert snapshots[0]["dir"] != snapshots[1]["dir"]
    assert store.run_path(ids[0], "same-run") != store.run_path(ids[1], "same-run")


def _write_filter(path: Path, units: list[int], *, base: int, official_test: bool = False):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for unit in units:
        for t in range(6):
            row = {"Data_No": unit, "Differential_pressure": base + unit * 10 + t,
                   "Flow_rate": 1.0, "Time": t, "Dust_feed": 2.0, "Dust": "coal"}
            if official_test:
                row["RUL"] = 20
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_hse_manual_validation_and_author_test_never_train(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("HSE", "hse_filters")["project_id"]
    primary = tmp_path / "primary"
    validation = tmp_path / "validation"
    _write_filter(primary / "Train_Data_CSV.csv", [1, 2, 3, 4], base=100)
    _write_filter(primary / "Test_Data_CSV.csv", [1], base=200, official_test=True)
    _write_filter(validation / "Train_Data_CSV.csv", [1], base=300)
    spec = {
        "primary": {"mode": "folder", "path": str(primary)},
        "validation_mode": "folder", "validation": {"mode": "folder", "path": str(validation)},
        "signal_unit": "Pa",
        "thresholds": {"mode": "absolute", "direction": "above", "yellow": 300, "red": 600},
    }
    manifest = import_project(pid, spec, store=store)
    prepared = prepare_project(pid, manifest["manifest_id"], store=store)
    split = prepared["split"]
    assert split["validation"] == ["validation_Train_1"]
    assert split["test"] == ["Test_1"]
    assert all(uid.startswith("Train_") for uid in split["train"])
    assert "official_rul_at_prefix_end_s" not in load_snapshot(pid, store=store)["features"]
    _write_filter(validation / "Test_Data_CSV.csv", [2], base=400, official_test=True)
    with pytest.raises(ValueError, match="cannot be supplied as Validation"):
        import_project(pid, spec, store=store)


def test_hse_rejects_nonmonotonic_acquisition_time(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("HSE", "hse_filters")["project_id"]
    primary = tmp_path / "primary"
    _write_filter(primary / "Train_Data_CSV.csv", [1, 2, 3, 4], base=100)
    frame = pd.read_csv(primary / "Train_Data_CSV.csv")
    unit_one = frame.index[frame.Data_No == 1].tolist()
    frame.loc[unit_one[1], "Time"] = 3
    frame.loc[unit_one[2], "Time"] = 2
    frame.to_csv(primary / "Train_Data_CSV.csv", index=False)
    spec = {"primary": {"mode": "folder", "path": str(primary)},
            "signal_unit": "Pa",
            "thresholds": {"mode": "absolute", "direction": "above", "yellow": 300, "red": 600}}
    with pytest.raises(ValueError, match="Nonmonotonic source time"):
        import_project(pid, spec, store=store)
    assert store.get(pid)["source_manifest"] is None


def test_legacy_bearing_wrap_uses_combined_axis_and_causal_threshold_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw = tmp_path / "raw"
    (raw / "bearings").mkdir(parents=True)
    (raw / "bearings" / "1.csv").write_text("x\n1\n")
    monkeypatch.setattr("pdm.projects.data_raw", lambda: raw)
    monkeypatch.setattr("pdm.projects.data_processed", lambda: tmp_path / "processed")
    monkeypatch.setattr("pdm.projects.runs_root", lambda: tmp_path / "legacy-runs")
    features = pd.DataFrame(
        {"unit_id": unit, "timestamp_s": float(t), "horizontal_rms": 1 + i + t / 10,
         "vertical_rms": 2 + i + t / 10, "gap_before": t == 0}
        for i, unit in enumerate(("b1", "b2", "b3")) for t in range(6)
    )
    units = pd.DataFrame({"unit_id": ["b1", "b2", "b3"],
                          "origin_unit_id": ["b1", "b2", "b3"],
                          "observation_end_s": [5.0] * 3})
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda dataset_id: {
        "features": features, "units": units,
        "split": {"train": ["b1"], "validation": ["b2"], "test": ["b3"],
                  "protocol": "whole_unit_fixture"},
        "fingerprint": {"split_hash": "fixture"},
    })
    store = ProjectStore(tmp_path / "projects")
    project = store.register_legacy()[0]
    ref = prepare_project(project["project_id"], project["source_manifest"]["manifest_id"], store=store)
    loaded = load_snapshot(project["project_id"], store=store)
    assert loaded["features"].iloc[0]["signal"] == 2.0
    assert ref["schema"]["thresholds"]["mode"] == "initial_baseline_multiple"
    assert ref["schema"]["thresholds"]["baseline_n"] == 5
    assert ref["schema"]["thresholds"]["yellow_formula"] == "max(median + onset_sigma * population_sd, onset_ratio * median)"
    assert project["source_manifest"]["raw_path"] == str(raw / "bearings")


def test_xjtu_single_zip_maps_members_to_primary_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    source = tmp_path / "bearing-zip"
    source.mkdir()
    with zipfile.ZipFile(source / "bearings.zip", "w") as archive:
        archive.writestr("35Hz12kN/Bearing1_1/1.csv", "1,2\n")

    def synthetic_features(_cfg, raw_dir=None, progress=None):
        assert raw_dir is not None and (raw_dir / "primary" / "bearings.zip").is_file()
        return pd.DataFrame(
            {"unit_id": f"Bearing1_{unit}", "timestamp_s": float(t),
             "horizontal_rms": unit + t / 10, "vertical_rms": unit + t / 5,
             "gap_before": t == 0,
             "relpath": f"35Hz12kN/Bearing1_{unit}/{t+1}.csv"}
            for unit in range(1, 4) for t in range(4)
        )

    monkeypatch.setattr("pdm.data.bearings.extract_bearings_features", synthetic_features)
    monkeypatch.setattr("pdm.data.bearings.build_bearing_units", lambda features, _cfg: pd.DataFrame({
        "unit_id": [f"Bearing1_{i}" for i in range(1, 4)],
        "origin_unit_id": [f"Bearing1_{i}" for i in range(1, 4)],
    }))
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("Bearings", "xjtu_bearings")["project_id"]
    spec = {"primary": {"mode": "folder", "path": str(source)},
            "thresholds": {"mode": "absolute", "direction": "above", "yellow": 2, "red": 4}}
    manifest = import_project(pid, spec, store=store)
    prepared = prepare_project(pid, manifest["manifest_id"], store=store)
    assert prepared["split"]["realized_counts"] == {"train": 1, "validation": 1, "test": 1}
    assert set(load_snapshot(pid, store=store)["units"]["source_group"]) == {"primary"}


def test_replacement_import_activates_source_and_snapshot_together(tmp_path: Path):
    store, project, old_ref = make_contract_snapshot(tmp_path / "projects")
    pid = project["project_id"]
    before = store.get(pid)
    replacement = _write_generic(tmp_path / "replacement", [f"new{i}" for i in range(5)], offset=20)
    staged = import_project(pid, _spec(replacement), store=store)
    assert store.get(pid) == before
    staged_file = store.project_path(pid) / "source" / staged["manifest_id"] / "primary" / "signals.csv"
    staged_file.write_text("broken")
    with pytest.raises(ValueError, match="Imported source changed"):
        prepare_project(pid, staged["manifest_id"], store=store)
    assert store.get(pid) == before
    second = import_project(pid, _spec(replacement), store=store)
    new_ref = prepare_project(pid, second["manifest_id"], store=store)
    after = store.get(pid)
    assert after["source_manifest"]["manifest_id"] == second["manifest_id"]
    assert after["active_snapshot_id"] == new_ref["snapshot_id"] != old_ref["snapshot_id"]
