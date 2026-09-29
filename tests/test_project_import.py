from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from pdm.data.project_import import (
    XJTU_BASELINE_THRESHOLDS,
    _signal_schema,
    import_project,
    validate_thresholds,
)
from pdm.data.project_prepare import (
    ZONE_LIMITS_FILE,
    load_snapshot,
    load_zone_limits,
    move_units,
    prepare_project,
    preview_move,
    read_zone_limits,
    save_zone_limits,
)
from pdm.io_util import sha256_file
from pdm.project_zones import has_valid_rule
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


@pytest.fixture
def idle_worker(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_args, **_kwargs: False)


def _dir_hashes(directory: Path) -> dict[str, str]:
    return {path.name: sha256_file(path) for path in sorted(directory.iterdir())}


def _snapshot_dirs(store: ProjectStore, pid: str) -> list[str]:
    return sorted(path.name for path in (store.project_path(pid) / "snapshots").iterdir())


def test_move_units_publishes_new_snapshot_and_keeps_old(tmp_path: Path, idle_worker):
    store, project, parent = make_contract_snapshot(tmp_path / "projects")
    pid, old_id = project["project_id"], parent["snapshot_id"]
    old_hashes = _dir_hashes(parent["dir"])
    old_split = parent["split"]
    moved = sorted(old_split["train"])[:2]
    ref = move_units(pid, list(reversed(moved)), "test", expected_snapshot_id=old_id, store=store)

    assert _dir_hashes(parent["dir"]) == old_hashes
    assert load_snapshot(pid, old_id, store=store)["split"] == old_split
    record = store.get(pid)
    assert record["active_snapshot_id"] == ref["snapshot_id"] != old_id
    assert record["selected_run_id"] is None and record["state"] == "ready"
    assert ref["parent_snapshot_id"] == old_id
    loaded = load_snapshot(pid, store=store)
    split = loaded["split"]
    assert split["train"] == sorted(set(old_split["train"]) - set(moved))
    assert split["test"] == sorted([*old_split["test"], *moved])
    assert split["validation"] == sorted(old_split["validation"])
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        assert not set(split[a]) & set(split[b])
    assert split["realized_counts"] == {name: len(split[name]) for name in ("train", "validation", "test")}
    assert split["protocol"] == "whole_unit_project_v1_manual"
    assert split["parent_snapshot_id"] == old_id
    for key in ("seed", "desired_weights", "manual_modes"):
        assert split[key] == old_split[key]
    [entry] = split["manual_moves"]
    assert entry["unit_ids"] == moved and entry["to"] == "test"
    assert entry["from"] == {uid: "train" for uid in moved}
    assert entry["at"].endswith("+00:00")
    for name in ("features.parquet", "units.parquet", "feature_schema.json"):
        assert sha256_file(ref["dir"] / name) == old_hashes[name]
    report = loaded["report"]
    assert report["snapshot_id"] == ref["snapshot_id"] and report["parent_snapshot_id"] == old_id
    assert report["split_counts"] == split["realized_counts"]
    assert report["by_split"]["test"]["units"] == len(split["test"])
    assert sum(part["rows"] for part in report["by_split"].values()) == len(loaded["features"])
    for key in ("quality", "source_digest", "outcome_semantics"):
        assert report[key] == parent["report"][key]
    fingerprint = loaded["fingerprint"]
    assert fingerprint["parent_snapshot_id"] == old_id
    assert fingerprint["source_digest"] == parent["fingerprint"]["source_digest"]
    assert fingerprint["source_manifest_id"] == parent["fingerprint"]["source_manifest_id"]
    assert fingerprint["split_hash"] != parent["fingerprint"]["split_hash"]

    # A second move appends to the parent's history.
    back = move_units(pid, [moved[0]], "train", expected_snapshot_id=ref["snapshot_id"], store=store)
    assert [m["to"] for m in back["split"]["manual_moves"]] == ["test", "train"]
    assert back["split"]["manual_moves"][1]["from"] == {moved[0]: "test"}


def test_move_units_refuses_empty_set_unknown_and_same_set(tmp_path: Path, idle_worker):
    store, project, parent = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], parent["snapshot_id"]
    split = parent["split"]
    before = store.get(pid)
    cases = [
        ([], "train", "Choose at least one unit."),
        (["ghost"], "train", "Unknown unit: ghost."),
        ([split["train"][0]], "train", f"{split['train'][0]} is already in Training Data."),
        (list(split["test"]), "train", "Testing Data would have no units. Keep at least one unit in each set."),
    ]
    for units, destination, message in cases:
        assert preview_move(parent, units, destination)["problem"] == message
        with pytest.raises(ValueError) as caught:
            move_units(pid, units, destination, expected_snapshot_id=sid, store=store)
        assert str(caught.value) == message
    assert store.get(pid) == before
    assert _snapshot_dirs(store, pid) == [sid]
    ok = preview_move(parent, [split["train"][0]], "validation")
    assert ok["problem"] is None
    assert ok["counts"]["train"] == len(split["train"]) - 1
    assert ok["counts"]["validation"] == len(split["validation"]) + 1


def test_move_units_stale_snapshot_race(tmp_path: Path, idle_worker, monkeypatch: pytest.MonkeyPatch):
    store, project, parent = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], parent["snapshot_id"]
    unit = parent["split"]["train"][0]
    first = move_units(pid, [unit], "validation", expected_snapshot_id=sid, store=store)
    with pytest.raises(ValueError) as caught:
        move_units(pid, [unit], "validation", expected_snapshot_id=sid, store=store)
    assert str(caught.value) == "The data changed since this page loaded. Reload Data Quality and try again."
    assert _snapshot_dirs(store, pid) == sorted([sid, first["snapshot_id"]])

    # A concurrent publish while this move is staging is caught inside the lock.
    import pdm.data.project_prepare as prepare

    real_copy = prepare._copy_snapshot_file
    competing = "competingsnapshot"

    def racing_copy(source: Path, target: Path) -> None:
        if store.get(pid)["active_snapshot_id"] != competing:
            store.update(pid, active_snapshot_id=competing, selected_run_id=None)
        real_copy(source, target)

    monkeypatch.setattr(prepare, "_copy_snapshot_file", racing_copy)
    with pytest.raises(ValueError, match="The data changed since this page loaded"):
        move_units(pid, [unit], "train", expected_snapshot_id=first["snapshot_id"], store=store)
    assert _snapshot_dirs(store, pid) == sorted([sid, first["snapshot_id"]])
    assert store.get(pid)["active_snapshot_id"] == competing


def test_move_units_rollback_on_activation_failure(tmp_path: Path, idle_worker, monkeypatch: pytest.MonkeyPatch):
    store, project, parent = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], parent["snapshot_id"]
    before = store.get(pid)

    def broken(self, registry, project_id, snapshot_id):
        raise OSError("disk full")

    monkeypatch.setattr(ProjectStore, "_activate_snapshot_locked", broken)
    with pytest.raises(OSError, match="disk full"):
        move_units(pid, [parent["split"]["train"][0]], "test", expected_snapshot_id=sid, store=store)
    assert _snapshot_dirs(store, pid) == [sid]
    assert store.get(pid) == before
    load_snapshot(pid, sid, store=store)


def _hse_project(tmp_path: Path) -> tuple[ProjectStore, str, dict]:
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
    return store, pid, prepare_project(pid, manifest["manifest_id"], store=store)


def test_move_units_refuses_hse_author_test_leaving_test(tmp_path: Path, idle_worker):
    store, pid, parent = _hse_project(tmp_path)
    sid = parent["snapshot_id"]
    loaded = load_snapshot(pid, sid, store=store)
    assert preview_move(loaded, ["Train_1"], "test")["fixed_units"] == ["Test_1"]
    for destination in ("train", "validation"):
        assert preview_move(loaded, ["Test_1"], destination)["problem"] == "Official HSE test units stay in Testing Data."
        with pytest.raises(ValueError) as caught:
            move_units(pid, ["Test_1"], destination, expected_snapshot_id=sid, store=store)
        assert str(caught.value) == "Official HSE test units stay in Testing Data."
    moved = move_units(pid, ["Train_1"], "test", expected_snapshot_id=sid, store=store)
    assert moved["split"]["test"] == ["Test_1", "Train_1"]
    with pytest.raises(ValueError, match="Official HSE test units"):
        move_units(pid, ["Test_1", "Train_1"], "train", expected_snapshot_id=moved["snapshot_id"], store=store)


def test_move_units_refuses_linked_legacy(tmp_path: Path, idle_worker):
    store, project, parent = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], parent["snapshot_id"]
    registry = store._load()
    registry["projects"][pid]["storage_mode"] = "linked_legacy"
    store._save(registry)
    with pytest.raises(ValueError) as caught:
        move_units(pid, [parent["split"]["train"][0]], "test", expected_snapshot_id=sid, store=store)
    assert str(caught.value) == ("This project uses the published split of its source dataset. "
                                 "Create a new project to change the split.")
    assert _snapshot_dirs(store, pid) == [sid]


def test_move_units_refuses_while_heavy_job_active(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store, project, parent = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], parent["snapshot_id"]
    before = store.get(pid)
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_args, **_kwargs: True)
    with pytest.raises(RuntimeError) as caught:
        move_units(pid, [parent["split"]["train"][0]], "test", expected_snapshot_id=sid, store=store)
    assert str(caught.value) == "Wait for the current job to finish before changing sets."

    # A job that starts while the move is staging is caught inside the lock.
    calls = iter([False, True])
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_args, **_kwargs: next(calls))
    with pytest.raises(RuntimeError, match="Wait for the current job"):
        move_units(pid, [parent["split"]["train"][0]], "test", expected_snapshot_id=sid, store=store)
    assert store.get(pid) == before
    assert _snapshot_dirs(store, pid) == [sid]


def test_move_units_does_not_deadlock_or_hold_lock_while_copying(
    tmp_path: Path, idle_worker, monkeypatch: pytest.MonkeyPatch
):
    root = tmp_path / "projects"
    store, project, parent = make_contract_snapshot(root)
    pid, sid = project["project_id"], parent["snapshot_id"]
    unit = parent["split"]["train"][0]
    child = textwrap.dedent("""
        import json, sys, time
        import pdm.worker
        pdm.worker.heavy_job_active = lambda *args, **kwargs: False
        from pdm.data.project_prepare import move_units
        from pdm.projects import ProjectStore
        root, pid, sid, unit = sys.argv[1:5]
        store = ProjectStore(root)
        start = time.perf_counter()
        ref = move_units(pid, [unit], "validation", expected_snapshot_id=sid, store=store)
        print(json.dumps({"elapsed": time.perf_counter() - start, "snapshot_id": ref["snapshot_id"]}))
    """)
    done = subprocess.run([sys.executable, "-c", child, str(root), pid, sid, unit],
                          capture_output=True, text=True, timeout=30, check=True)
    result = json.loads(done.stdout.strip().splitlines()[-1])
    assert result["elapsed"] < 5
    assert store.get(pid)["active_snapshot_id"] == result["snapshot_id"]

    import pdm.data.project_prepare as prepare

    real_copy = prepare._copy_snapshot_file
    entered, release = threading.Event(), threading.Event()

    def blocking_copy(source: Path, target: Path) -> None:
        entered.set()
        release.wait(10)
        real_copy(source, target)

    monkeypatch.setattr(prepare, "_copy_snapshot_file", blocking_copy)
    outcome: dict = {}

    def run_move() -> None:
        try:
            outcome["ref"] = move_units(pid, [unit], "train", expected_snapshot_id=result["snapshot_id"], store=store)
        except BaseException as exc:  # noqa: BLE001
            outcome["error"] = exc

    mover = threading.Thread(target=run_move, daemon=True)
    mover.start()
    try:
        assert entered.wait(5)
        got: dict = {}
        reader = threading.Thread(target=lambda: got.update(record=store.get(pid)), daemon=True)
        start = time.perf_counter()
        reader.start()
        reader.join(1)
        assert not reader.is_alive(), "store.get blocked while move_units copied parquet"
        assert time.perf_counter() - start < 1
        assert got["record"]["active_snapshot_id"] == result["snapshot_id"]
    finally:
        release.set()
        mover.join(10)
    assert "error" not in outcome
    assert store.get(pid)["active_snapshot_id"] == outcome["ref"]["snapshot_id"]


def test_preview_move_without_units_frame(tmp_path: Path):
    snapshot = {"split": {"train": ["a", "b"], "validation": ["c"], "test": ["d"]}}
    result = preview_move(snapshot, ["a"], "test")
    assert result == {"counts": {"train": 1, "validation": 1, "test": 2}, "problem": None, "fixed_units": []}
    no_group = {**snapshot, "units": pd.DataFrame({"unit_id": ["a", "b", "c", "d"]})}
    assert preview_move(no_group, ["d"], "train")["fixed_units"] == []
    assert preview_move(no_group, ["d"], "train")["problem"] == (
        "Testing Data would have no units. Keep at least one unit in each set.")


def test_move_units_old_run_not_selectable_on_new_snapshot(tmp_path: Path, idle_worker):
    store, project, parent = make_contract_snapshot(tmp_path / "projects")
    pid, old_id = project["project_id"], parent["snapshot_id"]
    run_dir = store.run_path(pid, "run1")
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps(
        {"project_id": pid, "snapshot_id": old_id, "run_id": "run1",
         "task": "signal_forecast", "status": "completed"}))
    assert store.update(pid, selected_run_id="run1")["selected_run_id"] == "run1"
    ref = move_units(pid, [parent["split"]["train"][0]], "test", expected_snapshot_id=old_id, store=store)
    record = store.get(pid)
    assert record["active_snapshot_id"] == ref["snapshot_id"]
    assert record["selected_run_id"] is None
    with pytest.raises(ValueError, match="Selected run"):
        store.update(pid, selected_run_id="run1")
    assert (run_dir / "manifest.json").is_file()
    load_snapshot(pid, old_id, store=store)


def test_validate_thresholds_accepts_xjtu_baseline_rule_only():
    rule = validate_thresholds("xjtu_bearings", {"mode": "initial_baseline_multiple", "direction": "above"})
    assert rule == XJTU_BASELINE_THRESHOLDS
    custom = validate_thresholds("xjtu_bearings", {**XJTU_BASELINE_THRESHOLDS, "baseline_n": 4, "red_ratio": 3})
    assert (custom["baseline_n"], custom["red_ratio"]) == (4, 3.0)
    for kind, bad in (("generic_sensor_csv", {"mode": "initial_baseline_multiple"}),
                      ("hse_filters", {"mode": "initial_baseline_multiple"}),
                      ("xjtu_bearings", {"mode": "initial_baseline_multiple", "direction": "below"}),
                      ("xjtu_bearings", {"mode": "initial_baseline_multiple", "baseline_n": 0}),
                      ("xjtu_bearings", {"mode": "initial_baseline_multiple", "baseline_n": 2.5}),
                      ("xjtu_bearings", {"mode": "initial_baseline_multiple", "red_ratio": float("nan")}),
                      ("xjtu_bearings", {"mode": "relative"})):
        with pytest.raises(ValueError):
            validate_thresholds(kind, bad)


def test_signal_schema_rejects_baseline_rule_outside_xjtu_and_allows_unzoned_generic():
    baseline = {"mode": "initial_baseline_multiple", "direction": "above"}
    for kind in ("hse_filters", "generic_sensor_csv"):
        with pytest.raises(ValueError):
            _signal_schema(kind, {"signal_column": "vibration", "signal_unit": "g", "thresholds": baseline})
    with pytest.raises(ValueError):
        _signal_schema("hse_filters", {"thresholds": {}})
    schema = _signal_schema("generic_sensor_csv", {"signal_column": "vibration", "signal_unit": "g"})
    assert schema["thresholds"] == {} and not has_valid_rule(schema)


def test_first_generic_import_without_limits_is_not_zoned(tmp_path: Path):
    store = ProjectStore(tmp_path / "projects")
    pid = store.create("Project", "generic_sensor_csv")["project_id"]
    primary = _write_generic(tmp_path / "primary", [f"p{i}" for i in range(6)])
    manifest = import_project(pid, {**_spec(primary), "thresholds": {}}, store=store)
    prepare_project(pid, manifest["manifest_id"], store=store)
    assert not has_valid_rule(load_snapshot(pid, store=store)["schema"])


def test_save_zone_limits_writes_sidecar_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_a: False)
    store, project, ref = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], ref["snapshot_id"]
    before = store.get(pid)
    directory = store.snapshot_path(pid, sid)
    files = {name: (directory / name).read_bytes() for name in ("feature_schema.json", "processed_fingerprint.json")}
    rule = {"mode": "absolute", "direction": "below", "yellow": -0.2, "red": -0.6}
    save_zone_limits(pid, rule, expected_snapshot_id=sid, store=store)
    assert load_zone_limits(pid, sid, store=store) == rule
    stamp = (directory / ZONE_LIMITS_FILE).stat().st_mtime_ns
    save_zone_limits(pid, rule, expected_snapshot_id=sid, store=store)
    assert (directory / ZONE_LIMITS_FILE).stat().st_mtime_ns == stamp
    assert {name: (directory / name).read_bytes() for name in files} == files
    loaded = load_snapshot(pid, store=store)
    assert loaded["snapshot_id"] == sid and loaded["schema"]["thresholds"]["yellow"] == 0.4
    assert store.get(pid) == before
    with pytest.raises(ValueError):
        save_zone_limits(pid, {**rule, "yellow": -0.9}, expected_snapshot_id=sid, store=store)
    assert load_zone_limits(pid, sid, store=store) == rule


def test_save_zone_limits_refusals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    store, project, ref = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], ref["snapshot_id"]
    rule = {"mode": "absolute", "direction": "above", "yellow": 0.1, "red": 0.2}
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_a: True)
    with pytest.raises(RuntimeError, match="background job"):
        save_zone_limits(pid, rule, expected_snapshot_id=sid, store=store)
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_a: False)
    with pytest.raises(ValueError, match="changed since"):
        save_zone_limits(pid, rule, expected_snapshot_id="other", store=store)
    assert load_zone_limits(pid, sid, store=store) is None


def test_save_zone_limits_linked_legacy_writes_sidecar_not_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_a: False)
    store, project, ref = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], ref["snapshot_id"]
    directory = store.snapshot_path(pid, sid)
    schema_before = (directory / "feature_schema.json").read_bytes()
    fingerprint_before = (directory / "processed_fingerprint.json").read_bytes()
    registry = json.loads(store.registry_path.read_text())
    registry["projects"][pid]["storage_mode"] = "linked_legacy"
    store.registry_path.write_text(json.dumps(registry))
    rule = {"mode": "absolute", "direction": "above", "yellow": 0.1, "red": 0.2}
    assert save_zone_limits(pid, rule, expected_snapshot_id=sid, store=store) == rule
    assert load_zone_limits(pid, sid, store=store) == rule
    assert (directory / "feature_schema.json").read_bytes() == schema_before
    assert (directory / "processed_fingerprint.json").read_bytes() == fingerprint_before
    assert load_snapshot(pid, store=store)["schema"]["thresholds"]["yellow"] == 0.4


def test_move_units_keeps_saved_zone_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("pdm.worker.heavy_job_active", lambda *_a: False)
    store, project, ref = make_contract_snapshot(tmp_path / "projects")
    pid, sid = project["project_id"], ref["snapshot_id"]
    rule = {"mode": "absolute", "direction": "above", "yellow": 0.3, "red": 0.9}
    save_zone_limits(pid, rule, expected_snapshot_id=sid, store=store)
    unit = sorted(load_snapshot(pid, store=store)["split"]["train"])[-1]
    child = move_units(pid, [unit], "validation", expected_snapshot_id=sid, store=store)
    assert child["snapshot_id"] != sid
    assert load_zone_limits(pid, child["snapshot_id"], store=store) == rule
    assert load_snapshot(pid, child["snapshot_id"], store=store)["schema"]["thresholds"]["yellow"] == 0.4


def test_read_zone_limits_keeps_file_on_read_error_and_drops_bad_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sidecar = tmp_path / ZONE_LIMITS_FILE
    rule = {"mode": "absolute", "direction": "above", "yellow": 0.1, "red": 0.2}
    sidecar.write_text(json.dumps(rule))

    def denied(_path):
        raise PermissionError("sharing violation")

    with monkeypatch.context() as patch:
        patch.setattr("pdm.data.project_prepare.read_json", denied)
        assert read_zone_limits(tmp_path) is None
    assert sidecar.exists()
    assert read_zone_limits(tmp_path) == rule
    for bad in ("{not json", json.dumps({"mode": "absolute", "direction": "above", "yellow": 3, "red": 1}),
                json.dumps(["absolute"])):
        sidecar.write_text(bad)
        assert read_zone_limits(tmp_path) is None
        assert not sidecar.exists()
