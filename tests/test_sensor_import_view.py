"""Saved sensor roles stay visible without decoding signals or changing models."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pdm.io_util import atomic_write_json, read_json
from pdm.probabilistic import data
from pdm.probabilistic.contract import default_config
from pdm.projects import project_store


@dataclass
class Release:
    project: dict
    root: Path
    manifest: Path


@pytest.fixture
def release(tmp_path, monkeypatch):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("PDM_WORKER_ROOT", str(tmp_path / "worker"))
    root = tmp_path / "release"
    counts = {"train": 40, "validation": 8, "calibration": 10, "test": 10}
    manifest = dict(
        version="import-view-test",
        seed=13,
        schema=default_config()["schema"],
        target="vibration_rms_g",
        physical_unit="g",
        cadence_s=60,
        splits={"02_benchmark": counts},
        file_sha256={},
        total_rows=0,
        total_units=68,
    )
    global_unit = 0
    for part, count in counts.items():
        frames = []
        for unit in range(count):
            ticks = np.arange(135, dtype=float)
            times = ticks * 60
            if part == "train" and unit == 0:
                times[70:] += 120
            frames.append(
                pd.DataFrame(
                    dict(
                        unit_id=f"{part}-{unit}",
                        timestamp_s=times,
                        vibration_rms_g=0.3
                        + ticks * (0.001 + global_unit * 0.0000001)
                        + global_unit * 0.002,
                    )
                )
            )
            global_unit += 1
        path = root / "02_benchmark" / "sensor_csv" / part / "measurements.csv"
        path.parent.mkdir(parents=True)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        manifest["file_sha256"][str(path.relative_to(root))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        manifest["total_rows"] += count * 135
    manifest_path = root / "dataset_manifest.json"
    atomic_write_json(manifest_path, manifest)
    project = project_store().create("Saved Benchmark fixture", "generic_sensor_csv")
    return Release(project, root, manifest_path)


def original_import(release):
    store = project_store()
    pid = release.project["project_id"]
    directory = store.project_path(pid) / "snapshots" / "mapped-storage"
    snapshot = data.prepare_dataset(
        release.root, "02_benchmark", directory, manifest_path=release.manifest
    )
    store.update(
        pid,
        state="ready",
        active_snapshot_id=snapshot["snapshot_id"],
        source_manifest={
            "task": "probabilistic_signal_forecast",
            "manifest_path": str(release.manifest),
            "source_root": str(release.root),
            "suite": "02_benchmark",
            "snapshots": {snapshot["snapshot_id"]: str(directory)},
        },
    )
    return snapshot


def app(project_id, step="Import data"):
    view = AppTest.from_string("from pdm.project_ui import main\nmain()", default_timeout=30)
    view.session_state["project_id"] = project_id
    view.session_state["project_step"] = step
    return view.run()


def seal(directory):
    return {
        str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(directory).rglob("*")
        if path.is_file()
    }


def test_import_displays_four_saved_roles_without_decoding_or_mutating(release, monkeypatch):
    from pdm.project_ui import _snapshot_summaries

    snapshot = original_import(release)
    pid = release.project["project_id"]
    saved = seal(snapshot["directory"])
    project = project_store().get(pid)

    def forbidden(*args, **kwargs):
        raise AssertionError("Import overview must use admitted metadata, not read signal values")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    summary = _snapshot_summaries(pid, snapshot["snapshot_id"])
    assert summary["parts"]["calibration"] == dict(units=10, rows=1350, gaps=0)
    view = app(pid)
    assert not view.exception
    assert [row.value for row in view.subheader][:4] == [
        "Training Data",
        "Validation Data",
        "Calibration Data",
        "Testing Data",
    ]
    assert [str(row.value) for row in view.metric] == [
        "40",
        "1",
        "5400",
        "8",
        "0",
        "1080",
        "10",
        "0",
        "1350",
        "10",
        "0",
        "1350",
    ]
    assert not any("No data yet" in row.value for row in view.markdown)
    assert project_store().get(pid) == project
    assert seal(snapshot["directory"]) == saved


def test_import_does_not_add_model_notices_or_change_saved_configuration(release):
    snapshot = original_import(release)
    store = project_store()
    pid = release.project["project_id"]
    rid = "long-import-display-test"
    directory = store.run_path(pid, rid)
    directory.mkdir(parents=True)
    atomic_write_json(
        directory / "manifest.json",
        dict(
            project_id=pid,
            run_id=rid,
            snapshot_id=snapshot["snapshot_id"],
            protocol="stable_observed_trend_v5",
            task="signal_forecast",
            engine_id="lstm",
            status="completed",
            config=dict(width=0.45),
        ),
    )
    store.update(pid, selected_run_id=rid)
    view = app(pid)
    assert not view.exception
    assert not view.info
    assert store.get(pid)["selected_run_id"] == rid
    assert read_json(directory / "manifest.json")["config"]["width"] == 0.45


def test_calibration_view_opens_actual_calibration_rows_only(release, monkeypatch):
    from pdm.probabilistic import data

    original_import(release)
    pid = release.project["project_id"]
    opened = []
    loader = data.load_split

    def traced(snapshot, role):
        opened.append(role)
        return loader(snapshot, role)

    monkeypatch.setattr(data, "load_split", traced)
    view = app(pid)
    next(row for row in view.button if row.key == "import_view:calibration").click().run()
    assert not view.exception
    assert view.session_state["project_step"] == "Data Quality"
    assert view.session_state["quality_tab"] == "Calibration Data"
    assert opened == ["calibration"]
    assert len(view.get("plotly_chart")) == 1


def test_damaged_saved_snapshot_shows_error_instead_of_empty_cards(release):
    snapshot = original_import(release)
    path = Path(snapshot["directory"]) / "snapshot.json"
    path.chmod(0o600)
    path.write_text("{}")
    view = app(release.project["project_id"])
    assert not view.exception
    assert any("Saved data could not be read" in row.value for row in view.error)
    assert not any("No data yet" in row.value for row in view.markdown)
