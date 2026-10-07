"""The shared form preserves existing artifacts and submits real whole-unit imports."""

from pathlib import Path

import pytest

from pdm.projects import project_store
from tests.test_sensor_import_view import app, original_import, seal
from tests.test_sensor_import_view import release as sensor_release


@pytest.fixture
def release(tmp_path, monkeypatch):
    return sensor_release.__wrapped__(tmp_path, monkeypatch)


def full_form(view):
    assert not view.exception and not view.error
    assert {"Signal column", "Signal name", "Signal unit"} <= {row.label for row in view.text_input}
    assert {"Operating age and context (optional)", "Maintenance events (optional)"} <= {
        row.label for row in view.expander
    }
    assert view.get("popover")
    assert any(row.label == "Training folder" for row in view.file_uploader)
    assert any(row.label == "Import and check data" for row in view.button)


@pytest.mark.parametrize(
    "kind",
    [
        "generic_sensor_csv",
        "xjtu_bearings",
        "hse_filters",
        "synthetic_sanity",
        "synthetic_benchmark",
    ],
)
def test_source_formats_use_the_same_full_import_form(release, monkeypatch, kind):
    from pdm import project_ui

    project = project_store().create("Shared form " + kind, kind)
    seen = []
    presenter = project_ui._render_import

    def traced(*args):
        seen.append(args[1]["project_id"])
        return presenter(*args)

    monkeypatch.setattr(project_ui, "_render_import", traced)
    view = app(project["project_id"])
    full_form(view)
    assert seen == [project["project_id"]]


def test_saved_four_roles_keep_full_form_after_rename_and_do_not_write(release):
    snapshot = original_import(release)
    store = project_store()
    pid = release.project["project_id"]
    store.update(pid, name="Renamed Benchmark")
    before, record = seal(snapshot["directory"]), store.get(pid)
    view = app(pid)
    full_form(view)
    assert [row.value for row in view.subheader][:4] == [
        "Training Data",
        "Validation Data",
        "Calibration Data",
        "Testing Data",
    ]
    assert {"Validation data from", "Calibration data from", "Testing data from"} <= {
        row.label for row in view.selectbox
    }
    assert (
        next(row for row in view.text_input if row.label == "Signal column").value
        == "vibration_rms_g"
    )
    assert store.get(pid) == record and seal(snapshot["directory"]) == before


def primary_folder(view, release):
    next(row for row in view.radio if row.label == "Training source").set_value(
        "Server folder path"
    ).run()
    next(row for row in view.text_input if row.label == "Training server folder path").set_value(
        str(release.root / "02_benchmark" / "sensor_csv" / "train")
    ).run()


def submitted_plan(release, monkeypatch, *, invalid=False):
    import pdm.project_ui as ui

    previous = original_import(release)
    launched = []
    monkeypatch.setattr(ui, "spawn_worker", launched.append)
    monkeypatch.setattr(ui, "worker_alive", lambda: False)
    monkeypatch.setattr(ui, "status_for_project", lambda _: {"status": "not_ready"})
    view = app(release.project["project_id"])
    for label in ("Validation data from", "Calibration data from", "Testing data from"):
        next(row for row in view.selectbox if row.label == label).set_value(
            "Split from training"
        ).run()
    primary_folder(view, release)
    for part, weight in dict(
        train=45 if invalid else 40, validation=20, calibration=20, test=20
    ).items():
        view.number_input(key=f"import_weight_{part}").set_value(weight)
    view.number_input(key="import_seed").set_value(43)
    view.run()
    next(row for row in view.button if row.label == "Import and check data").click().run()
    return previous, view, launched


def test_form_submission_reaches_worker_and_allocates_real_calibration_units(release, monkeypatch):
    from pdm import worker
    from pdm.paths import worker_dir
    from pdm.probabilistic import workflow

    previous, view, launched = submitted_plan(release, monkeypatch)
    before = seal(previous["directory"])
    assert not view.exception and len(launched) == 1
    job = launched[0]
    assert job["kind"] == "project_import"
    assert job["source"]["weights"] == dict(train=0.4, validation=0.2, calibration=0.2, test=0.2)
    assert job["source"]["seed"] == 43
    assert job["source"]["signal_column"] == "vibration_rms_g"
    assert all(
        job["source"][f"{part}_mode"] == "auto" for part in ("validation", "calibration", "test")
    )
    worker_dir().mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("pdm.visualization.live.clear_live_activity", lambda: None)
    worker.run_job(job)
    current = workflow.snapshot_for_project(job["project_id"])
    ids = {
        part: set(current["split_manifest"][part]["units"])
        for part in ("train", "validation", "calibration", "test")
    }
    assert {part: len(values) for part, values in ids.items()} == dict(
        train=16, validation=8, calibration=8, test=8
    )
    assert sum(map(len, ids.values())) == len(set.union(*ids.values())) == 40
    assert set.union(*ids.values()) == set(previous["split_manifest"]["train"]["units"])
    assert worker.read_status()["status"] == "completed"
    assert seal(previous["directory"]) == before


def test_invalid_split_cannot_launch_or_replace_existing_data(release, monkeypatch):
    previous, view, launched = submitted_plan(release, monkeypatch, invalid=True)
    assert not view.exception and not launched
    assert any("Split weights must add to 100%" in row.value for row in view.error)
    assert (
        project_store().get(release.project["project_id"])["active_snapshot_id"]
        == previous["snapshot_id"]
    )


def test_failed_four_role_import_preserves_previous_snapshot_and_model(release):
    from pdm.probabilistic import workflow

    previous = original_import(release)
    before = seal(previous["directory"])
    store = project_store()
    pid = release.project["project_id"]
    from pdm.io_util import atomic_write_json

    rid = "long-preserved-model"
    run_directory = store.run_path(pid, rid)
    run_directory.mkdir(parents=True)
    atomic_write_json(run_directory / "manifest.json", dict(
        project_id=pid, run_id=rid, snapshot_id=previous["snapshot_id"],
        task="signal_forecast", status="completed", engine_id="lstm", config=dict(width=0.45)))
    store.update(pid, selected_run_id=rid)
    project = store.get(pid)
    plan = dict(
        primary={"mode": "folder", "path": str(Path(release.root) / "missing")},
        validation_mode="auto",
        calibration_mode="auto",
        test_mode="auto",
        validation=None,
        calibration=None,
        test=None,
        seed=42,
        weights=dict(train=0.4, validation=0.2, calibration=0.2, test=0.2),
        manifest_path=str(release.manifest),
    )
    with pytest.raises((ValueError, FileNotFoundError)):
        workflow.import_source_plan(pid, plan)
    assert store.get(pid) == project and seal(previous["directory"]) == before
