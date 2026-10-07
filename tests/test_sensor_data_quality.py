"""Sensor Data Quality renders observed data and persists only the limits sidecar."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from pdm.paths import project_root
from pdm.probabilistic import data
from pdm.probabilistic.contract import default_config
from pdm.projects import project_store


@pytest.fixture
def sensor_quality_project(tmp_path, monkeypatch):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("PDM_WORKER_ROOT", str(tmp_path / "worker"))
    source = tmp_path / "source"
    cfg = default_config()
    manifest = dict(version="quality-fixture", seed=13, schema=cfg["schema"],
                    target=cfg["target"], physical_unit="g", cadence_s=60,
                    splits={"01_sanity": dict.fromkeys(("train", "validation", "calibration", "test"), 2)},
                    file_sha256={})
    for role_index, role in enumerate(manifest["splits"]["01_sanity"]):
        frames = []
        for index in range(2):
            times = np.arange(135) * 60.
            if role == "train" and index == 0:
                times[75:] += 120.
            frames.append(pd.DataFrame(dict(unit_id=f"{role}-{index}", timestamp_s=times,
                          vibration_rms_g=.2 + role_index * .03 + index * .005 + np.arange(135) * (.003 + index * .0001))))
        path = source / "01_sanity" / "sensor_csv" / role / "measurements.csv"
        path.parent.mkdir(parents=True)
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        manifest["file_sha256"][str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (source / "dataset_manifest.json").write_text(json.dumps(manifest))
    store = project_store()
    project = store.create("Observed quality fixture", "generic_sensor_csv")
    snapshot = data.prepare_dataset(source, "01_sanity", store.project_path(project["project_id"]) / "snapshots" / "observed-storage")
    store.update(project["project_id"], state="ready", active_snapshot_id=snapshot["snapshot_id"],
                 source_manifest={"snapshots": {snapshot["snapshot_id"]: snapshot["directory"]}})
    return project["project_id"], snapshot


def quality_app(pid):
    app = AppTest.from_file(str(project_root() / "src/pdm/app.py"), default_timeout=20)
    app.session_state["project_id"] = pid
    app.session_state["project_step"] = "Data Quality"
    return app.run()


def chart_limits(app):
    figure = json.loads(app.get("plotly_chart")[0].proto.spec)
    return {shape["y0"] for shape in figure["layout"]["shapes"]
            if shape.get("y0") is not None and shape.get("y0") == shape.get("y1")}


def test_sensor_quality_has_observed_chart_limits_and_lazy_role_tabs(sensor_quality_project, monkeypatch):
    pid, _ = sensor_quality_project
    opened = []
    original = data.load_split

    def traced(snapshot, part):
        opened.append(part)
        return original(snapshot, part)

    monkeypatch.setattr(data, "load_split", traced)
    app = quality_app(pid)
    assert not app.exception and not app.error
    assert [tab.label for tab in app.tabs] == ["Training Data", "Validation Data", "Calibration Data", "Testing Data"]
    assert {"Yellow (g)", "Red (g)"} <= {row.label for row in app.number_input}
    assert len(app.get("plotly_chart")) == 1 and app.table
    assert chart_limits(app) == {.55, .8}
    assert opened and set(opened) == {"train"}
    app.session_state["quality_tab"] = "Testing Data"
    app.run()
    assert not app.exception and not app.error
    assert chart_limits(app) == {.55, .8}
    assert set(opened) == {"train", "test"}
    assert app.button(key="project_nav:Training").disabled is False


def test_sensor_limit_preview_cancel_save_and_reload_preserve_snapshot(sensor_quality_project):
    from pathlib import Path

    from pdm.long_forecast_data import source_for

    pid, snapshot = sensor_quality_project
    directory = Path(snapshot["directory"])
    before = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    app = quality_app(pid)
    next(n for n in app.number_input if n.label == "Yellow (g)").set_value(.6).run()
    assert not app.exception and chart_limits(app) == {.6, .8}
    assert not (directory / "zone_limits.json").exists()
    next(b for b in app.button if b.label == "Cancel").click().run()
    assert chart_limits(app) == {.55, .8}
    next(n for n in app.number_input if n.label == "Yellow (g)").set_value(.9).run()
    assert app.error and next(b for b in app.button if b.label == "Save").disabled
    next(n for n in app.number_input if n.label == "Yellow (g)").set_value(.6).run()
    next(b for b in app.button if b.label == "Save").click().run()
    assert not app.exception and not app.error
    reopened = quality_app(pid)
    assert chart_limits(reopened) == {.6, .8}
    assert next(n for n in reopened.number_input if n.label == "Yellow (g)").value == .6
    assert source_for(pid)["yellow"] == .6
    after = {str(p.relative_to(directory)): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
    assert {name: after[name] for name in before} == before
    assert set(after) - set(before) == {"zone_limits.json"}
    assert data.load_snapshot(directory) == snapshot
