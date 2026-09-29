"""Small saved-snapshot fixture shared by data, model, and UI tests."""

from __future__ import annotations

from pathlib import Path

from pdm.data.project_import import import_project
from pdm.data.project_prepare import prepare_project
from pdm.projects import ProjectStore

SOURCE = Path(__file__).parent / "fixtures" / "project_contract"


def make_contract_snapshot(root: Path) -> tuple[ProjectStore, dict, dict]:
    store = ProjectStore(root)
    project = store.create("Contract fixture", "generic_sensor_csv")
    manifest = import_project(
        project["project_id"],
        {
            "primary": {"mode": "folder", "path": str(SOURCE)},
            "signal_column": "vibration",
            "signal_label": "Signed vibration",
            "signal_unit": "g",
            "thresholds": {"mode": "absolute", "direction": "above", "yellow": 0.4, "red": 0.8},
            "seed": 19,
        },
        store=store,
    )
    snapshot = prepare_project(project["project_id"], manifest["manifest_id"], store=store)
    return store, project, snapshot
