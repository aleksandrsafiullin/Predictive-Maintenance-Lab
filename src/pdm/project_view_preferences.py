"""Project-scoped display selections, independent of data and saved models."""

from __future__ import annotations

from pdm.io_util import atomic_write_json, read_json
from pdm.projects import _file_lock, project_store

PREFERENCES_FILE = "view_preferences.json"


def load_preferences(project_id, snapshot_id, view):
    path = project_store().project_path(project_id) / PREFERENCES_FILE
    if path.is_symlink():
        raise ValueError("Display preferences cannot be a symlink")
    if not path.exists():
        return {}
    try:
        record = read_json(path)
    except ValueError:
        return {}
    if not isinstance(record, dict) or record.get("snapshot_id") != snapshot_id:
        return {}
    values = record.get(view)
    return values if isinstance(values, dict) else {}


def save_preferences(project_id, snapshot_id, view, changes):
    store = project_store()
    directory = store.project_path(project_id)
    path = directory / PREFERENCES_FILE
    with _file_lock(directory / ".view_preferences.lock"):
        if store.get(project_id).get("active_snapshot_id") != snapshot_id:
            return  # A callback from an older snapshot cannot replace current preferences.
        if path.is_symlink():
            raise ValueError("Display preferences cannot be a symlink")
        try:
            record = read_json(path) if path.exists() else {}
        except ValueError:
            record = {}
        if not isinstance(record, dict) or record.get("snapshot_id") != snapshot_id:
            record = {"snapshot_id": snapshot_id}
        previous = record.get(view)
        updated = {**(previous if isinstance(previous, dict) else {}), **changes}
        if updated != previous:
            atomic_write_json(path, {**record, view: updated})
