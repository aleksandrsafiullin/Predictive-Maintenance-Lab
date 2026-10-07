"""Browser-driven demo feed, isolated from the user's watched sensor folder."""

from __future__ import annotations

import shutil
import uuid
from functools import wraps
from pathlib import Path

from pdm.io_util import atomic_write_json, read_json
from pdm.live_monitor import live_dir
from pdm.project_snapshot import project_snapshot, role_frame
from pdm.projects import _file_lock


def _locked(function):
    @wraps(function)
    def call(project_id, *args, **kwargs):
        with _file_lock(live_dir(project_id) / "demo.lock"):
            return function(project_id, *args, **kwargs)

    return call


def demo_status(project_id):
    path = live_dir(project_id) / "demo.json"
    return read_json(path) if path.exists() else dict(state="idle")


def demo_units(project_id):
    split = project_snapshot(project_id)["split"]
    return [str(uid) for role in ("test", "validation") for uid in split.get(role, [])]


def _folder(project_id, state):
    root = live_dir(project_id) / "demo"
    folder = Path(state["folder"])
    if root.is_symlink() or folder.is_symlink() or folder.parent.resolve() != root.resolve():
        raise ValueError("Demo folder escapes its project")
    return folder


def _save(project_id, state):
    atomic_write_json(live_dir(project_id) / "demo.json", state)
    return state


@_locked
def start_demo(project_id, units):
    if demo_status(project_id)["state"] == "running":
        raise ValueError("The demo feed is already running")
    snapshot = project_snapshot(project_id)
    allowed = {
        str(uid): role for role in ("test", "validation") for uid in snapshot["split"].get(role, [])
    }
    if not units or len(set(units)) != len(units) or set(units) - allowed.keys():
        raise ValueError("Choose unique Testing or Validation machines")
    folder = live_dir(project_id) / "demo" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    return _save(
        project_id,
        dict(
            state="running",
            snapshot_id=snapshot["snapshot_id"],
            folder=str(folder),
            units={
                uid: dict(role=allowed[uid], written=0, total=None, file=f"machine-{i}.csv")
                for i, uid in enumerate(units)
            },
        ),
    )


@_locked
def advance_demo(project_id, rows_per_tick=10):
    if rows_per_tick not in (1, 5, 10, 30, 60):
        raise ValueError("Unsupported demo speed")
    state = demo_status(project_id)
    if state["state"] != "running":
        return state
    snapshot = project_snapshot(project_id)
    if snapshot["snapshot_id"] != state["snapshot_id"]:
        state.update(state="stopped", reason="The data snapshot changed")
        return _save(project_id, state)
    folder = _folder(project_id, state)
    for uid, record in state["units"].items():
        frame = role_frame(snapshot, record["role"], uid).sort_values("timestamp_s")
        columns = [
            name
            for name in ("unit_id", "timestamp_s", "signal", "gap_before", "component_cycle_id")
            if name in frame
        ]
        end = min(len(frame), record["written"] + rows_per_tick)
        chunk = frame.iloc[record["written"] : end][columns]
        path = folder / record["file"]
        if len(chunk):
            chunk.to_csv(path, mode="a", header=not path.exists(), index=False)
        record.update(written=end, total=len(frame))
    if all(record["written"] == record["total"] for record in state["units"].values()):
        state["state"] = "finished"
    return _save(project_id, state)


@_locked
def stop_demo(project_id):
    state = demo_status(project_id)
    if state["state"] == "running":
        state["state"] = "stopped"
        _save(project_id, state)


@_locked
def clear_demo(project_id):
    state = demo_status(project_id)
    if state.get("folder"):
        shutil.rmtree(_folder(project_id, state), ignore_errors=False)
    return _save(project_id, dict(state="idle"))
