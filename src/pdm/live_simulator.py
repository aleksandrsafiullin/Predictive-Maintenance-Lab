"""Demo feed: replay recorded machines into the live folder as if sensors were sending data.

Every tick appends the next measurement of each selected machine to ``sim_<unit>.csv``
in the watched folder, using the same CSV format the Live monitor reads. The values are
real recorded measurements; only the clock is accelerated. Progress is written to
``simulator.json`` beside the project's live config so the UI can show and stop it.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from pdm.io_util import atomic_write_json, read_json
from pdm.live_monitor import live_dir
from pdm.paths import project_root

SIM_PREFIX = "sim_"


def _status_path(project_id: str) -> Path:
    return live_dir(project_id) / "simulator.json"


def _stop_path(project_id: str) -> Path:
    return live_dir(project_id) / "simulator.stop"


def demo_units(project_id: str) -> dict[str, list[str]]:
    """Units of the active snapshot by split, for choosing what to replay."""
    from pdm.data.project_prepare import load_snapshot

    split = load_snapshot(project_id)["split"]
    return {name: [str(u) for u in split.get(name) or []] for name in ("test", "validation", "train")}


def clear_demo_files(folder: str | Path) -> int:
    """Remove only files this simulator wrote; anything else in the folder is left alone."""
    removed = 0
    for path in Path(folder).glob(f"{SIM_PREFIX}*.csv"):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def run_simulator(project_id: str, folder: str | Path, units: Sequence[str], *, tick_s: float = 1.0,
                  rows_per_tick: int = 1, signal_column: str | None = None, reset: bool = True) -> dict[str, Any]:
    from pdm.data.project_prepare import load_snapshot

    data = load_snapshot(project_id)
    column = signal_column or str(data["schema"].get("signal_column") or "signal")
    features = data["features"]
    series = {u: features[features.unit_id.astype(str) == str(u)].sort_values("timestamp_s")[["timestamp_s", "signal"]]
              for u in units}
    series = {u: s.reset_index(drop=True) for u, s in series.items() if len(s)}
    if not series:
        raise ValueError("None of the chosen units exist in the active snapshot")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if reset:
        clear_demo_files(folder)
    _stop_path(project_id).unlink(missing_ok=True)
    written = dict.fromkeys(series, 0)
    status = {"state": "running", "pid": os.getpid(), "folder": str(folder), "tick_s": tick_s,
              "rows_per_tick": rows_per_tick, "units": {u: {"written": 0, "total": len(s)} for u, s in series.items()},
              "started_at": time.time()}
    atomic_write_json(_status_path(project_id), status)
    try:
        while any(written[u] < len(s) for u, s in series.items()):
            if _stop_path(project_id).exists():
                status["state"] = "stopped"
                break
            for u, s in series.items():
                start, end = written[u], min(len(s), written[u] + rows_per_tick)
                if start >= end:
                    continue
                chunk = s.iloc[start:end]
                path = folder / f"{SIM_PREFIX}{u}.csv"
                pd.DataFrame({"unit_id": u, "timestamp_s": chunk.timestamp_s.to_numpy(),
                              column: chunk.signal.to_numpy()}).to_csv(
                    path, mode="a", header=not path.exists(), index=False)
                written[u] = end
                status["units"][u]["written"] = end
            atomic_write_json(_status_path(project_id), status)
            time.sleep(tick_s)
        else:
            status["state"] = "finished"
    finally:
        if status["state"] == "running":
            status["state"] = "stopped"
        status["finished_at"] = time.time()
        atomic_write_json(_status_path(project_id), status)
        _stop_path(project_id).unlink(missing_ok=True)
    return status


# ------------------------------------------------------------- process control

def simulator_status(project_id: str) -> dict[str, Any]:
    path = _status_path(project_id)
    if not path.exists():
        return {"state": "idle"}
    try:
        status = read_json(path)
    except (OSError, ValueError):
        return {"state": "idle"}
    if status.get("state") == "running" and not _alive(status.get("pid")):
        status["state"] = "stopped"
    return status


def _alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError, PermissionError):
        return False
    return True


def start_simulator_process(project_id: str, folder: str | Path, units: Sequence[str], *,
                            tick_s: float, rows_per_tick: int) -> int:
    if simulator_status(project_id).get("state") == "running":
        raise RuntimeError("The demo feed is already running")
    root = project_root()
    env = {**os.environ, "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "pdm", "live-simulate", "--project", project_id, "--folder", str(folder),
           "--units", ",".join(units), "--tick-s", str(tick_s), "--rows-per-tick", str(rows_per_tick)]
    log = (live_dir(project_id) / "simulator.log").open("w")
    proc = subprocess.Popen(cmd, cwd=str(root), env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    atomic_write_json(_status_path(project_id), {"state": "running", "pid": proc.pid, "folder": str(folder),
                                                 "units": {u: {"written": 0, "total": None} for u in units},
                                                 "started_at": time.time()})
    return proc.pid


def stop_simulator(project_id: str) -> None:
    _stop_path(project_id).touch()
    status = simulator_status(project_id)
    pid = status.get("pid")
    if status.get("state") == "running" and _alive(pid):
        for _ in range(20):
            if not _alive(pid):
                return
            time.sleep(0.1)
        os.kill(int(pid), signal.SIGTERM)
