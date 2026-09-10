from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pdm.io_util import read_json
from pdm.paths import dataset_runs, runs_root


def list_runs(dataset_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    roots = [dataset_runs(dataset_id)] if dataset_id else [p for p in runs_root().iterdir() if p.is_dir() and not p.name.startswith("_")]
    for root in roots:
        if not root.exists():
            continue
        ds = dataset_id or root.name
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            status_path = run_dir / "status.json"
            cfg_path = run_dir / "config.yaml"
            row: dict[str, Any] = {
                "dataset_id": ds,
                "run_id": run_dir.name,
                "path": str(run_dir),
                "has_best": (run_dir / "best.pt").exists(),
                "has_last": (run_dir / "last.pt").exists(),
            }
            if status_path.exists():
                try:
                    row.update(read_json(status_path))
                except Exception:
                    row["status"] = "unknown"
            else:
                row["status"] = "unknown"
            row.setdefault("run_id", run_dir.name)
            rows.append(row)
    rows.sort(key=lambda r: r.get("updated_at") or r.get("run_id") or "", reverse=True)
    return rows


def run_dir(dataset_id: str, run_id: str) -> Path:
    return dataset_runs(dataset_id) / run_id


def load_run_status(dataset_id: str, run_id: str) -> dict[str, Any]:
    p = run_dir(dataset_id, run_id) / "status.json"
    return read_json(p) if p.exists() else {}
