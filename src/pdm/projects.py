"""Persistent identities and app-owned paths for the project workflow.

Public records are plain JSON-compatible mappings.  Legacy source locations are
references only; every project's new snapshots and runs live under this store.
"""

from __future__ import annotations

import contextlib
import copy
import os
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from pdm.io_util import atomic_write_json, read_json
from pdm.paths import data_processed, data_raw, project_root, runs_root

SCHEMA_VERSION = 1
SOURCE_KINDS = frozenset({"xjtu_bearings", "hse_filters", "generic_sensor_csv"})
STORAGE_MODES = frozenset({"owned", "linked_legacy"})
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}\Z")
_EDITABLE = frozenset({"name", "state", "source_manifest", "active_snapshot_id", "selected_run_id"})
_ACTIVE_JOB_STATES = frozenset({"queued", "starting", "preparing", "training", "running", "stopping"})
_LEGACY = {
    "bearings": ("legacy-bearings", "XJTU-SY Bearings", "xjtu_bearings"),
    "filters": ("legacy-filters", "HSE Filters", "hse_filters"),
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_id(value: str, label: str) -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None or value in {".", ".."}:
        raise ValueError(f"Invalid {label}: {value!r}")
    return value


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Project name must not be empty")
    cleaned = value.strip()
    if len(cleaned) > 120 or any(ord(c) < 32 for c in cleaned):
        raise ValueError("Project name is too long or contains control characters")
    return cleaned


@contextlib.contextmanager
def _file_lock(path: Path) -> Iterator[None]:
    """Hold a process-level exclusive lock across registry read/modify/write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            if not handle.read(1):
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class ProjectStore:
    """Registry with one app-owned directory per active project."""

    def __init__(self, root: Path | None = None) -> None:
        configured = root if root is not None else os.environ.get("PDM_PROJECTS_ROOT")
        self.root = Path(configured) if configured is not None else project_root() / "data" / "projects"
        self.root = self.root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.root / "registry.json"
        self.lock_path = self.root / ".registry.lock"

    def _load(self) -> dict[str, Any]:
        if not self.registry_path.exists():
            return {"schema_version": SCHEMA_VERSION, "projects": {}, "legacy_tombstones": [], "archives": []}
        registry = read_json(self.registry_path)
        if not isinstance(registry, dict) or registry.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported project registry schema")
        if not isinstance(registry.get("projects"), dict):
            raise ValueError("Invalid project registry")
        registry.setdefault("legacy_tombstones", [])
        registry.setdefault("archives", [])
        return registry

    def _save(self, registry: dict[str, Any]) -> None:
        atomic_write_json(self.registry_path, registry)

    def _entry(self, registry: Mapping[str, Any], project_id: str) -> dict[str, Any]:
        _safe_id(project_id, "project ID")
        try:
            return registry["projects"][project_id]
        except KeyError as exc:
            raise KeyError(f"Unknown project ID: {project_id}") from exc

    def _owned_path(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts)
        cursor = self.root
        for part in parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Symlinks are forbidden in project-owned paths")
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("Project path escapes the project store")
        return path

    def launch_lock(self) -> contextlib.AbstractContextManager[None]:
        """Serialize worker launch with archive and registry mutations."""
        return _file_lock(self.lock_path)

    def _assert_unique_name(self, registry: Mapping[str, Any], name: str, *, except_id: str | None = None) -> None:
        folded = name.casefold()
        if any(
            pid != except_id and rec["name"].casefold() == folded
            for pid, rec in registry["projects"].items()
        ):
            raise ValueError(f"Project name already exists: {name}")

    def create(self, name: str, source_kind: str) -> dict[str, Any]:
        name = _name(name)
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"Unknown project source kind: {source_kind}")
        with _file_lock(self.lock_path):
            registry = self._load()
            self._assert_unique_name(registry, name)
            project_id = uuid.uuid4().hex
            path = self._owned_path(project_id)
            path.mkdir()
            for child in ("source", "snapshots", "runs"):
                (path / child).mkdir()
            record = {
                "project_id": project_id,
                "name": name,
                "source_kind": source_kind,
                "storage_mode": "owned",
                "state": "created",
                "active_snapshot_id": None,
                "selected_run_id": None,
                "source_manifest": None,
                "created_at": _utc_now(),
                "schema_version": SCHEMA_VERSION,
            }
            registry["projects"][project_id] = record
            try:
                self._save(registry)
            except Exception:
                shutil.rmtree(path)
                raise
            return copy.deepcopy(record)

    def list(self) -> list[dict[str, Any]]:
        with _file_lock(self.lock_path):
            records = self._load()["projects"].values()
            return copy.deepcopy(sorted(records, key=lambda rec: (rec["created_at"], rec["project_id"])))

    def get(self, project_id: str) -> dict[str, Any]:
        with _file_lock(self.lock_path):
            return copy.deepcopy(self._entry(self._load(), project_id))

    def update(self, project_id: str, **changes: Any) -> dict[str, Any]:
        unknown = changes.keys() - _EDITABLE
        if unknown:
            raise ValueError(f"Immutable or unknown project fields: {sorted(unknown)}")
        with _file_lock(self.lock_path):
            registry = self._load()
            record = self._entry(registry, project_id)
            if "name" in changes:
                changes["name"] = _name(changes["name"])
                self._assert_unique_name(registry, changes["name"], except_id=project_id)
            if "state" in changes and changes["state"] not in {"created", "imported", "ready", "error"}:
                raise ValueError("Invalid project state")
            if "source_manifest" in changes and changes["source_manifest"] is not None:
                if not isinstance(changes["source_manifest"], dict):
                    raise ValueError("source_manifest must be a mapping or null")
            for field in ("active_snapshot_id", "selected_run_id"):
                if field in changes and changes[field] is not None:
                    _safe_id(changes[field], field)
            if record["storage_mode"] == "linked_legacy" and "source_manifest" in changes:
                existing_id = record["source_manifest"]["legacy_dataset_id"]
                incoming = changes["source_manifest"]
                if not isinstance(incoming, dict) or incoming.get("legacy_dataset_id") != existing_id:
                    raise ValueError("Cannot remove a linked project's legacy dataset binding")
            updated = {**record, **changes}
            if updated["selected_run_id"] is not None:
                run_path = self._owned_path(project_id, "runs", updated["selected_run_id"])
                manifest_path = run_path / "manifest.json"
                if not manifest_path.is_file() or manifest_path.is_symlink():
                    raise ValueError("Selected run has no saved manifest")
                run = read_json(manifest_path)
                if (
                    run.get("project_id") != project_id
                    or run.get("snapshot_id") != updated["active_snapshot_id"]
                    or run.get("run_id") != updated["selected_run_id"]
                    or run.get("task") not in {"signal_forecast", "red_entry"}
                    or run.get("status") != "completed"
                ):
                    raise ValueError("Selected run is not a completed project run for this snapshot")
            registry["projects"][project_id] = updated
            self._save(registry)
            return copy.deepcopy(updated)

    def _activate_snapshot_locked(
        self, registry: dict[str, Any], project_id: str, snapshot_id: str
    ) -> dict[str, Any]:
        """Activate a published snapshot and clear the run selection.

        Caller must hold ``launch_lock()`` and pass the registry it loaded under
        that lock. This method takes no lock itself (the lock is not re-entrant).
        """
        _safe_id(snapshot_id, "active_snapshot_id")
        record = self._entry(registry, project_id)
        updated = {**record, "active_snapshot_id": snapshot_id, "selected_run_id": None, "state": "ready"}
        registry["projects"][project_id] = updated
        self._save(registry)
        return copy.deepcopy(updated)

    def project_path(self, project_id: str) -> Path:
        self.get(project_id)
        path = self._owned_path(project_id)
        if not path.is_dir():
            raise FileNotFoundError(f"Project directory is missing: {project_id}")
        return path

    def snapshot_path(self, project_id: str, snapshot_id: str) -> Path:
        _safe_id(snapshot_id, "snapshot ID")
        base = self.project_path(project_id)
        return self._owned_path(base.name, "snapshots", snapshot_id)

    def run_path(self, project_id: str, run_id: str) -> Path:
        _safe_id(run_id, "run ID")
        base = self.project_path(project_id)
        return self._owned_path(base.name, "runs", run_id)

    @staticmethod
    def _job_blocks(project_id: str, job: Mapping[str, Any] | None) -> bool:
        if not job or job.get("project_id") != project_id:
            return False
        state = str(job.get("status", job.get("state", ""))).lower()
        return state in _ACTIVE_JOB_STATES or bool(job.get("worker_alive"))

    def _active_worker_job(self) -> Mapping[str, Any] | None:
        # The worker owns the lifecycle. This read supplements the caller's
        # authoritative job record during the transition to project-scoped jobs.
        from pdm.worker import _pid_exists, read_status, worker_alive

        status = read_status()
        if status.get("project_id") is not None:
            live = worker_alive()
            # The worker removes worker.pid in finally, just before process
            # exit. A fresh terminal status still carries its PID; close that
            # teardown window without trusting a stale PID indefinitely.
            if not live and status.get("status") in {"completed", "failed", "cancelled"}:
                try:
                    pid = int(status["pid"])
                    age = time.time() - float(status["updated_at"])
                    live = pid > 0 and 0 <= age < 60 and _pid_exists(pid)
                except (KeyError, TypeError, ValueError, OverflowError):
                    pass
            return {**status, "worker_alive": live}
        return None

    def archive(
        self, project_id: str, *, active_job: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        with _file_lock(self.lock_path):
            registry = self._load()
            record = self._entry(registry, project_id)
            if self._job_blocks(project_id, active_job) or self._job_blocks(
                project_id, self._active_worker_job()
            ):
                raise RuntimeError(f"Project {project_id} has a queued or live job")
            source = self._owned_path(project_id)
            if not source.is_dir():
                raise FileNotFoundError(f"Project directory is missing: {project_id}")
            trash = self._owned_path(".trash")
            trash.mkdir(exist_ok=True)
            archive_id = f"{project_id}-{uuid.uuid4().hex}"
            destination = self._owned_path(".trash", archive_id)
            source.rename(destination)
            receipt = {
                "archive_id": archive_id,
                "project_id": project_id,
                "storage_mode": record["storage_mode"],
                "archived_at": _utc_now(),
                "archive_path": str(destination),
                "external_legacy_preserved": record["storage_mode"] == "linked_legacy",
                "project_record": copy.deepcopy(record),
            }
            del registry["projects"][project_id]
            if record["storage_mode"] == "linked_legacy":
                legacy_id = record["source_manifest"]["legacy_dataset_id"]
                registry["legacy_tombstones"].append(legacy_id)
                registry["legacy_tombstones"] = sorted(set(registry["legacy_tombstones"]))
            registry["archives"].append(receipt)
            try:
                atomic_write_json(destination / "archive_receipt.json", receipt)
                self._save(registry)
            except Exception:
                destination.rename(source)
                (source / "archive_receipt.json").unlink(missing_ok=True)
                raise
            return copy.deepcopy(receipt)

    @staticmethod
    def _legacy_exists(dataset_id: str) -> bool:
        raw = data_raw() / dataset_id
        processed = data_processed() / dataset_id
        if processed.is_dir() and any(processed.glob("features.parquet")):
            return True
        if processed.is_dir() and (processed / "manifest.json").is_file():
            return True
        if not raw.is_dir():
            return False
        return next(raw.rglob("*.csv"), None) is not None or next(raw.rglob("*.zip"), None) is not None

    def register_legacy(self) -> list[dict[str, Any]]:
        """Add links only for present datasets; never copy their large files."""
        with _file_lock(self.lock_path):
            registry = self._load()
            changed = False
            for dataset_id, (project_id, base_name, source_kind) in _LEGACY.items():
                if dataset_id in registry["legacy_tombstones"] or project_id in registry["projects"]:
                    continue
                if not self._legacy_exists(dataset_id):
                    continue
                name = base_name
                suffix = 0
                while any(rec["name"].casefold() == name.casefold() for rec in registry["projects"].values()):
                    suffix += 1
                    name = f"{base_name} (legacy {suffix})"
                self._assert_unique_name(registry, name)
                path = self._owned_path(project_id)
                path.mkdir(exist_ok=False)
                for child in ("source", "snapshots", "runs"):
                    (path / child).mkdir()
                registry["projects"][project_id] = {
                    "project_id": project_id,
                    "name": name,
                    "source_kind": source_kind,
                    "storage_mode": "linked_legacy",
                    "state": "created",
                    "active_snapshot_id": None,
                    "selected_run_id": None,
                    "source_manifest": {
                        "manifest_id": project_id,
                        "legacy_dataset_id": dataset_id,
                        "raw_path": str(data_raw() / dataset_id),
                        "processed_path": str(data_processed() / dataset_id),
                        "runs_path": str(runs_root() / dataset_id),
                    },
                    "created_at": _utc_now(),
                    "schema_version": SCHEMA_VERSION,
                }
                changed = True
            if changed:
                self._save(registry)
            return copy.deepcopy(
                [registry["projects"][pid] for pid, _, _ in _LEGACY.values() if pid in registry["projects"]]
            )


def project_store() -> ProjectStore:
    """Create a store with the current environment's configured root."""
    return ProjectStore()
