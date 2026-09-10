from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_raw() -> Path:
    return project_root() / "data" / "raw"


def data_processed() -> Path:
    return project_root() / "data" / "processed"


def runs_root() -> Path:
    return project_root() / "runs"


def configs_root() -> Path:
    return project_root() / "configs"


def reports_root() -> Path:
    return project_root() / "reports"


def worker_dir() -> Path:
    d = runs_root() / "_worker"
    d.mkdir(parents=True, exist_ok=True)
    return d


def dataset_raw(dataset_id: str) -> Path:
    return data_raw() / dataset_id


def dataset_processed(dataset_id: str) -> Path:
    return data_processed() / dataset_id


def dataset_runs(dataset_id: str) -> Path:
    return runs_root() / dataset_id
