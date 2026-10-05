#!/usr/bin/env python3
"""Run the four numeric engines sequentially through real application workers."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pdm.cli import spawn_worker
from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.projects import project_store
from pdm.signal_profiles import funnel_training_profile, learned_bearings_training_profile
from pdm.signal_training import ENGINES, _params, load_signal_run
from pdm.worker import read_status


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def source_hashes():
    paths = [*sorted((ROOT / "src/pdm").rglob("*.py")), Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def verify_run(saved: dict, config: dict) -> None:
    snapshot = load_snapshot(config["project_id"], config["snapshot_id"])
    training = snapshot["features"][snapshot["features"].unit_id.astype(str).isin(map(str, snapshot["split"]["train"]))]
    for key in ("project_id", "snapshot_id", "engine_id", "snapshot_fingerprint_sha256"):
        if saved.get(key) != config[key]:
            raise ValueError(f"Saved run {key} differs from frozen study")
    if saved["params"] != _params(config["engine_id"], config["params"], training):
        raise ValueError("Saved run parameters differ from frozen study")


def selected_datasets(projects: dict, datasets: list[str] | None) -> list[str]:
    """Validate exact manifest keys before any snapshot is decoded."""
    if datasets is None:
        return list(projects)
    if not datasets or any(not isinstance(key, str) or not key.strip() for key in datasets):
        raise ValueError("Dataset selector must contain nonempty manifest keys")
    if len(set(datasets)) != len(datasets):
        raise ValueError("Duplicate dataset selectors")
    unknown = [key for key in datasets if key not in projects]
    if unknown:
        raise ValueError(f"Unknown dataset selectors: {unknown}")
    return [key for key in projects if key in datasets]


def training_profile(snapshot: dict, engine: str) -> dict:
    """Match the normal Training UI route, preserving the entire resolved profile."""
    learned = learned_bearings_training_profile(snapshot, engine)
    return learned or funnel_training_profile(snapshot, engine)


def profile_policy(params: dict) -> dict:
    """Describe each resolved route without implying scientific acceptance."""
    learned = params["forecast_mode"] == "learned_joint_trajectories"
    return {
        "forecast_mode": params["forecast_mode"],
        "horizons_s": params["horizons_s"],
        "horizon_policy": (
            "dense cadence-aligned learned trajectory grid resolved by the profile"
            if learned else "Train-supported cadence-aligned sparse grid resolved by the profile"),
        "training_samples": params.get("training_samples"),
        "path_samples": params["path_samples"],
        "max_windows_per_unit": params.get("max_windows_per_unit"),
        "sampling_policy": "uniform origin index cap per physical group; resolved cap recorded above",
        "batch_sampling": params.get("batch_sampling") if learned else None,
        "checkpoint_selection_policy": (
            "best Validation physical-group-equal joint objective; restore best checkpoint; "
            "early stopping after patience epochs without min_delta improvement"
            if learned else "predeclared fixed final epoch/iterations/ridge; no Validation model selection"),
        "patience": params.get("patience") if learned else None,
        "min_delta": params.get("min_delta") if learned else None,
        "uncertainty_policy": (
            "learned uncalibrated simultaneous band; no residual OOF calibration"
            if learned else "grouped Train OOF residual paths; Validation-only interval calibration"),
        "cv_folds": None if learned else params.get("cv_folds"),
        "test_policy": "frozen evaluation only; no Test model selection",
    }


def freeze(directory: Path, resume: bool, datasets: list[str] | None = None) -> dict:
    rebuilt = read_json(ROOT / "output/dataset-reset-20261002/rebuild/rebuild_manifest.json")
    selected = selected_datasets(rebuilt["projects"], datasets)
    jobs = []
    for dataset, saved in rebuilt["projects"].items():
        if dataset not in selected:
            continue
        snapshot = load_snapshot(saved["project_id"], saved["snapshot_id"])
        if project_store().get(saved["project_id"])["active_snapshot_id"] != saved["snapshot_id"]:
            raise ValueError("Balanced project snapshot has changed")
        for engine in ENGINES:
            params = training_profile(snapshot, engine)
            jobs.append({"dataset": dataset, "project_id": saved["project_id"],
                         "snapshot_id": saved["snapshot_id"], "engine_id": engine,
                         "snapshot_fingerprint_sha256": digest(snapshot["fingerprint"]),
                         "params": params, "profile_policy": profile_policy(params)})
    contract = {
        "jobs": jobs, "implementation_hashes": source_hashes(),
        "nominal_coverage_policy": "0.9 engineering default; no operational acceptance target approved",
        "dataset_selector": None if datasets is None else selected,
        "horizon_policy": "per-job resolved profile and horizon grid recorded in profile_policy",
        "selection_policy": "per-job: learned Validation checkpoint selection/early stopping; residual fixed fit and Validation calibration; no Test selection",
        "sampling_policy": "per-job resolved samples, origin cap and batch sampling recorded in profile_policy",
        "evaluation_status": "explored_reused_data; no independent holdout",
    }
    proposed = {"contract": contract, "contract_hash": digest(contract)}
    path = directory / "frozen_contract.json"
    if path.exists():
        if not resume or read_json(path) != proposed:
            raise ValueError("Study is already frozen or has changed; use a new output directory")
        return proposed
    if resume:
        raise ValueError("No frozen study to resume")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Fresh study directory must be empty")
    atomic_write_json(path, proposed)
    return proposed


def run(directory: Path, resume: bool, timeout_s: float, datasets: list[str] | None = None) -> int:
    frozen = freeze(directory, resume, datasets)
    summary_path = directory / "summary.json"
    results = read_json(summary_path).get("results", []) if summary_path.exists() else []
    status = {"status": "running", "pid": os.getpid(), "results": results,
              "contract_hash": frozen["contract_hash"], "total_jobs": len(frozen["contract"]["jobs"])}
    atomic_write_json(summary_path, status)
    for index, config in enumerate(frozen["contract"]["jobs"]):
        key = f"{config['dataset']}:{config['engine_id']}"
        existing = next((row for row in reversed(results) if row["key"] == key and row["status"] == "completed"), None)
        if existing:
            saved = load_signal_run(config["project_id"], existing["run_id"])
            verify_run(saved, config)
            continue
        if source_hashes() != frozen["contract"]["implementation_hashes"]:
            raise ValueError("Implementation changed after freeze")
        snapshot = load_snapshot(config["project_id"], config["snapshot_id"])
        if digest(snapshot["fingerprint"]) != config["snapshot_fingerprint_sha256"]:
            raise ValueError("Dataset changed after freeze")
        job_id = uuid.uuid4().hex
        job = {"kind": "project_train", "task": "signal_forecast", "job_id": job_id,
               **{field: config[field] for field in ("project_id", "snapshot_id", "engine_id", "params")}}
        atomic_write_json(directory / f"job-{index + 1:02d}.json", job)
        status.update(current_job=key, current_job_number=index + 1, worker_job_id=job_id)
        atomic_write_json(summary_path, status)
        print(f"Starting {index + 1}/{status['total_jobs']}: {key}", flush=True)
        process = spawn_worker(job)
        started = time.monotonic()
        while process.poll() is None:
            progress = read_status()
            if progress.get("job_id") != job_id:
                raise RuntimeError("Application worker job identity changed")
            status["worker_status"] = progress
            atomic_write_json(summary_path, status)
            if time.monotonic() - started > timeout_s:
                raise TimeoutError("Worker still active after study wait limit; no process terminated")
            time.sleep(2)
        progress = read_status()
        if source_hashes() != frozen["contract"]["implementation_hashes"]:
            raise ValueError("Implementation changed while worker was training")
        if progress.get("job_id") != job_id:
            raise RuntimeError("Final worker identity changed")
        row = {"key": key, "project_id": config["project_id"], "engine_id": config["engine_id"],
               "duration_s": round(time.monotonic() - started, 3), "status": progress.get("status")}
        if progress.get("status") == "completed":
            saved = load_signal_run(config["project_id"], progress["run_id"])
            verify_run(saved, config)
            row.update(run_id=saved["run_id"], params=saved["params"], metrics=saved["metrics"],
                       interval_status=saved["interval_status"])
        else:
            row["error"] = progress.get("error") or progress.get("message")
        results.append(row)
        atomic_write_json(summary_path, status)
        print(f"Finished {key}: {row['status']}", flush=True)
    latest = {row["key"]: row for row in results}
    status.update(status="completed" if all(row["status"] == "completed" for row in latest.values()) else "failed",
                  current_job=None, worker_status=read_status())
    atomic_write_json(summary_path, status)
    return 0 if status["status"] == "completed" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=21600)
    parser.add_argument("--dataset", action="append", default=None,
                        help="Exact rebuild manifest key; repeat to select multiple datasets (default: all)")
    args = parser.parse_args()
    if args.prepare_only:
        frozen = freeze(args.output, args.resume, args.dataset)
        print(json.dumps({"jobs": len(frozen["contract"]["jobs"]), "contract_hash": frozen["contract_hash"]}))
        return 0
    try:
        return run(args.output, args.resume, args.timeout_s, args.dataset)
    except BaseException as exc:
        path = args.output / "summary.json"
        current = read_json(path) if path.exists() else {}
        current.update(status="failed", error=str(exc))
        if args.output.exists():
            atomic_write_json(path, current)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
