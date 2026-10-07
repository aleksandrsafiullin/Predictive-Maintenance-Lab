"""Immutable project runs, bounded background training and full-grid replay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np

from pdm import stable_forecast as stable
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.long_forecast import ENGINES, PROTOCOL, config, fit, metrics, predict, save
from pdm.long_forecast_data import at_origin, capacity, digest, read_part, source_for, windows
from pdm.paths import project_root
from pdm.projects import project_store

DIRECT_PROTOCOLS = {PROTOCOL, *stable.PROTOCOLS}


def list_runs(project_id):
    rows = []
    for path in project_store().project_path(project_id).joinpath("runs").glob("*/manifest.json"):
        record = read_json(path)
        if record.get("project_id") == project_id and record.get("status") == "completed":
            if record.get("protocol") in DIRECT_PROTOCOLS or (
                record.get("task") == "probabilistic_signal_forecast"
                and record.get("config", {}).get("forecast_mode") == "bounded_trend_v1"
            ):
                rows.append(record)
    return sorted(rows, key=lambda row: row.get("created_at", ""), reverse=True)


def load_run(project_id, run_id):
    directory = project_store().run_path(project_id, run_id)
    manifest = read_json(directory / "manifest.json")
    if (
        manifest.get("project_id") != project_id
        or manifest.get("run_id") != run_id
        or manifest.get("status") != "completed"
    ):
        raise ValueError("Invalid saved run identity")
    if manifest.get("protocol") in DIRECT_PROTOCOLS:
        for relative, checksum in manifest["files"].items():
            path = directory / relative
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve())
                or sha256_file(path) != checksum
            ):
                raise ValueError("Saved model checksum mismatch")
        model = joblib.load(directory / "model/model.joblib")
        if model["config"] != manifest["config"] or manifest["model_hash"] != sha256_file(
            directory / "model/model.joblib"
        ):
            raise ValueError("Frozen configuration mismatch")
        if model["protocol"] != manifest["protocol"] or (
            "effective_engine" in manifest and model["engine"] != manifest["effective_engine"]
        ):
            raise ValueError("Frozen architecture mismatch")
    else:
        from pdm.probabilistic.models import load_model

        model = load_model(directory / "model")
    return manifest, model


def replay(project_id, run_id, unit_id, origin, part="test", horizon=None, frozen=None, *, use_calibration=True):
    source = source_for(project_id)
    manifest, model = frozen if frozen is not None else load_run(project_id, run_id)
    if manifest.get("project_id") != project_id or manifest.get("run_id") != run_id:
        raise ValueError("Frozen run identity mismatch")
    if manifest["snapshot_id"] != source["snapshot_id"]:
        raise ValueError("This run belongs to a previous data snapshot")
    frame = read_part(source, part)
    if manifest.get("protocol") in stable.PROTOCOLS:
        x, n = stable.full_at_origin(frame, unit_id, origin, source["cadence_s"])
        outputs = stable.predict(model, [x], [n], horizon)[0]
        reference = None
    elif manifest.get("protocol") == PROTOCOL:
        x, n = at_origin(frame, unit_id, origin, source["cadence_s"])
        outputs = predict(model, x[None, :], np.array([n]), horizon)[0]
        reference = predict(
            dict(engine="robust_trend", config=model["config"]), x[None, :], np.array([n]), horizon
        )[0]
    else:
        x, n = at_origin(frame, unit_id, origin, source["cadence_s"])
        from pdm.probabilistic.models import predict as legacy_predict

        outputs = legacy_predict(model, x[-model["config"]["history_length"] :])[0]
        outputs = outputs[:horizon] if horizon else outputs
        reference = None
    calibration = None
    if use_calibration and manifest.get("protocol") in stable.PROTOCOLS:
        from pdm.corridor_calibration import apply_width, load_active

        calibration = load_active(project_id, run_id, manifest)
        if calibration is not None:
            outputs = apply_width(outputs, calibration)
    return dict(
        outputs=outputs,
        times=origin + source["cadence_s"] * np.arange(1, len(outputs) + 1),
        origin_s=origin,
        unit_id=unit_id,
        history_observations=n,
        source=source,
        input_hash=digest(x.tolist()),
        uncorrected_trend=reference,
        coverage_guarantee=False,
        calibration=calibration,
    )


def train(project_id, engine="gru", seed=21, overrides=None, report=None, stop=None, select=True):
    source = source_for(project_id)
    train_frame = read_part(source, "train")
    val_frame = read_part(source, "validation")
    support = None
    if engine in stable.ENGINES:
        support = stable.supported_horizon(train_frame, val_frame, source["cadence_s"])
        horizon = support["horizon"]
        cfg = stable.config(horizon, source["cadence_s"], seed=seed, **(overrides or {}))
        training = stable.full_windows(train_frame, horizon, source["cadence_s"], cfg["stride"])
        validation = stable.full_windows(val_frame, horizon, source["cadence_s"], cfg["stride"])
        model = stable.fit(training, validation, cfg, engine, report, stop)
        cfg = model["config"]
        validation_metrics = stable.metrics(
            model.pop("_validation_outputs"), validation, source["red"]
        )
    else:
        horizon = capacity(train_frame, source["cadence_s"])
        cfg = config(horizon, source["cadence_s"], seed=seed, **(overrides or {}))
        training = windows(train_frame, horizon, source["cadence_s"], cfg["stride"])
        validation = windows(val_frame, horizon, source["cadence_s"], cfg["stride"])
        model = fit(training, validation, cfg, engine, report, stop)
        validation_metrics = metrics(
            predict(model, validation["x"], validation["lengths"]), validation, source["red"]
        )
    if stop and stop():
        raise InterruptedError("Training cancelled before publication")
    run_id = "long-" + uuid.uuid4().hex
    directory = project_store().run_path(project_id, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    save(model, directory / "model")
    atomic_write_json(directory / "validation.json", validation_metrics)
    atomic_write_json(directory / "training.json", model["training"])
    atomic_write_json(
        directory / "origins.json",
        {
            part: dict(
                units=data["units"].tolist(),
                physical=data["physical"].tolist(),
                origins=data["origins"].tolist(),
            )
            for part, data in (("train", training), ("validation", validation))
        },
    )
    source_identity = {
        key: source.get(key)
        for key in (
            "project_id",
            "snapshot_id",
            "dataset_hash",
            "cadence_s",
            "unit",
            "red",
            "yellow",
            "direction",
        )
    }
    files = {
        str(path.relative_to(directory)): sha256_file(path)
        for path in directory.rglob("*")
        if path.is_file()
    }
    manifest = dict(
        run_id=run_id,
        project_id=project_id,
        snapshot_id=source["snapshot_id"],
        dataset_hash=source["dataset_hash"],
        task="signal_forecast",
        engine_id=engine,
        protocol=cfg["protocol"],
        status="completed",
        config=cfg,
        params=dict(
            forecast_mode=cfg["protocol"],
            history_length="all" if support else 240,
            hidden_size=cfg["hidden_size"],
            seed=seed,
            horizons_s=(np.arange(1, horizon + 1) * source["cadence_s"]).tolist(),
        ),
        model_hash=sha256_file(directory / "model/model.joblib"),
        files=files,
        source=source_identity,
        created_at=datetime.now(timezone.utc).isoformat(),
        validation=validation_metrics,
        selection_objective=model["training"]["validation_objective"],
        quality_accepted=False,
        effective_engine=model["engine"],
        learned_correction_selected=model["training"]["learned_correction_selected"],
        horizon_support=support,
        evaluation_status="exploratory_validation;separate_test_required",
    )
    atomic_write_json(directory / "manifest.json", manifest)
    if select and project_store().get(project_id)["active_snapshot_id"] == source["snapshot_id"]:
        project_store().update(project_id, selected_run_id=run_id)
    return manifest


def train_suite(project_id, seed=21, overrides=None, report=None, stop=None):
    report = report or (lambda _: None)
    snapshot_id = project_store().get(project_id)["active_snapshot_id"]
    runs = []
    for index, engine in enumerate(stable.ENGINES):
        if stop and stop():
            raise InterruptedError("Training cancelled")
        if project_store().get(project_id)["active_snapshot_id"] != snapshot_id:
            raise RuntimeError("Data snapshot changed during model comparison")

        def progress(record, engine=engine, index=index):
            report(dict(engine=engine, model_index=index + 1, model_total=3, **record))

        progress(dict(stage="starting", message=f"Training {stable.LABELS[engine]}"))
        run = train(project_id, engine, seed, overrides, progress, stop, select=False)
        if run["snapshot_id"] != snapshot_id:
            raise RuntimeError("Data snapshot changed during model comparison")
        runs.append(run)
        progress(dict(stage="saved", run_id=run["run_id"]))

    def score(run):
        val = run["validation"]
        return (
            -min(q["point_coverage"] for q in val["quarters"] if q["status"] == "measured"),
            run["selection_objective"],
        )

    best = min(runs, key=score)
    if project_store().get(project_id)["active_snapshot_id"] == best["snapshot_id"]:
        project_store().update(project_id, selected_run_id=best["run_id"])
    return dict(
        run_id=best["run_id"],
        runs=[r["run_id"] for r in runs],
        selected_engine=best["engine_id"],
        test_used=False,
    )


def status_path(project_id):
    return project_store().project_path(project_id) / "long_forecast_status.json"


def read_status(project_id):
    try:
        return read_json(status_path(project_id))
    except FileNotFoundError:
        return {}


def alive(status):
    if status.get("status") not in {"queued", "running", "stopping"}:
        return False
    try:
        os.kill(status["pid"], 0)
        return True
    except (ProcessLookupError, KeyError):
        return False


def active_locked(store):
    """Read status while holding the registry lock, without re-entering it."""
    for pid in store._load()["projects"]:
        path = store._owned_path(pid, "long_forecast_status.json")
        if path.exists() and alive(read_json(path)):
            return True
    return False


def launch(project_id, engine, seed, overrides=None):
    if engine not in (*ENGINES, *stable.ENGINES, "all"):
        raise ValueError("Unknown long forecast engine")
    if engine in (*stable.ENGINES, "all"):
        stable.config(1, seed=seed, **(overrides or {}))
    store = project_store()
    with store.launch_lock():
        from pdm.worker import heavy_job_active

        store._entry(store._load(), project_id)
        if heavy_job_active() or active_locked(store):
            raise RuntimeError("Wait for the current training job to finish")
        path = store._owned_path(project_id, "long_forecast_status.json")
        log = path.with_suffix(".log")
        stop_file = path.with_suffix(".stop")
        stop_file.unlink(missing_ok=True)
        job_id = uuid.uuid4().hex
        atomic_write_json(
            path,
            dict(status="queued", job_id=job_id, project_id=project_id, engine=engine, seed=seed),
        )
        env = {
            **os.environ,
            "PYTHONPATH": str(project_root() / "src")
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        with log.open("wb") as output:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "pdm.long_forecast_run",
                    "--project",
                    project_id,
                    "--engine",
                    engine,
                    "--seed",
                    str(seed),
                    "--job",
                    job_id,
                    "--overrides",
                    json.dumps(overrides or {}),
                ],
                cwd=project_root(),
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        # Child uses the same launch lock before publishing its first status.
        atomic_write_json(
            path,
            dict(
                status="queued",
                pid=process.pid,
                job_id=job_id,
                project_id=project_id,
                engine=engine,
                seed=seed,
            ),
        )
    return process.pid


def request_stop(project_id):
    status_path(project_id).with_suffix(".stop").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--engine", choices=(*ENGINES, *stable.ENGINES, "all"), default="all")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--job", default=None)
    parser.add_argument("--overrides", type=json.loads, default={})
    args = parser.parse_args()
    path = status_path(args.project)
    status = dict(
        status="running",
        pid=os.getpid(),
        job_id=args.job,
        project_id=args.project,
        engine=args.engine,
        seed=args.seed,
    )
    with project_store().launch_lock():
        atomic_write_json(path, status)

    def report(record):
        status.update(progress=record)
        atomic_write_json(path, status)
        print(json.dumps(record), flush=True)

    def stop():
        return path.with_suffix(".stop").exists()

    try:
        result = (
            train_suite(args.project, args.seed, overrides=args.overrides, report=report, stop=stop)
            if args.engine == "all"
            else train(
                args.project,
                args.engine,
                args.seed,
                overrides=args.overrides,
                report=report,
                stop=stop,
            )
        )
        status.update(
            status="completed", run_id=result["run_id"], runs=result.get("runs", [result["run_id"]])
        )
    except InterruptedError:
        status.update(status="stopped")
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        atomic_write_json(path, status)


if __name__ == "__main__":
    main()
