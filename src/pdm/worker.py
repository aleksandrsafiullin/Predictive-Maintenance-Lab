from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import traceback
from pathlib import Path

from pdm.io_util import atomic_write_json, read_json
from pdm.paths import worker_dir

_CURRENT_PROJECT_JOB: dict | None = None
_PROJECT_TERMINAL = {"completed", "failed", "cancelled"}
ACTIVE_JOB_STATES = frozenset({"queued", "starting", "running", "training", "preparing", "stopping"})


def status_path() -> Path:
    return worker_dir() / "status.json"


def stop_path() -> Path:
    return worker_dir() / "stop.flag"


def job_path() -> Path:
    return worker_dir() / "job.json"


def pid_path() -> Path:
    return worker_dir() / "worker.pid"


_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_ERROR_ACCESS_DENIED = 5


def _nt_pid_exists(pid: int) -> bool:
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        try:
            return True
        finally:
            kernel32.CloseHandle(handle)
    # Query denied still means the process exists; do not spawn a second job.
    return ctypes.get_last_error() == _ERROR_ACCESS_DENIED


def _pid_exists(pid: int) -> bool:
    """True if pid is a live process. Never sends SIGKILL or TerminateProcess."""
    if os.name == "nt":
        try:
            return _nt_pid_exists(pid)
        except (OSError, ValueError, AttributeError, TypeError, OverflowError):
            return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, AttributeError, TypeError, OverflowError):
        return False
    return True


def read_status() -> dict:
    p = status_path()
    if not p.exists():
        return {"status": "not_ready"}
    try:
        status = read_json(p)
        if status.get("project_id") and status.get("status") in {"queued", "running", "stopping"}:
            pid = status.get("pid")
            if not pid and pid_path().exists():
                try:
                    pid = int(pid_path().read_text(encoding="utf-8").strip())
                except (OSError, ValueError):
                    pid = None
            age = time.time() - float(status.get("updated_at", 0))
            if (pid and not _pid_exists(int(pid)) and age > 1) or (not pid and age > 10):
                status = {**status, "status": "failed", "stage": "worker_exit", "progress": None,
                          "message": "Worker exited without a terminal status", "error": "Worker process exited"}
                atomic_write_json(p, status)
        return status
    except Exception:
        return {"status": "unknown"}


def request_stop(expected_job_id: str | None = None) -> None:
    """Request cooperative stop, optionally bound to one project job identity."""
    status = read_status()
    if expected_job_id is None and status.get("project_id") and status.get("status") not in _PROJECT_TERMINAL:
        expected_job_id = str(status["job_id"])
    if expected_job_id is not None:
        if status.get("job_id") != expected_job_id:
            raise ValueError("Stop request does not match the active job")
        if status.get("status") in _PROJECT_TERMINAL:
            raise ValueError("Job has already finished")
        atomic_write_json(stop_path(), {"job_id": expected_job_id})
        atomic_write_json(status_path(), {**status, "status": "stopping", "stage": "stopping",
                                          "message": "Stop requested", "updated_at": time.time()})
    else:
        stop_path().write_text("stop\n", encoding="utf-8")


def clear_stop() -> None:
    if stop_path().exists():
        stop_path().unlink()


def worker_alive() -> bool:
    p = pid_path()
    if not p.exists():
        return False
    try:
        pid = int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return False
    if pid <= 0:
        return False
    try:
        return bool(_pid_exists(pid))
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, AttributeError, TypeError, OverflowError):
        return False


def heavy_job_active(project_id: str | None = None) -> bool:
    """True while any worker job is live or pending.

    Global on purpose: the single worker blocks every project, so ``project_id``
    never narrows the answer.
    """
    return worker_alive() or read_status().get("status") in ACTIVE_JOB_STATES


def write_status(payload: dict) -> None:
    payload = dict(payload)
    if _CURRENT_PROJECT_JOB is not None:
        for key in ("job_id", "project_id", "kind"):
            payload[key] = _CURRENT_PROJECT_JOB[key]
        payload.setdefault("stage", str(payload.get("status", "running")))
        payload.setdefault("progress", None)
        payload.setdefault("message", None)
        payload.setdefault("error", None)
        payload["updated_at"] = time.time()
    payload["pid"] = os.getpid()
    atomic_write_json(status_path(), payload)


def status_for_project(project_id: str) -> dict:
    status = read_status()
    return status if status.get("project_id") == project_id else {"status": "not_ready"}


def queued_project_job() -> dict | None:
    """Read the durable queue marker without relying on a possibly stale PID."""
    status = read_status()
    if status.get("project_id") and status.get("status") not in _PROJECT_TERMINAL:
        return status
    return None


def _project_stopped(job_id: str) -> bool:
    if not stop_path().exists():
        return False
    try:
        saved = read_json(stop_path())
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return saved.get("job_id") == job_id


def _cleanup_project_uploads(job: dict) -> None:
    """Remove only browser staging directories owned by this import job."""
    from pdm.projects import project_store

    uploads = project_store().project_path(job["project_id"]) / "uploads"
    if uploads.is_symlink() or not uploads.is_dir():
        return
    source = job.get("source") or {}
    for group in ("primary", "validation", "test"):
        spec = source.get(group)
        if not isinstance(spec, dict) or spec.get("mode") != "files":
            continue
        files = spec.get("files")
        if not isinstance(files, list) or not files:
            continue
        paths = [Path(item["path"]) for item in files if isinstance(item, dict) and isinstance(item.get("path"), str)]
        if len(paths) != len(files):
            continue
        stages = {path.relative_to(uploads).parts[0] for path in paths if path.is_relative_to(uploads)}
        if len(stages) != 1:
            continue
        stage_id = next(iter(stages))
        if len(stage_id) != 32 or any(c not in "0123456789abcdef" for c in stage_id):
            continue
        stage = uploads / stage_id
        if (not stage.is_dir() or stage.is_symlink() or
                not all(path.is_relative_to(stage) and len(path.relative_to(stage).parts) >= 2 for path in paths)):
            continue
        if not stage.resolve().is_relative_to(uploads.resolve()):
            continue
        shutil.rmtree(stage)


def run_job(job: dict) -> None:
    global _CURRENT_PROJECT_JOB
    kind = job.get("kind")
    is_project = kind in {"project_import", "project_train"}
    if is_project:
        if not isinstance(job.get("project_id"), str) or not isinstance(job.get("job_id"), str):
            raise ValueError("Project job requires project_id and job_id")
        _CURRENT_PROJECT_JOB = {"kind": kind, "project_id": job["project_id"], "job_id": job["job_id"]}
    else:
        clear_stop()
    pid_path().write_text(str(os.getpid()), encoding="utf-8")
    log_lines: list[str] = []

    if kind != "train":
        try:
            from pdm.visualization.live import clear_live_activity

            clear_live_activity()
        except Exception:  # noqa: BLE001
            pass

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg, flush=True)

    def stopped() -> bool:
        return _project_stopped(job["job_id"]) if is_project else stop_path().exists()

    try:
        if kind == "project_import":
            from pdm.data.project_import import import_project
            from pdm.data.project_prepare import prepare_project

            if stopped():
                raise InterruptedError("Project import cancelled before start")
            write_status({"status": "running", "stage": "import", "progress": 0.0,
                          "message": "Importing project source"})
            if job["source"].get("import_protocol") == "verified_sensor_release":
                from pdm.probabilistic.workflow import import_source_plan

                snapshot = import_source_plan(job["project_id"], job["source"], should_stop=stopped,
                    status_cb=lambda update: write_status({"status": "running", **update}))
            else:
                source = import_project(job["project_id"], job["source"], should_stop=stopped)
                if stopped():
                    raise InterruptedError("Project import cancelled")
                source_id = source.get("manifest_id")
                if not source_id:
                    raise ValueError("Import did not return a source manifest ID")
                write_status({"status": "running", "stage": "prepare", "progress": 0.5,
                              "message": "Preparing project snapshot"})
                snapshot = prepare_project(job["project_id"], source_id, should_stop=stopped)
            write_status({"status": "completed", "stage": "completed", "progress": 1.0,
                          "snapshot_id": snapshot["snapshot_id"], "message": "Project data ready"})
        elif kind == "project_train":
            from pdm.project_tasks import train_project_job, validate_training_job

            if stopped():
                raise InterruptedError("Project training cancelled before start")
            task = validate_training_job(job)
            write_status({"status": "running", "stage": "training", "progress": 0.0,
                          "task": task,
                          "message": "Training first RED entry model" if task == "red_entry" else "Training numeric signal model"})

            def project_progress(update: dict) -> None:
                write_status({"status": "running", **update})

            run = train_project_job(job, should_stop=stopped, status_cb=project_progress)
            write_status({"status": "completed", "stage": "completed", "progress": 1.0,
                          "task": task, "run_id": run["run_id"],
                          "message": "First RED entry model ready" if task == "red_entry" else "Signal model ready"})
        elif kind == "condition_study":
            from pdm.monitoring.study import run_condition_study

            run_condition_study(job, should_stop=stopped, log=log)
        elif kind == "monitor_evaluate":
            from pdm.monitoring.evaluation import evaluate_monitoring

            write_status({"kind": kind, "status": "training", "bundle_id": job["bundle_id"]})
            result = evaluate_monitoring(job["bundle_id"], job.get("split_name", "validation"), should_stop=stopped, log=log)
            write_status({"kind": kind, **result})
        elif kind == "training_study":
            from pdm.training_study import run_training_study

            run_training_study(job)
        elif kind == "future_red_matrix":
            from scripts import run_future_red_matrix

            dataset_id = str(job["dataset_id"])
            if dataset_id not in {"bearings", "filters"}:
                raise ValueError(f"Unsupported matrix dataset: {dataset_id}")
            architectures = tuple(job.get("architectures") or ("gru",))
            if not architectures or any(
                architecture not in {"gru", "lstm", "fly", "random"}
                for architecture in architectures
            ):
                raise ValueError("Choose at least one supported matrix model")
            args = run_future_red_matrix._parser().parse_args(
                ["--datasets", dataset_id, "--architectures", ",".join(architectures)]
            )
            write_status({
                "status": "training", "kind": kind, "dataset_id": dataset_id,
                "architectures": list(architectures), "message": "Training selected models",
            })
            output_dir = run_future_red_matrix.run(args)
            states = read_json(output_dir / "status.json")
            failures = [key for key, state in states.items() if state.get("status") != "completed"]
            write_status({
                "status": "failed" if failures else "completed", "kind": kind,
                "dataset_id": dataset_id, "run_id": output_dir.name,
                "architectures": list(architectures), "failed_models": failures,
                "message": "Training finished" if not failures else "Some models failed; inspect run details",
            })
        elif kind == "train_matrix":
            from pdm.batch import run_batch

            run_batch(job)
        elif kind == "prepare":
            from pdm.data.prepare import prepare_dataset

            write_status({"status": "preparing", "dataset_id": job["dataset_id"], "kind": kind})

            def progress(stage, info):
                write_status({"status": "preparing", "dataset_id": job["dataset_id"], "stage": stage, **info})

            prepare_dataset(job["dataset_id"], progress=progress)
            write_status({"status": "ready", "dataset_id": job["dataset_id"], "kind": kind})
        elif kind == "train_full_cns":
            from pdm.connectome.morphology import morphology_directory, prepare_morphology
            from pdm.train_full_cns import main as train_full_cns

            if job["dataset_id"] != "bearings":
                raise ValueError("Full CNS forecasting currently supports bearings")
            write_status({"status": "training", "kind": kind, "dataset_id": "bearings",
                          "message": "Preparing complete MaleCNS and fitting a new forecast readout"})
            if not (morphology_directory() / "manifest.json").is_file():
                prepare_morphology()
            try:
                arguments = []
                if job.get("training_protocol"):
                    path = job_path().parent / "full_cns_protocol.json"
                    atomic_write_json(path, job["training_protocol"])
                    arguments = ["--training-protocol", str(path), "--defer-test"]
                train_full_cns(arguments)
            except InterruptedError:
                write_status({"status": "cancelled", "kind": kind, "dataset_id": "bearings"})
            else:
                write_status({"status": "completed", "kind": kind, "dataset_id": "bearings"})
        elif kind == "train":
            from pdm.train import run_training

            write_status({"status": "training", "dataset_id": job["dataset_id"], "kind": kind})

            def status_cb(payload):
                payload = dict(payload)
                payload["kind"] = "train"
                write_status(payload)

            mw = job.get("max_windows_per_unit")
            if mw == "":
                mw = None
            run_training(
                job["dataset_id"],
                architecture=job.get("architecture", "gru"),
                max_epochs=job.get("max_epochs"),
                history_length=job.get("history_length"),
                history_mode=job.get("history_mode"),
                smoke=bool(job.get("smoke", False)),
                resume_run_id=job.get("resume_run_id"),
                device_pref=job.get("device", "auto"),
                log=log,
                should_stop=stopped,
                status_cb=status_cb,
                max_windows_per_unit=mw,
                n_nodes=job.get("n_nodes"),
                graph_mode=job.get("graph_mode"),
                readout=job.get("readout"),
                source_path=job.get("source_path"),
                seed=job.get("seed"),
                events_only=bool(job.get("events_only", False)),
                training_protocol=job.get("training_protocol"),
            )
        elif kind in {"evaluate", "replay_predict"}:
            from pdm.evaluate import evaluate_run

            write_status(
                {
                    "status": "training",
                    "dataset_id": job["dataset_id"],
                    "kind": kind,
                    "run_id": job["run_id"],
                }
            )
            eval_kwargs: dict = {
                "split_name": job.get("split_name", "test"),
                "policy_mode": job.get("policy_mode"),
                "device": job.get("device", "auto"),
            }
            for key in (
                "H_trigger",
                "warning_horizon_s",
                "confirmation_count",
                "minimum_action_lead_time",
                "max_useful_horizon_s",
                "with_trace",
            ):
                if key in job:
                    eval_kwargs[key] = job[key]
            metrics = evaluate_run(job["dataset_id"], job["run_id"], **eval_kwargs)
            final = "cancelled" if stop_path().exists() else "completed"
            write_status(
                {
                    "status": final,
                    "kind": kind,
                    "dataset_id": job["dataset_id"],
                    "run_id": job["run_id"],
                    "eval_id": metrics.get("eval_id"),
                    "eval_dir": metrics.get("eval_dir"),
                    "metrics_keys": list(metrics),
                }
            )
        elif kind == "trace":
            from pdm.visualization.trace import run_trace_job

            write_status(
                {
                    "status": "training",
                    "dataset_id": job["dataset_id"],
                    "kind": kind,
                    "run_id": job["run_id"],
                    "unit_id": job.get("unit_id"),
                }
            )
            rec = run_trace_job(
                job["dataset_id"],
                job["run_id"],
                job.get("unit_id"),
                should_stop=stopped,
                lazy=bool(job.get("lazy", False)),
                device=job.get("device", "cpu"),
            )
            final = (
                "cancelled"
                if stop_path().exists() or rec.get("status") == "cancelled"
                else "completed"
            )
            write_status(
                {
                    "status": final,
                    "kind": kind,
                    "dataset_id": job["dataset_id"],
                    "run_id": job["run_id"],
                    "unit_id": job.get("unit_id"),
                    "trace_dir": rec.get("trace_dir"),
                }
            )
        elif kind == "download":
            from pdm.data.download import download_dataset

            write_status({"status": "preparing", "dataset_id": job["dataset_id"], "kind": "download"})
            download_dataset(job["dataset_id"], local_path=job.get("local_path"))
            write_status({"status": "ready", "dataset_id": job["dataset_id"], "kind": "download"})
        else:
            raise ValueError(f"Unknown job kind {kind}")
    except InterruptedError as exc:
        if not is_project:
            raise
        write_status({"status": "cancelled", "stage": "cancelled", "progress": None,
                      "message": "Project job cancelled", "error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        write_status(
            {
                "status": "failed",
                "kind": kind,
                "dataset_id": job.get("dataset_id"),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
        if is_project:
            _CURRENT_PROJECT_JOB = None
            if _project_stopped(job["job_id"]):
                clear_stop()
            if kind == "project_import":
                try:
                    _cleanup_project_uploads(job)
                except (OSError, ValueError, KeyError):
                    pass  # Staging cleanup must not replace the import outcome.
        if pid_path().exists():
            try:
                pid_path().unlink()
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-file", default=str(job_path()))
    args = parser.parse_args(argv)
    job = json.loads(Path(args.job_file).read_text(encoding="utf-8"))
    run_job(job)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
