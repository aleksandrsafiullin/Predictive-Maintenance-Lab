from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path

from pdm.io_util import atomic_write_json, read_json
from pdm.paths import worker_dir


def status_path() -> Path:
    return worker_dir() / "status.json"


def stop_path() -> Path:
    return worker_dir() / "stop.flag"


def job_path() -> Path:
    return worker_dir() / "job.json"


def pid_path() -> Path:
    return worker_dir() / "worker.pid"


def read_status() -> dict:
    p = status_path()
    if not p.exists():
        return {"status": "not_ready"}
    try:
        return read_json(p)
    except Exception:
        return {"status": "unknown"}


def request_stop() -> None:
    stop_path().write_text("stop\n", encoding="utf-8")


def clear_stop() -> None:
    if stop_path().exists():
        stop_path().unlink()


def worker_alive() -> bool:
    p = pid_path()
    if not p.exists():
        return False
    try:
        pid = int(p.read_text().strip())
    except Exception:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def write_status(payload: dict) -> None:
    payload = dict(payload)
    payload["pid"] = os.getpid()
    atomic_write_json(status_path(), payload)


def run_job(job: dict) -> None:
    kind = job.get("kind")
    clear_stop()
    pid_path().write_text(str(os.getpid()), encoding="utf-8")
    log_lines: list[str] = []

    def log(msg: str) -> None:
        log_lines.append(msg)
        print(msg, flush=True)

    def stopped() -> bool:
        return stop_path().exists()

    try:
        if kind == "prepare":
            from pdm.data.prepare import prepare_dataset

            write_status({"status": "preparing", "dataset_id": job["dataset_id"], "kind": kind})

            def progress(stage, info):
                write_status({"status": "preparing", "dataset_id": job["dataset_id"], "stage": stage, **info})

            prepare_dataset(job["dataset_id"], progress=progress)
            write_status({"status": "ready", "dataset_id": job["dataset_id"], "kind": kind})
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
                smoke=bool(job.get("smoke", False)),
                resume_run_id=job.get("resume_run_id"),
                device_pref=job.get("device", "auto"),
                log=log,
                should_stop=stopped,
                status_cb=status_cb,
                max_windows_per_unit=mw,
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
            metrics = evaluate_run(
                job["dataset_id"],
                job["run_id"],
                warning_horizon_s=job.get("H_trigger", job.get("warning_horizon_s")),
                confirmation_count=job.get("confirmation_count"),
                minimum_action_lead_time=job.get("minimum_action_lead_time"),
                max_useful_horizon_s=job.get("max_useful_horizon_s"),
                device=job.get("device", "auto"),
            )
            write_status(
                {
                    "status": "completed",
                    "kind": kind,
                    "dataset_id": job["dataset_id"],
                    "run_id": job["run_id"],
                    "eval_id": metrics.get("eval_id"),
                    "eval_dir": metrics.get("eval_dir"),
                    "metrics_keys": list(metrics),
                }
            )
        elif kind == "download":
            from pdm.data.download import download_dataset

            write_status({"status": "preparing", "dataset_id": job["dataset_id"], "kind": "download"})
            download_dataset(job["dataset_id"], local_path=job.get("local_path"))
            write_status({"status": "ready", "dataset_id": job["dataset_id"], "kind": "download"})
        else:
            raise ValueError(f"Unknown job kind {kind}")
    except Exception as exc:  # noqa: BLE001
        write_status(
            {
                "status": "failed",
                "kind": kind,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise
    finally:
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
