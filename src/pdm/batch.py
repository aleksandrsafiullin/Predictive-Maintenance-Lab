"""Restartable, sequential training/evaluation matrix for the local worker."""
from __future__ import annotations

import contextlib
import gc
import json
import traceback
import uuid
from datetime import datetime, timezone

import pandas as pd
import torch

from pdm.alerts import build_alert_policy, save_alert_policy
from pdm.benchmark import compare_evaluations, load_evaluation, save_comparison
from pdm.config import load_dataset_config
from pdm.data.prepare import load_processed
from pdm.data.quality import training_admission
from pdm.evaluate import default_horizon_s, evaluate_run
from pdm.experiments import run_dir
from pdm.io_util import atomic_write_json, checkpoint_hash, read_json
from pdm.paths import runs_root
from pdm.train import run_training
from pdm.worker import stop_path, write_status


def matrix_tasks(datasets, include_ablation=True):
    tasks = []
    for ds in datasets:
        for arch in ("gru", "lstm", "fly_connectome_reservoir", "random_reservoir"):
            tasks.append({"dataset_id": ds, "architecture": arch, "seed": 42, "events_only": False, "role": "main"})
        if ds == "bearings":
            tasks.append({"dataset_id": ds, "architecture": "full_cns", "seed": 42, "events_only": False, "role": "main"})
    if include_ablation and "filters" in datasets:
        for seed in (42, 43, 44):
            for events_only in (False, True):
                if seed == 42 and not events_only:
                    continue  # Main GRU is also the paired ablation reference.
                tasks.append({"dataset_id": "filters", "architecture": "gru", "seed": seed, "events_only": events_only, "role": "ablation"})
    return tasks


def evaluation_splits(task):
    # The paired censoring experiment is a validation-only question. Its test
    # outcomes must not be used to choose a training-data policy.
    return ("validation",) if task.get("role") == "ablation" else ("validation", "test")


def run_batch(job):
    datasets = job.get("datasets") or ["bearings", "filters"]
    batch_id = job.get("batch_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
    if "/" in batch_id or "\\" in batch_id or batch_id.startswith("."):
        raise ValueError("Invalid batch identifier")
    directory = runs_root() / "batches" / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        datasets = list(manifest["dataset_versions"])
    else:
        manifest = {"batch_id": batch_id, "tasks": matrix_tasks(datasets, job.get("include_ablation", True)),
                    "dataset_versions": {ds: load_processed(ds)["dataset_version"] for ds in datasets}}
    for ds in datasets:
        p = load_processed(ds)
        if p["dataset_version"] != manifest["dataset_versions"][ds]:
            raise ValueError(f"{ds} changed since batch creation. Start a new batch.")
        training_admission(p, load_dataset_config(ds), 20)
    torch.set_num_threads(1)
    manifest["status"] = "running"
    atomic_write_json(manifest_path, manifest)
    with (directory / "batch.log").open("a", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        for index, task in enumerate(manifest["tasks"]):
            if task.get("status") == "completed":
                continue
            if stop_path().exists():
                manifest["status"] = "cancelled"
                break
            ds = task["dataset_id"]
            label = f"{ds} · {task['architecture']} · seed {task['seed']}" + (" · events only" if task["events_only"] else "")

            def status(payload):
                write_status({**payload, "kind": "train_matrix", "batch_id": batch_id, "dataset_id": ds,
                              "task_index": index + 1, "task_total": len(manifest["tasks"]), "message": label})

            try:
                task["status"] = "running"
                task.pop("error", None)
                atomic_write_json(manifest_path, manifest)
                status({"status": "training"})
                print(label, flush=True)
                if not task.get("run_id"):
                    if task["architecture"] == "full_cns":
                        from pdm.train_full_cns import main

                        result = main([])
                    else:
                        res = task["architecture"] in {"fly_connectome_reservoir", "random_reservoir"}
                        result = run_training(ds, architecture=task["architecture"], seed=task["seed"],
                                              max_epochs=30, history_length=20, max_windows_per_unit=0, smoke=False,
                                              device_pref="cpu", n_nodes=1000 if res else None,
                                              graph_mode="real_connectome" if res else None,
                                              readout=("ridge" if ds == "bearings" else "gradient") if res else None,
                                              events_only=task["events_only"], log=lambda line: print(line, flush=True),
                                              should_stop=lambda: stop_path().exists(), status_cb=status)
                    if result["status"] != "completed":
                        raise InterruptedError("Training cancelled")
                    task["run_id"] = result["run_id"]
                    atomic_write_json(manifest_path, manifest)
                rdir = run_dir(ds, task["run_id"])
                atomic_write_json(rdir / "benchmark_context.json", {"batch_id": batch_id, "role": task["role"], "evaluation_status": "exploratory_reused_holdout"})
                bundle = load_processed(ds)
                train_units = bundle["units"][bundle["units"].unit_id.isin(bundle["split"]["train"])]
                h = default_horizon_s(train_units)
                policy = build_alert_policy(H_trigger=h, confirmation_count=3, minimum_action_lead_time=h / 2,
                                            source="benchmark_validation", split="validation", unit_ids=bundle["split"]["validation"],
                                            checkpoint_hash=checkpoint_hash(rdir / "best.pt"))
                for part in evaluation_splits(task):
                    if stop_path().exists():
                        raise InterruptedError("Evaluation cancelled")
                    if task.get(f"{part}_eval_id"):
                        continue
                    status({"status": "evaluating", "stage": part, "run_id": task["run_id"]})
                    if part == "test":
                        save_alert_policy(rdir, policy, overwrite=False)
                    metrics = evaluate_run(ds, task["run_id"], split_name=part, device="cpu",
                                           should_stop=lambda: stop_path().exists(),
                                           progress_cb=lambda info: status({"status": "evaluating", "stage": part, "run_id": task["run_id"], **info}),
                                           policy_mode="frozen" if part == "test" else "research",
                                           **({"H_trigger": h, "confirmation_count": 3, "minimum_action_lead_time": h / 2} if part == "validation" else {}))
                    task[f"{part}_eval_id"] = metrics["eval_id"]
                    atomic_write_json(manifest_path, manifest)
                task["status"] = "completed"
            except InterruptedError:
                task["status"] = "cancelled"
                manifest["status"] = "cancelled"
                break
            except Exception as exc:
                task.update(status="failed", error=str(exc))
                print(traceback.format_exc(), flush=True)
            finally:
                atomic_write_json(manifest_path, manifest)
                gc.collect()
        else:
            manifest["status"] = "completed" if all(t.get("status") == "completed" for t in manifest["tasks"]) else "failed"
        manifest["comparisons"] = []
        manifest.pop("comparison_errors", None)
        for ds in datasets:
            main_tasks = [t for t in manifest["tasks"] if t["dataset_id"] == ds and t["role"] == "main" and t.get("status") == "completed"]
            if len(main_tasks) >= 2:
                for part in ("validation", "test"):
                    try:
                        comparison = compare_evaluations([load_evaluation(ds, t["run_id"], t[f"{part}_eval_id"]) for t in main_tasks])
                        manifest["comparisons"].append(str(save_comparison(comparison)))
                    except ValueError as exc:
                        manifest.setdefault("comparison_errors", []).append(str(exc))
                        manifest["status"] = "failed"
        records = []
        for task in manifest["tasks"]:
            if task["dataset_id"] == "filters" and task["architecture"] == "gru" and task.get("validation_eval_id"):
                ev = load_evaluation("filters", task["run_id"], task["validation_eval_id"])
                records.append({"seed": task["seed"], "events_only": task["events_only"], "run_id": task["run_id"],
                                **{k: ev["metrics"].get(k) for k in ("survival_nll", "event_mae", "n_observed_events", "nll_unit_count")}})
        if records:
            pd.DataFrame(records).to_csv(directory / "censoring_ablation.csv", index=False)
            print(json.dumps(records, indent=2), flush=True)
        atomic_write_json(manifest_path, manifest)
    write_status({"status": manifest["status"], "kind": "train_matrix", "batch_id": batch_id,
                  "manifest": str(manifest_path), "message": f"Batch {manifest['status']}"})
    return manifest
