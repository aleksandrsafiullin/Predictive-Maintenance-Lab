"""Bounded sequential optimization study. Test starts only after selection freezes."""
from __future__ import annotations

import copy
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pdm.alerts import build_alert_policy, save_alert_policy
from pdm.benchmark import compare_evaluations, load_evaluation, save_comparison
from pdm.data.prepare import load_processed
from pdm.evaluate import evaluate_run
from pdm.experiments import run_dir
from pdm.io_util import atomic_write_json, checkpoint_hash, read_json, sha256_file
from pdm.paths import runs_root
from pdm.study_metrics import warning_settings
from pdm.train import new_run_id, run_training
from pdm.training_protocol import fingerprint, protocol
from pdm.training_readout import training_folds
from pdm.worker import stop_path, write_status


class Study:
    def __init__(self, job):
        self.id = job.get("study_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if "/" in self.id or self.id.startswith("."):
            raise ValueError("Invalid study id")
        self.root = runs_root() / "training_studies" / self.id
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "manifest.json"
        if self.path.exists():
            self.manifest = read_json(self.path)
        else:
            batches = sorted((runs_root() / "batches").glob("*/manifest.json"), reverse=True)
            source = next((read_json(path) for path in batches if read_json(path).get("status") == "completed"), None)
            if source is None:
                raise ValueError("A completed reference matrix is required")
            self.manifest = {"study_id": self.id, "version": "training_study_v2", "status": "running",
                             "datasets": {ds: load_processed(ds)["dataset_version"] for ds in ("bearings", "filters")},
                             "reference_runs": [t for t in source["tasks"] if t["role"] == "main"],
                             "tasks": {}, "selections": {}, "warning_settings": {ds: warning_settings(ds) for ds in ("bearings", "filters")}}
        for ds, version in self.manifest["datasets"].items():
            if load_processed(ds)["dataset_version"] != version:
                raise ValueError("Source data changed during study")
        for key, task in self.manifest["tasks"].items():
            if ":adaptive:" in key and not task.get("parent"):
                task["parent"] = self.manifest["tasks"][task["dataset_id"] + ":diagnostic100"]["run_id"]
        self.save()

    def save(self):
        atomic_write_json(self.path, self.manifest)

    def progress(self, **payload):
        write_status({**payload, "kind": "training_study", "study_id": self.id})

    def log(self, message):
        with (self.root / "study.log").open("a", buffering=1) as handle:
            handle.write(str(message) + "\n")

    def check_stop(self):
        if stop_path().exists():
            raise InterruptedError("Study stopped")

    def fit(self, key, ds, arch, recipe, *, seed=42, fold=None, parent=None):
        self.check_stop()
        tasks = self.manifest["tasks"]
        if key not in tasks:
            tasks[key] = {"dataset_id": ds, "architecture": arch, "seed": seed, "recipe": copy.deepcopy(recipe),
                          "fold": fold, "parent": parent, "status": "pending", "run_id": None}
        task = tasks[key]
        if task["recipe"] != recipe or task.get("fold") != fold:
            self.manifest.setdefault("superseded_tasks", []).append({"key": key, **copy.deepcopy(task),
                 "reason": "Protocol correction or changed parent selection; checkpoint cannot be reused"})
            task = tasks[key] = {"dataset_id": ds, "architecture": arch, "seed": seed,
                   "recipe": copy.deepcopy(recipe), "fold": fold, "parent": parent, "status": "pending", "run_id": None}
        if parent is not None:
            task["parent"] = parent
        if task["status"] == "completed" and task.get("primary_score") is not None:
            self.save()
            return task
        rid = task.get("run_id")
        saved_status = run_dir(ds, rid) / "status.json" if rid else None
        recovered = saved_status is not None and saved_status.exists() and read_json(saved_status).get("status") == "completed"
        task["status"] = "training"
        self.save()
        self.progress(status="training", stage=key, message=key)
        if recovered:
            result = {"status": "completed", "run_id": rid}
        elif arch == "full_cns":
            from contextlib import redirect_stdout

            from pdm.train_full_cns import main

            path = self.root / (key.replace(":", "_") + "_protocol.json")
            atomic_write_json(path, recipe)
            with (self.root / "study.log").open("a", buffering=1) as output, redirect_stdout(output):
                result = main(["--seed", str(seed), "--training-protocol", str(path), "--defer-test"],
                              status_cb=lambda message: self.progress(status="training", stage=key, message=message, dataset_id=ds))
        else:
            rid = task.get("run_id")
            checkpoint = run_dir(ds, rid) / "last.pt" if rid else None
            if checkpoint is not None and checkpoint.exists():
                resume = rid
            else:
                resume = None
                rid = new_run_id(ds, arch, False)
                task["run_id"] = rid
                self.save()
            result = run_training(ds, architecture=arch, training_protocol=recipe, seed=seed,
                                  n_nodes=1000, graph_mode="real_connectome", split_override=fold,
                                  resume_run_id=resume, run_id_override=rid, device_pref="cpu",
                                  log=self.log, should_stop=lambda: stop_path().exists(),
                                  status_cb=lambda payload: self.progress(**{**payload, "stage": key, "message": key}))
        task["run_id"] = result["run_id"]
        if result["status"] != "completed":
            task["status"] = result["status"]
            self.save()
            raise InterruptedError("Training interrupted")
        root = run_dir(ds, task["run_id"])
        metrics = read_json(root / "validation_metrics.json")
        task["primary_score"] = metrics["best_metric"]
        task["best_epoch"] = metrics["best_epoch"]
        task["elapsed_s"] = float(pd.read_csv(root / "training_history.csv").get("elapsed_s", pd.Series([0.])).iloc[-1])
        task["status"] = "completed"
        atomic_write_json(root / "benchmark_context.json", {"study_id": self.id, "role": key,
                          "evaluation_status": "exploratory_reused_holdout"})
        self.save()
        return task

    def evaluate(self, task, split="validation", *, controls=False):
        self.check_stop()
        field = split + "_evaluation"
        if task.get(field):
            return task[field]
        ds, rid = task["dataset_id"], task["run_id"]
        settings = self.manifest["warning_settings"][ds]
        kwargs = {key: value for key, value in settings.items() if key != "reset_factor"}
        if split == "test" and not controls:
            frozen = build_alert_policy(**settings, source="training_study_validation", split="validation",
                                         unit_ids=load_processed(ds)["split"]["validation"],
                                         checkpoint_hash=checkpoint_hash(run_dir(ds, rid) / "best.pt"))
            saved_policy = save_alert_policy(run_dir(ds, rid), frozen, overwrite=False)
            if any(saved_policy.get(key) != value for key, value in settings.items()):
                # Archive any provisional UI policy before the first test score.
                save_alert_policy(run_dir(ds, rid), frozen, overwrite=True)
            kwargs = {"policy_mode": "frozen"}
        else:
            kwargs["policy_mode"] = "research"
        metrics = evaluate_run(ds, rid, split_name=split, device="cpu", **kwargs,
                               should_stop=lambda: stop_path().exists(),
                               progress_cb=lambda p: self.progress(status="evaluating", stage=split, run_id=rid, **p))
        task[field] = metrics["eval_id"]
        if split == "validation" and not controls:
            row = compare_evaluations([load_evaluation(ds, rid, metrics["eval_id"])])["table"].iloc[0]
            if not row.rank_eligible:
                raise ValueError(f"Saved candidate is ineligible: {rid}: {row.reason}")
            task["training_primary_score"] = task["primary_score"]
            task["primary_score"] = float(row.primary_score)
            task["warning_goal_met"] = bool(row.warning_goal_met)
        self.save()
        return metrics["eval_id"]

    @staticmethod
    def ordered(tasks):
        return sorted(tasks, key=lambda t: (t["primary_score"], t["recipe"]["feature_recipe"] != "base_v1", t.get("elapsed_s", 0)))

    def run(self):
        self.manifest["status"] = "running"
        self.save()
        # Pilot fitting starts first; the reference evaluation cannot affect it.
        for ds in self.manifest["datasets"]:
            if ds in self.manifest["selections"]:
                continue
            pilots = []
            diagnostic = self.fit(f"{ds}:diagnostic100", ds, "gru", protocol(ds, mode="diagnostic"))
            pilots.append(diagnostic)
            history = pd.read_csv(run_dir(ds, diagnostic["run_id"]) / "training_history.csv")
            best, bad, stop_epoch = np.inf, 0, min(30, len(history))
            # Counterfactual uses the former all-history MAE for bearings.
            metric_col = "val_all_history_mae_s" if ds == "bearings" else "val_metric"
            for row in history.iloc[:30].to_dict("records"):
                if row[metric_col] < best - 1e-8:
                    best, bad = row[metric_col], 0
                else:
                    bad += 1
                if bad >= 5:
                    stop_epoch = int(row["epoch"])
                    break
            self.manifest.setdefault("epoch_experiment", {})[ds] = {
                "run_id": diagnostic["run_id"], "former_stop_epoch": stop_epoch,
                "best_new_metric_before_former_stop": float(history.loc[history.epoch <= stop_epoch, "val_metric"].min()),
                "best_new_metric_through_100": float(history.val_metric.min()),
                "best_epoch": int(history.loc[history.val_metric.idxmin(), "epoch"]),
                "note": "Counterfactual on the new numeric implementation; original run remains separate."}
            for lr in (.001, .0003):
                pilots.append(self.fit(f"{ds}:adaptive:{lr}", ds, "gru", protocol(ds, learning_rate=lr), parent=diagnostic["run_id"]))
            parent = self.ordered(pilots)[0]
            full = {**parent["recipe"], "sampling": "full_pass", "window_loss_reduction": "fixed_batch_denominator"}
            pilots.append(self.fit(f"{ds}:full_pass", ds, "gru", full, parent=parent["run_id"]))
            if ds == "bearings":
                parent = pilots[-1]
                pilots.append(self.fit(f"{ds}:near_weight", ds, "gru", {**parent["recipe"], "near_weight": .5}, parent=parent["run_id"]))
            parent = self.ordered(pilots)[0]
            pilots.append(self.fit(f"{ds}:degradation", ds, "gru", {**parent["recipe"], "feature_recipe": "degradation_v1"}, parent=parent["run_id"]))
            finalists = self.ordered(pilots)[:2]
            bundle = load_processed(ds)
            folds = training_folds(ds, bundle["units"], bundle["split"]["train"])
            candidates = []
            for candidate_index, task in enumerate(finalists):
                results = []
                for number, fold in enumerate(folds):
                    result = self.fit(f"{ds}:cv:{candidate_index}:{number}", ds, "gru", task["recipe"], fold=fold, parent=task["run_id"])
                    results.append(result["primary_score"])
                candidates.append({"recipe": task["recipe"], "primary_score": float(np.average(results, weights=[len(f["validation"]) for f in folds])),
                                   "source_run": task["run_id"], "fold_scores": results, "elapsed_s": task.get("elapsed_s", 0)})
            chosen = self.ordered(candidates)[0]
            self.manifest["selections"][ds] = {"chosen": chosen, "candidates": candidates, "folds": folds,
                                                 "selection_rule": "equal-equipment grouped CV primary; ties: base features, time"}
            self.save()
        for task in self.manifest["reference_runs"]:
            self.evaluate(task, controls=True)
        from pdm.study_baselines import fit_baseline

        for ds in self.manifest["datasets"]:
            self.progress(status="training", stage="simple_baselines", dataset_id=ds)
            fit_baseline(ds, self.root / "baselines" / ds)
        main = []
        for ds, selected in self.manifest["selections"].items():
            recipe = selected["chosen"]["recipe"]
            architectures = ["gru", "lstm", "fly_connectome_reservoir", "random_reservoir"] + (["full_cns"] if ds == "bearings" else [])
            dataset_tasks = []
            for arch in architectures:
                result = self.fit(f"{ds}:main:{arch}:42", ds, arch, recipe)
                self.evaluate(result)
                dataset_tasks.append(result)
                main.append(result)
            # Locked by validation score before extra seeds or test evaluation.
            selected.setdefault("confirmation_architectures", [t["architecture"] for t in self.ordered(dataset_tasks)[:2]])
            self.save()
            for arch in selected["confirmation_architectures"]:
                for seed in (43, 44):
                    task = self.fit(f"{ds}:confirmation:{arch}:{seed}", ds, arch, recipe, seed=seed)
                    self.evaluate(task)
                    main.append(task)
        if "frozen_selection" not in self.manifest:
            self.manifest["frozen_selection"] = {"runs": [t["run_id"] for t in main],
                     "warning_settings": copy.deepcopy(self.manifest["warning_settings"]), "selection_hash": fingerprint(self.manifest["selections"]),
                     "baseline_sha256": {ds: sha256_file(self.root / "baselines" / ds / "model.json") for ds in self.manifest["datasets"]},
                     "frozen_at": datetime.now(timezone.utc).isoformat()}
            self.save()
        frozen = self.manifest["frozen_selection"]
        if frozen["runs"] != [t["run_id"] for t in main] or frozen["selection_hash"] != fingerprint(self.manifest["selections"]):
            raise ValueError("Candidate selection changed after test freeze; start a separate study")
        if frozen["warning_settings"] != self.manifest["warning_settings"]:
            raise ValueError("Warning rules changed after test freeze")
        for ds, expected in frozen.get("baseline_sha256", {}).items():
            if sha256_file(self.root / "baselines" / ds / "model.json") != expected:
                raise ValueError("A simple control changed after test freeze")
        for task in main:
            self.evaluate(task, "test")
        for task in self.manifest["reference_runs"]:
            self.evaluate(task, "test", controls=True)
        self.manifest["comparisons"] = []
        for ds in self.manifest["datasets"]:
            for part in ("validation", "test"):
                tasks = [t for t in main if t["dataset_id"] == ds and t["seed"] == 42]
                tasks += [t for t in self.manifest["reference_runs"] if t["dataset_id"] == ds]
                comparison = compare_evaluations([load_evaluation(ds, t["run_id"], t[part + "_evaluation"]) for t in tasks])
                self.manifest["comparisons"].append(str(save_comparison(comparison)))
        from pdm.study_report import build_study_report

        build_study_report(self.root, self.manifest)
        self.manifest["status"] = "completed"
        self.save()
        self.progress(status="completed", message="Training study completed", manifest=str(self.path))


def run_training_study(job):
    study = Study(job)
    try:
        study.run()
    except InterruptedError:
        for task in study.manifest["tasks"].values():
            if task.get("status") == "training":
                task["status"] = "cancelled"
        study.manifest["status"] = "cancelled"
        study.save()
        study.progress(status="cancelled", message="Study paused; completed tasks retained")
    except Exception as exc:
        for task in study.manifest["tasks"].values():
            if task.get("status") == "training":
                task.update(status="failed", error=str(exc))
        study.manifest.update(status="failed", error=str(exc))
        study.log(traceback.format_exc())
        study.save()
        study.progress(status="failed", error=str(exc))
        raise
    return study.manifest
