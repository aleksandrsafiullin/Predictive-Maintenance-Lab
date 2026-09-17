"""Bounded condition study on the existing worker and training-v2 engine."""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pdm.config import load_dataset_config
from pdm.data.prepare import dataset_fingerprint_for_run, load_processed
from pdm.data.quality import training_admission
from pdm.io_util import atomic_write_json, load_yaml, read_json, sha256_file
from pdm.monitoring.bundle import freeze_bundle
from pdm.monitoring.calibration import calibration_report
from pdm.monitoring.contracts import dataset_profile, unit_verification
from pdm.monitoring.normality import assess_normality, fit_reference
from pdm.monitoring.policy import default_policy
from pdm.monitoring.quality import quality_policy
from pdm.monitoring.signal_forecast import fit_sensor, supervised_targets
from pdm.paths import dataset_runs, project_root, runs_root
from pdm.splits import assert_split_coverage
from pdm.training_protocol import fingerprint, protocol
from pdm.training_readout import training_folds


def study_root(study_id):
    if Path(study_id).name != study_id:
        raise ValueError("Invalid study identifier")
    return runs_root() / "condition_studies" / study_id


def plan_study(config):
    cfg = load_yaml(Path(config)) if isinstance(config, (str, Path)) else dict(config)
    dataset = cfg["dataset_id"]
    data = load_processed(dataset)
    split, counts = training_admission(data, load_dataset_config(dataset), 20)
    assert_split_coverage(data["units"].copy(), split)
    folds = training_folds(dataset, data["units"], split["train"])
    jobs = []
    # Ten event fits; six horizon/quantile fits in each sensor stage.
    for mode in ("fixed_20", "fixed_40", "fixed_60", "variable_20_60"):
        jobs.append({"id": "event_" + mode, "kind": "event", "history_mode": mode, "recipe": "base_v1", "fold": 0, "seed": 42})
    for recipe in ("multiscale_trend_v2", "multiscale_no_age_v2"):
        jobs.append({"id": "event_" + recipe, "kind": "event", "history_mode": "selected", "recipe": recipe, "fold": 0, "seed": 42})
    jobs.append({"id": "event_group_confirmation", "kind": "event", "history_mode": "selected", "recipe": "selected", "fold": 1, "seed": 42})
    jobs.append({"id": "event_final", "kind": "event", "history_mode": "selected", "recipe": "selected", "fold": None, "seed": 42})
    for seed in (43, 44):
        jobs.append({"id": f"event_seed_{seed}", "kind": "event", "history_mode": "selected", "recipe": "selected", "fold": 0, "seed": seed})
    for phase in ("screen", "final"):
        for h in cfg["sensor_horizons"]:
            for q in (.05, .5, .95):
                jobs.append({"id": f"sensor_{phase}_{h}_{q}", "kind": "sensor", "phase": phase, "horizon": h, "quantile": q})
    if len(jobs) > cfg.get("fit_job_budget", 24):
        raise ValueError("Plan exceeds fit budget; reduce horizons before starting")
    return {"schema_version": "condition_study_v1", "config": cfg, "data_binding": dataset_fingerprint_for_run(data, split),
            "split": split, "folds": folds, "admission": counts, "fit_jobs": [{**j, "status": "pending"} for j in jobs],
            "planned_fits": len(jobs), "test_used_for_selection": False,
            "roles": {"train": "grouped selection and reference fitting", "validation": "final stopping and diagnostics; not independent calibration", "test": "frozen exploratory evaluation only"},
            "deferred": ["full_cns_training", "envelope_raw_snapshot_and_ablation", "residual_feature_ablation", "extra_long_horizons"],
            "operational_status": "laboratory_only", "status": "planned"}


def sensor_score(model, features, hold_ids):
    """Same physical output transform as runtime, evaluated only on a train fold."""
    cfg = model["config"]
    x, labels, meta = supervised_targets(features[features.unit_id.isin(hold_ids)], cfg["profile"], cfg["horizons"],
        tolerance=cfg["tolerance"], history_min=cfg["history_min"], history_max=cfg["history_max"],
        include_age=cfg["include_age"], reference=model["reference"], multiscale=cfg["multiscale"])
    rows = []
    for j, h in enumerate(cfg["horizons"]):
        if cfg["method"] == "persistence":
            predicted = np.maximum(0., x.current.to_numpy())
            lower = upper = None
        elif cfg["method"] == "causal_local_trend":
            predicted = np.maximum(0., x.current.to_numpy() + x.slope_20.to_numpy() * h)
            lower = upper = None
        else:
            estimates = np.column_stack([model["models"][h][q].predict(x[cfg["feature_names"]]) for q in (.05, .5, .95)])
            lower, predicted, upper = np.maximum(0., np.sort(estimates, axis=1)).T
        for uid, indices in meta.groupby("unit_id").groups.items():
            idx = np.asarray(indices)
            valid = np.isfinite(labels[idx, j])
            actual, pred = labels[idx, j][valid], predicted[idx][valid]
            rows.append({"method": cfg["method"], "horizon": h, "unit_id": uid, "targets": int(valid.sum()),
                "mae": float(np.mean(np.abs(actual - pred))) if valid.any() else None,
                "rmse": float(np.sqrt(np.mean((actual - pred)**2))) if valid.any() else None,
                "interval_coverage": float(np.mean((actual >= lower[idx][valid]) & (actual <= upper[idx][valid]))) if lower is not None and valid.any() else None})
    return pd.DataFrame(rows)


def run_condition_study(job, *, should_stop=None, log=print):
    from pdm.train import run_training
    from pdm.worker import write_status

    study_id = job.get("study_id") or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + uuid.uuid4().hex[:6]
    root = study_root(study_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "study_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        cfg = manifest["config"]
        current = plan_study(cfg)
        if current["data_binding"] != manifest["data_binding"]:
            raise ValueError("Study data identity changed; resume refused")
    else:
        manifest = plan_study(job["config"])
        manifest["study_id"] = study_id
        manifest["plan_hash"] = fingerprint(manifest)
        atomic_write_json(manifest_path, manifest)
    cfg = manifest["config"]
    dataset = cfg["dataset_id"]
    data = load_processed(dataset, manifest["data_binding"]["dataset_version"])
    profile = dataset_profile(dataset)
    atomic_write_json(root / "unit_verification.json", unit_verification(dataset))
    reference = fit_reference(data["features"], manifest["split"]["train"], dataset)
    atomic_write_json(root / "healthy_reference_manifest.json", reference)
    state_policy = default_policy(profile)
    # Statistical thresholds come only from selected candidate-healthy train
    # prefixes. No industrial hard limit is inferred from these statistics.
    scores = []
    for segment in reference["selected_segments"]:
        part = data["features"].loc[lambda f: f.unit_id.eq(segment["unit_id"]) & f.timestamp_s.between(segment["start"], segment["end"])]
        scores.extend(assess_normality(row, reference)["score"] for _, row in part.iterrows())
    if scores:
        state_policy.update(warning_enter=max(3., float(np.quantile(scores, .99))),
                            warning_exit=max(1., float(np.quantile(scores, .90))))
        if state_policy["warning_exit"] >= state_policy["warning_enter"]:
            state_policy["warning_exit"] = .7 * state_policy["warning_enter"]
        state_policy["threshold_provenance"] = "candidate_healthy_train_prefix_99pct_enter_90pct_exit; laboratory_only"
    atomic_write_json(root / "state_policy.json", state_policy)
    new_fits = 0
    max_new = job.get("max_new_fits", cfg["fit_job_budget"])

    def save():
        atomic_write_json(manifest_path, manifest)

    def stop():
        return bool(should_stop and should_stop())

    def can_start():
        return not stop() and new_fits < max_new

    def mark(entry, status, **values):
        entry.update(status=status, **values)
        save()
        write_status({"kind": "condition_study", "status": "training" if status == "running" else status,
                      "study_id": study_id, "job_id": entry["id"], "message": f"{study_id}: {entry['id']} {status}"})

    def selected():
        choices = [j for j in manifest["fit_jobs"] if j["kind"] == "event" and j.get("fold") == 0 and j.get("seed") == 42 and j["status"] == "completed" and j.get("quality_eligible", True)]
        if not choices:
            raise ValueError("No completed train-fold candidate")
        winner = min(choices, key=lambda j: (j["best_metric"], j["id"]))
        return winner["resolved_history"], winner["resolved_recipe"]

    manifest["status"] = "running"
    save()
    for entry in manifest["fit_jobs"]:
        if entry["kind"] != "event" or entry["status"] == "completed":
            continue
        if not can_start():
            break
        mode, recipe = entry["history_mode"], entry["recipe"]
        if mode == "selected" or recipe == "selected":
            chosen_mode, chosen_recipe = selected()
            mode = chosen_mode if mode == "selected" else mode
            recipe = chosen_recipe if recipe == "selected" else recipe
        rid = entry.get("run_id") or f"condition_{study_id}_{entry['id']}"
        saved = dataset_runs(dataset) / rid / "last.pt"
        entry.update(run_id=rid, resolved_history=mode, resolved_recipe=recipe)
        mark(entry, "running")
        new_fits += 1
        fit_protocol = protocol(dataset, mode="adaptive", learning_rate=.0003 if dataset == "filters" else .001,
                                sampling="full_pass", feature_recipe=recipe)
        if dataset == "filters":
            fit_protocol["selection_history_min"] = 60
        # Epoch/stopping rules remain protocol v2, not a shortened accuracy claim.
        result = run_training(dataset, architecture="gru", training_protocol=fit_protocol,
                    history_mode=mode, seed=entry["seed"], device_pref="cpu",
                    split_override=manifest["folds"][entry["fold"]] if entry["fold"] is not None else None,
                    resume_run_id=rid if saved.exists() else None, run_id_override=rid,
                    should_stop=stop, log=log)
        if result["status"] != "completed":
            mark(entry, "paused")
            break
        metrics = read_json(dataset_runs(dataset) / rid / "validation_metrics.json")
        mark(entry, "completed", best_metric=metrics["best_metric"],
             checkpoint_sha256=sha256_file(dataset_runs(dataset) / rid / "best.pt"))
    # Sensor submodels use an explicit ledger; a partial phase restarts its
    # unfinished fitted artifact and records the extra attempts in the budget.
    with threadpool_limits(limits=1):
        for phase in ("screen", "final"):
            jobs = [j for j in manifest["fit_jobs"] if j["kind"] == "sensor" and j["phase"] == phase]
            path = root / f"sensor_{phase}.joblib"
            if phase == "final" and manifest.get("sensor_selected_method") in {"persistence", "causal_local_trend"}:
                # An unselected quantile model does not need a final refit.
                if not path.exists():
                    baseline = fit_sensor(data["features"], manifest["split"]["train"], profile,
                        horizons=cfg["sensor_horizons"], method=manifest["sensor_selected_method"], reference=reference)
                    joblib.dump(baseline, path)
                    for entry in jobs:
                        mark(entry, "completed", fit_attempts=0, skipped="unselected_quantile_candidate", artifact=path.name)
                continue
            if all(j["status"] == "completed" for j in jobs) and path.exists():
                continue
            if stop() or new_fits + len(jobs) > max_new:
                break
            fit_ids = manifest["folds"][0]["train"] if phase == "screen" else manifest["split"]["train"]
            ref = fit_reference(data["features"], fit_ids, dataset)
            def on_fit(h, q):
                nonlocal new_fits
                entry = next(j for j in jobs if j["horizon"] == h and j["quantile"] == q)
                attempts = entry.get("fit_attempts", 0)
                if sum(j.get("fit_attempts", 1 if j["kind"] == "event" and j["status"] == "completed" else 0) for j in manifest["fit_jobs"]) >= cfg["fit_job_budget"]:
                    raise InterruptedError("Fit attempt budget exhausted")
                new_fits += 1
                mark(entry, "running", fit_attempts=attempts + 1)
            try:
                sensor = fit_sensor(data["features"], fit_ids, profile, horizons=cfg["sensor_horizons"],
                         method="multi_horizon_quantile_boosting", tolerance=.01, reference=ref,
                         on_fit=on_fit, should_stop=stop)
            except InterruptedError as exc:
                manifest["pause_reason"] = str(exc)
                break
            joblib.dump(sensor, path)
            for entry in jobs:
                mark(entry, "completed", artifact=path.name, sha256=sha256_file(path))
            if phase == "screen":
                tables = [sensor_score(sensor, data["features"], manifest["folds"][0]["validation"])]
                for method in ("persistence", "causal_local_trend"):
                    baseline = fit_sensor(data["features"], fit_ids, profile, horizons=cfg["sensor_horizons"], method=method, reference=ref)
                    tables.append(sensor_score(baseline, data["features"], manifest["folds"][0]["validation"]))
                comparisons = pd.concat(tables, ignore_index=True)
                comparisons.to_csv(root / "sensor_selection_per_unit.csv", index=False)
                rank = comparisons.groupby("method").mae.mean()
                manifest["sensor_selected_method"] = str(rank.idxmin())
                save()
    if (root / "sensor_final.joblib").exists() and manifest.get("sensor_selected_method"):
        method = manifest["sensor_selected_method"]
        if method != "multi_horizon_quantile_boosting":
            model = fit_sensor(data["features"], manifest["split"]["train"], profile,
                        horizons=cfg["sensor_horizons"], method=method, reference=reference)
            joblib.dump(model, root / "sensor_selected.joblib")
        else:
            import shutil
            shutil.copyfile(root / "sensor_final.joblib", root / "sensor_selected.joblib")
    rows = [{k: j.get(k) for k in ("id", "kind", "status", "run_id", "resolved_history", "resolved_recipe", "seed", "fold", "best_metric")} for j in manifest["fit_jobs"]]
    pd.DataFrame(rows).to_csv(root / "candidate_results.csv", index=False)
    manifest["status"] = "completed" if all(j["status"] == "completed" for j in manifest["fit_jobs"]) else "paused"
    save()
    write_status({"kind": "condition_study", "status": manifest["status"], "study_id": study_id})
    return manifest


def freeze_study(study_id, candidate_id="event_final"):
    root = study_root(study_id)
    manifest = read_json(root / "study_manifest.json")
    entry = next((j for j in manifest["fit_jobs"] if j["id"] == candidate_id), None)
    if not entry or entry["status"] != "completed" or entry.get("fold") is not None:
        raise ValueError("Freeze requires a completed final fit, not an internal selection fold")
    profile = dataset_profile(manifest["config"]["dataset_id"])
    data = load_processed(profile["dataset_id"], manifest["data_binding"]["dataset_version"])
    units = data["units"][data["units"].unit_id.isin(manifest["split"]["validation"])]
    calibration = calibration_report({}, units.unit_id.tolist(), int(units.event_observed.sum()), role="validation_used_for_checkpoint_stopping_and_diagnostics")
    sensor_path = root / "sensor_selected.joblib"
    if not sensor_path.exists():
        raise ValueError("Sensor selection/final fit is incomplete")
    sensor = joblib.load(sensor_path)
    saved_profile = sensor["config"]["profile"]
    if saved_profile != profile:
        corrected = {**saved_profile, "event_convention": "first_measurement_gt_600pa"}
        if profile["dataset_id"] != "filters" or saved_profile.get("event_convention") != "first_measurement_ge_600pa" or corrected != profile:
            raise ValueError("Sensor profile differs from the current verified source convention")
        # Preserve original frozen bytes. Only descriptive source metadata changes;
        # fitted estimators, input columns, horizons and transforms are identical.
        original = sensor_path
        sensor_path = root / "sensor_selected_source_gt.joblib"
        if not sensor_path.exists():
            sensor["config"]["profile"] = profile
            sensor["config"].pop("sensor_model_run_id")
            sensor["config"]["sensor_model_run_id"] = fingerprint(sensor["config"])[:20]
            joblib.dump(sensor, sensor_path)
        atomic_write_json(root / "source_convention_correction.json", {
            "original": original.name, "original_sha256": sha256_file(original),
            "corrected": sensor_path.name, "corrected_sha256": sha256_file(sensor_path),
            "reason": "Source _train_unit_table uses pressure > 600; independent monitoring hard limit uses >= 600",
            "estimators_retrained": False, "additional_fits": 0})
    payload = freeze_bundle(profile=profile, reference=read_json(root / "healthy_reference_manifest.json"),
                quality_policy=quality_policy(profile), state_policy=read_json(root / "state_policy.json"),
                event_run_id=entry["run_id"], sensor_path=sensor_path,
                dataset_version=manifest["data_binding"]["dataset_version"], calibration=calibration,
                selection_refs=[str((root / "candidate_results.csv").relative_to(project_root()))])
    manifest.setdefault("bundles", []).append(payload["bundle_id"])
    atomic_write_json(root / "study_manifest.json", manifest)
    return payload
