"""One immutable synthetic forecasting workflow shared by UI, CLI and workers.

Training never opens Calibration or Test. Child artifacts bind the saved model,
calibrator, dataset and origin; no evaluation can replace or update model weights.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.io_util import atomic_write_json, read_json
from pdm.paths import project_root
from pdm.probabilistic.contract import (
    SUPPORTED_ENGINES,
    TASK,
    canonical_hash,
    default_config,
)
from pdm.projects import _safe_id, project_store

ENGINES = SUPPORTED_ENGINES
SYNTHETIC_KINDS = {"synthetic_sanity": "01_sanity", "synthetic_benchmark": "02_benchmark"}
JOB_KINDS = frozenset({"probabilistic_import", "probabilistic_calibrate", "probabilistic_evaluate", "probabilistic_forecast"})
TREND_QUALITY_POLICY = {"target_point_containment":.90,"marked_undercoverage_below":.85,
                        "positive_persistence_MAE_skill":["all_validation","growing_validation"],
                        "required_earliest_reference_targets":True,"unsupported_targets":"unknown",
                        "interpretation":"exploratory Validation; not production acceptance"}
STEPS = ("Projects", "Import data", "Data Quality", "Training", "Calibration", "Evaluation", "Results")
DEFAULT_SOURCE = project_root() / "data" / "pdm_sensor_csv_v1" / "data"
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "pdm_synthetic_corridor_v1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json(value.tolist())
    if isinstance(value, np.generic):
        return _json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write(path: Path, data: dict) -> None:
    atomic_write_json(path, _json(data))


def _id(prefix: str) -> str:
    return prefix + "-" + uuid.uuid4().hex


def _stop(should_stop) -> None:
    if should_stop and should_stop():
        raise InterruptedError("Synthetic forecasting job cancelled")


def _emit(cb, stage: str, message: str, **extra) -> None:
    if cb:
        cb({"stage": stage, "message": message, **extra})


def _project(project_id: str) -> dict:
    return project_store().get(project_id)


def _path(project_id: str, run_id: str) -> Path:
    _safe_id(run_id, "run ID")
    return project_store().run_path(project_id, run_id)


def _publish(directory: Path, manifest: dict) -> dict:
    """Write terminal manifest once, covering every artifact already on disk."""
    if (directory / "manifest.json").exists():
        raise FileExistsError("An immutable artifact already exists")
    manifest = {**manifest, "created_at": datetime.now(timezone.utc).isoformat(),
                "files": {str(p.relative_to(directory)): _hash(p) for p in sorted(directory.rglob("*")) if p.is_file()}}
    manifest = _json(manifest)
    manifest["artifact_hash"] = canonical_hash(manifest)
    _write(directory / "manifest.json", manifest)
    return {**manifest, "directory": str(directory)}


def _verify(directory: Path) -> dict:
    manifest = read_json(directory / "manifest.json")
    identity = {key:value for key,value in manifest.items() if key != "artifact_hash"}
    if manifest.get("artifact_hash") != canonical_hash(identity):
        raise ValueError("Immutable artifact manifest checksum mismatch")
    for relative, checksum in manifest["files"].items():
        file = directory / relative
        if file.is_symlink() or not file.resolve().is_relative_to(directory.resolve()) or _hash(file) != checksum:
            raise ValueError(f"Immutable artifact checksum mismatch: {relative}")
    return {**manifest, "directory": str(directory)}


def create_project(name: str, suite: str) -> dict:
    if suite not in SYNTHETIC_KINDS.values():
        raise ValueError("Create separate Sanity and Benchmark projects; Stress is evaluation only")
    kind = next(kind for kind, value in SYNTHETIC_KINDS.items() if value == suite)
    return project_store().create(name, kind)


def snapshot_for_project(project_id: str, snapshot_id: str | None = None) -> dict:
    from pdm.probabilistic.data import load_snapshot

    project = _project(project_id)
    requested = snapshot_id or project.get("active_snapshot_id")
    if not requested:
        raise ValueError("Import all four sensor sources before training")
    _safe_id(requested, "snapshot ID")
    from pdm.project_snapshot import snapshot_directory
    directory = snapshot_directory(project_id,requested)
    snapshot = load_snapshot(directory)
    if snapshot["snapshot_id"] != requested or snapshot["config"].get("task") != TASK:
        raise ValueError("Snapshot identity is incompatible with this project")
    return snapshot


def import_observed_project(project_id,source,*,should_stop=None,status_cb=None):
    """One new-import dispatch; source labels select decoding, never the task."""
    project = _project(project_id)
    if source.get("task") != TASK:
        raise ValueError("New sensor imports must declare probabilistic_signal_forecast")
    if project["source_kind"] in SYNTHETIC_KINDS:
        from pdm.probabilistic.contract import default_config
        cfg = default_config()
        if (source.get("signal_column"),source.get("signal_unit"),source.get("cadence_s")) != (cfg["target"],cfg["unit"],cfg["cadence_s"]):
            raise ValueError("Declared signal fields do not match the verified sensor release")
        return import_source_plan(project_id,source,should_stop=should_stop,status_cb=status_cb)
    from pdm.data.project_import import import_project as stage_source
    from pdm.data.project_prepare import decode_imported_project
    from pdm.probabilistic.data import prepare_observations
    previous = project.get("source_manifest") or {}
    _stop(should_stop)
    _emit(status_cb,"import","Verifying observed sources")
    manifest = stage_source(project_id,source,should_stop=should_stop,activate=False)
    known_manifest = Path(source.get("manifest_path") or DEFAULT_ARCHIVE/"data"/"dataset_manifest.json")
    if known_manifest.is_file():
        known = read_json(known_manifest)
        originals = known.get("file_sha256",{})
        by_hash = {digest:name for name,digest in originals.items() if "/sensor_csv/" in name}
        matched = [by_hash[record["sha256"]] for record in manifest["files"] if record["sha256"] in by_hash]
        if matched:
            suites = {name.split("/")[0] for name in matched}
            if len(matched) != len(manifest["files"]) or len(suites) != 1 or "03_stress" in suites:
                raise ValueError("Verified exposed sources cannot mix suites, Stress, or unknown files in a training pool")
            manifest["verified_exposure"] = {"release_id":known.get("release_id") or f"{known['version']}:seed={known['seed']}",
                                            "source_suite":next(iter(suites)),"manifest_sha256":_hash(known_manifest),"files":matched,
                                            "target":known["target"],"unit":known["physical_unit"].split(" (")[0],"cadence_s":known["cadence_s"],"status":"reference_exposed"}
    features,units,quality,split = decode_imported_project(project_id,manifest,should_stop=should_stop)
    _emit(status_cb,"prepare","Preparing four independent whole-unit roles")
    parent = {"snapshot_id":project["active_snapshot_id"]} if project.get("active_snapshot_id") else None
    directory = project_store().project_path(project_id)/"snapshots"/_id("snapshot")
    snapshot = prepare_observations(features,units,split,directory,schema=manifest["signal_schema"],source_manifest=manifest,parent=parent,should_stop=should_stop,endpoint_records=quality.get("endpoint_records",()))
    _stop(should_stop)
    saved_source = {**manifest,"task":TASK,"source_plan":source,"snapshots":{**(previous.get("snapshots") or {}),snapshot["snapshot_id"]:snapshot["directory"]}}
    project_store().update(project_id,source_manifest=saved_source,active_snapshot_id=snapshot["snapshot_id"],selected_run_id=None,state="ready")
    return snapshot


def import_project(project_id: str, source_root: str, *, manifest_path=None, config=None, should_stop=None, status_cb=None) -> dict:
    from pdm.probabilistic.data import prepare_dataset

    project = _project(project_id)
    _stop(should_stop)
    _emit(status_cb, "import", "Verifying original sensor sources and four independent splits")
    root = project_store().project_path(project_id) / "snapshots" / _id("snapshot")
    snapshot = prepare_dataset(source_root, SYNTHETIC_KINDS[project["source_kind"]], root, manifest_path=manifest_path, config=config)
    _stop(should_stop)
    source = {"task": TASK, "source_root": str(Path(source_root).resolve()), "manifest_path": str(manifest_path) if manifest_path else None,
              "suite": snapshot["suite"], "evaluation_status": snapshot["evaluation_status"],
              "snapshots":{**((project.get("source_manifest") or {}).get("snapshots") or {}), snapshot["snapshot_id"]:snapshot["directory"]}}
    project_store().update(project_id, source_manifest=source, active_snapshot_id=snapshot["snapshot_id"], selected_run_id=None, state="ready")
    return snapshot


def import_source_plan(project_id: str, source_plan: dict, *, should_stop=None, status_cb=None) -> dict:
    """The shared Import form's verified source/split adapter, outside inference."""
    from pdm.probabilistic.source_adapter import prepare_source_plan

    project = _project(project_id)
    previous = project.get("source_manifest") or {}
    parent = snapshot_for_project(project_id) if project.get("active_snapshot_id") else None
    suite = SYNTHETIC_KINDS.get(project["source_kind"]) or (parent or {}).get("suite")
    if suite not in SYNTHETIC_KINDS.values() or (parent or {}).get("config", {}).get("observed_profile"):
        raise ValueError("The selected sources require a supported four-role sensor release")
    cfg = dict(parent["config"]) if parent else default_config()
    for field, expected in (("signal_column", cfg["target"]), ("signal_unit", cfg["unit"])):
        if field in source_plan and source_plan[field] != expected:
            raise ValueError(f"{field} must match the selected sensor release: {expected}")
    if source_plan.get("context_mapping") or source_plan.get("age_source", "unknown") != "unknown":
        raise ValueError("The verified sensor release contains only unit_id, timestamp_s and its declared signal column")
    default_label = {"vibration_rms_g": "Vibration RMS"}.get(cfg["target"], cfg["target"])
    label = source_plan.get("signal_label", cfg.get("signal_label", default_label))
    if label != cfg.get("signal_label", default_label):
        cfg["signal_label"] = label
    thresholds = source_plan.get("thresholds")
    if thresholds:
        from pdm.data.project_import import validate_absolute_thresholds
        rule = validate_absolute_thresholds(thresholds)
        if rule["direction"] != "above":
            raise ValueError("The verified vibration release requires increasing Yellow and Red limits")
        cfg.update(yellow=rule["yellow"], red=rule["red"])
    _stop(should_stop)
    _emit(status_cb,"import","Checking selected sources and assigning whole physical units")
    root = project_store().project_path(project_id)
    receipt = root / "source" / _id("import-plan") / "source_manifest.json"
    snapshot = prepare_source_plan(source_plan,suite,root/"snapshots"/_id("snapshot"),config=cfg,
                                   parent={"snapshot_id":parent["snapshot_id"],"dataset_hash":parent["dataset_hash"]} if parent else None,
                                   existing_directories=(previous.get("snapshots") or {}).values(),should_stop=should_stop,receipt_path=receipt)
    _stop(should_stop)
    source = {**previous,"task":TASK,"suite":snapshot["suite"],"evaluation_status":snapshot["evaluation_status"],
              "source_plan":source_plan,"import_receipt":str(receipt),"import_receipt_sha256":_hash(receipt),
              "snapshots":{**(previous.get("snapshots") or {}),snapshot["snapshot_id"]:snapshot["directory"]}}
    # Reusing an unchanged reference keeps its selected frozen model accessible.
    selected = project.get("selected_run_id") if project.get("active_snapshot_id") == snapshot["snapshot_id"] else None
    project_store().update(project_id,source_manifest=source,active_snapshot_id=snapshot["snapshot_id"],selected_run_id=selected,state="ready")
    return snapshot


def list_runs(project_id: str, *, current_snapshot=True) -> list[dict]:
    project = _project(project_id)
    records = []
    for path in sorted((project_store().project_path(project_id) / "runs").glob("*/manifest.json")):
        row = read_json(path)
        if row.get("task") == TASK:
            row = _verify(path.parent)
        if row.get("task") == TASK and row.get("status") == "completed" and (not current_snapshot or row.get("snapshot_id") == project.get("active_snapshot_id")):
            records.append({**row, "directory": str(path.parent)})
    return records


def load_run(project_id: str, run_id: str, *, require_current=True) -> tuple[dict, dict]:
    from pdm.probabilistic.models import load_model

    project = _project(project_id)
    manifest = _verify(_path(project_id, run_id))
    if manifest.get("task") != TASK or manifest.get("project_id") != project_id or manifest.get("run_id") != run_id or manifest.get("status") != "completed":
        raise ValueError("Invalid probabilistic model bundle")
    if require_current and manifest["snapshot_id"] != project.get("active_snapshot_id"):
        raise ValueError("This model belongs to an inactive snapshot")
    model = load_model(Path(manifest["directory"]) / "model")
    if model["model_hash"] != manifest["model_hash"] or model["config"] != manifest["config"] or model["training_summary"] != manifest["training_summary"]:
        raise ValueError("Model/config/selection summary differs from run binding")
    return manifest, model


def train_run(project_id: str, snapshot_id: str | None, engine_id: str, params: dict, *, should_stop=None, status_cb=None, checkpoint_directory=None) -> dict:
    from pdm.probabilistic.contract import default_config
    from pdm.probabilistic.data import load_split
    from pdm.probabilistic.models import fit_model, save_model
    from pdm.probabilistic.windows import build_windows

    if engine_id not in ENGINES:
        raise ValueError("Unsupported probabilistic engine")
    snapshot = snapshot_for_project(project_id, snapshot_id)
    for key in ("target","unit","schema","cadence_s","clock_tolerance_s","positive_domain","imported_rule"):
        if key in params and params[key] != snapshot["config"].get(key):
            raise ValueError(f"Training cannot replace the admitted physical profile: {key}")
    from pdm.data.project_import import validate_absolute_thresholds
    from pdm.project_snapshot import load_project_limits
    config = default_config(**{**snapshot["config"],**params,"engine_id":engine_id})
    bounded = config.get("forecast_mode") == "bounded_trend_v1"
    if bounded:
        if params.get("decision_quality_policy",TREND_QUALITY_POLICY) != TREND_QUALITY_POLICY:
            raise ValueError("Trend corridor quality policy must match the predeclared Validation criteria")
        config = default_config(**{**config,"decision_quality_policy":TREND_QUALITY_POLICY})
    _stop(should_stop)
    train_frame = load_split(snapshot,"train")
    if config.get("horizon_protocol") == "dense-v2":
        from pdm.probabilistic.horizons import train_horizon_profile
        horizon_profile = train_horizon_profile(train_frame,config)
        if config["max_horizon"] > horizon_profile["horizon"]:
            raise ValueError("Forecast grid exceeds admitted Train horizon capacity")
        if params.get("training_horizon_profile") is not None and params["training_horizon_profile"] != horizon_profile:
            raise ValueError("Train horizon provenance differs from the queued configuration")
        config = default_config(**{**config,"training_horizon_profile":horizon_profile})
    rule = load_project_limits(project_id,snapshot["snapshot_id"])
    if params.get("zone_rule_hash") is not None:
        requested_rule = validate_absolute_thresholds(params.get("zone_rule"))
        committed = rule or snapshot["config"].get("imported_rule")
        if canonical_hash(requested_rule) != params["zone_rule_hash"] or requested_rule != committed:
            raise ValueError("Saved Limits changed after this training job was queued; queue a new job")
    if rule or config.get("observed_profile"):
        rule = validate_absolute_thresholds(rule or config.get("imported_rule"))
        config = default_config(**{**config,"yellow":rule["yellow"],"red":rule["red"],"threshold_direction":rule["direction"],"zone_rule":rule,"zone_rule_hash":canonical_hash(rule)})
    _stop(should_stop)
    _emit(status_cb, "windows", "Preparing Train and Validation windows; holdouts remain closed")
    train = build_windows(train_frame, config, mode="train")
    validation_frame = load_split(snapshot,"validation")
    validation = build_windows(validation_frame, config, mode="validation")
    _stop(should_stop)
    run_id = _id("prob")
    directory = _path(project_id, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / "origin_manifest.json", {"train":train.get("origin_manifest"), "validation":validation.get("origin_manifest")})
    _write(directory / "admission_report.json", {"train":train.get("admission"), "validation":validation.get("admission")})
    _write(directory / "config.json", config)
    reference_validation = None
    if bounded:
        reference_validation = build_windows(validation_frame,config,mode="reference")
        _write(directory / "validation_reference_origins.json",reference_validation["origin_manifest"])
    fit_options = {"reference_validation_windows":reference_validation} if bounded else {}
    if config.get("training_population_protocol") == "balanced_rolling_reference_v1":
        reference_train = build_windows(train_frame,config,mode="reference")
        _write(directory / "train_reference_origins.json",reference_train["origin_manifest"])
        fit_options["reference_train_windows"] = reference_train
    model = fit_model(train, validation, config, should_stop=should_stop, status_cb=status_cb, checkpoint_directory=checkpoint_directory,**fit_options)
    _stop(should_stop)
    save_model(model, directory / "model")
    controls = validation_diagnostics(model, validation,reference_validation=reference_validation)
    _stop(should_stop)
    _write(directory / "validation_controls.json", controls)
    for name, value in {"config":config, "dataset_manifest":snapshot.get("dataset_manifest", {}), "split_manifest":snapshot.get("split_manifest", {}),
                        "training_summary":model["training_summary"], "environment":environment()}.items():
        _write(directory / (name + ".json"), value)
    (directory / "execution.log").write_text("Training opened only Train and Validation.\n", encoding="utf-8")
    selection = ("Selection uses the frozen rolling/earliest-reference Validation center objective. "
                 "Bounded corridor usefulness is measured empirically; width compliance is not quality acceptance."
                 if bounded else "Selection uses raw unit-balanced Validation pinball.")
    (directory / "report.md").write_text("# Training run\n\n"+selection+" Calibration and Test are separate actions.\n", encoding="utf-8")
    manifest = _publish(directory, {"run_id":run_id, "project_id":project_id, "snapshot_id":snapshot["snapshot_id"], "dataset_hash":snapshot["dataset_hash"],
                                   "release_id":snapshot["release_id"], "suite":snapshot["suite"], "task":TASK, "engine_id":engine_id, "status":"completed",
                                   "model_hash":model["model_hash"], "config":config, "training_summary":model["training_summary"], "validation_controls":controls, "evaluation_status":snapshot["evaluation_status"]})
    project_store().update(project_id, selected_run_id=run_id)
    return manifest



def validation_diagnostics(model: dict, windows: dict, *, reference_validation=None) -> dict:
    """Raw Validation controls; rising slice is declared from observed history only."""
    from pdm.probabilistic.evaluation import evaluate

    dense_v2 = model["config"].get("horizon_protocol") == "dense-v2"
    if dense_v2:
        from pdm.probabilistic.evaluation import raw_validation_metrics
        raw_metrics = raw_validation_metrics(model,windows)
    else:
        raw_metrics = evaluate(model, None, windows, protocol="rolling")["raw_metrics"]
    threshold = .5*model["config"]["scale_floor"] if model["config"].get("observed_profile") else .01
    keep = np.asarray(windows["x"])[:, -1] - np.asarray(windows["x"])[:, 0] >= threshold
    policy = f"observed last minus first history value >= {threshold:g} {model['config']['unit']}" if model["config"].get("observed_profile") else "observed last minus first history value >= 0.01 g"
    controls = {"all_validation":raw_metrics, "growing_policy":policy, "growing_origins":int(keep.sum())}
    if keep.any():
        selected = {}
        for key, value in windows.items():
            if isinstance(value, np.ndarray) and len(value) == len(keep):
                selected[key] = value[keep]
            elif key == "past_status" and isinstance(value, list) and len(value) == len(keep):
                selected[key] = [v for v, admitted in zip(value, keep, strict=True) if admitted]
            else:
                selected[key] = value
        controls["growing_validation"] = (raw_validation_metrics(model,selected) if dense_v2 else
                                          evaluate(model,None,selected,protocol="rolling")["raw_metrics"])
    else:
        controls["growing_validation"] = {"status":"not_evaluable", "reason":"no_observably_growing_validation_history"}
    if model["config"].get("forecast_mode") == "bounded_trend_v1":
        if reference_validation is None:
            raise ValueError("Trend corridor diagnostics require independent earliest-reference Validation windows")
        controls["earliest_reference_validation"] = raw_validation_metrics(model,reference_validation)
        policy = model["config"]["decision_quality_policy"]
        all_metrics,growing = controls["all_validation"],controls["growing_validation"]
        reference = controls["earliest_reference_validation"]
        def known(value):
            return value is not None and np.isfinite(value)
        containment = all_metrics.get("point_coverage")
        skills = [all_metrics.get("skill_persistence"),growing.get("skill_persistence")]
        supported = bool(controls["growing_origins"] and reference.get("known_targets",0) and
                         known(containment) and all(known(value) for value in skills))
        target_met = supported and containment >= policy["target_point_containment"] and all(value > 0 for value in skills)
        undercoverage = known(containment) and containment < policy["marked_undercoverage_below"]
        controls["usefulness_diagnostic"] = {"policy":policy,"status":"validation_target_met" if target_met else "validation_quality_unaccepted" if supported else "validation_quality_unknown",
                                              "marked_undercoverage":bool(undercoverage),"target_met":bool(target_met),
                                              "quality_accepted":False,"interpretation":"exploratory Validation; not production acceptance"}
    return controls

def environment() -> dict:
    import importlib.metadata

    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project_root(), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unavailable"
    sources = {str(p.relative_to(project_root())): _hash(p) for p in sorted((project_root() / "src" / "pdm" / "probabilistic").glob("*.py"))}
    versions = {}
    for package in ("numpy", "pandas", "torch", "scikit-learn", "streamlit"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return {"python":sys.version, "platform":platform.platform(), "commit":commit, "source_hashes":sources, "versions":versions,
            "determinism":"Seed and model settings recorded; cross-device bitwise equality is not claimed", "round_trip_tolerance":1e-6}


def _children(project_id: str, run_id: str, kind: str, *, require_current=True) -> list[dict]:
    load_run(project_id, run_id, require_current=require_current)
    rows = []
    for path in sorted((_path(project_id, run_id) / kind).glob("*/manifest.json")):
        manifest = _verify(path.parent)
        if manifest.get("run_id") != run_id or manifest.get("project_id") != project_id:
            raise ValueError("Child artifact has incompatible project/run identity")
        if manifest.get("status") == "completed":
            rows.append({**manifest, "directory":str(path.parent)})
    return sorted(rows, key=lambda x:x["created_at"])


def list_calibrations(project_id: str, run_id: str, *, require_current=True) -> list[dict]:
    return _children(project_id, run_id, "calibrations", require_current=require_current)


def list_evaluations(project_id: str, run_id: str, *, require_current=True) -> list[dict]:
    return _children(project_id, run_id, "evaluations", require_current=require_current)



def load_evaluation(project_id: str, run_id: str, evaluation_id: str) -> dict:
    _safe_id(evaluation_id, "evaluation ID")
    run, _model = load_run(project_id, run_id)
    saved = _verify(_path(project_id, run_id) / "evaluations" / evaluation_id)
    if saved.get("run_id") != run_id or saved.get("project_id") != project_id or saved.get("evaluation_id") != evaluation_id or saved.get("model_hash") != run["model_hash"] or saved.get("status") != "completed":
        raise ValueError("Evaluation belongs to another frozen bundle")
    return saved

def load_calibration(project_id: str, run_id: str, calibration_id: str, *, require_current=True) -> tuple[dict, dict]:
    from pdm.probabilistic.calibration import load_calibrator

    _safe_id(calibration_id, "calibration ID")
    run, model = load_run(project_id, run_id, require_current=require_current)
    saved = _verify(_path(project_id, run_id) / "calibrations" / calibration_id)
    calibrator = load_calibrator(Path(saved["directory"]) / "calibration.json", model=model)
    if any(saved.get(key) != value for key,value in {"project_id":project_id,"run_id":run_id,"snapshot_id":run["snapshot_id"],"calibration_id":calibration_id,"status":"completed","model_hash":run["model_hash"],"calibrator_hash":calibrator["calibrator_hash"]}.items()):
        raise ValueError("Calibration belongs to another frozen parent bundle")
    if saved.get("summary") != calibrator:
        raise ValueError("Calibration summary does not match the saved calibrator")
    snapshot = snapshot_for_project(project_id,run["snapshot_id"])
    physical = set(snapshot["split_manifest"]["calibration"]["physical_units"])
    from pdm.probabilistic.contract import reference_origin_policy
    if calibrator.get("calibration_snapshot_hash") != snapshot["dataset_hash"] or not set(calibrator.get("calibration_physical_units",[])).issubset(physical) or calibrator.get("origin_policy") != reference_origin_policy(model["config"]):
        raise ValueError("Calibration source scope does not match its frozen snapshot")
    return saved, calibrator


def calibrate_run(project_id: str, run_id: str, *, should_stop=None, status_cb=None) -> dict:
    from pdm.probabilistic.calibration import fit_calibrator, save_calibrator
    from pdm.probabilistic.data import load_split
    from pdm.probabilistic.windows import build_windows

    run, model = load_run(project_id, run_id)
    _stop(should_stop)
    message = ("Independent Calibration: empirical containment/support assessment; fixed corridor bounds stay unchanged"
               if model["config"].get("forecast_mode") == "bounded_trend_v1" else
               "Independent Calibration: fixed causal anchors and complete prefix support per horizon"
               if model["config"].get("horizon_protocol") == "dense-v2" else
               "Independent Calibration: one complete reference path per physical unit")
    _emit(status_cb, "calibration", message)
    snapshot = snapshot_for_project(project_id, run["snapshot_id"])
    windows = build_windows(load_split(snapshot, "calibration"), model["config"], mode="reference")
    identifier = _id("cal")
    directory = _path(project_id, run_id) / "calibrations" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / "origin_manifest.json", windows.get("origin_manifest", {}))
    _write(directory / "admission_report.json", windows.get("admission", {}))
    calibrator = fit_calibrator(model, windows, model["config"])
    _stop(should_stop)
    save_calibrator(calibrator, directory)
    return _publish(directory, {"calibration_id":identifier, "run_id":run_id, "project_id":project_id, "snapshot_id":run["snapshot_id"],
                               "model_hash":run["model_hash"], "calibrator_hash":calibrator["calibrator_hash"], "status":"completed",
                               "calibration_status":calibrator.get("status"), "n":calibrator.get("n"), "scope":calibrator.get("scope"), "summary":calibrator})


def freeze_bundle(project_id: str, run_id: str, calibration_id: str | None, *, protocol: str, evaluation_snapshot: dict | None = None) -> dict:
    from pdm.probabilistic.contract import reference_origin_policy
    run, model = load_run(project_id, run_id)
    calibrator = load_calibration(project_id, run_id, calibration_id)[1] if calibration_id else None
    return {"task":TASK, "run_id":run_id, "project_id":project_id, "snapshot_id":run["snapshot_id"], "training_dataset_hash":run["dataset_hash"],
            "model_hash":model["model_hash"], "calibration_id":calibration_id, "calibrator_hash":calibrator["calibrator_hash"] if calibrator else None,
            "config":model["config"], "protocol":protocol, "origin_policy":reference_origin_policy(model["config"]) if protocol != "rolling" else "causal-issued-every-5-observations", "evaluation_dataset_hash":evaluation_snapshot["dataset_hash"] if evaluation_snapshot else run["dataset_hash"],
            "frozen_at":datetime.now(timezone.utc).isoformat(), "selection":("Frozen rolling/earliest-reference Validation center objective; Test never selects configurations" if model["config"].get("forecast_mode") == "bounded_trend_v1" else "Raw Validation pinball only; Test never selects configurations")}


def _compatible(model: dict, snapshot: dict) -> None:
    config = snapshot.get("config") or snapshot.get("profile") or {}
    if not isinstance(config, dict):
        config = {}
    manifest = snapshot.get("dataset_manifest") or {}
    for key in ("target", "cadence_s"):
        found = config.get(key, manifest.get(key))
        if found is not None and found != model["config"].get(key):
            raise ValueError(f"External snapshot incompatible {key}")
    for key in ("unit", "positive_domain"):
        if key in config and config[key] != model["config"].get(key):
            raise ValueError(f"External snapshot incompatible {key}")



def evaluator_metadata(snapshot: dict, path: str) -> pd.DataFrame:
    """Verify evaluator-only labels against this release before joining IDs."""
    expected = snapshot["dataset_manifest"]["file_sha256"].get("units_EVALUATION_ONLY.csv")
    if not expected or _hash(Path(path)) != expected:
        raise ValueError("Evaluator metadata checksum is incompatible with this data release")
    meta = pd.read_csv(path)
    if not {"unit_id", "suite", "split", "mechanism"}.issubset(meta.columns):
        raise ValueError("Evaluator metadata schema is incomplete")
    meta = meta.loc[(meta.suite == snapshot["suite"]) & (meta.split == "test")].copy()
    if meta.unit_id.duplicated().any() or set(meta.unit_id) != set(snapshot["split_manifest"]["test"]["units"]):
        raise ValueError("Evaluator metadata unit identities do not match the evaluation release")
    physical = snapshot["release_id"] + ":" + meta.unit_id.astype(str)
    if "physical_unit_id" in meta and not (meta.physical_unit_id == physical).all():
        raise ValueError("Evaluator metadata physical release namespace mismatch")
    meta["physical_unit_id"] = physical
    return meta

def evaluate_run(project_id: str, run_id: str, calibration_id: str | None, *, protocol="reference", source_root=None, manifest_path=None, suite="03_stress", metadata_path=None, should_stop=None, status_cb=None) -> dict:
    from pdm.probabilistic.data import load_split, prepare_external
    from pdm.probabilistic.evaluation import evaluate
    from pdm.probabilistic.windows import build_windows

    if protocol not in {"reference", "rolling", "external"}:
        raise ValueError("Reference, rolling and external evaluation have separate scopes")
    run, model = load_run(project_id, run_id)
    calibrator = load_calibration(project_id, run_id, calibration_id)[1] if calibration_id else None
    _stop(should_stop)
    identifier = _id("eval")
    directory = _path(project_id, run_id) / "evaluations" / identifier
    directory.mkdir(parents=True, exist_ok=False)
    if protocol == "external":
        if not source_root:
            raise ValueError("External evaluation requires an explicit external sensor source")
        snapshot = prepare_external(source_root, suite, directory / "external_snapshots", manifest_path=manifest_path, config=model["config"])
        _compatible(model, snapshot)
    else:
        if source_root or manifest_path:
            raise ValueError("A different dataset is permitted only through explicit external evaluation")
        snapshot = snapshot_for_project(project_id, run["snapshot_id"])
    freeze = freeze_bundle(project_id, run_id, calibration_id, protocol=protocol, evaluation_snapshot=snapshot)
    if metadata_path:
        freeze["metadata"] = {"path":str(Path(metadata_path).resolve()), "sha256":_hash(Path(metadata_path)), "release_id":snapshot["release_id"], "access":"deferred_until_predictions"}
    _write(directory / "freeze_manifest.json", freeze)
    _emit(status_cb, "evaluation", f"Evaluating frozen bundle in separate {protocol} scope")
    windows = build_windows(load_split(snapshot, "test"), model["config"], mode="rolling" if protocol == "rolling" else "reference")
    _write(directory / "origin_manifest.json", windows.get("origin_manifest", {}))
    _write(directory / "admission_report.json", windows.get("admission", {}))
    _stop(should_stop)
    # Metadata is joined by the evaluator only after its numerical predictions.
    metadata = (lambda: evaluator_metadata(snapshot, metadata_path)) if metadata_path else None
    report = evaluate(model, calibrator, windows, protocol=protocol, metadata=metadata, observed_stream=lambda: load_split(snapshot, "test"))
    _stop(should_stop)
    for name, value in report.items():
        if isinstance(value, pd.DataFrame):
            if name == "predictions":
                value = value.assign(run_id=run_id, evaluation_id=identifier, calibrator_id=calibration_id)
                value.to_parquet(directory / "predictions.parquet", index=False)
            else:
                value.to_csv(directory / (name + ".csv"), index=False)
        else:
            _write(directory / (name + ".json"), value)
    report["metrics_by_horizon"].to_csv(directory / "trajectory_metrics.csv", index=False)
    if protocol == "rolling":
        report["rolling_table"].to_csv(directory / "rolling_metrics.csv", index=False)
    _write(directory / "environment.json", environment())
    interpretation = ("Fixed-width decision corridors have no nominal coverage guarantee. Point containment and center error are measured empirically over supported observations; unsupported targets remain unknown. Whole-path containment and RED geometry are separate diagnostics. All issued corridors remain in aggregate metrics."
                      if model["config"].get("forecast_mode") == "bounded_trend_v1" else
                      "Nominal 90% refers to the Calibration reference population. Rolling origins and external sources have their own measured scope. All wide forecasts remain in aggregate metrics.")
    (directory / "report.md").write_text(f"# {protocol} evaluation\n\nFrozen model: {run['model_hash']}\n\n{interpretation}\n", encoding="utf-8")
    return _publish(directory, {"evaluation_id":identifier, "run_id":run_id, "project_id":project_id, "snapshot_id":run["snapshot_id"], "model_hash":run["model_hash"],
                               "calibrator_hash":freeze["calibrator_hash"], "calibration_id":calibration_id, "status":"completed", "protocol":protocol,
                               "suite":snapshot["suite"], "release_id":snapshot["release_id"], "dataset_hash":snapshot["dataset_hash"],
                               "evaluation_status":snapshot["evaluation_status"], "summary":report.get("summary", {})})


def save_forecast(project_id: str, run_id: str, calibration_id: str | None, unit_id: str, origin_s: float, *, split="test", should_stop=None, status_cb=None) -> dict:
    from pdm.probabilistic.calibration import apply_calibrator, band_availability
    from pdm.probabilistic.data import load_split
    from pdm.probabilistic.models import predict
    from pdm.probabilistic.windows import observed_history

    if split not in {"train", "validation", "calibration", "test"}:
        raise ValueError("Unknown observation source")
    run, model = load_run(project_id, run_id, require_current=False)
    calibrator = load_calibration(project_id, run_id, calibration_id, require_current=False)[1] if calibration_id else None
    snapshot = snapshot_for_project(project_id, run["snapshot_id"])
    history = observed_history(load_split(snapshot, split), unit_id, float(origin_s), model["config"])
    _stop(should_stop)
    if history["status"] != "available":
        raise ValueError(history["status"])
    binding = {"model_hash":model["model_hash"], "calibrator_hash":calibrator["calibrator_hash"] if calibrator else None,
               "release_id":snapshot["release_id"], "physical_unit_id":history["physical_unit_id"], "unit_id":unit_id,
               "origin_s":float(origin_s), "history":np.asarray(history["x"]).tolist(), "config":model["config"]}
    identifier = "fc-" + hashlib.sha256(json.dumps(_json(binding), sort_keys=True).encode()).hexdigest()[:32]
    directory = _path(project_id, run_id) / "forecasts" / identifier
    if (directory / "manifest.json").exists():
        return _verify(directory)
    _emit(status_cb, "forecast", "Saving a forecast from observations available at Now")
    raw = predict(model, history)
    if model["config"].get("horizon_protocol") == "dense-v2":
        bands = {str(H):apply_calibrator(raw,calibrator,H,model_hash=model["model_hash"],config=model["config"])
                 for H in model["config"]["report_horizons"]}
    else:
        bands = {str(H):apply_calibrator(raw, calibrator, H, model_hash=model["model_hash"]) for H in (5, 15, 30, 60)}
    for band in bands.values():
        band["availability"] = band_availability(band, model["config"])
    _stop(should_stop)
    directory.mkdir(parents=True, exist_ok=False)
    outputs = {"decision_outputs":raw,"output_kind":"decision_corridor","coverage_guarantee":False,"nominal":None} if model["config"].get("forecast_mode") == "bounded_trend_v1" else {"raw_quantiles":raw}
    _write(directory / "forecast.json", {**binding, **outputs, "bands":bands, "past_status":history["past_status"]})
    return _publish(directory, {"forecast_id":identifier, "run_id":run_id, "project_id":project_id, "snapshot_id":run["snapshot_id"],
                               "calibration_id":calibration_id, "model_hash":model["model_hash"], "calibrator_hash":binding["calibrator_hash"],
                               "unit_id":unit_id, "origin_s":float(origin_s), "split":split, "status":"completed"})


def load_forecast(project_id: str, run_id: str, forecast_id: str) -> tuple[dict, dict]:
    from pdm.probabilistic.data import load_split
    from pdm.probabilistic.windows import observed_history

    _safe_id(forecast_id,"forecast ID")
    run,model = load_run(project_id,run_id,require_current=False)
    manifest = _verify(_path(project_id,run_id)/"forecasts"/forecast_id)
    if any(manifest.get(key) != value for key,value in {"forecast_id":forecast_id,"status":"completed","run_id":run_id,"project_id":project_id,"snapshot_id":run["snapshot_id"],"model_hash":run["model_hash"]}.items()):
        raise ValueError("Forecast belongs to another frozen run snapshot")
    body = read_json(Path(manifest["directory"])/"forecast.json")
    snapshot = snapshot_for_project(project_id,run["snapshot_id"])
    cal_id = manifest.get("calibration_id")
    calibrator = load_calibration(project_id,run_id,cal_id,require_current=False)[1] if cal_id else None
    cal_hash = calibrator["calibrator_hash"] if calibrator else None
    history = observed_history(load_split(snapshot,manifest["split"]),manifest["unit_id"],manifest["origin_s"],model["config"])
    if manifest.get("calibrator_hash") != cal_hash or any(body.get(key) != value for key,value in {"model_hash":run["model_hash"],"calibrator_hash":cal_hash,"unit_id":manifest["unit_id"],"origin_s":manifest["origin_s"],"config":model["config"],"release_id":snapshot["release_id"],"physical_unit_id":history["physical_unit_id"]}.items()):
        raise ValueError("Forecast body disagrees with its frozen manifest/model/calibrator")
    if history["status"] != "available" or body.get("history") != np.asarray(history["x"]).tolist() or body.get("past_status") != history["past_status"]:
        raise ValueError("Forecast causal history disagrees with its saved source")
    binding = {key:body[key] for key in ("model_hash","calibrator_hash","release_id","physical_unit_id","unit_id","origin_s","history","config")}
    expected = "fc-"+hashlib.sha256(json.dumps(_json(binding),sort_keys=True).encode()).hexdigest()[:32]
    if forecast_id != expected:
        raise ValueError("Forecast content-derived identity mismatch")
    return manifest,body


def validate_job(job: dict) -> None:
    if job.get("kind") not in JOB_KINDS:
        raise ValueError("Unsupported synthetic worker action")
    _project(job["project_id"])
    if job["kind"] == "probabilistic_import":
        if job.get("source_plan") is not None:
            from pdm.probabilistic.source_adapter import validate_source_plan

            validate_source_plan(job["source_plan"])
        elif not job.get("source_root"):
            raise ValueError("Import requires a source root")
    elif not job.get("run_id"):
        raise ValueError("Synthetic worker action requires a frozen run")
    if job["kind"] == "probabilistic_forecast" and (not isinstance(job.get("unit_id"), str) or not isinstance(job.get("origin_s"), (int,float)) or not np.isfinite(job["origin_s"])):
        raise ValueError("Replay requires unit and origin")


def dispatch_job(job: dict, *, should_stop=None, status_cb=None) -> dict:
    validate_job(job)
    common = {"should_stop":should_stop, "status_cb":status_cb}
    kind = job["kind"]
    if kind == "probabilistic_import":
        if job.get("source_plan") is not None:
            return import_source_plan(job["project_id"],job["source_plan"],**common)
        return import_project(job["project_id"], job["source_root"], manifest_path=job.get("manifest_path"), config=job.get("config"), **common)
    if kind == "probabilistic_calibrate":
        return calibrate_run(job["project_id"], job["run_id"], **common)
    if kind == "probabilistic_evaluate":
        return evaluate_run(job["project_id"], job["run_id"], job.get("calibration_id"), protocol=job.get("protocol", "reference"), source_root=job.get("source_root"),
                            manifest_path=job.get("manifest_path"), suite=job.get("suite", "03_stress"), metadata_path=job.get("metadata_path"), **common)
    return save_forecast(job["project_id"], job["run_id"], job.get("calibration_id"), job["unit_id"], job["origin_s"], split=job.get("split", "test"), **common)
