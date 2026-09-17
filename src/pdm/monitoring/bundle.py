"""Immutable bundle identities and hash-checked dependency loading."""
from __future__ import annotations

import platform
from pathlib import Path

import joblib

from pdm.data.prepare import load_processed, resolve_processed_dir
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.monitoring.calibration import verify_calibration
from pdm.monitoring.policy import validate_policy
from pdm.paths import dataset_runs, project_root, runs_root
from pdm.splits import assert_split_coverage
from pdm.train import _source_commit, load_trained_model
from pdm.training_protocol import fingerprint


def code_identity():
    root = project_root()
    return fingerprint({str(p.relative_to(root)): sha256_file(p) for p in sorted((root / "src" / "pdm").rglob("*.py"))})


def bundle_root(bundle_id):
    if Path(bundle_id).name != bundle_id:
        raise ValueError("Invalid bundle identifier")
    return runs_root() / "monitoring_bundles" / bundle_id


def freeze_bundle(*, profile, reference, quality_policy, state_policy, event_run_id,
                  sensor_path, dataset_version, calibration, selection_refs=None):
    validate_policy(state_policy, profile)
    data = load_processed(profile["dataset_id"], dataset_version)
    assert_split_coverage(data["units"].copy(), data["split"])
    dataset_dir = resolve_processed_dir(profile["dataset_id"], dataset_version)
    dependencies = {}
    def bind(path):
        path = Path(path).resolve()
        dependencies[str(path.relative_to(project_root()))] = sha256_file(path)
    for name in ("features.parquet", "units.parquet", "split.json", "processed_fingerprint.json", "quality_records.parquet"):
        bind(dataset_dir / name)
    if event_run_id:
        rdir = dataset_runs(profile["dataset_id"]) / event_run_id
        model, prep, meta = load_trained_model(rdir)
        del model
        run_binding = read_json(rdir / "dataset_fingerprint.json")
        if run_binding["dataset_version"] != dataset_version:
            raise ValueError("Event model dataset version differs from monitoring data")
        if read_json(rdir / "split.json").get("test") != data["split"].get("test"):
            raise ValueError("Cannot publish an internal-fold event model as final bundle")
        for name in ("best.pt", "preprocessing.json", "split.json", "experiment_snapshot.json", "dataset_fingerprint.json"):
            bind(rdir / name)
        if (rdir / "interval_profile.json").exists():
            bind(rdir / "interval_profile.json")
        for path in sorted((rdir / "connectome").glob("*")):
            if path.is_file():
                bind(path)
        history = prep.history_policy or {"mode": f"fixed_{meta['history_length']}"}
    else:
        history = None
    sensor_id = None
    for rel in selection_refs or []:
        bind(project_root() / rel)
    if sensor_path:
        bind(sensor_path)
        sensor = joblib.load(sensor_path)
        sensor_id = sensor["config"]["sensor_model_run_id"]
        if sensor["config"]["profile"] != profile:
            raise ValueError("Sensor profile mismatch")
        if set(sensor["config"]["fit_units"]) != set(data["split"]["train"]):
            raise ValueError("Final sensor must be fit only on the complete admitted train split")
    identity = {"dependencies": dependencies, "profile": profile, "reference": reference,
                "quality_policy": quality_policy, "state_policy": state_policy, "history_policy": history,
                "event_model_run_id": event_run_id, "sensor_model_run_id": sensor_id,
                "dataset_version": dataset_version}
    if calibration is not None:
        calibration = {**calibration, "identity": identity, "identity_hash": fingerprint(identity)}
    payload = {"schema_version": "monitoring_bundle_v1", **identity, "calibration_profile": calibration,
               "sensor_path": str(Path(sensor_path).resolve().relative_to(project_root())) if sensor_path else None,
               "source_commit": _source_commit(), "source_worktree_hash": code_identity(),
               "environment": {"python": platform.python_version(), "platform": platform.platform()},
               "selection_references": selection_refs or [], "frozen": True,
               "operational_status": "laboratory_only", "test_status": "exploratory_reused_holdout"}
    payload["bundle_id"] = fingerprint(payload)[:24]
    root = bundle_root(payload["bundle_id"])
    root.mkdir(parents=True, exist_ok=True)
    path = root / "monitoring_bundle.json"
    if path.exists() and read_json(path) != payload:
        raise ValueError("An immutable bundle already exists")
    atomic_write_json(path, payload)
    for filename, value in (("healthy_reference_manifest.json", reference), ("state_policy.json", state_policy),
                             ("calibration_report.json", calibration)):
        atomic_write_json(root / filename, value)
    from pdm.monitoring.contracts import unit_verification

    atomic_write_json(root / "unit_verification.json", unit_verification(profile["dataset_id"]))
    return payload


def load_bundle(bundle_id, *, load_models=True):
    payload = read_json(bundle_root(bundle_id) / "monitoring_bundle.json")
    saved = dict(payload)
    actual_id = saved.pop("bundle_id")
    if actual_id != bundle_id or fingerprint(saved)[:24] != bundle_id or not payload.get("frozen"):
        raise ValueError("Monitoring bundle identity changed after freeze")
    for rel, sha in payload["dependencies"].items():
        path = (project_root() / rel).resolve()
        if not path.is_relative_to(project_root()) or not path.is_file() or sha256_file(path) != sha:
            raise ValueError(f"Monitoring dependency hash mismatch: {rel}")
    validate_policy(payload["state_policy"], payload["profile"])
    identity = {k: payload[k] for k in ("dependencies", "profile", "reference", "quality_policy", "state_policy",
                "history_policy", "event_model_run_id", "sensor_model_run_id", "dataset_version")}
    if payload["calibration_profile"]:
        verify_calibration(payload["calibration_profile"], identity)
    predictor = sensor = None
    if load_models:
        if payload["event_model_run_id"]:
            from pdm.predict import Predictor

            model, prep, meta = load_trained_model(dataset_runs(payload["profile"]["dataset_id"]) / payload["event_model_run_id"])
            predictor = Predictor(model, prep, meta["history_length"], "cpu")
            interval = dataset_runs(payload["profile"]["dataset_id"]) / payload["event_model_run_id"] / "interval_profile.json"
            if interval.exists():
                from pdm.forecasting import load_interval_profile

                predictor.interval_profile = load_interval_profile(interval.parent)
        if payload["sensor_path"]:
            sensor = joblib.load(project_root() / payload["sensor_path"])
    return payload, predictor, sensor


def runtime_args(payload, predictor=None, sensor=None):
    return {"profile": payload["profile"], "reference": payload["reference"],
            "quality_policy": payload["quality_policy"], "state_policy": payload["state_policy"],
            "bundle_id": payload["bundle_id"], "predictor": predictor, "sensor_model": sensor}
