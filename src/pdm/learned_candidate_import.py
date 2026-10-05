"""Register retained experimental learned bundles without fitting or scoring them."""
from __future__ import annotations

import errno
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.learned_trajectory import MODE, _digest, _event_distribution_metadata, load_learned_bundle
from pdm.projects import project_store
from pdm.signal_training import _load_signal_run_directory, load_signal_run

_REQUIRED = {"checkpoint.pt", "learned_model.json", "objective_trace.json"}
_EVIDENCE = ("frozen_candidate.json", "status.json", "validation_metrics.json",
             "candidate_artifact_contract.json", "reload_verification.json")


def _regular(path: Path) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Candidate file must be a regular non-symlink file: {path}")


def _verify_hashes(directory: Path, artifacts: Mapping) -> None:
    if not isinstance(artifacts, Mapping) or not _REQUIRED.issubset(artifacts):
        raise ValueError("Candidate requires checkpoint, metadata and objective trace hashes")
    for name, digest in artifacts.items():
        if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
            raise ValueError("Unsafe candidate artifact name")
        _regular(directory / name)
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Candidate artifact hash mismatch: {name}")


def _timestamp(value: str | None, directory: Path) -> tuple[str, str]:
    if value is not None:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("Training timestamp must include a timezone")
        source = "explicit historical timestamp"
    else:
        parsed = datetime.fromtimestamp((directory / "status.json").stat().st_mtime, timezone.utc)
        source = "historical completed status file modification time"
    if parsed > datetime.now(timezone.utc):
        raise ValueError("Historical training timestamp cannot be in the future")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), source


def register_learned_candidate(
    project_id: str,
    candidate_path: str | Path,
    *,
    trusted_loader_manifest: Mapping | None = None,
    training_created_at: str | None = None,
    trusted_evaluation_partition: str | None = None,
) -> dict:
    """Import one existing candidate; leave project snapshot and selection untouched.

    ``trusted_loader_manifest`` is an already trusted minimal saved-bundle manifest,
    e.g. a retained best_models.json model's loader_manifest. Its hashes are checked
    against original files before any deserialization. No inference or historical
    fit-versus-reload comparison is performed; reload_verified means current state
    reconstruction succeeded. The return value is the standard loaded run.
    Legacy metrics without a partition require operator-verified attribution via
    ``trusted_evaluation_partition="Validation"``; explicit conflicts are rejected.
    Staging remains outside discoverable runs until the standard directory
    validator and learned state loader succeed. Published runs are preserved if
    final canonical verification fails, so they remain available for recovery.
    """
    directory = Path(candidate_path).absolute()
    if any(path.is_symlink() for path in (directory, *directory.parents)):
        raise ValueError("Candidate path cannot contain symlinks")
    if not directory.is_dir():
        raise ValueError(f"Candidate directory missing: {directory}")
    for name in ("frozen_candidate.json", "status.json", "validation_metrics.json", "learned_model.json"):
        _regular(directory / name)
    frozen = read_json(directory / "frozen_candidate.json")
    status = read_json(directory / "status.json")
    metadata = read_json(directory / "learned_model.json")
    metrics = read_json(directory / "validation_metrics.json")
    if status.get("stage", status.get("status")) != "completed":
        raise ValueError("Candidate is not completed")
    if frozen.get("project_id") != project_id or not frozen.get("snapshot_id"):
        raise ValueError("Frozen candidate project/snapshot binding missing or mismatched")
    if frozen.get("config") != metadata.get("config"):
        raise ValueError("Frozen candidate config differs from saved model")
    if trusted_evaluation_partition not in (None, "Validation"):
        raise ValueError("Trusted evaluation partition must be Validation")
    if "evaluation_partition" in metrics:
        if metrics["evaluation_partition"] != "Validation":
            raise ValueError("Only saved Validation metrics may be imported; explicit partition conflict")
        partition_source = "explicit saved validation_metrics.json evaluation_partition"
    elif trusted_evaluation_partition == "Validation":
        metrics = {**metrics, "evaluation_partition": "Validation"}
        partition_source = "operator verified historical candidate/register; legacy metrics missing partition field"
    else:
        raise ValueError("Legacy metrics missing evaluation_partition require trusted Validation attribution")
    if frozen.get("test_access") is not False or metadata.get("selection", {}).get("test_feedback") is not False:
        raise ValueError("Candidate must declare no Test access or selection feedback")
    if status.get("selection") is not None and status["selection"] != metadata["selection"]:
        raise ValueError("Completed status selection differs from saved model")

    data = load_snapshot(project_id, frozen["snapshot_id"], feature_partitions=("train", "validation"))
    if frozen.get("fingerprint") != data["fingerprint"]:
        raise ValueError("Frozen candidate fingerprint differs from bound snapshot")
    engine_id = frozen.get("engine_id")
    if engine_id not in {"gru", "lstm", "quantile_boosting", "full_cns"}:
        raise ValueError("Unsupported retained learned engine")
    if metadata["config"].get("forecast_mode") != MODE:
        raise ValueError("Candidate is not a learned joint trajectory model")
    contract_path = directory / "candidate_artifact_contract.json"
    if contract_path.exists():
        _regular(contract_path)
        minimal = read_json(contract_path)
    elif trusted_loader_manifest is not None:
        minimal = dict(trusted_loader_manifest)
    else:
        raise ValueError("Candidate requires a saved artifact contract or trusted loader manifest")
    _verify_hashes(directory, minimal.get("artifacts"))
    if trusted_loader_manifest is not None:
        trusted = dict(trusted_loader_manifest)
        _verify_hashes(directory, trusted.get("artifacts"))
        if trusted["artifacts"] != minimal["artifacts"]:
            raise ValueError("Trusted artifact hashes differ from candidate contract")
        for key in ("engine_id", "params", "scaler"):
            if trusted.get(key) != minimal.get(key):
                raise ValueError(f"Trusted {key} differs from candidate contract")
        if "connectome" in trusted:
            if minimal.get("connectome") not in (None, trusted["connectome"]):
                raise ValueError("Trusted connectome differs from candidate contract")
            minimal["connectome"] = trusted["connectome"]
    if (minimal.get("engine_id") != engine_id or minimal.get("params") != metadata["config"]
            or minimal.get("scaler") != metadata["scaler"]):
        raise ValueError("Candidate loader config, scaler or engine differs from frozen model")
    if engine_id == "full_cns":
        provenance = minimal.get("connectome", {})
        if (provenance.get("graph_mode") != "real_connectome" or provenance.get("is_synthetic") is not False
                or provenance.get("node_sampling") is not False or not provenance.get("graph_hash")):
            raise ValueError("Full candidate requires original complete real connectome provenance")
    loaded = load_learned_bundle({**minimal, "dir": directory})
    config = metadata["config"]
    event_metadata = _event_distribution_metadata(loaded["model"], config)
    created_at, timestamp_source = _timestamp(training_created_at, directory)
    source_hashes = dict(minimal["artifacts"])
    for name in _EVIDENCE:
        if (directory / name).exists():
            _regular(directory / name)
            source_hashes[name] = sha256_file(directory / name)
    identity = _digest({"project_id": project_id, "candidate": directory.name, "source_hashes": source_hashes,
                        "connectome": minimal.get("connectome"), "created_at": created_at})
    label = re.sub(r"[^a-zA-Z0-9_-]", "-", directory.name)[:40] or "candidate"
    run_id = f"signal-import-{label}-{identity[:20]}"
    store = project_store()
    destination = store.run_path(project_id, run_id)
    if destination.exists():
        existing = load_signal_run(project_id, run_id)
        load_learned_bundle(existing)
        if existing.get("import_provenance", {}).get("source_identity_sha256") != identity:
            raise ValueError("Existing import run has different provenance")
        return existing
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".signal-import-staging-", dir=store.project_path(project_id)))
    published = False
    try:
        for name in source_hashes:
            if name in {"checkpoint.pt", "encoder.joblib"}:
                try:
                    os.link(directory / name, stage / name)
                except OSError as exc:
                    if exc.errno != errno.EXDEV:
                        raise
                    shutil.copy2(directory / name, stage / name)
            else:
                shutil.copy2(directory / name, stage / name)
        _verify_hashes(stage, source_hashes)
        funnel = {"mode": MODE, "probabilistic_architecture_claim": True,
                  "nominal_coverage": config["nominal_coverage"], "path_samples": config["path_samples"],
                  "operational_coverage_approved": False}
        contract = {"project_id": project_id, "snapshot_id": frozen["snapshot_id"], "engine_id": engine_id,
                    "params": config, "schema": data["schema"], "snapshot_fingerprint_sha256": _digest(data["fingerprint"]),
                    "scaler": metadata["scaler"], "funnel": funnel, **event_metadata}
        if engine_id == "full_cns":
            contract["connectome"] = minimal["connectome"]
        atomic_write_json(stage / "training_contract.json", contract)
        atomic_write_json(stage / "model_input_contract.json", {
            "features": metadata["feature_names"], "fit_partition": "Train", "architecture": engine_id,
            "encoder_adapter": loaded["encoder"]["readout_policy"] if engine_id == "quantile_boosting"
            else "complete source connectome plus learned probabilistic readout" if engine_id == "full_cns"
            else "recurrent learned probabilistic encoder",
            "targets": "joint future nonnegative RMS path and first recorded-grid RED entry",
            "selection_partition": "Validation", "test_feedback": False,
            "unknown_suffix_policy": "masked path targets and censored RED likelihood; never negative padding",
            "normalization": "Train-only physical group balanced",
            "warning_distribution": "first entry mixture with horizon survival class",
            "phase_covariance": loaded["model"].phase_covariance,
            "path_head_outputs": loaded["model"].path_head[-1].out_features,
            "coverage_guarantee": False, **event_metadata,
        })
        artifacts = {name: sha256_file(stage / name) for name in (*source_hashes, "training_contract.json", "model_input_contract.json")}
        manifest = {**contract, "schema_version": "project_signal_forecast_learned_v1", "task": "signal_forecast",
                    "status": "completed", "run_id": run_id, "artifact": "checkpoint.pt", "artifacts": artifacts,
                    "selection": metadata["selection"], "interval_status": "learned_uncalibrated_simultaneous_band",
                    "created_at": created_at, "metrics": {"validation": metrics}, "reload_verified": True,
                    "accepted_for_production": False, "coverage_guarantee": False,
                    "import_provenance": {"kind": "historical_experimental_candidate", "candidate": directory.name,
                        "source_directory": str(directory), "source_identity_sha256": identity, "source_artifact_hashes": source_hashes,
                        "imported_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        "training_timestamp_source": timestamp_source,
                        "reload_verification_scope": "current artifact hashes and loaded model/encoder state; no inference equivalence audit",
                        "metrics_source": "saved Validation only; reused experimental evidence; no new evaluation",
                        "evaluation_partition_attribution": partition_source}}
        atomic_write_json(stage / "manifest.json", manifest)
        load_learned_bundle(_load_signal_run_directory(project_id, run_id, stage))
        if destination.exists():
            raise FileExistsError(f"Import destination already exists: {destination}")
        stage.rename(destination)
        published = True
        result = load_signal_run(project_id, run_id)
        load_learned_bundle(result)
        return result
    except BaseException:
        if not published:
            shutil.rmtree(stage)
        raise
