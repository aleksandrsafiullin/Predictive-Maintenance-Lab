"""Registration protocol tests use initialized checkpoints, never model training."""
from __future__ import annotations

import copy

import pytest
import torch

from pdm import learned_candidate_import as importing
from pdm import learned_trajectory as learned
from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json, read_json
from pdm.learned_candidate_import import register_learned_candidate
from pdm.signal_training import list_project_runs, load_signal_run
from tests.project_contract import make_contract_snapshot


def _bytes(directory):
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def _candidate(tmp_path, monkeypatch, engine="gru", saved_contract=True):
    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    store, project, ref = make_contract_snapshot(root)
    pid, sid = project["project_id"], ref["snapshot_id"]
    data = load_snapshot(pid, sid, store=store)
    config = learned.learned_params(engine, {"history_length": 2, "horizons_s": [10., 20.],
        "hidden_size": 16, "num_layers": 1, "rank": 2, "dropout": 0.,
        "epochs": 1, "cpu_threads": 1, "path_samples": 16}, data["features"])
    external = engine == "quantile_boosting"
    model = learned._make_model(engine, config, 2, 3 if external else None)
    bundle = {"model": model, "config": config, "scaler": {"mean": [0., 1.], "std": [1., 2.],
        "fit_units": data["split"]["train"]}, "external_scaler": {"mean": [0., 0., 0.],
        "std": [1., 1., 1.]} if external else None, "selection": {"test_feedback": False,
        "criterion": "validation_joint_objective", "best_epoch": 1},
        "feature_names": ["signal", "time"], "trace": [{"epoch": 1, "validation": {"total": 1.25}}],
        "encoder": {"kind": "quantile_boosting", "readout_policy": "saved_tree_features"} if external else None}
    directory = tmp_path / "retained-candidate"
    artifacts = learned.save_learned_bundle(bundle, directory)
    minimal = {"dir": str(directory), "engine_id": engine, "params": config,
        "scaler": bundle["scaler"], "artifacts": artifacts,
        **learned._event_distribution_metadata(model, config)}
    frozen = {"project_id": pid, "snapshot_id": sid, "engine_id": engine,
        "config": config, "fingerprint": data["fingerprint"], "test_access": False}
    metrics = {"evaluation_partition": "Validation", "physical_group_count": 2,
        "origins": 8, "horizons": [], "quality_target_met": False}
    atomic_write_json(directory / "frozen_candidate.json", frozen)
    atomic_write_json(directory / "status.json", {"stage": "completed", "selection": bundle["selection"]})
    atomic_write_json(directory / "validation_metrics.json", metrics)
    atomic_write_json(directory / "reload_verification.json", {"passed": True, "test_access": False})
    if saved_contract:
        atomic_write_json(directory / "candidate_artifact_contract.json", minimal)
    # Import is permitted to reconstruct saved state, never fit or infer.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Registration must not train or infer")
    monkeypatch.setattr(learned, "fit_learned_model", forbidden)
    monkeypatch.setattr(learned, "predict_learned", forbidden)
    return store, pid, directory, bundle, minimal, metrics


@pytest.mark.parametrize("engine", ["gru", "lstm", "quantile_boosting"])
def test_registration_reconstructs_canonical_bundle_and_preserves_originals(tmp_path, monkeypatch, engine):
    store, pid, source, bundle, minimal, metrics = _candidate(tmp_path, monkeypatch, engine)
    originals, record = _bytes(source), store.get(pid)
    run = register_learned_candidate(pid, source, training_created_at="2026-10-01T00:00:00Z")
    canonical = load_signal_run(pid, run["run_id"])
    reloaded = learned.load_learned_bundle(canonical)
    assert canonical["dir"] == store.run_path(pid, run["run_id"])
    assert canonical["params"] == bundle["config"]
    assert canonical["scaler"] == bundle["scaler"]
    assert canonical["selection"] == bundle["selection"]
    assert canonical["metrics"] == {"validation": metrics}
    assert canonical["created_at"] == "2026-10-01T00:00:00Z"
    assert canonical["reload_verified"] is True
    for name, value in bundle["model"].state_dict().items():
        assert torch.equal(value, reloaded["model"].state_dict()[name])
    for name, digest in minimal["artifacts"].items():
        assert canonical["artifacts"][name] == digest
        assert (canonical["dir"] / name).read_bytes() == originals[name]
    assert reloaded["encoder"] == bundle["encoder"]
    assert _bytes(source) == originals
    assert store.get(pid) == record
    published = _bytes(canonical["dir"])
    again = register_learned_candidate(pid, source, training_created_at="2026-10-01T00:00:00Z")
    assert again["run_id"] == run["run_id"]
    assert len(list_project_runs(pid)) == 1
    assert _bytes(canonical["dir"]) == published
    assert _bytes(source) == originals


def test_trusted_hashes_support_candidate_without_saved_contract(tmp_path, monkeypatch):
    store, pid, source, _, minimal, _ = _candidate(tmp_path, monkeypatch, saved_contract=False)
    originals = _bytes(source)
    run = register_learned_candidate(pid, source, trusted_loader_manifest=minimal)
    learned.load_learned_bundle(load_signal_run(pid, run["run_id"]))
    assert _bytes(source) == originals
    assert store.get(pid)["selected_run_id"] is None


@pytest.mark.parametrize("defect,match", [
    ("project", "project/snapshot"), ("fingerprint", "fingerprint"),
    ("unsafe_name", "Unsafe"), ("symlink", "symlink"),
    ("trusted_hash", "hash mismatch"), ("test_metrics", "Validation"),
    ("external_encoder", "verified artifact"),
])
def test_registration_rejects_untrusted_or_misbound_candidates(tmp_path, monkeypatch, defect, match):
    store, pid, source, _, minimal, _ = _candidate(tmp_path, monkeypatch,
        "quantile_boosting" if defect == "external_encoder" else "gru")
    trusted = None
    if defect in {"project", "fingerprint"}:
        frozen = read_json(source / "frozen_candidate.json")
        frozen["project_id" if defect == "project" else "fingerprint"] = "another-project-or-fingerprint"
        atomic_write_json(source / "frozen_candidate.json", frozen)
    elif defect == "unsafe_name":
        minimal["artifacts"]["../outside.pt"] = "0" * 64
        atomic_write_json(source / "candidate_artifact_contract.json", minimal)
    elif defect == "symlink":
        checkpoint = source / "checkpoint.pt"
        outside = tmp_path / "outside.pt"
        checkpoint.rename(outside)
        checkpoint.symlink_to(outside)
    elif defect == "trusted_hash":
        trusted = copy.deepcopy(minimal)
        trusted["artifacts"]["checkpoint.pt"] = "0" * 64
    elif defect == "test_metrics":
        atomic_write_json(source / "validation_metrics.json", {"evaluation_partition": "Test"})
    else:
        del minimal["artifacts"]["encoder.joblib"]
        atomic_write_json(source / "candidate_artifact_contract.json", minimal)
    originals = _bytes(source)
    with pytest.raises(ValueError, match=match):
        register_learned_candidate(pid, source, trusted_loader_manifest=trusted)
    assert list_project_runs(pid) == []
    assert _bytes(source) == originals
    assert store.get(pid)["selected_run_id"] is None


def test_idempotent_import_refuses_to_overwrite_tampered_published_run(tmp_path, monkeypatch):
    _, pid, source, _, _, _ = _candidate(tmp_path, monkeypatch)
    originals = _bytes(source)
    run = register_learned_candidate(pid, source)
    path = run["dir"] / "model_input_contract.json"
    path.write_text(path.read_text() + "\n")
    tampered = _bytes(run["dir"])
    with pytest.raises(ValueError, match="hash mismatch"):
        register_learned_candidate(pid, source)
    assert _bytes(run["dir"]) == tampered
    assert _bytes(source) == originals


def test_legacy_metrics_require_explicit_trusted_validation_partition(tmp_path, monkeypatch):
    _, pid, source, _, _, metrics = _candidate(tmp_path, monkeypatch)
    del metrics["evaluation_partition"]
    atomic_write_json(source / "validation_metrics.json", metrics)
    originals = _bytes(source)
    with pytest.raises(ValueError, match="Validation"):
        register_learned_candidate(pid, source)
    run = register_learned_candidate(pid, source, trusted_evaluation_partition="Validation")
    assert run["metrics"] == {"validation": {**metrics, "evaluation_partition": "Validation"}}
    assert _bytes(source) == originals


def test_explicit_test_metric_partition_cannot_be_overridden(tmp_path, monkeypatch):
    _, pid, source, _, _, metrics = _candidate(tmp_path, monkeypatch)
    metrics["evaluation_partition"] = "Test"
    atomic_write_json(source / "validation_metrics.json", metrics)
    with pytest.raises(ValueError, match="Validation"):
        register_learned_candidate(pid, source, trusted_evaluation_partition="Validation")
    assert list_project_runs(pid) == []


def test_full_candidate_requires_original_complete_connectome_provenance(tmp_path, monkeypatch):
    _, pid, source, _, minimal, _ = _candidate(tmp_path, monkeypatch, "quantile_boosting")
    frozen = read_json(source / "frozen_candidate.json")
    frozen["engine_id"] = minimal["engine_id"] = "full_cns"
    atomic_write_json(source / "frozen_candidate.json", frozen)
    atomic_write_json(source / "candidate_artifact_contract.json", minimal)
    with pytest.raises(ValueError, match="complete real connectome provenance"):
        register_learned_candidate(pid, source)
    assert list_project_runs(pid) == []


def test_staging_is_never_discoverable_as_a_completed_run(tmp_path, monkeypatch):
    store, pid, source, _, _, _ = _candidate(tmp_path, monkeypatch)
    validate = importing._load_signal_run_directory
    observed = []

    def observe(project_id, run_id, directory):
        assert directory.parent == store.project_path(pid)
        assert directory.name.startswith('.signal-import-staging-')
        assert run_id.startswith('signal-import-retained-candidate-')
        assert list_project_runs(pid) == []
        assert store.get(pid)['selected_run_id'] is None
        observed.append(directory)
        return validate(project_id, run_id, directory)

    monkeypatch.setattr(importing, '_load_signal_run_directory', observe)
    run = register_learned_candidate(pid, source)
    assert len(observed) == 1 and not observed[0].exists()
    assert [row['run_id'] for row in list_project_runs(pid)] == [run['run_id']]


def test_failed_staging_validation_leaves_project_and_originals_unchanged(tmp_path, monkeypatch):
    store, pid, source, _, _, _ = _candidate(tmp_path, monkeypatch)
    originals, record = _bytes(source), store.get(pid)

    def fail(*args):
        raise ValueError('staged verification failed')

    monkeypatch.setattr(importing, '_load_signal_run_directory', fail)
    with pytest.raises(ValueError, match='staged verification failed'):
        register_learned_candidate(pid, source)
    assert list_project_runs(pid) == []
    assert not list(store.project_path(pid).glob('.signal-import-staging-*'))
    assert store.get(pid) == record
    assert _bytes(source) == originals


def test_postpublication_error_preserves_verified_model_for_recovery(tmp_path, monkeypatch):
    store, pid, source, _, _, _ = _candidate(tmp_path, monkeypatch)
    originals, record = _bytes(source), store.get(pid)

    def fail(*args):
        raise OSError('postpublication read failed')

    monkeypatch.setattr(importing, 'load_signal_run', fail)
    with pytest.raises(OSError, match='postpublication read failed'):
        register_learned_candidate(pid, source)
    runs = list_project_runs(pid)
    assert len(runs) == 1
    loaded = load_signal_run(pid, runs[0]['run_id'])
    learned.load_learned_bundle(loaded)
    assert loaded['dir'].is_dir()
    assert store.get(pid) == record
    assert _bytes(source) == originals
