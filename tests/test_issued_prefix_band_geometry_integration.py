"""Synthetic engineering evidence only; no production data or quality claims."""
import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pdm import learned_trajectory as learned
from pdm.io_util import sha256_file
from pdm.models.signal_distribution import SignalDistribution
from tests.test_post_factor_smoothness_integration import config as base_config
from tests.test_post_factor_smoothness_integration import frame

CONTRACT = "path_band_geometry_contract"


def config(engine="gru", **overrides):
    return base_config(engine, **{"post_factor_smoothness_weight": 0.0,
                                 "path_band_geometry": "issued_prefix", **overrides})


def save_run(bundle, directory, full=False):
    artifacts = learned.save_learned_bundle(bundle, directory)
    fields = learned._event_distribution_metadata(bundle["model"], bundle["config"])
    for name in ("training_contract.json", "model_input_contract.json"):
        (directory / name).write_text(json.dumps(fields))
        artifacts[name] = sha256_file(directory / name)
    run = {"dir": directory, "engine_id": "gru", "params": bundle["config"],
           "scaler": bundle["scaler"], "artifacts": artifacts, **copy.deepcopy(fields)}
    if full:
        run.update(schema_version=1, selection=copy.deepcopy(bundle["selection"]))
    return run


def assert_models_equal(a, b):
    for key, value in a["model"].state_dict().items():
        assert torch.equal(value, b["model"].state_dict()[key]), key
    for key, value in learned.predict_learned(a, frame(.1)).items():
        np.testing.assert_array_equal(value, learned.predict_learned(b, frame(.1))[key])


@pytest.mark.parametrize("value", [None, True, False, 0, 1, [], {}, "", "issued", "Issued_prefix", "issued_prefix ", np.nan])
def test_strict_selector_config_validation(value):
    with pytest.raises(ValueError, match="path_band_geometry"):
        config(path_band_geometry=value)


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_legacy_missing_default_trace_rng_weights_and_predictions_exact(engine, tmp_path):
    explicit = config(engine, path_band_geometry="observed_prefix")
    missing = {k: v for k, v in explicit.items() if k != "path_band_geometry"}
    features = pd.DataFrame({"unit_id": ["s"] * 5, "timestamp_s": np.arange(5) * 60., "gap_before": False})
    resolved = learned.learned_params(engine, missing, features)
    assert resolved == explicit
    bundles, rng = [], []
    for cfg in (missing, explicit, resolved):
        bundles.append(learned.fit_learned_model(engine, frame(), frame(.1), cfg))
        rng.append((torch.random.get_rng_state().clone(), copy.deepcopy(np.random.get_state())))
    for other, state in zip(bundles[1:], rng[1:]):
        assert bundles[0]["trace"] == other["trace"]
        assert bundles[0]["selection"] == other["selection"]
        assert torch.equal(rng[0][0], state[0])
        assert rng[0][1][0] == state[1][0]
        np.testing.assert_array_equal(rng[0][1][1], state[1][1])
        assert rng[0][1][2:] == state[1][2:]
        assert_models_equal(bundles[0], other)
    bundle = bundles[0]
    artifacts = learned.save_learned_bundle(bundle, tmp_path)
    metadata = json.loads((tmp_path / "learned_model.json").read_text())
    assert CONTRACT not in metadata and CONTRACT not in metadata["selection"]
    assert "path_band_geometry" not in metadata
    loaded = learned.load_learned_bundle({"dir": tmp_path, "engine_id": engine,
                                        "params": missing, "artifacts": artifacts})
    assert_models_equal(bundle, loaded)


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_positive_propagation_checkpoint_exact_reload_and_unchanged_sampler(engine, tmp_path, monkeypatch):
    models, snapshots, calls = [], [], []
    original_factory, objective = learned._make_model, SignalDistribution.objective
    def factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        models.append(model)
        return model
    def spy(self, *args, **kwargs):
        calls.append((self.training, kwargs.copy()))
        return objective(self, *args, **kwargs)
    monkeypatch.setattr(learned, "_make_model", factory)
    monkeypatch.setattr(SignalDistribution, "objective", spy)
    def report(record):
        if record["stage"] == "training":
            snapshots.append(copy.deepcopy(models[0].state_dict()))
    cfg = config(engine, min_delta=1e10)
    bundle = learned.fit_learned_model(engine, frame(), frame(.1), cfg, report=report)
    positive_calls = calls.copy()
    assert positive_calls and all(c[1]["path_band_geometry"] == "issued_prefix" for c in positive_calls)
    assert {c[0] for c in positive_calls} == {True, False}
    assert bundle["selection"]["best_epoch"] == 1
    assert bundle["selection"]["best_score"] == bundle["trace"][0]["validation"]["total"]
    assert len(snapshots) == 3 and bundle["selection"]["restored_best_checkpoint"]
    assert any(not torch.equal(snapshots[0][k], snapshots[-1][k]) for k in snapshots[0])
    for k, value in bundle["model"].state_dict().items():
        assert torch.equal(value, snapshots[0][k])
    assert not bundle["selection"]["test_feedback"]
    fields = learned._event_distribution_metadata(bundle["model"], cfg)
    assert fields[CONTRACT] == bundle["selection"][CONTRACT]
    assert fields[CONTRACT]["inference_change"] is False
    artifacts = learned.save_learned_bundle(bundle, tmp_path)
    loaded = learned.load_learned_bundle({"dir": tmp_path, "engine_id": engine, "params": cfg, "artifacts": artifacts})
    assert_models_equal(bundle, loaded)
    changed = {**bundle, "config": {**cfg, "path_band_geometry": "observed_prefix"}}
    for k, value in learned.predict_learned(bundle, frame(.1)).items():
        np.testing.assert_array_equal(value, learned.predict_learned(changed, frame(.1))[k])
    legacy_model = original_factory(engine, changed["config"], 2, None)
    assert legacy_model.state_dict().keys() == bundle["model"].state_dict().keys()
    assert sum(p.numel() for p in legacy_model.parameters()) == sum(p.numel() for p in bundle["model"].parameters())
    calls.clear()
    learned.fit_learned_model(engine, frame(), frame(.1), changed["config"])
    assert len(calls) == len(positive_calls)
    assert [c[0] for c in calls] == [c[0] for c in positive_calls]
    for (_, old), (_, positive) in zip(calls, positive_calls):
        assert "path_band_geometry" not in old
        assert old.keys() == positive.keys() - {"path_band_geometry"}
        for key in old:
            if isinstance(old[key], torch.Tensor):
                assert torch.equal(old[key], positive[key]), key
            elif key != "generator":
                assert old[key] == positive[key], key


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_unknown_future_nan_inf_do_not_change_fit_or_validation(engine):
    cfg = config(engine)
    baseline = learned.fit_learned_model(engine, frame(), frame(.1), cfg)
    train, validation = frame(), frame(.1)
    for data in (train, validation):
        data["y"][~data["mask"]] = np.resize([np.nan, np.inf, -np.inf], (~data["mask"]).sum())
    other = learned.fit_learned_model(engine, train, validation, cfg)
    assert baseline["trace"] == other["trace"]
    assert baseline["selection"] == other["selection"]
    assert_models_equal(baseline, other)


@pytest.fixture(scope="module")
def positive_bundle():
    return learned.fit_learned_model("gru", frame(), frame(.1), config())


@pytest.mark.parametrize("surface", ["learned_model.json", "manifest", "training_contract.json", "model_input_contract.json", "selection"])
@pytest.mark.parametrize("operation", ["missing", "tampered"])
def test_rehashed_contract_surface_rejected(positive_bundle, tmp_path, surface, operation):
    run = save_run(positive_bundle, tmp_path)
    target = "learned_model.json" if surface == "selection" else surface
    saved = run if target == "manifest" else json.loads((tmp_path / target).read_text())
    parent = saved["selection"] if surface == "selection" else saved
    if operation == "missing":
        del parent[CONTRACT]
    else:
        parent[CONTRACT]["band_source"] = "target-dependent masked paths"
    if target != "manifest":
        (tmp_path / target).write_text(json.dumps(saved))
        run["artifacts"][target] = sha256_file(tmp_path / target)
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


def test_all_contract_fields_mandatory(positive_bundle, tmp_path):
    expected = positive_bundle["selection"][CONTRACT]
    for key in expected:
        run = save_run(positive_bundle, tmp_path)
        saved = json.loads((tmp_path / "learned_model.json").read_text())
        del saved[CONTRACT][key]
        (tmp_path / "learned_model.json").write_text(json.dumps(saved))
        run["artifacts"]["learned_model.json"] = sha256_file(tmp_path / "learned_model.json")
        with pytest.raises(ValueError):
            learned.load_learned_bundle(run)


@pytest.mark.parametrize("mutation", ["missing", "best_epoch", "contract"])
def test_full_run_selection_required(positive_bundle, tmp_path, mutation):
    run = save_run(positive_bundle, tmp_path, full=True)
    assert learned.load_learned_bundle(run)["selection"] == run["selection"]
    if mutation == "missing":
        del run["selection"]
    elif mutation == "best_epoch":
        run["selection"]["best_epoch"] += 1
    else:
        run["selection"][CONTRACT]["geometry"] = "observed_prefix"
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize("name", ["training_contract.json", "model_input_contract.json"])
@pytest.mark.parametrize("field,wrong", [("path_band_geometry", "observed_prefix"), ("nominal_coverage", .5), ("path_objective_horizons_s", [180.])])
def test_contract_duplicate_params_bound(positive_bundle, tmp_path, name, field, wrong):
    run = save_run(positive_bundle, tmp_path)
    saved = json.loads((tmp_path / name).read_text())
    saved["params"] = copy.deepcopy(positive_bundle["config"])
    for valid in (True, False):
        if not valid:
            saved["params"][field] = wrong
        (tmp_path / name).write_text(json.dumps(saved))
        run["artifacts"][name] = sha256_file(tmp_path / name)
        if valid:
            learned.load_learned_bundle(run)
        else:
            with pytest.raises(ValueError):
                learned.load_learned_bundle(run)


def test_fields_only_and_minimal_saved_contracts_legal(positive_bundle, tmp_path):
    run = save_run(positive_bundle, tmp_path)
    assert_models_equal(positive_bundle, learned.load_learned_bundle(run))
    minimal = {k: run[k] for k in ("dir", "engine_id", "params", "artifacts")}
    assert_models_equal(positive_bundle, learned.load_learned_bundle(minimal))


@pytest.mark.parametrize("surface", ["learned_model.json", "selection"])
def test_legacy_unexpected_contract_rejected(positive_bundle, tmp_path, surface):
    bundle = learned.fit_learned_model("gru", frame(), frame(.1), config(path_band_geometry="observed_prefix"))
    run = save_run(bundle, tmp_path)
    saved = json.loads((tmp_path / "learned_model.json").read_text())
    parent = saved["selection"] if surface == "selection" else saved
    parent[CONTRACT] = copy.deepcopy(positive_bundle["selection"][CONTRACT])
    (tmp_path / "learned_model.json").write_text(json.dumps(saved))
    run["artifacts"]["learned_model.json"] = sha256_file(tmp_path / "learned_model.json")
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


def test_actual_isolated_synthetic_worker_accepts_selector_and_binds_contracts(tmp_path, monkeypatch):
    from pdm.cli import spawn_worker
    from pdm.data.project_import import import_project
    from pdm.data.project_prepare import prepare_project
    from pdm.projects import ProjectStore
    from pdm.signal_training import load_signal_run
    from pdm.worker import job_path, read_status

    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("PDM_WORKER_ROOT", str(tmp_path / "worker"))
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame([{"unit_id": f"synthetic-{u}", "timestamp_s": float(t * 60),
                   "rms": .2 + u * .11 + t * .12}
                  for u in range(8) for t in range(12)]).to_csv(source / "synthetic.csv", index=False)
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Synthetic issued-prefix engineering QA", "generic_sensor_csv")
    manifest = import_project(project["project_id"], {
        "primary": {"mode": "folder", "path": str(source)},
        "signal_column": "rms", "signal_label": "Synthetic RMS", "signal_unit": "g",
        "thresholds": {"mode": "absolute", "direction": "above", "yellow": .9, "red": 1.5},
        "seed": 19}, store=store)
    snapshot = prepare_project(project["project_id"], manifest["manifest_id"], store=store)
    params = {"forecast_mode": learned.MODE, "history_length": 2,
              "horizons_s": [60., 120., 180.], "path_objective_horizons_s": [60., 180.],
              "epochs": 2, "hidden_size": 16, "num_layers": 1, "rank": 2,
              "batch_size": 16, "max_windows_per_unit": 5, "path_samples": 16,
              "training_samples": 8, "cpu_threads": 1, "seed": 927,
              "event_distribution": "finite_horizon_mixture",
              "path_band_geometry": "issued_prefix"}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast",
                            "job_id": "isolated-issued-prefix-synthetic-gru",
                            "project_id": project["project_id"], "snapshot_id": snapshot["snapshot_id"],
                            "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=60) == 0, read_status()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed", state
    assert json.loads(job_path().read_text())["params"]["path_band_geometry"] == "issued_prefix"
    run = load_signal_run(project["project_id"], state["run_id"])
    assert run["reload_verified"] and run["params"]["path_band_geometry"] == "issued_prefix"
    expected = run[CONTRACT]
    assert run["selection"][CONTRACT] == expected
    for name in ("learned_model.json", "training_contract.json", "model_input_contract.json"):
        saved = json.loads((run["dir"] / name).read_text())
        assert saved[CONTRACT] == expected
        assert sha256_file(run["dir"] / name) == run["artifacts"][name]
    assert run["selection"]["test_feedback"] is False
    learned.load_learned_bundle(run)


@pytest.mark.parametrize("omissions", [("training_contract.json",), ("model_input_contract.json",),
                                      ("training_contract.json", "model_input_contract.json")])
def test_full_schema_requires_both_verified_contracts(positive_bundle, tmp_path, omissions):
    run = save_run(positive_bundle, tmp_path, full=True)
    for name in omissions:
        del run["artifacts"][name]
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


def test_minimal_with_explicit_selection_without_schema_remains_legal(positive_bundle, tmp_path):
    artifacts = learned.save_learned_bundle(positive_bundle, tmp_path)
    run = {"dir": tmp_path, "engine_id": "gru", "params": positive_bundle["config"],
           "artifacts": artifacts, "selection": copy.deepcopy(positive_bundle["selection"])}
    assert_models_equal(positive_bundle, learned.load_learned_bundle(run))
    del run["selection"][CONTRACT]
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


def test_partial_target_objective_band_matches_runtime_band_and_origin_weights(monkeypatch):
    import pdm.models.signal_distribution as distribution
    from pdm.trajectory_objectives import path_losses

    seen, weight_calls = [], []
    objective_weights = learned._objective_weights
    def weights(data):
        result = objective_weights(data)
        weight_calls.append({key: value.copy() for key, value in result.items()})
        return result
    def spy(paths, target, mask, **kwargs):
        result = path_losses(paths, target, mask, **kwargs)
        if kwargs.get("path_band_geometry") == "issued_prefix":
            lower, upper = SignalDistribution.simultaneous_band(paths, kwargs["coverage"])
            scale = kwargs["signal_scale"].reshape(-1, 1)
            count = mask.sum(-1).clamp_min(1)
            valid = mask.any(-1)
            actual = torch.where(mask, target, torch.zeros_like(target)) / scale
            lo, hi = lower / scale, upper / scale
            expected_width = ((hi - lo) * mask).sum(-1) / count * valid
            expected_miss = ((torch.relu(lo - actual) * mask).amax(-1)
                             + (torch.relu(actual - hi) * mask).amax(-1)) * valid
            torch.testing.assert_close(result["width"], expected_width, rtol=0, atol=0)
            torch.testing.assert_close(result["miss"], expected_miss, rtol=0, atol=0)
            seen.append(bool((~mask).any()))
        return result
    monkeypatch.setattr(distribution, "path_losses", spy)
    monkeypatch.setattr(learned, "_objective_weights", weights)
    learned.fit_learned_model("gru", frame(), frame(.1), config(epochs=1))
    assert seen and any(seen)
    positive_weights = weight_calls.copy()
    weight_calls.clear()
    learned.fit_learned_model("gru", frame(), frame(.1), config(epochs=1, path_band_geometry="observed_prefix"))
    assert len(positive_weights) == len(weight_calls) == 2
    for expected, actual in zip(positive_weights, weight_calls):
        assert expected.keys() == actual.keys()
        for key in expected:
            np.testing.assert_array_equal(expected[key], actual[key])
