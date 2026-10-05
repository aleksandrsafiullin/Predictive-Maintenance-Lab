"""Independent bounded synthetic engineering QA, never warning-quality evidence."""
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

CONTRACT = "red_corridor_objective_contract"
FIELDS = ("red_corridor_width_weight", "red_corridor_miss_weight", "red_corridor_scale_s")
TERMS = ("red_corridor_width", "red_corridor_miss")


def config(engine="gru", **overrides):
    return base_config(engine, **{"post_factor_smoothness_weight": 0.,
                                 "red_corridor_width_weight": .7,
                                 "red_corridor_miss_weight": 1.3,
                                 "red_corridor_scale_s": 120., **overrides})


def equal(a, b):
    for key, value in a["model"].state_dict().items():
        assert torch.equal(value, b["model"].state_dict()[key]), key
    expected = learned.predict_learned(a, frame(.1))
    actual = learned.predict_learned(b, frame(.1))
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])


def save_run(bundle, directory, full=False):
    artifacts = learned.save_learned_bundle(bundle, directory)
    fields = learned._event_distribution_metadata(bundle["model"], bundle["config"])
    for name in ("training_contract.json", "model_input_contract.json"):
        (directory / name).write_text(json.dumps(fields))
        artifacts[name] = sha256_file(directory / name)
    run = {"dir": directory, "engine_id": "gru", "params": bundle["config"],
           "artifacts": artifacts, **copy.deepcopy(fields)}
    if full:
        run.update(schema_version=1, selection=copy.deepcopy(bundle["selection"]))
    return run


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_positive_fit_joint_selection_restores_actual_tensor_and_runtime_unchanged(engine, tmp_path, monkeypatch):
    factory, objective = learned._make_model, SignalDistribution.objective
    models, checkpoints, calls = [], [], []
    def capture(*args, **kwargs):
        model = factory(*args, **kwargs)
        models.append(model)
        return model
    def spy(self, *args, **kwargs):
        calls.append((self.training, kwargs.copy()))
        return objective(self, *args, **kwargs)
    monkeypatch.setattr(learned, "_make_model", capture)
    monkeypatch.setattr(SignalDistribution, "objective", spy)
    def report(record):
        if record["stage"] == "training":
            checkpoints.append(copy.deepcopy(models[0].state_dict()))
    cfg = config(engine, min_delta=1e10)
    bundle = learned.fit_learned_model(engine, frame(), frame(.1), cfg, report=report)
    assert {training for training, _ in calls} == {True, False}
    for _, kwargs in calls:
        assert kwargs[FIELDS[0]] == .7 and kwargs[FIELDS[1]] == 1.3
        assert kwargs["red_corridor_scale_steps"] == 2.
        assert tuple(kwargs["event_objective_prefix_lengths"]) == (1, 3)
    weights = {k: cfg[k + "_weight"] for k in ("energy", "width", "miss", "event", *TERMS)}
    weights["phase_energy"] = cfg["phase_weight"]
    assert len(weights) == 7
    for record in bundle["trace"]:
        for phase in ("train", "validation"):
            terms = record[phase]
            assert all(np.isfinite(v) for v in terms.values())
            assert terms["total"] == pytest.approx(sum(weights[k] * terms[k] for k in weights), rel=2e-6)
    assert bundle["selection"]["best_epoch"] == 1
    assert len(checkpoints) == 3
    assert any(not torch.equal(checkpoints[0][k], checkpoints[-1][k]) for k in checkpoints[0])
    for key, value in bundle["model"].state_dict().items():
        assert torch.equal(value, checkpoints[0][key])
    assert bundle["selection"]["best_score"] == bundle["trace"][0]["validation"]["total"]
    assert bundle["selection"]["restored_best_checkpoint"] and not bundle["selection"]["test_feedback"]
    fields = learned._event_distribution_metadata(bundle["model"], cfg)
    assert bundle["selection"][CONTRACT] == fields[CONTRACT]
    artifacts = learned.save_learned_bundle(bundle, tmp_path)
    equal(bundle, learned.load_learned_bundle({"dir": tmp_path, "engine_id": engine, "params": cfg, "artifacts": artifacts}))
    equal(bundle, {**bundle, "config": {**cfg, FIELDS[0]: 0., FIELDS[1]: 0.}})
    old_model = factory(engine, {**cfg, FIELDS[0]: 0., FIELDS[1]: 0.}, 2, None)
    assert old_model.state_dict().keys() == bundle["model"].state_dict().keys()
    assert sum(p.numel() for p in old_model.parameters()) == sum(p.numel() for p in bundle["model"].parameters())


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_missing_zero_legacy_rng_trace_tensors_metadata_exact(engine, tmp_path):
    explicit = config(engine, **{FIELDS[0]: 0., FIELDS[1]: 0., FIELDS[2]: 1800.})
    missing = {k: v for k, v in explicit.items() if k not in FIELDS}
    bundles, states = [], []
    for cfg in (missing, explicit):
        bundles.append(learned.fit_learned_model(engine, frame(), frame(.1), cfg))
        states.append((torch.random.get_rng_state().clone(), copy.deepcopy(np.random.get_state())))
    assert bundles[0]["trace"] == bundles[1]["trace"]
    assert bundles[0]["selection"] == bundles[1]["selection"]
    assert torch.equal(states[0][0], states[1][0])
    np.testing.assert_array_equal(states[0][1][1], states[1][1][1])
    assert states[0][1][2:] == states[1][1][2:]
    equal(*bundles)
    artifacts = learned.save_learned_bundle(bundles[0], tmp_path)
    metadata = json.loads((tmp_path / "learned_model.json").read_text())
    assert CONTRACT not in metadata and CONTRACT not in metadata["selection"]
    assert not set(TERMS) & bundles[0]["trace"][0]["train"].keys()
    equal(bundles[0], learned.load_learned_bundle({"dir": tmp_path, "engine_id": engine, "params": missing, "artifacts": artifacts}))


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_unknown_future_nan_inf_fit_validation_invariance(engine):
    baseline = learned.fit_learned_model(engine, frame(), frame(.1), config(engine))
    train, val = frame(), frame(.1)
    for data in (train, val):
        data["y"][~data["mask"]] = np.resize([np.nan, np.inf, -np.inf], (~data["mask"]).sum())
    mutated = learned.fit_learned_model(engine, train, val, config(engine))
    assert baseline["trace"] == mutated["trace"] and baseline["selection"] == mutated["selection"]
    equal(baseline, mutated)


def test_no_additional_sample_calls_or_generator_draws(monkeypatch):
    sample = SignalDistribution.sample
    calls = []
    def spy(self, *args, **kwargs):
        generator = args[3] if len(args) > 3 else kwargs.get("generator")
        before = generator.get_state().clone()
        result = sample(self, *args, **kwargs)
        calls.append((self.training, before, generator.get_state().clone(), kwargs.copy()))
        return result
    monkeypatch.setattr(SignalDistribution, "sample", spy)
    learned.fit_learned_model("gru", frame(), frame(.1), config(epochs=1))
    positive = calls.copy()
    calls.clear()
    learned.fit_learned_model("gru", frame(), frame(.1), config(epochs=1, **{FIELDS[0]: 0., FIELDS[1]: 0.}))
    assert len(calls) == len(positive) > 0
    for old, new in zip(calls, positive):
        assert old[0] == new[0] and old[3] == new[3]
        assert torch.equal(old[1], new[1]) and torch.equal(old[2], new[2])


@pytest.mark.parametrize("sampling", ["row_permutation", "physical_group"])
def test_width_and_miss_optimizer_have_distinct_physical_populations(sampling, monkeypatch):
    objective, optimizer = SignalDistribution.objective, learned._optimizer_term
    pointers, retained, calls = {}, [], []
    def spy(self, *args, **kwargs):
        result = objective(self, *args, **kwargs)
        for term in TERMS:
            retained.append(result[term])
            pointers[result[term].data_ptr()] = term
        return result
    def optimizer_spy(values, weights, indices, probabilities):
        if values.data_ptr() in pointers:
            calls.append((pointers[values.data_ptr()], weights.copy(), indices.copy(), probabilities))
        return optimizer(values, weights, indices, probabilities)
    monkeypatch.setattr(SignalDistribution, "objective", spy)
    monkeypatch.setattr(learned, "_optimizer_term", optimizer_spy)
    train = frame()
    train["current"][0] = 1.1  # currently RED is excluded even with future evidence
    learned.fit_learned_model("gru", train, frame(.1), config(epochs=1, batch_size=3, batch_sampling=sampling))
    expected = {TERMS[0]: np.array([0, .25, .25, .25, .25], np.float32),
                TERMS[1]: np.array([0, .5, 0, .5, 0], np.float32)}
    assert {term for term, *_ in calls} == set(TERMS)
    for term, weights, indices, probabilities in calls:
        np.testing.assert_array_equal(weights, expected[term])
        if sampling == "physical_group":
            np.testing.assert_array_equal(probabilities, learned._weights(train)[indices])
        else:
            assert probabilities is None


@pytest.fixture(scope="module")
def positive_bundle():
    return learned.fit_learned_model("gru", frame(), frame(.1), config())


@pytest.mark.parametrize("surface", ["learned_model.json", "selection", "manifest", "training_contract.json", "model_input_contract.json"])
def test_every_contract_field_deleted_or_tampered_rejected_after_rehash(positive_bundle, tmp_path, surface):
    contract = positive_bundle["selection"][CONTRACT]
    for field in [None, *contract]:
        for operation in ("delete", "tamper"):
            run = save_run(positive_bundle, tmp_path)
            target = "learned_model.json" if surface == "selection" else surface
            saved = run if target == "manifest" else json.loads((tmp_path / target).read_text())
            parent = saved["selection"] if surface == "selection" else saved
            if field is None:
                if operation == "delete":
                    del parent[CONTRACT]
                else:
                    parent[CONTRACT] = {"invalid": True}
            elif operation == "delete":
                del parent[CONTRACT][field]
            else:
                parent[CONTRACT][field] = {"invalid": True}
            if target != "manifest":
                (tmp_path / target).write_text(json.dumps(saved))
                run["artifacts"][target] = sha256_file(tmp_path / target)
            with pytest.raises(ValueError):
                learned.load_learned_bundle(run)


@pytest.mark.parametrize("name", ["training_contract.json", "model_input_contract.json"])
@pytest.mark.parametrize("field,wrong", [(FIELDS[0], 0.), (FIELDS[1], 0.), (FIELDS[2], 1800.), ("nominal_coverage", .5), ("event_objective_horizons_s", [180.]), ("training_samples", 4)])
def test_rehashed_duplicate_params_bound(positive_bundle, tmp_path, name, field, wrong):
    run = save_run(positive_bundle, tmp_path)
    saved = json.loads((tmp_path / name).read_text())
    saved["params"] = copy.deepcopy(positive_bundle["config"])
    saved["params"][field] = wrong
    (tmp_path / name).write_text(json.dumps(saved))
    run["artifacts"][name] = sha256_file(tmp_path / name)
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize("omission", ["training_contract.json", "model_input_contract.json", "selection"])
def test_full_run_requires_contracts_and_selection(positive_bundle, tmp_path, omission):
    run = save_run(positive_bundle, tmp_path, full=True)
    equal(positive_bundle, learned.load_learned_bundle(run))
    if omission == "selection":
        del run[omission]
    else:
        del run["artifacts"][omission]
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("bad", [True, False, np.bool_(True), "1", None, -1., np.nan, np.inf, -np.inf])
def test_strict_bad_config(field, bad):
    with pytest.raises(ValueError, match="red_corridor"):
        config(**{field: bad})


def test_strict_positive_samples_and_zero_scale_before_rng():
    for overrides in ({FIELDS[2]: 0.}, {"training_samples": 1}, {"training_samples": True}, {"training_samples": 2.5}):
        state = torch.random.get_rng_state().clone()
        with pytest.raises(ValueError):
            config(**overrides)
        assert torch.equal(state, torch.random.get_rng_state())


def test_isolated_real_worker_positive_contract_reload(tmp_path, monkeypatch):
    from pdm.cli import spawn_worker
    from pdm.data.project_import import import_project
    from pdm.data.project_prepare import prepare_project
    from pdm.projects import ProjectStore
    from pdm.signal_training import load_signal_run
    from pdm.worker import read_status
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    monkeypatch.setenv("PDM_WORKER_ROOT", str(tmp_path / "worker"))
    source = tmp_path / "source"
    source.mkdir()
    pd.DataFrame([{"unit_id": f"synthetic-{u}", "timestamp_s": float(t * 60), "rms": .2 + u * .11 + t * .12}
                  for u in range(8) for t in range(12)]).to_csv(source / "synthetic.csv", index=False)
    store = ProjectStore(tmp_path / "projects")
    project = store.create("Synthetic MC decision engineering QA", "generic_sensor_csv")
    manifest = import_project(project["project_id"], {"primary": {"mode": "folder", "path": str(source)},
        "signal_column": "rms", "signal_label": "Synthetic RMS", "signal_unit": "g",
        "thresholds": {"mode": "absolute", "direction": "above", "yellow": .9, "red": 1.5}, "seed": 19}, store=store)
    snapshot = prepare_project(project["project_id"], manifest["manifest_id"], store=store)
    params = {**config(epochs=2), "forecast_mode": learned.MODE, "max_windows_per_unit": 5}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast", "job_id": "isolated-mc-red-synthetic",
        "project_id": project["project_id"], "snapshot_id": snapshot["snapshot_id"], "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=60) == 0, read_status()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed", state
    run = load_signal_run(project["project_id"], state["run_id"])
    assert run["reload_verified"] and run["selection"]["test_feedback"] is False
    expected = run[CONTRACT]
    assert run["selection"][CONTRACT] == expected
    for name in ("learned_model.json", "training_contract.json", "model_input_contract.json"):
        assert json.loads((run["dir"] / name).read_text())[CONTRACT] == expected
        assert sha256_file(run["dir"] / name) == run["artifacts"][name]
    learned.load_learned_bundle(run)


def test_event_prefix_mean_uses_reused_full_draws_and_saved_scale(monkeypatch):
    import pdm.models.signal_distribution as distribution
    helper, objective = distribution.mc_red_decision_regularizer, SignalDistribution.objective
    records, checked = [], []
    def helper_spy(logits, entries, **kwargs):
        result = helper(logits, entries, **kwargs)
        length = kwargs["prefix_length"]
        # Independent empirical bracket construction; no target-derived geometry.
        edge = entries.clamp(0, length).double()
        left = torch.where(edge == length, length + 1, edge)
        right = edge + 1
        alpha = (1 - kwargs["coverage"]) / 2
        lo = torch.quantile(left, alpha, dim=0, interpolation="lower")
        hi = torch.quantile(right, 1 - alpha, dim=0, interpolation="higher")
        expected = torch.where(kwargs["already_red"], 0, torch.relu(hi - lo))
        torch.testing.assert_close(result["width_term"], expected, rtol=0, atol=0)
        records.append((entries.data_ptr(), length, result))
        return result
    def objective_spy(self, *args, **kwargs):
        records.clear()
        result = objective(self, *args, **kwargs)
        assert [r[1] for r in records] == [1, 3]
        assert len({r[0] for r in records}) == 1
        for term, field in zip(TERMS, ("width_term", "compatible_miss_term")):
            expected = sum(r[2][field] for r in records) / 2 / 2
            torch.testing.assert_close(result[term], expected, rtol=0, atol=0)
        checked.append(self.training)
        return result
    monkeypatch.setattr(distribution, "mc_red_decision_regularizer", helper_spy)
    monkeypatch.setattr(SignalDistribution, "objective", objective_spy)
    learned.fit_learned_model("gru", frame(), frame(.1), config(epochs=1))
    assert set(checked) == {True, False}
