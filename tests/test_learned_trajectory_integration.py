"""Engineering QA for saved learned paths; synthetic data is not Bearings quality evidence."""
from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from pdm.cli import spawn_worker
from pdm.data.project_import import import_project
from pdm.data.project_prepare import load_snapshot, prepare_project
from pdm.io_util import sha256_file
from pdm.learned_trajectory import (
    MODE,
    fit_learned_model,
    forecast_learned_prefix,
    learned_params,
    load_learned_bundle,
    predict_learned,
    save_learned_bundle,
)
from pdm.project_results_ui import replay_figure
from pdm.projects import ProjectStore
from pdm.signal_inference import forecast_prefix
from pdm.signal_training import load_signal_run
from pdm.trajectory_data import build_trajectory_frame, build_trajectory_prefix, slice_frame
from pdm.trajectory_evaluation import entry_corridor, evaluate_trajectory_model
from pdm.worker import read_status


@pytest.fixture(scope="module")
def learned_worker(tmp_path_factory):
    root = tmp_path_factory.mktemp("positive-rms-learned")
    with pytest.MonkeyPatch.context() as env:
        env.setenv("PDM_PROJECTS_ROOT", str(root / "projects"))
        env.setenv("PDM_WORKER_ROOT", str(root / "worker"))
        source = root / "source"
        source.mkdir()
        # Whole physical units and high precision positive measurements, real import/prepare.
        pd.DataFrame([
            {"unit_id": f"rms-{u}", "timestamp_s": float(t * 60),
             "rms": 0.2345678912345678 + u * .11 + t * .12}
            for u in range(8) for t in range(12)
        ]).to_csv(source / "rms.csv", index=False)
        store = ProjectStore(root / "projects")
        project = store.create("Synthetic positive RMS engineering QA", "generic_sensor_csv")
        manifest = import_project(project["project_id"], {
            "primary": {"mode": "folder", "path": str(source)},
            "signal_column": "rms", "signal_label": "RMS vibration", "signal_unit": "g",
            "thresholds": {"mode": "absolute", "direction": "above", "yellow": .9, "red": 1.5},
            "seed": 19,
        }, store=store)
        snapshot = prepare_project(project["project_id"], manifest["manifest_id"], store=store)
        params = {"forecast_mode": MODE, "history_length": 2, "horizons_s": [60., 120., 180.],
                  "epochs": 2, "hidden_size": 16, "num_layers": 1, "rank": 2,
                  "batch_size": 16, "max_windows_per_unit": 5, "path_samples": 16,
                  "training_samples": 4, "max_iter": 2, "cpu_threads": 1, "seed": 734}
        process = spawn_worker({"kind": "project_train", "task": "signal_forecast",
                                "job_id": "synthetic-learned-gru-qa",
                                "project_id": project["project_id"],
                                "snapshot_id": snapshot["snapshot_id"], "engine_id": "gru", "params": params})
        try:
            assert process.wait(timeout=90) == 0, read_status()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
        state = read_status()
        assert state["status"] == "completed", state
        assert state["job_id"] == "synthetic-learned-gru-qa"
        run = load_signal_run(project["project_id"], state["run_id"])
        data = load_snapshot(project["project_id"], snapshot["snapshot_id"])
        uid = str(data["split"]["test"][0])
        prefix = data["features"].loc[data["features"].unit_id.astype(str) == uid].sort_values("timestamp_s").iloc[:3].copy()
        yield root, run, data, uid, prefix


def test_actual_worker_saved_contract_bindings_and_hashes(learned_worker):
    _, run, data, _, _ = learned_worker
    assert run["params"]["forecast_mode"] == MODE
    assert run["snapshot_id"] == data["snapshot_id"]
    assert run["snapshot_fingerprint_sha256"] == hashlib.sha256(
        json.dumps(data["fingerprint"], sort_keys=True, default=str).encode()).hexdigest()
    assert run["reload_verified"] and run["selection"]["restored_best_checkpoint"]
    assert run["selection"]["test_feedback"] is False
    assert run["selection"]["best_epoch"] in {1, 2}
    assert run["metrics"]["validation"]["evaluation_partition"] == "Validation"
    assert run["metrics"]["validation"]["status"] == "exploratory"
    assert run["metrics"]["test"]["evaluation_partition"] == "Test"
    assert run["metrics"]["test"]["status"] == "exploratory_reused_test"
    for name, digest in run["artifacts"].items():
        assert sha256_file(run["dir"] / name) == digest
    trace = json.loads((run["dir"] / "objective_trace.json").read_text())
    assert len(trace) == 2
    assert {"energy", "width", "miss", "event", "phase_energy"} <= set(trace[0]["train"])
    frame = build_trajectory_prefix(data, learned_worker[-1], run["params"])
    before = predict_learned(load_learned_bundle(run), frame)
    after = predict_learned(load_learned_bundle(run), frame)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


def test_group_sampler_actual_worker_manifest_reload_and_event_prefix_binding(learned_worker):
    # This operates only in the fixture's isolated synthetic project/worker roots.
    _, original, data, _, prefix = learned_worker
    params = {**original["params"], "batch_sampling": "physical_group",
              "event_objective_horizons_s": [60., 180.]}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast",
                            "job_id": "synthetic-group-sampling-qa",
                            "project_id": original["project_id"], "snapshot_id": data["snapshot_id"],
                            "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=90) == 0, read_status()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed" and state["job_id"] == "synthetic-group-sampling-qa"
    run = load_signal_run(original["project_id"], state["run_id"])
    assert run["params"]["batch_sampling"] == run["selection"]["batch_sampling"] == "physical_group"
    assert run["selection"]["event_objective_horizons_s"] == [60., 180.]
    assert run["reload_verified"] and not run["selection"]["test_feedback"]
    for name, digest in run["artifacts"].items():
        assert sha256_file(run["dir"] / name) == digest
    bundle = load_learned_bundle(run)
    frame = build_trajectory_prefix(data, prefix, run["params"])
    first, reloaded = predict_learned(bundle, frame), predict_learned(load_learned_bundle(run), frame)
    for key in first:
        np.testing.assert_array_equal(first[key], reloaded[key])


def test_event_head_rate_actual_worker_manifest_reload(learned_worker):
    # The fixture owns isolated synthetic project/worker roots, never real Bearings.
    _, original, data, _, prefix = learned_worker
    params = {**original["params"], "event_head_learning_rate_multiplier": 10.,
              "phase_covariance": "separate", "batch_sampling": "physical_group",
              "event_objective_horizons_s": [60., 180.]}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast",
                            "job_id": "synthetic-event-head-rate-qa",
                            "project_id": original["project_id"], "snapshot_id": data["snapshot_id"],
                            "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=90) == 0, read_status()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed" and state["job_id"] == "synthetic-event-head-rate-qa"
    run = load_signal_run(original["project_id"], state["run_id"])
    assert run["params"]["event_head_learning_rate_multiplier"] == 10.
    rates = run["selection"]["optimizer_rates"]
    assert rates["parameter_group_policy"] == "all_other_parameters_then_event_head"
    assert rates["base_learning_rate"] == run["params"]["learning_rate"]
    assert rates["event_head_learning_rate"] == 10 * rates["base_learning_rate"]
    assert rates["event_head_decay_per_step"] == 10 * rates["base_decay_per_step"]
    assert run["reload_verified"] and not run["selection"]["test_feedback"]
    assert run["params"]["phase_covariance"] == "separate"
    for name, digest in run["artifacts"].items():
        assert sha256_file(run["dir"] / name) == digest
    frame = build_trajectory_prefix(data, prefix, run["params"])
    first = predict_learned(load_learned_bundle(run), frame)
    second = predict_learned(load_learned_bundle(run), frame)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_finite_horizon_family_actual_worker_reload_and_prefix(learned_worker):
    # Real dispatch in isolated synthetic roots checks integration, not Bearings quality.
    _, original, data, uid, prefix = learned_worker
    params = {**original["params"], "event_distribution": "finite_horizon_mixture",
              "phase_covariance": "separate", "batch_sampling": "physical_group",
              "event_objective_horizons_s": [60., 180.],
              "event_head_learning_rate_multiplier": 1.}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast",
                            "job_id": "synthetic-finite-horizon-family-qa",
                            "project_id": original["project_id"], "snapshot_id": data["snapshot_id"],
                            "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=90) == 0, read_status()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed"
    assert state["job_id"] == "synthetic-finite-horizon-family-qa"
    run = load_signal_run(original["project_id"], state["run_id"])
    assert run["params"]["event_distribution"] == "finite_horizon_mixture"
    assert run["selection"]["optimizer_rates"]["parameter_group_policy"] == "legacy_single_group"
    assert run["reload_verified"] and not run["selection"]["test_feedback"]
    for name, digest in run["artifacts"].items():
        assert sha256_file(run["dir"] / name) == digest
    bundle = load_learned_bundle(run)
    assert bundle["model"].event_distribution == "finite_horizon_mixture"
    assert bundle["model"].effective_event_location == "bounded_horizon_logistic"
    expected_contract = {"horizon_steps": 3, "horizons_s": [60., 120., 180.],
                         "median_lower_steps": .05, "median_upper_steps": 3,
                         "component_truncation": "per component at saved horizon",
                         "survival_semantics": "no recorded entry through saved horizon"}
    for metadata in (bundle, run,
                     json.loads((run["dir"] / "training_contract.json").read_text()),
                     json.loads((run["dir"] / "model_input_contract.json").read_text())):
        assert metadata["event_distribution"] == "finite_horizon_mixture"
        assert metadata["effective_event_location"] == "bounded_horizon_logistic"
        assert metadata["event_distribution_contract"] == expected_contract
    frame = build_trajectory_prefix(data, prefix, run["params"])
    first = predict_learned(bundle, frame)
    reloaded = predict_learned(load_learned_bundle(run), frame)
    for key in first:
        np.testing.assert_array_equal(first[key], reloaded[key])
    full = forecast_learned_prefix(run, data, prefix, uid)
    short = forecast_learned_prefix(run, data, prefix, uid, prediction_horizon_s=60.)
    assert full["status"] == short["status"] == "available"
    np.testing.assert_array_equal(np.asarray(full["sampled_paths"])[:, :2], short["sampled_paths"])
    probabilities = first["event_probabilities"][0]
    collapsed = np.r_[probabilities[:1], probabilities[1:].sum()]
    expected = entry_corridor(collapsed, [60.], run["params"]["nominal_coverage"],
                              float(prefix.timestamp_s.iloc[-1]))
    assert short["red_entry_corridor"] == expected


def test_coupled_path_timing_actual_worker_contract_reload_and_prefix(learned_worker):
    # Isolated synthetic dispatch verifies integration, not Bearings quality.
    _, original, data, uid, prefix = learned_worker
    params = {**original["params"], "event_distribution": "finite_horizon_mixture",
              "path_distribution": "coupled_timing_mixture", "phase_covariance": "separate",
              "batch_sampling": "physical_group", "training_samples": 8,
              "path_objective_horizons_s": [60., 180.],
              "event_objective_horizons_s": [60., 180.]}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast",
                            "job_id": "synthetic-coupled-path-timing-qa",
                            "project_id": original["project_id"], "snapshot_id": data["snapshot_id"],
                            "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=60) == 0, read_status()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed"
    assert state["job_id"] == "synthetic-coupled-path-timing-qa"
    run = load_signal_run(original["project_id"], state["run_id"])
    assert run["reload_verified"] and not run["selection"]["test_feedback"]
    assert run["funnel"]["operational_coverage_approved"] is False
    for name, digest in run["artifacts"].items():
        assert sha256_file(run["dir"] / name) == digest
    bundle = load_learned_bundle(run)
    model = bundle["model"]
    assert model.path_distribution == "coupled_timing_mixture" and model.path_components == 3
    assert model.path_head[-1].out_features == 3 * 3 * (6 + 3 * 2)
    for metadata in (bundle, run,
                     json.loads((run["dir"] / "training_contract.json").read_text()),
                     json.loads((run["dir"] / "model_input_contract.json").read_text())):
        assert metadata["path_distribution"] == "coupled_timing_mixture"
        assert metadata["path_components"] == 3
        assert metadata["path_distribution_contract"] == run["path_distribution_contract"]
        assert metadata["path_distribution_contract"]["path_head_outputs"] == model.path_head[-1].out_features
    frame = build_trajectory_prefix(data, prefix, run["params"])
    first, reloaded = predict_learned(bundle, frame), predict_learned(load_learned_bundle(run), frame)
    for key in first:
        np.testing.assert_array_equal(first[key], reloaded[key])
    full = forecast_learned_prefix(run, data, prefix, uid)
    short = forecast_learned_prefix(run, data, prefix, uid, prediction_horizon_s=60.)
    assert full["status"] == short["status"] == "available"
    np.testing.assert_array_equal(np.asarray(full["sampled_paths"])[:, :2], short["sampled_paths"])
    probability = first["event_probabilities"][0]
    expected = entry_corridor(np.r_[probability[:1], probability[1:].sum()], [60.],
                              run["params"]["nominal_coverage"], float(prefix.timestamp_s.iloc[-1]))
    assert short["red_entry_corridor"] == expected


def test_variable_causal_context_actual_worker_reload_and_prefix(learned_worker):
    # Isolated synthetic worker checks the packed-input persistence contract.
    _, original, data, uid, prefix = learned_worker
    params = {**original["params"], "event_distribution": "finite_horizon_mixture",
              "path_distribution": "single", "phase_covariance": "separate",
              "recurrent_context_mode": "variable_causal", "max_history_length": 8,
              "batch_sampling": "physical_group", "training_samples": 8,
              "path_objective_horizons_s": [60., 180.],
              "event_objective_horizons_s": [60., 180.]}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast",
                            "job_id": "synthetic-variable-causal-context-qa",
                            "project_id": original["project_id"], "snapshot_id": data["snapshot_id"],
                            "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=60) == 0, read_status()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed"
    assert state["job_id"] == "synthetic-variable-causal-context-qa"
    run = load_signal_run(original["project_id"], state["run_id"])
    assert run["reload_verified"] and not run["selection"]["test_feedback"]
    assert run["funnel"]["operational_coverage_approved"] is False
    for name, digest in run["artifacts"].items():
        assert sha256_file(run["dir"] / name) == digest
    bundle = load_learned_bundle(run)
    assert bundle["model"].recurrent_context_mode == "variable_causal"
    assert bundle["model"].min_history_length == 2
    assert bundle["model"].max_history_length == 8
    for metadata in (bundle, run,
                     json.loads((run["dir"] / "training_contract.json").read_text()),
                     json.loads((run["dir"] / "model_input_contract.json").read_text())):
        assert metadata["recurrent_context_mode"] == "variable_causal"
        assert metadata["recurrent_context_contract"] == run["recurrent_context_contract"]
        assert metadata["recurrent_context_contract"]["minimum_real_observations"] == 2
        assert metadata["recurrent_context_contract"]["maximum_real_observations"] == 8
    frame = build_trajectory_prefix(data, prefix, run["params"])
    assert frame["history_lengths"].tolist() == [len(prefix)]
    first, reloaded = predict_learned(bundle, frame), predict_learned(load_learned_bundle(run), frame)
    for key in first:
        np.testing.assert_array_equal(first[key], reloaded[key])
    full = forecast_learned_prefix(run, data, prefix, uid)
    short = forecast_learned_prefix(run, data, prefix, uid, prediction_horizon_s=60.)
    assert full["status"] == short["status"] == "available"
    np.testing.assert_array_equal(np.asarray(full["sampled_paths"])[:, :2], short["sampled_paths"])
    probability = first["event_probabilities"][0]
    expected = entry_corridor(np.r_[probability[:1], probability[1:].sum()], [60.],
                              run["params"]["nominal_coverage"], float(prefix.timestamp_s.iloc[-1]))
    assert short["red_entry_corridor"] == expected


def test_post_factor_smoothness_actual_worker_reload_and_prefix(learned_worker):
    # Synthetic isolated roots exercise real worker dispatch and all contracts.
    _, original, data, uid, prefix = learned_worker
    params = {**original["params"], "event_distribution": "finite_horizon_mixture",
              "path_distribution": "single", "phase_covariance": "separate",
              "post_factor_smoothness_weight": 5.0,
              "batch_sampling": "physical_group", "training_samples": 8,
              "path_objective_horizons_s": [60., 180.],
              "event_objective_horizons_s": [60., 180.]}
    process = spawn_worker({"kind": "project_train", "task": "signal_forecast",
                            "job_id": "synthetic-post-factor-smoothness-qa",
                            "project_id": original["project_id"], "snapshot_id": data["snapshot_id"],
                            "engine_id": "gru", "params": params})
    try:
        assert process.wait(timeout=60) == 0, read_status()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=15)
    state = read_status()
    assert state["status"] == "completed"
    assert state["job_id"] == "synthetic-post-factor-smoothness-qa"
    run = load_signal_run(original["project_id"], state["run_id"])
    assert run["reload_verified"] and not run["selection"]["test_feedback"]
    assert run["funnel"]["operational_coverage_approved"] is False
    for name, digest in run["artifacts"].items():
        assert sha256_file(run["dir"] / name) == digest
    bundle = load_learned_bundle(run)
    contract = run["post_factor_smoothness_contract"]
    assert contract["weight"] == 5.0 and contract["sampler_change"] is False
    assert contract["objective_prefix_horizons_s"] == [60., 180.]
    assert run["selection"]["post_factor_smoothness_contract"] == contract
    for metadata in (bundle, run,
                     json.loads((run["dir"] / "training_contract.json").read_text()),
                     json.loads((run["dir"] / "model_input_contract.json").read_text())):
        assert metadata["post_factor_smoothness_contract"] == contract
    trace = json.loads((run["dir"] / "objective_trace.json").read_text())
    weights = {"energy": params.get("energy_weight", 1.0),
               "width": params.get("width_weight", .05), "miss": params.get("miss_weight", 2.0),
               "event": params.get("event_weight", 1.0), "phase_energy": params.get("phase_weight", .5),
               "post_factor_smoothness": 5.0}
    for epoch in trace:
        for partition in ("train", "validation"):
            terms = epoch[partition]
            assert terms["post_factor_smoothness"] >= 0
            assert terms["total"] == pytest.approx(sum(terms[k] * w for k, w in weights.items()))
    frame = build_trajectory_prefix(data, prefix, run["params"])
    first, reloaded = predict_learned(bundle, frame), predict_learned(load_learned_bundle(run), frame)
    for key in first:
        np.testing.assert_array_equal(first[key], reloaded[key])
    full = forecast_learned_prefix(run, data, prefix, uid)
    short = forecast_learned_prefix(run, data, prefix, uid, prediction_horizon_s=60.)
    assert full["status"] == short["status"] == "available"
    np.testing.assert_array_equal(np.asarray(full["sampled_paths"])[:, :2], short["sampled_paths"])


def test_public_inference_exact_anchor_deterministic_and_suffix_invariant(learned_worker, monkeypatch):
    _, run, data, uid, prefix = learned_worker
    issued = float(prefix.timestamp_s.iloc[-1])
    first = forecast_prefix(run["project_id"], run["run_id"], uid, issued)
    assert first == forecast_prefix(run["project_id"], run["run_id"], uid, issued)
    assert first["status"] == "available"
    actual = float(prefix.signal.iloc[-1])
    assert actual != float(np.float32(actual)), "Fixture must detect accidental float32 anchoring"
    assert first["points"][0] == {"target_time_s": issued, "value": actual,
                                 "lower": actual, "upper": actual, "kind": "anchor"}
    paths = np.asarray(first["sampled_paths"])
    assert paths.shape == (16, 4)
    assert (paths[:, 0] == actual).all() and (paths >= 0).all()
    changed = copy.deepcopy(data)
    hidden = (changed["features"].unit_id.astype(str) == uid) & (changed["features"].timestamp_s > issued)
    changed["features"].loc[hidden, "signal"] = 99999.
    changed["features"].loc[hidden, "gap_before"] = True
    changed["features"].loc[hidden, "timestamp_s"] += 123456.
    monkeypatch.setattr("pdm.signal_inference.load_snapshot", lambda *_: changed)
    assert first == forecast_prefix(run["project_id"], run["run_id"], uid, issued)
    frame = build_trajectory_prefix(data, prefix, run["params"])
    pred = predict_learned(load_learned_bundle(run), frame)
    expected = entry_corridor(pred["event_probabilities"][0], run["params"]["horizons_s"],
                              run["params"]["nominal_coverage"], issued)
    assert first["red_entry_corridor"] == expected


def test_single_prefix_matches_batch_and_reordered_batch(learned_worker):
    _, run, data, uid, prefix = learned_worker
    bundle = load_learned_bundle(run)
    all_rows = build_trajectory_frame(data, [uid], run["params"])
    one = build_trajectory_prefix(data, prefix, run["params"])
    index = int(np.flatnonzero(all_rows["as_of_s"] == prefix.timestamp_s.iloc[-1])[0])
    batch = predict_learned(bundle, all_rows)
    solo = predict_learned(bundle, one)
    rev = predict_learned(bundle, slice_frame(all_rows, np.arange(len(all_rows["x"]))[::-1]))
    for key in solo:
        axis = 1 if key == "paths" else 0
        np.testing.assert_array_equal(np.take(batch[key], [index], axis=axis), solo[key])
        np.testing.assert_array_equal(np.flip(rev[key], axis=axis), batch[key])


def test_insufficient_history_since_last_gap_is_unavailable(learned_worker):
    _, run, data, uid, prefix = learned_worker
    short = forecast_learned_prefix(run, data, prefix.iloc[:1], uid)
    assert short["status"] == "unavailable" and not short["points"]
    prefix = prefix.copy()
    prefix.loc[prefix.index[-1], "gap_before"] = True
    gap = forecast_learned_prefix(run, data, prefix, uid)
    assert gap["status"] == "unavailable" and not gap["points"] and not gap["sampled_paths"]
    assert "history" in gap["reason"].lower() or "observations" in gap["reason"].lower()


def test_corrupted_hash_rejected(learned_worker):
    _, run, _, _, _ = learned_worker
    target = run["dir"] / "checkpoint.pt"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"QA corruption")
        with pytest.raises(ValueError, match="hash mismatch"):
            load_signal_run(run["project_id"], run["run_id"])
        with pytest.raises(ValueError, match="hash mismatch"):
            load_learned_bundle(run)
    finally:
        target.write_bytes(original)


@pytest.mark.parametrize("engine", ["gru", "lstm", "quantile_boosting"])
def test_three_cheap_engines_synthetic_fit_and_exact_reload(learned_worker, tmp_path, engine):
    _, run, data, _, _ = learned_worker
    config = learned_params(engine, run["params"], data["features"].loc[
        data["features"].unit_id.astype(str).isin(data["split"]["train"])])
    train = build_trajectory_frame(data, data["split"]["train"], config, cap=3)
    validation = build_trajectory_frame(data, data["split"]["validation"], config, cap=3)
    bundle = fit_learned_model(engine, train, validation, config)
    artifacts = save_learned_bundle(bundle, tmp_path)
    loaded = load_learned_bundle({"dir": tmp_path, "artifacts": artifacts, "params": config,
                                  "engine_id": engine})
    before, after = predict_learned(bundle, validation), predict_learned(loaded, validation)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert before["paths"].shape == (16, len(validation["x"]), 3)
    assert np.all(before["paths"] >= 0)
    np.testing.assert_allclose(before["event_probabilities"].sum(1), 1, atol=1e-6)
    if engine == "quantile_boosting":
        unbound = {name: digest for name, digest in artifacts.items() if name != "encoder.joblib"}
        with pytest.raises(ValueError, match="verified artifact"):
            load_learned_bundle({"dir": tmp_path, "artifacts": unbound, "params": config,
                                 "engine_id": engine})


@pytest.mark.parametrize("survival", [.05, .06, .1])
def test_corridor_prior_grid_lower_edge_and_open_unconditional_survival(survival):
    finite = entry_corridor([0., 1., 0., 0.], [60., 120., 180.], issued=1000.)
    assert finite["earliest_s"] == 1060. and finite["latest_s"] == 1120.
    opened = entry_corridor([.2, .3, .5 - survival, survival], [60., 120., 180.], issued=1000.)
    assert opened["latest_s"] is None and opened["right_censored"]
    assert opened["conditional_latest_s"] is not None
    assert opened["conditioning"] == "unconditional_including_no_entry"
    assert opened["observed_future_used"] is False


def test_evaluation_uses_issued_constructor_and_separate_conditional_metrics(learned_worker, monkeypatch):
    _, run, data, uid, _ = learned_worker
    frame = build_trajectory_frame(data, [uid], run["params"])
    bundle = load_learned_bundle(run)
    calls = []
    def observed(*args, **kwargs):
        answer = entry_corridor(*args, **kwargs)
        calls.append(answer)
        return answer
    monkeypatch.setattr("pdm.trajectory_evaluation.entry_corridor", observed)
    metrics = evaluate_trajectory_model(lambda batch: predict_learned(bundle, batch), frame, run["params"])
    assert calls
    assert metrics["coverage_guarantee"] is False
    assert metrics["physical_group_count"] == 1
    assert any(row["incomplete_paths"] for row in metrics["horizons"])
    for row in metrics["horizons"]:
        assert {"red_bracket_containment", "red_finite_corridor_fraction",
                "conditional_red_bracket_coverage", "conditional_red_mean_width_s"} <= row.keys()


def test_english_plotly_learned_band_and_schema_smoke(learned_worker):
    _, run, _, uid, prefix = learned_worker
    result = forecast_prefix(run["project_id"], run["run_id"], uid, float(prefix.timestamp_s.iloc[-1]))
    fig = replay_figure(result, run["schema"])
    lower = next(t for t in fig.data if t.name == "Forecast band")
    upper = next(t for t in fig.data if t.name == "Forecast band upper")
    assert lower.fill == "tonexty" and lower.fillcolor
    assert list(lower.x) == list(upper.x)
    assert lower.y[0] == upper.y[0] == float(prefix.signal.iloc[-1])
    assert not any("\u0400" <= c <= "\u04ff" for c in fig.to_json())
    assert result["funnel"]["simultaneous"] and result["funnel"]["coverage_guarantee"] is False


def test_combined_objective_backward_preserves_caller_evidence_masks():
    import torch

    from pdm.models.signal_distribution import SignalDistribution

    model = SignalDistribution("gru", 2, 3, hidden_size=16, num_layers=1, rank=2)
    moments = model(torch.zeros(3, 2, 2), torch.tensor([.2, .3, .4]))
    observed = torch.tensor([True, False, False])
    allowed = torch.tensor([[False, True, False, False], [False] * 4, [False] * 4])
    prefix = torch.tensor([0, 1, 0])
    mask = torch.tensor([[True, True, True], [True, False, False], [False] * 3])
    initial = [value.clone() for value in (observed, allowed, prefix, mask)]
    terms = model.objective(
        moments, torch.tensor([[.5, 3.1, 3.5], [.4, 0., 0.], [0., 0., 0.]]), mask, 3.,
        event_allowed_mask=allowed, event_observed_mask=observed, no_entry_prefix=prefix,
        n_samples=8, generator=torch.Generator().manual_seed(99))
    for before, after in zip(initial, (observed, allowed, prefix, mask)):
        assert torch.equal(before, after)
    terms["total"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.event_head.weight.grad.abs().sum() > 0


def test_declared_prefix_funnel_uses_exact_evaluation_band_and_collapsed_survival(learned_worker):
    from pdm.trajectory_evaluation import _band

    _, run, data, uid, prefix = learned_worker
    full = forecast_learned_prefix(run, data, prefix, uid)
    shorter = forecast_learned_prefix(run, data, prefix, uid, prediction_horizon_s=120.)
    assert shorter['funnel']['issued_horizon_s'] == 120.
    assert shorter['funnel']['band_scope'] == 'declared_prefix_horizon'
    assert len(shorter['points']) == 3
    np.testing.assert_array_equal(np.asarray(shorter['sampled_paths']), np.asarray(full['sampled_paths'])[:, :3])
    paths = np.asarray(shorter['sampled_paths'])[:, None, 1:]
    lower, upper = _band(paths, run['params']['nominal_coverage'])
    np.testing.assert_array_equal([p['lower'] for p in shorter['points'][1:]], lower[0])
    np.testing.assert_array_equal([p['upper'] for p in shorter['points'][1:]], upper[0])
    frame = build_trajectory_prefix(data, prefix, run['params'])
    output = predict_learned(load_learned_bundle(run), frame)
    p = output['event_probabilities'][0]
    expected = entry_corridor(np.r_[p[:2], p[2:].sum()], [60.,120.],
                              run['params']['nominal_coverage'], float(prefix.timestamp_s.iloc[-1]))
    assert shorter['red_entry_corridor'] == expected
    with pytest.raises(ValueError, match='Prediction span'):
        forecast_learned_prefix(run,data,prefix,uid,prediction_horizon_s=10000.)


def test_input_feature_order_must_match_saved_contract(learned_worker):
    _, run, data, _, prefix = learned_worker
    frame = build_trajectory_prefix(data, prefix, run["params"])
    frame["feature_names"] = list(reversed(frame["feature_names"]))
    with pytest.raises(ValueError, match="input feature contract"):
        predict_learned(load_learned_bundle(run), frame)


def test_unknown_evaluation_targets_cannot_change_reported_metrics(learned_worker):
    _, run, data, uid, _ = learned_worker
    frame = build_trajectory_frame(data, [uid], run['params'])
    bundle = load_learned_bundle(run)
    left = evaluate_trajectory_model(lambda batch: predict_learned(bundle,batch), frame,run['params'])
    frame['y'][~frame['mask']] = np.nan
    right = evaluate_trajectory_model(lambda batch: predict_learned(bundle,batch), frame,run['params'])
    assert left == right
    assert all('energy_signal_red_scaled' in row for row in right['horizons'])


def test_candidate_snapshot_filters_test_features_before_decode(learned_worker, monkeypatch):
    _, run, original, _, _ = learned_worker
    calls = []
    read = pd.read_parquet

    def capture(path, *args, **kwargs):
        if str(path).endswith("features.parquet"):
            calls.append(kwargs.get("filters"))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", capture)
    subset = load_snapshot(run["project_id"], run["snapshot_id"], feature_partitions=("train", "validation"))
    admitted = original["split"]["train"] + original["split"]["validation"]
    assert calls == [[("unit_id", "in", admitted)]]
    assert set(subset["features"].unit_id) == set(admitted)
    assert not set(subset["features"].unit_id) & set(original["split"]["test"])
    assert subset["fingerprint"] == original["fingerprint"]
    assert subset["loaded_feature_partitions"] == ["train", "validation"]
    pd.testing.assert_frame_equal(subset["features"].reset_index(drop=True),
                                  original["features"].loc[original["features"].unit_id.isin(admitted)].reset_index(drop=True))


@pytest.mark.parametrize("partitions", [(), [], "train", ("unknown",), ("train", "train"), (True,)])
def test_snapshot_refuses_invalid_feature_partition_filter(learned_worker, partitions):
    _, run, _, _, _ = learned_worker
    with pytest.raises(ValueError, match="Feature partitions"):
        load_snapshot(run["project_id"], run["snapshot_id"], feature_partitions=partitions)
