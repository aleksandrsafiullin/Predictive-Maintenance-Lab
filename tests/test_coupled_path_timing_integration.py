"""Small synthetic integration only; no Bearings data or quality evidence."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pdm import learned_trajectory as learned
from pdm.io_util import sha256_file
from pdm.trajectory_data import build_trajectory_frame, slice_frame


def config(engine="gru", selector="coupled_timing_mixture"):
    features = pd.DataFrame(
        {
            "unit_id": ["synthetic"] * 4,
            "timestamp_s": np.arange(4) * 60.0,
            "gap_before": [False] * 4,
        }
    )
    return learned.learned_params(
        engine,
        {
            "horizons_s": [60.0, 120.0, 180.0],
            "history_length": 2,
            "hidden_size": 16,
            "num_layers": 1,
            "rank": 2,
            "epochs": 2,
            "batch_size": 4,
            "path_samples": 16,
            "training_samples": 8,
            "cpu_threads": 1,
            "seed": 927,
            "dropout": 0.0,
            "event_distribution": "finite_horizon_mixture",
            "path_distribution": selector,
            "phase_covariance": "separate",
            "path_objective_horizons_s": [60.0, 180.0],
            "event_objective_horizons_s": [60.0, 180.0],
        },
        features,
    )


def frame(group, offset=0):
    x = np.arange(16, dtype=np.float32).reshape(4, 2, 2) / 20 + offset
    observed = np.array([True, False, True, False])
    allowed = np.zeros((4, 4), bool)
    allowed[observed, 1] = True
    allowed[~observed, 3] = True
    y = np.tile([0.5, 1.2, 1.4], (4, 1)).astype(np.float32)
    y[~observed] = [0.5, 0.6, 0.7]
    return {
        "x": x,
        "raw_x": x[:, :, :1],
        "y": y,
        "mask": np.ones((4, 3), bool),
        "current": np.full(4, 0.3, np.float32),
        "red_threshold": np.ones(4, np.float32),
        "event_allowed": allowed,
        "event_observed": observed,
        "no_entry_prefix": np.where(observed, 0, 3),
        "physical_unit_id": [group] * 4,
        "feature_names": ["f1", "f2"],
    }


@pytest.fixture(scope="module")
def coupled_bundle():
    return learned.fit_learned_model("gru", frame("train"), frame("validation", 0.1), config())


def save_contracts(bundle, directory):
    artifacts = learned.save_learned_bundle(bundle, directory)
    fields = learned._event_distribution_metadata(bundle["model"], bundle["config"])
    for name in ("training_contract.json", "model_input_contract.json"):
        saved = copy.deepcopy(fields)
        if name == "model_input_contract.json":
            saved["phase_covariance"] = bundle["model"].phase_covariance
            saved["path_head_outputs"] = bundle["model"].path_head[-1].out_features
        (directory / name).write_text(json.dumps(saved))
        artifacts[name] = sha256_file(directory / name)
    return {
        "dir": directory,
        "engine_id": "gru",
        "params": bundle["config"],
        "scaler": bundle["scaler"],
        "artifacts": artifacts,
        **copy.deepcopy(fields),
    }


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_actual_tiny_joint_fit_and_exact_reload(engine, tmp_path):
    resolved = config(engine)
    bundle = learned.fit_learned_model(engine, frame("train"), frame("validation", 0.1), resolved)
    assert bundle["selection"]["best_score"] == min(
        r["validation"]["total"] for r in bundle["trace"]
    )
    assert bundle["selection"]["restored_best_checkpoint"]
    assert (
        not bundle["selection"]["test_feedback"] and not bundle["selection"]["coverage_guarantee"]
    )
    for record in bundle["trace"]:
        assert all(
            np.isfinite(v) for phase in ("train", "validation") for v in record[phase].values()
        )
    fields = learned._event_distribution_metadata(bundle["model"], resolved)
    assert fields["path_components"] == 3
    assert fields["path_distribution_contract"]["path_head_outputs"] == 3 * 3 * (6 + 3 * 2)
    artifacts = learned.save_learned_bundle(bundle, tmp_path)
    loaded = learned.load_learned_bundle(
        {"dir": tmp_path, "artifacts": artifacts, "engine_id": engine, "params": resolved, **fields}
    )
    for name, value in bundle["model"].state_dict().items():
        assert torch.equal(value, loaded["model"].state_dict()[name])
    before = learned.predict_learned(bundle, frame("validation", 0.1))
    after = learned.predict_learned(loaded, frame("validation", 0.1))
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    assert before["paths"].shape == (16, 4, 3)
    assert before["event_probabilities"].shape == (4, 4)


@pytest.mark.parametrize("engine", ["gru", "lstm", "full_cns", "quantile_boosting"])
def test_all_engine_shared_factory_shapes(engine):
    torch.manual_seed(927)
    external = 5 if engine in {"full_cns", "quantile_boosting"} else None
    model = learned._make_model(engine, config(engine), 2, external).eval()
    moments = model(
        torch.zeros(2, 2, 2),
        torch.tensor([0.3, 0.4]),
        None if external is None else torch.ones(2, external),
    )
    assert model.path_components == 3 and model.path_head[-1].out_features == 108
    assert moments["mean"].shape == (2, 3, 3)
    assert moments["pre_factors"].shape == (2, 3, 3, 2)
    assert moments["joint_event_logits"].shape == (2, 3, 4)
    assert moments["component_logits"].shape == (2, 3)
    paths = model.sample(moments, 1.0, 8, torch.Generator().manual_seed(8))
    assert paths.shape == (8, 2, 3) and torch.isfinite(paths).all()


def test_future_mutation_changes_labels_but_never_causal_prediction():
    resolved = config()
    features = pd.DataFrame(
        {
            "unit_id": ["synthetic"] * 10,
            "physical_unit_id": ["physical"] * 10,
            "timestamp_s": np.arange(10) * 60.0,
            "gap_before": [False] * 10,
            "signal": np.linspace(0.2, 0.7, 10),
        }
    )
    data = {
        "features": features,
        "schema": {"thresholds": {"mode": "absolute", "direction": "above", "red": 1.0}},
    }
    before = slice_frame(build_trajectory_frame(data, ["synthetic"], resolved), np.array([0]))
    changed = features.copy()
    changed.loc[changed.timestamp_s > 60, "signal"] = 9.0
    after = slice_frame(
        build_trajectory_frame({**data, "features": changed}, ["synthetic"], resolved),
        np.array([0]),
    )
    for key in ("x", "raw_x", "current"):
        np.testing.assert_array_equal(before[key], after[key])
    assert not np.array_equal(before["y"], after["y"]) and not np.array_equal(
        before["event_allowed"], after["event_allowed"]
    )
    torch.manual_seed(31)
    model = learned._make_model("gru", resolved, 13, None).eval()
    bundle = {
        "model": model,
        "encoder": None,
        "config": resolved,
        "feature_names": before["feature_names"],
        "scaler": {"mean": [0.0] * 13, "std": [1.0] * 13},
    }
    outputs = [learned.predict_learned(bundle, value) for value in (before, after)]
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])


def test_full_metadata_and_saved_contracts_roundtrip(coupled_bundle, tmp_path):
    run = save_contracts(coupled_bundle, tmp_path)
    expected = learned._event_distribution_metadata(
        coupled_bundle["model"], coupled_bundle["config"]
    )
    metadata = json.loads((tmp_path / "learned_model.json").read_text())
    for key, value in expected.items():
        assert metadata[key] == run[key] == value
    for name in ("training_contract.json", "model_input_contract.json"):
        saved = json.loads((tmp_path / name).read_text())
        for key, value in expected.items():
            assert saved[key] == value
    assert learned.load_learned_bundle(run)["model"].path_components == 3


FIELDS = [
    ("path_distribution", "single"),
    ("path_components", 2),
    ("path_distribution_contract", {}),
    ("path_distribution_contract.parameters_per_component_horizon", 99),
    ("path_distribution_contract.component_count", 2),
    ("path_distribution_contract.head_layout", "wrong"),
    ("path_distribution_contract.component_association", "wrong"),
    ("path_distribution_contract.horizon_steps", 4),
    ("path_distribution_contract.horizons_s", [60.0, 120.0, 240.0]),
    ("path_distribution_contract.phase_covariance", "shared"),
    ("path_distribution_contract.path_head_outputs", 99),
    ("path_distribution_contract.shared_hidden_stem", False),
    ("path_distribution_contract.latent_scope", "wrong"),
    ("path_distribution_contract.joint_law", "wrong"),
    ("path_distribution_contract.survival_component_law", "wrong"),
    ("path_distribution_contract.already_red_component_law", "wrong"),
    ("path_distribution_contract.conditional_phase_score", "wrong"),
]


@pytest.mark.parametrize(
    "surface",
    ["learned_model.json", "manifest", "training_contract.json", "model_input_contract.json"],
)
@pytest.mark.parametrize("field,wrong", FIELDS)
@pytest.mark.parametrize("operation", ["missing", "contradictory"])
def test_valid_rehash_cannot_hide_path_contract_tampering(
    coupled_bundle, tmp_path, surface, field, wrong, operation
):
    run = save_contracts(coupled_bundle, tmp_path)
    saved = run if surface == "manifest" else json.loads((tmp_path / surface).read_text())
    pieces = field.split(".")
    parent = saved
    for key in pieces[:-1]:
        parent = parent[key]
    if operation == "missing":
        del parent[pieces[-1]]
    else:
        parent[pieces[-1]] = wrong
    if surface != "manifest":
        (tmp_path / surface).write_text(json.dumps(saved))
        run["artifacts"][surface] = sha256_file(tmp_path / surface)
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize("operation", ["missing", "contradictory"])
def test_coupled_top_level_head_size_is_mandatory(coupled_bundle, tmp_path, operation):
    run = save_contracts(coupled_bundle, tmp_path)
    saved = json.loads((tmp_path / "learned_model.json").read_text())
    if operation == "missing":
        del saved["path_head_outputs"]
    else:
        saved["path_head_outputs"] = 99
    (tmp_path / "learned_model.json").write_text(json.dumps(saved))
    run["artifacts"]["learned_model.json"] = sha256_file(tmp_path / "learned_model.json")
    with pytest.raises(ValueError, match="head"):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize("surface", ["learned_model.json", "manifest"])
def test_single_config_rejects_contradictory_coupled_labels(tmp_path, surface):
    cfg = config(selector="single")
    bundle = learned.fit_learned_model("gru", frame("train"), frame("val", 0.1), cfg)
    run = save_contracts(bundle, tmp_path)
    value = run if surface == "manifest" else json.loads((tmp_path / surface).read_text())
    value["path_distribution"] = "coupled_timing_mixture"
    if surface != "manifest":
        (tmp_path / surface).write_text(json.dumps(value))
        run["artifacts"][surface] = sha256_file(tmp_path / surface)
    with pytest.raises(ValueError, match="path"):
        learned.load_learned_bundle(run)


def test_old_single_missing_selector_preserves_fit_rng_loss_and_exact_reload(tmp_path):
    explicit = config(selector="single")
    missing = dict(explicit)
    del missing["path_distribution"]
    before = learned.fit_learned_model("gru", frame("train"), frame("val", 0.1), explicit)
    rng_after_explicit = torch.random.get_rng_state().clone()
    legacy = learned.fit_learned_model("gru", frame("train"), frame("val", 0.1), missing)
    assert torch.equal(rng_after_explicit, torch.random.get_rng_state())
    assert before["trace"] == legacy["trace"] and before["selection"] == legacy["selection"]
    for key, value in before["model"].state_dict().items():
        assert torch.equal(value, legacy["model"].state_dict()[key])
    assert "path_distribution" not in learned._event_distribution_metadata(legacy["model"], missing)
    artifacts = learned.save_learned_bundle(legacy, tmp_path)
    loaded = learned.load_learned_bundle(
        {"dir": tmp_path, "engine_id": "gru", "params": missing, "artifacts": artifacts}
    )
    assert loaded["model"].path_distribution == "single" and loaded["model"].path_components == 1
    outputs = [learned.predict_learned(x, frame("val", 0.1)) for x in (before, loaded)]
    for key in outputs[0]:
        np.testing.assert_array_equal(outputs[0][key], outputs[1][key])


@pytest.mark.parametrize("field,wrong", [("phase_covariance", "shared"), ("path_head_outputs", 99)])
def test_valid_rehash_input_top_fields_cannot_contradict_nested_contract(
    coupled_bundle, tmp_path, field, wrong
):
    run = save_contracts(coupled_bundle, tmp_path)
    name = "model_input_contract.json"
    value = json.loads((tmp_path / name).read_text())
    expected = copy.deepcopy(value["path_distribution_contract"])
    value[field] = wrong
    assert value["path_distribution_contract"] == expected
    (tmp_path / name).write_text(json.dumps(value))
    run["artifacts"][name] = sha256_file(tmp_path / name)
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)


@pytest.mark.parametrize(
    "field,value",
    [
        ("path_distribution", "coupled_timing_mixture"),
        ("path_components", 3),
        ("path_distribution_contract", {}),
    ],
)
def test_path_only_manifest_still_triggers_complete_provenance_validation(
    coupled_bundle, tmp_path, field, value
):
    run = save_contracts(coupled_bundle, tmp_path)
    run = {k: run[k] for k in ("dir", "engine_id", "params", "scaler", "artifacts")}
    run[field] = value
    with pytest.raises(ValueError):
        learned.load_learned_bundle(run)
