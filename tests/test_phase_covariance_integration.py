"""Synthetic fitted-bundle checks; no real Bearings quality evidence."""

import copy
import json

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.io_util import sha256_file
from pdm.learned_trajectory import (
    fit_learned_model,
    learned_params,
    load_learned_bundle,
    predict_learned,
    save_learned_bundle,
)
from pdm.models.signal_distribution import SignalDistribution
from pdm.trajectory_data import build_trajectory_frame, build_trajectory_prefix


@pytest.mark.parametrize("engine", ["gru", "lstm"])
def test_separate_phase_fitted_bundle_reload_and_causal_prefix(tmp_path, engine):
    features = pd.DataFrame([
        {"unit_id": uid, "physical_unit_id": uid, "timestamp_s": float(i * 60),
         "signal": value, "gap_before": i == 0}
        for uid, values in (
            ("train-crossing", [.3, .4, .8, 1.5, 3.1, 4., 2., 5.]),
            ("train-censored", [.6, .5, .7, .5, .8, .6, .9, .8]),
            ("validation", [.5, .6, 1., 2., 3.2, 4., 2.5, 4.5]),
        )
        for i, value in enumerate(values)
    ])
    data = {"features": features, "schema": {"thresholds": {
        "mode": "absolute", "direction": "above", "red": 3.,
    }}}
    train_features = features[features.unit_id.str.startswith("train-")]
    config = learned_params(engine, {
        "phase_covariance": "separate", "batch_sampling": "physical_group",
        "history_length": 2, "horizons_s": [60., 120., 180.],
        "epochs": 2, "hidden_size": 16, "num_layers": 1, "rank": 2,
        "batch_size": 8, "training_samples": 8, "path_samples": 16,
        "cpu_threads": 1, "seed": 821, "dropout": 0.,
        "event_objective_horizons_s": [60., 180.],
    }, train_features)
    train = build_trajectory_frame(data, ["train-crossing", "train-censored"], config)
    validation = build_trajectory_frame(data, ["validation"], config)
    bundle = fit_learned_model(engine, train, validation, config)
    assert bundle["selection"]["restored_best_checkpoint"]
    assert bundle["selection"]["test_feedback"] is False
    assert all(np.isfinite(row["validation"]["total"]) for row in bundle["trace"])
    artifacts = save_learned_bundle(bundle, tmp_path)
    run = {"dir": tmp_path, "engine_id": engine, "params": config,
           "scaler": bundle["scaler"], "artifacts": artifacts}
    loaded = load_learned_bundle(run)
    expected, actual = predict_learned(bundle, validation), predict_learned(loaded, validation)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    assert loaded["phase_covariance"] == "separate"
    assert loaded["path_head_outputs"] == 3 * (6 + 3 * 2)
    prefix = features[features.unit_id == "validation"].iloc[:3]
    original = build_trajectory_prefix(data, prefix, config)
    changed = copy.deepcopy(data)
    changed["features"].loc[(features.unit_id == "validation") &
                            (features.timestamp_s > 120.), "signal"] = 9999.
    other = build_trajectory_prefix(changed, prefix, config)
    for key, value in predict_learned(loaded, original).items():
        np.testing.assert_array_equal(value, predict_learned(loaded, other)[key])
    metadata_path = tmp_path / "learned_model.json"
    metadata = json.loads(metadata_path.read_text())
    for field, value, message in (
        ("phase_covariance", "shared", "phase covariance"),
        ("path_head_outputs", 1, "path head size"),
    ):
        changed_metadata = {**metadata, field: value}
        metadata_path.write_text(json.dumps(changed_metadata))
        bound_artifacts = {**artifacts, "learned_model.json": sha256_file(metadata_path)}
        with pytest.raises(ValueError, match=message):
            load_learned_bundle({**run, "artifacts": bound_artifacts})


def test_separate_phase_unknown_targets_cannot_change_loss_or_gradients():
    torch.manual_seed(761)
    model = SignalDistribution("gru", 2, 4, hidden_size=8, num_layers=1,
                               rank=2, dropout=0., phase_covariance="separate")
    reference = copy.deepcopy(model)
    mask = torch.tensor([[True, True, False, False], [False] * 4])
    target = torch.tensor([[.6, .7, 0., 0.], [0.] * 4])
    hidden_changed = target.masked_fill(~mask, float("nan"))
    evidence = {"no_entry_prefix": torch.tensor([2, 0]), "n_samples": 8,
                "path_objective_prefix_lengths": [2, 4],
                "event_objective_prefix_lengths": [2, 4]}
    results = []
    for candidate, actual in ((model, target), (reference, hidden_changed)):
        moments = candidate(torch.ones(2, 2, 2), torch.tensor([.5, .5]))
        terms = candidate.objective(moments, actual, mask, 3.,
                                    generator=torch.Generator().manual_seed(46), **evidence)
        terms["total"].sum().backward()
        results.append(terms)
    for key in results[0]:
        torch.testing.assert_close(results[0][key], results[1][key], rtol=0., atol=0.)
    for left, right in zip(model.parameters(), reference.parameters()):
        assert torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad, rtol=0., atol=0.)
