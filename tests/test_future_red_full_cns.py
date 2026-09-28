from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from pdm.future_red_full_cns import (
    _add_full_cns_baselines,
    _finalize_full_cns_run_manifest,
    load_full_cns_future_red_model,
    predict_full_cns_future_red_origins,
    prepare_full_cns_future_red,
    verify_full_cns_future_red_artifacts,
)
from pdm.models.full_cns import FullCNSReservoir


class TinyFullCNS:
    """Deterministic tiny fixture; production adapter still builds real MaleCNS."""

    def __init__(self):
        self.n_nodes = 4
        self.n_readout_features = 2
        self.is_synthetic = False
        self.provenance = {"graph_mode": "real_connectome", "graph_hash": "fixture", "n_nodes": 4}
        self.alpha = 0.2
        self.seed = 42
        self.spectral_radius = 0.9
        self.input_scale = 0.1
        self.graph_hash = "fixture"
        self.W_in = np.ones((4, 1), dtype=np.float32)
        self.b_res = np.zeros(4, dtype=np.float32)
        self.W_res = np.eye(4, dtype=np.float32)
        self.pool_index = np.array([0, 0, 1, 1], dtype=np.int32)
        self.calls = 0

    def pooled_trajectory(self, inputs, gap_before=None, *, should_stop=None):
        self.calls += 1
        values = np.asarray(inputs, dtype=np.float32)
        gaps = np.zeros(len(values), dtype=bool) if gap_before is None else np.asarray(gap_before, bool)
        state = 0.0
        result = []
        for index, row in enumerate(values):
            if gaps[index]:
                state = 0.0
            if should_stop and should_stop():
                raise InterruptedError
            state = 0.5 * state + float(self.W_in[0, 0] * row[2])
            result.append([state, -state])
        return np.asarray(result, dtype=np.float32)


def _fixture():
    rows = []
    for unit, baseline in (("train", 1.0), ("validation", 100.0), ("test", 1000.0)):
        for minute in range(4):
            row = {
                "unit_id": unit,
                "timestamp_s": float(minute * 60),
                "gap_before": minute == 0,
                "operating_age_s": float(minute * 60),
                "rpm": 2100.0,
                "load_kn": 12.0,
            }
            row.update({f"{channel}_{name}": baseline + minute for channel in ("horizontal", "vertical")
                        for name in ("rms", "std", "abs_peak", "peak_to_peak", "crest_factor", "kurtosis")})
            row.update({f"{channel}_band_{band}": baseline + minute + band for channel in ("horizontal", "vertical")
                        for band in range(4)})
            rows.append(row)
    features = pd.DataFrame(rows)
    units = pd.DataFrame({"unit_id": ["train", "validation", "test"],
                          "event_time_s": [600.0, 600.0, 600.0]})
    split = {"dataset_id": "bearings", "train": ["train"], "validation": ["validation"], "test": ["test"]}
    target_rows = []
    for unit in split["train"] + split["validation"] + split["test"]:
        part = "train" if unit == "train" else "validation" if unit == "validation" else "test"
        for i in range(4):
            known = i < 2
            target_rows.append({"unit_id": unit, "split": part, "timestamp_s": float(i * 60),
                                "target": (1 if i == 0 else 0) if known else None,
                                "target_known": known,
                                "first_red_timestamp_s": 60.0,
                                "at_risk": i < 3})
    targets = pd.DataFrame(target_rows)
    return {"features": features, "units": units, "split": split,
            "dataset_version": "tiny-v1", "fingerprint": {"snapshot": "tiny"}}, targets


def _mini_full_cns_reservoir(input_size):
    provenance = {"graph_hash": "fixture", "seed": 42, "n_nodes": 4,
                  "graph_mode": "real_connectome"}
    return FullCNSReservoir(
        np.full((4, input_size), 0.01, dtype=np.float32), sparse.eye(4, dtype=np.float32),
        np.zeros(4, dtype=np.float32), ["a", "b", "c", "d"], provenance,
        leak=0.2, time_scale_s=1.0, head="rul", pool_index=np.array([0, 0, 1, 1]),
    )


def test_full_cns_preparation_aligns_all_origins_and_masks_unknown_rows():
    data, targets = _fixture()
    prepared = prepare_full_cns_future_red(
        data,
        (targets, {"artifact_id": "targets-v1", "horizon_s": 1800.0, "targets_sha256": "abc",
                   "zone_policy": {"baseline_n_measurements": 2, "red_threshold_ratio": 1.5,
                                   "trend_points": 2}}),
        config={"feature_recipe": "base_v1", "features": {"log1p_features": []}},
        model=TinyFullCNS(),
    )

    assert prepared["X_train"].shape == (4, 2 + 22)
    assert prepared["X_validation"].shape == (4, 2 + 22)
    assert prepared["y_train"].tolist()[:2] == [1.0, 0.0]
    assert np.isnan(prepared["y_train"][2:]).all()
    assert prepared["rows_test"].target_known.tolist() == [True, True, False, False]
    assert prepared["rows_test"].at_risk.tolist() == [True, True, True, False]
    assert prepared["provenance"]["n_neurons_computed"] == 4
    assert prepared["provenance"]["model_population"] == "all_classified_neurons"
    assert "episode start" in prepared["provenance"]["history_contract"]
    assert "replay the preprocessed prefix" in prepared["provenance"]["checkpoint_replay_requirement"]
    # Standardization comes only from the declared training unit even though
    # validation and test sensor values are deliberately far outside its range.
    scaler = prepared["scaler"]
    assert scaler["fit_unit_ids"] == ["train"]
    assert scaler["preprocessor"]["time_scale_s"] == 1.0
    assert scaler["mean"][scaler["feature_names"].index("horizontal_rms")] == 2.5


def test_full_cns_baselines_share_origins_and_ignore_future_sensor_rows():
    data, targets = _fixture()
    prepared = prepare_full_cns_future_red(
        data,
        (targets, {"artifact_id": "targets-v1", "horizon_s": 1800.0, "targets_sha256": "abc",
                   "zone_policy": {"baseline_n_measurements": 2, "red_threshold_ratio": 1.5,
                                   "trend_points": 2}}),
        config={"feature_recipe": "base_v1", "features": {"log1p_features": []}},
        model=TinyFullCNS(),
    )
    _add_full_cns_baselines(prepared, data["features"])
    for baseline in prepared["baseline_probabilities"].values():
        for part in ("train", "validation", "test"):
            assert len(baseline[part]) == len(prepared[f"rows_{part}"])

    baseline_prefix = prepared["baseline_probabilities"]["trend_to_red"]["test"][:2].copy()
    changed_features = data["features"].copy()
    future = changed_features.unit_id.eq("test") & changed_features.timestamp_s.gt(60)
    changed_features.loc[future, ["horizontal_rms", "vertical_rms"]] = 1e9
    counterfactual = {key: value for key, value in prepared.items()
                      if key not in {"baseline_probabilities", "baseline_definitions"}}
    _add_full_cns_baselines(counterfactual, changed_features)
    assert np.array_equal(baseline_prefix,
                          counterfactual["baseline_probabilities"]["trend_to_red"]["test"][:2])


def test_full_cns_adapter_rejects_split_mismatch_and_synthetic_model():
    data, targets = _fixture()
    targets.loc[targets.unit_id == "test", "split"] = "train"
    try:
        prepare_full_cns_future_red(data, targets, model=TinyFullCNS())
    except ValueError as exc:
        assert "split" in str(exc).lower()
    else:
        raise AssertionError("expected a target split mismatch to fail")

    data, targets = _fixture()
    fake = TinyFullCNS()
    fake.is_synthetic = True
    try:
        prepare_full_cns_future_red(data, targets, model=fake)
    except ValueError as exc:
        assert "real MaleCNS" in str(exc)
    else:
        raise AssertionError("expected a synthetic model to fail")


def test_full_cns_cache_key_includes_seed_and_frozen_weights(tmp_path):
    data, targets = _fixture()
    kwargs = {"config": {"feature_recipe": "base_v1", "features": {"log1p_features": []}},
              "cache_dir": tmp_path}
    baseline_model = TinyFullCNS()
    baseline = prepare_full_cns_future_red(data, targets, model=baseline_model, **kwargs)
    assert baseline_model.calls == 3

    # Same graph, data, and scaler, but a different seed must miss all per-unit
    # cache entries even if this fixture's resulting weights are otherwise equal.
    seeded_model = TinyFullCNS()
    seeded_model.seed = 43
    prepare_full_cns_future_red(data, targets, model=seeded_model, **kwargs)
    assert seeded_model.calls == 3

    # The cache also tracks actual projection/operator arrays, so a changed
    # frozen sensor projection cannot reuse the baseline representation.
    weighted_model = TinyFullCNS()
    weighted_model.W_in[0, 0] = 2.0
    weighted = prepare_full_cns_future_red(data, targets, model=weighted_model, **kwargs)
    assert weighted_model.calls == 3
    assert not np.allclose(baseline["X_train"][:, :2], weighted["X_train"][:, :2])


def test_full_cns_readout_is_hashed_and_replay_matches_prepared_origins(tmp_path):
    data, targets = _fixture()
    prepared = prepare_full_cns_future_red(
        data, targets, config={"feature_recipe": "base_v1", "features": {"log1p_features": []}},
        model=_mini_full_cns_reservoir(22),
    )
    run = tmp_path
    body_dir = run / "full_cns"
    body = _mini_full_cns_reservoir(len(prepared["scaler"]["feature_names"]))
    body.graph_payload = {"n_nodes": 4, "node_order": body.node_order,
                          "edges": [], "recurrent_file": "recurrent.npz"}
    body.layout_payload = {"node_order": body.node_order, "positions": {}, "position_kind": {}}
    body.write_artifacts(body_dir)
    (body_dir / "model_meta.json").write_text(json.dumps({
        "n_nodes": 4, "graph_hash": "fixture", "leak": 0.2,
        "time_scale_s": 1.0, "head": "rul",
    }))
    (run / "full_cns_adapter.json").write_text(json.dumps(prepared["provenance"]))
    dim = prepared["X_train"].shape[1]
    layer = torch.nn.Linear(dim, 1)
    with torch.no_grad():
        layer.weight.copy_(torch.linspace(-0.1, 0.1, dim).reshape(1, -1))
        layer.bias.fill_(0.03)
    torch.save({"state_dict": layer.state_dict()}, run / "readout.pt")
    result = {"model_contract": prepared["model_contract"], "provenance": prepared["provenance"]}
    _finalize_full_cns_run_manifest(run, result)

    loaded = load_full_cns_future_red_model(run)
    replay = predict_full_cns_future_red_origins(loaded, data)
    expected_logits = prepared["X_test"] @ layer.weight.detach().numpy().reshape(-1) + layer.bias.item()
    expected = 1 / (1 + np.exp(-expected_logits))
    np.testing.assert_allclose(replay.loc[replay.unit_id.eq("test"), "probability"],
                               expected, rtol=0, atol=1e-7)
    changed = data["features"].copy()
    future = changed.unit_id.eq("test") & changed.timestamp_s.gt(60)
    changed.loc[future, ["horizontal_rms", "vertical_rms"]] = 1e9
    counterfactual = predict_full_cns_future_red_origins(loaded, changed)
    base_prefix = replay.loc[replay.unit_id.eq("test"), "probability"].to_numpy()[:2]
    changed_prefix = counterfactual.loc[counterfactual.unit_id.eq("test"), "probability"].to_numpy()[:2]
    np.testing.assert_array_equal(base_prefix, changed_prefix)
    assert loaded["contract"]["uses_rul_target"] is False
    assert loaded["contract"]["target_horizon_s"] == 1800.0
    assert "readout.pt" in verify_full_cns_future_red_artifacts(run)["artifact_hashes"]

    with (run / "readout.pt").open("ab") as checkpoint:
        checkpoint.write(b"tampered")
    try:
        verify_full_cns_future_red_artifacts(run)
    except ValueError as exc:
        assert "readout.pt" in str(exc)
    else:
        raise AssertionError("expected a readout hash mismatch to fail")
