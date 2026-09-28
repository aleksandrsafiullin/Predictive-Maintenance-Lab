from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from pdm.future_red_models import (
    build_future_red_baselines,
    evaluate_future_red_baselines,
    event_level_metrics,
    fit_future_red_model,
    fit_train_scaler,
    load_future_red_model,
    predict_future_red_origins,
    prepare_future_red_sequences,
    select_validation_threshold,
    transform_future_red_features,
    verify_future_red_reservoir_artifacts,
)


def test_future_red_scaler_uses_only_train_units_and_forbids_rul_inputs():
    features = pd.DataFrame({"unit_id": ["a", "a", "b"], "timestamp_s": [0, 1, 0],
                             "sensor": [1.0, 3.0, 1000.0]})
    scaler = fit_train_scaler(features, ["a"], ["sensor"])
    assert scaler["mean"] == [2.0]
    assert scaler["scale"] == [1.0]
    with pytest.raises(ValueError, match="target leakage"):
        fit_train_scaler(features.assign(RUL=0), ["a"], ["RUL"])


def test_sequences_are_causal_mask_unknown_labels_and_stop_at_gaps():
    features = pd.DataFrame({
        "unit_id": ["a"] * 5,
        "timestamp_s": [0, 1, 2, 10, 11],
        "sensor": [0.0, 1.0, 2.0, 100.0, 101.0],
        "gap_before": [False, False, False, True, False],
    })
    targets = pd.DataFrame({
        "unit_id": ["a"] * 5, "timestamp_s": [0, 1, 2, 10, 11],
        "split": ["train"] * 5, "at_risk": [True] * 5,
        "target": [0, 0, 1, np.nan, 1],
        "target_known": [True, True, True, False, True],
        "first_red_timestamp_s": [2, 2, 2, 2, 2],
    })
    prepared = prepare_future_red_sequences(features, targets, {"train": ["a"], "validation": [],
                                                                "test": []},
                                            history_length=3, feature_names=["sensor"])
    assert prepared["X_train"].shape == (5, 3, 1)
    # At t=2, the window contains only [0, 1, 2], so no future value leaks backward.
    mean, scale = prepared["scaler"]["mean"][0], prepared["scaler"]["scale"][0]
    np.testing.assert_allclose(prepared["X_train"][2, :, 0],
                               (np.array([0.0, 1.0, 2.0]) - mean) / scale, atol=1e-6)
    # The first row after the gap is left padded from its new segment, excluding t=2.
    expected_after_gap = (100.0 - mean) / scale
    np.testing.assert_allclose(prepared["X_train"][3, :, 0], [expected_after_gap] * 3, atol=1e-6)
    assert np.isnan(prepared["y_train"][3])
    assert prepared["rows_train"].target_known.tolist() == [True, True, True, False, True]


def test_threshold_and_event_metrics_leave_censored_non_events_unresolved():
    threshold, info = select_validation_threshold(np.array([0, 0, 1, 1]),
                                                  np.array([0.1, 0.2, 0.8, 0.9]))
    assert info["selection"] == "validation_max_f1"
    assert threshold <= 0.8
    rows = pd.DataFrame({
        "unit_id": ["event", "event", "event", "censored"],
        "timestamp_s": [0.0, 5.0, 10.0, 0.0],
        "target": [0.0, np.nan, 1.0, 0.0],
        "target_known": [True, False, True, True],
        "first_red_timestamp_s": [10.0, 10.0, 10.0, np.nan],
    })
    summary = event_level_metrics(rows, np.array([0.9, 0.1, 0.1, 0.99]), threshold=0.5)
    assert summary["event_observed_units"] == 1
    assert summary["missed_event_units"] == 1
    assert summary["unknown_censored_units"] == 1
    assert summary["early_or_unmatched_warning_units"] == 1
    assert summary["alerted_censored_unknown_units"] == 1


def test_baselines_align_to_all_origins_and_use_causal_trend():
    features = pd.DataFrame({"unit_id": ["u"] * 3, "timestamp_s": [0.0, 1.0, 2.0],
                             "differential_pressure": [500.0, 550.0, 580.0],
                             "gap_before": [False, False, False]})
    rows = pd.DataFrame({"unit_id": ["u"] * 3, "timestamp_s": [0.0, 1.0, 2.0],
                         "target": [0.0, 1.0, np.nan], "target_known": [True, True, False],
                         "at_risk": [True, True, False], "first_red_timestamp_s": [1.5] * 3})
    prepared = {"rows_train": rows, "rows_validation": rows.iloc[0:0], "rows_test": rows.iloc[0:0],
                "target_manifest": {"dataset_id": "filters"}}
    baselines = build_future_red_baselines("filters", features, prepared, horizon_s=20,
                                           zone_policy={"red_limit_pa": 600.0})
    assert len(baselines["trend_to_red"]["train"]) == len(rows)
    assert baselines["trend_to_red"]["train"].tolist() == [0.0, 1.0, 1.0]
    scores = evaluate_future_red_baselines(prepared, baselines)
    assert scores["trend_to_red"]["test"]["test_event_recall"] == "not_reported_by_design"


def test_filter_feature_transform_does_not_include_row_or_target_metadata():
    raw = pd.DataFrame({"unit_id": ["a", "b"], "timestamp_s": [0.0, 0.1],
                        "operating_age_s": [0.0, 0.1], "delta_t_s": [0.0, 0.1],
                        "differential_pressure": [10.0, 20.0], "delta_pressure": [0.0, 10.0],
                        "flow_rate": [1.0, 1.0], "dust_feed": [0.1, 0.1],
                        "dust": ["low", "high"], "gap_before": [False, False],
                        "observation_end_s": [0.0, 0.1], "event_time_s": [0.0, 0.1],
                        "RUL": [0.0, 0.0]})
    split = {"train": ["a"], "validation": ["b"], "test": []}
    encoded, names = transform_future_red_features("filters", raw, split)
    assert "differential_pressure" in names
    assert "dust__low" in names
    assert "timestamp_s" not in names
    assert "unit_id" not in names
    assert "operating_age_s" not in names
    assert "delta_t_s" in names
    assert not {"observation_end_s", "event_time_s", "rul"} & {n.lower() for n in names}
    assert "dust__high" not in names
    assert encoded["dust__low"].tolist() == [1.0, 0.0]


def test_prepared_future_red_inputs_contain_only_sensor_features():
    raw = pd.DataFrame({"unit_id": ["a", "a", "b", "b"],
                        "timestamp_s": [0.0, 0.1, 0.0, 0.1],
                        "operating_age_s": [0.0, 0.1, 0.0, 0.1],
                        "delta_t_s": [0.0, 0.1, 0.0, 0.1],
                        "differential_pressure": [10.0, 20.0, 30.0, 40.0],
                        "delta_pressure": [0.0, 10.0, 0.0, 10.0],
                        "flow_rate": [1.0] * 4, "dust_feed": [0.1] * 4,
                        "dust": ["low"] * 4, "gap_before": [False] * 4,
                        "observation_end_s": [0.1] * 4, "event_time_s": [0.1] * 4,
                        "RUL": [0.0] * 4})
    split = {"train": ["a"], "validation": ["b"], "test": []}
    encoded, names = transform_future_red_features("filters", raw, split)
    targets = pd.DataFrame({"unit_id": raw.unit_id, "timestamp_s": raw.timestamp_s,
                            "split": ["train", "train", "validation", "validation"],
                            "target": [0.0, 1.0, 0.0, 1.0], "target_known": [True] * 4,
                            "at_risk": [True] * 4,
                            "first_red_timestamp_s": [0.1, 0.1, 0.1, 0.1],
                            "observation_end_s": [0.1] * 4, "event_time_s": [0.1] * 4,
                            "RUL": [0.0] * 4})
    prepared = prepare_future_red_sequences(encoded, targets, split, history_length=2,
                                            feature_names=names)

    input_names = {name.lower() for name in prepared["feature_names"]}
    assert "differential_pressure" in input_names
    assert "delta_t_s" in input_names
    assert not input_names & {"operating_age_s", "observation_end_s", "event_time_s", "rul"}
    assert prepared["X_train"].shape[-1] == len(prepared["feature_names"])
    assert prepared["X_validation"].shape[-1] == len(prepared["feature_names"])


def test_recurrent_binary_trainer_saves_frozen_split_predictions(tmp_path):
    def rows(part):
        return pd.DataFrame({"unit_id": [part + "0", part + "1"], "split": [part] * 2,
                             "timestamp_s": [0.0, 1.0], "target": [0, 1],
                             "target_known": [True, True], "at_risk": [True, True],
                             "first_red_timestamp_s": [np.nan, 1.0]})

    prepared = {"feature_names": ["sensor"], "scaler": {"mean": [0.0], "scale": [1.0]},
                "target_manifest": {"dataset_id": "bearings"}}
    for part in ("train", "validation", "test"):
        prepared[f"rows_{part}"] = rows(part)
        prepared[f"X_{part}"] = np.array([[[0.0], [0.0]], [[1.0], [1.0]]], dtype=np.float32)
        prepared[f"y_{part}"] = np.array([0.0, 1.0], dtype=np.float32)
    result = fit_future_red_model("bearings", "gru", prepared=prepared, output_dir=tmp_path,
                                  hidden_size=4, epochs=2, batch_size=2, patience=2)
    saved = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(saved) == 6
    assert set(saved.split) == {"train", "validation", "test"}
    assert result["threshold_selection"]["selection_unit"] == "origin_rows"
    assert (tmp_path / "model.pt").is_file()
    loaded, contract = load_future_red_model(tmp_path / "model.pt")
    replay = predict_future_red_origins(loaded, prepared, split="test", model_contract=contract)
    expected = saved.loc[saved.split.eq("test"), "probability"].to_numpy()
    np.testing.assert_allclose(replay, expected, rtol=0, atol=1e-7)
    assert contract["history_length"] == 2
    assert result["model_contract"] == contract
    assert result["artifact_hashes"]["model.pt"]
    with (tmp_path / "model.pt").open("ab") as checkpoint:
        checkpoint.write(b"tampered")
    with pytest.raises(ValueError, match="checkpoint hash"):
        load_future_red_model(tmp_path / "model.pt")


def test_reservoir_artifact_verifier_checks_manifest_hashes(tmp_path):
    artifacts = ("reservoir_weights.npz", "graph.json", "readout.pt")
    hashes = {}
    for name in artifacts:
        path = tmp_path / name
        path.write_bytes(name.encode())
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "model_contract": {"version": "future_red_replay_v1",
                           "architecture": "fly_connectome_reservoir"},
        "artifact_hashes": hashes,
    }))
    assert verify_future_red_reservoir_artifacts(tmp_path)["artifact_hashes"] == hashes
    (tmp_path / "graph.json").write_text("changed")
    with pytest.raises(ValueError, match="graph.json"):
        verify_future_red_reservoir_artifacts(tmp_path)
