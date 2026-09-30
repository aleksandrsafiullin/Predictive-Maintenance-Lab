"""Recursive inference must stay causal, bounded, and honest about crossing."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from pdm import signal_inference as inference


@pytest.fixture
def rollout_model(monkeypatch):
    data = {"features": pd.DataFrame({
        "unit_id": ["train"] * 6 + ["test"] * 9,
        "timestamp_s": list(range(6)) + list(range(9)),
        "signal": [0.0] * 15, "gap_before": [False] * 15}),
        "split": {"train": ["train"], "test": ["test"]}}
    run = {"snapshot_id": "snapshot", "params": {"history_length": 3, "horizons_s": [1., 2., 3.]},
           "schema": {"thresholds": {"mode": "absolute", "red": 5., "yellow": 2., "direction": "above"}},
           "engine_id": "gru", "scaler": {}, "artifact": "model", "artifacts": {"model": "digest"}}
    inputs = []

    def predict(model, engine, frame, scaler):
        inputs.append(frame["x"].copy())
        pred = frame["x"][:, -1, 0, None] + np.asarray(run["params"]["horizons_s"])
        return pred, pred - .5, pred + .5

    monkeypatch.setattr(inference, "load_signal_run", lambda *args: run)
    monkeypatch.setattr(inference, "load_snapshot", lambda *args: data)
    monkeypatch.setattr(inference, "_load_model", lambda *args: object())
    monkeypatch.setattr(inference, "_predict", predict)
    return data, run, inputs


def test_rollout_reaches_red_and_keeps_predicting_after_it(rollout_model):
    data, _, inputs = rollout_model
    result = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert result["crossing"] == {"status": "predicted", "time_s": 7.}
    assert result["points"][-1]["target_time_s"] >= 17.
    assert result["rollout"]["reason"] == "red_tail"
    np.testing.assert_equal(inputs[1], [[[1.], [2.], [3.]]])
    assert all(p["lower"] is None and p["upper"] is None for p in result["points"][3:])
    assert all(p["kind"] == "recursive" for p in result["points"][3:])
    # Both future values and their timestamps are display-only data.
    later = (data["features"].unit_id == "test") & (data["features"].timestamp_s > 2)
    data["features"].loc[later, "signal"] = 1e6
    data["features"].loc[later, "timestamp_s"] += 1000
    data["features"].loc[later, "gap_before"] = True
    again = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert again == result


def test_no_crossing_search_is_finite_and_horizon_is_configurable(rollout_model):
    _, run, _ = rollout_model
    run["schema"]["thresholds"]["red"] = 1000.
    short = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=12)
    long = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert short["crossing"] == {"status": "none_within_horizon", "time_s": None}
    assert short["points"][-1]["target_time_s"] == 14.
    assert long["points"][-1]["target_time_s"] == 32.
    assert long["rollout"]["reason"] == "search_limit"
    assert long["points"][:len(short["points"])] == short["points"]


def test_below_threshold_and_already_red_get_a_tail(rollout_model, monkeypatch):
    _, run, _ = rollout_model
    run["schema"]["thresholds"].update(red=-5., yellow=-2., direction="below")
    monkeypatch.setattr(inference, "_predict", lambda m, e, f, s: (
        f["x"][:, -1, 0, None] - np.array([1., 2., 3.]), None, None))
    result = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert result["crossing"]["time_s"] == 7.
    assert result["points"][-1]["target_time_s"] >= 17.
    run["schema"]["thresholds"]["red"] = 0.
    already = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert already["crossing"] == {"status": "already_red", "time_s": 2.}
    assert already["points"][-1]["target_time_s"] >= 12.


def test_sparse_heads_interpolate_inputs_but_plot_only_model_outputs(rollout_model):
    _, run, inputs = rollout_model
    run["params"]["horizons_s"] = [3.]
    result = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert result["rollout"]["interpolated_inputs"]
    np.testing.assert_equal(inputs[1], [[[1.], [2.], [3.]]])
    assert [p["target_time_s"] for p in result["points"]][:3] == [5., 8., 11.]


def test_nonfinite_continuation_stops_without_inventing_a_crossing(rollout_model, monkeypatch):
    _, run, _ = rollout_model
    run["schema"]["thresholds"]["red"] = 1000.
    predict = inference._predict
    count = 0

    def broken(*args):
        nonlocal count
        count += 1
        return predict(*args) if count == 1 else (np.full((1, 3), np.nan), None, None)

    monkeypatch.setattr(inference, "_predict", broken)
    result = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert result["rollout"]["status"] == "stopped"
    assert len(result["points"]) == 3
    assert result["crossing"]["time_s"] is None


def test_gaps_require_new_history_and_irregular_history_blocks_extension(rollout_model):
    data, _, _ = rollout_model
    data["features"].loc[8, "gap_before"] = True
    result = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert not result["points"] and "last gap" in result["reason"]
    data["features"].loc[8, "gap_before"] = False
    data["features"] = data["features"].astype({"timestamp_s": float})
    data["features"].loc[7, "timestamp_s"] = .5
    result = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    assert result["rollout"]["status"] == "unavailable"
    assert len(result["points"]) == 3


def test_thresholds_only_change_crossing_and_length_not_shared_predictions(rollout_model):
    _, run, _ = rollout_model
    low = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30)
    high_rule = copy.deepcopy(run["schema"]["thresholds"])
    high_rule["red"] = 1000.
    high = inference.forecast_prefix("p", "r", "test", 2., thresholds=high_rule, rollout_steps=30)
    assert high["points"][:len(low["points"])] == low["points"]


def test_progress_is_incremental_immutable_and_matches_final_predictions(rollout_model):
    updates = []
    result = inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30, progress_cb=updates.append)
    assert len(updates) > 1 and len(updates[0]["points"]) == 3
    assert updates[0]["crossing"]["status"] == "pending"
    assert updates[0]["rollout"]["status"] == "running"
    assert any(update["crossing"]["status"] == "predicted" for update in updates)
    for update in updates:
        assert update["points"] == result["points"][:len(update["points"])]


def test_cancellation_interrupts_rollout_and_reaches_model_kernel(rollout_model, monkeypatch):
    calls, stop = [], False
    predict = inference._predict

    def cancellable(*args, should_stop=None):
        calls.append(should_stop)
        return predict(*args)

    def cancel_after_first(_update):
        nonlocal stop
        stop = True

    monkeypatch.setattr(inference, "_predict", cancellable)
    with pytest.raises(InterruptedError, match="cancelled"):
        inference.forecast_prefix("p", "r", "test", 2., rollout_steps=30,
                                  should_stop=lambda: stop, progress_cb=cancel_after_first)
    assert all(check is not None for check in calls)
