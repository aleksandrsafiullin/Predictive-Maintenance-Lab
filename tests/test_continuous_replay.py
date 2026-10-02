from __future__ import annotations

import inspect
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.forecasting import predict_failure_interval
from pdm.models.reservoir import LeakyESN
from pdm.predict import Predictor
from pdm.preprocessing import Preprocessor
from pdm.replay import _continuous_bearing_predictions, replay_unit
from pdm.visualization.simulation import continuous_forecast_history, continuous_trace
from pdm.windows import recompute_filter_gap_before


@pytest.fixture
def continuous_case():
    frame = pd.DataFrame({"unit_id": "bearing", "timestamp_s": np.arange(30) * 60.0,
                          "operating_age_s": np.arange(30) * 60.0, "rpm": 1800, "load_kn": 4})
    for axis in ("horizontal", "vertical"):
        for stat in ("rms", "std", "abs_peak", "peak_to_peak", "crest_factor", "kurtosis",
                     "band_0", "band_1", "band_2", "band_3"):
            frame[f"{axis}_{stat}"] = np.linspace(0.1, 2.0, len(frame))
    frame["gap_before"] = False
    model = LeakyESN(torch.tensor([[.3], [-.2], [.1]]), torch.eye(3) * .25,
                     torch.zeros(3), state_mode="continuous", time_scale_s=600)
    model.node_order = ["a", "b", "c"]
    model.rul_transform = "log1p"
    model.readout.load_ridge_vector(np.array([.2, -.3, .4, -.1, 4.0]))
    prep = Preprocessor(["horizontal_rms"], [], [0], [1], 600,
                        fill_values={"horizontal_rms": 0}, dataset_id="bearings")
    predictor = Predictor(model, prep, history_length=3)
    profile = {"version": 1, "warmup_measurements": 3, "smoothing_tau_s": 300,
               "residual_q_low": -1.0, "residual_q_high": 2.0}
    return frame, predictor, profile


def _replay(frame, predictor, profile=None, truth=None):
    return replay_unit(frame, predictor, dataset_id="bearings", unit_id="bearing", run_id="run",
                       history_length=3, warning_horizon_s=600, forecast_profile=profile,
                       truth_units=truth)["predictions"]


def test_continuous_evaluation_matches_neural_prefix_forecast_and_gap_reset(continuous_case, monkeypatch):
    frame, predictor, profile = continuous_case
    frame.loc[12, "gap_before"] = True
    original = predictor.model.forward_states
    calls = Mock(wraps=original)
    monkeypatch.setattr(predictor.model, "forward_states", calls)
    # The batched path must not call the old O(T²) prefix predictor.
    monkeypatch.setattr(predictor, "predict_from_history", Mock(side_effect=AssertionError("window path used")))
    actual = _replay(frame, predictor, profile)
    assert calls.call_count == 2
    assert actual.prediction_status.iloc[[0, 1, 12, 13]].eq("Collecting history").all()
    assert actual.forecast_method.eq("continuous_empirical_interval").all()
    for i in (2, 8, 11, 14, 22, 29):
        trace = continuous_trace(frame.iloc[:i + 1], predictor.model, predictor.prep, 3)
        expected = predict_failure_interval(trace["timestamps_s"], trace["raw_rul_s"], profile).iloc[-1]
        for key in ("raw_rul_s", "predicted_rul_s", "lower_rul_s", "upper_rul_s"):
            assert actual.iloc[i][key] == pytest.approx(expected[key], rel=2e-6, abs=1e-3)


def test_continuous_batching_is_causal_and_does_not_consume_outcomes(continuous_case):
    frame, predictor, profile = continuous_case
    clean = _replay(frame, predictor, profile)
    poisoned = frame.copy()
    poisoned.loc[20:, "horizontal_rms"] = 1000000
    poisoned["actual_rul_s"] = -1e12
    poisoned["target_rul_s"] = 1e12
    truth = pd.DataFrame({"unit_id": ["bearing"], "event_time_s": [9999999], "event_observed": [1]})
    changed = _replay(poisoned, predictor, profile, truth)
    columns = ["timestamp_s", "raw_rul_s", "predicted_rul_s", "lower_rul_s", "upper_rul_s", "prediction_status"]
    pd.testing.assert_frame_equal(clean.loc[:19, columns], changed.loc[:19, columns])
    short = _replay(frame.iloc[:20], predictor, profile)
    pd.testing.assert_frame_equal(clean.loc[:19, columns], short[columns], rtol=2e-6, atol=1e-3)


def test_unprofiled_continuous_replay_preserves_raw_predictor_contract(continuous_case):
    frame, predictor, _ = continuous_case
    result = _replay(frame, predictor)
    assert "lower_rul_s" not in result
    assert result.forecast_method.eq("continuous_raw_readout").all()
    assert result.predicted_rul_s.iloc[:2].isna().all()
    ordinary = predictor.predict_from_history(frame)
    assert result.prediction_status.iloc[-1] == ordinary["status"] == "ok"
    assert result.predicted_rul_s.iloc[-1] == pytest.approx(ordinary["predicted_rul_s"])


def test_continuous_profile_warmup_must_match_saved_model(continuous_case):
    frame, predictor, profile = continuous_case
    with pytest.raises(ValueError, match="warmup"):
        _replay(frame, predictor, {**profile, "warmup_measurements": 2})


def _forbid_chart_shortcuts(monkeypatch):
    monkeypatch.setattr("pdm.predict.Predictor", Mock(side_effect=AssertionError("per-row Predictor used")))
    monkeypatch.setattr(
        "pdm.visualization.simulation.window_forecast_history",
        Mock(side_effect=AssertionError("window forecast used")),
    )
    monkeypatch.setattr(
        "pdm.forecasting.predict_failure_interval",
        Mock(side_effect=AssertionError("interval profile used")),
    )


def _assert_forecast_signature():
    names = set(inspect.signature(continuous_forecast_history).parameters)
    assert "show_gt" not in names
    assert "official_rul_at_prefix_end_s" not in names
    assert not any("visit" in name.lower() for name in names)
    for banned in ("predictions", "event_time", "event_time_s", "profile", "forecast_profile"):
        assert banned not in names


def _assert_earlier_segments_only(cache, trace, model):
    assert isinstance(cache, list) and cache
    active_ts = np.asarray(trace["timestamps_s"], dtype=float).reshape(-1)
    identity = (id(model), model.W_in._version, model.W_res._version, model.b_res._version, float(model.alpha))
    assert len(cache) == 1
    for entry in cache:
        assert entry["model_identity"] == identity
        stamps = np.asarray(entry["timestamps_s"], dtype=float).reshape(-1)
        assert entry["length"] == len(stamps) == len(entry["raw_rul_s"])
        assert entry["start_timestamp_s"] == float(stamps[0])
        assert not np.array_equal(stamps, active_ts)
        assert float(stamps[-1]) < float(active_ts[0])


@pytest.mark.parametrize("architecture", ["leaky", "full_cns"])
def test_continuous_forecast_reuses_traced_segment(continuous_case, monkeypatch, architecture):
    frame, predictor, _profile = continuous_case
    if architecture == "full_cns":
        from tests.test_future_red_full_cns import _mini_full_cns_reservoir

        model = _mini_full_cns_reservoir(1)
        assert model.pool_index is not None and model.pool_index.tolist() == [0, 0, 1, 1]
        model.rul_transform = "log1p"
        model.readout.load_ridge_vector(np.array([0.2, -0.3, 0.4, -0.1, 0.05, 4.0]))
    else:
        model = predictor.model
        assert isinstance(model, LeakyESN)
        assert not hasattr(model, "pool_index")
    prep, history_length = predictor.prep, predictor.history_length
    prefix = frame.copy()
    prefix.loc[12, "gap_before"] = True
    gap_at = int(np.flatnonzero(prefix["gap_before"].fillna(False).to_numpy(dtype=bool))[-1])
    post_gap = prefix.iloc[gap_at:].reset_index(drop=True)
    trace = continuous_trace(post_gap, model, prep, history_length)
    if architecture == "full_cns":
        assert "activity_history" in trace
    _assert_forecast_signature()
    reference = Predictor(model, prep, history_length)
    cached_segments = []
    with monkeypatch.context() as guarded:
        _forbid_chart_shortcuts(guarded)
        original = model.forward_states
        calls = Mock(wraps=original)
        guarded.setattr(model, "forward_states", calls)
        first = continuous_forecast_history(
            prefix, model, prep, history_length, trace=trace, cached_segments=cached_segments
        )
        assert calls.call_count == 1
        _assert_earlier_segments_only(cached_segments, trace, model)
        calls.reset_mock()
        second = continuous_forecast_history(
            prefix, model, prep, history_length, trace=trace, cached_segments=cached_segments
        )
        assert calls.call_count == 0
        _assert_earlier_segments_only(cached_segments, trace, model)
        short = prefix.iloc[:20].copy()
        short_result = continuous_forecast_history(
            short, model, prep, history_length, trace=trace, cached_segments=cached_segments
        )
        assert calls.call_count == 1
        _assert_earlier_segments_only(cached_segments, trace, model)
    assert list(second.columns) == ["timestamp_s", "raw_rul_s", "predicted_rul_s"]
    assert len(first) == len(second) == len(prefix)
    active = np.asarray(trace["raw_rul_s"], dtype=float).reshape(-1)
    np.testing.assert_array_equal(second["raw_rul_s"].to_numpy(dtype=float)[-len(active) :], active)
    predicted = second["predicted_rul_s"].to_numpy(dtype=float)
    raw = second["raw_rul_s"].to_numpy(dtype=float)
    assert np.isnan(predicted[: history_length - 1]).all()
    assert np.isnan(predicted[gap_at : gap_at + history_length - 1]).all()
    np.testing.assert_array_equal(predicted[history_length - 1 : gap_at], raw[history_length - 1 : gap_at])
    np.testing.assert_array_equal(predicted[gap_at + history_length - 1 :], raw[gap_at + history_length - 1 :])
    expected = _continuous_bearing_predictions(prefix, reference, None)
    np.testing.assert_allclose(
        predicted, expected["predicted_rul_s"].to_numpy(dtype=float), equal_nan=True, rtol=0, atol=0
    )
    short_end = float(short["timestamp_s"].iloc[-1])
    assert len(short_result) == len(short)
    assert np.all(short_result["timestamp_s"].to_numpy(dtype=float) <= short_end)
    np.testing.assert_array_equal(
        short_result["timestamp_s"].to_numpy(dtype=float), short["timestamp_s"].to_numpy(dtype=float)
    )


def test_filter_forecast_ignores_poisoned_gap_flags(monkeypatch):
    n = 14
    step = 6.0
    timestamps = np.arange(n) * step
    timestamps[8:] += 100.0
    prefix = pd.DataFrame({
        "dataset_id": ["filters"] * n,
        "unit_id": ["filter-a"] * n,
        "timestamp_s": timestamps,
        "operating_age_s": timestamps,
        "delta_t_s": np.r_[0.0, np.diff(timestamps)],
        "differential_pressure": np.linspace(10.0, 80.0, n),
        "delta_pressure": np.zeros(n),
        "flow_rate": np.full(n, 80.0),
        "dust_feed": np.full(n, 100.0),
        "dust": ["A3"] * n,
        "gap_before": np.zeros(n, dtype=bool),
    })
    assert str(prefix["dataset_id"].iloc[0]) == "filters"
    model = LeakyESN(
        torch.tensor([[0.3, 0.05], [-0.2, 0.1], [0.1, -0.05]]),
        torch.eye(3) * 0.25,
        torch.zeros(3),
        state_mode="continuous",
        time_scale_s=600,
    )
    model.node_order = ["a", "b", "c"]
    model.rul_transform = "log1p"
    model.readout.load_ridge_vector(np.array([0.2, -0.3, 0.4, -0.1, 0.05, 4.0]))
    history_length = 3
    prep = Preprocessor(
        ["differential_pressure", "delta_t_s"],
        [],
        [0.0, 0.0],
        [1.0, 1.0],
        600,
        fill_values={"differential_pressure": 0.0, "delta_t_s": 0.0},
        dataset_id="filters",
        gap_multiplier=2.5,
        sampling_interval_s=step,
    )
    assert prep.dataset_id == "filters"
    recomputed = recompute_filter_gap_before(
        prefix,
        gap_multiplier=prep.gap_multiplier,
        sampling_interval_s=prep.sampling_interval_s,
        causal=True,
    )
    stored = prefix["gap_before"].to_numpy(dtype=bool)
    causal = recomputed["gap_before"].to_numpy(dtype=bool)
    assert not np.array_equal(stored, causal)
    gap_at = int(np.flatnonzero(causal)[0])
    assert not bool(stored[gap_at])
    predictor = Predictor(model, prep, history_length)
    with monkeypatch.context() as guarded:
        _forbid_chart_shortcuts(guarded)
        result = continuous_forecast_history(prefix, model, prep, history_length)
    expected = []
    for index in range(len(prefix)):
        value = predictor.predict_from_history(prefix.iloc[: index + 1])["predicted_rul_s"]
        expected.append(np.nan if value is None else float(value))
    np.testing.assert_allclose(
        result["predicted_rul_s"].to_numpy(dtype=float),
        np.asarray(expected, dtype=float),
        equal_nan=True,
        rtol=0,
        atol=0,
    )
    assert result["predicted_rul_s"].iloc[gap_at : gap_at + history_length - 1].isna().all()

    def keep_stored(frame, **_kwargs):
        return frame.sort_values("timestamp_s").reset_index(drop=True)

    with monkeypatch.context() as guarded:
        _forbid_chart_shortcuts(guarded)
        guarded.setattr("pdm.windows.recompute_filter_gap_before", keep_stored)
        poisoned = continuous_forecast_history(prefix, model, prep, history_length)
    assert np.isfinite(poisoned["predicted_rul_s"].iloc[gap_at])
    assert not np.allclose(
        result["predicted_rul_s"].to_numpy(dtype=float),
        poisoned["predicted_rul_s"].to_numpy(dtype=float),
        equal_nan=True,
        rtol=0,
        atol=0,
    )
