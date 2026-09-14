from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.forecasting import predict_failure_interval
from pdm.models.reservoir import LeakyESN
from pdm.predict import Predictor
from pdm.preprocessing import Preprocessor
from pdm.replay import replay_unit
from pdm.visualization.simulation import continuous_trace


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
