from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.connectome.sources import load_synthetic_fixture
from pdm.models import FlyConnectomeReservoir
from pdm.predict import Predictor
from pdm.preprocessing import Preprocessor
from pdm.visualization.overlay import build_work_overlay_figure
from pdm.visualization.simulation import continuous_trace, simulate_step


def setup_model():
    model = FlyConnectomeReservoir(load_synthetic_fixture().graph, input_size=1,
                                  state_mode="continuous", seed=4, time_scale_s=100)
    prep = Preprocessor(feature_names=["horizontal_rms"], log1p_features=[],
                        scaler_mean=[0.0], scaler_scale=[1.0], time_scale_s=100,
                        fill_values={"horizontal_rms": 0.0}, dataset_id="bearings")
    return model, prep


def test_continuous_trace_prefix_cache_rewind_and_real_processing(tiny_bearing_tables):
    feat, _ = tiny_bearing_tables
    model, prep = setup_model()
    a = simulate_step(feat, "Bearing1_1", 1200, model, prep, 5)
    b = simulate_step(feat, "Bearing1_1", 1260, model, prep, 5, previous_trace=a)
    full = simulate_step(feat, "Bearing1_1", 1260, model, prep, 5)
    np.testing.assert_allclose(b["states"], full["states"], atol=1e-7)
    assert len(b["states"]) == 21
    changed = feat.copy()
    changed.loc[changed.timestamp_s > 1200, "horizontal_rms"] = 1e9
    unchanged = simulate_step(changed, "Bearing1_1", 1200, model, prep, 5, previous_trace=b)
    np.testing.assert_array_equal(a["states"], unchanged["states"])
    np.testing.assert_array_equal(a["raw_rul_s"], unchanged["raw_rul_s"])
    proc = b["processing"]
    rebuilt = ((1-proc["leak"])*proc["previous_state"][-1] + proc["leak"] *
               np.tanh(proc["input_drive"][-1]+proc["recurrent_drive"][-1]+proc["bias"]))
    np.testing.assert_allclose(rebuilt, b["states"][-1], atol=1e-7)
    contrib = b["contributions"]
    np.testing.assert_allclose(contrib["raw"], contrib["intercept"] + contrib["input"].sum(axis=1)
                               + contrib["neuron"].sum(axis=1), atol=1e-6)
    with torch.no_grad():
        model.W_in.add_(0.1)
    cached = simulate_step(feat, "Bearing1_1", 1320, model, prep, 5, previous_trace=b)
    fresh = simulate_step(feat, "Bearing1_1", 1320, model, prep, 5)
    np.testing.assert_array_equal(cached["states"], fresh["states"])


def test_continuous_predictor_contract_empty_warmup_and_gap(tiny_bearing_tables):
    feat, _ = tiny_bearing_tables
    model, prep = setup_model()
    unit = feat[feat.unit_id == "Bearing1_1"].copy()
    predictor = Predictor(model, prep, history_length=5)
    assert predictor.predict_from_history(unit.iloc[:0])["status"] == "Collecting history"
    early = continuous_trace(unit.iloc[:1], model, prep, 5)
    assert early["status"] == "Collecting history"
    assert early["states"].shape == (1, model.n_nodes)
    normal = predictor.predict_from_history(unit.iloc[:20])
    traced = predictor.predict_from_history(unit.iloc[:20], with_trace=True)
    assert normal["status"] == "ok"
    assert normal["predicted_rul_s"] == traced["predicted_rul_s"]
    unit.loc[unit.index[15], "gap_before"] = True
    after = continuous_trace(unit.iloc[:20], model, prep, 5)
    reset = continuous_trace(unit.iloc[15:20], model, prep, 5)
    np.testing.assert_array_equal(after["states"], reset["states"])
    assert after["n_history"] == 5
    torch.testing.assert_close(model.forward_states(torch.from_numpy(reset["inputs"])), torch.from_numpy(after["states"]))


def test_failure_interval_chart_band_and_gt_independence(tiny_bearing_tables):
    feat, _ = tiny_bearing_tables
    points = pd.DataFrame({"timestamp_s": [1200, 1260], "predicted_rul_s": [900, 850],
                           "lower_rul_s": [400, 390], "upper_rul_s": [1400, 1300]})
    kwargs = dict(dataset_id="bearings", unit_features=feat[feat.unit_id == "Bearing1_1"],
                  now_timestamp_s=1260, predicted_rul_s=850, history_length=21,
                  event_time_s=1800, event_observed=True, prediction_interval_s=(390, 1300),
                  predicted_rul_by_time=points, actual_rul_s=np.array([600, 540]))
    hidden = build_work_overlay_figure(**kwargs, show_gt=False)
    shown = build_work_overlay_figure(**kwargs, show_gt=True)
    for key in ("predicted_zone_x0", "predicted_zone_x1", "stored_predicted_rul_s", "interval_lower_s", "interval_upper_s"):
        assert hidden.layout.meta[key] == shown.layout.meta[key]
    assert shown.layout.meta["predicted_zone_x0"] == (1260+390)/60
    assert shown.layout.meta["predicted_zone_x1"] == pytest.approx((1260+1300)/60)
    band = next(trace for trace in shown.data if trace.name == "empirical forecast interval")
    assert band.fill == "tonexty"
