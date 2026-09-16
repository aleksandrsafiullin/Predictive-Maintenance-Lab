from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.feature_recipes import enrich_features
from pdm.losses import weibull_nll, weibull_nll_seconds
from pdm.training_protocol import FullPassSampler, TrainingControl, protocol, window_weights


def test_baseline_cohort_join_tolerates_csv_noise_but_never_fills_missing_measurements():
    from pdm.study_baselines import align_cohort_features

    reference = pd.DataFrame({"unit_id": ["a", "a"], "timestamp_s": [966., 984.]})
    source = pd.DataFrame({"unit_id": ["a", "a"], "timestamp_s": [966.0000000000001, 983.9999999999999],
                           "sensor": [7., 8.]})
    assert align_cohort_features(reference, source).sensor.tolist() == [7., 8.]
    with pytest.raises(ValueError, match="no exact source measurement"):
        align_cohort_features(reference, source.iloc[:1])
    with pytest.raises(pd.errors.MergeError):
        align_cohort_features(reference, pd.concat([source, source.iloc[:1]]))


def test_simple_baseline_reports_missing_prefix_endpoints():
    from pdm.study_report import score_frame

    frame = pd.DataFrame({"unit_id": ["a", "a", "b", "b"], "timestamp_s": [0., 6., 0., 6.],
                          "prediction_status": "ok", "predicted_rul_s": [10., np.nan, 30., 20.],
                          "actual_rul_s": [16., 10., 26., 20.]})
    result = score_frame(frame, "filters", "test")
    assert result["primary_score"] == 0.
    assert result["primary_points_available"] == 1
    assert result["primary_points_expected"] == 2
    assert result["primary_coverage"] == .5
    assert result["prediction_coverage"] == .75


def test_stop_waits_for_minimum_and_meaningful_improvement():
    cfg = protocol("bearings")
    state = TrainingControl()
    assert not state.update(100, 1, cfg)
    for epoch in range(2, 21):
        assert not state.update(99.5, epoch, cfg)
    assert state.update(99.5, 21, cfg)
    assert state.stop_reason == "early_stopping"
    state = TrainingControl()
    cfg = protocol("bearings", mode="diagnostic")
    for epoch in range(1, 100):
        assert not state.update(100, epoch, cfg)
    assert state.update(100, 100, cfg)
    assert state.stop_reason == "max_epochs"


def test_full_pass_and_unit_near_weights():
    index = [("a", 0, 19, x, x, 1) for x in (60, 120, 4000)]
    index += [("b", 0, 19, x, x, 1) for x in (4000, 5000)]
    sampler = FullPassSampler(index, 42)
    sampler.set_epoch(3)
    order = list(sampler)
    assert sorted(order) == list(range(5))
    assert list(sampler) == order
    w = window_weights(index, .5)
    assert w[:3].sum() == pytest.approx(w[3:].sum())
    assert w.mean() == pytest.approx(1)
    assert w[0] > w[2]
    assert w[3] == pytest.approx(w[4])


def test_short_last_batch_preserves_total_window_weights():
    from pdm.training_protocol import weighted_batch_loss

    cfg = protocol("bearings", sampling="full_pass")
    index = [("a", 0, 19, 60, 0, 1)] * 32 + [("b", 0, 19, 60, 0, 1)] * 3
    weights = torch.tensor(window_weights(index))
    predictions = torch.ones(35, requires_grad=True)
    objective = weighted_batch_loss(predictions[:32], weights[:32], cfg)
    objective += weighted_batch_loss(predictions[32:], weights[32:], cfg)
    objective.backward()
    assert predictions.grad[:32].sum() == pytest.approx(predictions.grad[32:].sum())


def test_weibull_density_units_and_gradients():
    duration = torch.tensor([1e-6, 1.0, 1000.0], dtype=torch.float64)
    scale = torch.tensor([1.0, 10.0, 500.0], dtype=torch.float64, requires_grad=True)
    shape = torch.tensor([.4, 2.0, 8.0], dtype=torch.float64, requires_grad=True)
    event = torch.tensor([0., 1., 1.])
    got = weibull_nll_seconds(duration, event, scale, shape)
    ref = -torch.where(event.bool(), torch.distributions.Weibull(scale, shape).log_prob(duration), -(duration/scale)**shape)
    torch.testing.assert_close(got, ref)
    got.sum().backward()
    assert torch.isfinite(scale.grad).all() and torch.isfinite(shape.grad).all()
    normalized = weibull_nll(duration, event, scale / 60, shape, 60)
    torch.testing.assert_close(got, normalized + event * np.log(60))


def test_nonfinite_raw_readout_is_not_hidden_by_log_clipping():
    from pdm.forecasting import score_readout

    rows = pd.DataFrame({"unit_id": ["unit-a"], "timestamp_s": [1200.], "gap_before": [True]})
    with pytest.raises(FloatingPointError, match="unit-a.*1200"):
        score_readout(np.ones((1, 2)), rows, np.array([np.inf, 0.]), 60., {"rul_transform": "log1p"})


@pytest.mark.parametrize("dataset_id", ["bearings", "filters"])
def test_feature_prefix_parity_and_gap(dataset_id):
    n = 45
    x = np.arange(n, dtype=float)
    frame = pd.DataFrame({"unit_id": "a", "timestamp_s": x * 6, "gap_before": x == 25,
                          "differential_pressure": 100 + x*x, "dust_feed": 2.0})
    for channel in ("horizontal", "vertical"):
        for name in ("rms", "abs_peak", "kurtosis"):
            frame[f"{channel}_{name}"] = x + 1
    all_features = enrich_features(frame, dataset_id, "degradation_v1")
    names = list(set(all_features) - set(frame))
    for size in range(1, n + 1):
        prefix = enrich_features(frame.iloc[:size], dataset_id, "degradation_v1")
        np.testing.assert_allclose(prefix[names], all_features.iloc[:size][names], atol=1e-10)
    slopes = [c for c in names if "slope" in c]
    assert (all_features.loc[25, slopes] == 0).all()
    if dataset_id == "filters":
        assert all_features.loc[25, "dust_accumulated_internal"] == 0
        assert all_features.loc[24, "dust_accumulated_internal"] == 288


@pytest.mark.parametrize("seed", [42, 43])
def test_training_resume_matches_uninterrupted(tmp_path, monkeypatch, tiny_bearing_tables, seed):
    from pdm import train
    from pdm import training_engine as engine
    from pdm.splits import split_hash

    features, units = tiny_bearing_tables
    split = {"dataset_id": "bearings", "train": ["Bearing1_1", "Bearing1_2", "Bearing1_3"],
             "validation": ["Bearing1_4"], "test": ["Bearing1_5"]}
    processed = {"features": features, "units": units, "split": split}
    monkeypatch.setattr(engine, "load_processed", lambda _: processed)
    monkeypatch.setattr(engine, "training_admission", lambda *args: (split, []))
    monkeypatch.setattr(engine, "dataset_runs", lambda _: tmp_path)
    monkeypatch.setattr(engine, "dataset_fingerprint_for_run", lambda p, s: {"dataset_version": "fixture", "split_hash": split_hash(s)})
    monkeypatch.setattr(train, "_experiment_snapshot", lambda *args: {})
    cfg = protocol("bearings", mode="adaptive", sampling="full_pass", feature_recipe="degradation_v1")
    cfg.update(max_epochs=4, min_epochs=2)
    engine.train_v2("bearings", architecture="gru", training_protocol=cfg, run_id_override="full", seed=seed)
    stopped = {"value": False}

    def progress(status):
        if status.get("epoch") == 2:
            stopped["value"] = True

    engine.train_v2("bearings", architecture="gru", training_protocol=cfg, run_id_override="resumed", seed=seed,
                    status_cb=progress, should_stop=lambda: stopped["value"])
    engine.train_v2("bearings", architecture="gru", training_protocol=cfg, resume_run_id="resumed")
    a = torch.load(tmp_path / "full/last.pt", weights_only=False)
    b = torch.load(tmp_path / "resumed/last.pt", weights_only=False)
    for key in a["model_state_dict"]:
        torch.testing.assert_close(a["model_state_dict"][key], b["model_state_dict"][key], rtol=0, atol=0)
    assert a["training_state"]["scheduler"] == b["training_state"]["scheduler"]
    assert a["training_state"]["control"] == b["training_state"]["control"]


def test_validation_object_without_windows_cannot_disappear(monkeypatch, tiny_bearing_tables):
    from pdm import training_engine as engine

    features, units = tiny_bearing_tables
    short = features[features.unit_id.eq("Bearing1_4")].iloc[:10]
    features = pd.concat([features[~features.unit_id.eq("Bearing1_4")], short])
    split = {"train": ["Bearing1_1", "Bearing1_2", "Bearing1_3"],
             "validation": ["Bearing1_4", "Bearing1_5"], "test": []}
    monkeypatch.setattr(engine, "load_processed", lambda _: {"features": features, "units": units, "split": split})
    monkeypatch.setattr(engine, "training_admission", lambda *args: (split, []))
    with pytest.raises(ValueError, match="no eligible windows.*Bearing1_4"):
        engine.train_v2("bearings", architecture="gru", training_protocol=protocol("bearings"))


def test_constant_zero_is_not_a_useful_warning_and_censored_is_unknown():
    from pdm.study_metrics import useful_warning_metrics, warning_settings

    frame = pd.DataFrame({"unit_id": "a", "timestamp_s": np.arange(100) * 60.,
                          "predicted_rul_s": 0., "prediction_status": "ok"})
    units = pd.DataFrame({"unit_id": ["a"], "event_time_s": [5940.], "observation_end_s": [5940.], "event_observed": [1]})
    score, episodes = useful_warning_metrics(frame, units, warning_settings("bearings"))
    assert score["timely_recall"] == 0
    assert score["useful_precision"] == 0
    assert episodes.outcome.tolist() == ["too_early"]
    # A short observed series can accidentally place an always-on alarm in the
    # useful band. It still cannot qualify a constant-zero model as useful.
    short = frame.iloc[-25:].copy()
    short_score, _ = useful_warning_metrics(short, units, warning_settings("bearings"))
    assert short_score["timely_recall"] == 1
    assert short_score["degenerate_constant_zero"]
    assert not short_score["warning_goal_met"]
    units.loc[0, "event_observed"] = 0
    units.loc[0, "event_time_s"] = np.nan
    frame = frame.iloc[-4:]
    score, episodes = useful_warning_metrics(frame, units, warning_settings("bearings"))
    assert episodes.outcome.tolist() == ["unknown"]
    assert score["useful_precision"] is None
    assert not score["warning_goal_met"]


def test_unscorable_failure_stays_in_warning_target_denominator():
    from pdm.study_metrics import useful_warning_metrics, warning_settings

    first = pd.DataFrame({"unit_id": "a", "timestamp_s": np.arange(100) * 60.,
                          "predicted_rul_s": 5000., "prediction_status": "ok"})
    first.loc[first.timestamp_s >= 4200, "predicted_rul_s"] = 1000.
    second = first.copy().assign(unit_id="b", prediction_status="Collecting history", predicted_rul_s=np.nan)
    units = pd.DataFrame({"unit_id": ["a", "b"], "event_time_s": [5940.] * 2,
                          "observation_end_s": [5940.] * 2, "event_observed": [1] * 2})
    result, _ = useful_warning_metrics(pd.concat([first, second]), units, warning_settings("bearings"))
    assert result["scorable_timely_recall"] == 1
    assert result["timely_recall"] == .5
    assert result["unscorable_warning_units"] == 1
    assert not result["warning_goal_met"]


def test_feature_recipe_replay_matches_ordinary_prediction(tiny_filter_tables):
    from pdm.config import load_dataset_config
    from pdm.models import build_model
    from pdm.predict import Predictor
    from pdm.preprocessing import apply_preprocessor, fit_preprocessor
    from pdm.replay import replay_unit
    from pdm.visualization.simulation import window_forecast_history

    features, units = tiny_filter_tables
    split = {"train": ["Filter_1", "Filter_3"], "validation": ["Filter_2"], "test": ["Filter_101"]}
    cfg = load_dataset_config("filters")
    cfg["feature_recipe"] = "degradation_v1"
    prep, _ = fit_preprocessor("filters", features, units, split, cfg)
    model = build_model(architecture="gru", input_size=len(prep.feature_names), head="weibull", time_scale_s=prep.time_scale_s)
    predictor = Predictor(model, prep, 20)
    prefix = features[features.unit_id.eq("Filter_2")].iloc[:30]
    encoded = apply_preprocessor(prep, prefix)
    for size in (20, 25, 30):
        ordinary = predictor.predict_from_history(prefix.iloc[:size])
        cached = predictor.predict_encoded_prefix(prefix.iloc[:size], encoded.iloc[:size])
        traced = predictor.predict_from_history(prefix.iloc[:size], with_trace=True)
        assert ordinary["predicted_rul_s"] == cached["predicted_rul_s"] == traced["predicted_rul_s"]
    chart = window_forecast_history(prefix, model, prep, 20)
    replay = replay_unit(prefix, predictor, dataset_id="filters", unit_id="Filter_2", run_id="fixture",
                         history_length=20, warning_horizon_s=420, truth_units=units)["predictions"]
    np.testing.assert_allclose(chart.predicted_rul_s, replay.predicted_rul_s, equal_nan=True, rtol=0, atol=0)


def test_scheduler_reduces_on_fifth_bad_epoch_and_restores():
    cfg = protocol("bearings")
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.AdamW([parameter], lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold_mode="abs", threshold=1., **cfg["scheduler"])
    scheduler.step(100.)
    for _ in range(4):
        scheduler.step(99.5)
        assert optimizer.param_groups[0]["lr"] == .001
    saved = scheduler.state_dict()
    restored = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **cfg["scheduler"])
    restored.load_state_dict(saved)
    restored.step(99.5)
    assert optimizer.param_groups[0]["lr"] == .0005
    for _ in range(100):
        restored.step(99.5)
    assert optimizer.param_groups[0]["lr"] == .00001


def test_weibull_extreme_finite_values_pass_gradcheck():
    duration = torch.tensor([1e-9, 1e9, 1e6], dtype=torch.float64)
    event = torch.tensor([0., 1., 0.], dtype=torch.float64)
    logscale = torch.tensor([-18., 18., 13.], dtype=torch.float64, requires_grad=True)
    logshape = torch.tensor([.2, 1., 2.], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda a, b: weibull_nll_seconds(duration, event, a.exp(), b.exp()),
                                   (logscale, logshape), eps=1e-6)


def test_reservoir_cache_matches_computation_and_binds_graph(tmp_path, tiny_bearing_tables, monkeypatch):
    from pdm.config import load_dataset_config
    from pdm.models.reservoir import LeakyESN
    from pdm.predict import forecast_tensors
    from pdm.preprocessing import fit_preprocessor
    from pdm.train import UnitWindowDataset, _collate
    from pdm.training_engine import state_cache
    from pdm.windows import build_windows

    features, units = tiny_bearing_tables
    split = {"train": ["Bearing1_1", "Bearing1_2", "Bearing1_3"], "validation": ["Bearing1_4"], "test": ["Bearing1_5"]}
    prep, encoded = fit_preprocessor("bearings", features, units, split, load_dataset_config("bearings"))
    windows = build_windows(features, units, 20, "bearings")
    windows = windows[windows.unit_id.eq("Bearing1_4")]
    dataset = UnitWindowDataset(encoded[encoded.unit_id.eq("Bearing1_4")], windows, prep.feature_names, "bearings", prep.time_scale_s)
    torch.manual_seed(42)
    model = LeakyESN(torch.randn(4, len(prep.feature_names)), torch.randn(4, 4) * .1, torch.randn(4), time_scale_s=prep.time_scale_s)
    model.provenance = {"graph_hash": "fixture"}
    first = state_cache(model, dataset, prep, {"data": "fixture"}, tmp_path)
    batch = _collate([dataset[i] for i in range(len(dataset))])
    torch.testing.assert_close(first, model.forward_states(batch["x"])[:, -1, :])
    direct = forecast_tensors(model, batch["x"])["point"]
    cached = forecast_tensors(model, batch["x"], first)["point"]
    torch.testing.assert_close(direct, cached, rtol=0, atol=0)
    original = model.forward_states
    monkeypatch.setattr(model, "forward_states", lambda *_: pytest.fail("Cache hit recomputed states"))
    torch.testing.assert_close(first, state_cache(model, dataset, prep, {"data": "fixture"}, tmp_path), rtol=0, atol=0)
    monkeypatch.setattr(model, "forward_states", original)
    model.W_res.add_(.2)
    changed = state_cache(model, dataset, prep, {"data": "fixture"}, tmp_path)
    assert not torch.equal(first, changed)
    assert len(list((tmp_path / "state_cache").glob("*.npy"))) == 2


@pytest.mark.parametrize("dataset_id", ["bearings", "filters"])
def test_current_feature_baselines_fit_only_train_and_keep_warmup(tmp_path, monkeypatch, request, dataset_id):
    from pdm import study_baselines as baselines

    features, units = request.getfixturevalue("tiny_bearing_tables" if dataset_id == "bearings" else "tiny_filter_tables")
    train_ids = ["Bearing1_1", "Bearing1_2", "Bearing1_3"] if dataset_id == "bearings" else ["Filter_1", "Filter_3", "Filter_4", "Filter_5"]
    uid = "Bearing1_4" if dataset_id == "bearings" else "Filter_2"
    split = {"train": train_ids, "validation": [uid], "test": []}
    bundle = {"features": features, "units": units, "split": split, "dataset_version": "fixture"}
    monkeypatch.setattr(baselines, "load_processed", lambda *args: bundle)
    model = baselines.fit_baseline(dataset_id, tmp_path)
    assert model["training_ids"] == train_ids
    source = features[features.unit_id.eq(uid)].reset_index(drop=True)
    reference = source[["unit_id", "timestamp_s"]].copy()
    reference["prediction_status"] = ["Collecting history"] * 19 + ["ok"] * (len(source) - 19)
    reference["actual_rul_s"] = float(units.set_index("unit_id").loc[uid, "event_time_s"]) - source.timestamp_s
    reference["run_id"] = "reference_fixture"
    prediction = baselines.evaluate_baseline(model, reference, tmp_path / "predictions.csv")
    assert prediction.predicted_rul_s.iloc[:19].isna().all()
    assert np.isfinite(prediction.predicted_rul_s.iloc[19:]).all()
    assert prediction.run_id.eq("baseline_" + model["name"]).all()
    if dataset_id == "filters":
        eligible = prediction.prediction_status.eq("ok") & prediction.outcome_duration_s.gt(0)
        assert np.isfinite(prediction.loc[eligible, "survival_nll"]).all()
        assert prediction.loc[prediction.outcome_duration_s.le(0), "survival_nll"].isna().all()
    # Held-out values can alter their predictions, never refit the saved model.
    sensor = "horizontal_rms" if dataset_id == "bearings" else "differential_pressure"
    features.loc[features.unit_id.eq(uid), sensor] *= 100
    repeated = baselines.fit_baseline(dataset_id, tmp_path)
    assert repeated == model
