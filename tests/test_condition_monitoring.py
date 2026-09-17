"""Behavioral acceptance fixtures are synthetic, never experimental results."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.history import history_policy, history_windows, resolved_length
from pdm.models.recurrent import PDMNet
from pdm.monitoring.calibration import calibration_report, verify_calibration, weibull_probabilities
from pdm.monitoring.contracts import dataset_profile, horizon_label, observed_prefix
from pdm.monitoring.evaluation import score_episodes, signal_metrics
from pdm.monitoring.journal import append_feedback
from pdm.monitoring.normality import assess_normality, fit_reference
from pdm.monitoring.policy import action_timing, default_policy
from pdm.monitoring.quality import assess_quality, quality_policy
from pdm.monitoring.runtime import monitoring_step, replay_monitoring
from pdm.monitoring.signal_forecast import (
    fit_sensor,
    forecast_signal,
    median_crossing,
    supervised_targets,
)
from pdm.monitoring.state import initial_state, update_state
from pdm.windows import raw_numeric_columns


def fixture_frame(n=80, dataset="bearings", uid="a"):
    dt = 60. if dataset == "bearings" else 6.
    frame = pd.DataFrame({c: np.ones(n) for c in raw_numeric_columns(dataset)})
    frame["unit_id"] = uid
    frame["dataset_id"] = dataset
    frame["timestamp_s"] = np.arange(1, n + 1) * dt
    frame["operating_age_s"] = frame.timestamp_s
    frame["gap_before"] = False
    frame["quality_gap_before"] = False
    frame["regime_id"] = "known"
    if dataset == "bearings":
        frame["rpm"], frame["load_kn"] = 2100., 12.
        frame["horizontal_rms"] = 1 + .01 * np.sin(np.arange(n))
        frame["vertical_rms"] = frame.horizontal_rms
    else:
        frame["dust"] = "A"
        frame["flow_rate"], frame["dust_feed"] = 120., 200.
        frame["differential_pressure"] = 100 + .1 * np.arange(n)
    return frame


def setup(frame):
    ds = str(frame.dataset_id.iloc[0])
    profile = dataset_profile(ds)
    reference = fit_reference(frame, ["a"], ds)
    policy = default_policy(profile)
    policy.update(warning_enter=5., warning_exit=2.)
    return {"profile": profile, "reference": reference, "state_policy": policy,
            "quality_policy": quality_policy(profile), "bundle_id": "synthetic_test"}


@pytest.mark.parametrize("mode", ["fixed_20", "fixed_40", "fixed_60", "variable_20_60"])
@pytest.mark.parametrize("n", [19, 20, 39, 40, 59, 60, 61])
def test_history_boundaries(mode, n):
    frame = fixture_frame(n)
    prep = SimpleNamespace(history_policy=history_policy(mode), dataset_id="bearings")
    expected = max(20, min(n, 60)) if mode == "variable_20_60" else int(mode.split("_")[1])
    assert resolved_length(frame, prep, 20) == expected
    frame.loc[n - 1, "gap_before"] = True
    assert resolved_length(frame, prep, 20) == (20 if mode == "variable_20_60" else expected)


@pytest.mark.parametrize("architecture", ["gru", "lstm"])
def test_padding_does_not_update_last_valid_state(architecture):
    torch.manual_seed(12)
    model = PDMNet(3, architecture=architecture).eval()
    a, b = torch.randn(1, 20, 3), torch.randn(1, 39, 3)
    padded = torch.randn(2, 70, 3) * 1000
    padded[0, :20], padded[1, :39] = a[0], b[0]
    with torch.no_grad():
        actual = model(padded, torch.tensor([20, 39]))
        expected = torch.cat([model(a), model(b)])
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_variable_training_really_contains_multiple_lengths_and_resets():
    f = fixture_frame(200)
    f.loc[70, "gap_before"] = True
    units = pd.DataFrame([{"unit_id": "a", "event_time_s": 13000., "observation_end_s": 13000., "event_observed": 1}])
    w = history_windows(f, units, "bearings", history_policy("variable_20_60"), training=True)
    lengths = w.end_index - w.start_index + 1
    assert lengths.min() == 20 and lengths.max() == 60 and lengths.nunique() > 30
    assert not ((w.start_index < 70) & (w.end_index >= 70)).any()


def test_reference_ignores_holdout_and_future_and_is_provisional():
    f = pd.concat([fixture_frame(), fixture_frame(uid="test")])
    reference = fit_reference(f, ["a"], "bearings")
    altered = f.copy()
    altered.loc[(altered.unit_id == "test") | (altered.timestamp_s > 1200), "horizontal_rms"] = 1000
    assert fit_reference(altered, ["a"], "bearings") == reference
    assert reference["status"] == "provisional"
    row = f.iloc[-1].copy()
    row["rpm"] = 100
    assert assess_normality(row, reference)["model_applicability"] == "out_of_domain"
    assert fit_reference(f, ["missing"], "bearings")["status"] == "unavailable"


def test_quality_invalid_stale_clock_and_high_real_signal():
    f = fixture_frame()
    p = dataset_profile("bearings")
    q = quality_policy(p)
    f.loc[79, "horizontal_rms"] = 100000
    assert assess_quality(f, p, q, 4800)["measurement_usable"]
    assert assess_quality(f, p, q, 5100)["data_quality_status"] == "stale"
    f.loc[79, "horizontal_rms"] = np.nan
    assert assess_quality(f, p, q, 4800)["data_quality_status"] == "invalid"
    f.loc[78, "timestamp_s"] = 4800
    assert "clock" in assess_quality(f, p, q, 4800)["reason_codes"][0]


def test_runtime_future_truth_and_seek_independence():
    f = fixture_frame(45)
    args = setup(f)
    sensor = fit_sensor(f, ["a"], args["profile"], horizons=[60, 120], method="causal_local_trend")
    args["sensor_model"] = sensor
    rows, state = replay_monitoring(f, as_of=2400, **args)
    changed = f.copy()
    changed.loc[changed.timestamp_s > 2400, "horizontal_rms"] = 1000
    changed["RUL"] = 0
    changed["life_fraction"] = .99
    changed["event_time_s"] = 1
    other, seek = replay_monitoring(changed, as_of=2400, **args)
    assert rows == other and state == seek
    repeated, repeat_state = monitoring_step(changed, as_of=2400, previous=json.loads(json.dumps(state)), **args)
    assert repeat_state == state
    assert repeated["condition"] == rows[-1]["condition"]
    assert not {"RUL", "event_time_s", "life_fraction"} & set(observed_prefix(changed, "bearings", 2400))


def test_signal_targets_mask_each_horizon_gap_repair_and_end():
    f = fixture_frame(45)
    f.loc[30, "maintenance_reset"] = True
    p = dataset_profile("bearings")
    _, labels, meta = supervised_targets(f, p, [60, 120, 3600], tolerance=.01)
    idx = meta.index[meta.issued_at == 29 * 60][0]
    assert np.isfinite(labels[idx, 0]) and np.isnan(labels[idx, 1:]).all()
    idx = meta.index[meta.issued_at == 30 * 60][0]
    assert np.isnan(labels[idx]).all()


def test_forecast_falls_and_never_uses_rul_or_future_context():
    f = fixture_frame()
    f["horizontal_rms"] = 100 - np.arange(80)
    p = dataset_profile("bearings")
    model = fit_sensor(f, ["a"], p, horizons=[60, 120], method="causal_local_trend")
    rows = forecast_signal(f, model, 2400)
    assert rows[1]["point"] < rows[0]["point"]
    assert median_crossing(rows, None)["time"] is None
    f.loc[f.timestamp_s > 2400, ["rpm", "horizontal_rms"]] = 100000
    f["RUL"] = 0
    assert forecast_signal(f, model, 2400) == rows
    assert not forecast_signal(f, None, 2400)


def test_horizon_unknown_and_cumulative_probabilities():
    assert horizon_label(as_of=10, horizon=20, observation_end=15) is None
    assert horizon_label(as_of=10, horizon=20, observation_end=50) == 0
    assert horizon_label(as_of=10, horizon=20, observation_end=50, event_time=25, outcome="observed_failure") == 1
    rows = weibull_probabilities(50., 2., [10., 20., 100.])
    cumulative = [r["cumulative"] for r in rows]
    assert 0 <= cumulative[0] <= cumulative[1] <= cumulative[2] <= 1
    assert sum(r["bin_probability"] for r in rows) == pytest.approx(cumulative[-1])


def test_calibration_is_bound_and_not_operator_probability():
    report = calibration_report({"history": 20}, ["a", "b", "c"], 3)
    assert not report["operator_probabilities_allowed"] and not report["coverage_guarantee"]
    with pytest.raises(ValueError, match="identity"):
        verify_calibration(report, {"history": 60})


def test_persistent_yellow_spike_recovery_and_episode_identity():
    f = fixture_frame(60)
    args = setup(f)
    f.loc[23, "horizontal_rms"] = 3
    f.loc[30:38, "horizontal_rms"] = 3
    rows, state = replay_monitoring(f, **args)
    assert rows[23]["condition"]["display_zone"] == "green"
    assert rows[32]["condition"]["display_zone"] == "yellow"
    assert rows[32]["event_forecast"]["point"] is None
    assert rows[39]["condition"]["display_zone"] == "yellow"
    assert rows[43]["condition"]["display_zone"] == "green"
    assert len(state["episodes"]) == 1 and state["episodes"][0]["resolved_at"] == rows[43]["as_of"]


def test_hard_limit_before_warmup_ood_and_loss_preserves_latch():
    f = fixture_frame(5, "filters")
    args = setup(fixture_frame(30, "filters"))
    f.loc[0, "differential_pressure"] = 650
    f.loc[0, "dust"] = "unseen"
    result, state = monitoring_step(f.iloc[:1], as_of=6, **args)
    assert result["condition"]["display_zone"] == "red"
    assert result["model_applicability"] == "out_of_domain"
    result, state = monitoring_step(f.iloc[:1], as_of=60, previous=state, **args)
    assert result["condition"]["display_zone"] == "gray"
    assert result["condition"]["critical_latch"]
    assert state["active_episode"]
    broken = f.iloc[:1].copy()
    broken["differential_pressure"] = np.nan
    result, _ = monitoring_step(broken, as_of=6, **args)
    assert result["condition"]["display_zone"] == "gray"


def test_ack_is_append_only_and_not_a_fault_or_resolution(tmp_path):
    record = {"episode_id": "a", "unit_id": "bearing", "at": 123., "status": "acknowledged",
              "source": "operator", "verification_status": "unverified"}
    path = tmp_path / "feedback.jsonl"
    first = append_feedback(path, record)
    append_feedback(path, record)
    assert len(path.read_text().splitlines()) == 1
    assert first["confirmed_cause"] is None and not first["automatic_retraining"]


def test_action_timing_rejects_internal_units_wide_interval_and_ood():
    p = dataset_profile("filters")
    policy = default_policy(p)
    assert action_timing({}, p, policy, "in_domain", "valid")["planning_margin"] is None
    p = dataset_profile("bearings")
    policy = default_policy(p)
    policy["action_profile"].update(required_action_lead_time=500, safety_buffer=100, verification_status="laboratory_definition")
    policy.update(prognostic_methods=["empirical"], max_interval_width=300)
    event = {"event_definition_id": p["event_definition_id"], "time_basis": p["time_basis"], "timing_validated": True,
             "validated_horizon": 900, "interval": {"lower": 400, "upper": 600, "method": "empirical"}}
    assert action_timing(event, p, policy, "in_domain", "valid")["planning_margin"] == -200
    assert not action_timing(event, p, policy, "out_of_domain", "valid")["prognostic_escalation_eligible"]
    event["interval"]["upper"] = 10000
    assert action_timing(event, p, policy, "in_domain", "valid")["reason"] == "forecast_uncertainty_too_high"


def test_risk_path_can_escalate_without_anomaly():
    p = dataset_profile("bearings")
    policy = default_policy(p)
    policy.update(warning_enter=5., warning_exit=2., prognostic_methods=["empirical"], max_interval_width=300)
    policy["action_profile"].update(required_action_lead_time=500, safety_buffer=100, verification_status="laboratory_definition")
    event = {"event_definition_id": p["event_definition_id"], "time_basis": p["time_basis"], "timing_validated": True,
             "validated_horizon": 900, "interval": {"lower": 400, "upper": 600, "method": "empirical"}, "risk_band": "urgent"}
    state = initial_state("a", "b")
    for t in (60, 120, 180):
        cond = {"segment_id": "s", "measurement": {},
                "quality": {"data_quality_status": "valid", "last_measurement_at": t, "reason_codes": []},
                "normality": {"model_applicability": "in_domain", "score": 0., "reason_codes": []}}
        state = update_state(state, cond, event, p, policy, t)
    assert state["display_zone"] == "red"


def test_no_predictions_do_not_win_and_censored_diagnostic_is_unknown():
    forecasts = pd.DataFrame({"unit_id": ["a"], "issued_at": [0.], "horizon": [60.], "point": [None], "lower": [None], "upper": [None]})
    summary, _ = signal_metrics(forecasts, np.array([[2.]]), pd.DataFrame({"unit_id": ["a"], "issued_at": [0.]}), [60.])
    assert summary.iloc[0].prediction_coverage == 0 and pd.isna(summary.iloc[0].unit_balanced_mae)
    units = pd.DataFrame({"unit_id": ["a", "b"], "observation_end_s": [10., 30.], "event_observed": [0, 1], "event_time_s": [np.nan, 30.]})
    episodes = [{"episode_id": "e", "unit_id": "a", "confirmed_at": 5., "kind": "diagnostic"},
                {"episode_id": "f", "unit_id": "a", "confirmed_at": 6., "kind": "prognostic", "issued_horizon": 60.}]
    outcomes, counts = score_episodes(episodes, units, minimum_lead=10)
    assert outcomes[0]["category"] == "unknown_diagnostic_outcome"
    assert outcomes[1]["category"] == "unknown_insufficient_followup"
    assert counts["observed_events"] == 1 and counts["unmatched_events"] == 1


def test_bundle_and_dependency_tampering_is_rejected(tmp_path, monkeypatch):
    from pdm.io_util import atomic_write_json, sha256_file
    from pdm.monitoring import bundle as module
    from pdm.training_protocol import fingerprint

    monkeypatch.setattr(module, "project_root", lambda: tmp_path)
    monkeypatch.setattr(module, "runs_root", lambda: tmp_path / "runs")
    dep = tmp_path / "sensor.joblib"
    dep.write_bytes(b"synthetic dependency")
    profile = dataset_profile("bearings")
    payload = {"profile": profile, "state_policy": default_policy(profile), "frozen": True,
        "dependencies": {"sensor.joblib": sha256_file(dep)}, "reference": {}, "quality_policy": {},
        "history_policy": {}, "event_model_run_id": None, "sensor_model_run_id": None,
        "dataset_version": "test", "calibration_profile": None}
    bid = fingerprint(payload)[:24]
    payload["bundle_id"] = bid
    path = module.bundle_root(bid) / "monitoring_bundle.json"
    atomic_write_json(path, payload)
    assert module.load_bundle(bid, load_models=False)[0] == payload
    changed = dict(payload)
    changed["dataset_version"] = "changed"
    atomic_write_json(path, changed)
    with pytest.raises(ValueError, match="identity"):
        module.load_bundle(bid, load_models=False)
    atomic_write_json(path, payload)
    dep.write_bytes(b"changed model")
    with pytest.raises(ValueError, match="hash"):
        module.load_bundle(bid, load_models=False)


def test_actual_quantile_fit_and_crossing_repair():
    from threadpoolctl import threadpool_limits

    frame = fixture_frame(55)
    frame["horizontal_rms"] = 3 - .02 * np.arange(55)
    profile = dataset_profile("bearings")
    with threadpool_limits(limits=1):
        model = fit_sensor(frame, ["a"], profile, horizons=[60, 120], method="multi_horizon_quantile_boosting", max_iter=3)
        rows = forecast_signal(frame, model, 2400)
    assert all(0 <= row["lower"] <= row["point"] <= row["upper"] for row in rows)
    assert all(row["interval_method"] == "pointwise_quantile_regression_5_95" for row in rows)
    class Constant:
        def __init__(self, value):
            self.value = value
        def predict(self, _):
            return np.array([self.value])
    for h in model["models"]:
        model["models"][h] = {.05: Constant(8), .5: Constant(4), .95: Constant(-2)}
    rows = forecast_signal(frame, model, 2400)
    assert all((r["lower"], r["point"], r["upper"]) == (0., 4., 8.) for r in rows)


def test_channel_limit_is_independent_of_broken_covariate_but_not_stale():
    frame = fixture_frame(30, "filters")
    args = setup(frame)
    frame.loc[29, "flow_rate"] = np.nan
    frame.loc[29, "differential_pressure"] = 700.
    result, _ = monitoring_step(frame, as_of=180., **args)
    assert result["condition"]["display_zone"] == "red"
    result, _ = monitoring_step(frame, as_of=300., **args)
    assert result["condition"]["display_zone"] == "gray"


def test_serialized_episode_resolves_in_same_log_and_cannot_cross_equipment():
    frame = fixture_frame(50)
    args = setup(frame)
    frame.loc[25:30, "horizontal_rms"] = 8
    state = None
    for i in range(len(frame)):
        row, state = monitoring_step(frame.iloc[:i+1], as_of=float(frame.timestamp_s.iloc[i]), previous=state, **args)
        state = json.loads(json.dumps(state))
    assert len(state["episodes"]) == 1 and state["episodes"][0]["resolved_at"] is not None
    with pytest.raises(ValueError, match="another unit"):
        monitoring_step(fixture_frame(50, uid="other"), as_of=3000., previous=state, **args)


def test_fragment_audit_retains_high_amplitude_and_detects_clipping():
    from pdm.data.quality import fragment_diagnostics

    x = np.arange(32, dtype=float) * 10000
    audit = fragment_diagnostics(x, expected_samples=32)
    assert not audit["amplitude_only_exclusion"] and audit["diagnostic_reasons"] == ""
    audit = fragment_diagnostics(np.ones(32), expected_samples=32, sensor_range=(-1, 1))
    assert audit["constant_fragment"] and audit["saturation_fraction"] == 1 and audit["repeated_fragment"]


def test_invalid_text_clock_and_old_broken_sensor_never_impute_forecast():
    frame = fixture_frame(65)
    args = setup(frame)
    model = fit_sensor(frame, ["a"], args["profile"], horizons=[60], method="persistence")
    args["sensor_model"] = model
    frame["horizontal_rms"] = frame.horizontal_rms.astype(object)
    frame.loc[45, "horizontal_rms"] = "broken"
    result, _ = monitoring_step(frame, as_of=3900, **args)
    assert result["data_quality_status"] == "insufficient_history"
    assert all(row["point"] is None for row in result["signal_forecasts"])
    frame.loc[64, "timestamp_s"] = np.nan
    result, _ = monitoring_step(frame, as_of=3900, **args)
    assert result["data_quality_status"] == "invalid"
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mode", ["fixed_20", "fixed_40", "fixed_60", "variable_20_60"])
def test_training_windows_never_cross_maintenance(mode):
    frame = fixture_frame(160)
    frame.loc[70, "maintenance_reset"] = True
    units = pd.DataFrame([{"unit_id": "a", "event_time_s": 10000., "observation_end_s": 10000., "event_observed": 1}])
    windows = history_windows(frame, units, "bearings", history_policy(mode), training=True)
    assert not ((windows.start_index < 70) & (windows.end_index >= 70)).any()


def test_historical_fragment_damage_and_sensor_range_reset_warmup():
    frame = fixture_frame(80, "filters")
    profile = dataset_profile("filters")
    policy = quality_policy(profile)
    frame.loc[65, "differential_pressure"] = 3000
    assert assess_quality(frame, profile, policy, 480)["available_measurements"] == 14
    frame.loc[65, "differential_pressure"] = 100
    frame["sample_count_ok"] = True
    frame.loc[70, "sample_count_ok"] = False
    assert assess_quality(frame, profile, policy, 480)["available_measurements"] == 9


def test_multiscale_feature_context_resets_at_maintenance():
    from pdm.multiscale_features import enrich
    frame = fixture_frame(90)
    frame.loc[70, "maintenance_reset"] = True
    full = enrich(frame, "bearings")
    restarted = enrich(frame.iloc[70:].copy(), "bearings")
    for column in ("horizontal_rms_slope_60", "horizontal_rms_available_60"):
        np.testing.assert_allclose(full.loc[70:, column], restarted[column])


def test_operating_regime_explains_same_observed_level_without_adapting_reference():
    a = fixture_frame(30)
    b = fixture_frame(30, uid="b")
    b["rpm"] = 2400.
    b["horizontal_rms"] += 4.
    reference = fit_reference(pd.concat([a, b]), ["a", "b"], "bearings")
    same_signal = b.iloc[-1].copy()
    normal = assess_normality(same_signal, reference)
    same_signal["rpm"] = 2100.
    anomalous = assess_normality(same_signal, reference)
    assert anomalous["residuals"]["horizontal_rms"]["z"] > 100
    assert abs(normal["residuals"]["horizontal_rms"]["z"]) < 3
    assert reference["automatic_adaptation"] is False


def test_episode_score_uses_first_issue_and_preserves_short_event_denominator():
    units = pd.DataFrame([
        {"unit_id": "a", "event_time_s": 200., "observation_end_s": 200., "event_observed": 1},
        {"unit_id": "short", "event_time_s": 30., "observation_end_s": 30., "event_observed": 1},
        {"unit_id": "maint", "event_time_s": np.nan, "observation_end_s": 250., "event_observed": 0, "event_source": "preventive maintenance"},
    ])
    episodes = [
        {"episode_id": "a1", "unit_id": "a", "confirmed_at": 100., "kind": "prognostic", "issued_horizon": 80., "updates": [{"as_of": 150., "type": "update"}]},
        {"episode_id": "m1", "unit_id": "maint", "confirmed_at": 100., "kind": "prognostic", "issued_horizon": 80.},
    ]
    outcomes, summary = score_episodes(episodes, units, minimum_lead=30., maximum_horizon=80.)
    assert outcomes[0]["category"] == "early"
    assert outcomes[1]["category"] == "unknown_informative_intervention"
    assert summary["observed_events"] == 2 and summary["timely_recall"] == 0


def test_multiscale_v2_all_units_finite_and_no_age_is_saved():
    from pdm.feature_recipes import enrich_features
    from pdm.preprocessing import fit_preprocessor
    frame = pd.concat([fixture_frame(80), fixture_frame(80, uid="b")], ignore_index=True)
    frame.loc[100, "maintenance_reset"] = True
    result = enrich_features(frame, "bearings", "multiscale_trend_v2")
    assert np.isfinite(result.filter(regex="_slope_").to_numpy()).all()
    units = pd.DataFrame([{"unit_id": uid, "event_time_s": 5000., "observation_end_s": 5000., "event_observed": 1} for uid in ("a", "b")])
    prep, _ = fit_preprocessor("bearings", frame, units, {"train": ["a"]}, {"feature_recipe": "multiscale_no_age_v2"})
    assert "operating_age_s" not in prep.feature_names
    assert prep.to_dict()["feature_recipe_parameters"]["version"] == 2


def test_filter_source_event_and_engineering_limit_equality_are_explicitly_distinct():
    from pdm.data.filters import _train_unit_table
    frame = fixture_frame(1, "filters")
    frame["author_split"] = "author_train"
    frame["author_data_no"] = 1
    frame["origin_unit_id"] = 1
    frame["differential_pressure"] = 600.
    source = _train_unit_table(frame, 600.)
    assert source.event_observed.iloc[0] == 0
    assert dataset_profile("filters")["event_convention"] == "first_measurement_gt_600pa"
    result, _ = monitoring_step(frame, as_of=6., **setup(frame))
    assert result["condition"]["display_zone"] == "red"
