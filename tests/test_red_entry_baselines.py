import json

import numpy as np
import pandas as pd
import pytest

from pdm.red_entry_baselines import (
    always_no_entry,
    fit_age_baseline,
    last_value_signal,
    predict_age_baseline,
    trend_to_red,
)


def episodes(rows):
    return pd.DataFrame([{ "unit_id": uid, "physical_unit_id": physical, "split": "train",
                          "first_event_verified": event, "age_at_start_s": start,
                          "age_at_start_source": "counter", "age_at_event_s": end if event else np.nan,
                          "age_at_event_source": "counter" if event else "unknown",
                          "age_at_censor_s": end if not event else np.nan,
                          "age_at_censor_source": "counter" if not event else "unknown"}
                         for uid, physical, start, end, event in rows])


def test_conditional_age_risk_uses_delayed_entry_and_censoring():
    data = episodes([("a", "a", 0, 10, True), ("b", "b", 0, 20, True),
                     ("c", "c", 15, 30, False)])
    state = fit_age_baseline(data, train_unit_ids=["a", "b", "c"])
    group = next(iter(state["groups"].values()))
    assert group["curve"][0]["risk_weight"] == 2  # c enters after first event
    assert group["curve"][0]["survival"] == .5
    assert group["curve"][1]["survival"] == .25
    early = predict_age_baseline(state, 0, [10, 20, 30], age_source="counter")
    assert early["probabilities"] == [.5, .75, .75]
    late = predict_age_baseline(state, 15, [5, 15], age_source="counter")
    assert late["probabilities"] == [.5, .5]
    assert late["hazard"] == [.5, 0.0]


def test_no_events_is_zero_only_within_observed_support():
    state = fit_age_baseline(episodes([("a", "a", 0, 10, False), ("b", "b", 0, 20, False)]),
                             train_unit_ids=["a", "b"])
    result = predict_age_baseline(state, 5, [5, 15, 16])
    assert result["probabilities"] == [0., 0., None]
    assert result["hazard"] == [0., 0., None]
    assert result["status"] == "partial_support"
    assert predict_age_baseline(state, 20, [1])["probabilities"] == [None]
    assert predict_age_baseline(state, None, [1])["reason"] == "unknown_age"


def test_only_train_ids_affect_fit_and_known_nontrain_is_rejected():
    train = episodes([("a", "a", 0, 10, True), ("b", "b", 0, 20, False)])
    outside = episodes([("test", "test", 0, 1, True)]).assign(split="test")
    combined = pd.concat([train, outside], ignore_index=True)
    a = fit_age_baseline(train, train_unit_ids=["a", "b"])
    b = fit_age_baseline(combined, train_unit_ids=["a", "b"])
    assert a == b
    with pytest.raises(ValueError, match="Train"):
        fit_age_baseline(combined, train_unit_ids=["a", "b", "test"])


def test_cycle_weight_normalizes_physical_units_and_group_support_counts_equipment():
    data = episodes([("a1", "a", 0, 10, True), ("a2", "a", 0, 10, True),
                     ("b", "b", 0, 20, False)])
    state = fit_age_baseline(data, train_unit_ids=["a1", "a2", "b"])
    result = predict_age_baseline(state, 0, [10, 20])
    assert result["probabilities"] == [.5, .5]
    insufficient = fit_age_baseline(data.iloc[:2], train_unit_ids=["a1", "a2"])
    assert predict_age_baseline(insufficient, 0, [1])["reason"] == "insufficient_physical_units"


def test_unknown_gap_history_is_censored_at_reliable_age_not_later_red():
    data = episodes([("a", "a", 0, 4, False), ("b", "b", 0, 6, False)])
    data["first_red_timestamp_s"] = [100, 200]
    data["age_at_event_s"] = [100, 200]
    data["censor_reason"] = "gap"
    state = fit_age_baseline(data, train_unit_ids=["a", "b"])
    result = predict_age_baseline(state, 0, [4, 6, 7])
    assert result["probabilities"] == [0, 0, None]
    assert next(iter(state["groups"].values()))["observed_event_count"] == 0


def test_regime_and_age_source_are_separate_no_unknown_or_changing_regime_fallback():
    data = episodes([("a", "a", 0, 10, True), ("b", "b", 0, 20, False)])
    data["regime_metadata"] = [{"at_start": {"rpm": 100}, "at_event": {"rpm": 100}},
                               {"at_start": {"rpm": 100}, "at_censor": {"rpm": 100}}]
    state = fit_age_baseline(data, train_unit_ids=["a", "b"])
    assert predict_age_baseline(state, 0, [10], regime={"rpm": 100}, age_source="counter")["probabilities"] == [.5]
    assert predict_age_baseline(state, 0, [10], regime={"rpm": 200})["probabilities"] == [None]
    assert predict_age_baseline(state, 0, [10], regime={"rpm": 100}, age_source="laboratory_proxy")["probabilities"] == [None]
    data.at[1, "regime_metadata"] = {"at_start": {"rpm": 100}, "at_censor": {"rpm": 200}}
    state = fit_age_baseline(data, train_unit_ids=["a", "b"])
    assert state["excluded_counts"] == {"regime_changed": 1}


def test_unobserved_age_holes_and_survival_zero_are_unsupported():
    data = episodes([("a", "a", 0, 10, False), ("b", "b", 20, 30, False)])
    state = fit_age_baseline(data, train_unit_ids=["a", "b"])
    assert predict_age_baseline(state, 5, [5, 6])["probabilities"] == [0, None]
    assert predict_age_baseline(state, 15, [1])["probabilities"] == [None]
    data = episodes([("a", "a", 0, 10, True), ("b", "b", 20, 30, False)])
    state = fit_age_baseline(data, train_unit_ids=["a", "b"])
    assert predict_age_baseline(state, 25, [1])["reason"] == "no_surviving_risk_set"


def test_json_roundtrip_predictions_and_fit_hash_are_reproducible():
    data = episodes([("a", "a", 0, 10, True), ("b", "b", 0, 20, False)])
    state = fit_age_baseline(data, train_unit_ids=["a", "b"])
    reloaded = json.loads(json.dumps(state, allow_nan=False))
    assert predict_age_baseline(state, 0, [10, 20]) == predict_age_baseline(reloaded, 0, [10, 20])
    assert state == fit_age_baseline(data, train_unit_ids=["b", "a"])


def test_unknown_age_is_excluded_without_timestamp_age_imputation():
    data = episodes([("a", "a", 0, 10, True), ("b", "b", 0, 20, False)])
    data["age_at_start_source"] = "unknown"
    data["observation_start_s"] = 0
    data["observation_end_s"] = 999
    state = fit_age_baseline(data, train_unit_ids=["a", "b"])
    assert not state["groups"]
    assert state["excluded_counts"] == {"unknown_or_inconsistent_age": 2}


def prefix(values, **extra):
    return pd.DataFrame({"timestamp_s": range(len(values)), "signal": values, **extra})


def test_causal_trend_direction_boundaries_and_auxiliary_numeric_baseline():
    schema = {"thresholds": {"mode": "absolute", "red": 10}}
    frame = prefix([2., 4., 6.])
    result = trend_to_red(frame, schema, [1, 2, 3])
    assert result["probabilities"] == [0, 1, 1]
    assert result["crossing_delay_s"] == 2
    assert always_no_entry([1, 2])["probabilities"] == [0, 0]
    numeric = last_value_signal(frame, [1, 2])
    assert numeric["values"] == [6, 6]
    assert "probabilities" not in numeric
    below = {"thresholds": {"mode": "absolute", "red": 0, "direction": "below"}}
    assert trend_to_red(prefix([6., 4., 2.]), below, [1, 2])["probabilities"] == [1, 1]


def test_float_clock_trend_preserves_input_and_absorbing_probability():
    frame = pd.DataFrame({"timestamp_s": [100., 101.], "signal": [1., 2.]})
    before = frame.copy(deep=True)
    result = trend_to_red(frame, {"thresholds": {"mode": "absolute", "red": 3}}, [1, 2, 3])
    assert result["probabilities"] == [1., 1., 1.]
    assert result["hazard"] == [1., 0., 0.]
    np.testing.assert_array_equal(1 - np.cumprod(1 - np.asarray(result["hazard"])), [1., 1., 1.])
    pd.testing.assert_frame_equal(frame, before)


def test_trend_mask_unknown_gap_quality_and_post_event_until_explicit_cycle():
    schema = {"thresholds": {"mode": "absolute", "red": 10}}
    assert trend_to_red(prefix([1, 2, 3], gap_before=[False, True, False]), schema, [1])["reason"] == "unknown_first_event_history"
    assert trend_to_red(prefix([1, np.nan, 3]), schema, [1])["probabilities"] == [None]
    assert trend_to_red(prefix([1, 10, 3]), schema, [1])["reason"] == "event_or_post_event"
    result = trend_to_red(prefix([10, 1, 2], cycle_id=["a", "b", "b"], gap_before=[False, True, False]), schema, [1])
    assert result["status"] == "reference"


def test_relative_baseline_warmup_and_prefix_immutability():
    schema = {"thresholds": {"mode": "initial_baseline_multiple", "baseline_n": 3, "red_ratio": 2}}
    assert trend_to_red(prefix([2, 2]), schema, [1])["reason"] == "rule_unavailable"
    frame = prefix([2., 2., 2., 3.])
    old = frame.copy(deep=True)
    result = trend_to_red(frame, schema, [1, 2])
    assert result["red_limit"] == 4
    pd.testing.assert_frame_equal(frame, old)


def test_engine_adapter_uses_raw_prefixes_preserves_nan_support_and_json_state():
    from pdm.red_entry_baselines import fit_baseline_model, predict_baseline_hazard
    events = episodes([("a", "a", 0, 10, True), ("b", "b", 0, 20, False)])
    origins = pd.DataFrame({"unit_id": ["a", "a"], "timestamp_s": [1., 2.], "at_risk": [True, False]})
    raw = prefix([1., 2.], operating_age_s=[0., 5.], operating_age_known=[True, True],
                 operating_age_source=["counter"] * 2)
    batch = {"origins": origins, "prefixes": [raw, raw]}
    model, selection = fit_baseline_model("kaplan_meier", batch, {},
                                          {"horizons_s": [5, 15, 16], "events": events,
                                           "train_unit_ids": ["a", "b"]})
    json.dumps(model, allow_nan=False)
    predicted = predict_baseline_hazard(model, batch, {})
    assert predicted[0, :2].tolist() == [.5, 0]
    assert np.isnan(predicted[0, 2])
    assert np.isnan(predicted[1]).all()
    assert selection["selection"] == "fixed_reference_no_validation_fitting"
    schema = {"thresholds": {"mode": "absolute", "red": 10}}
    batch = {"origins": origins.iloc[:1], "prefixes": [prefix([2., 4.])], "schema": schema}
    trend, _ = fit_baseline_model("trend_to_red", batch, {}, {"horizons_s": [1, 3]})
    assert predict_baseline_hazard(trend, batch, {})[0].tolist() == [0., 1.]


def test_adapter_rejects_future_prefix_and_observes_cancellation():
    from pdm.red_entry_baselines import fit_baseline_model, predict_baseline_hazard
    origins = pd.DataFrame({"unit_id": ["a"], "timestamp_s": [0.], "at_risk": [True]})
    batch = {"origins": origins, "prefixes": [prefix([1, 2])],
             "schema": {"thresholds": {"mode": "absolute", "red": 10}}}
    model, _ = fit_baseline_model("trend_to_red", batch, {}, {"horizons_s": [1]})
    with pytest.raises(ValueError, match="future"):
        predict_baseline_hazard(model, batch)
    with pytest.raises(InterruptedError):
        fit_baseline_model("always_no_entry", batch, {}, {"horizons_s": [1]}, should_stop=lambda: True)
