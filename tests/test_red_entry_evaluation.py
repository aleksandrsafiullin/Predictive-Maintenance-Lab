import numpy as np
import pandas as pd
import pytest

from pdm.red_entry_calibration import (
    apply_hazard_calibrator,
    event_corridor,
    event_quantiles,
    fit_hazard_calibrator,
    hazard_to_cdf,
    load_calibrator,
    save_calibrator,
)
from pdm.red_entry_evaluation import evaluate_red_entry
from pdm.red_entry_policy import alert_episodes


def origins(times, event=None, end=100):
    return pd.DataFrame([dict(unit_id="u", physical_unit_id="p", episode_id="u:0", timestamp_s=t,
                              at_risk=True, risk_status="at_risk", zone="green", is_running=True,
                              event_observed=event is not None, followup_duration_s=(event if event is not None else end)-t)
                         for t in times])


def events(t):
    return pd.DataFrame([dict(unit_id="u", physical_unit_id="p", episode_id="u:0",
                              first_event_verified=True, first_red_timestamp_s=t)])


def test_quantile_null_and_absolute_interval_support():
    cdf = hazard_to_cdf([[.1, .2, .3]])
    assert np.diff(cdf).min() >= 0
    assert event_quantiles(cdf, [10, 20, 30])[0] == [10, None, None]
    assert event_corridor(cdf, [10, 20, 30], 100)["latest_s"] is None
    report = event_corridor([[.1, .95, .99]], [10, 20, 30], 100,
                            calibration={"interval_support_validated": True, "interval_independent_event_units": 5},
                            requirements={"nominal_interval_coverage": .8, "interval_coverage_evidence_requirement": 5})
    assert report["earliest_s"] == 110 and report["latest_s"] == 120


def test_calibration_groups_monotone_identity_and_json(tmp_path):
    h, y, mask = np.array([[.2,.3],[.4,.5]]), np.array([[0,1],[1,0]]), np.ones((2,2),bool)
    a = fit_hazard_calibrator(h,y,mask,["a","b"])
    assert a["status"] == "insufficient_event_evidence"
    np.testing.assert_array_equal(apply_hazard_calibrator(h,a),h)
    a = fit_hazard_calibrator(h,y,mask,["a","b"],requirements={"minimum_independent_event_evidence":2})
    assert a["status"] == "fitted" and not a["interval_support_validated"]
    assert np.diff(hazard_to_cdf(apply_hazard_calibrator(h,a))).min() >= 0
    path = tmp_path / "calibration.json"
    save_calibrator(a,path)
    assert load_calibrator(path) == a
    with pytest.raises(ValueError):
        fit_hazard_calibrator(h,y,mask,["a","b"],train_physical_ids=["a"])
    shared = fit_hazard_calibrator(h,y,mask,["a","b"],calibration_role="shared_validation",
                                  requirements={"minimum_independent_event_evidence":2})
    assert shared["status"] == "insufficient_event_evidence"


def test_censored_known_negative_and_incomplete_unknown():
    o = origins([0,20,40],end=50)
    r = evaluate_red_entry(o,pd.DataFrame(),np.ones((3,1)),[30],policy={"threshold":.9,"release_threshold":.9},
                           laboratory_elapsed_assumption=True)
    assert r["alert_episodes"][0]["horizon_outcome"] == "false"
    assert r["brier_by_horizon"][0]["known_origins"] == 2
    assert r["brier_by_horizon"][0]["unknown_origins"] == 1
    assert r["metrics"]["useful_event_recall"] is None
    assert not r["quality_gate"]["can_pass"]


def test_perpetual_early_alert_never_becomes_timely_and_budget_rejects():
    o = origins([0,10,20,30,40,50,60,70,80,90],event=100)
    r = evaluate_red_entry(o,events(100),np.ones((10,1)),[30],
        requirements={"minimum_action_lead_s":10,"max_alert_time_fraction":.1},
        laboratory_elapsed_assumption=True)
    assert r["events"][0]["status"] == "early"
    assert r["metrics"]["useful_event_recall"] == 0
    assert "max_alert_time_fraction_not_met" in r["quality_gate"]["reason_codes"]


def test_late_and_repeated_episodes_use_first_confirmed():
    o = origins(list(range(0,100,10)),event=100)
    p = np.zeros((10,1))
    p[9] = 1
    r = evaluate_red_entry(o,events(100),p,[30],requirements={"minimum_action_lead_s":20})
    assert r["events"][0]["status"] == "late"
    p[0] = 1
    p[7:9] = 1
    r = evaluate_red_entry(o,events(100),p,[30],requirements={"minimum_action_lead_s":20})
    assert r["events"][0]["status"] == "early"
    assert r["events"][0]["confirmed_episode_count"] == 2


def test_confirmation_seconds_variable_cadence_gap_unknown_and_cooldown():
    o = origins([0,1,9,10,20,30])
    p = np.ones((6,1))
    ep = alert_episodes(o,p,[30],policy={"confirmation_s":10})
    assert ep[0]["start_s"] == 10
    o.loc[3,"gap_before"] = True
    ep = alert_episodes(o,p,[30],policy={"confirmation_s":10})
    assert ep[0]["start_s"] == 20
    o.loc[4,"context_known"] = False
    o["context_known"] = o.context_known.fillna(True)
    ep = alert_episodes(o,p,[30],policy={"confirmation_s":10})
    assert not ep


def test_exposure_excludes_gap_stopped_and_unknown_intervals():
    o = origins([0,10,20,30,40])
    o["gap_before"] = [False,False,True,False,False]
    o["is_running"] = [True,True,True,False,False]
    r = evaluate_red_entry(o,pd.DataFrame(),np.ones((5,1)),[30],laboratory_elapsed_assumption=True)
    assert r["metrics"]["operating_exposure_s"] == 10


def test_explicit_synthetic_validation_gate_and_budget_failure():
    o = origins(list(range(0,100,10)),event=100)
    p = np.zeros((10,1))
    p[7:10] = 1
    requirements = {"minimum_action_lead_s":10,"max_alert_time_fraction":1,
                    "max_false_alert_episodes_per_100_operating_hours":1,
                    "nominal_interval_coverage":.8,"maximum_interval_width_s":30,
                    "minimum_independent_event_evidence":1,"required_regime_coverage":["lab"],
                    "interval_coverage_evidence_requirement":1}
    provenance = {"independent_holdout_available":True,"physical_identity_verified":True,
                  "physical_ids_by_part":{"holdout":["p"]},"exposed_physical_ids":[]}
    evidence = {"interval_support_validated":True,"interval_independent_event_units":1,
                "interval_coverage":1,"interval_maximum_width_s":20,
                "regime_support_validated":True,"observed_regimes":["lab"]}
    kwargs = dict(requirements=requirements,provenance=provenance,validation_evidence=evidence,
                  model_frozen=True,policy_frozen=True)
    report = evaluate_red_entry(o,events(100),p,[30],**kwargs)
    assert report["quality_gate"]["status"] == "passed"
    requirements["max_alert_time_fraction"] = .01
    assert not evaluate_red_entry(o,events(100),p,[30],**kwargs)["quality_gate"]["can_pass"]
    requirements["max_alert_time_fraction"] = 1
    provenance["exposed_physical_ids"] = ["p"]
    assert not evaluate_red_entry(o,events(100),p,[30],**kwargs)["quality_gate"]["can_pass"]


def test_no_exposure_metadata_is_unavailable():
    o = origins([0,10,20]).drop(columns="is_running")
    report = evaluate_red_entry(o,pd.DataFrame(),np.zeros((3,1)),[30])
    assert report["metrics"]["operating_exposure_s"] is None
    assert report["metrics"]["false_alert_episodes_per_100_operating_hours"] is None


def test_policy_selector_development_only_unset_and_both_budgets():
    from pdm.red_entry_policy import select_alert_policy

    o = origins(list(range(0,100,10)),event=100).assign(split="validation")
    e = events(100).assign(split="validation")
    p = np.zeros((10,1))
    p[7:10] = .6
    unset = select_alert_policy(o,e,p,[30])
    assert unset["status"] == "requirements_unset" and unset["config"]["threshold"] == .5
    assert unset["frozen"] and not unset["selection_quality_claim"]
    with pytest.raises(ValueError):
        select_alert_policy(o,e,p,[30],selection_part="holdout")
    with pytest.raises(ValueError):
        select_alert_policy(o.assign(split="test"),e,p,[30])
    required = {"minimum_action_lead_s":10,"max_alert_time_fraction":1,
                "max_false_alert_episodes_per_100_operating_hours":1,
                "nominal_interval_coverage":.8,"maximum_interval_width_s":30,
                "minimum_independent_event_evidence":1,"required_regime_coverage":["lab"],
                "interval_coverage_evidence_requirement":1}
    candidates = [{"threshold":.5},{"threshold":.75}]
    selected = select_alert_policy(o,e,p,[30],requirements=required,candidate_policies=candidates)
    assert selected["status"] == "selected_development" and selected["config"]["threshold"] == .5
    required["max_alert_time_fraction"] = .01
    selected = select_alert_policy(o,e,p,[30],requirements=required,candidate_policies=[candidates[0]])
    assert selected["status"] == "not_passed"
    required["max_alert_time_fraction"] = 1
    required["max_false_alert_episodes_per_100_operating_hours"] = 0
    selected = select_alert_policy(o,e,np.ones((10,1)),[30],requirements=required,
                                  candidate_policies=[candidates[0]])
    assert selected["status"] == "not_passed"


@pytest.mark.parametrize("predictions", [[0, 1, 1], [0, 0, 0], [1, 1, 1]])
def test_unset_action_lead_never_reports_useful_recall(predictions):
    report = evaluate_red_entry(origins([60, 80, 90], event=100), events(100),
                                np.array(predictions)[:, None], [30],
                                requirements={"minimum_action_lead_s": None})
    assert all(report["metrics"][key] is None for key in (
        "useful_event_recall", "reachable_useful_event_recall", "green_useful_event_recall"))
    assert report["metrics"]["event_count"] == 1
    assert sum(report["metrics"]["event_status_counts"].values()) == 1
    scope = report["metric_scope"]["useful_event_recall"]
    assert scope["status"] == "unavailable"
    assert scope["reason_codes"] == ["minimum_action_lead_s_unset"]
    assert scope["known_usefulness_coverage"] == 0
    assert report["metric_scope"]["reachability"] == "policy_horizon_history_only"
    assert report["metrics"]["reachable_event_count"] == 1
    if predictions == [0, 1, 1]:
        assert report["events"][0]["lead_s"] == 20
        assert report["metrics"]["event_status_counts"]["action_lead_requirement_unset"] == 1


@pytest.mark.parametrize("mixed", [False, True])
def test_unknown_event_outcomes_are_unavailable_not_missed(mixed):
    unknown = origins([70, 80, 90], event=100)
    event_rows = events(100)
    probabilities = np.array([[np.nan], [1.0], [1.0]])
    if mixed:
        known = origins([70, 80, 90], event=100).assign(unit_id="v", physical_unit_id="q", episode_id="v:0")
        unknown = pd.concat([unknown, known], ignore_index=True)
        event_rows = pd.concat([event_rows, events(100).assign(unit_id="v", physical_unit_id="q", episode_id="v:0")])
        probabilities = np.vstack([probabilities, np.ones((3, 1))])
    report = evaluate_red_entry(unknown, event_rows, probabilities, [30],
                                requirements={"minimum_action_lead_s": 10})
    metrics = report["metrics"]
    assert all(metrics[key] is None for key in (
        "useful_event_recall", "reachable_useful_event_recall", "green_useful_event_recall"))
    assert metrics["event_count"] == 1 + mixed
    assert metrics["event_status_counts"]["unknown"] == 1
    assert metrics["event_status_counts"]["missed"] == 0
    assert metrics["event_status_counts"]["useful"] == int(mixed)
    assert sum(metrics["event_status_counts"].values()) == metrics["event_count"]
    scope = report["metric_scope"]["useful_event_recall"]
    assert scope["reason_codes"] == ["unknown_event_outcomes"]
    assert scope["verified_event_count"] == metrics["event_count"]
    assert scope["known_usefulness_event_count"] == int(mixed)
    assert scope["known_usefulness_coverage"] == (0.5 if mixed else 0)
    assert report["brier_by_horizon"][0]["unknown_origins"] == 1
    assert report["brier_by_horizon"][0]["known_origins"] == 2 + 3 * mixed


@pytest.mark.parametrize("probabilities,status,recall", [
    ([1, 1, 1, 1], "early", 0),
    ([0, 1, 1, 1], "useful", 1),
    ([0, 0, 0, 1], "late", 0),
    ([0, 0, 0, 0], "missed", 0),
])
def test_complete_event_useful_recall_preserves_known_policy_outcomes(probabilities, status, recall):
    report = evaluate_red_entry(origins([60, 70, 80, 90], event=100), events(100),
                                np.array(probabilities)[:, None], [30],
                                requirements={"minimum_action_lead_s": 20})
    assert report["events"][0]["status"] == status
    assert all(report["metrics"][key] == recall for key in (
        "useful_event_recall", "reachable_useful_event_recall", "green_useful_event_recall"))
    assert report["metric_scope"]["useful_event_recall"]["status"] == "available"
    assert report["metric_scope"]["useful_event_recall"]["known_usefulness_coverage"] == 1


def test_policy_selector_rejects_unknown_event_recall_without_error():
    from pdm.red_entry_policy import select_alert_policy

    required = {"minimum_action_lead_s": 10, "max_alert_time_fraction": 1,
                "max_false_alert_episodes_per_100_operating_hours": 1,
                "nominal_interval_coverage": .8, "maximum_interval_width_s": 30,
                "minimum_independent_event_evidence": 1, "required_regime_coverage": ["lab"],
                "interval_coverage_evidence_requirement": 1}
    selected = select_alert_policy(origins([70, 80, 90], event=100).assign(split="validation"),
                                   events(100).assign(split="validation"),
                                   [[np.nan], [1], [1]], [30], requirements=required,
                                   candidate_policies=[{"threshold": .5}])
    assert selected["status"] == "not_passed"
    assert selected["candidate_metrics"][0]["metrics"]["useful_event_recall"] is None
    assert not selected["candidate_metrics"][0]["budget_feasible"]


def synthetic_gate_evidence():
    return dict(
        requirements={"minimum_action_lead_s": 10, "max_alert_time_fraction": 1,
                      "max_false_alert_episodes_per_100_operating_hours": 1,
                      "nominal_interval_coverage": .8, "maximum_interval_width_s": 30,
                      "minimum_independent_event_evidence": 1, "required_regime_coverage": ["lab"],
                      "interval_coverage_evidence_requirement": 1},
        provenance={"independent_holdout_available": True, "physical_identity_verified": True,
                    "physical_ids_by_part": {"holdout": ["p"]}, "exposed_physical_ids": []},
        validation_evidence={"interval_support_validated": True, "interval_independent_event_units": 1,
                             "interval_coverage": 1, "interval_maximum_width_s": 20,
                             "regime_support_validated": True, "observed_regimes": ["lab"]},
        model_frozen=True, policy_frozen=True)


@pytest.mark.parametrize("inactive_probability", [np.nan, 1.0])
def test_retained_confirmed_inactive_rows_leave_policy_scope_and_preserve_brier(inactive_probability):
    pre = origins(list(range(0, 100, 10)), event=100)
    predictions = np.zeros((10, 2))
    predictions[7:] = 1
    kwargs = synthetic_gate_evidence()
    baseline = evaluate_red_entry(pre, events(100), predictions, [20, 30], **kwargs)
    inactive = origins([100, 110, 120], event=100).assign(
        at_risk=False, risk_status=["event_observed", "post_event", "already_red"], context_known=False)
    retained = pd.concat([pre.assign(context_known=True), inactive], ignore_index=True)
    report = evaluate_red_entry(retained, events(100),
                                np.vstack([predictions, np.full((3, 2), inactive_probability)]),
                                [20, 30], **kwargs)
    assert baseline["quality_gate"]["can_pass"] and report["quality_gate"]["can_pass"]
    scope = report["metric_scope"]
    assert scope["policy_prediction_coverage"] == 1
    assert scope["policy_prediction_eligible_origin_count"] == 10
    assert scope["policy_prediction_available_origin_count"] == 10
    assert scope["policy_prediction_known_inactive_excluded_origin_count"] == 3
    assert scope["common_origin_prediction_count"] == 10
    assert report["events"] == baseline["events"]
    assert report["alert_episodes"] == [dict(baseline["alert_episodes"][0], termination="unknown")]
    assert [{key: value for key, value in alert.items() if key != "horizon_outcome"}
            for alert in report["alert_episodes"]] == alert_episodes(
                retained, np.vstack([predictions, np.full((3, 2), inactive_probability)]), [20, 30])
    for before, after in zip(baseline["brier_by_horizon"], report["brier_by_horizon"]):
        for key in ("brier", "known_origins", "unknown_origins", "physical_units", "per_unit"):
            assert after[key] == before[key]
        assert after["prediction_coverage"] == 1
        assert after["prediction_eligible_origin_count"] == after["prediction_available_origin_count"] == 10
        assert after["known_inactive_excluded_origin_count"] == 3


@pytest.mark.parametrize("unreliable", [
    {"context_known": False}, {"quality_ok": False},
    {"at_risk": False, "risk_status": "quality_unavailable"},
    {"at_risk": False, "risk_status": "rule_unavailable"},
    {"at_risk": False, "risk_status": "unknown_event_history"},
])
def test_finite_unreliable_near_event_predictions_remain_partial_and_unknown(unreliable):
    frame = origins([70, 80, 90], event=100).assign(context_known=True, quality_ok=True)
    for key, value in unreliable.items():
        frame.loc[1, key] = value
    report = evaluate_red_entry(frame, events(100), np.zeros((3, 2)), [20, 30],
                                **synthetic_gate_evidence())
    scope = report["metric_scope"]
    assert scope["policy_prediction_coverage"] == pytest.approx(2/3)
    assert scope["policy_prediction_eligible_origin_count"] == 3
    assert scope["policy_prediction_available_origin_count"] == 2
    assert scope["policy_prediction_known_inactive_excluded_origin_count"] == 0
    assert scope["common_origin_prediction_count"] == 2
    assert report["events"][0]["status"] == "unknown"
    assert report["metrics"]["useful_event_recall"] is None
    assert not report["quality_gate"]["can_pass"]
    assert "partial_policy_prediction_coverage" in report["quality_gate"]["reason_codes"]
    for brier in report["brier_by_horizon"]:
        assert brier["prediction_coverage"] == pytest.approx(2/3)
        assert brier["known_origins"] == 2 and brier["unknown_origins"] == 1
        assert brier["brier"] == (0.5 if brier["horizon_s"] == 20 else 1)


@pytest.mark.parametrize("empty", [False, True])
def test_no_eligible_policy_origins_never_passes_coverage_gate(empty):
    frame = origins([100, 110], event=100).assign(
        at_risk=False, risk_status=["event_observed", "post_event"])
    if empty:
        frame = frame.iloc[:0]
    report = evaluate_red_entry(frame, events(100), np.ones((len(frame), 1)), [30],
                                **synthetic_gate_evidence())
    scope = report["metric_scope"]
    assert scope["policy_prediction_coverage"] == 0
    assert scope["policy_prediction_eligible_origin_count"] == 0
    assert scope["policy_prediction_available_origin_count"] == 0
    assert scope["policy_prediction_known_inactive_excluded_origin_count"] == len(frame)
    assert scope["common_origin_prediction_count"] == 0
    assert not report["quality_gate"]["can_pass"]
    assert "no_eligible_policy_origins" in report["quality_gate"]["reason_codes"]
    brier = report["brier_by_horizon"][0]
    assert brier["brier"] is None
    assert brier["known_origins"] == brier["unknown_origins"] == 0


def test_common_prediction_count_requires_all_horizons_and_reliable_origin():
    report = evaluate_red_entry(origins([70, 80, 90], event=100), events(100),
                                [[np.nan, .5], [.5, .5], [.5, .5]], [20, 30])
    assert report["metric_scope"]["policy_prediction_coverage"] == 1
    assert report["metric_scope"]["common_origin_prediction_count"] == 2
    assert report["brier_by_horizon"][0]["prediction_coverage"] == pytest.approx(2/3)
    assert report["brier_by_horizon"][1]["prediction_coverage"] == 1
