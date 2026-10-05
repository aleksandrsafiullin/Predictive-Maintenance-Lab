import numpy as np
import pandas as pd
import pytest

from pdm.red_entry_targets import build_red_entry_targets


def snapshot(values, times=None, **columns):
    n = len(values)
    return {"snapshot_id": "fixture", "schema": {"signal_column": "signal", "signal_unit": "Pa",
             "thresholds": {"mode": "absolute", "direction": "above", "yellow": 5, "red": 10}},
            "split": {"train": ["u"], "validation": [], "test": []},
            "features": pd.DataFrame({"unit_id": ["u"] * n, "signal": values,
                                      "timestamp_s": list(range(n)) if times is None else times, **columns})}


def test_right_inclusive_event_and_post_event_masks():
    result = build_red_entry_targets(snapshot([1, 1, 10, 1, 11]), [1, 2, 3])
    assert result["hazard_targets"][0].tolist() == [0, 1, 0]
    assert result["hazard_mask"][0].tolist() == [True, True, False]
    assert result["hazard_targets"][1].tolist() == [1, 0, 0]
    assert result["hazard_mask"][2:].sum() == 0
    assert result["origins"].risk_status.tolist() == ["at_risk", "at_risk", "event_observed", "post_event", "post_event"]
    assert result["events"].iloc[0].transition_left_s == 1
    assert result["events"].iloc[0].transition_right_s == 2


def test_right_censor_only_full_bins_and_no_failure():
    result = build_red_entry_targets(snapshot([1, 2, 3], times=[0, 1, 2.5]), [1, 2, 3])
    assert result["hazard_mask"][0].tolist() == [True, True, False]
    assert result["hazard_targets"].sum() == 0
    assert result["origins"].iloc[0].followup_duration_s == 2.5
    event = result["events"].iloc[0]
    assert pd.isna(event.first_red_timestamp_s)
    assert event.censor_reason == "observation_end"
    assert not event.first_event_verified


@pytest.mark.parametrize("extra,reason", [({"gap_before": [False, False, True, False, False]}, "gap"),
                                            ({"quality_ok": [True, True, False, True, True]}, "quality"),
                                            ({"usable": [True, True, False, True, True]}, "quality")])
def test_break_censors_at_prior_usable_row_and_history_stays_unknown(extra, reason):
    result = build_red_entry_targets(snapshot([1, 2, 3, 4, 10], **extra), [1, 2, 4])
    assert result["hazard_mask"][0].tolist() == [True, False, False]
    assert result["origins"].iloc[0].followup_duration_s == 1
    assert result["origins"].iloc[0].censor_reason == reason
    assert not result["origins"].iloc[3].at_risk
    assert result["origins"].iloc[3].risk_status == "unknown_event_history"
    assert result["hazard_targets"].sum() == 0
    assert not result["events"].iloc[0].first_event_verified


def test_explicit_new_cycle_reopens_after_event_gap_boundary_is_not_missing_history():
    result = build_red_entry_targets(snapshot([1, 10, 1, 2, 10], cycle_id=["a", "a", "b", "b", "b"],
                                              gap_before=[True, False, True, False, False],
                                              physical_unit_id=["equipment"] * 5), [1, 2, 3])
    assert result["origins"].iloc[2].at_risk
    assert result["hazard_targets"][2].tolist() == [0, 1, 0]
    assert len(result["events"]) == 2
    assert set(result["origins"].physical_unit_id) == {"equipment"}


def test_cycle_boundary_censors_followup():
    result = build_red_entry_targets(snapshot([1, 2, 10], cycle_id=["a", "a", "b"]), [1, 2])
    assert result["hazard_mask"][0].tolist() == [True, False]
    assert result["origins"].iloc[0].censor_reason == "new_cycle"


def test_replacement_reopens_even_with_unchanged_external_cycle_id():
    source = snapshot([1, 10, 1, 2, 10], component_cycle_id=["same"] * 5,
                      component_replaced=[False, False, True, False, False])
    result = build_red_entry_targets(source, [1, 2])
    assert len(result["events"]) == 2
    assert result["origins"].iloc[2].at_risk
    assert result["hazard_targets"][2].tolist() == [0, 1]
    source["features"]["signal"] = [1, 2, 1, 2, 10]
    result = build_red_entry_targets(source, [1, 2])
    assert result["origins"].iloc[0].censor_reason == "new_cycle"
    assert result["hazard_mask"][0].tolist() == [True, False]


def test_relative_baseline_is_unavailable_until_nth_causal_sample():
    source = snapshot([2, 2, 2, 4, 2])
    source["schema"]["thresholds"] = {"mode": "initial_baseline_multiple", "baseline_n": 3, "red_ratio": 2}
    result = build_red_entry_targets(source, [1, 2])
    assert result["origins"].risk_status.tolist()[:4] == ["rule_unavailable", "rule_unavailable", "at_risk", "event_observed"]
    assert result["origins"].red_limit.iloc[:2].isna().all()
    assert result["hazard_mask"][:2].sum() == 0
    assert result["hazard_targets"][2].tolist() == [1, 0]


def test_risk_status_prefix_invariant_to_future_values_and_breaks():
    short = snapshot([1, 2, 3])
    long = snapshot([1, 2, 3, 10, 1], gap_before=[False, False, False, True, False])
    a, b = [build_red_entry_targets(s, [1, 2])["origins"] for s in (short, long)]
    pd.testing.assert_frame_equal(a[["risk_status", "at_risk", "zone", "red_limit"]],
                                  b.iloc[:3][["risk_status", "at_risk", "zone", "red_limit"]])


def test_effective_rule_changes_hash_and_targets_without_mutating_saved_schema():
    source = snapshot([1, 8, 10])
    default = build_red_entry_targets(source, [1, 2])
    override = {**source["schema"], "thresholds": {"mode": "absolute", "red": 8}}
    changed = build_red_entry_targets(source, [1, 2], schema=override)
    assert default["identity"]["red_rule"]["red_rule_hash"] != changed["identity"]["red_rule"]["red_rule_hash"]
    assert default["target_hash"] != changed["target_hash"]
    assert changed["hazard_targets"][0].tolist() == [1, 0]
    assert source["schema"]["thresholds"]["red"] == 10


def test_below_inclusive_rule_and_separate_real_endpoint_records():
    source = snapshot([20, 10, 5], planned_maintenance=[False, False, True], maintenance_timestamp_s=[np.nan, np.nan, 2])
    source["schema"]["thresholds"]["direction"] = "below"
    result = build_red_entry_targets(source, [1, 2])
    assert result["hazard_targets"][0].tolist() == [1, 0]
    assert result["events"].iloc[0].endpoint_records[-1]["maintenance_timestamp_s"] == 2


@pytest.mark.parametrize("grid", [[], [0], [2, 1], [1, 1], [np.inf]])
def test_invalid_grids_rejected(grid):
    with pytest.raises(ValueError):
        build_red_entry_targets(snapshot([1, 2]), grid)


def test_event_age_metadata_requires_explicit_known_source_and_censors_before_gap():
    source = snapshot([1, 2, 1, 10], times=[100, 101, 110, 111],
                      gap_before=[False, False, True, False],
                      operating_age_s=[1000, 1001, 1010, 1011], operating_age_known=[True] * 4,
                      operating_age_source=["counter"] * 4, rpm=[100, 100, 200, 200])
    event = build_red_entry_targets(source, [1, 2])["events"].iloc[0]
    assert event.observation_start_s == 100
    assert event.age_at_start_s == 1000
    assert event.age_at_censor_s == 1001
    assert event.age_at_censor_source == "counter"
    assert pd.isna(event.age_at_event_s)  # recorded RED after gap is not verified first event
    assert event.reliable_followup_end_s == 101
    assert event.regime_metadata["at_censor"]["rpm"] == 100
    source["features"]["operating_age_source"] = "unknown"
    event = build_red_entry_targets(source, [1, 2])["events"].iloc[0]
    assert pd.isna(event.age_at_start_s)
    assert pd.isna(event.age_at_censor_s)


def test_verified_event_preserves_lab_proxy_age_separately_from_timestamp():
    source = snapshot([1, 10], operating_age_s=[900, 901], operating_age_known=[True, True],
                      operating_age_source=["laboratory_proxy"] * 2)
    event = build_red_entry_targets(source, [1, 2])["events"].iloc[0]
    assert event.age_at_event_s == 901
    assert event.age_at_event_source == "laboratory_proxy"
    assert pd.isna(event.age_at_censor_s)
