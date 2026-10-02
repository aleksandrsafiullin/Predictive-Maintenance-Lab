"""Shared project zone rule. Frames here are synthetic, not XJTU-SY/HSE data."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pdm.project_zones import (
    ZONES,
    describe_rule,
    has_valid_rule,
    is_beyond,
    label_unit,
    resolve_thresholds,
    zone_counts,
    zone_labels,
)


def _unit(values, uid="U1", gaps=None):
    n = len(values)
    return pd.DataFrame({
        "unit_id": [uid] * n,
        "timestamp_s": np.arange(n, dtype=float) * 10.0,
        "signal": np.asarray(values, float),
        "gap_before": [True] + [False] * (n - 1) if gaps is None else gaps,
    })


ABOVE = {"signal_unit": "g", "thresholds": {"mode": "absolute", "direction": "above", "yellow": 0.4, "red": 0.8}}
BELOW = {"signal_unit": "bar", "thresholds": {"mode": "absolute", "direction": "below", "yellow": -1.0, "red": -2.0}}
BASELINE = {"thresholds": {"mode": "initial_baseline_multiple", "direction": "above", "yellow": None, "red": None,
                           "baseline_n": 3, "onset_sigma": 1.5, "onset_ratio": 1.1, "red_ratio": 1.7}}


def test_is_beyond_inclusive_and_nan_false():
    assert is_beyond([0.79, 0.8, 0.9, np.nan], 0.8, "above").tolist() == [False, True, True, False]
    assert is_beyond([-1.9, -2.0, -3.0, np.nan], -2.0, "below").tolist() == [False, True, True, False]
    assert bool(is_beyond(0.8, 0.8, "above"))


def test_absolute_above_boundaries():
    labels = label_unit(_unit([0.1, 0.4, 0.5, 0.8, 1.2]), ABOVE)
    assert labels.zone.tolist() == ["green", "yellow", "yellow", "red", "red"]
    assert labels.yellow_limit.eq(0.4).all() and labels.red_limit.eq(0.8).all()
    assert list(labels.columns) == ["unit_id", "timestamp_s", "signal", "gap_before", "zone",
                                    "yellow_limit", "red_limit"]


def test_absolute_below_signed_mirrored():
    labels = label_unit(_unit([0.5, -1.0, -1.5, -2.0, -5.0]), BELOW)
    assert labels.zone.tolist() == ["green", "yellow", "yellow", "red", "red"]
    text = describe_rule(BELOW)
    assert "yellow at ≤ -1 bar" in text and "red at ≤ -2 bar" in text


def test_absolute_without_yellow_is_green_or_red_only():
    schema = {"thresholds": {"mode": "absolute", "direction": "above", "yellow": None, "red": 0.8}}
    assert label_unit(_unit([0.1, 0.5, 0.8]), schema).zone.tolist() == ["green", "green", "red"]
    assert "yellow" not in describe_rule(schema)
    assert "red at ≥ 0.8" in describe_rule(schema)


def test_baseline_mode_uses_first_n_rows_and_leaves_n_minus_one_unknown():
    values = [1.0, 1.2, 0.9, 1.0, 1.6, 2.5, 3.0]
    unit = _unit(values)
    labels = label_unit(unit, BASELINE)
    assert (labels.zone == "unknown").sum() == 2
    assert labels.zone.iloc[:2].tolist() == ["unknown", "unknown"]
    expected = resolve_thresholds(BASELINE, unit.iloc[:3])
    assert expected["status"] == "available"
    assert labels.yellow_limit.iloc[2:].eq(expected["yellow"]).all()
    assert labels.red_limit.iloc[2:].eq(expected["red"]).all()
    assert labels.yellow_limit.iloc[:2].isna().all() and labels.red_limit.iloc[:2].isna().all()
    assert labels.zone.iloc[2:].tolist() == zone_labels(values[2:], expected).tolist()
    # Future rows never move the baseline.
    later = unit.copy()
    later.loc[5:, "signal"] = 1e6
    assert label_unit(later, BASELINE).red_limit.iloc[2] == expected["red"]
    text = describe_rule(BASELINE)
    for token in ("first 3 measurements", "1.5 × SD", "1.1 × median", "red = 1.7 × median",
                  "The first 2 measurements are not zoned."):
        assert token in text


def test_baseline_unit_shorter_than_n_is_all_unknown():
    assert set(label_unit(_unit([1.0, 1.1]), BASELINE).zone) == {"unknown"}


def test_describe_rule_appends_note_and_default_baseline_numbers():
    schema = {"signal_unit": "Pa", "thresholds": {
        "mode": "absolute", "direction": "above", "yellow": 300.0, "red": 600.0,
        "note": "Provisional laboratory bands; not confirmed industrial fault limits"}}
    text = describe_rule(schema)
    assert "yellow at ≥ 300 Pa, red at ≥ 600 Pa" in text
    assert text.endswith("Provisional laboratory bands; not confirmed industrial fault limits.")
    default = describe_rule({"thresholds": {"mode": "initial_baseline_multiple", "direction": "above"}})
    assert "first 5 measurements" in default and "3 × SD" in default and "2 × median" in default


def test_gaps_do_not_change_labels():
    values = [0.1, 0.5, 0.9, 0.2, 0.85, 0.45]
    base = label_unit(_unit(values), ABOVE)
    gapped = label_unit(_unit(values, gaps=[True, False, True, True, False, True]), ABOVE)
    assert base.zone.tolist() == gapped.zone.tolist()
    base_b = label_unit(_unit(values), BASELINE)
    gapped_b = label_unit(_unit(values, gaps=[True, True, True, False, True, False]), BASELINE)
    assert base_b.zone.tolist() == gapped_b.zone.tolist()


@pytest.mark.parametrize("thresholds", [
    {"mode": "absolute", "direction": "sideways", "yellow": 0.4, "red": 0.8},
    {"mode": "absolute", "direction": "above", "yellow": 0.4},
    {"mode": "absolute", "direction": "above", "yellow": 0.4, "red": float("nan")},
    {"mode": "absolute", "direction": "above", "yellow": float("inf"), "red": 0.8},
    {"mode": "percentile", "direction": "above"},
])
def test_invalid_rule_is_unknown_and_not_zoned(thresholds):
    schema = {"thresholds": thresholds}
    labels = label_unit(_unit([0.1, 0.5, 1.0]), schema)
    assert set(labels.zone) == {"unknown"}
    assert labels.red_limit.isna().all() and labels.yellow_limit.isna().all()
    assert "not zoned" in describe_rule(schema)


def test_zone_counts_sum_per_unit_labels():
    features = pd.concat([_unit([0.1, 0.5, 0.9], "A"), _unit([0.9, 0.95], "B"), _unit([0.0], "C")])
    counts = zone_counts(features, ["A", "B", "missing"], ABOVE)
    assert set(counts) == set(ZONES)
    assert counts == {"green": 1, "yellow": 1, "red": 3, "unknown": 0}
    counts_b = zone_counts(features, ["A", "B", "C"], BASELINE)
    expected = pd.concat([label_unit(features[features.unit_id == u], BASELINE) for u in "ABC"]).zone.value_counts()
    assert counts_b == {z: int(expected.get(z, 0)) for z in ZONES}
    assert counts_b["unknown"] == 2 + 2 + 1


def test_missing_signal_is_unknown_not_green():
    assert label_unit(_unit([0.1, np.nan, 0.9]), ABOVE).zone.tolist() == ["green", "unknown", "red"]


def test_baseline_with_missing_value_is_unavailable_and_unknown():
    unit = _unit([1.0, np.nan, 1.1, 1.0, 50.0])
    assert resolve_thresholds(BASELINE, unit.iloc[:3])["status"] == "unavailable"
    labels = label_unit(unit, BASELINE)
    assert set(labels.zone) == {"unknown"}
    assert labels.red_limit.isna().all() and labels.yellow_limit.isna().all()


def test_has_valid_rule_checks_schema_not_caption():
    assert not has_valid_rule({"signal_label": "Vibration", "signal_unit": "g"})
    assert not has_valid_rule({"thresholds": {}})
    assert has_valid_rule(ABOVE) and has_valid_rule(BELOW) and has_valid_rule(BASELINE)
    assert not has_valid_rule({"thresholds": {**BASELINE["thresholds"], "red_ratio": float("nan")}})
    assert not has_valid_rule({"thresholds": {**BASELINE["thresholds"], "baseline_n": 0}})
    contract = {"thresholds": {"mode": "absolute", "direction": "above", "yellow": 0.4, "red": 0.8}}
    assert has_valid_rule(contract)


@pytest.mark.parametrize("thresholds", [
    {"mode": "absolute", "direction": "sideways", "yellow": 0.4, "red": 0.8},
    {"mode": "absolute", "direction": "above", "yellow": 0.4},
    {"mode": "percentile", "direction": "above"},
])
def test_invalid_rule_is_not_valid_and_described_as_not_zoned(thresholds):
    assert not has_valid_rule({"thresholds": thresholds})
    assert "not zoned" in describe_rule({"thresholds": thresholds})


def test_quality_ui_branch_ignores_no_zones_copy():
    import pdm.project_quality_ui as quality_ui
    labelled = label_unit(_unit([0.1, 0.5, 0.9]), ABOVE)
    names = [trace.name for trace in quality_ui.zone_figure(labelled, ABOVE, "Vibration", "g", "dark").data]
    assert names == ["Vibration", "Green · 1", "Yellow · 1", "Red · 1"]
    assert describe_rule(ABOVE) not in names
    bare = quality_ui.zone_figure(labelled, {"signal_label": "Vibration"}, "Vibration", "g", "dark")
    assert [trace.name for trace in bare.data] == ["Vibration"]


def test_label_unit_sorts_by_timestamp():
    unit = _unit([0.1, 0.5, 0.9]).iloc[::-1]
    labels = label_unit(unit, ABOVE)
    assert labels.timestamp_s.is_monotonic_increasing
    assert labels.zone.tolist() == ["green", "yellow", "red"]
