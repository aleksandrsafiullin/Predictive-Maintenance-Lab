"""Filter zones are measured sensor states, never time-to-event labels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.monitoring.contracts import dataset_profile
from pdm.monitoring.evaluation import _sensor_zone_columns
from pdm.monitoring.filter_zones import (
    assess_filter_sensor_zone,
    export_filter_zone_labels,
    filter_pressure_band,
)
from pdm.monitoring.policy import default_policy
from pdm.monitoring.quality import quality_policy
from pdm.monitoring.runtime import monitoring_step
from pdm.monitoring.ui import _history_ribbon_values
from pdm.ui_theme import TOKENS


def rows(pressure, *, flow=120., feed=200.):
    n = len(pressure)
    return pd.DataFrame({
        "dataset_id": "filters", "unit_id": "filter-1",
        "timestamp_s": np.arange(n, dtype=float) * 6.,
        "differential_pressure": pressure,
        "flow_rate": flow, "dust_feed": feed, "dust": "A",
        "operating_age_s": np.arange(n, dtype=float) * 6.,
        "delta_t_s": 6., "delta_pressure": pd.Series(pressure).diff().fillna(0.),
        "gap_before": False, "quality_gap_before": False,
        "regime_id": "known",
    })


def good_quality():
    return {"critical_channel_usable": True, "data_quality_status": "valid"}


def test_red_zone_requires_observed_pressure_crossing():
    frame = rows([100., 200., 599., 600.])
    result = assess_filter_sensor_zone(frame, quality=good_quality())
    assert result["zone"] == "configured_600_pa_limit"
    assert result["pressure_pa"] == 600.
    assert "dataset_time" not in result["label"]


def test_configured_provisional_pressure_bands_and_context_gate():
    below = assess_filter_sensor_zone(rows([299.9]), quality=good_quality())
    assert below["zone"] == "below_provisional_pressure_band"
    assert below["display_zone"] == "green"
    assert "healthy" not in below["label"].lower()
    assert below["pressure_pa"] == 299.9

    at_warning = assess_filter_sensor_zone(rows([300.]), quality=good_quality())
    assert at_warning["zone"] == "pressure_warning_band"
    assert at_warning["display_zone"] == "yellow"
    assert at_warning["policy"]["version"] == "filter_sensor_zones_v2"
    assert at_warning["policy"]["yellow_limit_source"]

    no_flow = rows([350.], flow=0.)
    assert assess_filter_sensor_zone(no_flow, quality=good_quality())["display_zone"] == "gray"


@pytest.mark.parametrize(("pressure", "flow", "feed", "expected"), [
    (299.9, 100., 10., "green"), (300., 100., 10., "yellow"),
    (599.9, 100., 10., "yellow"), (600., 100., 10., "red"),
    (300., 0., 10., "gray"), (300., 100., np.nan, "gray"),
    (650., 0., np.nan, "red"), (np.nan, 100., 10., "gray"),
])
def test_runtime_and_label_band_classifier_parity(pressure, flow, feed, expected):
    frame = rows([pressure], flow=flow, feed=feed)
    runtime = assess_filter_sensor_zone(frame, quality=good_quality())
    label = filter_pressure_band(pressure, flow, feed,
        pressure_quality_valid=True, row_quality_valid=True)
    assert runtime["display_zone"] == expected
    assert runtime["display_zone"] == label["display_zone"]


def test_row_zone_is_available_during_valid_warmup_and_stale_as_of_is_gray():
    from pdm.monitoring.quality import assess_quality

    frame = rows([100.])
    profile = dataset_profile("filters")
    qpolicy = quality_policy(profile)
    warmup_quality = assess_quality(frame, profile, qpolicy, 0.)
    assert warmup_quality["data_quality_status"] == "insufficient_history"
    assert assess_filter_sensor_zone(frame, quality=warmup_quality)["display_zone"] == "green"
    stale_quality = assess_quality(frame, profile, qpolicy, 1000.)
    assert stale_quality["data_quality_status"] == "stale"
    assert assess_filter_sensor_zone(frame, quality=stale_quality)["display_zone"] == "gray"


def test_yellow_trend_is_supplementary_and_requires_stable_measured_context():
    pressure = [100., 100.5, 101.5, 102.5, 103.5]
    normality = {"reference_status": "provisional", "regime_key": "dust=A|flow=2|feed=2",
                 "residuals": {"differential_pressure": {
        "expected": 101., "scale": 1.}}}
    frame = rows(pressure)
    result = assess_filter_sensor_zone(frame, quality=good_quality(), normality=normality)
    assert result["zone"] == "below_provisional_pressure_band"
    assert result["pressure_trend"]["direction"] == "rising"
    assert result["pressure_trend"]["significant"]
    assert result["display_zone"] == "green"
    assert "significant_rising_pressure_trend" in result["supplementary_evidence"]
    assert result["policy"]["verification_status"] == "provisional_laboratory_rule"

    changed_flow = frame.copy()
    changed_flow.loc[4, "flow_rate"] = 180.
    result = assess_filter_sensor_zone(changed_flow, quality=good_quality(), normality=normality)
    assert result["zone"] == "below_provisional_pressure_band"
    assert not result["supplementary_evidence"]


def test_missing_pressure_or_poor_flow_never_becomes_green():
    missing = rows([100., 110., 120., 130., np.nan])
    result = assess_filter_sensor_zone(missing, quality=good_quality())
    assert result["zone"] == "unknown"

    poor_flow = rows([100., 110., 120., 130., 140.], flow=0.)
    result = assess_filter_sensor_zone(poor_flow, quality=good_quality())
    assert result["zone"] == "unknown"


def test_reference_band_uses_only_candidate_reference_and_measured_state():
    reference = {"reference_status": "provisional", "regime_key": "dust=A|flow=2|feed=2",
                 "residuals": {"differential_pressure": {
        "expected": 100., "scale": 10.}}}
    frame = rows([90., 100., 95., 99., 98.])
    result = assess_filter_sensor_zone(frame, quality=good_quality(), normality=reference)
    assert result["zone"] == "below_provisional_pressure_band"
    assert result["reference"]["yellow_pressure_pa"] == 130.
    stable = assess_filter_sensor_zone(rows([100., 100., 100., 100., 100.]),
                                       quality=good_quality(), normality=reference)
    assert stable["zone"] == "below_provisional_pressure_band"
    assert stable["display_zone"] == "green"
    unverified = assess_filter_sensor_zone(rows([100.] * 5), quality=good_quality(),
                                          normality={**reference, "reference_status": "unavailable"})
    assert unverified["zone"] == "below_provisional_pressure_band"
    assert "healthy" not in unverified["label"].lower()

    result = assess_filter_sensor_zone(rows([100., 110., 120., 130., 140.]),
                                       quality=good_quality(), normality=reference)
    assert result["zone"] == "below_provisional_pressure_band"
    assert result["display_zone"] == "green"
    assert "measured_pressure_above_provisional_reference_band" in result["supplementary_evidence"]


def test_monitoring_runtime_exposes_filter_sensor_zone_without_event_model():
    frame = rows(np.linspace(100., 180., 25))
    profile = dataset_profile("filters")
    reference = {"status": "unavailable", "dataset_id": "filters", "regimes": {},
                 "model_regimes": [], "residuals": {}}
    policy = default_policy(profile)
    warmup_result, _ = monitoring_step(frame.iloc[:1], profile=profile, reference=reference,
        quality_policy=quality_policy(profile), state_policy=policy,
        bundle_id="synthetic", as_of=float(frame.timestamp_s.iloc[0]))
    assert warmup_result["sensor_zone"]["zone"] == "below_provisional_pressure_band"
    result, _ = monitoring_step(frame, profile=profile, reference=reference,
        quality_policy=quality_policy(profile), state_policy=policy,
        bundle_id="synthetic", as_of=float(frame.timestamp_s.iloc[-1]))
    assert result["sensor_zone"]["zone"] == "below_provisional_pressure_band"
    assert result["sensor_zone"]["pressure_pa"] == 180.
    assert result["sensor_zone"]["flow_rate"] == 120.
    assert result["sensor_zone"]["dust_feed"] == 200.
    assert result["sensor_zone"]["policy"]["version"] == "filter_sensor_zones_v2"
    assert result["sensor_zone"]["pressure_trend"]["significant"] is False


def test_flattened_evaluation_keeps_sensor_zone_fields():
    fields = _sensor_zone_columns({"zone": "pressure_warning_band", "display_zone": "yellow",
        "status": "provisional", "reason": "measured_pressure_at_or_above_provisional_300_pa_band",
        "supplementary_evidence": ["significant_rising_pressure_trend"],
        "pressure_pa": 350., "flow_rate": 120., "dust_feed": 200.,
        "policy": {"version": "filter_sensor_zones_v2"}})
    assert fields["sensor_zone"] == "pressure_warning_band"
    assert fields["sensor_zone_color"] == "yellow"
    assert fields["sensor_pressure_pa"] == 350.
    assert fields["sensor_flow_rate_recorded"] == 120.
    assert fields["sensor_dust_feed_recorded"] == 200.
    assert fields["sensor_zone_evidence"] == "significant_rising_pressure_trend"


def test_filter_history_ribbon_uses_measured_sensor_zones_over_model_state():
    rows_in = [
        {"condition": {"display_zone": "red"},
         "sensor_zone": {"display_zone": "yellow", "label": "Pressure at warning band"}},
        {"condition": {"display_zone": "green"},
         "sensor_zone": {"display_zone": "gray", "label": "Assessment unavailable"}},
    ]
    title, colors, labels = _history_ribbon_values(rows_in, "filters", "light")
    assert title == "Filter sensor-zone history"
    assert colors == [TOKENS["light"]["zone_yellow"], TOKENS["light"]["zone_unknown"]]
    assert labels == ["Pressure at warning band", "Assessment unavailable"]


def test_legacy_filter_history_ribbon_is_explicitly_model_condition():
    rows_in = [{"condition": {"display_zone": "green"}}]
    title, colors, labels = _history_ribbon_values(rows_in, "filters", "dark")
    assert "Model condition history" in title
    assert "predates sensor zones" in title
    assert colors == [TOKENS["dark"]["zone_green"]]
    assert labels == ["Normal"]


def test_filter_zone_export_is_versioned_complete_and_hash_bound(tmp_path, monkeypatch):
    from pdm.data import prepare

    features = pd.DataFrame({
        "unit_id": ["u1", "u2", "u3", "u4", "u5"], "timestamp_s": [1., 2., 3., 4., 5.],
        "differential_pressure": [100., 300., 600., 250., 650.],
        "flow_rate": [100., 100., 100., 0., 0.], "dust_feed": [10., 10., 10., 10., 10.],
    })
    quality = pd.DataFrame({"unit_id": ["u1", "u2", "u3", "u4", "u5"],
        "timestamp_s": [1., 2., 3., 4., 5.], "status": ["admitted", "attention", "admitted", "admitted", "excluded"],
        "reasons": ["", "unusual_signal_retained", "", "", "invalid"]})
    source_dir = tmp_path / "processed"
    source_dir.mkdir()
    features.to_parquet(source_dir / "features.parquet", index=False)
    pd.DataFrame({"unit_id": ["u1", "u2", "u3", "u4", "u5"]}).to_parquet(
        source_dir / "units.parquet", index=False)
    split = {"dataset_id": "filters", "train": ["u1", "u4", "u5"],
             "validation": ["u2"], "test": ["u3"]}
    (source_dir / "split.json").write_text(json.dumps(split), encoding="utf-8")
    (source_dir / "feature_schema.json").write_text(json.dumps({"version": "test-v1"}), encoding="utf-8")
    quality.to_parquet(source_dir / "quality_records.parquet", index=False)
    from pdm.io_util import sha256_file

    data = {"features": features, "split": {"dataset_id": "filters", "train": ["u1", "u4", "u5"],
            "validation": ["u2"], "test": ["u3"]}, "dataset_version": "test-v1",
            "fingerprint": {"features_hash": sha256_file(source_dir / "features.parquet"),
                            "units_hash": sha256_file(source_dir / "units.parquet"),
                            "split_json_hash": sha256_file(source_dir / "split.json"),
                            "feature_schema_hash": sha256_file(source_dir / "feature_schema.json"),
                            "quality_records_hash": sha256_file(source_dir / "quality_records.parquet")},
            "report": {"quality": {"admitted_measurements": 4}}, "dir": source_dir}
    monkeypatch.setattr(prepare, "load_processed", lambda dataset, version=None: data)
    result = export_filter_zone_labels(output_root=tmp_path / "artifacts")
    manifest = result["manifest"]
    labels_path = Path(result["directory"]) / "labels.csv"
    table = pd.read_csv(labels_path)
    assert len(table) == manifest["row_count"] == 4
    assert table.display_zone.tolist() == ["green", "yellow", "red", "gray"]
    assert set(table.display_zone).issubset({"green", "yellow", "red", "gray"})
    assert table.sensor_zone.tolist() == ["below_provisional_pressure_band", "pressure_warning_band", "configured_600_pa_limit", "unknown"]
    assert manifest["class_counts"] == {"green": 1, "yellow": 1, "red": 1, "unknown": 1}
    assert sum(manifest["class_counts"].values()) == manifest["row_count"]
    assert sum(sum(counts.values()) for counts in manifest["class_counts_by_split"].values()) == manifest["row_count"]
    assert manifest["class_counts_by_split"]["validation"]["yellow"] == 1
    assert manifest["endpoint_or_rul_used_for_classification"] is False
    assert hashlib.sha256(labels_path.read_bytes()).hexdigest() == manifest["labels_sha256"]
    assert json.loads((labels_path.parent / "manifest.json").read_text())["artifact_id"] == result["artifact_id"]
    configured = export_filter_zone_labels(output_root=tmp_path / "configured",
                                           policy={"yellow_limit_pa": 250.})
    assert configured["manifest"]["policy"]["yellow_limit_pa"] == 250.
    assert configured["manifest"]["policy"]["yellow_limit_policy_id"] == "filter_pressure_warning_250pa_v1"

    # Existing artifact identity must not mask any source file drift.
    for filename in ("features.parquet", "units.parquet", "split.json", "feature_schema.json",
                     "quality_records.parquet"):
        path = source_dir / filename
        original = path.read_bytes()
        path.write_bytes(original + b"drift")
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            export_filter_zone_labels(output_root=tmp_path / "artifacts")
        path.write_bytes(original)


def test_zones_labels_rejects_dataset_incompatible_threshold_flags(capsys):
    from pdm.cli import main

    with pytest.raises(SystemExit) as err:
        main(["zones-labels", "--dataset", "filters", "--red-ratio", "2"])
    assert err.value.code == 2
    assert "--red-ratio applies only to bearing" in capsys.readouterr().err
    with pytest.raises(SystemExit) as err:
        main(["zones-labels", "--dataset", "bearings", "--yellow-limit-pa", "300"])
    assert err.value.code == 2
    assert "--yellow-limit-pa applies only to filter" in capsys.readouterr().err
