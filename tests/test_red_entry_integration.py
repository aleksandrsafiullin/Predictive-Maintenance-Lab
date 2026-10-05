"""Real frozen-run tests of policy chronology, partial support and exposure."""
import json

import numpy as np
import pandas as pd
import pytest

import pdm.red_entry_inference as inference
import pdm.red_entry_training as training
from pdm.data.project_prepare import freeze_red_entry_exposure
from pdm.red_entry_evaluation import evaluate_red_entry
from pdm.red_entry_protocol import build_run_contract


def fixture_snapshot():
    frame = pd.DataFrame(
        [
            {
                "unit_id": u,
                "physical_unit_id": u,
                "timestamp_s": float(t),
                "signal": float(t + 1),
                "gap_before": False,
            }
            for u in ("a", "b", "c")
            for t in range(6)
        ]
    )
    return {
        "snapshot_id": "s1",
        "features": frame,
        "units": pd.DataFrame({"unit_id": ["a", "b", "c"], "physical_unit_id": ["a", "b", "c"]}),
        "split": {"train": ["a"], "validation": ["b"], "test": ["c"], "holdout": []},
        "schema": {
            "source_kind": "generic_sensor_csv",
            "signal_column": "signal",
            "signal_unit": "u",
            "thresholds": {"mode": "absolute", "red": 5.0, "direction": "above"},
            "columns": {
                "signal": {"role": "sensor"},
                "unit_id": {"role": "metadata"},
                "timestamp_s": {"role": "metadata"},
            },
        },
    }


class Store:
    def __init__(self, root):
        self.root = root

    def run_path(self, p, r):
        return self.root / p / "runs" / r

    def project_path(self, p):
        return self.root / p

    def get(self, p):
        return {"active_snapshot_id": "other"}





@pytest.fixture
def bound(monkeypatch, tmp_path):
    data = fixture_snapshot()
    data["features"]["is_running"] = True
    data["schema"]["thresholds"]["red"] = 50.0
    store = Store(tmp_path)
    monkeypatch.setattr(training, "load_snapshot", lambda *a: data)
    monkeypatch.setattr(training, "load_zone_limits", lambda *a: None)
    monkeypatch.setattr(training, "project_store", lambda: store)
    return data, store


def test_partial_tail_preserves_supported_metrics_and_frozen_policy(bound):
    _, store = bound
    manifest = training.train_red_entry_run("p", "s1", "always_no_entry", {"horizons_s": [1., 2., 100.]})
    run = training.load_red_entry_run("p", manifest["run_id"])
    policy = json.loads((run["dir"] / "policy.json").read_text())
    assert policy == run["contract"]["policy"] and policy["frozen"]
    assert policy["status"] == "requirements_unset"
    for part in ("validation", "test"):
        report = manifest["metrics"][part]
        assert [entry["brier"] for entry in report["brier_by_horizon"]] == [0., 0., None]
        assert report["brier_by_horizon"][2]["prediction_coverage"] == 0
        assert not report["quality_gate"]["can_pass"]
    forecast = inference.forecast_red_entry_prefix("p", manifest["run_id"], "c", 2.)
    assert forecast["warning"]["status"] == "unavailable"
    assert not forecast["warning"]["operational_quality_claim"]


def test_missing_prediction_never_bridges_release_or_invents_exposure():
    rows = pd.DataFrame(dict(unit_id=["u"]*3, timestamp_s=[0., 1., 2.],
        at_risk=[True]*3, risk_status=["at_risk"]*3, is_running=[True,False,True],
        event_observed=[False]*3, followup_duration_s=[10.]*3))
    probabilities = np.array([[.9], [np.nan], [.9]])
    report = evaluate_red_entry(rows, pd.DataFrame(), probabilities, [3.],
        policy={"threshold": .8, "release_threshold": .4, "confirmation_s": 2.})
    assert report["alert_episodes"] == []
    assert report["metrics"]["operating_exposure_s"] == 0
    assert report["metric_scope"]["policy_prediction_coverage"] == pytest.approx(2/3)
    assert "partial_policy_prediction_coverage" in report["quality_gate"]["reason_codes"]
    rows["is_running"] = True
    rows["context_known"] = [True,False,True]
    assert evaluate_red_entry(rows, pd.DataFrame(), probabilities, [3.])["metrics"]["operating_exposure_s"] == 2


def test_frozen_inference_replays_confirmation_release_and_uses_calibrator(bound, monkeypatch):
    data, _ = bound
    manifest = training.train_red_entry_run("p", "s1", "always_no_entry", {"horizons_s": [1.]})
    run = training.load_red_entry_run("p", manifest["run_id"])
    run["contract"]["policy"]["config"].update(threshold=.7, release_threshold=.5, confirmation_s=2., cooldown_s=2.)
    run["contract"]["calibration"]["temperature"] = .5
    monkeypatch.setattr(inference, "load_red_entry_run", lambda *a: run)
    def probabilities(model, batch, **kwargs):
        return np.array([[.8] if t != 1 else [.1] for t in batch["origins"].timestamp_s])
    monkeypatch.setattr(inference, "predict_hazard", probabilities)
    assert inference.forecast_red_entry_prefix("p", manifest["run_id"], "c", 2.)["warning"]["status"] == "inactive"
    result = inference.forecast_red_entry_prefix("p", manifest["run_id"], "c", 4.)
    assert result["warning"]["status"] == "active"
    assert result["warning"]["confirmed_at_s"] == 4.
    assert result["probability_by_horizon"][0]["probability"] == pytest.approx(16/17)
    data["features"].loc[data["features"].timestamp_s > 4, "signal"] = 999.
    assert inference.forecast_red_entry_prefix("p", manifest["run_id"], "c", 4.) == result


def test_real_run_origin_cap_keeps_full_chronology(bound):
    data, _ = bound
    data["features"].loc[data["features"].timestamp_s == 1, "is_running"] = False
    manifest = training.train_red_entry_run("p", "s1", "always_no_entry", {
        "horizons_s": [1.], "maximum_origins_per_physical_unit": 2, "history_length": 1})
    report = manifest["metrics"]["test"]
    assert report["metrics"]["operating_exposure_s"] == 3.
    assert report["metric_scope"]["policy_prediction_coverage"] == pytest.approx(1/3)
    assert not report["quality_gate"]["can_pass"]


def test_persisted_physical_exposure_survives_reassignment(tmp_path):
    data = fixture_snapshot()
    store = Store(tmp_path)
    first = freeze_red_entry_exposure("p", data, store=store)
    assert "c" in first["exposed_physical_ids"]
    data["split"] = {"train": ["a"], "validation": ["b"], "test": [], "holdout": ["c"]}
    restored = freeze_red_entry_exposure("p", data, store=store)
    contract = build_run_contract(data, horizons_s=[1.], split_provenance=restored)
    assert "c" in restored["exposed_physical_ids"]
    assert not contract["split_provenance"]["independent_holdout_available"]


def test_policy_freezes_before_test_target_access(bound, monkeypatch):
    import pdm.red_entry_policy as policies
    calls = []
    original_targets = training.build_red_entry_targets
    original_policy = policies.select_alert_policy
    def targets(data, *args, **kwargs):
        calls.append("all_targets" if "c" in set(data["features"].unit_id) else "development_targets")
        return original_targets(data, *args, **kwargs)
    def policy(origins, events, *args, **kwargs):
        assert set(origins.split) == {"validation"}
        calls.append("freeze_policy")
        return original_policy(origins, events, *args, **kwargs)
    monkeypatch.setattr(training, "build_red_entry_targets", targets)
    monkeypatch.setattr(policies, "select_alert_policy", policy)
    training.train_red_entry_run("p", "s1", "always_no_entry", {"horizons_s": [1.]})
    assert calls == ["development_targets", "freeze_policy", "all_targets"]
