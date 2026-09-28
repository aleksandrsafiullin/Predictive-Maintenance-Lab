from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import pdm.health_zones as hz


def _unit(rms, uid="Bearing9_9"):
    """Labelled synthetic bearing: one measurement per minute, given RMS curve."""
    n = len(rms)
    row = {f"{c}_{s}": 1.0 for c in hz.CHANNELS for s in hz.SIGNALS}
    df = pd.DataFrame([row] * n)
    df["horizontal_rms"] = rms
    df["vertical_rms"] = np.asarray(rms) * 0.8
    df["unit_id"] = uid
    df["timestamp_s"] = np.arange(n, dtype=float) * 60.0
    df["rpm"], df["load_kn"] = 2100.0, 12.0
    return df


def _degrading(n=120, start=60, seed=0):
    rng = np.random.default_rng(seed)
    rms = 1.0 + 0.01 * rng.standard_normal(n)
    rms[start:] += np.linspace(0.5, 6.0, n - start)
    return rms


def test_labels_red_last_30_minutes_and_yellow_from_onset():
    lab = hz.label_unit(_unit(_degrading()), hz.DEFAULT_CONFIG)
    z = lab["true_zone"].to_numpy()
    assert (z[lab["rul_s"] <= 1800] == 2).all()
    assert (z[lab["rul_s"] > 1800] < 2).all()
    assert (z[:60] == 0).all() and (z[60:89] == 1).all()


def test_small_drift_is_not_onset():
    rms = 1.0 + np.linspace(0, 0.1, 200)  # +10 % drift stays below the 1.25x ratio
    assert hz.onset_index(rms, hz.DEFAULT_CONFIG) is None


def test_smoothing_needs_confirmation_and_is_sticky():
    raw = np.array([0, 2, 0, 1, 1, 1, 0, 1, 2, 2, 0, 0])
    out = hz.smooth_zones(raw, escalate_n=2, deescalate_n=3)
    assert out.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2]


def test_features_are_causal():
    unit = hz.label_unit(_unit(_degrading()), hz.DEFAULT_CONFIG)
    full = hz.unit_features(unit, hz.DEFAULT_CONFIG)
    for k in (3, 30, 70, 100):
        prefix = hz.unit_features(unit.iloc[:k], hz.DEFAULT_CONFIG)
        np.testing.assert_allclose(prefix, full[:k])
    assert full.shape[1] == len(hz.feature_names(hz.DEFAULT_CONFIG))


def test_prediction_at_t_ignores_future_rows():
    cfg = hz.DEFAULT_CONFIG
    torch.manual_seed(0)
    model = hz.ZoneGRU(len(hz.feature_names(cfg)), 8)
    scaler = hz.Scaler(np.zeros(len(hz.feature_names(cfg))), np.ones(len(hz.feature_names(cfg))))
    rms = _degrading()
    full = hz.predict_unit(model, scaler, cfg, _unit(rms))
    cut = hz.predict_unit(model, scaler, cfg, _unit(rms[:70]))
    np.testing.assert_allclose(cut["p_red"].to_numpy(), full["p_red"].to_numpy()[:70], rtol=1e-5, atol=1e-6)


def test_rule_baseline_latches_and_escalates():
    feats = _unit(_degrading(start=40))
    pred = hz.rule_baseline(feats, ["Bearing9_9"], hz.DEFAULT_CONFIG, red_ratio=3.0)
    z = pred["zone"].to_numpy()
    assert (np.diff(z) >= 0).all()
    assert z[:40].max() == 0 and z[-1] == 2 and 1 in z


def test_metrics_report_first_red_and_misses():
    lab = hz.label_unit(_unit(_degrading()), hz.DEFAULT_CONFIG)
    perfect = lab.assign(zone=lab["true_zone"])
    m = hz.zone_metrics(perfect)
    assert m["unit_balanced_balanced_accuracy"] == pytest.approx(1.0)
    assert m["per_unit"]["Bearing9_9"]["first_red_min_before_failure"] == pytest.approx(30.0)
    never = lab.assign(zone=0)
    m = hz.zone_metrics(never)
    assert m["units_red_raised"] == 0 and m["per_unit"]["Bearing9_9"]["minutes_green_while_red"] == 31


def test_health_zones_view_without_model(monkeypatch):
    from streamlit.testing.v1 import AppTest

    import pdm.health_zones_ui as ui
    from pdm.paths import project_root

    monkeypatch.setattr(ui, "list_zone_runs", lambda dataset_id="bearings": [])
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=30)
    at.session_state["screen_selection"] = "Model Report"
    at.session_state["report_view"] = "Health zones"
    at.run()
    assert not at.exception
    assert any("pdm zones-train" in str(c.value) for c in at.code)
