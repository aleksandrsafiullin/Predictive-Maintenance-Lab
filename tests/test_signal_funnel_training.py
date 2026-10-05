"""Joint mode protocol tests, separate from model-quality evidence."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

import pdm.signal_training as training
from pdm.signal_funnel import load_sampler
from tests.project_contract import make_contract_snapshot


def config(features, **overrides):
    return training._params("gru", {"forecast_mode": "joint_residual_paths", "history_length": 2,
                                     "horizons_s": [10., 20.], "epochs": 1, "hidden_size": 4,
                                     "max_windows_per_unit": 5, **overrides}, features)


def test_common_params_and_fixed_full_cns_config():
    features = pd.DataFrame({"unit_id": ["a"]*4, "timestamp_s": [0., 1., 2., 3.]})
    for engine in training.ENGINES:
        params = training._params(engine, {"forecast_mode": "joint_residual_paths"}, features)
        assert params["nominal_coverage"] == .9 and params["path_samples"] == 256
        assert params["cv_folds"] == 3 and params["epochs"] == 12
    with pytest.raises(ValueError):
        training._params("gru", {"forecast_mode": "joint_residual_paths", "max_windows_per_unit": 0}, features)


def test_physical_overlap_rejected_and_uniform_cap_across_cycles():
    features = pd.DataFrame({"unit_id": ["a"]*12+["b"]*12, "timestamp_s": list(range(12))*2,
                             "signal": list(range(12))*2, "gap_before": [False]*24})
    physical = {"a": "machine", "b": "machine"}
    params = training._params("gru", {"forecast_mode": "joint_residual_paths", "history_length": 2,
                                      "horizons_s": [1.], "max_windows_per_unit": 5}, features)
    windows = training._joint_windows(features, ["a", "b"], params, physical)
    assert len(windows["x"]) == 5
    assert windows["window_counts"]["machine"] == {"eligible": 20, "retained": 5}
    changed = features.copy()
    changed["signal"] = -999
    other = training._joint_windows(changed, ["a", "b"], params, physical)
    assert other["as_of_s"] == windows["as_of_s"]
    assert other["unit_id"] == windows["unit_id"]
    units = pd.DataFrame({"unit_id": ["a", "b", "c"], "physical_unit_id": ["machine", "machine", "c"]})
    with pytest.raises(ValueError, match="Physical"):
        training._physical_map({"units": units, "features": features,
                               "split": {"train": ["a"], "validation": ["b"], "test": ["c"]}})


def test_oof_scaler_and_fits_never_include_held_group(tmp_path, monkeypatch):
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    store, project, snapshot = make_contract_snapshot(tmp_path / "projects")
    data = training.load_snapshot(project["project_id"], snapshot["snapshot_id"])
    physical = training._physical_map(data)
    fitting = []
    def fit(engine, frame, params, scaler, stop, report):
        fitting.append(copy.deepcopy(scaler))
        assert set(frame["unit_id"]) <= set(scaler["fit_units"])
        return object()
    def predict(model, engine, frame, scaler, **kwargs):
        return np.repeat(frame["x"][:, -1, 0, None], frame["y"].shape[1], axis=1), None, None
    monkeypatch.setattr(training, "_fit_fixed", fit)
    monkeypatch.setattr(training, "_predict", predict)
    model, scaler, selection, bank = training._joint_fit(data, "gru", config(data["features"]), physical,
                                                        lambda: False, lambda payload: None)
    train_ids = set(map(str, data["split"]["train"]))
    assert len(fitting) == selection["cv_folds_actual"] + 1
    for fold in selection["folds"]:
        assert not set(fold["held_physical_groups"]) & set(fold["fit_physical_groups"])
        expected = data["features"].loc[data["features"].unit_id.astype(str).isin(fold["fit_units"]), "signal"].mean()
        assert fold["scaler"]["mean"] == pytest.approx(expected)
        assert set(fold["fit_units"]) | set(fold["held_units"]) == train_ids
    assert scaler["fit_units"] == sorted(train_ids)
    assert bank["excluded_incomplete_count"] > 0
    assert np.isfinite(bank["residuals"]).all()


@pytest.mark.parametrize("engine", ["gru", "lstm", "quantile_boosting", "full_cns"])
def test_joint_run_artifacts_reload_and_sparse_calibration(tmp_path, monkeypatch, engine, request):
    if engine == "full_cns":
        request.getfixturevalue("signal_cns_fixture")
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(tmp_path / "projects"))
    store, project, snapshot = make_contract_snapshot(tmp_path / "projects")
    params = {"forecast_mode": "joint_residual_paths", "history_length": 2, "horizons_s": [10., 20.],
              "epochs": 1, "hidden_size": 4, "batch_size": 16, "max_iter": 2,
              "max_windows_per_unit": 5, "path_samples": 8, "cv_folds": 3}
    run = training.train_signal_run(project["project_id"], snapshot["snapshot_id"], engine, params)
    loaded = training.load_signal_run(project["project_id"], run["run_id"])
    assert run["selection"]["criterion"] == "predeclared_fixed_configuration"
    assert run["selection"]["validation_role"] == "calibration_only"
    assert run["funnel"]["calibration_status"] == "insufficient_calibration"
    assert run["metrics"]["test"]["funnel"]["joint_energy_score"] is not None
    bank = load_sampler(loaded["dir"] / run["funnel"]["artifact"], run["artifacts"][run["funnel"]["artifact"]])
    assert bank["complete_vector_count"] > 0
    artifact = loaded["dir"] / run["funnel"]["metadata"]
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        training.load_signal_run(project["project_id"], run["run_id"])


def test_calibration_origin_does_not_shift_past_targetless_initial_segment():
    features = pd.DataFrame({"unit_id": ["u"]*8, "timestamp_s": [0., 1., 2., 10., 11., 12., 13., 14.],
                             "signal": np.arange(8.), "gap_before": [True, False, False, True, False, False, False, False]})
    params = training._params("gru", {"history_length": 2, "horizons_s": [2., 3.]}, features)
    regular = training._windows(features, ["u"], params)
    calibration = training._joint_windows(features, ["u"], {**params, "include_targetless": True}, {"u": "physical"})
    assert regular["as_of_s"][0] == 11.
    assert calibration["as_of_s"][0] == 1.
    assert not calibration["mask"][0].any()


def test_unfittable_oof_horizon_is_excluded_not_filled(monkeypatch):
    parts = []
    for uid, count in [("short", 4), ("long", 9), ("validation", 9), ("test", 9)]:
        parts.append(pd.DataFrame({"unit_id": [uid]*count, "timestamp_s": np.arange(count, dtype=float),
                                   "signal": np.arange(count, dtype=float), "gap_before": [False]*count}))
    features = pd.concat(parts, ignore_index=True)
    data = {"features": features, "units": features[["unit_id"]].drop_duplicates(),
            "split": {"train": ["short", "long"], "validation": ["validation"], "test": ["test"]},
            "schema": {"output_domain": "real"}}
    physical = training._physical_map(data)
    params = training._params("gru", {"forecast_mode": "joint_residual_paths", "history_length": 2,
                                      "horizons_s": [1., 5.]}, features[features.unit_id.isin(data["split"]["train"])])
    monkeypatch.setattr(training, "_fit_fixed", lambda *_args: object())
    monkeypatch.setattr(training, "_predict", lambda model, engine, frame, scaler, **kwargs:
                        (np.repeat(frame["x"][:, -1, 0, None], 2, axis=1), None, None))
    _, _, selection, bank = training._joint_fit(data, "gru", params, physical, lambda: False, lambda _: None)
    assert any(row.get("status") == "insufficient_fit_horizon_support" for row in selection["folds"])
    assert bank["status"] == "insufficient_support"
    assert bank["excluded_unfittable_fold_origin_count"] > 0
    assert bank["complete_vector_count"] == 0


def test_oof_clock_allocation_balances_support_and_ignores_signals_and_holdouts():
    parts = []
    for uid, count in [("long_a", 12), ("long_b", 11), ("short_a", 5), ("short_b", 30),
                       ("cycle_a", 5), ("validation", 50), ("test", 60)]:
        parts.append(pd.DataFrame({"unit_id": [uid]*count, "timestamp_s": np.arange(count, dtype=float),
                                   "signal": np.arange(count, dtype=float), "gap_before": [False]*count}))
    features = pd.concat(parts, ignore_index=True)
    features.loc[features.unit_id.eq("short_b"), "gap_before"] = np.arange(30) % 3 == 0
    train_ids = ["long_a", "long_b", "short_a", "short_b", "cycle_a"]
    physical = {uid: uid for uid in features.unit_id.unique()}
    physical["cycle_a"] = "long_a"
    params = config(features, horizons_s=[1., 7.], max_windows_per_unit=128)
    held_folds, metadata = training._joint_oof_allocation(features, train_ids, params, physical)
    assert len(held_folds) == 3
    assert sorted(group for held in held_folds for group in held) == ["long_a", "long_b", "short_a", "short_b"]
    assert next(i for i, held in enumerate(held_folds) if "long_a" in held) != next(
        i for i, held in enumerate(held_folds) if "long_b" in held)
    assert {row["unit_id"] for row in metadata["group_clock_geometry"]["long_a"]["units"]} == {"long_a", "cycle_a"}
    assert metadata["group_clock_geometry"]["short_b"]["max_continuous_duration_s"] == 2.
    changed = features.copy()
    changed["signal"] = -999.
    holdouts = changed.unit_id.isin(["validation", "test"])
    changed.loc[holdouts, "timestamp_s"] *= 10000.
    changed.loc[holdouts, "gap_before"] = True
    changed["event_label"] = True
    assert training._joint_oof_allocation(changed, list(reversed(train_ids)), params, physical) == (held_folds, metadata)
    for held in held_folds:
        fit_ids = [uid for uid in train_ids if physical[uid] not in held]
        held_ids = [uid for uid in train_ids if physical[uid] in held]
        assert ("long_a" in held_ids) == ("cycle_a" in held_ids)
        assert not {physical[uid] for uid in fit_ids} & {physical[uid] for uid in held_ids}
        frame = training._joint_windows(features, fit_ids, params, physical)
        assert frame["mask"].any(axis=0).all()


def test_display_metrics_use_raw_fallback_outside_calibration_scope():
    frame = {"x": np.zeros((2, 2, 1)), "y": np.full((2, 1), 10.), "mask": np.ones((2, 1), bool),
             "unit_id": ["u", "u"], "physical_unit_id": ["p", "p"], "as_of_s": [1., 2.],
             "origin_within_calibration_scope": [True, False]}
    paths = np.zeros((2, 8, 2))
    params = {"nominal_coverage": .9, "horizons_s": [1.]}
    calibration = {"status": "calibrated", "expansion": 20.,
                   "guarantee_scope": "predeclared_earliest_origin_per_physical_group_only"}
    metrics = training._joint_metrics(paths, frame, params, {"std": 1.}, calibration)
    assert metrics["interval_kind"] == "raw_empirical_pointwise_band_from_joint_samples"
    assert metrics["whole_path_coverage"] == 0.
    display = metrics["display_band"]
    assert display["status"] == "scope_aware_mixed_bands"
    assert display["calibrated_origin_count"] == 1 and display["raw_origin_count"] == 1
    assert display["whole_path_coverage"] == .5
    assert display["band_width_by_horizon"] == [0., 20.]
    assert display["earliest_origin_diagnostics"]["whole_path_coverage"] == 1.
    assert display["all_origin_coverage_guarantee"] is False
