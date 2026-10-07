"""Bounded widths, independent fitting, persisted settings and real replay."""

import copy

import numpy as np
import pytest

from pdm import corridor_calibration as cal
from pdm.io_util import atomic_write_json, read_json
from pdm.projects import project_store
from tests.test_sensor_import_view import app, original_import, seal
from tests.test_sensor_import_view import release as sensor_release


def population(y, *, mask=None, physical=None, split="calibration"):
    y = np.asarray(y, float)
    mask = np.ones_like(y, bool) if mask is None else np.asarray(mask, bool)
    raw = np.ones((*y.shape, 3), float)
    raw[..., 0], raw[..., 2] = 0.775, 1.225
    return raw, dict(y=y, mask=mask, physical=np.asarray(physical or list(map(str, range(len(y))))), split=split)


def test_hard_maximum_preserves_center_and_reports_unmet_target():
    raw, data = population([[2, 3, 4], [2, 3, 4]])
    result = cal.fit_width(raw, data)
    assert result["width"] == .45 and result["required_width"] > .45
    assert result["upper_limit_reached"] and not result["target_met"]
    corrected = cal.apply_width(raw, result)
    np.testing.assert_array_equal(corrected[..., 1], raw[..., 1])
    np.testing.assert_allclose((corrected[..., 2] - corrected[..., 0]) / corrected[..., 1], .45)
    assert not result["coverage_guarantee"]


def test_floor_and_target_quantile_choose_width_inside_limits():
    raw, data = population([[1, 1], [1, 1]])
    assert cal.fit_width(raw, data)["width"] == .20
    raw, data = population([[1.05, 1.15], [1.05, 1.15]])
    result = cal.fit_width(raw, data)
    assert result["width"] == pytest.approx(.30) and result["target_met"]


def test_units_have_equal_weight_despite_unequal_number_of_origins():
    raw, data = population([[1]] * 100 + [[1.2]], physical=["a"] * 100 + ["b"])
    result = cal.fit_width(raw, data, dict(target_coverage=.75))
    assert result["width"] == pytest.approx(.4)


def test_unknown_tail_is_not_covered_or_a_complete_trajectory():
    raw, data = population([[1, 999], [1, 999]], mask=[[True, False]] * 2)
    result = cal.fit_width(raw, data)
    assert result["supported_points"] == 2 and result["point_coverage"] == 1
    assert result["whole_path_coverage"] is None and result["support_by_lead"] == [2, 0]
    with pytest.raises(ValueError, match="unknown"):
        cal.fit_width(raw, data, dict(scope="whole_paths"))


def test_whole_path_target_uses_complete_paths_and_100_percent_obeys_cap():
    raw, data = population([[1, 1.10], [1, 1.20], [1, 999]], mask=[[True, True], [True, True], [True, False]])
    result = cal.fit_width(raw, data, dict(scope="whole_paths", target_coverage=1.0))
    assert result["width"] == pytest.approx(.4) and result["whole_path_coverage"] == 1
    raw, data = population([[1, 4]])
    result = cal.fit_width(raw, data, dict(target_coverage=1.0))
    assert result["width"] == .45 and not result["target_met"]


@pytest.mark.parametrize("policy", [dict(min_width=.5), dict(max_width=.1), dict(max_width=float("nan")),
    dict(max_width=2), dict(target_coverage=0), dict(target_coverage=1.1), dict(origin_stride=0)])
def test_invalid_settings_fail(policy):
    with pytest.raises(ValueError):
        cal.settings(policy)


@pytest.mark.parametrize("part", ["train", "validation", "test"])
def test_held_out_role_shield(part):
    raw, data = population([[1]], split=part)
    with pytest.raises(ValueError, match="Only Calibration"):
        cal.fit_width(raw, data)


@pytest.fixture
def release(tmp_path, monkeypatch):
    return sensor_release.__wrapped__(tmp_path, monkeypatch)


def test_settings_ui_saves_reopens_and_rejects_invalid_bounds_without_writes(release):
    original_import(release)
    pid = release.project["project_id"]
    view = app(pid, "Data Quality")
    view.session_state["quality_tab"] = "Calibration Data"
    view.run()
    assert not view.exception and not view.error
    assert next(w for w in view.number_input if w.label == "Minimum full width (%)").value == 20.0
    assert next(w for w in view.number_input if w.label == "Maximum full width (%)").value == 45.0
    next(w for w in view.number_input if w.label == "Minimum full width (%)").set_value(25.0).run()
    next(w for w in view.button if w.label == "Save calibration settings").click().run()
    saved = cal.load_settings(pid, project_store().get(pid)["active_snapshot_id"])
    assert saved["min_width"] == .25
    view = app(pid, "Data Quality")
    view.session_state["quality_tab"] = "Calibration Data"
    view.run()
    assert next(w for w in view.number_input if w.label == "Minimum full width (%)").value == 25.0
    next(w for w in view.number_input if w.label == "Maximum full width (%)").set_value(10.0).run()
    assert view.error and next(w for w in view.button if w.label == "Save calibration settings").disabled
    assert cal.load_settings(pid, project_store().get(pid)["active_snapshot_id"]) == saved


def test_real_calibration_reads_only_calibration_and_changes_bounds_in_actual_replay(release, monkeypatch):
    from pdm import long_forecast_run as runs
    from pdm.long_forecast_data import read_part

    snapshot = original_import(release)
    pid = release.project["project_id"]
    manifest = runs.train(pid, "gru", overrides=dict(epochs=1, hidden_size=8, threads=1))
    rid = manifest["run_id"]
    model_dir = project_store().run_path(pid, rid)
    before = seal(snapshot["directory"]), seal(model_dir), project_store().get(pid)
    opened = []

    def only_calibration(source, role):
        opened.append(role)
        assert role == "calibration"
        return read_part(source, role)

    monkeypatch.setattr(cal, "read_part", only_calibration)
    result = cal.calibrate(pid, rid)
    assert opened == ["calibration"]
    assert cal.load_active(pid, rid, manifest) == result
    assert seal(snapshot["directory"]) == before[0]
    after = seal(model_dir)
    assert {name: after[name] for name in before[1]} == before[1]
    assert all(name.startswith("corridor_calibrations/") for name in set(after) - set(before[1]))
    assert project_store().get(pid) == before[2]
    source = runs.source_for(pid)
    uid = read_part(source, "test").unit_id.iloc[0]
    raw = runs.replay(pid, rid, uid, 59 * 60, use_calibration=False)
    corrected = runs.replay(pid, rid, uid, 59 * 60)
    np.testing.assert_array_equal(corrected["outputs"][:, 1], raw["outputs"][:, 1])
    widths = np.diff(corrected["outputs"][:, [0, 2]], axis=1).ravel() / corrected["outputs"][:, 1]
    assert np.all(widths >= .20 - 1e-12) and np.all(widths <= .45 + 1e-12)
    assert corrected["calibration"]["calibration_id"] == result["calibration_id"]
    incompatible = {**manifest, "model_hash": "changed"}
    with pytest.raises(ValueError, match="binding"):
        cal.load_active(pid, rid, incompatible)
    payload = model_dir / "corridor_calibrations" / (result["calibration_id"] + ".json")
    corrupt = read_json(payload)
    corrupt["width"] = .99
    atomic_write_json(payload, corrupt)
    with pytest.raises(ValueError, match="integrity"):
        runs.replay(pid, rid, uid, 59 * 60)


def test_failed_calibration_does_not_replace_active_result(release, monkeypatch):
    from pdm import long_forecast_run as runs

    original_import(release)
    pid = release.project["project_id"]
    manifest = runs.train(pid, "gru", overrides=dict(epochs=1, hidden_size=8, threads=1))
    rid = manifest["run_id"]
    active = cal.calibrate(pid, rid)
    before = seal(project_store().run_path(pid, rid))
    reader = cal.read_part

    def overlapping(source, part):
        frame = reader(source, part)
        frame["physical_unit_id"] = read_json(project_store().run_path(pid, rid) / "origins.json")["train"]["physical"][0]
        return frame

    monkeypatch.setattr(cal, "read_part", overlapping)
    with pytest.raises(ValueError, match="overlap"):
        cal.calibrate(pid, rid)
    assert seal(project_store().run_path(pid, rid)) == before
    assert cal.load_active(pid, rid, manifest) == active


def test_calibration_apply_button_and_forecast_switch_use_the_saved_model(release):
    from pdm import long_forecast_run as runs

    original_import(release)
    pid = release.project["project_id"]
    manifest = runs.train(pid, "gru", overrides=dict(epochs=1, hidden_size=8, threads=1))
    view = app(pid, "Data Quality")
    view.session_state["quality_tab"] = "Calibration Data"
    view.run()
    assert next(w for w in view.selectbox if w.label == "Model to calibrate").value == manifest["run_id"]
    next(w for w in view.button if w.label == "Calibrate and apply to Forecast").click().run()
    assert not view.exception and not view.error
    active = cal.load_active(pid, manifest["run_id"], manifest)
    assert active is not None and .20 <= active["width"] <= .45
    assert next(w for w in view.metric if w.label == "Applied full width").value == f"{100 * active['width']:.1f}%"
    results = app(pid, "Results")
    assert not results.exception and not results.error
    switch = next(w for w in results.checkbox if w.label == "Use saved Calibration corridor")
    assert switch.value
    assert any("Calibration corridor · full width" in w.value for w in results.caption)
    switch.uncheck().run()
    assert not results.exception and not results.error
    assert not any("Calibration corridor · full width" in w.value for w in results.caption)


def test_input_arrays_and_raw_outputs_remain_unchanged():
    raw, windows = population([[1.05, 1.15]])
    before = copy.deepcopy((raw, windows))
    cal.apply_width(raw, cal.fit_width(raw, windows))
    np.testing.assert_array_equal(raw, before[0])
    for key in ("mask", "y", "physical"):
        np.testing.assert_array_equal(windows[key], before[1][key])
