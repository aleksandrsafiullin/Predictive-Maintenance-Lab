"""Bounded training/inference contracts; these checks do not establish model quality."""
from __future__ import annotations

import copy

import pytest
import torch
from streamlit.testing.v1 import AppTest

from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.project_results_ui import replay_figure
from pdm.signal_inference import forecast_prefix
from pdm.signal_training import load_signal_run, train_signal_run
from pdm.trend_corridor import (
    CONTRACT,
    LEGACY_CONTRACT,
    MODE,
    CorridorNet,
    boundary_red_entry,
    corridor_params,
    interval_loss,
)
from tests.project_contract import make_contract_snapshot


@pytest.fixture
def contract(tmp_path, monkeypatch):
    root = tmp_path/"projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    return make_contract_snapshot(root)


def config(engine="gru"):
    return corridor_params(engine, dict(forecast_mode=MODE, history_length=4,
                           hidden_size=8, epochs=2, batch_size=16, max_iter=3,
                           horizons_s=[10., 20., 30.], max_windows_per_unit=32), None)


@pytest.mark.parametrize("nonnegative", [True, False])
def test_two_learned_boundaries_always_obey_width_limit_and_receive_gradients(nonnegative):
    model = CorridorNet("gru", config())
    x = torch.randn(5, 4, 1)
    current = torch.tensor([0., .1, 1., 100., 1000.] if nonnegative else [-100., -1., 0., 1., 100.])
    lo, hi = model(x, current, 1., nonnegative)
    level = (hi+lo)/2
    relative = (hi-lo)/level.abs().clamp_min(1e-12)
    active = level != 0
    assert torch.all(relative[active] >= .2-1e-6)
    assert torch.all(relative[active] <= .3+1e-6)
    if nonnegative:
        assert (lo > 0).all()
    target = level.detach()+.3
    objective = interval_loss(lo, hi, target, torch.ones_like(lo, dtype=torch.bool),
                              torch.ones_like(lo)/lo.numel(), 1., .9, nonnegative)
    objective.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.head.weight.grad.abs().sum() > 0


def test_loss_scores_misses_and_does_not_treat_missing_targets_as_zeros():
    lo, hi = torch.tensor([[.9, .9]]), torch.tensor([[1.1, 1.1]])
    mask = torch.tensor([[True, False]])
    weights = torch.ones_like(lo)
    def score(target):
        return interval_loss(lo, hi, torch.tensor([target]), mask, weights, 1., .9, True)
    assert score([1., 0.]) == score([1., float("nan")])
    assert score([2., float("nan")]) > score([1., 0.])


@pytest.mark.parametrize("engine", ["gru", "lstm", "quantile_boosting", "full_cns"])
def test_real_save_reload_inference_and_future_suffix_invariance(contract, engine, request, monkeypatch):
    if engine == "full_cns":
        request.getfixturevalue("signal_cns_fixture")
    _, project, snapshot = contract
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    manifest = train_signal_run(pid, sid, engine, config(engine))
    assert manifest["corridor_contract"] == CONTRACT
    assert manifest["reload_verified"] and manifest["quality_accepted"] is False
    assert manifest["selection"]["test_feedback"] is False
    assert manifest["metrics"]["validation"]["maximum_relative_width"] <= .300001
    load_signal_run(pid, manifest["run_id"])
    data = load_snapshot(pid, sid)
    uid = str(data["split"]["test"][0])
    result = forecast_prefix(pid, manifest["run_id"], uid, 130., prediction_horizon_s=20.)
    assert result["status"] == "available"
    assert len(result["points"]) == 2
    assert all(p["target_time_s"] <= 150. for p in result["points"])
    full = forecast_prefix(pid, manifest["run_id"], uid, 130.)
    assert full["points"][:2] == result["points"]
    assert result["red_entry_corridor"]["source"] == "trend_corridor_boundaries"
    changed = copy.deepcopy(data)
    hidden = (changed["features"].unit_id == uid) & (changed["features"].timestamp_s > 130.)
    changed["features"].loc[hidden, "signal"] = 100000.
    import pdm.signal_inference as inference
    monkeypatch.setattr(inference, "load_snapshot", lambda *_a, **_k: changed)
    assert forecast_prefix(pid, manifest["run_id"], uid, 130., prediction_horizon_s=20.) == result
    short = forecast_prefix(pid, manifest["run_id"], uid, 10.)
    assert short["status"] == "unavailable" and not short["points"]


def test_red_window_is_derived_only_from_boundary_crossings_and_can_remain_open():
    rule = dict(status="available", red=3., direction="above")
    points = [dict(target_time_s=10., lower=2.5, upper=2.8),
              dict(target_time_s=20., lower=2.9, upper=3.2),
              dict(target_time_s=30., lower=3.1, upper=3.4)]
    closed = boundary_red_entry(points, 2., rule, 0.)
    assert (closed["status"], closed["earliest_s"], closed["latest_s"]) == ("derived", 10., 30.)
    opened = boundary_red_entry(points[:2], 2., rule, 0.)
    assert opened["status"] == "open" and opened["latest_s"] is None
    assert boundary_red_entry(points[:1], 2., rule, 0.)["status"] == "none_within_horizon"
    assert boundary_red_entry(points, 3., rule, 0.)["status"] == "already_red"
    assert boundary_red_entry(points, 2., rule, 0., previously_red=True)["status"] == "previously_red"
    falling = [{**p, "lower": 6-p["upper"], "upper": 6-p["lower"]} for p in points]
    down = boundary_red_entry(falling, 4., {**rule, "direction": "below"}, 0.)
    assert (down["earliest_s"], down["latest_s"]) == (10., 30.)


def test_width_contract_cannot_be_changed_by_rehashing_saved_artifacts(contract):
    store, project, snapshot = contract
    manifest = train_signal_run(project["project_id"], snapshot["snapshot_id"], "gru", config())
    directory = store.run_path(project["project_id"], manifest["run_id"])
    saved = read_json(directory/"training_contract.json")
    saved["corridor_contract"]["maximum_relative_width"] = .9
    manifest["corridor_contract"] = saved["corridor_contract"]
    atomic_write_json(directory/"training_contract.json", saved)
    manifest["artifacts"]["training_contract.json"] = sha256_file(directory/"training_contract.json")
    atomic_write_json(directory/"manifest.json", manifest)
    with pytest.raises(ValueError, match="corridor contract"):
        load_signal_run(project["project_id"], manifest["run_id"])


def test_saved_v1_run_retains_its_original_width_and_replay_metadata(contract):
    store, project, snapshot = contract
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    manifest = train_signal_run(pid, sid, "gru", config())
    uid = str(load_snapshot(pid, sid)["split"]["test"][0])
    widened = forecast_prefix(pid, manifest["run_id"], uid, 130.)
    directory = store.run_path(pid, manifest["run_id"])
    # V1 and V2 share checkpoint structure; decoding must use the saved contract.
    for name in ("corridor_model.json", "training_contract.json"):
        saved = read_json(directory/name)
        saved["corridor_contract"] = LEGACY_CONTRACT
        atomic_write_json(directory/name, saved)
        manifest["artifacts"][name] = sha256_file(directory/name)
    manifest["corridor_contract"] = LEGACY_CONTRACT
    manifest["schema_version"] = "project_trend_corridor_v1"
    atomic_write_json(directory/"manifest.json", manifest)
    load_signal_run(pid, manifest["run_id"])
    original = forecast_prefix(pid, manifest["run_id"], uid, 130.)
    assert original["corridor_contract"] == LEGACY_CONTRACT
    assert original["funnel"]["target_relative_width"] == .10
    assert original["funnel"]["maximum_relative_width"] == .15
    for old, new in zip(original["points"], widened["points"]):
        assert old["value"] == pytest.approx(new["value"], abs=1e-6)
        assert old["upper"]-old["lower"] == pytest.approx((new["upper"]-new["lower"])/2, abs=1e-6)
    assert forecast_prefix(pid, manifest["run_id"], uid, 10.)["corridor_contract"] == LEGACY_CONTRACT


@pytest.mark.parametrize("logit,half_width", [(-100., .10), (100., .15)])
def test_width_endpoints_are_plus_minus_ten_and_fifteen_percent(logit, half_width):
    model = CorridorNet("gru", config())
    with torch.no_grad():
        model.head.bias[len(config()["horizons_s"]):] = logit
    lower, upper = model(torch.zeros(1, 4, 1), torch.tensor([100.]), 1., True)
    assert torch.allclose(lower, torch.full_like(lower, 100*(1-half_width)))
    assert torch.allclose(upper, torch.full_like(upper, 100*(1+half_width)))


def test_chart_draws_the_model_boundaries_without_a_center_prediction():
    result = dict(as_of_s=0., status="available", funnel=dict(mode=MODE),
                  observed_prefix=[dict(timestamp_s=0., signal=1.)],
                  points=[dict(target_time_s=10., value=2., lower=1.9, upper=2.1, kind="direct")],
                  thresholds=dict(status="available", red=3., yellow=2.5, direction="above"),
                  red_entry_corridor=dict(status="none_within_horizon"))
    fig = replay_figure(result, dict(signal_unit="g", signal_label="Signal"))
    traces = {trace.name: trace for trace in fig.data}
    assert "Forecast horizon" not in traces
    assert list(traces["Trend corridor upper"].y) == [1., 2.1]
    assert list(traces["Trend corridor"].y) == [1., 1.9]
    assert traces["Trend corridor"].fill == "tonexty"


def test_training_ui_has_one_corridor_task_and_submits_it(contract, monkeypatch):
    _, project, snapshot = contract
    import pdm.project_training_ui as ui
    monkeypatch.setattr(ui, "worker_alive", lambda: False)
    monkeypatch.setattr(ui, "read_status", lambda: {})
    monkeypatch.setattr(ui, "status_for_project", lambda _: {})
    jobs = []
    monkeypatch.setattr(ui, "spawn_worker", lambda job: jobs.append(job))
    pid, sid = project["project_id"], snapshot["snapshot_id"]
    at = AppTest.from_string(f"from pdm.data.project_prepare import load_snapshot\n"
                             f"from pdm.project_training_ui import render_training\n"
                             f"render_training({pid!r}, load_snapshot({pid!r}, {sid!r}))").run()
    assert not at.exception
    assert all(w.label != "Training task" for w in at.selectbox)
    assert any("Training task · Trend corridor" in c.value for c in at.caption)
    assert any("±10%" in c.value and "±15%" in c.value for c in at.caption)
    next(b for b in at.button if b.label == "Train model").click().run()
    assert not at.exception
    assert len(jobs) == 1 and jobs[0]["params"]["forecast_mode"] == MODE
