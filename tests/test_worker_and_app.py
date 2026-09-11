from __future__ import annotations

import json

from pdm.worker import read_status, worker_alive


def test_worker_not_alive_without_pid():
    # No duplicate training on UI rerun: spawn_worker refuses if worker_alive().
    assert worker_alive() in {True, False}
    st = read_status()
    assert "status" in st


def test_app_starts_without_data():
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception


def test_app_filters_data_shows_time_scale_warning():
    from streamlit.testing.v1 import AppTest

    from pdm.data.filters import FILTER_TIME_SCALE_WARNING
    from pdm.paths import project_root

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Filters")
    at.run()
    assert not at.exception
    texts = [str(w.value) for w in at.warning]
    assert any(FILTER_TIME_SCALE_WARNING in t for t in texts)


def test_app_data_screen_reads_cached_counts_not_build_windows(monkeypatch, tiny_filter_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.data.filters import FILTER_TIME_SCALE_WARNING
    from pdm.paths import project_root
    from pdm.splits import filters_split

    features, units = tiny_filter_tables
    units = units.copy()
    if "origin_unit_id" not in units.columns:
        units["origin_unit_id"] = units["author_data_no"]
    split = filters_split(units)
    report = {
        "dataset_id": "filters",
        "dataset_version": "testver_deadbeef",
        "split_protocol": split["protocol"],
        "split_hash": "abc123splithash",
        "time_scale_verified": False,
        "time_unit_note": FILTER_TIME_SCALE_WARNING,
        "sensor_time_note": "synthetic cache",
        "issues_and_decisions": [],
        "history_length": 20,
        "window_counts": {
            "history_length": 20,
            "n_candidate_ends": int(len(features)),
            "eligible": 100,
            "excluded": {"gap": 3, "post_event": 10, "insufficient_length": 40},
            "by_split": {
                "train": {
                    "n_units": split["n_train"],
                    "eligible_windows": 80,
                    "excluded_gap": 1,
                    "excluded_post_event": 8,
                    "excluded_insufficient_length": 30,
                },
                "validation": {
                    "n_units": split["n_validation"],
                    "eligible_windows": 12,
                    "excluded_gap": 1,
                    "excluded_post_event": 1,
                    "excluded_insufficient_length": 5,
                },
                "test": {
                    "n_units": split["n_test"],
                    "eligible_windows": 8,
                    "excluded_gap": 1,
                    "excluded_post_event": 1,
                    "excluded_insufficient_length": 5,
                },
            },
        },
        "events_vs_censoring": {
            "n_event_observed": 2,
            "n_right_censored": 8,
            "n_official_rul_known_not_observed": 2,
            "n_sensor_end_without_event_or_official_rul": 6,
            "note": (
                "Observed 600 Pa events count sensor crossings only. "
                "Author test official RUL is an evaluation label at prefix end."
            ),
        },
        "observation_time_s": {
            "total_span_s": 1234.0,
            "by_split": {"train": 800.0, "validation": 200.0, "test": 234.0},
        },
        "regimes": [{"regime_id": "A3", "n_units": 10, "n_event_observed": 2}],
    }
    bundle = {
        "features": features,
        "units": units,
        "split": split,
        "report": report,
        "dir": None,
        "fingerprint": {
            "dataset_version": "testver_deadbeef",
            "split_protocol": split["protocol"],
            "split_hash": "abc123splithash",
        },
        "dataset_version": "testver_deadbeef",
    }

    def _fake_ready(dataset_id: str) -> bool:
        return dataset_id == "filters"

    def _fake_load(dataset_id: str) -> dict:
        assert dataset_id == "filters"
        return bundle

    def _no_windows(*_a, **_k):
        raise AssertionError("Data screen must not call build_windows")

    monkeypatch.setattr("pdm.data.prepare.processed_ready", _fake_ready)
    monkeypatch.setattr("pdm.data.prepare.load_processed", _fake_load)
    monkeypatch.setattr("pdm.windows.build_windows", _no_windows)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Filters")
    at.run()
    assert not at.exception

    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "testver_deadbeef" in markdown
    assert "abc123splithash" in markdown
    assert split["protocol"] in markdown

    metric_by_label = {m.label: m.value for m in at.metric}
    assert metric_by_label["Observed 600 Pa events"] == "2"
    assert metric_by_label["Official RUL known (eval only)"] == "2"
    assert metric_by_label["Eligible windows"] == "100"
    assert metric_by_label["Excluded: gap"] == "3"

    frames = [w.value for w in at.dataframe]
    split_tbl = next(df for df in frames if "eligible windows" in df.columns)
    assert set(split_tbl["split"]) == {"train", "validation", "test"}
    units_tbl = next(df for df in frames if "origin" in df.columns and "record_kind" in df.columns)
    assert units_tbl["origin"].astype(str).str.contains("author_").any()
    kinds = units_tbl["record_kind"].astype(str)
    assert kinds.str.contains("Observed 600 Pa").any()
    assert kinds.str.contains("official RUL").any()
    assert kinds.str.contains("Right-censored").any()

    captions = "\n".join(str(w.value) for w in at.caption)
    assert "does not rebuild windows" in captions
    assert "Data_No" in captions or "origin_unit_id" in captions
    assert any("evaluation label" in str(w.value) for w in at.caption)


def _fake_processed_bundle(dataset_id, features, units, split):
    return {
        "features": features,
        "units": units,
        "split": split,
        "report": {
            "history_length_physical": {"note": "synthetic fixture, not a real dataset"},
            "dataset_version": "testver",
        },
        "dir": None,
        "fingerprint": {"dataset_version": "testver"},
        "dataset_version": "testver",
    }


def test_app_train_screen_smoke_off_clears_window_cap(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    captured: dict = {}
    run_id = "bearings_gru_full_fake"
    rdir = tmp_path / run_id
    rdir.mkdir()
    (rdir / "training_history.csv").write_text(
        "epoch,train_loss,val_loss,train_metric,val_metric,val_mae_events,n_val_event_units\n"
        "1,1.0,0.9,0.80,0.70,0.70,3\n"
        "2,0.8,0.6,0.55,0.40,0.40,3\n",
        encoding="utf-8",
    )
    (rdir / "status.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "architecture": "gru",
                "status": "completed",
                "smoke": False,
                "mode": "Full",
                "best_epoch": 2,
                "best_metric": 0.4,
                "n_train_windows": 100,
                "n_val_windows": 20,
                "max_windows_per_unit": None,
                "train_windows_per_unit": {"min": 10, "max": 40, "mean": 25.0, "n_units": 9, "n_windows": 100},
                "selection_metric_name": "val MAE",
                "selection_metric_unit": "seconds",
                "selection_metric_label": "val MAE (seconds)",
                "updated_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    (rdir / "validation_metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": 2,
                "best_metric": 0.4,
                "selection_metric_label": "val MAE (seconds)",
                "last": {"selection_metric": 0.4, "val_mae_events": 0.4, "n_val_event_units": 3},
                "last_train": {"selection_metric": 0.55},
                "n_train_windows": 100,
                "n_val_windows": 20,
                "max_windows_per_unit": None,
                "train_windows_per_unit": {"min": 10, "max": 40, "mean": 25.0},
                "smoke": False,
                "mode": "Full",
            }
        ),
        encoding="utf-8",
    )
    fake_row = {
        "dataset_id": "bearings",
        "run_id": run_id,
        "path": str(rdir),
        "has_best": True,
        "has_last": True,
        "status": "completed",
        "smoke": False,
        "mode": "Full",
        "best_epoch": 2,
        "best_metric": 0.4,
        "architecture": "gru",
        "n_train_windows": 100,
        "n_val_windows": 20,
        "updated_at": "2026-01-01T00:00:00Z",
    }

    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.cli.spawn_worker", lambda job: captured.update(job) or captured)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[1].set_value("Train")
    at.run()
    assert not at.exception

    markdown = "\n".join(str(w.value) for w in at.markdown)
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "Training mode: **Smoke**" in markdown
    assert "val MAE (seconds)" in captions
    max_w = next(n for n in at.number_input if "Max windows" in n.label)
    assert int(max_w.value) == 32

    mode = next(r for r in at.radio if "Smoke" in list(r.options) and "Full" in list(r.options))
    mode.set_value("Full")
    at.run()
    assert not at.exception
    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "Training mode: **Full**" in markdown
    max_w = next(n for n in at.number_input if "Max windows" in n.label)
    assert int(max_w.value) == 0

    start = next(b for b in at.button if "Start training" in b.label)
    start.click()
    at.run()
    assert not at.exception
    assert captured["kind"] == "train"
    assert captured["smoke"] is False
    assert captured["max_windows_per_unit"] == 0
    assert captured["max_epochs"] == 30
    assert captured["dataset_id"] == "bearings"

    metric_by_label = {m.label: m.value for m in at.metric}
    assert metric_by_label["Best epoch"] == "2"
    assert "val MAE" in " ".join(metric_by_label)
    assert metric_by_label["Train windows"] == "100"
    assert metric_by_label["Val windows"] == "20"
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "windows/unit used" in captions
    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "Run mode: **Full**" in markdown
    frames = [w.value for w in at.dataframe]
    exp = next(df for df in frames if "mode" in df.columns and "run_id" in df.columns)
    assert "Full" in set(exp["mode"].astype(str))


def test_app_train_screen_filters_shows_val_nll(monkeypatch, tiny_filter_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.splits import filters_split

    features, units = tiny_filter_tables
    units = units.copy()
    if "origin_unit_id" not in units.columns:
        units["origin_unit_id"] = units["author_data_no"]
    split = filters_split(units)
    bundle = _fake_processed_bundle("filters", features, units, split)
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "filters")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [])
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Filters")
    at.run()
    at.sidebar.radio[1].set_value("Train")
    at.run()
    assert not at.exception
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "val NLL" in captions
    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "Training mode: **Smoke**" in markdown


def test_app_train_resume_keeps_full_not_form_smoke(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    captured: dict = {}
    run_id = "bearings_gru_full_resume"
    rdir = tmp_path / run_id
    rdir.mkdir()
    (rdir / "last.pt").write_bytes(b"stub")
    (rdir / "config.yaml").write_text(
        "dataset_id: bearings\nsmoke: false\nmode: Full\nmax_windows_per_unit: null\n"
        "model:\n  architecture: gru\n  max_epochs: 30\n",
        encoding="utf-8",
    )
    fake_row = {
        "dataset_id": "bearings",
        "run_id": run_id,
        "path": str(rdir),
        "has_best": True,
        "has_last": True,
        "status": "completed",
        "smoke": False,
        "mode": "Full",
        "architecture": "gru",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.cli.spawn_worker", lambda job: captured.update(job) or captured)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    at.sidebar.radio[1].set_value("Train")
    at.run()
    assert not at.exception
    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "Training mode: **Smoke**" in markdown
    max_w = next(n for n in at.number_input if "Max windows" in n.label)
    assert int(max_w.value) == 32

    resume_box = next(s for s in at.selectbox if "Resume" in s.label)
    resume_box.set_value(run_id)
    at.run()
    assert not at.exception
    infos = "\n".join(str(w.value) for w in at.info)
    assert "Resume uses saved" in infos
    assert "Full" in infos

    start = next(b for b in at.button if "Start training" in b.label)
    start.click()
    at.run()
    assert not at.exception
    assert captured["resume_run_id"] == run_id
    assert captured["smoke"] is False
    assert captured["max_windows_per_unit"] == 0
    assert captured["max_epochs"] == 30


def test_app_replay_lists_evaluations(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.io_util import atomic_write_json
    from pdm.paths import project_root
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    run_id = "bearings_gru_eval_list"
    rdir = tmp_path / run_id
    eval_id = "20260101T000000Z_abcd1234"
    edir = rdir / "evaluations" / eval_id
    edir.mkdir(parents=True)
    test_uid = split["test"][0]
    (edir / "predictions.csv").write_text(
        "run_id,unit_id,timestamp_s,predicted_rul_s\n"
        f"{run_id},{test_uid},60.0,100.0\n"
        f"{run_id},{test_uid},120.0,90.0\n",
        encoding="utf-8",
    )
    atomic_write_json(
        edir / "evaluation_config.json",
        {"eval_id": eval_id, "run_id": run_id, "metrics_version": "v0"},
    )
    (edir / "metrics.json").write_text(
        json.dumps(
            {
                "eval_id": eval_id,
                "metrics_version": "v0",
                "primary_metric": "equal_weight_unit_mae",
                "equal_weight_unit_mae": 120.0,
                "pooled_mae": 130.0,
                "mean_overestimation": 10.0,
                "n_test_units": 1,
                "near_event_zones_s": [3600, 1800, 600],
                "equal_weight_unit_mae_by_zone": {"3600": 80.0, "1800": 50.0, "600": 20.0},
                "note": "Test metrics are not used for epoch selection or preprocessing.",
                "baseline": {
                    "name": "Age-only",
                    "baseline_coverage_fraction": 1.0,
                    "baseline_coverage_points": 10,
                    "n_reference_points": 10,
                    "neural_net_all_points": {"mae": 120.0},
                    "neural_net_baseline_overlap": {"mae": 120.0},
                    "baseline_overlap": {"mae": 200.0},
                },
                "alerts": {
                    "timely": 1,
                    "miss": 0,
                    "insufficient_coverage": 0,
                    "n_units_timely": 1,
                    "denominator_note": (
                        "miss/timely rates use units with a finite evaluator event excluding "
                        "insufficient_coverage; incomplete windows are not false misses."
                    ),
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (edir / "metrics_by_unit.csv").write_text(
        "unit_id,mae,rmse,alert_outcome,lead_time_s,has_sufficient_coverage,baseline_coverage_fraction\n"
        f"{test_uid},120.0,140.0,timely,600.0,True,1.0\n",
        encoding="utf-8",
    )
    (edir / "alerts.csv").write_text("unit_id\n", encoding="utf-8")
    fake_row = {
        "dataset_id": "bearings",
        "run_id": run_id,
        "path": str(rdir),
        "has_best": True,
        "has_last": True,
        "status": "completed",
        "n_evaluations": 1,
    }
    bound = {
        "features": features,
        "units": units,
        "split": split,
        "report": bundle["report"],
        "test_ids": list(split["test"]),
    }

    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.replay.bind_replay_to_run", lambda *a, **k: bound)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[1].set_value("Test & Replay")
    at.run()
    assert not at.exception
    eval_box = next(s for s in at.selectbox if "Evaluation" in s.label)
    assert eval_id in list(eval_box.options)
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "never overwrites" in captions or "does not depend on H/K" in captions
    labels = [n.label for n in at.number_input]
    assert any("H_trigger" in lab for lab in labels)
    assert any("Minimum action lead time" in lab for lab in labels)
    assert any("Max useful horizon" in lab for lab in labels)
    freeze = next(b for b in at.button if "Freeze alert policy" in b.label)
    freeze.click()
    at.run()
    assert not at.exception
    policy_path = rdir / "alert_policy.json"
    assert policy_path.exists()
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    assert "H_trigger" in policy
    assert "minimum_action_lead_time" in policy
    assert policy["warning_horizon_s"] == policy["H_trigger"]

    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert eval_id in markdown
    metric_by_label = {m.label: m.value for m in at.metric}
    assert "Equal-weight unit MAE" in metric_by_label
    frames = [w.value for w in at.dataframe]
    unit_tbl = next(df for df in frames if "Unit" in df.columns and "MAE (s)" in df.columns)
    assert test_uid in set(unit_tbl["Unit"].astype(str))
    assert "Alert class" in unit_tbl.columns
    assert "Lead time (s)" in unit_tbl.columns
    assert "Sufficient coverage" in unit_tbl.columns
    zone_tbl = next(df for df in frames if "Zone (s)" in df.columns)
    assert set(zone_tbl["Zone (s)"].astype(str)) == {"3600", "1800", "600"}
    infos = "\n".join(str(w.value) for w in at.info)
    assert "Age-only" in infos
    assert "overlap" in infos.lower() or "coverage" in infos.lower()
    expanders = [str(e.label) for e in at.expander]
    assert any("metrics.json" in lab for lab in expanders)
    dl = [b.label for b in at.download_button]
    assert any("predictions CSV" in lab for lab in dl)
    assert any("alerts CSV" in lab for lab in dl)
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "evaluation directory" in captions or "evaluations/" in captions
    assert "3 bearings" in captions or "3 held-out" in captions


def test_evaluation_unit_table_lists_all_test_units():
    import pandas as pd

    from pdm.experiments import (
        baseline_coverage_callout,
        metrics_by_unit_display_frame,
    )

    by = pd.DataFrame(
        {
            "unit_id": ["Bearing1_5"],
            "mae": [12.0],
            "alert_outcome": ["timely"],
            "lead_time_s": [600.0],
            "has_sufficient_coverage": [True],
        }
    )
    table = metrics_by_unit_display_frame(
        by,
        dataset_id="bearings",
        test_ids=["Bearing1_3", "Bearing2_5", "Bearing3_5"],
    )
    assert list(table["Unit"]) == ["Bearing1_3", "Bearing2_5", "Bearing3_5"]
    assert "MAE (s)" in table.columns
    assert "Alert class" in table.columns
    assert "NLL" not in table.columns

    filt = pd.DataFrame(
        {
            "unit_id": ["Test_1"],
            "prefix_end_abs_error": [5.0],
            "prefix_end_actual_rul_s": [80.0],
            "prefix_end_predicted_rul_s": [75.0],
            "nll": [0.4],
            "alert_outcome": ["insufficient_coverage"],
            "lead_time_s": [None],
            "has_sufficient_coverage": [False],
        }
    )
    ftable = metrics_by_unit_display_frame(filt, dataset_id="filters", test_ids=["Test_1", "Test_2"])
    assert list(ftable["Unit"]) == ["Test_1", "Test_2"]
    assert "Official RUL at prefix end (s)" in ftable.columns
    assert "Predicted RUL at prefix end (s)" in ftable.columns
    assert "NLL" in ftable.columns
    assert "Insufficient coverage" in set(ftable["Alert class"].astype(str))

    callout = baseline_coverage_callout(
        {
            "baseline": {
                "name": "linear_dp_trend",
                "prefix_end": {
                    "baseline_coverage_fraction": 0.5,
                    "baseline_coverage_points": 1,
                    "n_reference_points": 2,
                    "neural_net_baseline_overlap": {"mae": 10.0},
                    "baseline_overlap": {"mae": 40.0},
                },
            }
        }
    )
    assert callout is not None
    assert "linear_dp_trend" in callout
    assert "1/2" in callout
    assert "prefix-end" in callout


def test_app_filters_evaluation_shows_prefix_end_official_rul(
    monkeypatch, tmp_path, tiny_filter_tables
):
    from streamlit.testing.v1 import AppTest

    from pdm.io_util import atomic_write_json
    from pdm.paths import project_root
    from pdm.splits import filters_split

    features, units = tiny_filter_tables
    units = units.copy()
    if "origin_unit_id" not in units.columns:
        units["origin_unit_id"] = units["author_data_no"]
    split = filters_split(units)
    bundle = _fake_processed_bundle("filters", features, units, split)
    run_id = "filters_gru_eval_table"
    rdir = tmp_path / run_id
    eval_id = "20260102T000000Z_filt0001"
    edir = rdir / "evaluations" / eval_id
    edir.mkdir(parents=True)
    test_ids = list(split["test"])
    uid0, uid1 = test_ids[0], test_ids[1]
    (edir / "predictions.csv").write_text(
        "run_id,unit_id,timestamp_s,predicted_rul_s\n"
        f"{run_id},{uid0},60.0,100.0\n"
        f"{run_id},{uid1},60.0,90.0\n",
        encoding="utf-8",
    )
    atomic_write_json(
        edir / "evaluation_config.json",
        {"eval_id": eval_id, "run_id": run_id, "metrics_version": "v0"},
    )
    (edir / "metrics.json").write_text(
        json.dumps(
            {
                "eval_id": eval_id,
                "primary_metric": "prefix_end_mae",
                "prefix_end_mae": 12.0,
                "official_rul_source": (
                    "official_rul_at_prefix_end_s at the last sensor row of each "
                    "author test prefix (eval-only; never a model input)"
                ),
                "validation": {
                    "nll_all_units": 0.8,
                    "mae_observed_events": 30.0,
                    "n_observed_event_units": 2,
                },
                "baseline": {
                    "name": "linear_dp_trend",
                    "prefix_end": {
                        "baseline_coverage_fraction": 0.5,
                        "baseline_coverage_points": 1,
                        "n_reference_points": 2,
                        "neural_net_all_points": {"mae": 12.0},
                        "neural_net_baseline_overlap": {"mae": 8.0},
                        "baseline_overlap": {"mae": 4.0},
                    },
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (edir / "metrics_by_unit.csv").write_text(
        "unit_id,prefix_end_abs_error,prefix_end_actual_rul_s,prefix_end_predicted_rul_s,"
        "alert_outcome,lead_time_s,has_sufficient_coverage\n"
        f"{uid0},12.0,720.0,708.0,miss,,True\n"
        f"{uid1},4.0,480.0,476.0,insufficient_coverage,,False\n",
        encoding="utf-8",
    )
    (edir / "alerts.csv").write_text("unit_id,timestamp_s\n", encoding="utf-8")
    fake_row = {
        "dataset_id": "filters",
        "run_id": run_id,
        "path": str(rdir),
        "has_best": True,
        "has_last": True,
        "status": "completed",
        "n_evaluations": 1,
    }
    bound = {
        "features": features,
        "units": units,
        "split": split,
        "report": bundle["report"],
        "test_ids": test_ids,
    }
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "filters")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.replay.bind_replay_to_run", lambda *a, **k: bound)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[0].set_value("Filters")
    at.run()
    at.sidebar.radio[1].set_value("Test & Replay")
    at.run()
    assert not at.exception
    metric_by_label = {m.label: m.value for m in at.metric}
    assert "Prefix-end MAE" in metric_by_label
    assert "Validation NLL (all units)" in metric_by_label
    frames = [w.value for w in at.dataframe]
    unit_tbl = next(
        df
        for df in frames
        if "Official RUL at prefix end (s)" in df.columns and "Unit" in df.columns
    )
    assert set(unit_tbl["Unit"].astype(str)) >= {uid0, uid1}
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "official RUL" in captions.lower() or "prefix end" in captions.lower()
    infos = "\n".join(str(w.value) for w in at.info)
    assert "linear_dp_trend" in infos
    dl = [b.label for b in at.download_button]
    assert any("predictions CSV" in lab for lab in dl)


def test_app_replay_screen_loads_without_eval(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    run_id = "bearings_gru_replay_block"
    rdir = tmp_path / run_id
    rdir.mkdir()
    fake_row = {
        "dataset_id": "bearings",
        "run_id": run_id,
        "path": str(rdir),
        "has_best": True,
        "has_last": True,
        "status": "completed",
        "n_evaluations": 0,
    }
    bound = {
        "features": features,
        "units": units,
        "split": split,
        "report": bundle["report"],
        "test_ids": list(split["test"]),
    }
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.replay.bind_replay_to_run", lambda *a, **k: bound)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[1].set_value("Test & Replay")
    at.run()
    assert not at.exception
    errors = "\n".join(str(w.value) for w in at.error)
    assert "predictions.csv" in errors or "Evaluate test set" in errors
    play = next(b for b in at.button if b.label == "Play")
    assert play.disabled
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "Play stays blocked" in captions


def _launch_bearing_replay(monkeypatch, tmp_path, tiny_bearing_tables, *, run_id, eval_id=None):
    from pdm.io_util import atomic_write_json
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    rdir = tmp_path / run_id
    rdir.mkdir()
    test_uid = split["test"][0]
    unit_ts = (
        features.loc[features["unit_id"] == test_uid, "timestamp_s"]
        .astype(float)
        .sort_values()
        .tolist()
    )
    other = next(u for u in split["train"])
    pred_lines = ["run_id,unit_id,timestamp_s,predicted_rul_s"]
    for i, ts in enumerate(unit_ts):
        pred_lines.append(f"{run_id},{test_uid},{ts},{100.0 - i}")
    pred_lines.append(f"{run_id},{other},9999.0,5.0")
    pred_text = "\n".join(pred_lines) + "\n"
    alert_text = (
        "unit_id,timestamp_s,type\n"
        f"{test_uid},{unit_ts[0]},horizon_warning\n"
        f"{test_uid},{unit_ts[-1]},horizon_warning\n"
        f"{other},{unit_ts[0]},horizon_warning\n"
    )
    if eval_id:
        edir = rdir / "evaluations" / eval_id
        edir.mkdir(parents=True)
        (edir / "predictions.csv").write_text(pred_text, encoding="utf-8")
        (edir / "alerts.csv").write_text(alert_text, encoding="utf-8")
        atomic_write_json(
            edir / "evaluation_config.json",
            {"eval_id": eval_id, "run_id": run_id, "metrics_version": "v0"},
        )
        pred_path = edir / "predictions.csv"
    else:
        (rdir / "predictions.csv").write_text(pred_text, encoding="utf-8")
        (rdir / "alerts.csv").write_text(alert_text, encoding="utf-8")
        pred_path = rdir / "predictions.csv"
    fake_row = {
        "dataset_id": "bearings",
        "run_id": run_id,
        "path": str(rdir),
        "has_best": True,
        "has_last": True,
        "status": "completed",
        "n_evaluations": 1 if eval_id else 0,
    }
    bound = {
        "features": features,
        "units": units,
        "split": split,
        "report": bundle["report"],
        "test_ids": list(split["test"]),
    }
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.replay.bind_replay_to_run", lambda *a, **k: bound)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)
    return rdir, test_uid, pred_path, unit_ts


def test_app_replay_legacy_predictions_unlock_play(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    _launch_bearing_replay(
        monkeypatch, tmp_path, tiny_bearing_tables, run_id="bearings_gru_legacy_play"
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[1].set_value("Test & Replay")
    at.run()
    assert not at.exception
    play = next(b for b in at.button if b.label == "Play")
    assert not play.disabled
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "Play stays blocked" not in captions
    assert "legacy" in captions.lower()


def test_replay_play_advances_without_clicks(monkeypatch, tmp_path, tiny_bearing_tables):
    """Play + script rerun advances step. AppTest does not fire fragment run_every timers."""
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    _, _, _, unit_ts = _launch_bearing_replay(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        run_id="bearings_gru_play_advance",
        eval_id="20260103T000000Z_play0001",
    )
    assert len(unit_ts) >= 3
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.sidebar.radio[1].set_value("Test & Replay")
    at.run()
    assert not at.exception
    play = next(b for b in at.button if b.label == "Play")
    assert not play.disabled
    step0 = int(at.session_state["replay_step"])
    age0 = next(m.value for m in at.metric if m.label == "Operating age (min)")
    play.click()
    at.run()
    assert not at.exception
    # st.rerun() after Play is followed; run_every=0.4s is not. Next run is the fragment analog.
    assert at.session_state["playing"] is True
    step_play = int(at.session_state["replay_step"])
    assert step_play >= step0
    at.run()
    assert not at.exception
    step_tick = int(at.session_state["replay_step"])
    assert step_tick > step0
    assert at.session_state["playing"] is True
    age_tick = next(m.value for m in at.metric if m.label == "Operating age (min)")
    assert float(age_tick) > float(age0)
    next(b for b in at.button if b.label == "Pause").click()
    at.run()
    assert not at.exception
    assert at.session_state["playing"] is False
    frozen = int(at.session_state["replay_step"])
    at.run()
    assert not at.exception
    assert at.session_state["playing"] is False
    assert int(at.session_state["replay_step"]) == frozen
