from __future__ import annotations

import json
import os
import types

import pytest

from pdm.evaluate import METRICS_VERSION
from pdm.worker import worker_alive


def _dataset_radio(at):
    """Find the Dataset radio by its options, not sidebar index."""
    for radio in at.sidebar.radio:
        opts = list(radio.options)
        if "Bearings" in opts and "Filters" in opts:
            return radio
    raise AssertionError("Dataset radio not found")


def _screen_radio(at):
    """Find the Screen radio by its options containing known screen names."""
    for radio in at.sidebar.radio:
        opts = list(radio.options)
        if "Data Quality" in opts and "Training" in opts and "Model Report" in opts:
            return radio
    raise AssertionError("Screen radio not found")


class _CFunc:
    """Callable with restype/argtypes so tests can stand in for ctypes prototypes."""

    def __init__(self, fn):
        self._fn = fn
        self.restype = None
        self.argtypes = None

    def __call__(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


def _isolate_pid_file(monkeypatch, tmp_path, contents: str | None = None):
    import pdm.worker as worker

    pid_file = tmp_path / "worker.pid"
    if contents is not None:
        pid_file.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(worker, "pid_path", lambda: pid_file)
    return pid_file


def _mock_nt_kernel32(monkeypatch, *, handle, last_error: int = 87):
    import ctypes

    import pdm.worker as worker

    closed: list[object] = []

    def open_process(access, inherit, pid):
        assert access == 0x1000
        assert inherit in {False, 0}
        assert pid == 4321
        return handle

    def close_handle(h):
        closed.append(h)
        return 1

    kernel32 = types.SimpleNamespace(
        OpenProcess=_CFunc(open_process),
        CloseHandle=_CFunc(close_handle),
    )

    def win_dll(name, use_last_error=False):
        assert name == "kernel32"
        assert use_last_error is True
        return kernel32

    monkeypatch.setattr(worker.os, "name", "nt")
    monkeypatch.setattr(ctypes, "WinDLL", win_dll, raising=False)
    monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error, raising=False)
    return closed


def test_worker_not_alive_without_pid(monkeypatch, tmp_path):
    # No duplicate training on UI rerun: spawn_worker refuses if worker_alive().
    _isolate_pid_file(monkeypatch, tmp_path)
    assert worker_alive() is False


@pytest.mark.parametrize("contents", ["", "nope", "0", "-1"])
def test_worker_alive_invalid_pid_file(monkeypatch, tmp_path, contents):
    _isolate_pid_file(monkeypatch, tmp_path, contents)
    assert worker_alive() is False


def test_worker_alive_posix_current_pid(monkeypatch, tmp_path):
    import pdm.worker as worker

    monkeypatch.setattr(worker.os, "name", "posix")
    monkeypatch.setattr(worker.os, "kill", lambda _pid, _sig: None)
    _isolate_pid_file(monkeypatch, tmp_path, str(os.getpid()))
    assert worker_alive() is True


def test_worker_alive_posix_dead_pid(monkeypatch, tmp_path):
    import pdm.worker as worker

    monkeypatch.setattr(worker.os, "name", "posix")
    _isolate_pid_file(monkeypatch, tmp_path, "99999999")

    def _missing(_pid, _sig):
        raise ProcessLookupError()

    monkeypatch.setattr(worker.os, "kill", _missing)
    assert worker_alive() is False


def test_worker_alive_windows_openprocess_handle(monkeypatch, tmp_path):
    closed = _mock_nt_kernel32(monkeypatch, handle=4242)
    _isolate_pid_file(monkeypatch, tmp_path, "4321")
    assert worker_alive() is True
    assert closed == [4242]


def test_worker_alive_windows_openprocess_zero(monkeypatch, tmp_path):
    closed = _mock_nt_kernel32(monkeypatch, handle=0, last_error=87)
    _isolate_pid_file(monkeypatch, tmp_path, "4321")
    assert worker_alive() is False
    assert closed == []


def test_worker_alive_windows_access_denied_is_alive(monkeypatch, tmp_path):
    closed = _mock_nt_kernel32(monkeypatch, handle=0, last_error=5)
    _isolate_pid_file(monkeypatch, tmp_path, "4321")
    assert worker_alive() is True
    assert closed == []


def test_worker_alive_windows_ctypes_error_returns_false(monkeypatch, tmp_path):
    import ctypes

    import pdm.worker as worker

    def _boom(name, use_last_error=False):
        raise AttributeError("kernel32")

    monkeypatch.setattr(worker.os, "name", "nt")
    monkeypatch.setattr(ctypes, "WinDLL", _boom, raising=False)
    _isolate_pid_file(monkeypatch, tmp_path, "4321")
    assert worker_alive() is False


def test_worker_alive_windows_overflow_pid_returns_false(monkeypatch, tmp_path):
    import ctypes

    import pdm.worker as worker

    def open_process(access, inherit, pid):
        raise OverflowError("int too long")

    kernel32 = types.SimpleNamespace(
        OpenProcess=_CFunc(open_process),
        CloseHandle=_CFunc(lambda _h: 1),
    )
    monkeypatch.setattr(worker.os, "name", "nt")
    monkeypatch.setattr(ctypes, "WinDLL", lambda *_a, **_k: kernel32, raising=False)
    _isolate_pid_file(monkeypatch, tmp_path, str((1 << 32) + 1))
    assert worker_alive() is False


def test_worker_alive_kill_valueerror_returns_bool(monkeypatch, tmp_path):
    import pdm.worker as worker

    monkeypatch.setattr(worker.os, "name", "posix")
    _isolate_pid_file(monkeypatch, tmp_path, str(os.getpid()))

    def _windows_like(_pid, _sig):
        raise ValueError("unsupported signal: 0")

    monkeypatch.setattr(worker.os, "kill", _windows_like)
    result = worker_alive()
    assert result is False
    assert isinstance(result, bool)


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
    _dataset_radio(at).set_value("Filters")
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
    _dataset_radio(at).set_value("Filters")
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
    from pdm.data.quality import QUALITY_POLICY_HASH

    return {
        "features": features,
        "units": units,
        "split": split,
        "report": {
            "history_length_physical": {"note": "synthetic fixture, not a real dataset"},
            "dataset_version": "testver",
        },
        "dir": None,
        "fingerprint": {"dataset_version": "testver", "quality_policy_hash": QUALITY_POLICY_HASH},
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
    _screen_radio(at).set_value("Training")
    at.run()
    assert not at.exception

    mode = next(r for r in at.radio if "Smoke" in list(r.options) and "Full" in list(r.options))
    assert mode.value == "Full"
    mode.set_value("Smoke")
    at.run()
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
    assert captured["max_epochs"] == 100
    assert captured["training_protocol"]["version"] == "training_v2"
    assert captured["training_protocol"]["selection_metric"] == "near_30m_mae_s"
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
    _dataset_radio(at).set_value("Filters")
    at.run()
    _screen_radio(at).set_value("Training")
    at.run()
    assert not at.exception
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "survival_nll (internal seconds)" in captions
    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "Training mode: **Full**" in markdown


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
    _screen_radio(at).set_value("Training")
    at.run()
    assert not at.exception
    next(r for r in at.radio if r.label == "Training mode").set_value("Smoke")
    at.run()
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
    val_uid = split["validation"][0]
    test_uid = split["test"][0]
    (edir / "predictions.csv").write_text(
        "run_id,unit_id,timestamp_s,predicted_rul_s\n"
        f"{run_id},{val_uid},60.0,100.0\n"
        f"{run_id},{val_uid},120.0,90.0\n",
        encoding="utf-8",
    )
    atomic_write_json(
        edir / "evaluation_config.json",
        {
            "eval_id": eval_id,
            "run_id": run_id,
            "metrics_version": METRICS_VERSION,
            "evaluate_mask": {
                "split": "validation",
                "unit_ids": list(split["validation"]),
                "blind_benchmark": False,
            },
        },
    )
    (edir / "metrics.json").write_text(
        json.dumps(
            {
                "eval_id": eval_id,
                "metrics_version": METRICS_VERSION,
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
        f"{val_uid},120.0,140.0,timely,600.0,True,1.0\n",
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
        "validation_ids": list(split["validation"]),
        "run_fingerprint": {"checkpoint_hash": "ckpt_freeze_test"},
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
    at.session_state["report_view"] = "Evaluation settings"
    _screen_radio(at).set_value("Model Report")
    at.run()
    assert not at.exception
    mode = next(r for r in at.radio if "Validation" in list(r.options) and "Research" in list(r.options))
    assert mode.value == "Validation"
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
    assert policy["source"] == "validation_ui"
    assert policy["split"] == "validation"
    assert policy["unit_ids"] == list(split["validation"])
    assert policy.get("policy_hash")
    assert policy.get("frozen_at")
    assert policy.get("checkpoint_hash") == "ckpt_freeze_test"

    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert eval_id in markdown
    metric_by_label = {m.label: m.value for m in at.metric}
    assert "Equal-weight unit MAE" in metric_by_label
    frames = [w.value for w in at.dataframe]
    unit_tbl = next(df for df in frames if "Unit" in df.columns and "MAE (s)" in df.columns)
    listed = set(unit_tbl["Unit"].astype(str))
    assert val_uid in listed
    assert test_uid not in listed
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
        unit_ids=["Bearing1_3", "Bearing2_5", "Bearing3_5"],
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
    ftable = metrics_by_unit_display_frame(filt, dataset_id="filters", unit_ids=["Test_1", "Test_2"])
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
        {"eval_id": eval_id, "run_id": run_id, "metrics_version": METRICS_VERSION},
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
    _dataset_radio(at).set_value("Filters")
    at.run()
    at.session_state["report_view"] = "Evaluation settings"
    _screen_radio(at).set_value("Model Report")
    at.run()
    assert not at.exception
    next(r for r in at.radio if "Validation" in list(r.options) and "Research" in list(r.options)).set_value(
        "Research"
    )
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
    at.session_state["report_view"] = "Evaluation settings"
    _screen_radio(at).set_value("Model Report")
    at.run()
    assert not at.exception
    errors = "\n".join(str(w.value) for w in at.error)
    assert "predictions.csv" in errors or "Evaluate test set" in errors
    infos = "\n".join(str(w.value) for w in at.info)
    assert "Run Evaluate validation set." in infos
    assert "Run Evaluate test set." not in infos
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
            {"eval_id": eval_id, "run_id": run_id, "metrics_version": METRICS_VERSION},
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
    at.session_state["report_view"] = "Evaluation settings"
    _screen_radio(at).set_value("Model Report")
    at.run()
    assert not at.exception
    next(r for r in at.radio if "Validation" in list(r.options) and "Research" in list(r.options)).set_value(
        "Research"
    )
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
    at.session_state["report_view"] = "Evaluation settings"
    _screen_radio(at).set_value("Model Report")
    at.run()
    assert not at.exception
    next(r for r in at.radio if "Validation" in list(r.options) and "Research" in list(r.options)).set_value(
        "Research"
    )
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


_HK_JOB_KEYS = (
    "H_trigger",
    "warning_horizon_s",
    "confirmation_count",
    "minimum_action_lead_time",
    "max_useful_horizon_s",
)


def _set_replay_mode(at, mode: str):
    radio = next(r for r in at.radio if "Validation" in list(r.options) and "Research" in list(r.options))
    radio.set_value(mode)
    at.run()
    assert not at.exception
    return at


def _open_replay_screen(at, *, mode: str | None = None):
    at.session_state["report_view"] = "Evaluation settings"
    _screen_radio(at).set_value("Model Report")
    at.run()
    assert not at.exception
    if mode and mode != "Validation":
        _set_replay_mode(at, mode)
    return at


def _bearing_replay_harness(monkeypatch, tmp_path, tiny_bearing_tables, *, run_id, captured=None):
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    rdir = tmp_path / run_id
    rdir.mkdir(parents=True, exist_ok=True)
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
        "validation_ids": list(split["validation"]),
        "run_fingerprint": {"checkpoint_hash": "ckpt_modes"},
    }
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda ds: ds == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda ds: bundle)
    monkeypatch.setattr("pdm.experiments.list_runs", lambda ds=None: [fake_row])
    monkeypatch.setattr("pdm.experiments.run_dir", lambda ds, rid: rdir)
    monkeypatch.setattr("pdm.replay.bind_replay_to_run", lambda *a, **k: bound)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)
    if captured is not None:
        monkeypatch.setattr("pdm.cli.spawn_worker", lambda job: captured.update(job) or captured)
    return rdir, split


def _write_simple_eval(
    rdir,
    eval_id,
    *,
    run_id,
    split_name,
    unit_ids,
    blind,
    pred_uid,
    alerts_body: str | None = None,
    alert_policy: dict | None = None,
):
    from pdm.io_util import atomic_write_json

    edir = rdir / "evaluations" / eval_id
    edir.mkdir(parents=True)
    (edir / "predictions.csv").write_text(
        "run_id,unit_id,timestamp_s,predicted_rul_s\n"
        f"{run_id},{pred_uid},60.0,100.0\n"
        f"{run_id},{pred_uid},120.0,90.0\n",
        encoding="utf-8",
    )
    cfg = {
        "eval_id": eval_id,
        "run_id": run_id,
        "metrics_version": METRICS_VERSION,
        "evaluate_mask": {
            "split": split_name,
            "unit_ids": [str(u) for u in unit_ids],
            "blind_benchmark": bool(blind),
        },
    }
    if alert_policy is not None:
        cfg["alert_policy"] = dict(alert_policy)
    atomic_write_json(edir / "evaluation_config.json", cfg)
    (edir / "metrics.json").write_text(
        json.dumps(
            {
                "eval_id": eval_id,
                "metrics_version": METRICS_VERSION,
                "primary_metric": "equal_weight_unit_mae",
                "equal_weight_unit_mae": 1.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (edir / "metrics_by_unit.csv").write_text(
        "unit_id,mae,rmse,alert_outcome,lead_time_s,has_sufficient_coverage,baseline_coverage_fraction\n"
        f"{pred_uid},1.0,1.0,timely,600.0,True,1.0\n",
        encoding="utf-8",
    )
    (edir / "alerts.csv").write_text(alerts_body or "unit_id\n", encoding="utf-8")
    return edir


def test_evaluations_for_mode_filters_legacy_and_blind():
    from pdm.experiments import evaluation_mask_view, evaluations_for_mode

    val = {
        "eval_id": "val",
        "evaluate_mask": {"split": "validation", "blind_benchmark": False, "unit_ids": ["V"]},
    }
    research = {"eval_id": "research", "evaluate_mask": evaluation_mask_view(None)}
    blind = {
        "eval_id": "blind",
        "evaluate_mask": {"split": "test", "blind_benchmark": True, "unit_ids": ["T"]},
    }
    rows = [val, research, blind]
    assert [e["eval_id"] for e in evaluations_for_mode(rows, "Validation")] == ["val"]
    assert [e["eval_id"] for e in evaluations_for_mode(rows, "Test")] == ["blind"]
    assert [e["eval_id"] for e in evaluations_for_mode(rows, "Research")] == ["research", "blind"]


def test_app_replay_test_mode_freeze_does_not_write(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    rdir, _split = _bearing_replay_harness(
        monkeypatch, tmp_path, tiny_bearing_tables, run_id="bearings_gru_test_no_freeze"
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at, mode="Test")
    freeze = next(b for b in at.button if "Freeze alert policy" in b.label)
    assert freeze.disabled
    assert not (rdir / "alert_policy.json").exists()


def test_app_replay_validation_evaluate_job_research_no_policy_write(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    captured: dict = {}
    rdir, _split = _bearing_replay_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        run_id="bearings_gru_val_eval",
        captured=captured,
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at)
    eval_btn = next(b for b in at.button if "Evaluate validation set" in b.label)
    eval_btn.click()
    at.run()
    assert not at.exception
    assert captured.get("kind") == "evaluate"
    assert captured.get("split_name") == "validation"
    assert captured.get("policy_mode") == "research"
    assert "H_trigger" in captured
    assert "confirmation_count" in captured
    assert not (rdir / "alert_policy.json").exists()


def test_app_replay_test_evaluate_missing_freeze_no_spawn(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    captured: dict = {}
    _bearing_replay_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        run_id="bearings_gru_test_eval_missing",
        captured=captured,
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at, mode="Test")
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "Freeze from Validation first" in captions
    eval_btn = next(b for b in at.button if "Evaluate test set" in b.label)
    eval_btn.click()
    at.run()
    assert not at.exception
    errors = "\n".join(str(w.value) for w in at.error)
    assert "Freeze from Validation first" in errors
    assert captured == {}


def test_app_replay_test_evaluate_frozen_omits_hk(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    captured: dict = {}
    rdir, _split = _bearing_replay_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        run_id="bearings_gru_test_eval_frozen",
        captured=captured,
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at)
    next(b for b in at.button if "Freeze alert policy" in b.label).click()
    at.run()
    assert (rdir / "alert_policy.json").exists()
    captured.clear()
    _set_replay_mode(at, "Test")
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "Test evaluation uses the frozen validation-selected policy." in captions
    next(b for b in at.button if "Evaluate test set" in b.label).click()
    at.run()
    assert not at.exception
    assert captured.get("kind") == "evaluate"
    assert captured.get("split_name") == "test"
    assert captured.get("policy_mode") == "frozen"
    assert set(_HK_JOB_KEYS).isdisjoint(captured)


def test_app_replay_test_play_uses_frozen_hk_not_widgets(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.replay import rescore_replay_alerts as real_rescore

    seen: dict = {}

    def _wrap(predictions, policy, **kwargs):
        seen["H_trigger"] = float(policy["H_trigger"])
        seen["confirmation_count"] = int(policy.get("confirmation_count") or 0)
        return real_rescore(predictions, policy, **kwargs)

    monkeypatch.setattr("pdm.replay.rescore_replay_alerts", _wrap)
    rdir, split = _bearing_replay_harness(
        monkeypatch, tmp_path, tiny_bearing_tables, run_id="bearings_gru_test_frozen_play"
    )
    test_uid = split["test"][0]
    _write_simple_eval(
        rdir,
        "20260107T000000Z_blindhk",
        run_id="bearings_gru_test_frozen_play",
        split_name="test",
        unit_ids=list(split["test"]),
        blind=True,
        pred_uid=test_uid,
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at)
    h = next(n for n in at.number_input if "H_trigger" in n.label)
    h.set_value(7.0)
    at.run()
    next(b for b in at.button if "Freeze alert policy" in b.label).click()
    at.run()
    frozen = json.loads((rdir / "alert_policy.json").read_text(encoding="utf-8"))
    frozen_h = float(frozen["H_trigger"])
    assert frozen_h == 7.0 * 60.0
    h = next(n for n in at.number_input if "H_trigger" in n.label)
    h.set_value(41.0)
    at.run()
    widget_h = float(next(n for n in at.number_input if "H_trigger" in n.label).value) * 60.0
    assert widget_h != frozen_h
    _set_replay_mode(at, "Test")
    leftover = float(next(n for n in at.number_input if "H_trigger" in n.label).value) * 60.0
    assert leftover == widget_h
    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "420 s (7 min)" in markdown
    assert "2460 s (41 min)" not in markdown
    view = at.session_state["_replay_view"]
    assert float(view["h_s"]) == frozen_h
    play = next(b for b in at.button if b.label == "Play")
    assert not play.disabled
    assert seen.get("H_trigger") == frozen_h
    assert seen.get("H_trigger") != leftover


def test_app_replay_test_play_uses_stored_eval_h_without_freeze(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.replay import rescore_replay_alerts as real_rescore

    seen: dict = {}

    def _wrap(predictions, policy, **kwargs):
        seen["H_trigger"] = float(policy["H_trigger"])
        return real_rescore(predictions, policy, **kwargs)

    monkeypatch.setattr("pdm.replay.rescore_replay_alerts", _wrap)
    rdir, split = _bearing_replay_harness(
        monkeypatch, tmp_path, tiny_bearing_tables, run_id="bearings_gru_test_stored_h"
    )
    stored_h = 123.0
    _write_simple_eval(
        rdir,
        "20260108T000000Z_storedh",
        run_id="bearings_gru_test_stored_h",
        split_name="test",
        unit_ids=list(split["test"]),
        blind=True,
        pred_uid=split["test"][0],
        alert_policy={
            "H_trigger": stored_h,
            "warning_horizon_s": stored_h,
            "minimum_action_lead_time": 60.0,
            "confirmation_count": 2,
        },
    )
    assert not (rdir / "alert_policy.json").exists()
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at)
    next(n for n in at.number_input if "H_trigger" in n.label).set_value(41.0)
    at.run()
    leftover = float(next(n for n in at.number_input if "H_trigger" in n.label).value) * 60.0
    assert leftover != stored_h
    _set_replay_mode(at, "Test")
    assert not (rdir / "alert_policy.json").exists()
    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "123 s (2.05 min)" in markdown
    assert "2460 s (41 min)" not in markdown
    play = next(b for b in at.button if b.label == "Play")
    assert not play.disabled
    assert seen.get("H_trigger") == stored_h
    assert seen.get("H_trigger") != leftover


def test_app_replay_test_play_blocked_without_freeze_or_stored_h(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.replay import rescore_replay_alerts as real_rescore

    seen: dict = {}

    def _wrap(predictions, policy, **kwargs):
        seen["H_trigger"] = float(policy["H_trigger"])
        return real_rescore(predictions, policy, **kwargs)

    monkeypatch.setattr("pdm.replay.rescore_replay_alerts", _wrap)
    rdir, split = _bearing_replay_harness(
        monkeypatch, tmp_path, tiny_bearing_tables, run_id="bearings_gru_test_no_policy"
    )
    _write_simple_eval(
        rdir,
        "20260109T000000Z_nopolicy",
        run_id="bearings_gru_test_no_policy",
        split_name="test",
        unit_ids=list(split["test"]),
        blind=True,
        pred_uid=split["test"][0],
    )
    assert not (rdir / "alert_policy.json").exists()
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at)
    next(n for n in at.number_input if "H_trigger" in n.label).set_value(41.0)
    at.run()
    leftover = float(next(n for n in at.number_input if "H_trigger" in n.label).value) * 60.0
    assert leftover == 41.0 * 60.0
    _set_replay_mode(at, "Test")
    play = next(b for b in at.button if b.label == "Play")
    assert play.disabled
    errors = "\n".join(str(w.value) for w in at.error)
    assert "Freeze from Validation first" in errors
    assert "widget" in errors.lower()
    markdown = "\n".join(str(w.value) for w in at.markdown)
    assert "2460 s (41 min)" not in markdown
    assert "unavailable" in markdown
    assert seen == {}


def test_app_replay_test_mode_excludes_validation_eval(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    rdir, split = _bearing_replay_harness(
        monkeypatch, tmp_path, tiny_bearing_tables, run_id="bearings_gru_eval_picker"
    )
    val_id = "20260104T000000Z_val0001"
    test_id = "20260104T000000Z_test0001"
    _write_simple_eval(
        rdir,
        val_id,
        run_id="bearings_gru_eval_picker",
        split_name="validation",
        unit_ids=list(split["validation"]),
        blind=False,
        pred_uid=split["validation"][0],
    )
    _write_simple_eval(
        rdir,
        test_id,
        run_id="bearings_gru_eval_picker",
        split_name="test",
        unit_ids=list(split["test"]),
        blind=True,
        pred_uid=split["test"][0],
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at)
    val_box = next(s for s in at.selectbox if "Evaluation" in s.label)
    assert val_id in list(val_box.options)
    assert test_id not in list(val_box.options)
    _set_replay_mode(at, "Test")
    test_box = next(s for s in at.selectbox if "Evaluation" in s.label)
    assert test_id in list(test_box.options)
    assert val_id not in list(test_box.options)


def test_app_replay_play_disabled_uid_not_in_mask(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    rdir, split = _bearing_replay_harness(
        monkeypatch, tmp_path, tiny_bearing_tables, run_id="bearings_gru_play_mask"
    )
    eval_id = "20260105T000000Z_mask0001"
    _write_simple_eval(
        rdir,
        eval_id,
        run_id="bearings_gru_play_mask",
        split_name="validation",
        unit_ids=["Bearing1_1"],
        blind=False,
        pred_uid=split["validation"][0],
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at)
    play = next(b for b in at.button if b.label == "Play")
    assert play.disabled
    errors = "\n".join(str(w.value) for w in at.error)
    assert "unit mask" in errors.lower() or "not in this evaluation" in errors.lower()


def test_app_replay_research_rescore_does_not_rewrite_alerts(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    rdir, split = _bearing_replay_harness(
        monkeypatch, tmp_path, tiny_bearing_tables, run_id="bearings_gru_research_rescore"
    )
    eval_id = "20260106T000000Z_blind01"
    marker = "FROZEN_BLIND_ALERT_ROW\n"
    edir = _write_simple_eval(
        rdir,
        eval_id,
        run_id="bearings_gru_research_rescore",
        split_name="test",
        unit_ids=list(split["test"]),
        blind=True,
        pred_uid=split["test"][0],
        alerts_body=marker,
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at, mode="Research")
    captions = "\n".join(str(w.value) for w in at.caption)
    assert "Research — not a blind benchmark." in captions
    h = next(n for n in at.number_input if "H_trigger" in n.label)
    h.set_value(float(h.value) + 1.0)
    at.run()
    assert not at.exception
    assert edir.joinpath("alerts.csv").read_text(encoding="utf-8") == marker
    play = next(b for b in at.button if b.label == "Play")
    assert not play.disabled
    unit_box = next(s for s in at.selectbox if s.label == "Unit")
    assert split["validation"][0] not in list(unit_box.options)
    assert split["test"][0] in list(unit_box.options)


def test_app_replay_research_evaluate_job_is_test_research(
    monkeypatch, tmp_path, tiny_bearing_tables
):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    captured: dict = {}
    rdir, split = _bearing_replay_harness(
        monkeypatch,
        tmp_path,
        tiny_bearing_tables,
        run_id="bearings_gru_research_eval",
        captured=captured,
    )
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    _open_replay_screen(at, mode="Research")
    freeze = next(b for b in at.button if "Freeze alert policy" in b.label)
    assert freeze.disabled
    next(b for b in at.button if "Evaluate test set" in b.label).click()
    at.run()
    assert not at.exception
    assert captured.get("kind") == "evaluate"
    assert captured.get("split_name") == "test"
    assert captured.get("policy_mode") == "research"
    assert "H_trigger" in captured
    assert not (rdir / "alert_policy.json").exists()
    unit_box = next(s for s in at.selectbox if s.label == "Unit")
    assert split["validation"][0] not in list(unit_box.options)


def test_stop_flag_sets_cancelled_not_completed(monkeypatch, tmp_path):
    from pdm.worker import read_status, run_job, stop_path

    def fake_eval(dataset_id, run_id, **kwargs):
        stop_path().write_text("stop\n", encoding="utf-8")
        return {"eval_id": "e1", "eval_dir": str(tmp_path / "e1")}

    wdir = tmp_path / "worker"
    wdir.mkdir()
    monkeypatch.setattr("pdm.worker.worker_dir", lambda: wdir)
    monkeypatch.setattr("pdm.evaluate.evaluate_run", fake_eval)
    run_job({"kind": "evaluate", "dataset_id": "bearings", "run_id": "r1"})
    status = read_status()["status"]
    assert status == "cancelled"
    assert status not in {"completed", "stopped"}


def test_app_operational_explorer_by_label_no_exception(tmp_path, monkeypatch):
    """The one-flow explorer stays selectable without legacy comparison/mode widgets."""
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.visualization.explorer import clear_soma_table_cache

    monkeypatch.setattr("pdm.connectome.anatomy.default_soma_dir", lambda: tmp_path)
    clear_soma_table_cache()
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    radio = _screen_radio(at)
    opts = list(radio.options)
    assert "Data Quality" in opts
    assert "Training" in opts
    assert "Model Report" in opts
    assert "Compare Models" in opts
    radio.set_value("Model Report")
    at.run()
    assert not at.exception
    parts = []
    for attr in ("caption", "markdown", "info", "warning", "error", "title", "header"):
        for widget in getattr(at, attr, []):
            parts.append(str(getattr(widget, "value", widget)))
    text = "\n".join(parts)
    assert "Real model states and causal forecasts from recorded measurements" in text
    assert "Architecture comparison" not in text
    assert not any(widget.label == "Mode" for widget in at.radio)
    assert not any(widget.label == "Build trace" for widget in at.button)
    at.session_state["report_view"] = "Evaluation settings"
    radio.set_value("Model Report")
    at.run()
    assert not at.exception
