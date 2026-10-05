from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

from pdm.evaluate import METRICS_VERSION
from pdm.worker import worker_alive


class _WorkflowSelection:
    """Small compatibility adapter for tests that select workflow state."""

    def __init__(self, at, *, kind):
        self.at = at
        self.kind = kind

    @property
    def options(self):
        if self.kind == "dataset":
            return ["Bearings", "Filters"]
        return ["Data Quality", "Training", "Model Report", "Compare Models"]

    @property
    def value(self):
        if self.kind == "dataset":
            return "Filters" if self.at.session_state.get("workflow_dataset") == "filters" else "Bearings"
        return self.at.session_state.get("screen_selection", "Data Quality")

    def set_value(self, value):
        if self.kind == "dataset":
            options = list(self.at.selectbox)
            project_picker = next((w for w in options if w.label == "Project"), None)
            source_picker = next((w for w in options if w.label == "Source format"), None)
            if project_picker is not None and value.lower() in list(project_picker.options):
                project_picker.set_value(value.lower())
            elif source_picker is not None:
                source_picker.set_value("HSE filters" if value.lower() == "filters" else "XJTU-SY bearings")
            else:
                self.at.session_state["workflow_dataset"] = value.lower()
        else:
            self.at.session_state["workflow_step"] = {
                "Data Quality": "Data Quality", "Training": "Training",
                "Model Report": "Results", "Compare Models": "Compare",
            }[value]
            # app.py retains this alias for report-specific legacy state.
            self.at.session_state["screen_selection"] = value
        return self


def _dataset_radio(at):
    """Set the selected imported project without relying on the removed dataset radio."""
    return _WorkflowSelection(at, kind="dataset")


def _screen_radio(at):
    """Set the workflow step while preserving legacy screen state used by app internals."""
    return _WorkflowSelection(at, kind="step")


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


@pytest.mark.parametrize(("status", "alive", "expected"), [
    *[(state, False, True) for state in ("queued", "starting", "running", "training", "preparing", "stopping")],
    *[(state, False, False) for state in ("completed", "failed", "cancelled", "not_ready")],
    ("completed", True, True),
])
def test_heavy_job_active_is_global_worker_predicate(monkeypatch, status, alive, expected):
    import pdm.worker as worker

    monkeypatch.setattr(worker, "worker_alive", lambda: alive)
    monkeypatch.setattr(worker, "read_status", lambda: {"status": status, "project_id": "other"})
    assert worker.heavy_job_active() is expected
    assert worker.heavy_job_active("this-project") is expected


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

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception


_WIDGET_COLLECTIONS = (
    "button_group", "radio", "selectbox", "multiselect", "select_slider", "checkbox", "toggle",
    "text_input", "text_area", "number_input", "slider", "date_input", "time_input", "color_picker",
)


def _product_app(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    return AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15), root


def _theme_controls(at):
    return [widget for attr in _WIDGET_COLLECTIONS for widget in getattr(at, attr)
            if getattr(widget, "label", None) == "Appearance"]


def _theme_control(at):
    widgets = _theme_controls(at)
    assert len(widgets) == 1
    return widgets[0]


def _help_of(widget) -> str:
    return (getattr(widget, "help", None) or widget.proto.help or "").strip()


def _theme_markers(at) -> set[str]:
    import re

    return {theme for m in at.markdown
            for theme in re.findall(r'class="brain-lab-shell" data-theme="(\w+)"', str(m.value))}


def _light_marker(at) -> bool:
    return _theme_markers(at) == {"light"}


def _assert_light(at):
    assert not at.exception
    assert at.session_state["_pdm_theme"] == "light"
    assert _theme_control(at).value == "light"
    assert _light_marker(at)


def test_import_navigation_and_theme_in_empty_workspace(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    monkeypatch.setattr("pdm.paths.data_raw", lambda: raw_root)
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda _dataset: False)
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    assert any(widget.label == "Source format" for widget in at.selectbox)
    assert any(widget.label == "Local file or folder path" for widget in at.text_input)
    assert any(widget.label == "Source files" for widget in at.file_uploader)
    assert any("Import data" in button.label for button in at.button)
    assert not any(r.label in {"Dataset", "Screen"} for r in at.radio)
    nav = [button for button in at.button if any(step in button.label for step in ("Data Quality", "Training", "Results", "Compare"))]
    assert nav and all(button.disabled for button in nav)

    control = _theme_control(at)
    assert control.value == "dark"
    control.set_value("light")
    at.run()
    assert not at.exception
    assert _theme_control(at).value == "light"


def test_import_and_quality_controls_expose_help(monkeypatch, tmp_path):
    import pandas as pd

    from pdm import ui_copy
    from pdm.projects import project_store
    from pdm.zone_limit_proposal import propose_absolute_limits

    at, _ = _product_app(monkeypatch, tmp_path)
    at.run()
    assert not at.exception
    assert _help_of(next(w for w in at.text_input if w.label == "Project name"))
    assert _help_of(next(w for w in at.selectbox if w.label == "Source format"))
    assert _help_of(next(b for b in at.button if b.label == "Create project"))

    pid = project_store().create("Motor", "generic_sensor_csv")["project_id"]
    at, _ = _product_app(monkeypatch, tmp_path)
    at.session_state["project_id"] = pid
    at.session_state["project_step"] = "Import data"
    at.run()
    assert not at.exception
    controls = [w for w in [*at.radio, *at.number_input, *at.text_input, *at.selectbox]
                if w.label not in {"Appearance", "Project", "Age source"}
                and not str(w.key or "").startswith(("import_context:", "import_context_unit:"))]
    assert controls
    missing = [w.label for w in controls if not _help_of(w)]
    assert not missing
    assert len(at.file_uploader) and all(_help_of(w) for w in at.file_uploader)
    assert _help_of(next(b for b in at.button if b.label == "Import and check data"))

    store = project_store()
    store.update(pid, active_snapshot_id="snapshot1", state="ready")
    features = pd.DataFrame({"unit_id": ["train1", "train1", "val1", "test1"],
                             "timestamp_s": [0.0, 1.0, 0.0, 0.0],
                             "signal": [-2.0, -1.0, 1.0, 2.0],
                             "gap_before": [False, True, False, False]})
    snapshot = {"project_id": pid, "snapshot_id": "snapshot1",
                "features": features,
                "split": {"train": ["train1"], "validation": ["val1"], "test": ["test1"]},
                "schema": {"signal_label": "Vibration", "signal_unit": "g"},
                "report": {"by_split": {"train": {"rejected_signal_rows": 1}}}}
    monkeypatch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
    monkeypatch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
    monkeypatch.setattr("pdm.project_quality_ui.available_signal_engines",
                        lambda _pid, _sid: [{"engine_id": "gru", "available": True}])
    at, _ = _product_app(monkeypatch, tmp_path)
    at.session_state["project_id"] = pid
    at.session_state["project_step"] = "Data Quality"
    at.run()
    assert not at.exception
    inspect = [w for w in at.selectbox if w.label.startswith("Inspect ")]
    assert inspect and all(_help_of(w) for w in inspect)
    assert not at.multiselect
    move_to = [w for w in at.selectbox if w.label == "Move to"]
    replace_with = [w for w in at.selectbox if w.label == "Replace with"]
    assert len(move_to) == 1 and len(replace_with) == 1
    move_unit = next(b for b in at.button if b.label == "Move unit")
    replace_unit = next(b for b in at.button if b.label == "Replace unit")
    assert _help_of(move_to[0]) == ui_copy.QUALITY_MOVE_TO_HELP
    assert _help_of(move_unit) == ui_copy.QUALITY_MOVE_SUBMIT_HELP
    assert _help_of(replace_with[0]) == ui_copy.QUALITY_REPLACE_WITH_HELP
    assert _help_of(replace_unit) == ui_copy.QUALITY_REPLACE_SUBMIT_HELP
    assert move_unit.proto.type != "primary" and replace_unit.proto.type != "primary"
    proposal = propose_absolute_limits(features, snapshot["split"]["train"], "above")
    suggest = next(b for b in at.button if b.label == "Suggest from Training Data")
    assert suggest.disabled
    assert _help_of(suggest) == ui_copy.QUALITY_SUGGEST_LIMITS_HELP
    assert suggest.proto.type != "primary"
    assert ui_copy.QUALITY_SUGGEST_LIMITS_CAPTION in [str(item.value) for item in at.caption]
    assert proposal["reason"] in [str(item.value) for item in at.caption]
    assert not at.exception
    assert len(at.metric) and all(_help_of(m) for m in at.metric if m.label != "Known operating age")
    assert _help_of(next(b for b in at.button if b.label == "Continue to Training"))
    admitted = next(m for m in at.metric if m.label == "Admitted rows")
    assert _help_of(admitted) == ui_copy.QUALITY_ADMITTED_ROWS_HELP
    assert ui_copy.QUALITY_ADMITTED_ROWS_HELP == (
        "Rows kept after import checks. Rows the import rejected are not counted; "
        "any remaining missing values are listed below."
    )


def test_ui_copy_help_strings_follow_rules():
    import re

    from pdm import ui_copy

    constants = {name: value for name, value in vars(ui_copy).items()
                 if not name.startswith("_") and isinstance(value, str)}
    assert "APPEARANCE_HELP" in constants
    for name, text in constants.items():
        assert len(text) <= 220, name
        assert text.endswith("."), name
        assert "!" not in text, name
        assert not re.search(r"\blikely\b", text, re.IGNORECASE), name


def _product_screens(monkeypatch, tmp_path):
    """Yield (expected title, ran AppTest) for Projects, Import data, Data Quality, Training."""
    import pandas as pd

    from pdm.projects import project_store
    from tests.project_contract import make_contract_snapshot

    at, root = _product_app(monkeypatch, tmp_path)
    yield "Projects", at.run()

    store = project_store()
    pid = store.create("Motor", "generic_sensor_csv")["project_id"]
    at, _ = _product_app(monkeypatch, tmp_path)
    at.session_state["project_id"] = pid
    at.session_state["project_step"] = "Import data"
    yield "Import data", at.run()

    store.update(pid, active_snapshot_id="snapshot1", state="ready")
    features = pd.DataFrame({"unit_id": ["train1", "train1", "val1", "test1"],
                             "timestamp_s": [0.0, 1.0, 0.0, 0.0],
                             "signal": [-2.0, -1.0, 1.0, 2.0],
                             "gap_before": [False, True, False, False]})
    snapshot = {"project_id": pid, "snapshot_id": "snapshot1", "features": features,
                "split": {"train": ["train1"], "validation": ["val1"], "test": ["test1"]},
                "schema": {"signal_label": "Vibration", "signal_unit": "g"},
                "report": {"by_split": {"train": {"rejected_signal_rows": 1}}}}
    with monkeypatch.context() as patch:
        patch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
        patch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
        patch.setattr("pdm.project_quality_ui.available_signal_engines",
                      lambda _pid, _sid: [{"engine_id": "gru", "available": True}])
        at, _ = _product_app(monkeypatch, tmp_path)
        at.session_state["project_id"] = pid
        at.session_state["project_step"] = "Data Quality"
        yield "Data Quality", at.run()

    _, project, _ = make_contract_snapshot(root)
    at, _ = _product_app(monkeypatch, tmp_path)
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Training"
    yield "Training", at.run()


def test_product_screens_single_primary_button(monkeypatch, tmp_path):
    expected = {"Projects": "Create project", "Import data": "Import and check data",
                "Data Quality": "Continue to Training", "Training": "Train model"}
    seen = []
    for title, at in _product_screens(monkeypatch, tmp_path):
        assert not at.exception, title
        if title == "Data Quality":
            assert any(widget.label == "Move to" for widget in at.selectbox)
            assert any("would have no units" in str(item.value) for item in at.warning)
            assert not at.multiselect
        # AppTest lists st.form_submit_button entries in at.button too (proto.is_form_submitter).
        primaries = [b for b in at.button if b.proto.type == "primary"
                     and not str(b.key or "").startswith("project_nav:")]
        assert [b.label for b in primaries] == [expected[title]], title
        seen.append(title)
    assert seen == list(expected)


def test_product_titles_match_default_task(monkeypatch, tmp_path):
    for title, at in _product_screens(monkeypatch, tmp_path):
        assert not at.exception, title
        assert [t.value for t in at.title] == [title]
        # Stylesheet markdown names the class; only the header element counts.
        has_desc = any('<p class="pdm-page-desc">' in str(m.value) for m in at.markdown)
        if title in {"Data Quality", "Training"}:
            assert not has_desc, title
        else:
            assert has_desc, title


def test_training_form_boosting_has_tree_iterations_and_corridor_epochs(monkeypatch, tmp_path):
    from pdm import ui_copy
    from tests.project_contract import make_contract_snapshot

    at, root = _product_app(monkeypatch, tmp_path)
    _, project, _ = make_contract_snapshot(root)
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Training"
    at.run()
    assert not at.exception
    assert [s.value for s in at.subheader][:3] == ["Data window", "Model size", "Repeatability"]
    model = next(w for w in at.selectbox if w.label == "Model")
    model.set_value("quantile_boosting")
    at.run()
    assert not at.exception
    labels = {w.label for w in at.number_input}
    assert "Training epochs" in labels
    assert {"Random seed", "Boosting iterations"} <= labels
    assert ui_copy.TRAIN_BOOSTING_NO_EPOCHS_CAPTION == (
        "Quantile boosting uses a fixed iteration count."
    )
    next(w for w in at.selectbox if w.label == "Model").set_value("gru")
    at.run()
    assert not at.exception
    labels = {w.label for w in at.number_input}
    assert {"Training epochs", "Hidden units", "Batch size", "Random seed"} <= labels
    assert ui_copy.TRAIN_BOOSTING_NO_EPOCHS_CAPTION not in [c.value for c in at.caption]


def test_theme_light_survives_project_navigation_and_actions(monkeypatch, tmp_path):
    from tests.project_contract import make_contract_snapshot

    at, root = _product_app(monkeypatch, tmp_path)
    _, project, _ = make_contract_snapshot(root)
    contract_pid = project["project_id"]
    at.run()
    assert not at.exception
    _theme_control(at).set_value("light")
    at.run()
    _assert_light(at)

    next(widget for widget in at.text_input if widget.label == "Project name").set_value("Theme check")
    next(button for button in at.button if button.label == "Create project").click()
    at.run()
    _assert_light(at)

    at.button(key="project_nav:Projects").click()
    at.run()
    _assert_light(at)

    at.button(key=f"open:{contract_pid}").click()
    at.run()
    _assert_light(at)

    for step in ("Data Quality", "Training"):
        at.button(key=f"project_nav:{step}").click()
        at.run()
        _assert_light(at)
        assert at.session_state["project_step"] == step


def test_theme_light_survives_legacy_workflow_steps(monkeypatch, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda dataset: dataset == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda _dataset: bundle)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.session_state["workflow_dataset"] = "bearings"
    at.session_state["workflow_step"] = "Data Quality"
    at.session_state["screen_selection"] = "Data Quality"
    at.run()
    assert not at.exception
    _theme_control(at).set_value("light")
    at.run()
    _assert_light(at)
    next(button for button in at.button if button.label == "Continue to Training").click()
    at.run()
    _assert_light(at)
    assert at.session_state["workflow_step"] == "Training"


def test_theme_restored_from_query_param_on_fresh_session(monkeypatch, tmp_path):
    at, _ = _product_app(monkeypatch, tmp_path)
    at.query_params["theme"] = "light"
    at.run()
    _assert_light(at)


def test_theme_single_control_in_sidebar(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    at, _ = _product_app(monkeypatch, tmp_path)
    at.run()
    assert not at.exception
    assert len(_theme_controls(at)) == 1
    assert len([w for attr in _WIDGET_COLLECTIONS for w in getattr(at.sidebar, attr)
                if getattr(w, "label", None) == "Appearance"]) == 1
    legacy = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    legacy.run()
    assert not legacy.exception
    assert len(_theme_controls(legacy)) == 1


def test_theme_light_survives_results_replay_fragment():
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    at = AppTest.from_file(str(project_root() / "tests" / "project_results_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    assert _light_marker(at)
    next(button for button in at.button if button.label == "Play").click()
    at.run()
    at.run()
    assert not at.exception
    assert at.session_state["_pdm_theme"] == "light"
    assert _light_marker(at)


def test_theme_set_none_is_ignored(monkeypatch):
    import pdm.ui_theme as ui_theme

    fake_st = types.SimpleNamespace(
        session_state={"_pdm_theme": "light", "_pdm_theme_widget": None},
        query_params={"theme": "light"},
        context=types.SimpleNamespace(theme=types.SimpleNamespace(type="dark")),
    )
    monkeypatch.setattr(ui_theme, "st", fake_st)
    ui_theme.set_theme(None)
    assert fake_st.session_state == {"_pdm_theme": "light", "_pdm_theme_widget": "light"}
    assert fake_st.query_params == {"theme": "light"}
    ui_theme.set_theme("sepia")
    assert fake_st.session_state["_pdm_theme"] == "light"
    ui_theme.set_theme("dark")
    assert fake_st.session_state["_pdm_theme"] == "dark"
    assert fake_st.query_params == {"theme": "dark"}


_EXPLORER_CSS = Path(__file__).resolve().parents[1] / "src" / "pdm" / "visualization" / "explorer.css"
_LEGACY_CSS_LITERALS = (
    "#0b0f14 #10151c #141a22 #19212b #111820 #151d26 #1a232d #222e3a #2a3542 #202a35 #384656 "
    "#354351 #253a4b #f3f5f7 #d8e0e8 #aab7c5 #8493a3 #55c9dc #83dce8 #071317 #e7b866 #f0d39f "
    "#4ca982 #d36d6d #f4f5f7 #eceef1 #f7f8fa #f8f9fb #f4f6f8 #e9edf1 #d9dee5 #e6e9ee #c6ced8 "
    "#d9e0e7 #deeff2 #20252b #303943 #5b6672 #737f8c #087f91 #086c7b #96620a #805507 #237a53 "
    "#b34444 #34414d #4c5661 #697480 #f0f2f5 #e8ebef #e2e7eb --gdg-"
).split()


def _theme_leaks(theme: str) -> list[str]:
    from pdm.ui_theme import DARK_EXCLUSIVE, LIGHT_EXCLUSIVE, TOKENS, theme_css

    other = "dark" if theme == "light" else "light"
    css = theme_css(theme).lower()
    exclusive = DARK_EXCLUSIVE if theme == "light" else LIGHT_EXCLUSIVE
    differing = {str(value) for key, value in TOKENS[other].items()
                 if isinstance(value, str) and value != TOKENS[theme].get(key)}
    leaks = [value for value in {*exclusive, *differing} if value.lower() in css]
    combined = css + _EXPLORER_CSS.read_text(encoding="utf-8").lower()
    leaks += [literal for literal in _LEGACY_CSS_LITERALS if literal in combined]
    return leaks


def test_light_theme_css_contains_no_dark_colors():
    from pdm.ui_theme import TOKENS, theme_css

    css = theme_css("light")
    assert css.startswith('body:has(.brain-lab-shell[data-theme="light"])')
    assert css.count("{") == 1
    assert "color-scheme:light" in css
    assert f"--pdm-bg:{TOKENS['light']['bg']}" in css
    assert "--lab-panel:var(--pdm-surface)" in css
    assert "series-cycle" not in css and "heatmap-scale" not in css
    assert not _theme_leaks("light")


def test_dark_theme_css_contains_no_light_colors():
    from pdm.ui_theme import TOKENS, theme_css

    css = theme_css("dark")
    assert css.startswith('body:has(.brain-lab-shell[data-theme="dark"])')
    assert css.count("{") == 1
    assert "color-scheme:dark" in css
    assert f"--pdm-bg:{TOKENS['dark']['bg']}" in css
    assert "series-cycle" not in css and "heatmap-scale" not in css
    assert not _theme_leaks("dark")


def test_explorer_css_has_no_color_literals():
    import re

    css = _EXPLORER_CSS.read_text(encoding="utf-8")
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b", css)
    assert not re.findall(r"\b(?:rgba?|hsla?)\(", css)
    assert not re.findall(
        r"(?<![-\w])(?:white|black|red|green|blue|yellow|orange|gray|grey|cyan|magenta)(?![-\w])", css,
        flags=re.IGNORECASE,
    )


def test_explorer_css_min_font_size_11px():
    import re

    css = _EXPLORER_CSS.read_text(encoding="utf-8")
    sizes = [float(s) for s in re.findall(r"font-size:\s*(\d+(?:\.\d+)?)px", css)]
    assert sizes and min(sizes) >= 11
    rem_sizes = [float(s) for s in re.findall(r"font-size:\s*(0?\.\d+)r?em", css)]
    assert all(size * 16 >= 11 for size in rem_sizes), rem_sizes
    assert "font-size:clamp(" not in css.replace(" ", "")


def test_explorer_css_has_no_glow_or_gradient():
    import re

    css = _EXPLORER_CSS.read_text(encoding="utf-8")
    rules = re.findall(r"[^{}]*\{[^{}]*\}", re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL))
    # These hard color stops encode progress, rather than decorative gradients.
    # Match both the exact selector and declaration so unrelated rules (including
    # additional slider/rail gradients) cannot inherit a broad component exception.
    allowed_gradients = {
        'body:has(.brain-lab-shell) [data-testid="stSlider"] '
        '[data-orientation="horizontal"] > [data-orientation="horizontal"] > div:first-child': (
            "background:linear-gradient(to right,var(--lab-accent) 0%,"
            "var(--lab-accent) var(--lab-progress,0%),"
            "var(--pdm-border-soft) var(--lab-progress,0%),var(--pdm-border-soft) 100%);"
        ),
        'body:has(.brain-lab-shell) [class*="st-key-pdm-workflow-rail"] > '
        '[data-testid="stElementContainer"]:has(button[kind="primary"]):not(:first-child)::after': (
            "background:linear-gradient(var(--pdm-accent) 23px,var(--pdm-border) 23px);"
        ),
    }
    gradient_rules = []
    for rule in rules:
        selector, declarations = rule[:-1].split("{", 1)
        if re.search(r"\b[\w-]*gradient\s*\(", declarations, flags=re.IGNORECASE):
            gradient_rules.append((re.sub(r"\s+", " ", selector.strip()),
                                   re.sub(r"\s+", "", declarations)))
    assert sorted(gradient_rules) == sorted(
        (selector, re.sub(r"\s+", "", declarations))
        for selector, declarations in allowed_gradients.items()
    )
    assert "radial-gradient(" not in css and "conic-gradient(" not in css
    assert not re.findall(r"box-shadow:\s*0(?:px)?\s+0(?:px)?\s+[1-9]", css)
    assert "--pdm-focus-ring" in css
    focus_rule = next(rule for rule in rules if ":focus-visible," in rule and "outline" in rule)
    assert "outline: 2px solid var(--pdm-accent)" in focus_rule and "outline-offset: 2px" in focus_rule


_PDM_SRC = Path(__file__).resolve().parents[1] / "src" / "pdm"
_UI_MODULES = (
    "app.py", "project_chart_style.py", "project_quality_ui.py", "project_results_ui.py",
    "project_training_ui.py", "project_ui.py", "lab_ui.py", "health_zones_ui.py",
    "filter_health_zones_ui.py", "visualization/overlay.py", "visualization/recurrent_trace.py",
    "visualization/presentation.py", "monitoring/ui.py", "future_red_ui.py",
)
_CSS_COLOR_NAMES = "white|black|red|green|blue|yellow|orange|gray|grey|cyan|magenta"


def test_ui_modules_have_no_hardcoded_color_literals():
    import re

    patterns = (
        r"[\"']#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?[\"']",
        r"#[0-9a-fA-F]{3,8}\b",
        r"\brgba?\(",
        rf"(?:color|bgcolor|fillcolor|line_color|bordercolor)\s*=\s*[\"'](?:{_CSS_COLOR_NAMES})[\"']",
        rf"[\"'](?:color|bgcolor)[\"']\s*:\s*[\"'](?:{_CSS_COLOR_NAMES})[\"']",
    )
    found = []
    for name in _UI_MODULES:
        source = (_PDM_SRC / name).read_text(encoding="utf-8")
        for pattern in patterns:
            found += [f"{name}: {match}" for match in re.findall(pattern, source, flags=re.IGNORECASE)]
    assert not found, found


def test_no_builtin_plotly_bw_templates():
    banned = tuple("plotly_" + suffix for suffix in ("dark", "white"))
    offenders = [str(path.relative_to(_PDM_SRC)) for path in _PDM_SRC.rglob("*.py")
                 if any(name in path.read_text(encoding="utf-8") for name in banned)]
    assert not offenders


def _plotly_payload(fig) -> str:
    return json.dumps(fig.to_plotly_json(), default=str).lower()


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_signal_chart_uses_only_its_theme_colors(theme):
    import plotly.graph_objects as go

    from pdm.project_chart_style import style_signal_chart
    from pdm.ui_theme import DARK_EXCLUSIVE, FONT_STACK, LIGHT_EXCLUSIVE, TOKENS

    fig = style_signal_chart(go.Figure(go.Scatter(x=[0, 1], y=[1, 2])), theme)
    payload = _plotly_payload(fig)
    other = DARK_EXCLUSIVE if theme == "light" else LIGHT_EXCLUSIVE
    assert not [value for value in other if value.lower() in payload]
    t = TOKENS[theme]
    layout = fig.layout
    assert layout.paper_bgcolor == layout.plot_bgcolor == t["chart_bg"]
    assert layout.font.color == t["chart_text"] and layout.font.family == FONT_STACK
    assert layout.hovermode == "x unified"
    assert (layout.hoverlabel.bgcolor, layout.hoverlabel.bordercolor, layout.hoverlabel.font.color) == (
        t["surface"], t["border"], t["text"])
    assert layout.yaxis.showgrid is True and layout.yaxis.gridcolor == t["chart_grid"]
    assert layout.xaxis.showgrid is False
    assert layout.xaxis.linecolor == layout.yaxis.linecolor == t["chart_axis"]
    assert layout.margin.t == 16
    fig.add_trace(go.Scatter(x=[0, 1], y=[2, 3]))
    assert style_signal_chart(fig, theme).layout.margin.t == 48


def test_signal_chart_light_has_no_dark_colors():
    import plotly.graph_objects as go

    from pdm.project_chart_style import style_signal_chart
    from pdm.ui_theme import DARK_EXCLUSIVE, LIGHT_EXCLUSIVE

    light = _plotly_payload(style_signal_chart(go.Figure(go.Scatter(x=[0, 1], y=[1, 2])), "light"))
    assert not [value for value in DARK_EXCLUSIVE if value.lower() in light]
    dark = _plotly_payload(style_signal_chart(go.Figure(go.Scatter(x=[0, 1], y=[1, 2])), "dark"))
    assert not [value for value in LIGHT_EXCLUSIVE if value.lower() in dark]


def test_replay_figure_uses_theme_tokens():
    from pdm.project_results_ui import replay_figure
    from pdm.ui_theme import DARK_EXCLUSIVE, TOKENS

    result = {"project_id": "project1", "run_id": "run1", "snapshot_id": "snapshot1", "as_of_s": 3.0,
              "observed_prefix": [{"timestamp_s": 2.0, "signal": 2.0}, {"timestamp_s": 3.0, "signal": 3.0}],
              "points": [{"target_time_s": 4.0, "value": 3.5, "lower": 3.0, "upper": 4.0}],
              "thresholds": {"status": "available", "mode": "absolute", "direction": "above",
                             "yellow": 10, "red": 12},
              "crossing": {"status": "none_within_horizon", "time_s": None}}
    fig = replay_figure(result, {"signal_label": "Signal", "signal_unit": "g"}, "light")
    payload = _plotly_payload(fig)
    light = TOKENS["light"]
    for key in ("series_forecast", "series_observed", "series_band", "series_band_line",
                "zone_yellow_fill", "zone_red_fill", "series_reference"):
        assert light[key].lower() in payload, key
    assert not [value for value in DARK_EXCLUSIVE if value.lower() in payload]


def test_zone_colors_keep_health_zone_roles():
    from pdm.ui_theme import TOKENS, series_cycle, zone_colors

    for theme in ("light", "dark"):
        t = TOKENS[theme]
        assert zone_colors(theme) == {"green": t["zone_green"], "yellow": t["zone_yellow"], "red": t["zone_red"],
                                      "unknown": t["zone_unknown"], "gray": t["zone_unknown"]}
        assert series_cycle(theme) == t["series_cycle"]


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_comparison_actual_color_is_reference_and_distinct_from_models(theme):
    from pdm.lab_ui import comparison_colors
    from pdm.ui_theme import TOKENS

    t = TOKENS[theme]
    actual, lower_bound, models = comparison_colors(theme)
    assert actual == t["series_reference"]
    assert lower_bound == t["series_observed"]
    assert models and actual not in models and lower_bound not in models
    assert models == [c for c in t["series_cycle"] if c not in {t["series_observed"], t["series_reference"]}]


def test_theme_css_covers_native_surfaces():
    from pdm.ui_theme import theme_css

    css = theme_css("light") + theme_css("dark") + _EXPLORER_CSS.read_text(encoding="utf-8")
    for needle in (
        'data-testid="stTooltipContent"', '[role="tooltip"]', 'data-baseweb="popover"',
        'data-testid="stSelectboxVirtualDropdown"', '[role="listbox"]', '[role="option"]',
        'data-testid="stMainMenuPopover"', 'data-testid="stFileUploaderDropzone"',
        '[data-testid="stNumberInput"] button', 'data-testid="stCheckbox"', 'data-testid="stRadio',
        'data-testid="stProgress"', 'data-testid="stToast"', 'data-testid="stHeader"',
        'data-testid="stToolbar"', 'data-testid="stCode"', 'data-testid="stTable"', "color-scheme",
    ):
        assert needle in css, needle
    assert 'body:has(.brain-lab-shell[data-theme="light"]) [data-testid="stDataFrame"] canvas' in css
    assert '[data-theme="dark"]) [data-testid="stDataFrame"] canvas' not in css
    assert "stTable\"] canvas" not in css


def test_product_tables_use_st_table(monkeypatch, tmp_path):
    import pandas as pd

    from pdm.paths import project_root
    from pdm.projects import project_store

    at, _ = _product_app(monkeypatch, tmp_path)
    project = project_store().create("Motor", "generic_sensor_csv")
    project_store().update(project["project_id"], active_snapshot_id="snapshot1", state="ready")
    features = pd.DataFrame({"unit_id": ["train1", "train1", "val1", "test1"],
                             "timestamp_s": [0.0, 1.0, 0.0, 0.0],
                             "signal": [-2.0, -1.0, 1.0, 2.0],
                             "gap_before": [False, True, False, False]})
    snapshot = {"project_id": project["project_id"], "snapshot_id": "snapshot1",
                "features": features,
                "split": {"train": ["train1"], "validation": ["val1"], "test": ["test1"]},
                "schema": {"signal_label": "Vibration", "signal_unit": "g"},
                "report": {"by_split": {"train": {"rejected_signal_rows": 1}}}}
    monkeypatch.setattr("pdm.project_ui.load_snapshot", lambda _pid: snapshot)
    monkeypatch.setattr("pdm.project_ui.list_project_runs", lambda _pid: [])
    monkeypatch.setattr("pdm.project_quality_ui.available_signal_engines",
                        lambda _pid, _sid: [{"engine_id": "gru", "available": True}])
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Data Quality"
    at.run()
    assert not at.exception
    assert len(at.table) >= 1
    assert len(at.dataframe) == 0
    sample_tables = [table.value for table in at.table
                     if list(table.value.columns) == ["Time (s)", "Vibration (g)", "Record position", "Zone"]]
    assert len(sample_tables) == 1
    first = sample_tables[0]
    assert list(first.columns) == ["Time (s)", "Vibration (g)", "Record position", "Zone"]
    assert list(first["Record position"]) == ["Start of record", "Gap before"]
    for name in ("project_quality_ui.py", "project_results_ui.py"):
        assert "st.dataframe" not in (project_root() / "src" / "pdm" / name).read_text(encoding="utf-8")


def test_config_toml_locks_streamlit_base_dark():
    import tomllib

    from pdm.paths import project_root
    from pdm.ui_theme import TOKENS

    with (project_root() / ".streamlit" / "config.toml").open("rb") as handle:
        config = tomllib.load(handle)
    theme = config["theme"]
    assert theme["base"] == "dark"
    assert theme["backgroundColor"].lower() == str(TOKENS["dark"]["bg"]).lower()
    assert "light" not in theme
    assert "dark" not in theme
    assert config["server"]["address"] == "127.0.0.1"


def test_prepared_project_can_advance_from_quality_to_training(monkeypatch, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda dataset: dataset == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda _dataset: bundle)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.session_state["workflow_dataset"] = "bearings"
    at.session_state["workflow_step"] = "Data Quality"
    at.session_state["screen_selection"] = "Data Quality"
    at.run()
    assert not at.exception
    assert at.session_state["workflow_step"] == "Data Quality"
    training = next(button for button in at.button if "Training" in button.label)
    assert not training.disabled
    training.click()
    at.run()
    assert not at.exception
    assert at.session_state["workflow_step"] == "Training"


def test_training_button_dispatches_selected_models(monkeypatch, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    jobs = []
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda dataset: dataset == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda _dataset: bundle)
    monkeypatch.setattr("pdm.cli.spawn_worker", lambda job: jobs.append(job))
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.session_state["workflow_dataset"] = "bearings"
    at.session_state["workflow_step"] = "Training"
    at.session_state["screen_selection"] = "Training"
    at.run()
    assert not at.exception
    next(button for button in at.button if button.label == "Start training").click()
    at.run()
    assert not at.exception
    assert jobs == [{"kind": "future_red_matrix", "dataset_id": "bearings", "architectures": ["gru"]}]


def test_training_controls_expose_help(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from tests.project_contract import make_contract_snapshot

    root = tmp_path / "projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(root))
    _, project, _ = make_contract_snapshot(root)
    at = AppTest.from_file(str(project_root() / "src" / "pdm" / "app.py"), default_timeout=15)
    at.session_state["project_id"] = project["project_id"]
    at.session_state["project_step"] = "Training"
    at.run()
    assert not at.exception
    assert all(w.label != "Training task" for w in at.selectbox)
    widgets = [*at.selectbox, *at.number_input, *at.text_input]
    for label in ("Model", "History samples", "Forecast span (minutes)", "Random seed",
                  "Training epochs", "Hidden units", "Batch size"):
        assert _help_of(next(w for w in widgets if w.label == label)), label
    assert _help_of(next(b for b in at.button if b.label == "Train model"))

    next(w for w in at.selectbox if w.label == "Model").set_value("quantile_boosting")
    at.run()
    assert not at.exception
    assert _help_of(next(w for w in at.number_input if w.label == "Boosting iterations"))
    assert "trend corridor" in _help_of(next(w for w in at.selectbox if w.label == "Model"))


def test_training_stop_help_copy():
    from pdm import ui_copy
    from pdm.paths import project_root

    assert ui_copy.TRAIN_STOP_HELP == (
        "Asks the job to stop after its current safe step. No model is saved from a stopped run; "
        "earlier saved runs stay available."
    )
    assert "ui_copy.TRAIN_STOP_HELP" in Path(project_root() / "src/pdm/project_training_ui.py").read_text()


def test_training_mae_help_mentions_equal_units():
    from pdm import ui_copy

    assert "each unit counts equally" in ui_copy.TRAIN_VALIDATION_MAE_HELP
    assert "each unit counts equally" in ui_copy.TRAIN_TEST_MAE_HELP
    assert "trend corridor" in ui_copy.TRAIN_MODEL_HELP
    assert "likely range" not in ui_copy.TRAIN_MODEL_HELP


def test_future_red_training_controls_expose_help(monkeypatch, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.splits import bearings_split

    features, units = tiny_bearing_tables
    split = bearings_split(units)
    bundle = _fake_processed_bundle("bearings", features, units, split)
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda dataset: dataset == "bearings")
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda _dataset: bundle)
    monkeypatch.setattr("pdm.cli.spawn_worker", lambda job: None)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.session_state["workflow_dataset"] = "bearings"
    at.session_state["workflow_step"] = "Training"
    at.session_state["screen_selection"] = "Training"
    at.run()
    assert not at.exception
    assert _help_of(next(w for w in at.multiselect if w.label == "Model families"))
    assert _help_of(next(b for b in at.button if b.label == "Start training"))


def test_app_filters_data_shows_source_seconds_note():
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    _dataset_radio(at).set_value("Filters")
    at.run()
    assert not at.exception
    assert next(widget for widget in at.selectbox if widget.label == "Source format").value == "HSE filters"
    assert any(widget.label == "Source files" for widget in at.file_uploader)
    assert any("local path" in str(widget.value).lower() for widget in at.caption)


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

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.session_state["workflow_dataset"] = "filters"
    at.session_state["workflow_step"] = "Data Quality"
    at.session_state["screen_selection"] = "Data Quality"
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
    assert "Window counts" in captions or "history_length" in captions
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


def _isolate_legacy_navigation(monkeypatch, tmp_path, dataset_id, tables):
    """Legacy UI tests must not depend on the user's prepared data or saved runs."""
    from pdm.splits import bearings_split, filters_split

    features, units = tables
    split = (bearings_split if dataset_id == "bearings" else filters_split)(units)
    bundle = _fake_processed_bundle(dataset_id, features, units, split)
    monkeypatch.setattr("pdm.data.prepare.processed_ready", lambda value: value == dataset_id)
    monkeypatch.setattr("pdm.data.prepare.load_processed", lambda _value: bundle)
    monkeypatch.setattr("pdm.paths.runs_root", lambda: tmp_path / "runs")
    monkeypatch.setattr("pdm.experiments.list_runs", lambda _value: [
        {"run_id": "synthetic-navigation-fixture", "has_legacy_predictions": True}])
    manifest = {"run_id": "synthetic-matrix-fixture", "run_config": {
        "datasets": [dataset_id], "architectures": ["gru"],
        "target_artifacts": {dataset_id: {"horizon_s": 1800 if dataset_id == "bearings" else 20}}}}
    monkeypatch.setattr("pdm.future_red_ui.latest_matrix", lambda *_args: (tmp_path, manifest))
    monkeypatch.setattr("pdm.future_red_ui.load_test_metrics", lambda *_args: [
        {"dataset_id": dataset_id, "architecture": "gru", "split": "test"}])
    monkeypatch.setattr("pdm.future_red_ui.matching_full_cns_run", lambda *_args: None)


def test_app_training_screen_shows_future_red_matrix_without_rul_controls(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    _isolate_legacy_navigation(monkeypatch, tmp_path, "bearings", tiny_bearing_tables)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    assert list(_screen_radio(at).options) == ["Data Quality", "Training", "Model Report", "Compare Models"]
    _screen_radio(at).set_value("Training")
    at.run()
    assert not at.exception
    markdown = "\n".join(str(w.value) for w in at.markdown)
    headers = "\n".join(str(w.value) for w in at.header)
    code = "\n".join(str(w.value) for w in at.code)
    assert "Future-red entry training — Bearings" in headers
    assert "1,800 seconds (30 minutes)" in markdown
    assert "--datasets bearings --architectures gru,lstm,fly,random" in code
    assert "Test split snapshot" in markdown
    assert any(button.label == "Start training" for button in at.button)
    assert not any("Train full MaleCNS" in button.label for button in at.button)
    assert not any(r.label == "Training mode" for r in at.radio)
    assert not any("Resume" in widget.label for widget in at.selectbox)
    assert not any({"Smoke", "Full"}.issubset(set(widget.options)) for widget in at.radio)
    _screen_radio(at).set_value("Compare Models")
    at.run()
    assert not at.exception
    assert any(widget.label == "Comparison target" for widget in at.selectbox)
    assert any("Compare models" in str(widget.value) for widget in at.header)


def test_app_training_screen_filters_uses_20_second_future_red_target(monkeypatch, tmp_path, tiny_filter_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    _isolate_legacy_navigation(monkeypatch, tmp_path, "filters", tiny_filter_tables)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.session_state["workflow_dataset"] = "filters"
    at.session_state["workflow_step"] = "Training"
    at.session_state["screen_selection"] = "Training"
    at.run()
    _screen_radio(at).set_value("Training")
    at.run()
    assert not at.exception
    markdown = "\n".join(str(w.value) for w in at.markdown)
    code = "\n".join(str(w.value) for w in at.code)
    assert "20 seconds" in markdown
    assert "--datasets filters --architectures gru,lstm,fly,random" in code
    assert any(button.label == "Start training" for button in at.button)
    assert not any(r.label == "Training mode" for r in at.radio)
    _screen_radio(at).set_value("Model Report")
    at.run()
    warnings = "\n".join(str(widget.value) for widget in at.warning)
    assert "no observed RED-entry events" in warnings
    assert any(r.label == "Report view" and "Health zones" in r.options for r in at.radio)


def test_app_training_screen_hides_legacy_resume_controls(monkeypatch, tmp_path, tiny_bearing_tables):
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root

    _isolate_legacy_navigation(monkeypatch, tmp_path, "bearings", tiny_bearing_tables)
    monkeypatch.setattr("pdm.worker.worker_alive", lambda: False)
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    _screen_radio(at).set_value("Training")
    at.run()
    assert not at.exception
    labels = [widget.label for widget in at.selectbox]
    assert not any("Resume" in label or "Architecture" in label for label in labels)
    assert any("Start training" in button.label for button in at.button)
    markdown = "\n".join(str(widget.value) for widget in at.markdown)
    assert "remaining useful life" in markdown


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

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.session_state["report_view"] = "Historical RUL — Evaluation settings"
    _screen_radio(at).set_value("Model Report")
    at.run()
    assert not at.exception
    mode = next(r for r in at.radio if "Validation" in list(r.options) and "Research" in list(r.options))
    assert mode.value == "Validation"
    eval_box = next(s for s in at.selectbox if "Evaluation" in s.label)
    assert eval_id in list(eval_box.options)
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
    dl = [b.label for b in at.download_button]
    assert any("predictions CSV" in lab for lab in dl)
    assert any("alerts CSV" in lab for lab in dl)
    captions = "\n".join(str(w.value) for w in at.caption)
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

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.session_state["workflow_dataset"] = "filters"
    at.session_state["workflow_step"] = "Results"
    at.session_state["screen_selection"] = "Model Report"
    at.run()
    assert not at.exception
    at.session_state["report_view"] = "Historical RUL — Evaluation settings"
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

    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.session_state["report_view"] = "Historical RUL — Evaluation settings"
    _screen_radio(at).set_value("Model Report")
    at.run()
    assert not at.exception
    errors = "\n".join(str(w.value) for w in at.error)
    assert "Play is blocked" in errors
    infos = "\n".join(str(w.value) for w in at.info)
    assert "No evaluation yet" in infos or "Evaluate validation set" in infos
    play = next(b for b in at.button if b.label == "Play")
    assert play.disabled


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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.session_state["report_view"] = "Historical RUL — Evaluation settings"
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    at.run()
    assert not at.exception
    at.session_state["report_view"] = "Historical RUL — Evaluation settings"
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
    at.session_state["report_view"] = "Historical RUL — Evaluation settings"
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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


def test_future_red_matrix_worker_runs_selected_dataset_and_models(monkeypatch, tmp_path):
    from scripts import run_future_red_matrix

    from pdm.worker import read_status, run_job

    worker_dir = tmp_path / "worker"
    worker_dir.mkdir()
    output_dir = tmp_path / "matrix_run"
    output_dir.mkdir()
    (output_dir / "status.json").write_text(
        json.dumps({name: {"status": "completed"} for name in ("gru", "fly")}),
        encoding="utf-8",
    )
    captured = {}

    def fake_run(args):
        captured["dataset"] = args.datasets
        captured["architectures"] = args.architectures
        return output_dir

    monkeypatch.setattr("pdm.worker.worker_dir", lambda: worker_dir)
    monkeypatch.setattr(run_future_red_matrix, "run", fake_run)

    run_job({"kind": "future_red_matrix", "dataset_id": "filters", "architectures": ["gru", "fly"]})

    assert captured == {"dataset": "filters", "architectures": "gru,fly"}
    status = read_status()
    assert status["status"] == "completed"
    assert status["kind"] == "future_red_matrix"
    assert status["dataset_id"] == "filters"
    assert status["run_id"] == output_dir.name
    assert status["architectures"] == ["gru", "fly"]


def test_browser_import_persists_both_required_filter_files(monkeypatch, tmp_path):
    from io import BytesIO

    from pdm.app import _store_upload

    monkeypatch.setattr("pdm.paths.data_raw", lambda: tmp_path)
    files = []
    for name in ("Train_Data_CSV.csv", "Test_Data_CSV.csv"):
        stream = BytesIO((name + "\n").encode())
        stream.name = name
        files.append(stream)

    source = _store_upload("filters", files)

    assert source is not None
    assert {path.name for path in Path(source).iterdir()} == {
        "Train_Data_CSV.csv", "Test_Data_CSV.csv",
    }
    assert (Path(source) / "Train_Data_CSV.csv").read_bytes() == b"Train_Data_CSV.csv\n"


def test_app_operational_explorer_by_label_no_exception(tmp_path, monkeypatch, tiny_bearing_tables):
    """The one-flow explorer stays selectable without legacy comparison/mode widgets."""
    from streamlit.testing.v1 import AppTest

    from pdm.paths import project_root
    from pdm.visualization.explorer import clear_soma_table_cache

    _isolate_legacy_navigation(monkeypatch, tmp_path, "bearings", tiny_bearing_tables)
    monkeypatch.setattr("pdm.connectome.anatomy.default_soma_dir", lambda: tmp_path)
    clear_soma_table_cache()
    at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
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
    assert "Future-red entry model report — Bearings" in text
    assert "Architecture comparison" not in text
    report_view = next(widget for widget in at.radio if widget.label == "Report view")
    assert "Future-red entry" in report_view.options
    assert "Historical RUL — Condition & Forecast" in report_view.options
    assert "Historical RUL — Model replay" in report_view.options
    assert "Historical RUL — Evaluation settings" in report_view.options
    assert "Historical RUL — Experimental / Model activity" in report_view.options
    assert "Health zones" in report_view.options
    assert not any(widget.label == "Mode" for widget in at.radio)
    assert not any(widget.label == "Build trace" for widget in at.button)
    legacy_at = AppTest.from_file(str(project_root() / "tests" / "legacy_app_harness.py"), default_timeout=15)
    legacy_at.session_state["screen_selection"] = "Model Report"
    legacy_at.session_state["report_view"] = "Evaluation settings"
    legacy_at.run()
    assert not legacy_at.exception
    assert next(widget for widget in legacy_at.radio if widget.label == "Report view").value == (
        "Historical RUL — Evaluation settings"
    )
