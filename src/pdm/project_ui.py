"""Persistent Create ML workflow for local sensor projects."""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import PurePosixPath

import streamlit as st

from pdm.cli import spawn_worker
from pdm.data.prepare import processed_ready
from pdm.data.project_import import XJTU_BASELINE_THRESHOLDS, validate_thresholds
from pdm.data.project_prepare import (
    MANUAL_SPLIT_PROTOCOL,
    load_snapshot,
    prepare_project,
    read_zone_limits,
)
from pdm.io_util import read_json
from pdm.project_quality_ui import (
    PARTS,
    QUALITY_DESCRIPTION,
    part_summary,
    render_quality,
)
from pdm.project_results_ui import render_results
from pdm.project_training_ui import render_training
from pdm.projects import project_store
from pdm.signal_training import list_project_runs
from pdm.ui_copy import (
    CREATE_PROJECT_HELP,
    IMPORT_CARD_EMPTY,
    IMPORT_FOLDER_HELP,
    IMPORT_SIGNAL_COLUMN_HELP,
    IMPORT_SIGNAL_NAME_HELP,
    IMPORT_SIGNAL_UNIT_HELP,
    IMPORT_SOURCE_MODE_HELP,
    IMPORT_SPLIT_SEED_HELP,
    IMPORT_SPLIT_SETTINGS_HELP,
    IMPORT_SPLIT_WEIGHTS_CAPTION,
    IMPORT_SUBMIT_HELP,
    IMPORT_TEST_FROM_HELP,
    IMPORT_TEST_WEIGHT_HELP,
    IMPORT_TRAIN_WEIGHT_HELP,
    IMPORT_VALIDATION_FROM_HELP,
    IMPORT_VALIDATION_WEIGHT_HELP,
    IMPORT_VIEW_HELP,
    PROJECT_NAME_HELP,
    PROJECT_SOURCE_FORMAT_HELP,
    QUALITY_ADMITTED_ROWS_HELP,
    QUALITY_CONTINUE_HELP,
    QUALITY_GAPS_HELP,
    QUALITY_UNITS_HELP,
)
from pdm.ui_theme import empty_state, page_header, render_theme_control
from pdm.worker import read_status, status_for_project, worker_alive

STEPS = ("Projects", "Import data", "Data Quality", "Training", "Results")
SOURCE_KINDS = {
    "generic_sensor_csv": "Sensor CSV",
    "xjtu_bearings": "XJTU-SY bearings",
    "hse_filters": "HSE filters",
}
HOLDOUT_SOURCES = {"Split from training": "auto", "Separate folder": "folder"}
SPLIT_DEFAULTS = {"import_weight_train": 70, "import_weight_validation": 15, "import_weight_test": 15,
                  "import_seed": 42}
FIRST_IMPORT_THRESHOLDS = {
    "generic_sensor_csv": {},
    "hse_filters": {"mode": "absolute", "direction": "above", "yellow": 300.0, "red": 600.0},
    "xjtu_bearings": XJTU_BASELINE_THRESHOLDS,
}


def _reset_project_session() -> None:
    for key in list(st.session_state):
        if str(key).startswith(("project_play:", "play_slider:", "play_toggle:", "play_reset:",
                                 "result_run:", "result_unit:", "quality_unit_", "quality_move", "folder:", "path:",
                                 "source_mode:", "validation_mode", "test_mode", "quality_tab",
                                 "import_weight_", "import_seed", "import_split_settings", "import_view:")):
            st.session_state.pop(key, None)
    st.session_state.pop("project_last_forecast", None)
    st.session_state.pop("result_active_pair", None)
    st.session_state["project_step"] = "Projects"


def _new_name(projects: list[dict]) -> str:
    existing = {str(project.get("name", "")).casefold() for project in projects}
    number = 1
    while f"new project {number}" in existing:
        number += 1
    return f"New project {number}"


def _open_step(project: dict) -> str:
    if project.get("state") == "ready" and project.get("active_snapshot_id"):
        return "Data Quality"
    if project.get("storage_mode") == "linked_legacy":
        manifest = project.get("source_manifest") or {}
        legacy_dataset = manifest.get("legacy_dataset_id")
        if legacy_dataset and processed_ready(legacy_dataset):
            return "Data Quality"
    return "Import data"


def _stage_uploads(store, project_id: str, files, group: str,
                   created_roots: list | None = None) -> dict:
    if not files:
        raise ValueError(f"Choose a {group} folder.")
    project_root = store.project_path(project_id)
    uploads_root = project_root / "uploads"
    if uploads_root.is_symlink():
        raise ValueError("Project upload path is unsafe.")
    base = uploads_root / uuid.uuid4().hex / group
    base.mkdir(parents=True, exist_ok=False)
    if created_roots is not None:
        created_roots.append(base.parent)
    if not base.resolve().is_relative_to(project_root.resolve()):
        raise ValueError("Project upload path is unsafe.")
    staged = []
    seen = set()
    try:
        for uploaded in files:
            raw = str(uploaded.name).replace("\\", "/")
            relative = PurePosixPath(raw)
            if (relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts)
                    or ":" in relative.parts[0] or any("\x00" in part for part in relative.parts)):
                raise ValueError("Folder contains an unsafe file path.")
            clean = str(relative)
            if clean.casefold() in seen:
                raise ValueError(f"Folder contains a duplicate path: {clean}")
            seen.add(clean.casefold())
            target = base.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            uploaded.seek(0)
            with target.open("xb") as output:
                while chunk := uploaded.read(1024 * 1024):
                    output.write(chunk)
            staged.append({"path": str(target), "relative_path": clean})
    except Exception:
        shutil.rmtree(base.parent)
        raise
    return {"mode": "files", "files": staged}


def _source_input(store, project_id: str, label: str, key: str) -> dict | None:
    mode = st.radio(f"{label} source", ["Choose folder", "Server folder path"], horizontal=True, key=f"source_mode:{key}",
                    help=IMPORT_SOURCE_MODE_HELP)
    if mode == "Choose folder":
        files = st.file_uploader(f"{label} folder", accept_multiple_files="directory", key=f"folder:{key}",
                                 help=IMPORT_FOLDER_HELP)
        return {"mode": "uploads", "files": files, "group": key} if files else None
    path = st.text_input(f"{label} server folder path", key=f"path:{key}",
                         help="This path must be visible to the computer running the app.").strip()
    return {"mode": "folder", "path": path} if path else None


def _materialize_source(store, project_id: str, source: dict | None,
                        created_roots: list) -> dict | None:
    if source and source.get("mode") == "uploads":
        return _stage_uploads(store, project_id, source["files"], source["group"], created_roots)
    return source


def _auto_shares(weights: dict, val_mode: str, test_mode: str) -> dict[str, float]:
    automatic = ["train"] + [name for name, mode in (("validation", val_mode), ("test", test_mode)) if mode == "auto"]
    total = sum(float(weights[name]) for name in automatic)
    return {name: float(weights[name]) / total for name in automatic}


@st.cache_data(show_spinner=False)
def _snapshot_summaries(project_id: str, snapshot_id: str) -> dict:
    snapshot = load_snapshot(project_id, snapshot_id)
    parts = {}
    for part, _ in PARTS:
        summary = part_summary(snapshot["features"], snapshot["split"], part)
        parts[part] = {"units": summary["units"], "rows": summary["rows"], "gaps": summary["gaps"]}
    return {"parts": parts, "protocol": str(snapshot["split"].get("protocol") or "")}


def _open_quality_tab(name: str) -> None:
    st.session_state["quality_tab"] = name
    st.session_state["project_step"] = "Data Quality"


def _card_counts(part: str, name: str, summaries: dict | None) -> None:
    counts = (summaries or {}).get("parts", {}).get(part)
    with st.container(key=f"pdm-card-head-{part}"):
        st.subheader(name, anchor=False)
        if counts:
            st.button("View", key=f"import_view:{part}", type="tertiary", help=IMPORT_VIEW_HELP,
                      on_click=_open_quality_tab, args=(name,), width="content")
    with st.container(key=f"pdm-card-body-{part}"):
        if not counts:
            st.markdown("**No data yet**")
            st.caption(IMPORT_CARD_EMPTY)
            return
        units, gaps, rows = st.columns([5, 5, 8], gap="small")
        units.metric("Units", counts["units"], help=QUALITY_UNITS_HELP)
        gaps.metric("Gaps", counts["gaps"], help=QUALITY_GAPS_HELP)
        rows.metric("Admitted rows", counts["rows"], help=QUALITY_ADMITTED_ROWS_HELP)


def _holdout_source(label: str, key: str, help_text: str) -> str:
    if st.session_state.get(key) not in HOLDOUT_SOURCES:
        st.session_state.pop(key, None)
    return HOLDOUT_SOURCES[st.selectbox(f"{label} data from", list(HOLDOUT_SOURCES), key=key, help=help_text)]


def _import_cards(store, project: dict, summaries: dict | None
                  ) -> tuple[dict | None, dict | None, dict | None, str, str]:
    pid = project["project_id"]
    kind = project["source_kind"]
    names = dict(PARTS)
    weights = {part: st.session_state.get(f"import_weight_{part}", SPLIT_DEFAULTS[f"import_weight_{part}"])
               for part in names}
    columns = st.columns(3, gap="medium")
    with columns[0], st.container(border=True, key="pdm-import-card-train"):
        _card_counts("train", names["train"], summaries)
        with st.container(key="pdm-card-source-train"):
            st.caption(f"Source: {SOURCE_KINDS.get(kind, kind)}")
            primary = _source_input(store, pid, "Training", f"{pid}:primary")
    with columns[1], st.container(border=True, key="pdm-import-card-validation"):
        _card_counts("validation", names["validation"], summaries)
        with st.container(key="pdm-card-source-validation"):
            val_mode = _holdout_source("Validation", "validation_mode", IMPORT_VALIDATION_FROM_HELP)
            validation_slot = st.container()
    with columns[2], st.container(border=True, key="pdm-import-card-test"):
        _card_counts("test", names["test"], summaries)
        with st.container(key="pdm-card-source-test"):
            test_mode = _holdout_source("Testing", "test_mode", IMPORT_TEST_FROM_HELP)
            hse_test_auto = kind == "hse_filters" and test_mode == "auto"
            shares = _auto_shares(weights, val_mode, "folder" if hse_test_auto else test_mode)
            if hse_test_auto:
                st.caption("Official HSE test units (Test_Data_CSV.csv) in the Training folder stay in Testing.")
            elif test_mode == "auto":
                st.caption("Automatically split from training data")
                st.caption(f"{round(100 * shares['test'])}% of the training pool")
            test = _source_input(store, pid, "Testing", f"{pid}:testing") if test_mode == "folder" else None
    with validation_slot:
        if val_mode == "auto":
            st.caption("Automatically split from training data")
            st.caption(f"{round(100 * shares['validation'])}% of the training pool")
            validation = None
        else:
            validation = _source_input(store, pid, "Validation", f"{pid}:validation")
    return primary, validation, test, val_mode, test_mode


def _split_settings(any_auto: bool) -> tuple[int, int, int, int]:
    values = {key: st.session_state.get(key, default) for key, default in SPLIT_DEFAULTS.items()}
    disabled = not any_auto
    c1, c2 = st.columns([1, 4], vertical_alignment="center")
    c2.caption(f"{values['import_weight_train']} / {values['import_weight_validation']} / "
               f"{values['import_weight_test']} · seed {values['import_seed']}")
    if disabled:
        st.caption("Both holdouts use separate folders; split settings do not apply.")
    with c1.popover("Split settings", key="import_split_settings", help=IMPORT_SPLIT_SETTINGS_HELP):
        st.caption(IMPORT_SPLIT_WEIGHTS_CAPTION)
        train_pct = st.number_input("Train weight (%)", min_value=1, max_value=98, value=70, disabled=disabled,
                                    key="import_weight_train", help=IMPORT_TRAIN_WEIGHT_HELP)
        val_pct = st.number_input("Validation weight (%)", min_value=1, max_value=98, value=15, disabled=disabled,
                                  key="import_weight_validation", help=IMPORT_VALIDATION_WEIGHT_HELP)
        test_pct = st.number_input("Test weight (%)", min_value=1, max_value=98, value=15, disabled=disabled,
                                   key="import_weight_test", help=IMPORT_TEST_WEIGHT_HELP)
        seed = st.number_input("Split seed", min_value=0, max_value=2**31 - 1, value=42, disabled=disabled,
                               key="import_seed", help=IMPORT_SPLIT_SEED_HELP)
    return train_pct, val_pct, test_pct, seed


def _saved_schema(store, project: dict) -> dict:
    """Schema of the snapshot being replaced, with its saved display limits if any.

    Re-import defaults only; the worker re-validates what is sent.
    """
    sid = project.get("active_snapshot_id")
    if project.get("state") != "ready" or not sid:
        return {}
    try:
        directory = store.snapshot_path(project["project_id"], str(sid))
        path = directory / "feature_schema.json"
        schema = {} if path.is_symlink() else read_json(path)
    except (OSError, ValueError, KeyError):
        return {}
    if not isinstance(schema, dict):
        return {}
    limits = read_zone_limits(directory)
    return {**schema, "thresholds": limits} if limits else schema


def _import_thresholds(kind: str, saved: dict, signal_column: str, signal_unit: str) -> dict:
    """The replaced snapshot's rule for the same signal, else first-import defaults."""
    same_signal = bool(saved) and (kind != "generic_sensor_csv" or (
        saved.get("signal_column"), saved.get("signal_unit")) == (signal_column, signal_unit))
    if same_signal:
        try:
            return validate_thresholds(kind, saved.get("thresholds"))
        except ValueError:
            pass
    return dict(FIRST_IMPORT_THRESHOLDS.get(kind, {}))


@st.fragment(run_every=1.5)
def _import_status(project_id: str) -> None:
    status = status_for_project(project_id)
    if status.get("kind") != "project_import":
        return
    state = status.get("status")
    if state in {"queued", "running", "preparing", "stopping"}:
        st.info(str(status.get("message") or "Checking the selected files…"))
    elif state == "completed":
        st.success("Import and data checks complete.")
        if st.session_state.get(f"import_terminal:{project_id}") != status.get("job_id"):
            st.session_state[f"import_terminal:{project_id}"] = status.get("job_id")
            st.rerun(scope="app")
    elif state in {"failed", "cancelled"}:
        st.error(str(status.get("error") or status.get("message") or "Import did not complete."))
        previous = project_store().get(project_id)
        if previous.get("state") == "ready" and previous.get("active_snapshot_id"):
            st.caption("Your previous saved data and model runs remain available.")


def _render_import(store, project: dict) -> None:
    pid = project["project_id"]
    page_header("Import data", "Choose sensor files by folder. For very large data, enter a folder path on this computer.")
    kind = project["source_kind"]
    _import_status(pid)
    summaries, load_error = None, None
    if project.get("state") == "ready" and project.get("active_snapshot_id"):
        try:
            summaries = _snapshot_summaries(pid, str(project["active_snapshot_id"]))
        except Exception as exc:
            load_error = str(exc)
    if summaries:
        st.caption("Counts are the saved snapshot.")
    primary, validation, test, val_mode, test_mode = _import_cards(store, project, summaries)
    if load_error:
        st.caption(f"Saved data could not be read: {load_error}")
    if summaries and summaries.get("protocol") == MANUAL_SPLIT_PROTOCOL:
        st.caption("Current sets include manual moves from Data Quality. Importing again creates a fresh split.")
    train_pct, val_pct, test_pct, seed = _split_settings("auto" in {val_mode, test_mode})
    saved = _saved_schema(store, project)
    with st.container(border=True, key="pdm-import-signal"):
        st.subheader("Signal", anchor=False)
        if kind == "generic_sensor_csv":
            st.caption("Each CSV needs unit_id, timestamp_s, and the selected numeric signal. Time is in seconds. Other columns are descriptive in this first version.")
            s1, s2, s3 = st.columns(3)
            signal_column = s1.text_input("Signal column", value=str(saved.get("signal_column") or "signal"),
                                          help=IMPORT_SIGNAL_COLUMN_HELP)
            signal_label = s2.text_input("Signal name", value=str(saved.get("signal_label") or "Signal"),
                                         help=IMPORT_SIGNAL_NAME_HELP)
            signal_unit = s3.text_input("Signal unit", value=str(saved.get("signal_unit") or "unit"),
                                        help=IMPORT_SIGNAL_UNIT_HELP)
        elif kind == "xjtu_bearings":
            signal_column, signal_label, signal_unit = "combined_rms", "Combined max-axis RMS", "g"
            st.caption("XJTU-SY vibration fragments produce max-axis RMS acceleration in g. Acquisition time is recorded in seconds.")
        else:
            signal_column, signal_label, signal_unit = "differential_pressure", "Differential pressure", "Pa"
            st.caption("HSE differential pressure is measured in Pa and source Time is seconds. Official test RUL stays for evaluation only.")
        st.caption("Yellow and red limits are set on Data Quality after import.")
    status = status_for_project(pid)
    running = worker_alive() or status.get("status") in {"queued", "running", "training", "preparing", "stopping"}
    if running:
        global_status = read_status()
        detail = "this project" if global_status.get("project_id") == pid else "another project"
        st.info(f"A job is active for {detail}. Wait for it to finish before importing.")
    if st.button("Import and check data", type="primary", disabled=running, help=IMPORT_SUBMIT_HELP):
        created_roots = []
        job_id = None
        launched = False
        try:
            if not primary:
                raise ValueError("Choose a Training folder or provide its server path.")
            if val_mode == "folder" and not validation:
                raise ValueError("Choose the separate Validation folder.")
            if test_mode == "folder" and not test:
                raise ValueError("Choose the separate Testing folder.")
            if int(train_pct + val_pct + test_pct) != 100:
                raise ValueError("Split weights must add to 100%.")
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", signal_column.strip()):
                raise ValueError("Enter a valid signal column name.")
            if not signal_label.strip() or not signal_unit.strip():
                raise ValueError("Signal name and unit are required.")
            thresholds = _import_thresholds(kind, saved, signal_column.strip(), signal_unit.strip())
            primary = _materialize_source(store, pid, primary, created_roots)
            validation = _materialize_source(store, pid, validation, created_roots)
            test = _materialize_source(store, pid, test, created_roots)
            source = {"primary": primary, "validation_mode": val_mode, "validation": validation,
                      "test_mode": test_mode, "test": test, "seed": int(seed),
                      "weights": {"train": float(train_pct) / 100, "validation": float(val_pct) / 100,
                                  "test": float(test_pct) / 100},
                      "signal_column": signal_column.strip(), "signal_label": signal_label.strip(),
                      "signal_unit": signal_unit.strip(), "thresholds": thresholds}
            job_id = uuid.uuid4().hex
            spawn_worker({"kind": "project_import", "job_id": job_id, "project_id": pid, "source": source})
            launched = True
            st.session_state["project_step"] = "Data Quality"
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            own_job_started = bool(job_id and status_for_project(pid).get("job_id") == job_id)
            if not launched and not own_job_started:
                uploads_root = (store.project_path(pid) / "uploads").resolve()
                for root in created_roots:
                    if not root.is_symlink() and root.exists() and root.resolve().is_relative_to(uploads_root):
                        shutil.rmtree(root)
            st.error(f"Import could not start: {exc}")


def _render_projects(store, projects: list[dict], selected: dict | None) -> None:
    page_header("Projects", "Create an independent project for each data source and model run.")
    with st.form("create_project"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Project name", value=_new_name(projects), help=PROJECT_NAME_HELP)
        kind = c2.selectbox("Source format", list(SOURCE_KINDS), format_func=lambda value: SOURCE_KINDS[value],
                            help=PROJECT_SOURCE_FORMAT_HELP)
        submitted = st.form_submit_button("Create project", type="primary", help=CREATE_PROJECT_HELP)
    if submitted:
        try:
            project = store.create(name.strip(), kind)
            _reset_project_session()
            st.session_state["project_id"] = project["project_id"]
            st.session_state["project_step"] = "Import data"
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Project could not be created: {exc}")
    if not projects:
        empty_state("Open a project", "No projects yet. Create one to import data.")
    else:
        st.subheader("Open a project", anchor=False)
        with st.container(border=True, key="pdm-project-list"):
            for project in projects:
                pid = project["project_id"]
                with st.container(key=f"pdm-project-row-{pid}"):
                    c1, c2 = st.columns([6, 1], vertical_alignment="center")
                    c1.write(f"**{project['name']}** · {SOURCE_KINDS.get(project['source_kind'], project['source_kind'])}")
                    if c2.button("Open", key=f"open:{pid}", width="stretch"):
                        st.session_state["project_id"] = pid
                        st.session_state["project_step"] = _open_step(project)
                        st.rerun()
    if selected:
        with st.expander(f"Delete {selected['name']}"):
            st.caption("Delete moves this project's owned files to a recoverable archive. Linked source files and historical research runs stay in place.")
            confirmed = st.checkbox("I want to archive this project", key=f"archive_confirm:{selected['project_id']}")
            if st.button("Delete project", disabled=not confirmed, key=f"archive:{selected['project_id']}"):
                try:
                    receipt = store.archive(selected["project_id"])
                    st.session_state["archive_receipt"] = receipt
                    st.session_state.pop("project_id", None)
                    _reset_project_session()
                    st.rerun()
                except (OSError, ValueError, RuntimeError) as exc:
                    st.error(f"Project could not be archived: {exc}")
    receipt = st.session_state.pop("archive_receipt", None)
    if receipt:
        st.success(f"Project archived. Recovery receipt: {receipt.get('archive_id', 'saved')}")


def _maybe_wrap_legacy(project: dict) -> dict:
    if project.get("active_snapshot_id") or project.get("storage_mode") != "linked_legacy":
        return project
    source = project.get("source_manifest") or {}
    legacy_dataset = source.get("legacy_dataset_id") if isinstance(source, dict) else None
    if not legacy_dataset or not processed_ready(legacy_dataset):
        return project
    manifest_id = source.get("manifest_id") if isinstance(source, dict) else None
    if not manifest_id:
        manifest_id = "legacy-bearings" if project.get("source_kind") == "xjtu_bearings" else "legacy-filters"
    with st.spinner("Opening existing prepared data"):
        prepare_project(project["project_id"], manifest_id)
    return project_store().get(project["project_id"])


def main() -> None:
    from pdm.visualization.presentation import apply_explorer_style

    st.markdown('<span class="pdm-project-workflow" aria-hidden="true"></span>', unsafe_allow_html=True)
    store = project_store()
    store.register_legacy()
    projects = store.list()
    by_id = {project["project_id"]: project for project in projects}
    selected_id = st.session_state.get("project_id")
    if selected_id not in by_id:
        if selected_id:
            _reset_project_session()
        selected_id = None
        st.session_state.pop("project_id", None)
    selected = by_id.get(selected_id)
    current_status = status_for_project(selected_id) if selected_id else {}
    importing_current = (current_status.get("kind") == "project_import"
                         and current_status.get("status") in {"queued", "running", "preparing", "stopping"})
    with st.sidebar:
        st.caption("APPEARANCE")
        theme = render_theme_control(st.sidebar)
    apply_explorer_style(theme)
    with st.sidebar:
        st.caption("PROJECT")
        if projects:
            options = [None, *by_id]
            chosen = st.selectbox("Project", options, index=options.index(selected_id),
                                  format_func=lambda pid: "Choose a project" if pid is None else by_id[pid]["name"],
                                  label_visibility="collapsed")
            if chosen != selected_id:
                _reset_project_session()
                if chosen:
                    st.session_state["project_id"] = chosen
                else:
                    st.session_state.pop("project_id", None)
                st.rerun()
        else:
            st.caption("No projects yet")
        st.caption("WORKFLOW")
        step = st.session_state.get("project_step", "Projects")
        snapshot_ready = bool(selected and selected.get("state") == "ready" and selected.get("active_snapshot_id"))
        runs_ready = False
        if snapshot_ready:
            try:
                runs_ready = any(row.get("run_id") == selected.get("selected_run_id")
                                 and row.get("project_id") == selected_id
                                 and row.get("snapshot_id") == selected["active_snapshot_id"]
                                 and row.get("task") == "signal_forecast"
                                 and row.get("status") == "completed"
                                 for row in list_project_runs(selected_id))
            except (OSError, ValueError, KeyError, RuntimeError):
                runs_ready = False
        with st.container(key="pdm-workflow-rail"):
            for name in STEPS:
                enabled = name == "Projects" or bool(selected) and (
                    name in {"Import data", "Data Quality"} or
                    name == "Training" and snapshot_ready and not importing_current or
                    name == "Results" and runs_ready and not importing_current)
                if st.button(name, key=f"project_nav:{name}", disabled=not enabled,
                             type="primary" if step == name else "secondary", width="stretch"):
                    st.session_state["project_step"] = name
                    st.rerun()
    if selected and step in {"Data Quality", "Training", "Results"}:
        try:
            selected = _maybe_wrap_legacy(selected)
        except (OSError, ValueError, RuntimeError) as exc:
            st.warning(f"Existing prepared data could not be opened: {exc}")
    if step == "Projects" or selected is None:
        _render_projects(store, projects, selected)
    elif step == "Import data":
        _render_import(store, selected)
    elif step == "Data Quality":
        if selected.get("state") == "ready" and selected.get("active_snapshot_id"):
            try:
                if current_status.get("kind") == "project_import":
                    _import_status(selected_id)
                    if importing_current:
                        st.caption("Showing the previous saved data while the replacement import is checked.")
                snapshot = load_snapshot(selected_id)
                ready = render_quality(snapshot, theme, storage_mode=selected.get("storage_mode", "owned"))
                if ready and st.button("Continue to Training", type="primary", disabled=importing_current,
                                       help=QUALITY_CONTINUE_HELP):
                    st.session_state["project_step"] = "Training"
                    st.rerun()
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                st.error(f"Prepared data could not be opened: {exc}")
        else:
            page_header("Data Quality", QUALITY_DESCRIPTION)
            _import_status(selected_id)
            if selected.get("storage_mode") == "linked_legacy":
                st.info("This linked source has no prepared sensor table yet. Prepare the legacy dataset, then reopen this project, or create a new project to import its folder.")
            elif status_for_project(selected_id).get("status") == "not_ready":
                empty_state("No data yet", "Import data to inspect Train, Validation, and Test.")
    elif step == "Training" and selected.get("state") == "ready" and not importing_current:
        try:
            render_training(selected_id, load_snapshot(selected_id))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Training data could not be opened: {exc}")
    elif step == "Results" and selected.get("state") == "ready" and not importing_current:
        try:
            render_results(selected_id, load_snapshot(selected_id), selected.get("selected_run_id"), theme)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            st.error(f"Results could not be opened: {exc}")
    else:
        empty_state("Not ready yet", "Finish importing and checking the current source before training or opening results.")
