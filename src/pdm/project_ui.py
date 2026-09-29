"""Persistent Create ML workflow for local sensor projects."""
from __future__ import annotations

import re
import shutil
import uuid
from pathlib import PurePosixPath

import streamlit as st

from pdm.cli import spawn_worker
from pdm.data.prepare import processed_ready
from pdm.data.project_prepare import load_snapshot, prepare_project
from pdm.project_quality_ui import QUALITY_DESCRIPTION, render_quality
from pdm.project_results_ui import render_results
from pdm.project_training_ui import render_training
from pdm.projects import project_store
from pdm.signal_training import list_project_runs
from pdm.ui_copy import (
    CREATE_PROJECT_HELP,
    IMPORT_FOLDER_HELP,
    IMPORT_RED_CONDITION_HELP,
    IMPORT_RED_LIMIT_HELP,
    IMPORT_SIGNAL_COLUMN_HELP,
    IMPORT_SIGNAL_NAME_HELP,
    IMPORT_SIGNAL_UNIT_HELP,
    IMPORT_SOURCE_MODE_HELP,
    IMPORT_SPLIT_SEED_HELP,
    IMPORT_SPLIT_WEIGHTS_CAPTION,
    IMPORT_SUBMIT_HELP,
    IMPORT_TEST_MODE_HELP,
    IMPORT_TEST_WEIGHT_HELP,
    IMPORT_TRAIN_WEIGHT_HELP,
    IMPORT_VALIDATION_MODE_HELP,
    IMPORT_VALIDATION_WEIGHT_HELP,
    IMPORT_YELLOW_LIMIT_HELP,
    PROJECT_NAME_HELP,
    PROJECT_SOURCE_FORMAT_HELP,
    QUALITY_CONTINUE_HELP,
)
from pdm.ui_theme import empty_state, page_header, render_theme_control
from pdm.worker import read_status, status_for_project, worker_alive

STEPS = ("Projects", "Import data", "Data Quality", "Training", "Results")
SOURCE_KINDS = {
    "generic_sensor_csv": "Sensor CSV",
    "xjtu_bearings": "XJTU-SY bearings",
    "hse_filters": "HSE filters",
}


def _reset_project_session() -> None:
    for key in list(st.session_state):
        if str(key).startswith(("project_play:", "play_slider:", "play_toggle:", "play_reset:",
                                 "result_run:", "result_unit:", "quality_unit_", "folder:", "path:",
                                 "source_mode:", "validation_mode", "test_mode")):
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


def _source_widgets(store, project: dict) -> tuple[dict | None, dict | None, dict | None, str, str]:
    pid = project["project_id"]
    kind = project["source_kind"]
    with st.container(border=True, key="pdm-import-training"):
        st.subheader("Training source", anchor=False)
        st.caption(f"Source: {SOURCE_KINDS.get(kind, kind)} · Validation and Testing can each be automatic or a separate folder.")
        primary = _source_input(store, pid, "Training", f"{pid}:primary")
    with st.container(border=True, key="pdm-import-holdout"):
        st.subheader("Validation and Testing", anchor=False)
        c1, c2 = st.columns(2)
        with c1:
            val_mode = st.radio("Validation", ["Automatic holdout", "Separate folder"], key="validation_mode",
                                help=IMPORT_VALIDATION_MODE_HELP)
            validation = _source_input(store, pid, "Validation", f"{pid}:validation") if val_mode == "Separate folder" else None
        with c2:
            test_mode = st.radio("Testing", ["Automatic holdout", "Separate folder"], key="test_mode",
                                 help=IMPORT_TEST_MODE_HELP)
            test = _source_input(store, pid, "Testing", f"{pid}:testing") if test_mode == "Separate folder" else None
    return primary, validation, test, "folder" if val_mode == "Separate folder" else "auto", "folder" if test_mode == "Separate folder" else "auto"


def _valid_thresholds(direction: str, yellow: float, red: float) -> None:
    if direction == "above" and yellow >= red:
        raise ValueError("For an increasing warning signal, the yellow limit must be below red.")
    if direction == "below" and yellow <= red:
        raise ValueError("For a decreasing warning signal, the yellow limit must be above red.")


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
    primary, validation, test, val_mode, test_mode = _source_widgets(store, project)
    with st.expander("Automatic split weights · 70 / 15 / 15", expanded=False):
        st.caption(IMPORT_SPLIT_WEIGHTS_CAPTION)
        c1, c2, c3 = st.columns(3)
        train_pct = c1.number_input("Train weight (%)", min_value=1, max_value=98, value=70,
                                    help=IMPORT_TRAIN_WEIGHT_HELP)
        val_pct = c2.number_input("Validation weight (%)", min_value=1, max_value=98, value=15,
                                  help=IMPORT_VALIDATION_WEIGHT_HELP)
        test_pct = c3.number_input("Test weight (%)", min_value=1, max_value=98, value=15,
                                   help=IMPORT_TEST_WEIGHT_HELP)
        seed = st.number_input("Split seed", min_value=0, max_value=2**31 - 1, value=42,
                               help=IMPORT_SPLIT_SEED_HELP)
    with st.container(border=True, key="pdm-import-signal"):
        st.subheader("Signal and limits", anchor=False)
        if kind == "generic_sensor_csv":
            st.caption("Each CSV needs unit_id, timestamp_s, and the selected numeric signal. Time is in seconds. Other columns are descriptive in this first version.")
            s1, s2, s3 = st.columns(3)
            signal_column = s1.text_input("Signal column", value="signal", help=IMPORT_SIGNAL_COLUMN_HELP)
            signal_label = s2.text_input("Signal name", value="Signal", help=IMPORT_SIGNAL_NAME_HELP)
            signal_unit = s3.text_input("Signal unit", value="unit", help=IMPORT_SIGNAL_UNIT_HELP)
        elif kind == "xjtu_bearings":
            signal_column, signal_label, signal_unit = "combined_rms", "Combined max-axis RMS", "g"
            st.caption("XJTU-SY vibration fragments produce max-axis RMS acceleration in g. Acquisition time is recorded in seconds.")
        else:
            signal_column, signal_label, signal_unit = "differential_pressure", "Differential pressure", "Pa"
            st.caption("HSE differential pressure is measured in Pa and source Time is seconds. Official test RUL stays for evaluation only.")
        direction_label = st.radio("Red condition", ["Signal rises above limits", "Signal falls below limits"], horizontal=True,
                                   help=IMPORT_RED_CONDITION_HELP)
        direction = "above" if direction_label.startswith("Signal rises") else "below"
        c1, c2 = st.columns(2)
        yellow = c1.number_input(f"Yellow limit ({signal_unit})", value=1.0 if direction == "above" else -1.0, format="%.6f",
                                 help=IMPORT_YELLOW_LIMIT_HELP)
        red = c2.number_input(f"Red limit ({signal_unit})", value=2.0 if direction == "above" else -2.0, format="%.6f",
                              help=IMPORT_RED_LIMIT_HELP)
        st.caption("Limits are instantaneous in the signal's native unit. Set these from your operating rules; the app does not infer fault limits.")
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
            _valid_thresholds(direction, float(yellow), float(red))
            primary = _materialize_source(store, pid, primary, created_roots)
            validation = _materialize_source(store, pid, validation, created_roots)
            test = _materialize_source(store, pid, test, created_roots)
            source = {"primary": primary, "validation_mode": val_mode, "validation": validation,
                      "test_mode": test_mode, "test": test, "seed": int(seed),
                      "weights": {"train": float(train_pct) / 100, "validation": float(val_pct) / 100,
                                  "test": float(test_pct) / 100},
                      "signal_column": signal_column.strip(), "signal_label": signal_label.strip(),
                      "signal_unit": signal_unit.strip(),
                      "thresholds": {"mode": "absolute", "direction": direction,
                                     "yellow": float(yellow), "red": float(red)}}
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
                                  format_func=lambda pid: "Choose a project" if pid is None else by_id[pid]["name"])
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
                ready = render_quality(snapshot, theme)
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
