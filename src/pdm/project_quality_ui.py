"""Plain-language quality view for one immutable project snapshot."""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm import project_zones
from pdm.data.project_prepare import (
    load_zone_limits,
    move_units,
    preview_move,
    save_zone_limits,
)
from pdm.project_chart_style import add_threshold_layers, style_signal_chart
from pdm.signal_training import available_signal_engines
from pdm.ui_copy import (
    IMPORT_RED_CONDITION_HELP,
    IMPORT_RED_LIMIT_HELP,
    IMPORT_YELLOW_LIMIT_HELP,
    QUALITY_ADMITTED_ROWS_HELP,
    QUALITY_GAPS_HELP,
    QUALITY_INSPECT_UNIT_HELP,
    QUALITY_MOVE_DONE,
    QUALITY_MOVE_FIXED_HSE,
    QUALITY_MOVE_JOB_ACTIVE,
    QUALITY_MOVE_LEGACY,
    QUALITY_MOVE_PREVIEW,
    QUALITY_MOVE_SUBMIT_HELP,
    QUALITY_MOVE_TEST_OPTIMISM,
    QUALITY_MOVE_TO_HELP,
    QUALITY_MOVE_UNITS_HELP,
    QUALITY_NO_ZONES,
    QUALITY_TABS_CAPTION,
    QUALITY_UNITS_HELP,
    QUALITY_ZONE_MODEL_CAPTION,
    QUALITY_ZONE_SUMMARY_HELP,
)
from pdm.ui_theme import page_header, tokens, zone_colors
from pdm.worker import heavy_job_active

QUALITY_DESCRIPTION = "Each set contains whole physical units. Rows from one unit stay in one set."
PARTS = (("train", "Training Data"), ("validation", "Validation Data"), ("test", "Testing Data"))
ZONE_NAMES = {"green": "Green", "yellow": "Yellow", "red": "Red", "unknown": "Not zoned"}
PART_BY_NAME = {name: part for part, name in PARTS}
MOVE_FLASH_KEY = "quality_move_flash"
DIRECTIONS = {"above": "Above", "below": "Below"}
DEFAULT_LIMITS = {"hse_filters": (300.0, 600.0)}


def limits_key(project_id: str, snapshot_id: str) -> str:
    """Session key for the last valid edited absolute rule of one snapshot."""
    return f"quality_limits:{project_id}:{snapshot_id}"


def _state_key(name: str, project_id: str, snapshot_id: str) -> str:
    return f"quality_limit_{name}:{project_id}:{snapshot_id}"


def valid_thresholds(direction: str, yellow: float, red: float) -> None:
    if direction == "above" and yellow >= red:
        raise ValueError("For an increasing warning signal, the yellow limit must be below red.")
    if direction == "below" and yellow <= red:
        raise ValueError("For a decreasing warning signal, the yellow limit must be above red.")


@st.cache_data(show_spinner=False, max_entries=64)
def split_zone_counts(project_id: str, snapshot_id: str, part: str, rule_key: str,
                      _features: pd.DataFrame, _unit_ids: tuple[str, ...], _schema: dict) -> dict[str, int]:
    """Snapshot rows are immutable; ``rule_key`` covers edited or saved display limits."""
    return project_zones.zone_counts(_features, _unit_ids, _schema)


def zone_schema(schema: dict, project_id: str, snapshot_id: str, saved_limits: dict | None = None) -> dict:
    """Schema whose thresholds are this session's edit, else the saved display limits, else the import rule."""
    rule = st.session_state.get(limits_key(project_id, snapshot_id)) or saved_limits
    return {**schema, "thresholds": dict(rule)} if rule else schema


def _widget_rule(project_id: str, snapshot_id: str) -> dict:
    """Absolute rule from the limit widgets; raises ``ValueError`` with the user-facing message."""
    state = st.session_state
    try:
        raw = state[_state_key("direction", project_id, snapshot_id)]
        direction = raw if raw in DIRECTIONS else ("above" if str(raw).startswith("Signal rises") else "below")
        yellow = float(state[_state_key("yellow", project_id, snapshot_id)])
        red = float(state[_state_key("red", project_id, snapshot_id)])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Enter numeric yellow and red limits.") from exc
    if not (math.isfinite(yellow) and math.isfinite(red)):
        raise ValueError("Yellow and red limits must be finite numbers.")
    valid_thresholds(direction, yellow, red)
    return {"mode": "absolute", "direction": direction, "yellow": yellow, "red": red}


def committed_rule(schema: dict, saved_limits: dict | None) -> dict | None:
    """Last committed absolute display rule: the sidecar, else a valid absolute import rule."""
    if saved_limits:
        return saved_limits
    imported = dict(schema.get("thresholds") or {})
    if imported.get("mode", "absolute") == "absolute" and project_zones.has_valid_rule(schema) \
            and imported.get("yellow") is not None and imported.get("red") is not None:
        return imported
    return None


def _seed(schema: dict, committed: dict | None) -> tuple[str, float, float]:
    if committed:
        direction = committed.get("direction", "above")
        return (direction if direction in DIRECTIONS else "above",
                float(committed["yellow"]), float(committed["red"]))
    yellow, red = DEFAULT_LIMITS.get(str(schema.get("source_kind") or ""), (1.0, 2.0))
    return "above", float(yellow), float(red)


def _same_limits(a: tuple[str, float, float], b: tuple[str, float, float]) -> bool:
    return a[0] == b[0] and math.isclose(a[1], b[1], abs_tol=1e-12) and math.isclose(a[2], b[2], abs_tol=1e-12)


def _on_limits_change(project_id: str, snapshot_id: str, committed: dict | None) -> None:
    state = st.session_state
    error_key = _state_key("error", project_id, snapshot_id)
    state.pop(_state_key("status", project_id, snapshot_id), None)
    try:
        rule = _widget_rule(project_id, snapshot_id)
    except ValueError as exc:
        state[error_key] = str(exc)
        return
    state.pop(error_key, None)
    values = (rule["direction"], rule["yellow"], rule["red"])
    if committed and _same_limits(values, _seed({}, committed)):
        state.pop(limits_key(project_id, snapshot_id), None)
    else:
        state[limits_key(project_id, snapshot_id)] = rule


def _on_limits_save(project_id: str, snapshot_id: str) -> None:
    state = st.session_state
    status_key = _state_key("status", project_id, snapshot_id)
    try:
        rule = _widget_rule(project_id, snapshot_id)
    except ValueError as exc:
        state[_state_key("error", project_id, snapshot_id)] = str(exc)
        return
    try:
        save_zone_limits(project_id, rule, expected_snapshot_id=snapshot_id)
    except (KeyError, ValueError, RuntimeError, OSError) as exc:
        state[status_key] = str(exc)
        return
    for key in (limits_key(project_id, snapshot_id), status_key, _state_key("error", project_id, snapshot_id)):
        state.pop(key, None)


def _on_limits_cancel(project_id: str, snapshot_id: str, schema: dict) -> None:
    """Widget keys must be overwritten here: the render-time seed only fills missing keys."""
    state = st.session_state
    direction, yellow, red = _seed(schema, committed_rule(schema, load_zone_limits(project_id, snapshot_id)))
    state[_state_key("direction", project_id, snapshot_id)] = direction
    state[_state_key("yellow", project_id, snapshot_id)] = yellow
    state[_state_key("red", project_id, snapshot_id)] = red
    for name in ("error", "status"):
        state.pop(_state_key(name, project_id, snapshot_id), None)
    state.pop(limits_key(project_id, snapshot_id), None)


def _render_limits(schema: dict, project_id: str, snapshot_id: str,
                   saved_limits: dict | None, job_running: bool) -> None:
    state = st.session_state
    unit = str(schema.get("signal_unit") or "")
    keys = {name: _state_key(name, project_id, snapshot_id) for name in ("direction", "yellow", "red")}
    edited = state.get(limits_key(project_id, snapshot_id))
    committed = committed_rule(schema, saved_limits)
    seed = _seed(schema, committed)
    if any(key not in state for key in keys.values()):
        direction, yellow, red = _seed(schema, edited) if edited else seed
        state[keys["direction"]] = direction
        state[keys["yellow"]], state[keys["red"]] = yellow, red
        state.pop(_state_key("error", project_id, snapshot_id), None)
    args = (project_id, snapshot_id, committed)
    suffix = f" ({unit})" if unit else ""
    with st.container(border=True, key="pdm-quality-limits"):
        st.radio("Limits", list(DIRECTIONS), format_func=DIRECTIONS.get, horizontal=True,
                 key=keys["direction"], help=IMPORT_RED_CONDITION_HELP, on_change=_on_limits_change, args=args)
        st.number_input(f"Yellow{suffix}", format="%.2f", step=0.01, key=keys["yellow"],
                        help=IMPORT_YELLOW_LIMIT_HELP, on_change=_on_limits_change, args=args)
        st.number_input(f"Red{suffix}", format="%.2f", step=0.01, key=keys["red"],
                        help=IMPORT_RED_LIMIT_HELP, on_change=_on_limits_change, args=args)
        error = state.get(_state_key("error", project_id, snapshot_id))
        if error:
            st.error(error)
        status = state.get(_state_key("status", project_id, snapshot_id))
        if status:
            st.error(status)
        if job_running:
            st.caption("A background job is running. Save is available after it finishes.")
        try:
            current = (str(state[keys["direction"]]), float(state[keys["yellow"]]), float(state[keys["red"]]))
            dirty = not _same_limits(current, seed)
        except (KeyError, TypeError, ValueError):
            dirty = True
        dirty = dirty or edited is not None or bool(error)
        cancel_col, save_col = st.columns(2, gap="small")
        cancel_col.button("Cancel", key=f"quality_limit_cancel:{project_id}:{snapshot_id}", width="stretch",
                          disabled=not dirty, on_click=_on_limits_cancel, args=(project_id, snapshot_id, schema))
        save_col.button("Save", key=f"quality_limit_save:{project_id}:{snapshot_id}",
                        width="stretch", disabled=edited is None or bool(error) or job_running,
                        on_click=_on_limits_save, args=(project_id, snapshot_id))


def unit_thresholds(labelled: pd.DataFrame, schema: dict) -> dict:
    rule = schema.get("thresholds") or {}
    prefix = labelled
    if rule.get("mode") == "initial_baseline_multiple":
        prefix = labelled.iloc[:int(rule.get("baseline_n", 5))]
    return project_zones.resolve_thresholds(schema, prefix)


def zone_figure(labelled: pd.DataFrame, schema: dict, label: str, unit: str, theme: str) -> go.Figure:
    fig = go.Figure()
    x, y = gap_safe_trace(labelled)
    if not project_zones.has_valid_rule(schema):
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name=label,
                                 line={"color": tokens(theme)["series_observed"], "width": 2}, marker={"size": 4}))
        fig.update_layout(height=280, margin={"l": 20, "r": 20, "t": 15, "b": 25},
                          xaxis_title="Time (s)", yaxis_title=f"{label} ({unit})", showlegend=False)
        return style_signal_chart(fig, theme)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, showlegend=False, hoverinfo="skip",
                             line={"color": tokens(theme)["series_observed"], "width": 2}))
    colors = zone_colors(theme)
    suffix = f" {unit}" if unit else ""
    for zone in project_zones.ZONES:
        rows = labelled.loc[labelled["zone"] == zone]
        if rows.empty:
            continue
        fig.add_trace(go.Scatter(
            x=rows["timestamp_s"].tolist(), y=rows["signal"].tolist(), mode="markers",
            name=f"{ZONE_NAMES[zone]} · {len(rows)}", marker={"size": 6, "color": colors[zone]},
            hovertemplate=(f"Time %{{x:g}} s<br>{label} %{{y:g}}{suffix}"
                           f"<br>Zone: {ZONE_NAMES[zone]}<extra></extra>")))
    thresholds = unit_thresholds(labelled, schema)
    if thresholds.get("status") == "available":
        values = [float(v) for v in labelled["signal"] if np.isfinite(v)]
        add_threshold_layers(fig, thresholds, values, theme)
    fig.update_layout(height=300, margin={"l": 20, "r": 20, "t": 30, "b": 25},
                      xaxis_title="Time (s)", yaxis_title=f"{label} ({unit})",
                      showlegend=True, legend={"orientation": "h", "y": 1.12, "x": 0})
    return style_signal_chart(fig, theme)


def gap_safe_trace(frame: pd.DataFrame) -> tuple[list[float | None], list[float | None]]:
    x, y = [], []
    for row in frame.itertuples(index=False):
        if bool(getattr(row, "gap_before")) and x:
            x.append(None)
            y.append(None)
        x.append(float(getattr(row, "timestamp_s")))
        y.append(float(getattr(row, "signal")))
    return x, y


def part_summary(features: pd.DataFrame, split: dict, part: str) -> dict:
    ids = {str(uid) for uid in split.get(part) or []}
    rows = features.loc[features["unit_id"].astype(str).isin(ids)].sort_values(["unit_id", "timestamp_s"]).copy()
    signal = pd.to_numeric(rows.get("signal", pd.Series(dtype=float)), errors="coerce")
    time = pd.to_numeric(rows.get("timestamp_s", pd.Series(dtype=float)), errors="coerce")
    gap = rows.get("gap_before", pd.Series(False, index=rows.index)).fillna(False).astype(bool)
    first = ~rows["unit_id"].astype(str).duplicated()
    return {
        "units": len(ids),
        "rows": len(rows),
        "missing_signal": int((~np.isfinite(signal)).sum()),
        "gaps": int((gap & ~first).sum()),
        "time_start": float(time.min()) if len(time) and np.isfinite(time.min()) else None,
        "time_end": float(time.max()) if len(time) and np.isfinite(time.max()) else None,
        "signal_min": float(signal.min()) if len(signal) and np.isfinite(signal.min()) else None,
        "signal_max": float(signal.max()) if len(signal) and np.isfinite(signal.max()) else None,
        "frame": rows,
    }


def training_admission(features: pd.DataFrame, split: dict) -> tuple[bool, str]:
    summaries = {part: part_summary(features, split, part) for part, _ in PARTS}
    empty = [part.title() for part, value in summaries.items() if not value["units"] or not value["rows"]]
    if empty:
        return False, "Training needs admitted measurements and at least one physical unit in " + ", ".join(empty) + "."
    return True, "Train, Validation, and Test have admitted measurements. Training uses Train units; Validation selects the model; Test is held out until evaluation."


def _do_move(project_id: str, snapshot_id: str, part: str) -> None:
    """Button callback: runs before widgets re-render, so their keys can be dropped here."""
    units = list(st.session_state.get(f"quality_move_units:{part}") or [])
    name = str(st.session_state.get(f"quality_move_to:{part}") or "")
    destination = PART_BY_NAME.get(name)
    if destination is None:
        st.session_state[MOVE_FLASH_KEY] = ("warning", "Choose the set to move the units to.")
        return
    try:
        move_units(project_id, units, destination, expected_snapshot_id=snapshot_id)
    except (ValueError, RuntimeError, OSError) as exc:
        st.session_state[MOVE_FLASH_KEY] = ("warning", str(exc))
        return
    for key in list(st.session_state):
        if str(key).startswith(("quality_move_units:", "quality_move_to:", "quality_unit_")):
            st.session_state.pop(key, None)
    st.session_state[MOVE_FLASH_KEY] = ("success", QUALITY_MOVE_DONE.format(n=len(units), name=name))


def _render_move(snapshot: dict, part: str, storage_mode: str, job_active: bool) -> None:
    with st.expander("Move units", expanded=False):
        if storage_mode != "owned":
            st.info(QUALITY_MOVE_LEGACY)
            return
        split = snapshot["split"]
        fixed = set(preview_move(snapshot, (), "train")["fixed_units"])
        movable = sorted(str(uid) for uid in split.get(part) or [] if str(uid) not in fixed)
        if part == "test" and fixed:
            st.caption(QUALITY_MOVE_FIXED_HSE.format(n=len(fixed)))
        key = f"quality_move_units:{part}"
        if key in st.session_state:
            st.session_state[key] = [uid for uid in st.session_state[key] if uid in movable]
        chosen = st.multiselect("Units to move", movable, key=key,
                                disabled=job_active, help=QUALITY_MOVE_UNITS_HELP)
        targets = [name for other, name in PARTS if other != part]
        target = st.selectbox("Move to", targets, key=f"quality_move_to:{part}",
                              disabled=job_active, help=QUALITY_MOVE_TO_HELP)
        if part == "test" or target == "Testing Data":
            st.caption(QUALITY_MOVE_TEST_OPTIMISM)
        problem = None
        if chosen:
            preview = preview_move(snapshot, chosen, PART_BY_NAME[target])
            problem = preview["problem"]
            if problem:
                st.warning(problem)
            else:
                st.caption(QUALITY_MOVE_PREVIEW.format(**preview["counts"]))
        if job_active:
            st.caption(QUALITY_MOVE_JOB_ACTIVE)
        st.button("Move selected units", disabled=not chosen or bool(problem) or job_active,
                  key=f"quality_move:{part}", help=QUALITY_MOVE_SUBMIT_HELP, on_click=_do_move,
                  args=(str(snapshot["project_id"]), str(snapshot["snapshot_id"]), part))


def render_quality(snapshot: dict, theme: str = "dark", *, storage_mode: str = "owned") -> bool:
    features = snapshot["features"]
    split = snapshot["split"]
    schema = snapshot["schema"]
    report = snapshot.get("report") or {}
    label = str(schema.get("signal_label") or schema.get("signal_column") or "Signal")
    unit = str(schema.get("signal_unit") or "")
    page_header("Data Quality", QUALITY_DESCRIPTION)
    flash = st.session_state.pop(MOVE_FLASH_KEY, None)
    if flash:
        (st.success if flash[0] == "success" else st.warning)(flash[1])
    project_id, snapshot_id = str(snapshot.get("project_id")), str(snapshot.get("snapshot_id"))
    job_running = heavy_job_active()
    job_active = storage_mode == "owned" and job_running
    saved_limits = load_zone_limits(project_id, snapshot_id)
    zones_schema = zone_schema(schema, project_id, snapshot_id, saved_limits)
    rule_key = json.dumps(zones_schema.get("thresholds"), sort_keys=True, default=str)
    st.caption(QUALITY_TABS_CAPTION)
    summaries = {part: part_summary(features, split, part) for part, _ in PARTS}
    zoned = project_zones.has_valid_rule(zones_schema)
    drew_limits = False
    tabs = st.tabs([name for _, name in PARTS], key="quality_tab", on_change="rerun")
    for tab, (part, name) in zip(tabs, PARTS, strict=True):
        summary = summaries[part]
        is_open = tab.open is not False
        with tab:
            c1, c2, c3 = st.columns(3)
            c1.metric("Units", summary["units"], help=QUALITY_UNITS_HELP)
            c2.metric("Admitted rows", summary["rows"], help=QUALITY_ADMITTED_ROWS_HELP)
            c3.metric("Gaps", summary["gaps"], help=QUALITY_GAPS_HELP)
            if is_open and zoned and summary["rows"]:
                counts = split_zone_counts(project_id, snapshot_id, part, rule_key,
                                           features, tuple(str(uid) for uid in split.get(part) or []), zones_schema)
                st.caption(f"Zones: Green {counts['green']} · Yellow {counts['yellow']} · Red {counts['red']} · "
                           f"Not zoned {counts['unknown']} rows", help=QUALITY_ZONE_SUMMARY_HELP)
            if summary["time_start"] is not None:
                st.caption(f"Observed time: {summary['time_start']:g}–{summary['time_end']:g} s. "
                           f"{label}: {summary['signal_min']:g}–{summary['signal_max']:g} {unit}.")
            else:
                st.caption("No admitted measurements in this set.")
            if report.get("outcome_semantics") == "unlabelled_observations":
                st.caption("These are sensor histories without failure labels. Signal forecasting uses future measurements within each recorded history.")
            elif snapshot.get("units") is not None and "event_observed" in snapshot["units"]:
                unit_rows = snapshot["units"].loc[snapshot["units"]["unit_id"].astype(str).isin(
                    {str(uid) for uid in split.get(part) or []})]
                observed = int((pd.to_numeric(unit_rows["event_observed"], errors="coerce") == 1).sum())
                event_name = "Observed pressure crossings" if schema.get("source_kind") == "hse_filters" else "Recorded experiment endpoints"
                st.caption(f"{event_name}: {observed}. Unknown or censored outcomes: {len(unit_rows) - observed}.")
            findings = (report.get("by_split") or {}).get(part) or (report.get("sets") or {}).get(part) or {}
            rejected = findings.get("rejected_signal_rows", findings.get("rejected_rows", findings.get("rejected", 0)))
            if rejected or summary["missing_signal"]:
                st.write(f"Rejected or missing signal rows: {rejected}; remaining missing signal values: {summary['missing_signal']}.")
            if summary["gaps"]:
                st.caption("Gaps split the history. Training and forecasts do not cross them.")
            _render_move(snapshot, part, storage_mode, job_active)
            ids = sorted(str(uid) for uid in split.get(part) or [])
            if ids:
                selected = st.selectbox(f"Inspect {name} unit", ids, key=f"quality_unit_{part}",
                                        help=QUALITY_INSPECT_UNIT_HELP)
                if not is_open:
                    continue
                frame = summary["frame"].loc[summary["frame"]["unit_id"].astype(str) == selected]
                labelled = project_zones.label_unit(frame, zones_schema)
                chart_col, limits_col = st.columns([4, 1], gap="small", vertical_alignment="top", wrap=False)
                with chart_col:
                    st.plotly_chart(zone_figure(labelled, zones_schema, label, unit, theme),
                                    width="stretch", theme=None)
                if is_open and not drew_limits:
                    drew_limits = True
                    with limits_col:
                        _render_limits(schema, project_id, snapshot_id, saved_limits, job_running)
                st.caption("Admitted measurements for the selected unit")
                if zoned:
                    st.caption(f"{project_zones.describe_rule(zones_schema)} "
                               f"{QUALITY_ZONE_MODEL_CAPTION.format(label=label)}")
                else:
                    st.caption(QUALITY_NO_ZONES)
                display = labelled[["timestamp_s", "signal"]].rename(columns={
                    "timestamp_s": "Time (s)", "signal": f"{label} ({unit})",
                })
                display["Record position"] = ["Start of record" if index == 0 else
                                               "Gap before" if gap else "Continuous"
                                               for index, gap in enumerate(labelled["gap_before"])]
                display["Zone"] = labelled["zone"].map(ZONE_NAMES)
                with st.container(height=240, border=True, key=f"quality_table_{part}"):
                    st.table(display, hide_index=True, border="horizontal")
            elif is_open and not drew_limits:
                drew_limits = True
                _render_limits(schema, project_id, snapshot_id, saved_limits, job_running)
    ready, explanation = training_admission(features, split)
    if ready:
        try:
            engines = available_signal_engines(snapshot["project_id"], snapshot["snapshot_id"])
            if not any(engine.get("available") for engine in engines):
                ready = False
                explanation = "These sets are present, but their continuous histories are too short for a supported signal model. Add longer histories or adjust the unit split."
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            ready = False
            explanation = f"Training eligibility could not be checked: {exc}"
    (st.success if ready else st.warning)(explanation)
    return ready
