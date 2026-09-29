"""Plain-language quality view for one immutable project snapshot."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pdm.project_chart_style import style_signal_chart
from pdm.signal_training import available_signal_engines
from pdm.ui_copy import (
    QUALITY_ADMITTED_ROWS_HELP,
    QUALITY_GAPS_HELP,
    QUALITY_INSPECT_UNIT_HELP,
    QUALITY_TABS_CAPTION,
    QUALITY_UNITS_HELP,
)
from pdm.ui_theme import page_header, tokens

QUALITY_DESCRIPTION = "Each set contains whole physical units. Rows from one unit stay in one set."
PARTS = (("train", "Training Data"), ("validation", "Validation Data"), ("test", "Testing Data"))


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


def render_quality(snapshot: dict, theme: str = "dark") -> bool:
    features = snapshot["features"]
    split = snapshot["split"]
    schema = snapshot["schema"]
    report = snapshot.get("report") or {}
    label = str(schema.get("signal_label") or schema.get("signal_column") or "Signal")
    unit = str(schema.get("signal_unit") or "")
    page_header("Data Quality", QUALITY_DESCRIPTION)
    st.caption(QUALITY_TABS_CAPTION)
    summaries = {part: part_summary(features, split, part) for part, _ in PARTS}
    tabs = st.tabs([name for _, name in PARTS])
    for tab, (part, name) in zip(tabs, PARTS, strict=True):
        summary = summaries[part]
        with tab:
            c1, c2, c3 = st.columns(3)
            c1.metric("Units", summary["units"], help=QUALITY_UNITS_HELP)
            c2.metric("Admitted rows", summary["rows"], help=QUALITY_ADMITTED_ROWS_HELP)
            c3.metric("Gaps", summary["gaps"], help=QUALITY_GAPS_HELP)
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
            ids = sorted(str(uid) for uid in split.get(part) or [])
            if ids:
                selected = st.selectbox(f"Inspect {name} unit", ids, key=f"quality_unit_{part}",
                                        help=QUALITY_INSPECT_UNIT_HELP)
                frame = summary["frame"].loc[summary["frame"]["unit_id"].astype(str) == selected].sort_values("timestamp_s")
                fig = go.Figure()
                x, y = gap_safe_trace(frame)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers",
                                         name=label, line={"color": tokens(theme)["series_observed"], "width": 2},
                                         marker={"size": 4}))
                fig.update_layout(height=280, margin={"l": 20, "r": 20, "t": 15, "b": 25},
                                  xaxis_title="Time (s)", yaxis_title=f"{label} ({unit})", showlegend=False)
                st.plotly_chart(style_signal_chart(fig, theme), width="stretch", theme=None)
                st.caption("Admitted measurements for the selected unit")
                display = frame[["timestamp_s", "signal"]].rename(columns={
                    "timestamp_s": "Time (s)", "signal": f"{label} ({unit})",
                })
                display["Record position"] = ["Start of record" if index == 0 else
                                               "Gap before" if gap else "Continuous"
                                               for index, gap in enumerate(frame["gap_before"].fillna(False))]
                with st.container(height=240, border=True, key=f"quality_table_{part}"):
                    st.table(display, hide_index=True, border="horizontal")
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
