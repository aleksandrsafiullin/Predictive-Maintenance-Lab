"""Display already imported four-role sensor snapshots through owned storage."""

from __future__ import annotations

import streamlit as st

from pdm.project_snapshot import project_snapshot, role_frame, snapshot_directory
from pdm.ui_theme import page_header

PARTS = (
    ("train", "Training Data"),
    ("validation", "Validation Data"),
    ("calibration", "Calibration Data"),
    ("test", "Testing Data"),
)


def registered(project):
    if project.get("state") != "ready" or not project.get("active_snapshot_id"):
        return False
    directory = snapshot_directory(project["project_id"], project["active_snapshot_id"])
    return (directory / "snapshot.json").is_file() or (project.get("source_manifest") or {}).get(
        "task"
    ) == "probabilistic_signal_forecast"


def summaries(project_id, snapshot_id):
    view = project_snapshot(project_id, snapshot_id)
    raw = view["sensor_snapshot"]
    return {
        "format": "sensor",
        "parts": {
            part: {
                "units": raw["admission"].get(part, {}).get("units", 0),
                "rows": raw["admission"].get(part, {}).get("accepted_rows", 0),
                "gaps": raw["admission"].get(part, {}).get("gaps", 0),
            }
            for part, _ in PARTS
        },
        "protocol": str(raw["split_manifest"].get("protocol") or ""),
    }


def render_quality(project, theme="dark"):
    from pdm import project_zones
    from pdm.project_quality_ui import zone_figure

    view = project_snapshot(project["project_id"])
    summary = summaries(project["project_id"], view["snapshot_id"])
    schema = view["schema"]
    page_header("Data Quality", "Inspect the four saved data sets.")
    tabs = st.tabs([label for _, label in PARTS], key="quality_tab", on_change="rerun")
    for tab, (part, label) in zip(tabs, PARTS, strict=True):
        with tab:
            counts = summary["parts"][part]
            columns = st.columns(3)
            for column, name, key in zip(
                columns, ("Units", "Admitted rows", "Gaps"), ("units", "rows", "gaps"), strict=True
            ):
                column.metric(name, counts[key])
            ids = view["split"].get(part, [])
            if not ids:
                st.caption("No admitted measurements in this set.")
                continue
            uid = st.selectbox(f"Inspect {label} unit", ids, key=f"quality_unit_{part}")
            if tab.open is False:
                continue
            frame = role_frame(view, part, uid)
            labelled = project_zones.label_unit(frame, schema)
            st.plotly_chart(
                zone_figure(labelled, schema, schema["signal_label"], schema["signal_unit"], theme),
                width="stretch",
                theme=None,
            )
            with st.container(height=240, border=True, key=f"quality_table_{part}"):
                st.table(
                    labelled[["timestamp_s", "signal"]].rename(
                        columns={
                            "timestamp_s": "Time (s)",
                            "signal": f"{schema['signal_label']} ({schema['signal_unit']})",
                        }
                    ),
                    hide_index=True,
                    border="horizontal",
                )
    if st.button("Continue to Training", type="primary"):
        st.session_state["project_step"] = "Training"
        st.rerun()
