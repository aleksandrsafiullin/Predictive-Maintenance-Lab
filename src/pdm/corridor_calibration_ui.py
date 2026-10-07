"""Calibration settings in the common Data Quality Calibration tab."""

from __future__ import annotations

import streamlit as st

from pdm import corridor_calibration as calibration
from pdm import stable_forecast as stable
from pdm.long_forecast_run import alive, list_runs, read_status
from pdm.projects import project_store
from pdm.worker import heavy_job_active


def _keep_tab_open():
    st.session_state["quality_tab"] = "Calibration Data"


def render_controls(snapshot):
    pid, sid = snapshot["project_id"], snapshot["snapshot_id"]
    prefix = f"calibration:{pid}:{sid}:"
    with st.container(border=True, key="pdm-calibration-settings"):
        st.subheader("Forecast corridor calibration", anchor=False)
        st.caption("Full width is upper minus lower, as a percentage of the forecast center. 20–45% means ±10–22.5%. The center and model weights stay frozen.")
        saved = calibration.load_settings(pid, sid)
        columns = st.columns(3)
        minimum = columns[0].number_input("Minimum full width (%)", min_value=0.01, max_value=199.0,
                                          value=float(saved["min_width"] * 100), step=1.0, key=prefix + "min", on_change=_keep_tab_open)
        maximum = columns[1].number_input("Maximum full width (%)", min_value=0.01, max_value=199.0,
                                          value=float(saved["max_width"] * 100), step=1.0, key=prefix + "max", on_change=_keep_tab_open)
        target = columns[2].number_input("Target coverage (%)", min_value=1.0, max_value=100.0,
                                         value=float(saved["target_coverage"] * 100), step=1.0, key=prefix + "target", on_change=_keep_tab_open)
        c1, c2 = st.columns(2)
        scopes = {"points": "Observed points", "whole_paths": "Complete forecast trajectories"}
        scope = c1.selectbox("Coverage scope", list(scopes), format_func=scopes.get,
                             index=list(scopes).index(saved["scope"]), key=prefix + "scope", on_change=_keep_tab_open)
        stride = c2.number_input("Calibration origin step (observations)", min_value=1, max_value=4096,
                                  value=saved["origin_stride"], step=1, key=prefix + "stride", on_change=_keep_tab_open,
                                  help="Use a new causal origin every N observed measurements. Gaps reset the history. Each physical unit has equal total weight.")
        st.caption("Only Calibration errors determine one width for the full forecast. Maximum width is a hard limit; missing future observations do not count as covered. Coverage measured here is empirical, without a future coverage guarantee.")
        policy, error = None, None
        try:
            policy = calibration.settings(dict(min_width=minimum / 100, max_width=maximum / 100,
                target_coverage=target / 100, scope=scope, origin_stride=int(stride)))
        except ValueError as exc:
            error = str(exc)
            st.error(error)
        busy = heavy_job_active() or alive(read_status(pid))
        if st.button("Save calibration settings", disabled=busy or error is not None, key=prefix + "save", on_click=_keep_tab_open):
            try:
                calibration.save_settings(pid, sid, policy)
                st.success("Settings saved. Calibrate a model to apply them to Forecast.")
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
        runs = [row for row in list_runs(pid) if row.get("snapshot_id") == sid
                and row.get("protocol") in stable.PROTOCOLS]
        if not runs:
            st.caption("Train GRU, LSTM or MaleCNS on this snapshot before fitting its corridor width.")
            return
        ids = [row["run_id"] for row in runs]
        selected = project_store().get(pid).get("selected_run_id")
        rid = st.selectbox("Model to calibrate", ids,
            index=ids.index(selected) if selected in ids else 0, key=prefix + "model", on_change=_keep_tab_open,
            format_func=lambda value: stable.LABELS.get(runs[ids.index(value)]["engine_id"], "Model") + " · " + value[-8:])
        manifest = runs[ids.index(rid)]
        if st.button("Calibrate and apply to Forecast", type="primary", disabled=busy or error is not None,
                     key=prefix + "apply", on_click=_keep_tab_open):
            try:
                with st.spinner("Fitting corridor width on Calibration data"):
                    progress = st.progress(0.0)
                    calibration.calibrate(pid, rid, policy, progress=progress.progress)
                    progress.empty()
                st.success("Calibration saved and activated for this model.")
            except (OSError, ValueError, RuntimeError) as exc:
                st.error(str(exc))
        active = calibration.load_active(pid, rid, manifest)
        if not active:
            st.caption("This model currently uses its saved training width. Calibration has not been applied.")
            return
        c1, c2, c3 = st.columns(3)
        c1.metric("Applied full width", f"{100 * active['width']:.1f}%")
        c2.metric("Calibration point coverage", f"{100 * active['point_coverage']:.1f}%")
        c3.metric("Complete trajectory coverage", "Unknown" if active["whole_path_coverage"] is None
                  else f"{100 * active['whole_path_coverage']:.1f}%")
        st.caption(f"{active['physical_units']} physical units · {active['supported_points']} supported points · {active['complete_origins']} complete trajectories · settings frozen with this calibration")
        if active["settings"] != policy:
            st.caption("Displayed settings differ from the active calibration. Calibrate again to apply the new settings.")
        if active["upper_limit_reached"]:
            st.warning(f"Maximum width reached. Required width for the target: {100 * active['required_width']:.1f}%; applied: {100 * active['width']:.1f}%. Target coverage was not reached within the limit.")
