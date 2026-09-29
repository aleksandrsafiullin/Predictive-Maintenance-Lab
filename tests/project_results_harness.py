"""Isolated Streamlit replay harness for timer and cursor behavior."""

import pandas as pd
import streamlit as st

import pdm.project_results_ui as results_ui


def _forecast(project_id, run_id, unit_id, as_of_s, thresholds=None):
    return {"project_id": project_id, "run_id": run_id, "snapshot_id": "snapshot1",
            "as_of_s": as_of_s, "observed_prefix": [],
            "points": [{"target_time_s": as_of_s + 1, "value": as_of_s + 0.5}],
            "thresholds": {"status": "available", "mode": "absolute", "direction": "above",
                           "yellow": 10, "red": 12},
            "crossing": {"status": "none_within_horizon", "time_s": None}}


results_ui.forecast_prefix = _forecast
st.session_state["_pdm_theme"] = "light"
snapshot = {"snapshot_id": "snapshot1",
            "features": pd.DataFrame({"unit_id": ["unit1"] * 16, "timestamp_s": [float(n) for n in range(16)],
                                      "signal": [float(n) for n in range(16)], "gap_before": [False] * 16}),
            "schema": {"signal_label": "Signal", "signal_unit": "g", "source_kind": "generic_sensor_csv"}}
results_ui._play_fragment("project1", "run1", "unit1", snapshot, theme="light")
