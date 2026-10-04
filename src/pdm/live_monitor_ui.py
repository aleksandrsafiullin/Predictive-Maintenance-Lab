"""Live monitor screen: current zone and forecast for every machine in a watched folder."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pdm.data.project_prepare import load_zone_limits
from pdm.live_monitor import (
    DEFAULT_REFRESH_S,
    TEAMS_ENV,
    LiveModel,
    assess_fleet,
    default_folder,
    load_config,
    process_alerts,
    read_live_folder,
    recent_alerts,
    reset_alerts,
    save_config,
    send_test_alert,
    webhook_url,
)
from pdm.live_simulator import (
    clear_demo_files,
    demo_units,
    simulator_status,
    start_simulator_process,
    stop_simulator,
)
from pdm.project_results_ui import replay_figure
from pdm.signal_training import list_project_runs, load_signal_run
from pdm.ui_theme import empty_state, page_header

ZONE_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴", "unknown": "⚪"}
ZONE_TEXT = {"green": "Normal", "yellow": "Something is not right", "red": "Fix ASAP", "unknown": "Not zoned yet"}
ZONE_SHORT = {"green": "Normal", "yellow": "Not right", "red": "Fix ASAP", "unknown": "Not zoned"}
REFRESH_CHOICES = [5, 10, 30, 60, 300, 900]


@st.cache_resource(show_spinner=False, max_entries=8)
def _live_model(project_id: str, run_id: str, limits_key: str) -> LiveModel:
    # limits_key changes when Data Quality saves new limits, so the cache refreshes.
    return LiveModel(project_id, run_id)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def _forecast_text(result: dict) -> str:
    crossing = (result.get("crossing") or {}).get("status")
    if result["zone"] == "red":
        return "Already red"
    if result.get("status") != "available":
        return result.get("reason") or "No forecast"
    if crossing == "predicted":
        return f"Red in ~{_duration(result['red_in_s'])}"
    if crossing == "none_within_horizon":
        return "No red within forecast range"
    return "No limits for a red forecast"


def _saved_runs(project_id: str, snapshot_id: str) -> list[dict]:
    rows = []
    for row in list_project_runs(project_id):
        run_id = row.get("run_id")
        if not run_id or row.get("snapshot_id") != snapshot_id:
            continue
        try:
            load_signal_run(project_id, run_id)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
        rows.append(row)
    return rows


def _settings(project_id: str, snapshot: dict, config: dict, runs: list[dict]) -> dict:
    ids = [r["run_id"] for r in runs]
    labels = {r["run_id"]: f"{str(r.get('engine_id', '')).upper()} · {r['run_id'][-8:]}" for r in runs}
    c1, c2, c3 = st.columns([2, 3, 1])
    run_id = c1.selectbox("Saved model", ids, index=ids.index(config["run_id"]) if config.get("run_id") in ids else 0,
                          format_func=lambda r: labels.get(r, r), key=f"live_run:{project_id}")
    folder = c2.text_input("Watched folder", value=config["folder"], key=f"live_folder:{project_id}",
                           help="A folder on the computer running the app. Point it at a SharePoint library "
                                "synced with OneDrive to monitor files your team uploads.")
    refresh = c3.selectbox("Refresh every", REFRESH_CHOICES,
                           index=REFRESH_CHOICES.index(config["refresh_s"]) if config.get("refresh_s") in REFRESH_CHOICES
                           else REFRESH_CHOICES.index(DEFAULT_REFRESH_S),
                           format_func=lambda s: f"{s} s" if s < 60 else f"{s // 60} min", key=f"live_refresh:{project_id}")
    changes = {k: v for k, v in {"run_id": run_id, "folder": folder.strip(), "refresh_s": refresh}.items()
               if config.get(k) != v}
    if changes:
        config = save_config(project_id, changes)
    signal = str(snapshot["schema"].get("signal_column") or "signal")
    st.caption(f"Reads every `.csv` file in the folder. Columns: `unit_id`, `timestamp_s` (or `timestamp` as a "
               f"date-time) and `{signal}` (or `signal`). One row per measurement; each machine's file should start "
               "from its healthy period so baseline limits can be computed.")
    return config


def _demo_feed(project_id: str, config: dict) -> None:
    status = simulator_status(project_id)
    running = status.get("state") == "running"
    with st.expander("Demo feed · replay recorded machines as live data", expanded=True):
        st.caption("Writes real recorded measurements into the watched folder one by one, with an accelerated "
                   "clock. Use it to show how the monitor reacts as machines degrade.")
        groups = demo_units(project_id)
        options = groups["test"] + groups["validation"]
        c1, c2 = st.columns([3, 2])
        units = c1.multiselect("Machines to replay", options, default=options, disabled=running,
                               key=f"live_sim_units:{project_id}",
                               help="Test and Validation machines were not used to fit the model.")
        speed = c2.select_slider("Speed", options=[0.25, 0.5, 1.0, 2.0, 5.0], value=1.0, disabled=running,
                                 format_func=lambda s: f"{s:g} measurement/s per machine",
                                 key=f"live_sim_speed:{project_id}")
        b1, b2, b3 = st.columns(3)
        if b1.button("Start demo feed", type="primary", disabled=running or not units, width="stretch"):
            clear_demo_files(config["folder"])
            reset_alerts(project_id)
            start_simulator_process(project_id, config["folder"], units, tick_s=1.0 / speed, rows_per_tick=1)
            st.rerun()
        if b2.button("Stop demo feed", disabled=not running, width="stretch"):
            stop_simulator(project_id)
            st.rerun()
        if b3.button("Clear demo data", disabled=running, width="stretch",
                     help="Removes only files written by the demo feed, and resets the alert log."):
            clear_demo_files(config["folder"])
            reset_alerts(project_id)
            st.rerun()


def _alerts_panel(project_id: str, project_name: str, config: dict, model: LiveModel | None) -> dict:
    with st.expander("Teams alerts", expanded=False):
        st.markdown(
            "1. In Microsoft Teams, open the channel → **⋯** → **Workflows** → "
            "**Post to a channel when a webhook request is received**.\n"
            "2. Finish the wizard and copy the webhook URL it shows.\n"
            "3. Paste it below, turn alerts on and press **Send test alert**.")
        env_url = webhook_url({})
        if env_url:
            st.info(f"Using the webhook from the `{TEAMS_ENV}` environment variable.")
            url = config.get("teams_webhook_url", "")
        else:
            url = st.text_input("Webhook URL", value=config.get("teams_webhook_url", ""), type="password",
                                key=f"live_hook:{project_id}",
                                help=f"Stored only on this computer. Or set the {TEAMS_ENV} environment variable.")
        enabled = st.toggle("Send an alert when a machine turns yellow or red",
                            value=bool(config.get("alerts_enabled")), key=f"live_alerts_on:{project_id}")
        changes = {k: v for k, v in {"teams_webhook_url": url.strip(), "alerts_enabled": enabled}.items()
                   if config.get(k) != v}
        if changes:
            config = save_config(project_id, changes)
        if st.button("Send test alert", disabled=not webhook_url(config) or model is None):
            ok, message = send_test_alert(project_name, model, config)
            (st.success if ok else st.error)(message)
        st.caption("One alert per escalation: green → yellow, and yellow → red. A machine that returns to green "
                   "can alert again later.")
    return config


def render_live_monitor(project_id: str, project_name: str, snapshot: dict, theme: str = "dark") -> None:
    page_header("Live monitor", "Latest measurements from a watched folder, checked against the saved limits "
                                "and forecast by the saved model.")
    runs = _saved_runs(project_id, snapshot["snapshot_id"])
    if not runs:
        empty_state("No saved model yet", "Train a signal model first; the live monitor uses it for forecasts.")
        return
    config = load_config(project_id)
    if not config.get("folder"):
        config = save_config(project_id, {"folder": str(default_folder(project_id))})
    config = _settings(project_id, snapshot, config, runs)
    limits = load_zone_limits(project_id, snapshot["snapshot_id"])
    try:
        model = _live_model(project_id, config["run_id"], repr(sorted((limits or {}).items())))
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        st.error(f"The saved model cannot be loaded: {exc}")
        return
    _demo_feed(project_id, config)
    config = _alerts_panel(project_id, project_name, config, model)
    playing = simulator_status(project_id).get("state") == "running"
    _fleet(project_id, project_name, model, config, theme, run_every=config["refresh_s"] if not playing else 2)


def _fleet(project_id, project_name, model, config, theme, run_every):
    @st.fragment(run_every=run_every)
    def body():
        feed = simulator_status(project_id)
        if feed.get("units") and feed.get("state") in {"running", "finished", "stopped"}:
            done = sum(u.get("written") or 0 for u in feed["units"].values())
            total = sum(u.get("total") or 0 for u in feed["units"].values())
            st.caption(f"Demo feed {feed['state']} · {done} of {total or '…'} measurements written")
        data = read_live_folder(config["folder"], model.signal_column)
        if data.empty:
            empty_state("No measurements yet",
                        "Put CSV files in the watched folder, or start the demo feed above.")
            return
        results = assess_fleet(model, data)
        process_alerts(project_id, project_name, model, results, load_config(project_id))
        counts = {z: sum(r["zone"] == z for r in results) for z in ZONE_ICON}
        c = st.columns(4)
        c[0].metric("Machines", len(results))
        c[1].metric("🟢 Normal", counts["green"])
        c[2].metric("🟡 Not right", counts["yellow"])
        c[3].metric("🔴 Fix ASAP", counts["red"])
        unit_label = str(model.schema.get("signal_unit") or "")
        table = pd.DataFrame([{
            "Status": f"{ZONE_ICON[r['zone']]} {ZONE_SHORT[r['zone']]}",
            "Machine": r["unit_id"],
            f"Latest ({unit_label})": round(r["current"], 3),
            f"Yellow / red limit ({unit_label})": (
                f"{r['thresholds']['yellow']:.3g} / {r['thresholds']['red']:.3g}"
                if r["thresholds"].get("status") == "available" and r["thresholds"].get("yellow") is not None
                else "—"),
            "Forecast": _forecast_text(r),
            "Running time": _duration(r["as_of_s"] - r["observed_prefix"][0]["timestamp_s"]),
            "Measurements": r["n_rows"],
        } for r in results])
        st.dataframe(table, hide_index=True, width="stretch")
        # Stable alphabetical options: the urgency order changes between refreshes, and a
        # reordered option list must not silently switch the machine shown in the chart.
        by_id = {r["unit_id"]: r for r in results}
        ids = sorted(by_id)
        key = f"live_pick:{project_id}"
        if st.session_state.get(key) not in by_id:
            st.session_state[key] = results[0]["unit_id"]
        picked = st.selectbox("Machine detail", ids, key=key,
                              format_func=lambda u: f"{ZONE_ICON[by_id[u]['zone']]} {u}")
        result = by_id[picked]
        st.plotly_chart(replay_figure(result, model.schema, theme), width="stretch", theme=None,
                        key=f"live_chart:{project_id}")
        st.caption(f"{ZONE_ICON[result['zone']]} **{ZONE_TEXT[result['zone']]}** · {_forecast_text(result)}. "
                   "Blue points are the model's forecast from the latest measurement.")
        alerts = recent_alerts(project_id, 8)
        if alerts:
            st.subheader("Recent alerts")
            st.dataframe(pd.DataFrame([{
                "When (UTC)": a["logged_at"].replace("T", " ").replace("+00:00", ""),
                "Machine": a["unit_id"],
                "Change": f"{ZONE_ICON.get(a.get('previous'), '')} → {ZONE_ICON.get(a['zone'], '')} {ZONE_TEXT.get(a['zone'], '')}",
                "Teams": a["delivery"],
            } for a in alerts]), hide_index=True, width="stretch")

    body()
