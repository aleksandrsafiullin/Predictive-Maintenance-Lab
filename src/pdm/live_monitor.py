"""Live monitor: score the newest measurements in a watched folder with a saved signal model.

A watched folder (for example a SharePoint library synced by OneDrive) holds CSV files
with one row per measurement::

    unit_id,timestamp_s,<signal>
    pump-01,0,0.31

``<signal>`` is the project's signal column (``combined_rms`` for XJTU-SY bearings) or
``signal``/``value``. ``timestamp`` with date-times can replace ``timestamp_s``. Files may
be appended to or replaced; rows are de-duplicated per (unit_id, timestamp).

Each refresh reads the folder, applies the project's saved yellow/red rule to each
machine's latest measurement, runs the saved model on its most recent continuous
history (the same direct horizons as Results, never future rows) and reports the first
forecast point in red. Escalations to yellow or red can be posted to Microsoft Teams.
Everything here is file-based so the UI and the ``pdm live-watch`` CLI share one state.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pdm.io_util import atomic_write_json, read_json
from pdm.paths import project_root
from pdm.project_zones import resolve_thresholds, zone_labels

ZONE_RANK = {"unknown": -1, "green": 0, "yellow": 1, "red": 2}
SIGNAL_ALIASES = ("signal", "value")
TEAMS_ENV = "PDM_TEAMS_WEBHOOK_URL"
DEFAULT_REFRESH_S = 10


# ------------------------------------------------------------------ storage

def live_root() -> Path:
    return Path(os.environ.get("PDM_LIVE_ROOT") or project_root() / "data" / "live")


def live_dir(project_id: str) -> Path:
    path = live_root() / project_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_folder(project_id: str) -> Path:
    return live_dir(project_id) / "incoming"


def load_config(project_id: str) -> dict[str, Any]:
    path = live_dir(project_id) / "config.json"
    config = {"folder": str(default_folder(project_id)), "refresh_s": DEFAULT_REFRESH_S,
              "run_id": None, "teams_webhook_url": "", "alerts_enabled": False}
    if path.exists():
        try:
            config.update(read_json(path))
        except (OSError, ValueError):
            pass
    return config


def save_config(project_id: str, config: Mapping[str, Any]) -> dict[str, Any]:
    merged = {**load_config(project_id), **dict(config)}
    atomic_write_json(live_dir(project_id) / "config.json", merged)
    return merged


def webhook_url(config: Mapping[str, Any]) -> str:
    """The environment variable wins, so a shared machine never needs the URL on disk."""
    return (os.environ.get(TEAMS_ENV) or str(config.get("teams_webhook_url") or "")).strip()


# ------------------------------------------------------------------ reading

def read_live_folder(folder: str | Path, signal_column: str | None = None) -> pd.DataFrame:
    """All CSV rows in ``folder`` as unit_id, timestamp_s, signal (sorted, de-duplicated)."""
    folder = Path(folder)
    columns = ["unit_id", "timestamp_s", "signal"]
    if not folder.is_dir():
        return pd.DataFrame(columns=columns)
    frames = []
    for path in sorted(folder.glob("*.csv")):
        try:
            raw = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.EmptyDataError, pd.errors.ParserError):
            continue  # a file being written right now is read on the next refresh
        frame = _normalise(raw, signal_column)
        if frame is not None:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=columns)
    data = pd.concat(frames, ignore_index=True)
    data = data[np.isfinite(data.timestamp_s) & np.isfinite(data.signal)]
    data = data.drop_duplicates(["unit_id", "timestamp_s"], keep="last")
    return data.sort_values(["unit_id", "timestamp_s"]).reset_index(drop=True)


def _normalise(raw: pd.DataFrame, signal_column: str | None) -> pd.DataFrame | None:
    cols = {str(c).strip(): c for c in raw.columns}
    if "unit_id" not in cols:
        return None
    candidates = [c for c in (signal_column, *SIGNAL_ALIASES) if c]
    signal = next((cols[c] for c in candidates if c in cols), None)
    if signal is None:
        return None
    if "timestamp_s" in cols:
        t = pd.to_numeric(raw[cols["timestamp_s"]], errors="coerce")
    elif "timestamp" in cols:
        parsed = pd.to_datetime(raw[cols["timestamp"]], errors="coerce", utc=True)
        t = (parsed - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    else:
        return None
    return pd.DataFrame({"unit_id": raw[cols["unit_id"]].astype(str).str.strip(),
                         "timestamp_s": t.astype(float),
                         "signal": pd.to_numeric(raw[signal], errors="coerce").astype(float)})


def mark_gaps(unit: pd.DataFrame, cadence_s: float | None) -> pd.DataFrame:
    """Flag a break in continuous history where the clock jumps by more than 1.5 intervals."""
    unit = unit.sort_values("timestamp_s").reset_index(drop=True)
    gap = np.zeros(len(unit), dtype=bool)
    if len(unit):
        gap[0] = True
        if cadence_s and np.isfinite(cadence_s) and cadence_s > 0:
            gap[1:] = np.diff(unit.timestamp_s.to_numpy(float)) > 1.5 * float(cadence_s)
    return unit.assign(gap_before=gap)


# ------------------------------------------------------------------ scoring

class LiveModel:
    """A verified saved run plus everything needed to score live rows."""

    def __init__(self, project_id: str, run_id: str) -> None:
        from pdm.data.project_prepare import load_snapshot, load_zone_limits
        from pdm.projects import project_store
        from pdm.signal_inference import _load_model, training_cadence
        from pdm.signal_training import load_signal_run

        self.project_id, self.run_id = project_id, run_id
        self.run = load_signal_run(project_id, run_id)
        snapshot = load_snapshot(project_id, self.run["snapshot_id"])
        self.schema = dict(self.run["schema"])
        override = load_zone_limits(project_id, self.run["snapshot_id"])
        if override:
            self.schema["thresholds"] = dict(override)
        self.cadence_s = training_cadence(snapshot)
        self.history = int(self.run["params"]["history_length"])
        self.horizons = [float(h) for h in self.run["params"]["horizons_s"]]
        self.model = _load_model(project_id, run_id, self.run["artifacts"][self.run["artifact"]],
                                 str(project_store().root))

    @property
    def signal_column(self) -> str:
        return str(self.schema.get("signal_column") or "signal")

    def assess(self, unit: pd.DataFrame, now_s: float | None = None) -> dict[str, Any]:
        """Zone, forecast and predicted red entry for one machine's rows (all at/before now)."""
        from pdm.signal_inference import _crossing
        from pdm.signal_training import _predict, _segments

        uid = str(unit.unit_id.iloc[0])
        unit = mark_gaps(unit, self.cadence_s)
        thresholds = resolve_thresholds(self.schema, unit)
        current = float(unit.signal.iloc[-1])
        issued = float(unit.timestamp_s.iloc[-1])
        zone = str(zone_labels([current], thresholds)[0])
        result: dict[str, Any] = {
            "unit_id": uid, "as_of_s": issued, "current": current, "zone": zone, "n_rows": int(len(unit)),
            "thresholds": thresholds, "points": [], "status": "unavailable", "reason": None,
            "crossing": {"status": "unavailable", "time_s": None}, "red_in_s": None,
            "observed_prefix": [{"timestamp_s": float(t), "signal": float(v), "gap_before": bool(g)}
                                for t, v, g in zip(unit.timestamp_s, unit.signal, unit.gap_before, strict=True)],
            "age_s": None if now_s is None else max(0.0, float(now_s) - issued),
        }
        segment = _segments(unit, uid)[-1]
        if len(segment) < self.history:
            result["reason"] = f"Collecting history: {len(segment)} of {self.history} measurements"
            return result
        x = segment.signal.to_numpy(np.float32)[-self.history:].reshape(1, self.history, 1)
        frame = {"x": x, "y": np.zeros((1, len(self.horizons)), np.float32)}
        pred, lower, upper = _predict(self.model, self.run["engine_id"], frame, self.run["scaler"])
        points = [{"target_time_s": issued + h, "value": float(pred[0, j]),
                   "lower": float(lower[0, j]) if lower is not None else None,
                   "upper": float(upper[0, j]) if upper is not None else None}
                  for j, h in enumerate(self.horizons) if np.isfinite(pred[0, j])]
        crossing = _crossing(current, thresholds, points, issued)
        result.update(points=points, status="available" if points else "unavailable",
                      reason=None if points else "No supported forecast points", crossing=crossing,
                      red_in_s=None if crossing.get("time_s") is None else float(crossing["time_s"]) - issued)
        return result


def assess_fleet(model: LiveModel, data: pd.DataFrame) -> list[dict[str, Any]]:
    """One result per machine, most urgent first."""
    now = time.time()
    results = []
    for uid, unit in data.groupby("unit_id", sort=True):
        # Wall-clock age only makes sense for real date-times, not seconds since start.
        clock_now = now if float(unit.timestamp_s.max()) > 1e9 else None
        results.append(model.assess(unit, clock_now))
    return sorted(results, key=urgency_key)


def urgency_key(result: Mapping[str, Any]) -> tuple:
    rank = ZONE_RANK.get(result["zone"], -1)
    red_in = result.get("red_in_s")
    return (-rank, red_in if red_in is not None else float("inf"), result["unit_id"])


# ------------------------------------------------------------------ alerts

def _state_path(project_id: str) -> Path:
    return live_dir(project_id) / "alert_state.json"


def _log_path(project_id: str) -> Path:
    return live_dir(project_id) / "alerts.jsonl"


def detect_escalations(results: list[Mapping[str, Any]], state: dict[str, Any]) -> list[dict[str, Any]]:
    """Machines whose zone rose above the last alerted level. Returning to green re-arms."""
    events = []
    for r in results:
        uid, zone = r["unit_id"], r["zone"]
        if zone == "unknown":
            continue
        last = state.get(uid, "green")
        if zone == "green":
            state[uid] = "green"
        elif ZONE_RANK[zone] > ZONE_RANK.get(last, 0):
            state[uid] = zone
            events.append({"unit_id": uid, "zone": zone, "previous": last, "current": r["current"],
                           "as_of_s": r["as_of_s"], "red_in_s": r.get("red_in_s"),
                           "crossing_status": (r.get("crossing") or {}).get("status")})
    return events


def teams_card(event: Mapping[str, Any], schema: Mapping[str, Any], project_name: str) -> dict[str, Any]:
    """Adaptive Card envelope accepted by Teams Workflows and classic incoming webhooks."""
    zone = event["zone"]
    unit = str(schema.get("signal_unit") or "")
    label = str(schema.get("signal_label") or schema.get("signal_column") or "Signal")
    title = {"red": "🔴 Fix ASAP", "yellow": "🟡 Something is not right", "green": "🟢 Back to normal"}.get(zone, zone)
    if event.get("test"):
        title = "✅ Test alert from Predictive Maintenance Lab"
    facts = [{"title": "Machine", "value": str(event["unit_id"])},
             {"title": "Project", "value": project_name},
             {"title": label, "value": f"{float(event['current']):.3g} {unit}".strip()},
             {"title": "Change", "value": f"{event.get('previous', '—')} → {zone}"}]
    red_in = event.get("red_in_s")
    if zone == "yellow":
        facts.append({"title": "Forecast",
                      "value": f"red in about {red_in / 60:.0f} min" if red_in is not None
                      else "no red crossing within the model's horizon"})
    body = [{"type": "TextBlock", "text": title, "weight": "Bolder", "size": "Large", "wrap": True,
             "color": {"red": "Attention", "yellow": "Warning"}.get(zone, "Good")},
            {"type": "FactSet", "facts": facts},
            {"type": "TextBlock", "isSubtle": True, "wrap": True, "size": "Small",
             "text": "Signal zones follow the saved limits; the forecast is the saved model's direct output. "
                     "Laboratory MVP, not an equipment-protection system."}]
    return {"type": "message", "attachments": [{
        "contentType": "application/vnd.microsoft.card.adaptive", "contentUrl": None,
        "content": {"$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard", "version": "1.4", "body": body}}]}


def post_teams(url: str, payload: Mapping[str, Any], timeout: float = 10.0) -> tuple[bool, str]:
    import requests

    if not url:
        return False, "No Teams webhook URL configured"
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        return False, f"Could not reach Teams: {exc.__class__.__name__}"
    if 200 <= response.status_code < 300:
        return True, f"Delivered ({response.status_code})"
    return False, f"Teams answered {response.status_code}: {response.text[:200]}"


def process_alerts(project_id: str, project_name: str, model: LiveModel, results: list[Mapping[str, Any]],
                   config: Mapping[str, Any], *, sender: Callable = post_teams) -> list[dict[str, Any]]:
    """Detect escalations, post them when alerts are on, append them to the alert log."""
    path = _state_path(project_id)
    state = read_json(path) if path.exists() else {}
    events = detect_escalations(results, state)
    atomic_write_json(path, state)
    url = webhook_url(config)
    logged = []
    for event in events:
        if config.get("alerts_enabled") and url:
            ok, message = sender(url, teams_card(event, model.schema, project_name))
            delivery = "sent" if ok else "failed"
        else:
            ok, message = False, "Teams alerts are off" if not config.get("alerts_enabled") else "No webhook URL"
            delivery = "not sent"
        entry = {**event, "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "delivery": delivery, "message": message}
        with _log_path(project_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
        logged.append(entry)
    return logged


def recent_alerts(project_id: str, limit: int = 20) -> list[dict[str, Any]]:
    path = _log_path(project_id)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows[::-1]


def reset_alerts(project_id: str) -> None:
    for path in (_state_path(project_id), _log_path(project_id)):
        path.unlink(missing_ok=True)


def send_test_alert(project_name: str, model: LiveModel, config: Mapping[str, Any],
                    *, sender: Callable = post_teams) -> tuple[bool, str]:
    event = {"unit_id": "test-machine", "zone": "yellow", "previous": "green", "current": 0.0,
             "as_of_s": 0.0, "red_in_s": None, "test": True}
    return sender(webhook_url(config), teams_card(event, model.schema, project_name))


# ------------------------------------------------------------------ one cycle

def poll_once(project_id: str, project_name: str, config: Mapping[str, Any] | None = None,
              *, sender: Callable = post_teams) -> dict[str, Any]:
    """Read the folder, score every machine, process alerts. Shared by UI and CLI."""
    config = dict(config or load_config(project_id))
    run_id = config.get("run_id")
    if not run_id:
        raise ValueError("Choose a saved model for the live monitor")
    model = LiveModel(project_id, run_id)
    data = read_live_folder(config["folder"], model.signal_column)
    results = assess_fleet(model, data) if not data.empty else []
    alerts = process_alerts(project_id, project_name, model, results, config, sender=sender)
    return {"model": model, "results": results, "alerts": alerts, "rows": int(len(data)),
            "read_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
