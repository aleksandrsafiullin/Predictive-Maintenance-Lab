"""Read live sensor prefixes and reuse the saved Results forecasting contracts."""

from __future__ import annotations

import copy
import os
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.io_util import atomic_write_json, read_json
from pdm.paths import project_root
from pdm.project_zones import resolve_thresholds, zone_labels
from pdm.projects import _safe_id, project_store

DEFAULT_REFRESH_S = 10
SIGNAL_ALIASES = ("signal", "value")
ZONE_RANK = {"unknown": -1, "green": 0, "yellow": 1, "red": 2}


def live_dir(project_id):
    project_store().get(project_id)
    root = Path(os.environ.get("PDM_LIVE_ROOT") or project_root() / "data/live")
    path = root / _safe_id(project_id, "project ID")
    if path.is_symlink():
        raise ValueError("Live storage cannot be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_folder(project_id):
    return live_dir(project_id) / "incoming"


def load_config(project_id):
    path = live_dir(project_id) / "config.json"
    config = dict(folder=str(default_folder(project_id)), refresh_s=DEFAULT_REFRESH_S, run_id=None)
    if path.exists():
        config.update({k: v for k, v in read_json(path).items() if k in config})
    return config


def save_config(project_id, changes):
    config = {**load_config(project_id), **changes}
    atomic_write_json(live_dir(project_id) / "config.json", config)
    return config


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
    data = data[np.isfinite(data.timestamp_s) & np.isfinite(data.signal) & data.unit_id.ne("")]
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
    frame = pd.DataFrame(
        {
            "unit_id": raw[cols["unit_id"]].fillna("").astype(str).str.strip(),
            "timestamp_s": t.astype(float),
            "signal": pd.to_numeric(raw[signal], errors="coerce").astype(float),
        }
    )

    if "gap_before" in cols:
        frame["gap_before"] = raw[cols["gap_before"]].astype(str).str.lower().isin(["true", "1"])
    if "component_cycle_id" in cols:
        frame["component_cycle_id"] = raw[cols["component_cycle_id"]]
    return frame


def mark_gaps(unit: pd.DataFrame, cadence_s: float | None) -> pd.DataFrame:
    """Flag a break in continuous history where the clock jumps by more than 1.5 intervals."""
    unit = unit.sort_values("timestamp_s").reset_index(drop=True)
    gap = np.zeros(len(unit), dtype=bool)
    if len(unit):
        gap[0] = True
        if cadence_s and np.isfinite(cadence_s) and cadence_s > 0:
            gap[1:] = np.diff(unit.timestamp_s.to_numpy(float)) > 1.5 * float(cadence_s)
    if "gap_before" in unit:
        gap |= unit.gap_before.fillna(False).to_numpy(bool)
    return unit.assign(gap_before=gap)


def _long_run(run):
    from pdm.long_forecast_run import DIRECT_PROTOCOLS

    return run.get("protocol") in DIRECT_PROTOCOLS or (
        run.get("task") == "probabilistic_signal_forecast"
        and run.get("config", {}).get("forecast_mode") == "bounded_trend_v1"
    )


def available_runs(project_id, snapshot_id):
    """Current long forecasts and supported original direct signal models."""
    from pdm.long_forecast_run import list_runs
    from pdm.signal_training import list_project_runs

    candidates = {
        row["run_id"]: row for row in list_runs(project_id) + list_project_runs(project_id)
    }
    unsupported = {"joint_residual_paths", "learned_joint_trajectories", "bounded_trend_corridor"}
    return sorted(
        (
            row
            for row in candidates.values()
            if row.get("snapshot_id") == snapshot_id
            and (_long_run(row) or row.get("params", {}).get("forecast_mode") not in unsupported)
        ),
        key=lambda row: row.get("created_at", ""),
        reverse=True,
    )


class LiveModel:
    """A verified saved model; live inputs do not have to be in its training snapshot."""

    def __init__(self, project_id, run_id):
        from pdm.data.project_prepare import read_zone_limits
        from pdm.project_snapshot import project_snapshot

        self.project_id, self.run_id = project_id, run_id
        snapshot = project_snapshot(project_id)
        self.schema = copy.deepcopy(snapshot["schema"])
        limits = read_zone_limits(snapshot["dir"])
        if limits:
            self.schema["thresholds"] = limits
        record = read_json(project_store().run_path(project_id, run_id) / "manifest.json")
        if record.get("snapshot_id") != snapshot["snapshot_id"]:
            raise ValueError("This model belongs to a previous data snapshot")
        self.is_long = _long_run(record)
        if self.is_long:
            from pdm.long_forecast_data import MIN_HISTORY, source_for
            from pdm.long_forecast_run import load_run

            self.source = source_for(project_id)
            self.frozen = load_run(project_id, run_id)
            self.run = self.frozen[0]
            self.cadence_s = self.source["cadence_s"]
            self.history = MIN_HISTORY
        else:
            from pdm.signal_inference import _load_model, training_cadence
            from pdm.signal_training import load_signal_run

            self.run = load_signal_run(project_id, run_id)
            if run_id not in {
                row["run_id"] for row in available_runs(project_id, snapshot["snapshot_id"])
            }:
                raise ValueError("This saved forecast protocol does not support live inputs")
            self.cadence_s = training_cadence(snapshot)
            self.history = int(self.run["params"]["history_length"])
            self.model = _load_model(
                project_id,
                run_id,
                self.run["artifacts"][self.run["artifact"]],
                str(project_store().root),
            )

    @property
    def signal_column(self):
        return str(self.schema.get("signal_column") or "signal")

    def assess(self, unit, *, use_calibration=True):
        from pdm.signal_inference import _crossing

        unit = mark_gaps(unit, self.cadence_s)
        if unit.empty or unit.unit_id.nunique() != 1:
            raise ValueError("Provide measurements for exactly one machine")
        thresholds = resolve_thresholds(self.schema, unit)
        current, issued = float(unit.signal.iloc[-1]), float(unit.timestamp_s.iloc[-1])
        result = dict(
            unit_id=str(unit.unit_id.iloc[0]),
            as_of_s=issued,
            current=current,
            zone=str(zone_labels([current], thresholds)[0]),
            n_rows=len(unit),
            thresholds=thresholds,
            observed_prefix=unit[["timestamp_s", "signal", "gap_before"]].to_dict("records"),
            points=[],
            status="unavailable",
            reason=None,
            red_in_s=None,
            crossing=dict(status="unavailable", time_s=None),
            history_observations=0,
            calibration=None,
        )
        try:
            if self.is_long:
                from pdm.long_forecast_run import forecast_observations

                forecast = forecast_observations(
                    self.project_id,
                    self.run_id,
                    unit,
                    source=self.source,
                    frozen=self.frozen,
                    use_calibration=use_calibration,
                )
                points = [
                    dict(
                        target_time_s=float(t),
                        lower=float(row[0]),
                        value=float(row[1]),
                        upper=float(row[2]),
                    )
                    for t, row in zip(forecast["times"], forecast["outputs"], strict=True)
                ]
                result.update(
                    history_observations=forecast["history_observations"],
                    input_hash=forecast["input_hash"],
                    calibration=forecast["calibration"],
                    funnel=dict(mode="bounded_trend_corridor"),
                )
            else:
                from pdm.signal_training import _predict, _segments

                segment = _segments(unit, result["unit_id"])[-1]
                if len(segment) < self.history:
                    raise ValueError(
                        f"Collecting history: {len(segment)} of {self.history} measurements"
                    )
                horizons = self.run["params"]["horizons_s"]
                x = segment.signal.to_numpy(np.float32)[-self.history :].reshape(1, self.history, 1)
                pred, lower, upper = _predict(
                    self.model,
                    self.run["engine_id"],
                    dict(x=x, y=np.zeros((1, len(horizons)), np.float32)),
                    self.run["scaler"],
                )
                points = [
                    dict(
                        target_time_s=issued + h,
                        value=float(pred[0, j]),
                        lower=float(lower[0, j]) if lower is not None else None,
                        upper=float(upper[0, j]) if upper is not None else None,
                    )
                    for j, h in enumerate(horizons)
                    if np.isfinite(pred[0, j])
                ]
                result["history_observations"] = self.history
        except ValueError as error:
            result["reason"] = str(error)
            return result
        crossing = _crossing(current, thresholds, points, issued)
        result.update(
            points=points,
            status="available" if points else "unavailable",
            crossing=crossing,
            red_in_s=None if crossing["time_s"] is None else crossing["time_s"] - issued,
        )
        return result


def assess_fleet(model, data, *, use_calibration=True):
    results = [
        model.assess(unit, use_calibration=use_calibration)
        for _, unit in data.groupby("unit_id", sort=True)
    ]
    return sorted(
        results,
        key=lambda row: (
            -ZONE_RANK[row["zone"]],
            row["red_in_s"] if row["red_in_s"] is not None else float("inf"),
            row["unit_id"],
        ),
    )
