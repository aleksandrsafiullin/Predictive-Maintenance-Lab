from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from pdm.alerts import AlertEngine, alerts_from_predictions
from pdm.baselines import age_only_baseline_rul, filter_trend_baseline
from pdm.predict import Predictor
from pdm.windows import recompute_filter_gap_before, resolve_filter_gap_params

ReplayLog = list[dict[str, Any]]


def bind_replay_to_run(dataset_id: str, run_id: str, *, force: bool = False) -> dict[str, Any]:
    """Require fingerprint match, then bind replay to the run snapshot split."""
    from pdm.evaluate import bind_evaluation_to_run
    from pdm.experiments import run_dir

    return bind_evaluation_to_run(run_dir(dataset_id, run_id), dataset_id, force=force)


class ReplaySource:
    def __init__(
        self,
        measurements: pd.DataFrame,
        *,
        gap_multiplier: float | None = None,
        sampling_interval_s: float | None = None,
    ) -> None:
        self.measurements = measurements.sort_values("timestamp_s").reset_index(drop=True)
        ds = None
        if "dataset_id" in self.measurements.columns and len(self.measurements):
            ds = str(self.measurements["dataset_id"].iloc[0])
        self._recompute_filter_gaps = ds == "filters"
        if self._recompute_filter_gaps:
            self._gap_multiplier, self._sampling_interval_s = resolve_filter_gap_params(
                gap_multiplier=gap_multiplier,
                sampling_interval_s=sampling_interval_s,
                allow_yaml_fallback=True,
            )
        else:
            self._gap_multiplier = gap_multiplier
            self._sampling_interval_s = sampling_interval_s

    def __len__(self) -> int:
        return len(self.measurements)

    def prefix(self, step: int) -> pd.DataFrame:
        """Measurements with index <= step (0-based inclusive). No future rows."""
        if step < 0:
            return self.measurements.iloc[0:0]
        out = self.measurements.iloc[: step + 1]
        if not self._recompute_filter_gaps:
            return out
        return recompute_filter_gap_before(
            out,
            gap_multiplier=self._gap_multiplier,
            sampling_interval_s=self._sampling_interval_s,
            causal=True,
        )


def replay_unit(
    measurements: pd.DataFrame,
    predictor: Predictor,
    *,
    dataset_id: str,
    unit_id: str,
    run_id: str,
    history_length: int,
    warning_horizon_s: float,
    confirmation_count: int = 3,
    reset_factor: float = 1.2,
    train_units: pd.DataFrame | None = None,
    pressure_limit_pa: float = 600.0,
    truth_units: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Predictor never receives ground-truth RUL or future rows.

    Evaluator (this function, after prediction) may join actual RUL from truth_units.
    """
    src = ReplaySource(
        measurements,
        gap_multiplier=getattr(predictor, "gap_multiplier", None),
        sampling_interval_s=getattr(predictor, "sampling_interval_s", None),
    )
    engine = AlertEngine(warning_horizon_s, confirmation_count, reset_factor)
    engine.reset_unit(unit_id)
    preds: ReplayLog = []
    alerts: ReplayLog = []
    for step in range(len(src)):
        prefix = src.prefix(step)
        row = prefix.iloc[-1]
        t = float(row["timestamp_s"])
        pred = predictor.predict_from_history(prefix)
        # Includes short prefix and gap/quality warmup from valid_history_window.
        collecting = pred.get("status") == "Collecting history"
        observed_limit = False
        if dataset_id == "filters" and float(row.get("differential_pressure", 0)) > pressure_limit_pa:
            observed_limit = True
        snap = engine.update(
            timestamp_s=t,
            predicted_rul_s=pred.get("predicted_rul_s"),
            collecting=collecting,
            observed_limit=observed_limit,
            model_id=run_id,
        )
        baseline_rul = None
        if dataset_id == "bearings" and train_units is not None:
            baseline_rul = age_only_baseline_rul(
                float(row["operating_age_s"]),
                str(row.get("regime_id")) if "regime_id" in row else None,
                train_units,
            )
        elif dataset_id == "filters":
            hist = prefix.tail(history_length)
            tr = filter_trend_baseline(
                hist["timestamp_s"].to_numpy(),
                hist["differential_pressure"].to_numpy(),
                limit_pa=pressure_limit_pa,
            )
            baseline_rul = tr["predicted_rul_s"]
        rec = {
            "run_id": run_id,
            "unit_id": unit_id,
            "timestamp_s": t,
            "input_until_s": t,
            "predicted_rul_s": pred.get("predicted_rul_s"),
            "baseline_rul_s": baseline_rul,
            "prediction_status": pred.get("status") or "",
            "valid_history_reason": pred.get("valid_history_reason") or "",
            "observed_limit_reached": bool(observed_limit),
            "alert_status": snap["status"],
            "step": step,
        }
        if "differential_pressure" in row.index and pd.notna(row.get("differential_pressure")):
            rec["differential_pressure"] = float(row["differential_pressure"])
        if "time_original" in row.index and pd.notna(row.get("time_original")):
            rec["time_original"] = float(row["time_original"])
        # Evaluator-only actual RUL: computed after the prediction, never passed to Predictor.
        if truth_units is not None:
            meta = truth_units.set_index("unit_id").loc[unit_id]
            event_observed = int(meta.get("event_observed", 1 if dataset_id == "bearings" else 0))
            et = meta.get("event_time_s")
            if event_observed and et is not None and np.isfinite(et) and t < float(et):
                rec["actual_rul_s"] = float(et) - t
        preds.append(rec)
        if snap["opened_episode"]:
            alerts.append(snap["opened_episode"] | {"run_id": run_id})
    return {"predictions": pd.DataFrame(preds), "alerts": pd.DataFrame(alerts)}


def replay_session_key(
    dataset_id: str,
    run_id: str,
    unit_id: str,
    eval_id: str | None,
    policy_hash: str,
) -> tuple[str, str, str, str, str]:
    """Composite replay identity. Change resets step and clears session alerts."""
    return (
        str(dataset_id),
        str(run_id),
        str(unit_id),
        str(eval_id or ""),
        str(policy_hash or ""),
    )


def slice_predictions_to_replay_time(pred: pd.DataFrame, replay_time_s: float) -> pd.DataFrame:
    """Keep prediction rows with timestamp_s ≤ replay time. No future rows."""
    if pred is None or getattr(pred, "empty", True) or "timestamp_s" not in pred.columns:
        return pred.copy() if isinstance(pred, pd.DataFrame) else pd.DataFrame()
    ts = pd.to_numeric(pred["timestamp_s"], errors="coerce")
    return pred.loc[ts <= float(replay_time_s)].copy()


def filter_replay_alert_log(
    alerts: pd.DataFrame,
    *,
    unit_id: str,
    replay_time_s: float,
) -> pd.DataFrame:
    """Episodes for the selected unit at or before replay time. No future rows."""
    if alerts is None or not isinstance(alerts, pd.DataFrame):
        return pd.DataFrame()
    if alerts.empty:
        return alerts.copy()
    out = alerts
    if "unit_id" in out.columns:
        out = out[out["unit_id"].astype(str) == str(unit_id)]
    if "timestamp_s" in out.columns:
        ts = pd.to_numeric(out["timestamp_s"], errors="coerce")
        out = out.loc[ts <= float(replay_time_s)]
    return out.reset_index(drop=True)


def alert_status_at_time(
    steps: pd.DataFrame,
    *,
    unit_id: str,
    timestamp_s: float,
) -> str:
    """Latest engine status for this unit at or before t. Does not peek ahead."""
    if steps is None or not isinstance(steps, pd.DataFrame) or steps.empty:
        return "—"
    unit = steps[steps["unit_id"].astype(str) == str(unit_id)] if "unit_id" in steps.columns else steps
    if unit.empty or "timestamp_s" not in unit.columns:
        return "—"
    ts = pd.to_numeric(unit["timestamp_s"], errors="coerce")
    past = unit.loc[ts <= float(timestamp_s)]
    if past.empty:
        return "—"
    status = past.iloc[-1].get("status")
    return str(status) if status is not None and not (isinstance(status, float) and pd.isna(status)) else "—"


def rescore_replay_alerts(
    predictions: pd.DataFrame,
    policy: Mapping[str, Any],
    *,
    run_id: str | None = None,
    measurements: pd.DataFrame | None = None,
    pressure_limit_pa: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-run AlertEngine on frozen predicted_rul_s. Never calls the model or writes files.

    `pressure_limit_pa` is the run/config Δp threshold (same source as the plot line).
    Do not omit it in the UI and rely on a hardcoded Pa default.
    """
    if predictions is None or getattr(predictions, "empty", True):
        return alerts_from_predictions(
            predictions, policy, run_id=run_id, pressure_limit_pa=pressure_limit_pa
        )
    pred = predictions.copy()
    if (
        measurements is not None
        and not measurements.empty
        and "differential_pressure" in measurements.columns
        and "differential_pressure" not in pred.columns
        and "unit_id" in pred.columns
        and "timestamp_s" in pred.columns
    ):
        extra = measurements[["unit_id", "timestamp_s", "differential_pressure"]].copy()
        extra["unit_id"] = extra["unit_id"].astype(str)
        pred["unit_id"] = pred["unit_id"].astype(str)
        pred = pred.merge(extra, on=["unit_id", "timestamp_s"], how="left")
    return alerts_from_predictions(
        pred, policy, run_id=run_id, pressure_limit_pa=pressure_limit_pa
    )


END_OF_OBSERVED_DATA = "End of observed data"
PRESSURE_LIMIT_PA = 600.0
_VISIBLE_LOG_DROP = ("actual_rul_s",)


def _finite_scalar(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


def _meta_get(unit_meta: Mapping[str, Any] | pd.Series | None, key: str, default: Any = None) -> Any:
    if unit_meta is None:
        return default
    if isinstance(unit_meta, pd.Series):
        return unit_meta[key] if key in unit_meta.index else default
    return unit_meta.get(key, default)


def split_label_for_unit(split: Mapping[str, Any] | None, unit_id: str) -> str:
    """train / validation / test for the selected unit. Not a model input."""
    if not split:
        return "—"
    uid = str(unit_id)
    for part in ("train", "validation", "test"):
        ids = [str(x) for x in (split.get(part) or [])]
        if uid in ids:
            return part
    return "—"


def sensor_limit_reached(
    measurements: pd.DataFrame | None,
    *,
    pressure_limit_pa: float = PRESSURE_LIMIT_PA,
) -> bool:
    """True only when measured Δp exceeded the limit. Official RUL does not count."""
    if measurements is None or getattr(measurements, "empty", True):
        return False
    if "differential_pressure" not in measurements.columns:
        return False
    dp = pd.to_numeric(measurements["differential_pressure"], errors="coerce")
    return bool((dp > float(pressure_limit_pa)).any())


def unit_event_observed(unit_meta: Mapping[str, Any] | pd.Series | None) -> bool:
    raw = _meta_get(unit_meta, "event_observed", 0)
    val = _finite_scalar(raw)
    if val is None:
        return False
    return int(val) != 0


def official_rul_annotation_s(unit_meta: Mapping[str, Any] | pd.Series | None) -> float | None:
    """Official test RUL: evaluation annotation only, never a sensor event."""
    return _finite_scalar(_meta_get(unit_meta, "official_rul_at_prefix_end_s"))


def observation_end_s(
    unit_meta: Mapping[str, Any] | pd.Series | None,
    measurements: pd.DataFrame | None = None,
) -> float | None:
    obs = _finite_scalar(_meta_get(unit_meta, "observation_end_s"))
    if obs is not None:
        return obs
    if measurements is None or getattr(measurements, "empty", True):
        return None
    if "timestamp_s" not in measurements.columns:
        return None
    ts = pd.to_numeric(measurements["timestamp_s"], errors="coerce")
    if ts.notna().any():
        return float(ts.max())
    return None


def replay_end_state(
    *,
    dataset_id: str,
    unit_meta: Mapping[str, Any] | pd.Series | None,
    replay_time_s: float,
    prefix_measurements: pd.DataFrame | None = None,
    pressure_limit_pa: float = PRESSURE_LIMIT_PA,
    at_last_sample: bool = False,
    time_eps_s: float = 1e-9,
) -> dict[str, Any]:
    """Honest end-of-unit copy. Never invents a 600 Pa crossing from official RUL."""
    # Use unit meta, not the visible prefix — prefix max would look like "end" at every step.
    obs = _finite_scalar(_meta_get(unit_meta, "observation_end_s"))
    event = unit_event_observed(unit_meta)
    official = official_rul_annotation_s(unit_meta)
    sensor_600 = dataset_id == "filters" and sensor_limit_reached(
        prefix_measurements, pressure_limit_pa=pressure_limit_pa
    )
    t = _finite_scalar(replay_time_s)
    at_obs_end = bool(at_last_sample)
    if t is not None and obs is not None:
        at_obs_end = at_obs_end or t >= float(obs) - float(time_eps_s)
    message = None
    kind = "in_progress"
    if at_obs_end:
        if event:
            kind = "event_reached"
            message = "Observed 600 Pa event" if dataset_id == "filters" else "Last recorded sample"
        else:
            kind = "end_of_observed_data"
            message = END_OF_OBSERVED_DATA
    return {
        "at_observation_end": at_obs_end,
        "event_observed": event,
        "sensor_limit_reached": sensor_600,
        "message": message,
        "kind": kind,
        "official_rul_s": official,
        "observation_end_s": obs,
    }


def evaluator_overlay_rul(
    timestamps_s: pd.Series | np.ndarray,
    unit_meta: Mapping[str, Any] | pd.Series | None,
    *,
    dataset_id: str,
) -> np.ndarray:
    """Actual RUL overlay from an observed event_time only.

    Censored / official-RUL units return NaN. Official RUL is not a trajectory.
    Does not modify predicted_rul_s.
    """
    ts = np.asarray(timestamps_s, dtype=np.float64)
    out = np.full(ts.shape, np.nan, dtype=np.float64)
    if ts.size == 0 or unit_meta is None:
        return out
    et = _finite_scalar(_meta_get(unit_meta, "event_time_s"))
    if et is None:
        return out
    if dataset_id == "filters" and not unit_event_observed(unit_meta):
        return out
    mask = np.isfinite(ts) & (ts < float(et))
    out[mask] = float(et) - ts[mask]
    return out


def active_warning_episode_time_s(
    episodes: pd.DataFrame | None,
    *,
    unit_id: str,
    replay_time_s: float,
    steps: pd.DataFrame | None = None,
) -> float | None:
    """Confirmed warning timestamp if a warning is active at t. None otherwise."""
    if steps is not None and not getattr(steps, "empty", True) and "timestamp_s" in steps.columns:
        unit = steps[steps["unit_id"].astype(str) == str(unit_id)] if "unit_id" in steps.columns else steps
        if unit.empty:
            return None
        ts = pd.to_numeric(unit["timestamp_s"], errors="coerce")
        past = unit.loc[ts <= float(replay_time_s)]
        if past.empty:
            return None
        last = past.iloc[-1]
        active = False
        if "warning_active" in past.columns:
            active = bool(last.get("warning_active"))
        else:
            active = str(last.get("status") or "") == "Warning"
        if not active:
            return None
    else:
        status = alert_status_at_time(
            steps if steps is not None else pd.DataFrame(),
            unit_id=unit_id,
            timestamp_s=replay_time_s,
        )
        if status != "Warning":
            return None
    live = filter_replay_alert_log(
        episodes if episodes is not None else pd.DataFrame(),
        unit_id=unit_id,
        replay_time_s=replay_time_s,
    )
    if live.empty or "timestamp_s" not in live.columns:
        return None
    t_ep = _finite_scalar(live.iloc[-1].get("timestamp_s"))
    return t_ep


def visible_replay_slice(
    predictions: pd.DataFrame | None,
    alerts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Visible issued-forecast / alert log already capped at replay time. Drops GT."""
    frames: list[pd.DataFrame] = []
    pred = predictions.copy() if isinstance(predictions, pd.DataFrame) else pd.DataFrame()
    if not pred.empty:
        drop = [c for c in _VISIBLE_LOG_DROP if c in pred.columns]
        if drop:
            pred = pred.drop(columns=drop)
        pred.insert(0, "record_type", "prediction")
        frames.append(pred)
    al = alerts.copy() if isinstance(alerts, pd.DataFrame) else pd.DataFrame()
    if not al.empty:
        al.insert(0, "record_type", "alert")
        frames.append(al)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)
