from __future__ import annotations

from typing import Any, Callable

import pandas as pd

from pdm.alerts import AlertEngine
from pdm.baselines import age_only_baseline_rul, filter_trend_baseline
from pdm.predict import Predictor

ReplayLog = list[dict[str, Any]]


class ReplaySource:
    def __init__(self, measurements: pd.DataFrame) -> None:
        self.measurements = measurements.sort_values("timestamp_s").reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.measurements)

    def prefix(self, step: int) -> pd.DataFrame:
        """Measurements with index <= step (0-based inclusive). No future rows."""
        if step < 0:
            return self.measurements.iloc[0:0]
        return self.measurements.iloc[: step + 1]


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
    src = ReplaySource(measurements)
    engine = AlertEngine(warning_horizon_s, confirmation_count, reset_factor)
    engine.reset_unit(unit_id)
    preds: ReplayLog = []
    alerts: ReplayLog = []
    for step in range(len(src)):
        prefix = src.prefix(step)
        row = prefix.iloc[-1]
        t = float(row["timestamp_s"])
        collecting = len(prefix) < history_length
        pred = predictor.predict_from_history(prefix) if not collecting else {
            "predicted_rul_s": None,
            "status": "Collecting history",
            "n_history": len(prefix),
        }
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
            "alert_status": snap["status"],
            "step": step,
        }
        # Evaluator-only actual RUL: computed after the prediction, never passed to Predictor.
        if truth_units is not None:
            meta = truth_units.set_index("unit_id").loc[unit_id]
            et = meta.get("event_time_s")
            try:
                import math

                if et is not None and et == et and not (isinstance(et, float) and math.isinf(et)):
                    rec["actual_rul_s"] = float(et) - t
            except Exception:
                pass
        preds.append(rec)
        if snap["opened_episode"]:
            alerts.append(snap["opened_episode"] | {"run_id": run_id})
    return {"predictions": pd.DataFrame(preds), "alerts": pd.DataFrame(alerts)}
