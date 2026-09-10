from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pdm.alerts import AlertEngine, classify_alert_timing
from pdm.baselines import age_only_baseline_rul, filter_trend_baseline


def mae(y_hat: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(y_hat) - np.asarray(y)
    return float(np.mean(np.abs(d))) if d.size else float("nan")


def rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(y_hat) - np.asarray(y)
    return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")


def mean_overestimation(y_hat: np.ndarray, y: np.ndarray) -> float:
    d = np.maximum(np.asarray(y_hat) - np.asarray(y), 0.0)
    return float(np.mean(d)) if d.size else float("nan")


def summarize_rul_table(pred: pd.DataFrame, actual_col: str = "actual_rul_s", pred_col: str = "predicted_rul_s") -> dict[str, Any]:
    sub = pred.dropna(subset=[actual_col, pred_col])
    if sub.empty:
        return {"n_points": 0, "n_units": 0, "mae": None, "rmse": None, "mean_overestimation": None, "unit_mae": {}}
    y = sub[actual_col].to_numpy(dtype=np.float64)
    yhat = sub[pred_col].to_numpy(dtype=np.float64)
    unit_mae = {}
    for uid, g in sub.groupby("unit_id"):
        unit_mae[str(uid)] = mae(g[pred_col], g[actual_col])
    equal_unit_mae = float(np.mean(list(unit_mae.values()))) if unit_mae else float("nan")
    return {
        "n_points": int(len(sub)),
        "n_units": int(sub["unit_id"].nunique()),
        "mae": mae(yhat, y),
        "rmse": rmse(yhat, y),
        "mean_overestimation": mean_overestimation(yhat, y),
        "unit_mae": unit_mae,
        "equal_weight_unit_mae": equal_unit_mae,
    }


def compare_baseline(pred: pd.DataFrame) -> dict[str, Any]:
    both = pred.dropna(subset=["predicted_rul_s", "baseline_rul_s", "actual_rul_s"])
    nn_all = summarize_rul_table(pred)
    nn_overlap = summarize_rul_table(both)
    base = summarize_rul_table(both, pred_col="baseline_rul_s")
    n_base = int(pred["baseline_rul_s"].notna().sum()) if "baseline_rul_s" in pred.columns else 0
    n_all = int(len(pred))
    return {
        "neural_net_all_points": nn_all,
        "neural_net_baseline_overlap": nn_overlap,
        "baseline_overlap": base,
        "baseline_coverage_points": n_base,
        "baseline_coverage_fraction": (n_base / n_all) if n_all else 0.0,
    }


def default_horizon_s(train_units: pd.DataFrame, fraction: float = 0.10) -> float:
    durs = train_units["observation_end_s"].to_numpy(dtype=np.float64)
    if durs.size == 0:
        return 3600.0
    raw = float(np.median(durs)) * fraction
    # round to a convenient magnitude
    if raw >= 3600:
        return float(max(3600, round(raw / 3600) * 3600))
    if raw >= 60:
        return float(max(60, round(raw / 60) * 60))
    return float(max(1.0, round(raw)))


def attach_actual_rul(pred: pd.DataFrame, units: pd.DataFrame, dataset_id: str) -> pd.DataFrame:
    meta = units.set_index("unit_id")
    actual = []
    for _, row in pred.iterrows():
        u = meta.loc[row["unit_id"]]
        t = float(row["timestamp_s"])
        if dataset_id == "bearings":
            et = float(u["event_time_s"])
            actual.append(et - t if t < et else np.nan)
        else:
            et = u.get("event_time_s")
            if et is not None and np.isfinite(et):
                actual.append(float(et) - t)
            else:
                actual.append(np.nan)
    out = pred.copy()
    out["actual_rul_s"] = actual
    return out


def evaluate_run(
    dataset_id: str,
    run_id: str,
    *,
    warning_horizon_s: float | None = None,
    confirmation_count: int = 3,
    device: str = "cpu",
) -> dict[str, Any]:
    from pdm.config import load_dataset_config
    from pdm.data.prepare import load_processed
    from pdm.device import resolve_device
    from pdm.io_util import atomic_write_json
    from pdm.paths import dataset_runs
    from pdm.predict import Predictor
    from pdm.replay import replay_unit
    from pdm.train import load_trained_model

    processed = load_processed(dataset_id)
    features = processed["features"]
    units = processed["units"]
    split = processed["split"]
    cfg = load_dataset_config(dataset_id)
    rdir = dataset_runs(dataset_id) / run_id
    dev = resolve_device(device)
    model, prep, meta = load_trained_model(rdir, device=dev.torch_device, which="best")
    predictor = Predictor(model, prep, history_length=int(meta["history_length"]), device=dev.torch_device)
    train_units = units[units["unit_id"].isin(split["train"])]
    h = warning_horizon_s if warning_horizon_s is not None else default_horizon_s(
        train_units, float(cfg.get("alerts", {}).get("horizon_fraction_of_median_train", 0.10))
    )
    k = int(confirmation_count)
    test_ids = list(split["test"])
    pred_frames = []
    alert_frames = []
    for uid in test_ids:
        meas = features[features["unit_id"] == uid]
        out = replay_unit(
            meas,
            predictor,
            dataset_id=dataset_id,
            unit_id=uid,
            run_id=run_id,
            history_length=int(meta["history_length"]),
            warning_horizon_s=float(h),
            confirmation_count=k,
            reset_factor=float(cfg.get("alerts", {}).get("reset_factor", 1.2)),
            train_units=train_units,
            pressure_limit_pa=float(cfg.get("pressure_limit_pa", 600.0)),
            truth_units=units,
        )
        pred_frames.append(out["predictions"])
        if len(out["alerts"]):
            alert_frames.append(out["alerts"])
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    alerts = pd.concat(alert_frames, ignore_index=True) if alert_frames else pd.DataFrame()
    preds.to_csv(rdir / "predictions.csv", index=False)
    alerts.to_csv(rdir / "alerts.csv", index=False)

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "warning_horizon_s": float(h),
        "confirmation_count": k,
        "n_test_units": len(test_ids),
        "note": "Test metrics are not used for epoch selection or preprocessing.",
    }
    if dataset_id == "filters":
        last_pts = preds.sort_values("timestamp_s").groupby("unit_id").tail(1)
        metrics["primary_prefix_end_rul"] = summarize_rul_table(last_pts)
        metrics["additional_backtest_all_prefix_points"] = summarize_rul_table(preds)
        metrics["primary_note"] = (
            "Primary comparable result is RUL at the last available point of each author test prefix."
        )
    else:
        metrics["test_rul"] = summarize_rul_table(preds)
    metrics["baseline"] = compare_baseline(preds)
    metrics["alerts"] = _alert_summary(alerts, units, float(h), preds)
    atomic_write_json(rdir / "test_metrics.json", metrics)
    return metrics


def _alert_summary(alerts: pd.DataFrame, units: pd.DataFrame, horizon_s: float, preds: pd.DataFrame) -> dict[str, Any]:
    meta = units.set_index("unit_id")
    timely = early = none = insufficient = 0
    for uid in preds["unit_id"].unique() if len(preds) else []:
        u = meta.loc[uid]
        et = u.get("event_time_s")
        obs_end = float(u["observation_end_s"])
        if et is None or not np.isfinite(et):
            continue
        coverage_ok = obs_end >= (float(et) - horizon_s)
        unit_alerts = alerts[alerts["unit_id"] == uid] if len(alerts) and "unit_id" in alerts.columns else pd.DataFrame()
        if unit_alerts.empty:
            if not coverage_ok:
                insufficient += 1
            else:
                none += 1
            continue
        t0 = float(unit_alerts.iloc[0]["timestamp_s"])
        kind = classify_alert_timing(t0, float(et), horizon_s)
        if kind == "timely":
            timely += 1
        elif kind == "early_relative_to_selected_horizon":
            early += 1
        else:
            none += 1
    n_pred = len(preds)
    n_warn = int((preds["alert_status"] == "Warning").sum()) if n_pred and "alert_status" in preds.columns else 0
    return {
        "timely_warnings": timely,
        "early_relative_to_selected_horizon": early,
        "units_without_timely_warning": none,
        "insufficient_observed_coverage": insufficient,
        "fraction_time_in_warning": (n_warn / n_pred) if n_pred else 0.0,
        "denominator_units_scored": timely + early + none,
    }
