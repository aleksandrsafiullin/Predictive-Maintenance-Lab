from __future__ import annotations

import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from pdm.alerts import (
    ALERT_EPISODE_COLUMNS,
    alerts_from_predictions,
    classify_alert_outcome,
    confirmed_lead_time_s,
    copy_alert_policy_into_evaluation_config,
    has_sufficient_coverage,
    require_frozen_alert_policy,
    resolve_alert_policy,
)
from pdm.baselines import coverage_stats
from pdm.experiments import EVALUATIONS_DIRNAME

FINGERPRINT_COMPARE_FIELDS = (
    "dataset_version",
    "split_hash",
    "features_hash",
    "units_hash",
    "feature_pipeline_version",
    "checkpoint_hash",
)
_JSON_COMPARE_FIELDS = (
    "dataset_version",
    "split_hash",
    "features_hash",
    "units_hash",
    "feature_pipeline_version",
)
_PROCESSED_FILE_HASH_FIELDS = (
    ("features.parquet", "features_hash"),
    ("units.parquet", "units_hash"),
    ("split.json", "split_json_hash"),
    ("feature_schema.json", "feature_schema_hash"),
)


class IncompatibleDataError(RuntimeError):
    """Run snapshot does not match the current processed dataset."""

    def __init__(self, differing: list[str], detail: str = "") -> None:
        self.differing = list(differing)
        fields = ", ".join(self.differing) if self.differing else "(unknown)"
        msg = (
            "Incompatible data: run snapshot does not match the current processed dataset. "
            f"Differing fields: {fields}"
        )
        if detail:
            msg = f"{msg}. {detail}"
        super().__init__(msg)


DEFAULT_BEARINGS_NEAR_EVENT_ZONES_S = (3600.0, 1800.0, 600.0)
EXPERIMENT_SNAPSHOT_NAME = "experiment_snapshot.json"
_LEGACY_PRESSURE_LIMIT_PA = 600.0


def mae(y_hat: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(y_hat) - np.asarray(y)
    return float(np.mean(np.abs(d))) if d.size else float("nan")


def rmse(y_hat: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(y_hat) - np.asarray(y)
    return float(np.sqrt(np.mean(d * d))) if d.size else float("nan")


def mean_overestimation(y_hat: np.ndarray, y: np.ndarray) -> float:
    d = np.maximum(np.asarray(y_hat) - np.asarray(y), 0.0)
    return float(np.mean(d)) if d.size else float("nan")


def _finite_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if frame.empty or col not in frame.columns:
        return pd.Series(np.zeros(len(frame), dtype=bool), index=frame.index)
    arr = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=np.float64)
    return pd.Series(np.isfinite(arr), index=frame.index)


def _positive_actual_mask(frame: pd.DataFrame, actual_col: str = "actual_rul_s") -> pd.Series:
    """Finite actual RUL strictly before the event (endpoint rows are NaN / not > 0)."""
    if frame.empty or actual_col not in frame.columns:
        return pd.Series(np.zeros(len(frame), dtype=bool), index=frame.index)
    arr = pd.to_numeric(frame[actual_col], errors="coerce").to_numpy(dtype=np.float64)
    return pd.Series(np.isfinite(arr) & (arr > 0.0), index=frame.index)


def rul_score_frame(
    pred: pd.DataFrame,
    *,
    actual_col: str = "actual_rul_s",
    pred_col: str = "predicted_rul_s",
    max_actual_s: float | None = None,
) -> pd.DataFrame:
    """Rows eligible for RUL MAE: finite actual & pred, 0 < actual (endpoints excluded)."""
    if pred.empty or actual_col not in pred.columns or pred_col not in pred.columns:
        return pred.iloc[0:0].copy() if len(pred) else pred.copy()
    actual = pd.to_numeric(pred[actual_col], errors="coerce").to_numpy(dtype=np.float64)
    ok = _positive_actual_mask(pred, actual_col) & _finite_series(pred, pred_col)
    if max_actual_s is not None:
        ok = ok & (actual <= float(max_actual_s))
    return pred.loc[ok].copy()


def summarize_rul_table(
    pred: pd.DataFrame,
    actual_col: str = "actual_rul_s",
    pred_col: str = "predicted_rul_s",
    max_actual_s: float | None = None,
) -> dict[str, Any]:
    sub = rul_score_frame(pred, actual_col=actual_col, pred_col=pred_col, max_actual_s=max_actual_s)
    empty = {
        "n_points": 0,
        "n_units": 0,
        "mae": None,
        "rmse": None,
        "mean_overestimation": None,
        "unit_mae": {},
        "equal_weight_unit_mae": None,
    }
    if sub.empty:
        return empty
    y = sub[actual_col].to_numpy(dtype=np.float64)
    yhat = sub[pred_col].to_numpy(dtype=np.float64)
    unit_mae: dict[str, float] = {}
    for uid, g in sub.groupby("unit_id"):
        unit_mae[str(uid)] = mae(g[pred_col], g[actual_col])
    equal_unit_mae = float(np.mean(list(unit_mae.values()))) if unit_mae else None
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
    """NN vs baseline on the same finite-actual rows; coverage is vs that reference.

    Do not treat baseline's sparse hits as comparable to the NN's full scored set.
    """
    empty = summarize_rul_table(pred.iloc[0:0] if len(pred) else pred)
    if pred.empty:
        return {
            "neural_net_all_points": empty,
            "neural_net_baseline_overlap": empty,
            "baseline_overlap": empty,
            "baseline_coverage_points": 0,
            "baseline_coverage_fraction": 0.0,
            "n_reference_points": 0,
        }
    actual_ok = _positive_actual_mask(pred, "actual_rul_s")
    reference = pred.loc[actual_ok]
    n_ref = int(len(reference))
    base_vals = reference["baseline_rul_s"] if "baseline_rul_s" in reference.columns else []
    cov = coverage_stats(base_vals, n_reference=n_ref)
    if n_ref and "predicted_rul_s" in reference.columns and "baseline_rul_s" in reference.columns:
        both = reference.loc[
            _finite_series(reference, "predicted_rul_s") & _finite_series(reference, "baseline_rul_s")
        ]
    else:
        both = reference.iloc[0:0]
    return {
        "neural_net_all_points": summarize_rul_table(reference),
        "neural_net_baseline_overlap": summarize_rul_table(both),
        "baseline_overlap": summarize_rul_table(both, pred_col="baseline_rul_s"),
        "baseline_coverage_points": cov["n_finite"],
        "baseline_coverage_fraction": cov["coverage_fraction"],
        "n_reference_points": n_ref,
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
        elif int(u.get("event_observed", 0)):
            et = u.get("event_time_s")
            if et is not None and np.isfinite(et) and t < float(et):
                actual.append(float(et) - t)
            else:
                actual.append(np.nan)
        else:
            actual.append(np.nan)
    out = pred.copy()
    out["actual_rul_s"] = actual
    return out


def zone_key(zone_s: float) -> str:
    z = float(zone_s)
    if z.is_integer():
        return str(int(z))
    return str(z)


def load_near_event_zones_s(cfg: dict[str, Any] | None) -> list[float]:
    """Frozen config zones. Never derived from test residuals."""
    raw = ((cfg or {}).get("evaluation") or {}).get("near_event_zones_s")
    if not raw:
        nested = (cfg or {}).get("config")
        if isinstance(nested, dict):
            raw = (nested.get("evaluation") or {}).get("near_event_zones_s")
    if not raw:
        return [float(z) for z in DEFAULT_BEARINGS_NEAR_EVENT_ZONES_S]
    return [float(z) for z in raw]


def load_experiment_snapshot(rdir: Path) -> dict[str, Any] | None:
    """Resolved task snapshot written at train. None if the run is legacy."""
    path = Path(rdir) / EXPERIMENT_SNAPSHOT_NAME
    if not path.exists():
        return None
    from pdm.io_util import read_json

    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"experiment_snapshot.json must be an object: {path}")
    return data


def _pressure_limit_from_mapping(blob: Mapping[str, Any] | None) -> float | None:
    if not blob:
        return None
    val = blob.get("pressure_limit_pa")
    if val is None:
        nested = blob.get("config")
        if isinstance(nested, Mapping):
            val = nested.get("pressure_limit_pa")
    if val is None:
        return None
    return float(val)


def load_run_pressure_limit_pa(rdir: Path, fallback: float | None = None) -> float:
    """Snapshot Δp threshold when present; else caller fallback (legacy 04 YAML)."""
    snap = load_experiment_snapshot(rdir)
    if snap is not None:
        val = _pressure_limit_from_mapping(snap)
        if val is not None:
            return val
    if fallback is None:
        return float(_LEGACY_PRESSURE_LIMIT_PA)
    return float(fallback)


def resolve_run_task_config(
    rdir: Path, live_cfg: Mapping[str, Any] | None = None
) -> tuple[dict[str, Any], bool]:
    """Task config for evaluate. Snapshot wins; live YAML only if the file is missing."""
    live = dict(live_cfg or {})
    snap = load_experiment_snapshot(rdir)
    if snap is None:
        return live, True
    nested = snap.get("config")
    if isinstance(nested, dict):
        merged = dict(nested)
        for key, value in snap.items():
            if key != "config":
                merged[key] = value
        return merged, False
    return dict(snap), False


def equal_weight_unit_mae_by_zone(
    pred: pd.DataFrame,
    zones_s: list[float],
    *,
    actual_col: str = "actual_rul_s",
    pred_col: str = "predicted_rul_s",
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for zone in zones_s:
        stats = summarize_rul_table(pred, actual_col=actual_col, pred_col=pred_col, max_actual_s=float(zone))
        val = stats.get("equal_weight_unit_mae")
        out[zone_key(zone)] = None if val is None else float(val)
    return out


def last_row_per_unit(pred: pd.DataFrame) -> pd.DataFrame:
    if pred.empty or "unit_id" not in pred.columns:
        return pred.iloc[0:0].copy() if len(pred) else pred.copy()
    ordered = pred.sort_values(["unit_id", "timestamp_s"], kind="mergesort")
    return ordered.groupby("unit_id", as_index=False, sort=False).tail(1).copy()


def _official_rul_map(units: pd.DataFrame) -> dict[str, float]:
    if units.empty or "unit_id" not in units.columns:
        return {}
    if "official_rul_at_prefix_end_s" not in units.columns:
        return {}
    out: dict[str, float] = {}
    for _, row in units.iterrows():
        uid = str(row["unit_id"])
        try:
            val = float(row.get("official_rul_at_prefix_end_s"))
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            out[uid] = val
    return out


def filter_prefix_end_table(pred: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Last sensor row per author test prefix; actual = official_rul_at_prefix_end_s.

    Official RUL is eval-only and is never a model input.
    """
    ends = last_row_per_unit(pred)
    if ends.empty:
        return ends
    official = _official_rul_map(units)
    actual = [official.get(str(uid), np.nan) for uid in ends["unit_id"]]
    out = ends.copy()
    out["official_rul_at_prefix_end_s"] = actual
    out["actual_rul_s"] = actual
    return out


def filter_prefix_backtest_table(pred: pd.DataFrame, units: pd.DataFrame) -> pd.DataFrame:
    """Remaining-life trajectory from official prefix-end RUL. Eval-only backtest."""
    if pred.empty:
        return pred.copy()
    official = _official_rul_map(units)
    ends = last_row_per_unit(pred)
    t_last = {
        str(row["unit_id"]): float(row["timestamp_s"]) for _, row in ends.iterrows()
    }
    actual = []
    for _, row in pred.iterrows():
        uid = str(row["unit_id"])
        off = official.get(uid)
        last = t_last.get(uid)
        if off is None or last is None:
            actual.append(np.nan)
        else:
            actual.append(float(off) + (last - float(row["timestamp_s"])))
    out = pred.copy()
    out["actual_rul_s"] = actual
    return out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    return obj


def _abs_error(pred_v: Any, actual_v: Any) -> float | None:
    try:
        p = float(pred_v)
        a = float(actual_v)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(p) and math.isfinite(a)):
        return None
    return abs(p - a)


def _metrics_identity(
    *,
    eval_id: str,
    run_id: str,
    dataset_id: str,
    split: dict[str, Any],
    test_ids: list[str],
    n_prediction_rows: int,
    bound: dict[str, Any],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "eval_id": eval_id,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "n_test_units": len(test_ids),
        "n_prediction_rows": int(n_prediction_rows),
        "split_protocol": split.get("protocol"),
        "metrics_version": METRICS_VERSION,
        "note": "Test metrics are not used for epoch selection or preprocessing.",
    }
    if dataset_id == "filters":
        metrics.update(filter_time_scale_from_bound(bound))
        metrics["metric_time_unit"] = "s"
    if bound.get("forced"):
        metrics["compatibility_warning"] = (
            "Forced evaluation despite fingerprint mismatch. "
            f"Differing fields: {', '.join(bound['differing'])}"
        )
    return metrics


def _optional_number(value: Any, *, as_int: bool = False) -> float | int | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return int(x) if as_int else x


def _history_row_for_epoch(rdir: Path, epoch: int | None) -> dict[str, Any] | None:
    if epoch is None:
        return None
    path = Path(rdir) / "training_history.csv"
    if not path.exists():
        return None
    hist = pd.read_csv(path)
    if hist.empty or "epoch" not in hist.columns:
        return None
    epochs = pd.to_numeric(hist["epoch"], errors="coerce")
    match = hist.loc[epochs == int(epoch)]
    if match.empty:
        return None
    return match.iloc[-1].to_dict()


def _filter_validation_block(rdir: Path | None) -> dict[str, Any] | None:
    """Censored val NLL + observed-event MAE from the best.pt selection epoch, not last."""
    if rdir is None:
        return None
    path = Path(rdir) / "validation_metrics.json"
    if not path.exists():
        return None
    from pdm.io_util import read_json

    vm = read_json(path)
    last = dict(vm.get("last") or {})
    best_epoch = _optional_number(vm.get("best_epoch"), as_int=True)
    nll = _optional_number(vm.get("best_metric"))
    row = _history_row_for_epoch(rdir, int(best_epoch) if best_epoch is not None else None)
    mae = _optional_number(row.get("val_mae_events")) if row else None
    n_events = _optional_number(row.get("n_val_event_units"), as_int=True) if row else None
    if mae is None:
        last_sel = _optional_number(last.get("selection_metric", last.get("val_loss")))
        if nll is not None and last_sel is not None and abs(float(last_sel) - float(nll)) < 1e-8:
            mae = _optional_number(last.get("val_mae_events"))
            n_events = _optional_number(last.get("n_val_event_units"), as_int=True)
    return {
        "nll_all_units": nll,
        "mae_observed_events": mae,
        "n_observed_event_units": n_events,
        "n_val_windows": vm.get("n_val_windows"),
        "selection_metric_name": vm.get("selection_metric_name"),
        "selection_metric_unit": vm.get("selection_metric_unit"),
        "best_epoch": best_epoch,
        "best_metric": nll,
        "checkpoint_selection_epoch": best_epoch,
        "note": (
            "NLL (best_metric) and MAE are from the checkpoint-selection epoch "
            f"(best.pt / best_epoch={best_epoch}), not the last training epoch. "
            "NLL on all validation units; MAE only on observed 600 Pa events."
        ),
    }


def _bearings_by_unit(pred: pd.DataFrame, test_ids: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for uid in test_ids:
        g = pred[pred["unit_id"] == uid] if "unit_id" in pred.columns else pred.iloc[0:0]
        stats = summarize_rul_table(g)
        overlap = compare_baseline(g)
        rows.append(
            {
                "unit_id": uid,
                "n_points": stats["n_points"],
                "mae": stats["mae"],
                "rmse": stats["rmse"],
                "mean_overestimation": stats["mean_overestimation"],
                "baseline_mae": overlap["baseline_overlap"]["mae"],
                "baseline_n_points": overlap["baseline_overlap"]["n_points"],
                "baseline_coverage_fraction": overlap["baseline_coverage_fraction"],
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "n_points",
                "mae",
                "rmse",
                "mean_overestimation",
                "baseline_mae",
                "baseline_n_points",
                "baseline_coverage_fraction",
            ]
        )
    return pd.DataFrame(rows)


def _filters_by_unit(
    pred: pd.DataFrame,
    prefix_end: pd.DataFrame,
    backtest: pd.DataFrame,
    test_ids: list[str],
) -> pd.DataFrame:
    end_by = prefix_end.set_index("unit_id") if not prefix_end.empty and "unit_id" in prefix_end.columns else None
    rows: list[dict[str, Any]] = []
    for uid in test_ids:
        bt = backtest[backtest["unit_id"] == uid] if "unit_id" in backtest.columns else backtest.iloc[0:0]
        stats = summarize_rul_table(bt)
        rec: dict[str, Any] = {
            "unit_id": uid,
            "n_backtest_points": stats["n_points"],
            "backtest_mae": stats["mae"],
            "prefix_end_timestamp_s": None,
            "prefix_end_actual_rul_s": None,
            "prefix_end_predicted_rul_s": None,
            "prefix_end_abs_error": None,
            "prefix_end_baseline_rul_s": None,
            "prefix_end_baseline_abs_error": None,
            "prefix_end_baseline_finite": 0,
        }
        if end_by is not None and uid in end_by.index:
            e = end_by.loc[uid]
            if isinstance(e, pd.DataFrame):
                e = e.iloc[-1]
            rec["prefix_end_timestamp_s"] = float(e["timestamp_s"])
            rec["prefix_end_actual_rul_s"] = (
                float(e["actual_rul_s"]) if pd.notna(e.get("actual_rul_s")) else None
            )
            rec["prefix_end_predicted_rul_s"] = (
                float(e["predicted_rul_s"]) if pd.notna(e.get("predicted_rul_s")) else None
            )
            rec["prefix_end_abs_error"] = _abs_error(e.get("predicted_rul_s"), e.get("actual_rul_s"))
            rec["prefix_end_baseline_rul_s"] = (
                float(e["baseline_rul_s"])
                if "baseline_rul_s" in e.index and pd.notna(e.get("baseline_rul_s"))
                else None
            )
            rec["prefix_end_baseline_abs_error"] = _abs_error(
                e.get("baseline_rul_s") if "baseline_rul_s" in e.index else None,
                e.get("actual_rul_s"),
            )
            rec["prefix_end_baseline_finite"] = int(rec["prefix_end_baseline_rul_s"] is not None)
        rows.append(rec)
    if not rows:
        return pd.DataFrame(
            columns=[
                "unit_id",
                "n_backtest_points",
                "backtest_mae",
                "prefix_end_timestamp_s",
                "prefix_end_actual_rul_s",
                "prefix_end_predicted_rul_s",
                "prefix_end_abs_error",
                "prefix_end_baseline_rul_s",
                "prefix_end_baseline_abs_error",
                "prefix_end_baseline_finite",
            ]
        )
    return pd.DataFrame(rows)


def build_rul_metrics(
    pred: pd.DataFrame,
    *,
    dataset_id: str,
    units: pd.DataFrame,
    cfg: dict[str, Any],
    eval_id: str,
    run_id: str,
    split: dict[str, Any],
    test_ids: list[str],
    bound: dict[str, Any],
    rdir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    metrics = _metrics_identity(
        eval_id=eval_id,
        run_id=run_id,
        dataset_id=dataset_id,
        split=split,
        test_ids=test_ids,
        n_prediction_rows=int(len(pred)),
        bound=bound,
    )
    if dataset_id == "filters":
        prefix_end = filter_prefix_end_table(pred, units)
        backtest = filter_prefix_backtest_table(pred, units)
        end_stats = summarize_rul_table(prefix_end)
        back_stats = summarize_rul_table(backtest)
        end_base = compare_baseline(prefix_end)
        back_base = compare_baseline(backtest)
        metrics.update(
            {
                "primary_metric": "prefix_end_mae",
                "prefix_end_mae": end_stats.get("mae"),
                "equal_weight_unit_mae": end_stats.get("equal_weight_unit_mae"),
                "prefix_end": end_stats,
                "prefix_backtest": back_stats,
                "baseline": {
                    "name": "linear_dp_trend",
                    "prefix_end": end_base,
                    "prefix_backtest": back_base,
                },
                "official_rul_source": (
                    "official_rul_at_prefix_end_s at the last sensor row of each "
                    "author test prefix (eval-only; never a model input)"
                ),
            }
        )
        val_block = _filter_validation_block(rdir)
        if val_block is not None:
            metrics["validation"] = val_block
        by_unit = _filters_by_unit(pred, prefix_end, backtest, test_ids)
        return _json_safe(metrics), by_unit

    scored = summarize_rul_table(pred)
    zones = load_near_event_zones_s(cfg)
    metrics.update(
        {
            "primary_metric": "equal_weight_unit_mae",
            "equal_weight_unit_mae": scored.get("equal_weight_unit_mae"),
            "pooled_mae": scored.get("mae"),
            "pooled_rmse": scored.get("rmse"),
            "mean_overestimation": scored.get("mean_overestimation"),
            "n_scored_points": scored.get("n_points"),
            "n_scored_units": scored.get("n_units"),
            "unit_mae": scored.get("unit_mae") or {},
            "near_event_zones_s": [int(z) if float(z).is_integer() else float(z) for z in zones],
            "equal_weight_unit_mae_by_zone": equal_weight_unit_mae_by_zone(pred, zones),
            "baseline": {"name": "Age-only", **compare_baseline(pred)},
        }
    )
    by_unit = _bearings_by_unit(pred, test_ids)
    return _json_safe(metrics), by_unit


def fingerprint_mismatches(saved: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Field-by-field fingerprint diff. `split_hash` accepts legacy `split_fingerprint`.

    ``gap_rule_version`` is compared to the expected constant, not saved==current.
    Missing on both sides is a mismatch.
    """
    from pdm.splits import resolve_split_hash
    from pdm.windows import GAP_RULE_VERSION

    differing: list[str] = []
    for field in _JSON_COMPARE_FIELDS:
        if field == "split_hash":
            sv, cv = resolve_split_hash(saved), resolve_split_hash(current)
        else:
            sv, cv = saved.get(field), current.get(field)
        if sv != cv:
            differing.append(field)
    _extend_unique(differing, expected_constant_mismatches(saved, current, "gap_rule_version", GAP_RULE_VERSION))
    return differing


def expected_constant_mismatches(
    saved: dict[str, Any],
    current: dict[str, Any],
    field: str,
    expected: Any,
) -> list[str]:
    """True mismatch if saved or current is not ``expected`` (missing ≠ expected)."""
    if saved.get(field) != expected or current.get(field) != expected:
        return [field]
    return []


def assert_gap_rule_current(fp: Mapping[str, Any] | None) -> None:
    """Refuse train/eval when processed fingerprint is not ``GAP_RULE_VERSION``.

    Source is ``processed_fingerprint.json`` only. A ``feature_schema.json``
    value must not satisfy this gate.
    """
    from pdm.windows import GAP_RULE_VERSION

    blob = dict(fp or {})
    nested = blob.get("fingerprint")
    if isinstance(nested, dict):
        blob = nested
    version = blob.get("gap_rule_version")
    if version != GAP_RULE_VERSION:
        raise IncompatibleDataError(
            ["gap_rule_version"],
            detail=(
                f"gap_rule_version={version!r} is not {GAP_RULE_VERSION!r}; "
                "re-run prepare so gap flags are causal"
            ),
        )


require_current_gap_rule = assert_gap_rule_current


def _extend_unique(dst: list[str], extra: list[str]) -> list[str]:
    seen = set(dst)
    for field in extra:
        if field not in seen:
            dst.append(field)
            seen.add(field)
    return dst


def _processed_file_hash_mismatches(run_fp: dict[str, Any], processed_dir: Path) -> list[str]:
    """Rehash live processed files and compare SHA256s to the run fingerprint."""
    from pdm.io_util import sha256_file

    differing: list[str] = []
    for name, field in _PROCESSED_FILE_HASH_FIELDS:
        stored = run_fp.get(field)
        if not stored:
            continue
        path = processed_dir / name
        if not path.exists() or sha256_file(path) != stored:
            differing.append(field)
    return differing


def _live_split_hash_mismatch(run_fp: dict[str, Any], live_split: dict[str, Any]) -> list[str]:
    from pdm.splits import resolve_split_hash, split_hash

    saved = resolve_split_hash(run_fp)
    if not saved or not live_split:
        return []
    if split_hash(live_split) != saved:
        return ["split_hash"]
    return []


def _checkpoint_hash_mismatch(run_fp: dict[str, Any], rdir: Path) -> list[str]:
    """If the run fingerprint has checkpoint_hash, always verify best.pt bytes."""
    from pdm.io_util import checkpoint_hash

    stored = run_fp.get("checkpoint_hash")
    if not stored:
        return []
    ckpt = rdir / "best.pt"
    if not ckpt.exists() or checkpoint_hash(ckpt) != stored:
        return ["checkpoint_hash"]
    return []


def bind_evaluation_to_run(
    rdir: Path,
    dataset_id: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Abort on fingerprint mismatch; bind test IDs / protocol to `rdir/split.json`.

    Never returns the live split from `load_processed()` when a run snapshot exists.
    CLI `--force` skips the abort; UI must not pass force.
    Fingerprint / file-hash / split / gap_rule_version mismatches are data incompatible.
    METRICS_VERSION / coverage-definition drift is evaluation-method-changed, not IncompatibleDataError.
    """
    from pdm.data.prepare import load_processed, load_processed_fingerprint, resolve_processed_dir
    from pdm.experiments import load_run_snapshot
    from pdm.io_util import read_json, sha256_file
    from pdm.splits import resolve_split_hash, split_hash

    snap = load_run_snapshot(rdir)
    run_split = snap["split"]
    run_fp = dict(snap["fingerprint"] or {})
    if resolve_split_hash(run_fp) is None:
        run_fp["split_hash"] = split_hash(run_split)

    processed_dir = resolve_processed_dir(dataset_id)
    live_split_path = processed_dir / "split.json"
    live_split = read_json(live_split_path) if live_split_path.exists() else {}
    current_fp = dict(load_processed_fingerprint(processed_dir, live_split, dataset_id))
    if resolve_split_hash(current_fp) is None and live_split:
        current_fp["split_hash"] = split_hash(live_split)

    assert_gap_rule_current(current_fp)

    differing = fingerprint_mismatches(run_fp, current_fp)
    _extend_unique(differing, _processed_file_hash_mismatches(run_fp, processed_dir))
    _extend_unique(differing, _live_split_hash_mismatch(run_fp, live_split))
    _extend_unique(differing, _checkpoint_hash_mismatch(run_fp, rdir))

    exp_snap = load_experiment_snapshot(rdir)
    snapshot_missing = exp_snap is None
    snapshot_metrics_version = None if exp_snap is None else exp_snap.get("metrics_version")
    evaluation_method_changed = (not snapshot_missing) and snapshot_metrics_version != METRICS_VERSION
    stored_prep_hash = None if exp_snap is None else exp_snap.get("preprocessing_hash")
    if stored_prep_hash:
        prep_path = Path(rdir) / "preprocessing.json"
        if not prep_path.exists() or sha256_file(prep_path) != stored_prep_hash:
            _extend_unique(differing, ["preprocessing_hash"])

    if differing and not force:
        raise IncompatibleDataError(differing)

    processed = load_processed(dataset_id)
    return {
        "split": run_split,
        "features": processed["features"],
        "units": processed["units"],
        "report": processed.get("report") or {},
        "processed_dir": processed["dir"],
        "run_fingerprint": run_fp,
        "current_fingerprint": current_fp,
        "differing": differing,
        "forced": bool(differing and force),
        "protocol": run_split.get("protocol"),
        "test_ids": list(run_split.get("test") or []),
        "validation_ids": list(run_split.get("validation") or []),
        "snapshot_missing": snapshot_missing,
        "evaluation_method_changed": bool(evaluation_method_changed),
        "snapshot_metrics_version": snapshot_metrics_version,
    }


_FILTER_TIME_SCALE_REPORT_KEYS = (
    "time_unit_note",
    "time_to_seconds",
    "time_scale_verified",
    "original_time_unit",
)

METRICS_VERSION = "v1"
_ALERT_DEPENDENT_PRED_COLUMNS = ("alert_status",)
ALERT_UNIT_COLUMNS = (
    "n_alert_episodes",
    "timely_episodes",
    "too_early_episodes",
    "late_episodes",
    "alert_outcome",
    "lead_time_s",
    "has_sufficient_coverage",
    "evaluator_event_time_s",
    "alert_event_source",
    "time_in_warning_s",
    "observed_duration_s",
    "fraction_time_in_warning",
)
STUB_ALERT_COLUMNS = ALERT_EPISODE_COLUMNS + ("lead_time_s", "outcome")
_EVAL_CONFIG_FP_FIELDS = (
    "dataset_version",
    "split_hash",
    "features_hash",
    "units_hash",
    "feature_pipeline_version",
)


def new_eval_id() -> str:
    """UTC timestamp + short UUID; lexicographic order matches creation time."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def allocate_eval_staging(rdir: Path) -> tuple[str, Path, Path]:
    """Hidden `.tmp_<eval_id>/` for writes. Final `evaluations/<eval_id>/` is not created yet."""
    root = rdir / EVALUATIONS_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    last_err: OSError | None = None
    for _ in range(16):
        eval_id = new_eval_id()
        dest = root / eval_id
        staging = root / f".tmp_{eval_id}"
        if dest.exists() or staging.exists():
            continue
        try:
            staging.mkdir(parents=False, exist_ok=False)
            return eval_id, staging, dest
        except FileExistsError as exc:
            last_err = exc
    raise RuntimeError(f"Could not allocate a unique eval staging dir under {root}") from last_err


def publish_eval_dir(staging: Path, dest: Path) -> None:
    """Rename staging → final. Never overwrite an existing eval dir."""
    if dest.exists():
        raise RuntimeError(f"eval dir already exists, not overwriting: {dest}")
    staging.rename(dest)


def prediction_export_frame(pred: pd.DataFrame) -> pd.DataFrame:
    """Drop H/K / alert-policy columns so predictions.csv is independent of alert settings.

    Keeps current sensor/model state (`observed_limit_reached`, `prediction_status`,
    `valid_history_reason`, `differential_pressure`). Those are not future GT.
    """
    drop = [c for c in _ALERT_DEPENDENT_PRED_COLUMNS if c in pred.columns]
    return pred.drop(columns=drop) if drop else pred.copy()


def _evaluation_config(
    *,
    eval_id: str,
    run_id: str,
    dataset_id: str,
    run_fp: dict[str, Any],
    split: dict[str, Any],
    checkpoint_sha: str,
    test_ids: list[str] | None = None,
    split_name: Literal["validation", "test"] = "test",
    unit_ids: list[str] | None = None,
    source: str | None = None,
    blind_benchmark: bool = False,
) -> dict[str, Any]:
    from pdm.splits import resolve_split_hash, split_hash

    ids = [str(u) for u in (unit_ids if unit_ids is not None else (test_ids or []))]
    cfg: dict[str, Any] = {
        "eval_id": eval_id,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "checkpoint_hash": checkpoint_sha,
        "metrics_version": METRICS_VERSION,
        "evaluate_mask": {
            "split": split_name,
            "protocol": split.get("protocol"),
            "unit_ids": ids,
            "blind_benchmark": bool(blind_benchmark),
        },
    }
    if source is not None:
        cfg["source"] = str(source)
    for field in _EVAL_CONFIG_FP_FIELDS:
        if field == "split_hash":
            cfg["split_hash"] = resolve_split_hash(run_fp) or split_hash(split)
        else:
            cfg[field] = run_fp.get(field)
    return cfg


def load_evaluation_alert_policy(
    eval_dir: Path | str | None = None,
    *,
    config_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Nested `alert_policy` from `evaluation_config.json`. None if missing."""
    path: Path | None = None
    if config_path is not None:
        path = Path(config_path)
    elif eval_dir is not None:
        path = Path(eval_dir) / "evaluation_config.json"
    if path is None or not path.exists():
        return None
    from pdm.io_util import read_json

    cfg = read_json(path)
    block = cfg.get("alert_policy")
    if not isinstance(block, dict) or not block:
        return None
    return dict(block)


def filter_time_scale_from_bound(bound: dict[str, Any]) -> dict[str, Any]:
    """Stamp Time/RUL scale from bound processed data_report.json, not live YAML."""
    report = dict(bound.get("report") or {})
    if not any(k in report for k in _FILTER_TIME_SCALE_REPORT_KEYS):
        processed_dir = bound.get("processed_dir")
        if processed_dir is not None:
            path = Path(processed_dir) / "data_report.json"
            if path.exists():
                from pdm.io_util import read_json

                report = read_json(path)
    return {key: report[key] for key in _FILTER_TIME_SCALE_REPORT_KEYS if key in report}


POLICY_MODES = ("research", "frozen")
EVAL_SPLIT_NAMES = ("validation", "test")


def hk_overrides_present(
    *,
    warning_horizon_s: float | None = None,
    H_trigger: float | None = None,
    confirmation_count: int | None = None,
    minimum_action_lead_time: float | None = None,
    max_useful_horizon_s: float | None = None,
) -> bool:
    return any(
        v is not None
        for v in (
            warning_horizon_s,
            H_trigger,
            confirmation_count,
            minimum_action_lead_time,
            max_useful_horizon_s,
        )
    )


def policy_mode_from_hk_overrides(**hk: Any) -> str:
    """CLI: any H/K flag → research (never write). None of them → frozen."""
    return "research" if hk_overrides_present(**hk) else "frozen"


def normalize_policy_mode(policy_mode: str | None, *, hk_present: bool) -> str:
    if policy_mode is None or str(policy_mode).strip() == "":
        return "research" if hk_present else "frozen"
    mode = str(policy_mode).strip()
    if mode not in POLICY_MODES:
        raise ValueError(f"policy_mode must be 'research' or 'frozen', got {mode!r}")
    return mode


def research_eval_source(split_name: str) -> str:
    return "validation_eval" if split_name == "validation" else "research"


def generate_split_predictions(
    *,
    features: pd.DataFrame,
    predictor: Any,
    dataset_id: str,
    run_id: str,
    split_name: Literal["validation", "test"],
    unit_ids: list[str],
    history_length: int,
    policy: Mapping[str, Any],
    train_units: pd.DataFrame,
    pressure_limit_pa: float,
    truth_units: pd.DataFrame,
    forecast_profile: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Replay every listed unit. Predictions.csv stays H/K-independent after export."""
    from pdm.replay import replay_unit

    if split_name not in EVAL_SPLIT_NAMES:
        raise ValueError(f"split_name must be 'validation' or 'test', got {split_name!r}")
    h = float(policy["H_trigger"])
    k = int(policy["confirmation_count"])
    reset = float(policy.get("reset_factor", 1.2))
    pred_frames: list[pd.DataFrame] = []
    for uid in unit_ids:
        meas = features[features["unit_id"] == uid]
        out = replay_unit(
            meas,
            predictor,
            dataset_id=dataset_id,
            unit_id=str(uid),
            run_id=run_id,
            history_length=int(history_length),
            warning_horizon_s=float(h),
            confirmation_count=k,
            reset_factor=reset,
            train_units=train_units,
            pressure_limit_pa=pressure_limit_pa,
            truth_units=truth_units,
            forecast_profile=forecast_profile,
        )
        pred_frames.append(out["predictions"])
    if not pred_frames:
        return pd.DataFrame()
    return pd.concat(pred_frames, ignore_index=True)


def _write_reservoir_traces(
    *,
    dataset_id: str,
    run_id: str,
    unit_ids: list[str],
    features: pd.DataFrame,
    model: Any,
    prep: Any,
    history_length: int,
) -> None:
    """Persist per-unit traces next to the run. Does not touch predictions.csv."""
    from pdm.paths import run_traces_dir
    from pdm.visualization.export import save_trace
    from pdm.visualization.trace import is_reservoir_module, predict_with_trace

    if not is_reservoir_module(model):
        return
    hist_len = int(history_length)
    for uid in unit_ids:
        meas = features[features["unit_id"].astype(str) == str(uid)].copy()
        if meas.empty:
            continue
        meas.attrs["raw_features"] = True
        trace = predict_with_trace(
            meas,
            str(uid),
            model,
            prep,
            history_length=hist_len,
        )
        if trace["status"] != "predicted":
            continue
        save_trace(
            run_traces_dir(dataset_id, run_id, str(uid)),
            unit_id=str(uid),
            run_id=str(run_id),
            dataset_id=str(dataset_id),
            architecture=str(getattr(model, "architecture", "")),
            graph_hash=getattr(model, "graph_hash", None),
            n_nodes=int(getattr(model, "n_nodes", 0) or 0),
            history_length=hist_len,
            graph_mode=str(getattr(model, "graph_mode", "")),
            is_synthetic=bool(getattr(model, "is_synthetic", True)),
            states=trace["states"],
            inputs=trace["inputs"],
            contributions=trace["contributions"],
            frame_map=trace["frame_map"],
            node_order=trace["node_order"],
            predicted_rul_s=trace["predicted_rul_s"],
            raw_prediction=trace["raw_prediction"],
            status=trace["status"],
            time_scale_s=float(getattr(model, "time_scale_s", 1.0)),
            head=str(getattr(model, "head_type", "")),
        )


def evaluate_run(
    dataset_id: str,
    run_id: str,
    *,
    split_name: Literal["validation", "test"] = "test",
    policy_mode: str | None = None,
    warning_horizon_s: float | None = None,
    H_trigger: float | None = None,
    confirmation_count: int | None = None,
    minimum_action_lead_time: float | None = None,
    max_useful_horizon_s: float | None = None,
    device: str = "cpu",
    force: bool = False,
    with_trace: bool = False,
) -> dict[str, Any]:
    from pdm.config import load_dataset_config
    from pdm.device import resolve_device
    from pdm.io_util import atomic_write_json, atomic_write_text, checkpoint_hash, read_json
    from pdm.paths import dataset_runs
    from pdm.predict import Predictor
    from pdm.train import load_trained_model

    if split_name not in EVAL_SPLIT_NAMES:
        raise ValueError(f"split_name must be 'validation' or 'test', got {split_name!r}")

    rdir = dataset_runs(dataset_id) / run_id
    bound = bind_evaluation_to_run(rdir, dataset_id, force=force)
    features = bound["features"]
    units = bound["units"]
    split = bound["split"]
    run_fp = dict(bound["run_fingerprint"] or {})
    if split_name == "validation":
        unit_ids = [str(u) for u in (bound.get("validation_ids") or split.get("validation") or [])]
    else:
        unit_ids = [str(u) for u in (bound.get("test_ids") or split.get("test") or [])]

    hk_present = hk_overrides_present(
        warning_horizon_s=warning_horizon_s,
        H_trigger=H_trigger,
        confirmation_count=confirmation_count,
        minimum_action_lead_time=minimum_action_lead_time,
        max_useful_horizon_s=max_useful_horizon_s,
    )
    mode = normalize_policy_mode(policy_mode, hk_present=hk_present)

    live_cfg = load_dataset_config(dataset_id)
    cfg, snapshot_missing = resolve_run_task_config(rdir, live_cfg)
    snapshot_missing = bool(bound.get("snapshot_missing", snapshot_missing))
    alerts_cfg = dict(cfg.get("alerts") or {})
    train_units = units[units["unit_id"].isin(split["train"])]
    if mode == "frozen":
        # Ignore CLI/widget H/K. Missing file must fail — never first-write YAML defaults.
        policy = require_frozen_alert_policy(rdir)
        eval_source = str(policy.get("source") or "validation_ui")
        blind_benchmark = split_name == "test"
    else:
        default_h = default_horizon_s(
            train_units, float(alerts_cfg.get("horizon_fraction_of_median_train", 0.10))
        )
        eval_source = research_eval_source(split_name)
        policy = resolve_alert_policy(
            rdir,
            alerts_cfg=alerts_cfg,
            H_trigger=H_trigger,
            warning_horizon_s=warning_horizon_s,
            confirmation_count=confirmation_count,
            minimum_action_lead_time=minimum_action_lead_time,
            max_useful_horizon_s=max_useful_horizon_s,
            default_h_trigger=default_h,
            source=eval_source,
        )
        policy["source"] = eval_source
        blind_benchmark = False

    dev = resolve_device(device)
    model, prep, meta = load_trained_model(
        rdir,
        device=dev.torch_device,
        which="best",
        verify_checkpoint_hash=not force,
    )
    predictor = Predictor(model, prep, history_length=int(meta["history_length"]), device=dev.torch_device)
    forecast_profile = None
    if getattr(model, "state_mode", None) == "continuous" and (rdir / "interval_profile.json").is_file():
        from pdm.forecasting import load_interval_profile

        forecast_profile = load_interval_profile(rdir)
        # Reusing a previously inspected holdout cannot become a blind benchmark
        # merely because an alert policy was frozen later.
        if forecast_profile.get("evaluation_status") == "exploratory_reused_holdout":
            blind_benchmark = False
    h = float(policy["H_trigger"])
    k = int(policy["confirmation_count"])
    live_limit = live_cfg.get("pressure_limit_pa")
    if live_limit is None:
        live_limit = cfg.get("pressure_limit_pa")
    pressure_limit_pa = load_run_pressure_limit_pa(rdir, live_limit)
    preds = generate_split_predictions(
        features=features,
        predictor=predictor,
        dataset_id=dataset_id,
        run_id=run_id,
        split_name=split_name,
        unit_ids=unit_ids,
        history_length=int(meta["history_length"]),
        policy=policy,
        train_units=train_units,
        pressure_limit_pa=pressure_limit_pa,
        truth_units=units,
        forecast_profile=forecast_profile,
    )
    if not preds.empty:
        preds = attach_actual_rul(preds, units, dataset_id)
    pred_export = prediction_export_frame(preds)
    if with_trace:
        _write_reservoir_traces(
            dataset_id=dataset_id,
            run_id=run_id,
            unit_ids=unit_ids,
            features=features,
            model=model,
            prep=prep,
            history_length=int(meta["history_length"]),
        )

    expected_ckpt = run_fp.get("checkpoint_hash")
    ckpt_sha = checkpoint_hash(rdir / "best.pt")
    eval_id, staging, dest = allocate_eval_staging(rdir)
    try:
        eval_cfg = copy_alert_policy_into_evaluation_config(
            _evaluation_config(
                eval_id=eval_id,
                run_id=run_id,
                dataset_id=dataset_id,
                run_fp=run_fp,
                split=split,
                split_name=split_name,
                unit_ids=unit_ids,
                checkpoint_sha=str(ckpt_sha),
                source=eval_source,
                blind_benchmark=blind_benchmark,
            ),
            policy,
            metrics_version=METRICS_VERSION,
        )
        eval_cfg["snapshot_missing"] = snapshot_missing
        eval_cfg["evaluation_method_changed"] = bool(bound.get("evaluation_method_changed"))
        if bound.get("snapshot_metrics_version") is not None:
            eval_cfg["snapshot_metrics_version"] = bound.get("snapshot_metrics_version")
        eval_cfg["pressure_limit_pa"] = pressure_limit_pa
        if getattr(model, "state_mode", None) == "continuous":
            eval_cfg["state_mode"] = "continuous"
            eval_cfg["forecast_method"] = (
                "continuous_empirical_interval" if forecast_profile is not None else "continuous_raw_readout"
            )
        if forecast_profile is not None:
            from pdm.io_util import sha256_file

            eval_cfg["interval_profile_sha256"] = sha256_file(rdir / "interval_profile.json")
            eval_cfg["evaluation_status"] = forecast_profile.get("evaluation_status", "empirical_calibration")
            eval_cfg["interval_coverage_guarantee"] = False
        if expected_ckpt and str(expected_ckpt) != str(ckpt_sha):
            eval_cfg["expected_checkpoint_hash"] = str(expected_ckpt)
        atomic_write_json(staging / "evaluation_config.json", eval_cfg)
        atomic_write_text(staging / "predictions.csv", pred_export.to_csv(index=False))
        metrics, by_unit = build_rul_metrics(
            preds,
            dataset_id=dataset_id,
            units=units,
            cfg=cfg,
            eval_id=eval_id,
            run_id=run_id,
            split=split,
            test_ids=unit_ids,
            bound=bound,
            rdir=rdir,
        )
        for key in ("state_mode", "forecast_method", "interval_profile_sha256", "evaluation_status", "interval_coverage_guarantee"):
            if key in eval_cfg:
                metrics[key] = eval_cfg[key]
        atomic_write_json(staging / "metrics.json", metrics)
        atomic_write_text(staging / "metrics_by_unit.csv", by_unit.to_csv(index=False))
        run_alert_evaluation(
            pred_export,
            policy,
            units=units,
            eval_dir=staging,
            run_id=run_id,
            pressure_limit_pa=pressure_limit_pa,
        )
        metrics = read_json(staging / "metrics.json")
        publish_eval_dir(staging, dest)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    metrics["eval_dir"] = str(dest)
    metrics["warning_horizon_s"] = float(h)
    metrics["H_trigger"] = float(h)
    metrics["confirmation_count"] = k
    metrics["minimum_action_lead_time"] = float(policy["minimum_action_lead_time"])
    return metrics


def time_weighted_warning_share(
    timestamps_s: np.ndarray | list[float],
    warning_active: np.ndarray | list[bool],
) -> dict[str, float]:
    """Left-closed ΣΔt: status at t_i holds until t_{i+1}. Last sample adds 0."""
    ts = np.asarray(timestamps_s, dtype=np.float64)
    w = np.asarray(warning_active, dtype=bool)
    empty = {
        "fraction_time_in_warning": 0.0,
        "time_in_warning_s": 0.0,
        "observed_duration_s": 0.0,
    }
    if ts.size < 2 or w.size != ts.size:
        return empty
    dt = np.diff(ts)
    dt = np.where(np.isfinite(dt), np.maximum(dt, 0.0), 0.0)
    total = float(dt.sum())
    warn = float(dt[w[:-1]].sum())
    return {
        "fraction_time_in_warning": (warn / total) if total > 0.0 else 0.0,
        "time_in_warning_s": warn,
        "observed_duration_s": total,
    }


def _finite_or_none(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _evaluator_event_and_obs(
    rec: Mapping[str, Any],
    last_timestamp_s: float | None,
) -> tuple[float | None, float | None, str | None]:
    """Alert-eval event/obs only. Never a Predictor input or predictions.csv field.

    Filter test prefixes often have NaN `event_time_s` but a finite official RUL at
    prefix end. Overlay event = observation_end (or last timestamp) + official RUL.

    Replay UI must not treat ``official_rul_overlay`` as a sensor 600 Pa crossing.
    Censored / truncated files use ``End of observed data``; official RUL is an
    evaluation annotation only (see ``pdm.replay.replay_end_state``).
    """
    obs = _finite_or_none(rec.get("observation_end_s"))
    if obs is None:
        obs = last_timestamp_s
    event = _finite_or_none(rec.get("event_time_s"))
    if event is not None:
        return event, obs, "event_time_s"
    official = _finite_or_none(rec.get("official_rul_at_prefix_end_s"))
    if official is not None and obs is not None:
        return float(obs) + float(official), obs, "official_rul_overlay"
    return None, obs, None


def _unit_event_obs(
    units: pd.DataFrame,
    last_ts: Mapping[str, float] | None = None,
) -> dict[str, tuple[float | None, float | None, str | None]]:
    out: dict[str, tuple[float | None, float | None, str | None]] = {}
    if units is None or getattr(units, "empty", True) or "unit_id" not in units.columns:
        return out
    last = last_ts or {}
    for rec in units.to_dict(orient="records"):
        uid = str(rec["unit_id"])
        out[uid] = _evaluator_event_and_obs(rec, last.get(uid))
    return out


def _merge_alert_unit_metrics(by_unit: pd.DataFrame, alert_by_unit: pd.DataFrame) -> pd.DataFrame:
    drop = [c for c in ALERT_UNIT_COLUMNS if c in by_unit.columns]
    base = by_unit.drop(columns=drop) if drop else by_unit.copy()
    extra = alert_by_unit.copy() if alert_by_unit is not None else pd.DataFrame()
    if extra.empty:
        for col in ALERT_UNIT_COLUMNS:
            if col not in base.columns:
                base[col] = None
        return base
    extra["unit_id"] = extra["unit_id"].astype(str)
    if "unit_id" not in base.columns or base.empty:
        return extra
    base["unit_id"] = base["unit_id"].astype(str)
    return base.merge(extra, on="unit_id", how="left")


def summarize_alert_metrics(
    *,
    episodes: pd.DataFrame,
    steps: pd.DataFrame,
    units: pd.DataFrame,
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Score every confirmed episode. miss / insufficient_coverage are per unit with no alert.

    v1 coverage uses unit ``observation_end_s``. When ``minimum_action_lead_time == 0``,
    step timestamps decide whether any sample is strictly before the event.
    """
    last_ts: dict[str, float] = {}
    if not steps.empty and "unit_id" in steps.columns:
        for uid, g in steps.groupby("unit_id", sort=False):
            last_ts[str(uid)] = float(g["timestamp_s"].max())
    lookup = _unit_event_obs(units, last_ts)

    ep = episodes.copy()
    if ep.empty:
        ep = pd.DataFrame(columns=list(STUB_ALERT_COLUMNS))
    else:
        ep["unit_id"] = ep["unit_id"].astype(str)
        leads: list[float | None] = []
        outcomes: list[str | None] = []
        for rec in ep.to_dict(orient="records"):
            uid = str(rec["unit_id"])
            event, obs, _src = lookup.get(uid, (None, None, None))
            if obs is None:
                obs = last_ts.get(uid)
            if obs is None:
                obs = 0.0
            confirmed = float(rec["timestamp_s"])
            outcome = classify_alert_outcome(
                event_time_s=event,
                observation_end_s=float(obs),
                policy=policy,
                confirmed_alert_time_s=confirmed,
            )
            lead = (
                confirmed_lead_time_s(event_time_s=event, confirmed_alert_time_s=confirmed)
                if event is not None
                else None
            )
            leads.append(lead)
            outcomes.append(outcome)
        ep["lead_time_s"] = leads
        ep["outcome"] = outcomes

    scored_ids: list[str] = []
    if not steps.empty and "unit_id" in steps.columns:
        scored_ids = list(dict.fromkeys(steps["unit_id"].astype(str).tolist()))

    n_timely = n_too_early = n_late = 0
    if not ep.empty and "outcome" in ep.columns:
        n_timely = int((ep["outcome"] == "timely").sum())
        n_too_early = int((ep["outcome"] == "too_early").sum())
        n_late = int((ep["outcome"] == "late").sum())

    n_miss = n_insuf = n_event = n_cov = n_units_timely = 0
    warn_s = total_s = 0.0
    by_rows: list[dict[str, Any]] = []
    for uid in scored_ids:
        event, obs, event_source = lookup.get(uid, (None, None, None))
        unit_steps = steps[steps["unit_id"].astype(str) == uid]
        unit_ts = None
        if not unit_steps.empty and "timestamp_s" in unit_steps.columns:
            unit_ts = unit_steps["timestamp_s"].to_numpy(dtype=np.float64)
        tw = time_weighted_warning_share(
            unit_steps["timestamp_s"].to_numpy(dtype=np.float64),
            unit_steps["warning_active"].to_numpy(dtype=bool),
        )
        warn_s += tw["time_in_warning_s"]
        total_s += tw["observed_duration_s"]
        if obs is None:
            obs = last_ts.get(uid)
        unit_ep = ep[ep["unit_id"].astype(str) == uid] if not ep.empty and "unit_id" in ep.columns else ep
        n_ep = int(len(unit_ep))
        timely_n = too_early_n = late_n = 0
        first_lead = None
        coverage = None
        if event is not None:
            n_event += 1
            if obs is not None:
                coverage = has_sufficient_coverage(
                    float(obs), event, policy, timestamps_s=unit_ts
                )
                if coverage:
                    n_cov += 1
        if n_ep:
            outs = [o for o in unit_ep["outcome"].tolist() if o]
            timely_n = outs.count("timely")
            too_early_n = outs.count("too_early")
            late_n = outs.count("late")
            unique = list(dict.fromkeys(outs))
            unit_outcome = unique[0] if len(unique) == 1 else ("mixed" if unique else None)
            raw_lead = unit_ep["lead_time_s"].iloc[0]
            first_lead = _finite_or_none(raw_lead)
            if timely_n:
                n_units_timely += 1
        else:
            unit_outcome = classify_alert_outcome(
                event_time_s=event,
                observation_end_s=float(obs) if obs is not None else 0.0,
                policy=policy,
                confirmed_alert_time_s=None,
                timestamps_s=unit_ts,
            )
            if unit_outcome == "miss":
                n_miss += 1
            elif unit_outcome == "insufficient_coverage":
                n_insuf += 1
        by_rows.append(
            {
                "unit_id": uid,
                "n_alert_episodes": n_ep,
                "timely_episodes": timely_n,
                "too_early_episodes": too_early_n,
                "late_episodes": late_n,
                "alert_outcome": unit_outcome,
                "lead_time_s": first_lead,
                "has_sufficient_coverage": coverage,
                "evaluator_event_time_s": event,
                "alert_event_source": event_source,
                "time_in_warning_s": tw["time_in_warning_s"],
                "observed_duration_s": tw["observed_duration_s"],
                "fraction_time_in_warning": tw["fraction_time_in_warning"],
            }
        )

    lead_vals = []
    if not ep.empty and "lead_time_s" in ep.columns:
        lead_vals = [v for v in ep["lead_time_s"].tolist() if _finite_or_none(v) is not None]
    mean_lead = float(np.mean(lead_vals)) if lead_vals else None
    # Incomplete windows are a coverage class, not scored misses.
    n_scored = n_event - n_insuf
    miss_rate = (n_miss / n_scored) if n_scored else None
    timely_rate = (n_units_timely / n_scored) if n_scored else None
    metrics = {
        "timely": n_timely,
        "too_early": n_too_early,
        "late": n_late,
        "miss": n_miss,
        "insufficient_coverage": n_insuf,
        "n_episodes": int(len(ep)),
        "n_units_with_event": n_event,
        "n_units_sufficient_coverage": n_cov,
        "n_units_insufficient_coverage": n_insuf,
        "n_units_timely": n_units_timely,
        "n_units_scored": n_scored,
        "denominator_units_scored": n_scored,
        "miss_rate": miss_rate,
        "timely_rate": timely_rate,
        "denominator_note": (
            "miss/timely rates use units with a finite evaluator event excluding "
            "insufficient_coverage; incomplete windows are not false misses."
        ),
        "fraction_time_in_warning": (warn_s / total_s) if total_s > 0.0 else 0.0,
        "time_in_warning_s": warn_s,
        "observed_duration_s": total_s,
        "mean_lead_time_s": mean_lead,
    }
    by_unit = pd.DataFrame(by_rows)
    if by_unit.empty:
        by_unit = pd.DataFrame(columns=["unit_id", *ALERT_UNIT_COLUMNS])
    return metrics, by_unit, ep


def run_alert_evaluation(
    predictions: pd.DataFrame | Path | str,
    policy: Mapping[str, Any],
    *,
    units: pd.DataFrame,
    eval_dir: Path | None = None,
    run_id: str | None = None,
    pressure_limit_pa: float | None = None,
) -> dict[str, Any]:
    """Score alerts from frozen predictions.csv. Never regenerates or rewrites predictions."""
    from pdm.io_util import atomic_write_json, atomic_write_text, read_json

    pred_path: Path | None = None
    if isinstance(predictions, (str, Path)):
        pred_path = Path(predictions)
        pred_bytes = pred_path.read_bytes() if pred_path.exists() else None
        pred = pd.read_csv(pred_path)
    else:
        pred_bytes = None
        pred = predictions if predictions is not None else pd.DataFrame()

    episodes, steps = alerts_from_predictions(
        pred, policy, run_id=run_id, pressure_limit_pa=pressure_limit_pa
    )
    alert_block, by_unit, episodes = summarize_alert_metrics(
        episodes=episodes,
        steps=steps,
        units=units,
        policy=policy,
    )
    result: dict[str, Any] = {
        "alerts": episodes,
        "steps": steps,
        "metrics": {"alerts": alert_block},
        "by_unit": by_unit,
    }
    if eval_dir is None:
        return result

    dest = Path(eval_dir)
    atomic_write_text(dest / "alerts.csv", episodes.to_csv(index=False))
    cfg_path = dest / "evaluation_config.json"
    if cfg_path.exists():
        cfg = copy_alert_policy_into_evaluation_config(
            read_json(cfg_path), policy, metrics_version=METRICS_VERSION
        )
        atomic_write_json(cfg_path, cfg)
    metrics_path = dest / "metrics.json"
    if metrics_path.exists():
        existing = read_json(metrics_path)
        existing["alerts"] = alert_block
        existing["metrics_version"] = METRICS_VERSION
        atomic_write_json(metrics_path, _json_safe(existing))
    else:
        atomic_write_json(
            metrics_path,
            _json_safe({"alerts": alert_block, "metrics_version": METRICS_VERSION}),
        )
    by_path = dest / "metrics_by_unit.csv"
    if by_path.exists():
        merged = _merge_alert_unit_metrics(pd.read_csv(by_path), by_unit)
        atomic_write_text(by_path, merged.to_csv(index=False))
    else:
        atomic_write_text(by_path, by_unit.to_csv(index=False))
    if pred_path is not None and pred_bytes is not None:
        if pred_path.read_bytes() != pred_bytes:
            raise RuntimeError(f"run_alert_evaluation must not rewrite {pred_path}")
    return result
