from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from pdm.connectome.provenance import GRAPH_MODE_SYNTHETIC
from pdm.io_util import read_json
from pdm.paths import dataset_runs, runs_root
from pdm.splits import resolve_split_hash, split_hash

_RUN_IDENTITY_KEYS = (
    "architecture",
    "graph_mode",
    "is_synthetic",
    "parent_graph_hash",
    "graph_hash",
    "n_nodes",
    "split_hash",
    "leak",
    "spectral_radius",
    "state_mode",
    "history_length",
    "head",
    "smoke",
)

EVALUATIONS_DIRNAME = "evaluations"
LEGACY_PREDICTIONS_NAME = "predictions.csv"
LEGACY_METRICS_NAME = "test_metrics.json"
LEGACY_ALERTS_NAME = "alerts.csv"


def list_runs(dataset_id: str | None = None) -> list[dict[str, Any]]:
    rows = []
    roots = [dataset_runs(dataset_id)] if dataset_id else [p for p in runs_root().iterdir() if p.is_dir() and not p.name.startswith("_")]
    for root in roots:
        if not root.exists():
            continue
        ds = dataset_id or root.name
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir():
                continue
            status_path = run_dir / "status.json"
            row: dict[str, Any] = {
                "dataset_id": ds,
                "run_id": run_dir.name,
                "path": str(run_dir),
                "has_best": (run_dir / "best.pt").exists(),
                "has_last": (run_dir / "last.pt").exists(),
            }
            if status_path.exists():
                try:
                    row.update(read_json(status_path))
                except Exception:
                    row["status"] = "unknown"
            else:
                row["status"] = "unknown"
            row.setdefault("run_id", run_dir.name)
            row["n_evaluations"] = _count_evaluations(run_dir)
            row["has_legacy_predictions"] = (run_dir / LEGACY_PREDICTIONS_NAME).exists()
            _enrich_run_identity(row, run_dir)
            rows.append(row)
    def recency(row):
        try:
            return datetime.fromisoformat(row["updated_at"]).timestamp()
        except (KeyError, ValueError, TypeError):
            root = Path(row["path"])
            stamp = root / "best.pt" if (root / "best.pt").exists() else root
            return stamp.stat().st_mtime

    rows.sort(key=recency, reverse=True)
    return rows


def _optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = read_json(path)
    except Exception:
        return None
    return loaded if isinstance(loaded, dict) else None


def _copy_if_missing(row: dict[str, Any], key: str, value: Any) -> None:
    if value is None:
        return
    current = row.get(key)
    if current is None or current == "":
        row[key] = value


def _enrich_run_identity(row: dict[str, Any], run_path: Path) -> dict[str, Any]:
    """Fill architecture / graph identity from snapshot + provenance when status omits them."""
    snap = _optional_json(run_path / "experiment_snapshot.json") or {}
    preprocessing = _optional_json(run_path / "preprocessing.json") or {}
    row["feature_recipe"] = preprocessing.get("feature_recipe", "base_v1")
    row["feature_count"] = len(preprocessing.get("feature_names", []))
    row["feature_recipe_hash"] = preprocessing.get("feature_recipe_hash")
    model = snap.get("model") if isinstance(snap.get("model"), dict) else {}
    reservoir = model.get("reservoir") if isinstance(model.get("reservoir"), dict) else {}
    _copy_if_missing(row, "architecture", model.get("architecture"))
    _copy_if_missing(row, "head", model.get("head") or snap.get("head"))
    _copy_if_missing(row, "history_length", model.get("history_length"))
    _copy_if_missing(row, "smoke", snap.get("smoke") if "smoke" in snap else None)
    for key in (
        "graph_mode",
        "n_nodes",
        "graph_hash",
        "parent_graph_hash",
        "leak",
        "spectral_radius",
        "state_mode",
    ):
        _copy_if_missing(row, key, reservoir.get(key))

    prov = _optional_json(run_path / "connectome" / "provenance.json") or {}
    for key in ("graph_mode", "graph_hash", "parent_graph_hash", "n_nodes", "is_synthetic"):
        _copy_if_missing(row, key, prov.get(key))

    # Dataset defaults contain reservoir settings even for recurrent models.
    # They describe no part of a GRU/LSTM and must not classify it as synthetic.
    if row.get("architecture") in {"gru", "lstm"}:
        size = int(model["hidden_size"]) * int(model.get("recurrent_layers", 1)) if model.get("hidden_size") else None
        row.update(n_nodes=size,
                   state_mode="window_reset", is_synthetic=False, graph_mode=None,
                   graph_hash=None, parent_graph_hash=None)

    fp = _optional_json(run_path / "dataset_fingerprint.json") or {}
    for key in ("dataset_version", "quality_policy_hash", "quality_policy_version"):
        _copy_if_missing(row, key, fp.get(key))
    split_id = resolve_split_hash(fp) or resolve_split_hash(row) or resolve_split_hash(snap)
    if split_id is None:
        split_doc = _optional_json(run_path / "split.json")
        if split_doc is not None:
            split_id = resolve_split_hash(split_doc) or split_hash(split_doc)
    _copy_if_missing(row, "split_hash", split_id)

    mode = str(row.get("graph_mode") or "").strip().lower()
    if row.get("is_synthetic") is None and mode == GRAPH_MODE_SYNTHETIC:
        row["is_synthetic"] = True
    elif row.get("is_synthetic") is not None:
        row["is_synthetic"] = bool(row.get("is_synthetic"))
    for key in _RUN_IDENTITY_KEYS:
        row.setdefault(key, row.get(key))
    return row


def run_dir(dataset_id: str, run_id: str) -> Path:
    return dataset_runs(dataset_id) / run_id


def load_run_status(dataset_id: str, run_id: str) -> dict[str, Any]:
    p = run_dir(dataset_id, run_id) / "status.json"
    return read_json(p) if p.exists() else {}


def load_run_snapshot(run_path: Path) -> dict[str, Any]:
    """Load `split.json` + `dataset_fingerprint.json` from a run directory."""
    split_p = run_path / "split.json"
    if not split_p.exists():
        raise FileNotFoundError(f"Run snapshot missing split.json: {split_p}")
    fp_p = run_path / "dataset_fingerprint.json"
    return {
        "dir": run_path,
        "split": read_json(split_p),
        "fingerprint": read_json(fp_p) if fp_p.exists() else {},
        "has_fingerprint": fp_p.exists(),
        "best_pt": run_path / "best.pt",
        "last_pt": run_path / "last.pt",
    }


def evaluations_dir(run_path: Path) -> Path:
    return Path(run_path) / EVALUATIONS_DIRNAME


def _count_evaluations(run_path: Path) -> int:
    root = evaluations_dir(run_path)
    if not root.exists():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def evaluation_mask_view(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize `evaluate_mask`. Missing mask → split=test, not a blind benchmark."""
    raw = cfg.get("evaluate_mask") if isinstance(cfg, Mapping) else None
    if not isinstance(raw, Mapping):
        return {"split": "test", "blind_benchmark": False, "unit_ids": None}
    split = str(raw.get("split") or "test").strip() or "test"
    if split not in {"validation", "test"}:
        split = "test"
    ids_raw = raw.get("unit_ids")
    unit_ids = None if ids_raw is None else [str(u) for u in ids_raw]
    out: dict[str, Any] = {
        "split": split,
        "blind_benchmark": bool(raw.get("blind_benchmark")),
        "unit_ids": unit_ids,
    }
    if raw.get("protocol") is not None:
        out["protocol"] = raw.get("protocol")
    return out


def evaluations_for_mode(evals: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    """Filter eval rows for Validation / Test / Research. Test requires a blind benchmark."""
    label = str(mode).strip()
    want_split = "validation" if label == "Validation" else "test"
    require_blind = label == "Test"
    matching: list[dict[str, Any]] = []
    for row in evals:
        mask = row.get("evaluate_mask") or evaluation_mask_view(None)
        if str(mask.get("split") or "test") != want_split:
            continue
        if require_blind and not bool(mask.get("blind_benchmark")):
            continue
        matching.append(row)
    return matching


def mask_unit_ids(mask: Mapping[str, Any] | None, fallback: list[str] | None) -> list[str]:
    """Selected eval `evaluate_mask.unit_ids`, else the current mode split list."""
    ids = None if mask is None else mask.get("unit_ids")
    if ids is None:
        return [str(u) for u in (fallback or [])]
    return [str(u) for u in ids]


def empty_evaluation_artifacts(*, eval_id: str | None = None, legacy: bool = False) -> dict[str, Any]:
    return {
        "eval_id": eval_id,
        "eval_dir": None,
        "legacy": bool(legacy),
        "predictions": None,
        "alerts": None,
        "metrics": None,
        "metrics_by_unit": None,
        "evaluation_config": None,
        "evaluate_mask": evaluation_mask_view(None),
    }


def list_evaluations(run_path: Path) -> list[dict[str, Any]]:
    """Immutable eval dirs under `runs/<run_id>/evaluations/`, newest first."""
    root = evaluations_dir(run_path)
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in root.iterdir():
        if not path.is_dir() or path.name.startswith("."):
            continue
        row: dict[str, Any] = {
            "eval_id": path.name,
            "path": str(path),
            "has_predictions": (path / "predictions.csv").exists(),
            "has_alerts": (path / "alerts.csv").exists(),
            "has_metrics": (path / "metrics.json").exists(),
            "legacy": False,
        }
        cfg: dict[str, Any] | None = None
        cfg_path = path / "evaluation_config.json"
        if cfg_path.exists():
            try:
                loaded = read_json(cfg_path)
                if isinstance(loaded, dict):
                    cfg = loaded
                    for key in ("metrics_version", "checkpoint_hash", "split_hash"):
                        if key in cfg:
                            row[key] = cfg[key]
            except Exception:
                cfg = None
        row["evaluate_mask"] = evaluation_mask_view(cfg)
        rows.append(row)
    rows.sort(key=lambda r: r.get("eval_id") or "", reverse=True)
    return rows


def _artifacts_from_eval_dir(eval_dir: Path) -> dict[str, Any]:
    def _optional(name: str) -> Path | None:
        p = eval_dir / name
        return p if p.exists() else None

    cfg_path = _optional("evaluation_config.json")
    cfg = None
    if cfg_path is not None:
        try:
            loaded = read_json(cfg_path)
            if isinstance(loaded, dict):
                cfg = loaded
        except Exception:
            cfg = None
    return {
        "eval_id": eval_dir.name,
        "eval_dir": eval_dir,
        "legacy": False,
        "predictions": _optional("predictions.csv"),
        "alerts": _optional("alerts.csv"),
        "metrics": _optional("metrics.json"),
        "metrics_by_unit": _optional("metrics_by_unit.csv"),
        "evaluation_config": cfg_path,
        "evaluate_mask": evaluation_mask_view(cfg),
    }


def legacy_evaluation_artifacts(run_path: Path) -> dict[str, Any]:
    """Run-root predictions/metrics. Treated as test split, not a blind benchmark."""
    run_path = Path(run_path)
    pred = run_path / LEGACY_PREDICTIONS_NAME
    alerts = run_path / LEGACY_ALERTS_NAME
    metrics = run_path / LEGACY_METRICS_NAME
    return {
        "eval_id": None,
        "eval_dir": None,
        "legacy": True,
        "predictions": pred if pred.exists() else None,
        "alerts": alerts if alerts.exists() else None,
        "metrics": metrics if metrics.exists() else None,
        "metrics_by_unit": None,
        "evaluation_config": None,
        "evaluate_mask": evaluation_mask_view(None),
    }


def resolve_evaluation_artifacts(run_path: Path, eval_id: str | None = None) -> dict[str, Any]:
    """Prefer `evaluations/<eval_id>/`. Legacy root files are read-only fallback only."""
    run_path = Path(run_path)
    if eval_id:
        dest = evaluations_dir(run_path) / eval_id
        if dest.is_dir():
            return _artifacts_from_eval_dir(dest)
        return empty_evaluation_artifacts(eval_id=eval_id)
    rows = list_evaluations(run_path)
    if rows:
        return _artifacts_from_eval_dir(Path(rows[0]["path"]))
    return legacy_evaluation_artifacts(run_path)


PRIMARY_METRIC_LABELS = {
    "equal_weight_unit_mae": "Equal-weight unit MAE",
    "prefix_end_mae": "Prefix-end MAE",
}

_ALERT_CLASS_LABELS = {
    "timely": "Timely",
    "too_early": "Too early",
    "late": "Late",
    "miss": "Miss",
    "insufficient_coverage": "Insufficient coverage",
    "mixed": "Mixed",
}


def read_evaluation_metrics(artifacts: Mapping[str, Any]) -> dict[str, Any] | None:
    path = artifacts.get("metrics")
    if path is None:
        return None
    try:
        data = read_json(path)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def read_metrics_by_unit(artifacts: Mapping[str, Any]) -> pd.DataFrame | None:
    path = artifacts.get("metrics_by_unit")
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _finite(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def format_mae_s(val: Any, dataset_id: str) -> str:
    x = _finite(val)
    if x is None:
        return "—"
    if dataset_id == "bearings":
        return f"{x:.1f} s ({x / 60.0:.2f} min)"
    return f"{x:.1f} s"


def format_duration_s(val: Any) -> str:
    x = _finite(val)
    if x is None:
        return "—"
    return f"{x:.1f} s"


def _pct(val: Any) -> str:
    x = _finite(val)
    if x is None:
        return "—"
    return f"{100.0 * x:.1f}%"


def _yes_no(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    if val is True or val == 1 or str(val).strip().lower() in {"true", "yes", "1"}:
        return "Yes"
    if val is False or val == 0 or str(val).strip().lower() in {"false", "no", "0"}:
        return "No"
    return str(val)


def _alert_class_label(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "—"
    key = str(val).strip()
    if not key or key.lower() == "nan":
        return "—"
    return _ALERT_CLASS_LABELS.get(key, key.replace("_", " "))


def _baseline_compare_block(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The stored `compare_baseline` dict inside metrics.json (bearings or filters)."""
    if not metrics:
        return None
    base = metrics.get("baseline")
    if not isinstance(base, dict):
        return None
    if "baseline_coverage_fraction" in base:
        return base
    for key in ("prefix_end", "prefix_backtest"):
        inner = base.get(key)
        if isinstance(inner, dict) and "baseline_coverage_fraction" in inner:
            return inner
    return None


def baseline_coverage_from_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    block = _baseline_compare_block(metrics)
    if block is None:
        return None
    base = metrics.get("baseline") if metrics else None
    name = base.get("name") if isinstance(base, dict) else None
    nn_all = block.get("neural_net_all_points") or {}
    nn_overlap = block.get("neural_net_baseline_overlap") or {}
    base_overlap = block.get("baseline_overlap") or {}
    return {
        "name": name or "baseline",
        "coverage_fraction": _finite(block.get("baseline_coverage_fraction")),
        "coverage_points": _finite(block.get("baseline_coverage_points")),
        "n_reference_points": _finite(block.get("n_reference_points")),
        "neural_net_mae": _finite(nn_all.get("mae") if isinstance(nn_all, dict) else None),
        "neural_net_overlap_mae": _finite(nn_overlap.get("mae") if isinstance(nn_overlap, dict) else None),
        "baseline_mae": _finite(base_overlap.get("mae") if isinstance(base_overlap, dict) else None),
        "scope": "prefix_end" if isinstance(base, dict) and "prefix_end" in base else "all_scored",
    }


def baseline_coverage_callout(metrics: Mapping[str, Any] | None) -> str | None:
    cov = baseline_coverage_from_metrics(metrics)
    if cov is None:
        return None
    n_cov = cov["coverage_points"]
    n_ref = cov["n_reference_points"]
    pts = (
        f"{int(n_cov)}/{int(n_ref)}"
        if n_cov is not None and n_ref is not None
        else "unknown coverage"
    )
    name = str(cov["name"])
    scope = "prefix-end points" if cov["scope"] == "prefix_end" else "scored points"
    text = (
        f"{name} baseline has a finite RUL on {pts} {scope} "
        f"({_pct(cov['coverage_fraction'])}). Compare MAE only on that overlap; "
        "do not treat sparse baseline hits as a full-set score against the neural net."
    )
    if cov["baseline_mae"] is not None and cov["neural_net_overlap_mae"] is not None:
        text += (
            f" Overlap MAE: baseline {format_duration_s(cov['baseline_mae'])}, "
            f"neural net {format_duration_s(cov['neural_net_overlap_mae'])}."
        )
    return text


def evaluation_notes(metrics: Mapping[str, Any] | None, dataset_id: str) -> list[str]:
    notes: list[str] = []
    if not metrics:
        return notes
    if metrics.get("note"):
        notes.append(str(metrics["note"]))
    if dataset_id == "bearings":
        n = metrics.get("n_test_units")
        n_txt = f"{int(n)} test units" if _finite(n) is not None else "test units"
        notes.append(
            f"Each of the {n_txt} is a row (XJTU-SY protocol: 3 held-out bearings, "
            "instance 5 per regime). Endpoint rows (NaN actual RUL) are excluded from MAE. "
            "Near-event zones are frozen in config, not fit on test."
        )
    elif dataset_id == "filters":
        notes.append(
            "Primary MAE is at the last sensor row of each author test prefix. "
            "Actual RUL there is official_rul_at_prefix_end_s (evaluation-only, never a model input) — "
            "not a sensor-observed 600 Pa crossing. All prefix points are a secondary backtest."
        )
        src = metrics.get("official_rul_source")
        if src:
            notes.append(str(src))
    alerts = metrics.get("alerts") if isinstance(metrics.get("alerts"), dict) else {}
    denom = (alerts or {}).get("denominator_note")
    if denom:
        notes.append(str(denom))
    return notes


def evaluation_metric_cards(metrics: Mapping[str, Any] | None, dataset_id: str) -> list[dict[str, str]]:
    if not metrics:
        return []
    cards: list[dict[str, str]] = []
    key = metrics.get("primary_metric")
    if key:
        label = PRIMARY_METRIC_LABELS.get(str(key), str(key).replace("_", " "))
        cards.append({"label": label, "value": format_mae_s(metrics.get(key), dataset_id)})
    if dataset_id == "bearings":
        if metrics.get("pooled_mae") is not None:
            cards.append({"label": "Pooled MAE", "value": format_mae_s(metrics.get("pooled_mae"), dataset_id)})
        if metrics.get("mean_overestimation") is not None:
            cards.append(
                {
                    "label": "Mean overestimation",
                    "value": format_mae_s(metrics.get("mean_overestimation"), dataset_id),
                }
            )
    elif dataset_id == "filters":
        val = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
        nll = (val or {}).get("nll_all_units")
        if nll is not None:
            x = _finite(nll)
            cards.append({"label": "Validation NLL (all units)", "value": "—" if x is None else f"{x:.4g}"})
        mae_ev = (val or {}).get("mae_observed_events")
        if mae_ev is not None:
            n_ev = (val or {}).get("n_observed_event_units")
            suffix = f" (n={int(n_ev)})" if _finite(n_ev) is not None else ""
            cards.append(
                {
                    "label": "Validation MAE (observed events)",
                    "value": format_mae_s(mae_ev, dataset_id) + suffix,
                }
            )
    alerts = metrics.get("alerts") if isinstance(metrics.get("alerts"), dict) else {}
    if alerts:
        timely = alerts.get("n_units_timely", alerts.get("timely"))
        miss = alerts.get("miss")
        insuf = alerts.get("insufficient_coverage")
        if timely is not None or miss is not None or insuf is not None:
            bits = []
            if timely is not None:
                bits.append(f"timely {int(timely)}" if _finite(timely) is not None else f"timely {timely}")
            if miss is not None:
                bits.append(f"miss {int(miss)}" if _finite(miss) is not None else f"miss {miss}")
            if insuf is not None:
                bits.append(
                    f"insufficient coverage {int(insuf)}" if _finite(insuf) is not None else f"insufficient {insuf}"
                )
            cards.append({"label": "Alert classes", "value": ", ".join(bits) if bits else "—"})
    return cards


def near_event_zone_frame(metrics: Mapping[str, Any] | None) -> pd.DataFrame | None:
    if not metrics:
        return None
    zones = metrics.get("equal_weight_unit_mae_by_zone")
    if not isinstance(zones, dict) or not zones:
        return None
    order = metrics.get("near_event_zones_s") or list(zones.keys())
    rows = []
    for z in order:
        if isinstance(z, (int, float)) and float(z).is_integer():
            key = str(int(z))
        else:
            key = str(z)
        val = zones.get(key, zones.get(str(z)))
        rows.append({"Zone (s)": key, "Equal-weight unit MAE (s)": val})
    return pd.DataFrame(rows)


def metrics_by_unit_display_frame(
    by_unit: pd.DataFrame | None,
    *,
    dataset_id: str,
    unit_ids: list[str] | None = None,
) -> pd.DataFrame:
    """English per-unit table. Lists mask / mode units even if a CSV row is missing."""
    src = by_unit.copy() if by_unit is not None else pd.DataFrame()
    if not src.empty and "unit_id" in src.columns:
        src["unit_id"] = src["unit_id"].astype(str)
    elif src.empty:
        src = pd.DataFrame(columns=["unit_id"])
    if unit_ids:
        ordered = pd.DataFrame({"unit_id": [str(u) for u in unit_ids]})
        src = ordered.merge(src, on="unit_id", how="left")
    out = pd.DataFrame()
    if "unit_id" in src.columns:
        out["Unit"] = src["unit_id"].astype(str)
    else:
        out["Unit"] = []
    if "mae" in src.columns:
        mae = src["mae"]
    elif "prefix_end_abs_error" in src.columns:
        mae = src["prefix_end_abs_error"]
    else:
        mae = None
    if mae is not None:
        out["MAE (s)"] = mae
    if "nll" in src.columns:
        out["NLL"] = src["nll"]
    if dataset_id == "filters":
        if "prefix_end_actual_rul_s" in src.columns:
            out["Official RUL at prefix end (s)"] = src["prefix_end_actual_rul_s"]
        if "prefix_end_predicted_rul_s" in src.columns:
            out["Predicted RUL at prefix end (s)"] = src["prefix_end_predicted_rul_s"]
        if "backtest_mae" in src.columns:
            out["Prefix backtest MAE (s)"] = src["backtest_mae"]
    if "alert_outcome" in src.columns:
        out["Alert class"] = [_alert_class_label(v) for v in src["alert_outcome"]]
    if "lead_time_s" in src.columns:
        out["Lead time (s)"] = src["lead_time_s"]
    if "has_sufficient_coverage" in src.columns:
        out["Sufficient coverage"] = [_yes_no(v) for v in src["has_sufficient_coverage"]]
    if "baseline_coverage_fraction" in src.columns:
        out["Baseline coverage"] = [_pct(v) for v in src["baseline_coverage_fraction"]]
    return out
