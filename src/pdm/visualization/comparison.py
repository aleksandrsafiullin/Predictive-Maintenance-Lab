"""Collect and filter run metrics for cross-architecture comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from pdm.architectures import is_recurrent_nn, is_reservoir
from pdm.connectome.provenance import GRAPH_MODE_SYNTHETIC, SYNTHETIC_DISCLAIMER

SYNTHETIC_SECTION = SYNTHETIC_DISCLAIMER
BIOLOGICAL_SECTION = "Architecture comparison (same split)"
SYNTHETIC_LABEL = "[SYNTHETIC]"
SMOKE_NOTE = "Smoke test — not a quality benchmark"
_TABLE_COLUMNS = [
    "run_id",
    "architecture",
    "graph_mode",
    "n_nodes",
    "best_metric",
    "split_hash",
    "smoke",
    "matched_control",
    "warning",
    "label",
    "section",
]


def is_synthetic_run(run: Mapping[str, Any] | None) -> bool:
    """True for synthetic_fixture graphs or runs labeled synthetic. Never auto-promote."""
    rec = run or {}
    if bool(rec.get("is_synthetic")):
        return True
    mode = str(rec.get("graph_mode") or "").strip().lower()
    return mode == GRAPH_MODE_SYNTHETIC


def filter_synthetic_runs(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (real_runs, synthetic_runs). Synthetic is never mixed into real."""
    real_runs: list[dict] = []
    synthetic_runs: list[dict] = []
    for rec in runs or []:
        if is_synthetic_run(rec):
            synthetic_runs.append(rec)
        else:
            real_runs.append(rec)
    return real_runs, synthetic_runs


def _finite_number(val: Any) -> float | None:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        x = float(val)
    except (TypeError, ValueError):
        return None
    if not pd.notna(x):
        return None
    return x if x == x and abs(x) != float("inf") else None


def _norm_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return ""
    return str(val).strip()


def _same_number(left: Any, right: Any, *, as_int: bool = False) -> bool:
    a = _finite_number(left)
    b = _finite_number(right)
    if a is None or b is None:
        return False
    if as_int:
        return int(a) == int(b)
    return abs(a - b) <= 1e-8 * max(1.0, abs(a), abs(b))


def check_matched_control(fly_run: dict, random_run: dict) -> tuple[bool, str]:
    """Matched N/E control: same hyperparams/split, and random parent == fly graph hash."""
    fly = fly_run or {}
    rnd = random_run or {}
    fly_arch = _norm_str(fly.get("architecture")).lower()
    rnd_arch = _norm_str(rnd.get("architecture")).lower()
    if fly_arch and fly_arch != "fly_connectome_reservoir":
        return False, "Fly row is not architecture fly_connectome_reservoir."
    if rnd_arch and rnd_arch != "random_reservoir":
        return False, "Random row is not architecture random_reservoir."
    if _norm_str(fly.get("dataset_id")) != _norm_str(rnd.get("dataset_id")):
        return False, "dataset_id does not match; never mix bearings and filters."
    mismatches: list[str] = []
    if not _same_number(fly.get("n_nodes"), rnd.get("n_nodes"), as_int=True):
        mismatches.append("n_nodes")
    if not _same_number(fly.get("leak"), rnd.get("leak")):
        mismatches.append("leak")
    if not _same_number(fly.get("spectral_radius"), rnd.get("spectral_radius")):
        mismatches.append("spectral_radius")
    if _norm_str(fly.get("state_mode")).lower() != _norm_str(rnd.get("state_mode")).lower():
        mismatches.append("state_mode")
    if not _same_number(fly.get("history_length"), rnd.get("history_length"), as_int=True):
        mismatches.append("history_length")
    if _norm_str(fly.get("split_hash")) != _norm_str(rnd.get("split_hash")):
        mismatches.append("split_hash")
    parent = _norm_str(rnd.get("parent_graph_hash"))
    fly_hash = _norm_str(fly.get("graph_hash"))
    if not parent or not fly_hash or parent != fly_hash:
        mismatches.append("parent_graph_hash")
    if mismatches:
        if "parent_graph_hash" in mismatches:
            return (
                False,
                "Random run is not a matched control "
                "(parent_graph_hash must equal the fly run graph_hash).",
            )
        return False, "Matched-control fields differ: " + ", ".join(mismatches) + "."
    return True, ""


def _primary_metric_value(metrics: Mapping[str, Any] | None) -> Any:
    if not metrics:
        return None
    key = metrics.get("primary_metric")
    if key and key in metrics:
        return metrics.get(key)
    for fallback in ("equal_weight_unit_mae", "prefix_end_mae", "pooled_mae"):
        if metrics.get(fallback) is not None:
            return metrics.get(fallback)
    return None


def _baseline_mae(metrics: Mapping[str, Any] | None) -> Any:
    if not metrics:
        return None
    base = metrics.get("baseline")
    if not isinstance(base, dict):
        return None
    overlap = base.get("baseline_overlap")
    if isinstance(overlap, dict) and overlap.get("mae") is not None:
        return overlap.get("mae")
    for key in ("prefix_end", "prefix_backtest"):
        inner = base.get(key)
        if not isinstance(inner, dict):
            continue
        ov = inner.get("baseline_overlap")
        if isinstance(ov, dict) and ov.get("mae") is not None:
            return ov.get("mae")
    return None


def _nll_value(metrics: Mapping[str, Any] | None) -> Any:
    if not metrics:
        return None
    val = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    if val.get("nll_all_units") is not None:
        return val.get("nll_all_units")
    return metrics.get("nll")


def _alert_summary(metrics: Mapping[str, Any] | None) -> str | None:
    if not metrics:
        return None
    alerts = metrics.get("alerts") if isinstance(metrics.get("alerts"), dict) else {}
    if not alerts:
        return None
    bits = []
    for key, label in (("n_units_timely", "timely"), ("timely", "timely"), ("miss", "miss")):
        if key in alerts and alerts.get(key) is not None:
            bits.append(f"{label} {alerts.get(key)}")
            if label == "timely":
                break
    return ", ".join(bits) if bits else None


def _metric_dict_from_run(
    rec: Mapping[str, Any],
    *,
    metrics: Mapping[str, Any] | None,
    dataset_id: str,
) -> dict[str, Any]:
    arch = _norm_str(rec.get("architecture")).lower()
    row: dict[str, Any] = {
        "run_id": rec.get("run_id"),
        "architecture": arch or rec.get("architecture"),
        "graph_mode": rec.get("graph_mode"),
        "is_synthetic": bool(rec.get("is_synthetic")) or is_synthetic_run(rec),
        "parent_graph_hash": rec.get("parent_graph_hash"),
        "graph_hash": rec.get("graph_hash"),
        "n_nodes": rec.get("n_nodes") if is_reservoir(arch) else None,
        "split_hash": rec.get("split_hash"),
        "dataset_id": rec.get("dataset_id") or dataset_id,
        "best_metric": rec.get("best_metric"),
        "head": rec.get("head"),
        "smoke": bool(rec.get("smoke")),
        "leak": rec.get("leak"),
        "spectral_radius": rec.get("spectral_radius"),
        "state_mode": rec.get("state_mode"),
        "history_length": rec.get("history_length"),
        "status": rec.get("status"),
    }
    if metrics:
        primary = _primary_metric_value(metrics)
        if primary is not None:
            row["best_metric"] = primary
        row["primary_metric"] = metrics.get("primary_metric")
        row["equal_weight_unit_mae"] = metrics.get("equal_weight_unit_mae")
        row["prefix_end_mae"] = metrics.get("prefix_end_mae")
        row["baseline_mae"] = _baseline_mae(metrics)
        row["nll"] = _nll_value(metrics)
        row["alert_summary"] = _alert_summary(metrics)
        if metrics.get("split_hash") and not row.get("split_hash"):
            row["split_hash"] = metrics.get("split_hash")
    if is_recurrent_nn(arch):
        row["n_nodes"] = None
        row["graph_mode"] = row.get("graph_mode") or ""
    return row


def collect_run_metrics(dataset_id: str) -> list[dict]:
    """Return run metric dicts from finished eval artifacts (plus training identity)."""
    from pdm.experiments import (
        list_runs,
        read_evaluation_metrics,
        resolve_evaluation_artifacts,
        run_dir,
    )

    want = str(dataset_id)
    rows: list[dict] = []
    for rec in list_runs(want):
        ds = str(rec.get("dataset_id") or want)
        if ds != want:
            continue
        run_id = rec.get("run_id")
        if not run_id:
            continue
        path = rec.get("path")
        rdir = Path(path) if path else run_dir(want, str(run_id))
        metrics = None
        try:
            artifacts = resolve_evaluation_artifacts(rdir)
            metrics = read_evaluation_metrics(artifacts)
        except Exception:  # noqa: BLE001
            metrics = None
        row = _metric_dict_from_run(rec, metrics=metrics, dataset_id=want)
        row["dataset_id"] = want
        rows.append(row)
    return rows


def _architecture_label(rec: Mapping[str, Any], *, synthetic: bool) -> str:
    arch = _norm_str(rec.get("architecture")) or "unknown"
    if synthetic:
        return f"{arch} {SYNTHETIC_LABEL}"
    return arch


def _matched_flags(runs: list[dict]) -> dict[str, tuple[bool | None, str]]:
    """Per run_id: (matched_control, warning). GRU/LSTM are n/a (None, '')."""
    fly = [r for r in runs if _norm_str(r.get("architecture")).lower() == "fly_connectome_reservoir"]
    rnd = [r for r in runs if _norm_str(r.get("architecture")).lower() == "random_reservoir"]
    out: dict[str, tuple[bool | None, str]] = {}
    for rec in runs:
        rid = str(rec.get("run_id") or "")
        arch = _norm_str(rec.get("architecture")).lower()
        if is_recurrent_nn(arch) or not is_reservoir(arch):
            out[rid] = (None, "")
            continue
        if arch == "random_reservoir":
            if not fly:
                out[rid] = (False, "No fly_connectome_reservoir run on this split for a matched control.")
                continue
            ok_any = False
            warn = ""
            for parent in fly:
                matched, msg = check_matched_control(parent, rec)
                if matched:
                    ok_any = True
                    warn = ""
                    break
                warn = msg
            out[rid] = (ok_any, "" if ok_any else warn)
            continue
        if arch == "fly_connectome_reservoir":
            if not rnd:
                out[rid] = (None, "")
                continue
            ok_any = False
            warn = ""
            for other in rnd:
                matched, msg = check_matched_control(rec, other)
                if matched:
                    ok_any = True
                    warn = ""
                    break
                warn = msg
            out[rid] = (ok_any, "" if ok_any else warn)
            continue
        out[rid] = (None, "")
    return out


def _table_rows(runs: list[dict], *, section: str, synthetic: bool) -> list[dict[str, Any]]:
    flags = _matched_flags(runs)
    rows: list[dict[str, Any]] = []
    for rec in runs:
        rid = str(rec.get("run_id") or "")
        matched, warning = flags.get(rid, (None, ""))
        smoke = bool(rec.get("smoke"))
        label = _architecture_label(rec, synthetic=synthetic)
        if smoke:
            label = f"{label} (smoke)"
        n_nodes = rec.get("n_nodes") if is_reservoir(_norm_str(rec.get("architecture"))) else None
        rows.append(
            {
                "run_id": rec.get("run_id"),
                "architecture": rec.get("architecture"),
                "graph_mode": rec.get("graph_mode"),
                "n_nodes": n_nodes,
                "best_metric": rec.get("best_metric"),
                "split_hash": rec.get("split_hash"),
                "smoke": smoke,
                "matched_control": matched,
                "warning": warning,
                "label": label,
                "section": section,
                "baseline_mae": rec.get("baseline_mae"),
                "dataset_id": rec.get("dataset_id"),
            }
        )
    return rows


def build_comparison_table(
    dataset_id: str,
    run_ids: list[str] | None = None,
    *,
    runs: list[dict] | None = None,
) -> pd.DataFrame:
    """Same-split architecture table. Synthetic rows are labeled and never mixed with real."""
    want = str(dataset_id)
    source = list(runs) if runs is not None else collect_run_metrics(want)
    filtered: list[dict] = []
    allow = {str(r) for r in run_ids} if run_ids is not None else None
    for rec in source:
        ds = str(rec.get("dataset_id") or want)
        if ds != want:
            continue
        if allow is not None and str(rec.get("run_id")) not in allow:
            continue
        filtered.append(rec)
    real_runs, synthetic_runs = filter_synthetic_runs(filtered)
    rows = _table_rows(real_runs, section=BIOLOGICAL_SECTION, synthetic=False)
    rows.extend(_table_rows(synthetic_runs, section=SYNTHETIC_SECTION, synthetic=True))
    frame = pd.DataFrame(rows)
    for col in _TABLE_COLUMNS:
        if col not in frame.columns:
            frame[col] = None
    ordered = _TABLE_COLUMNS + [c for c in frame.columns if c not in _TABLE_COLUMNS]
    return frame[ordered]


def comparison_sections(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a comparison table into (real, synthetic) frames."""
    if table is None or table.empty or "section" not in table.columns:
        empty = table.iloc[0:0].copy() if isinstance(table, pd.DataFrame) else pd.DataFrame()
        return empty, empty
    real = table[table["section"] != SYNTHETIC_SECTION].copy()
    synth = table[table["section"] == SYNTHETIC_SECTION].copy()
    return real.reset_index(drop=True), synth.reset_index(drop=True)
