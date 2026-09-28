"""Causal future-red binary models for bearings and filter sensor zones.

This module intentionally consumes the versioned target artifact instead of
reconstructing labels from endpoint/RUL fields. Test labels are read only for
the final frozen evaluation.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from pdm.config import load_dataset_config
from pdm.data.prepare import load_processed
from pdm.feature_recipes import recipe_names
from pdm.models.recurrent import RecurrentEncoder
from pdm.preprocessing import (
    FEATURE_PIPELINE_VERSION,
    bearings_feature_frame,
    filters_feature_frame,
    raw_to_feature_frame,
)
from pdm.windows import FORBIDDEN_FEATURE_NAMES

FUTURE_RED_MODEL_VERSION = "future_red_binary_v1"
REPLAY_CONTRACT_VERSION = "future_red_replay_v1"
SPLITS = ("train", "validation", "test")
_NON_FEATURE_COLUMNS = {
    "unit_id", "origin_unit_id", "timestamp_s", "row_index", "sample_index",
    "fragment_index", "author_data_no", "author_split", "regime_id", "split",
}
_NON_SENSOR_MODEL_FEATURES = {"operating_age_s"}


def fit_train_scaler(features: pd.DataFrame, train_units: list[str] | set[str],
                     feature_names: list[str] | None = None) -> dict[str, Any]:
    """Fit a finite-value mean/std scaler on training-unit rows only."""
    names = list(feature_names) if feature_names is not None else [
        str(c) for c in features.columns
        if c not in _NON_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(features[c])
    ]
    # Lifetime is a direct proxy for proximity to an observed endpoint. Keep
    # the future-red classifier tied to its causal sensor history instead.
    names = [name for name in names if name.lower() not in _NON_SENSOR_MODEL_FEATURES]
    if not names:
        raise ValueError("No numeric sensor features available")
    forbidden = sorted(set(n.lower() for n in names) & FORBIDDEN_FEATURE_NAMES)
    if forbidden:
        raise ValueError(f"Refusing future-red target leakage features: {forbidden}")
    missing = sorted(set(names) - set(features.columns))
    if missing:
        raise ValueError(f"Missing requested sensor features: {missing}")
    train = features.loc[features.unit_id.astype(str).isin({str(x) for x in train_units}), names]
    if train.empty:
        raise ValueError("No feature rows for training units")
    values = train.to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    mean = np.divide(np.where(finite, values, 0.0).sum(axis=0), count,
                     out=np.zeros(len(names), dtype=np.float64), where=count > 0)
    centered = np.where(finite, values - mean, 0.0)
    variance = np.divide((centered * centered).sum(axis=0), count,
                         out=np.ones(len(names), dtype=np.float64), where=count > 0)
    scale = np.sqrt(variance)
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"feature_names": names, "mean": mean.tolist(), "scale": scale.tolist(),
            "fit_unit_ids": sorted({str(x) for x in train_units}), "fit_rows": int(len(train))}


def prepare_future_red_sequences(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    split: dict[str, Any],
    *,
    history_length: int = 20,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build right-aligned, past-only windows and train-only standardized inputs.

    A short prefix is left-padded with its earliest observed row. An observed
    gap is never crossed: continuity is interpreted in the supplied features
    and target artifact, and only the last ``history_length`` rows through the
    target timestamp are used.
    """
    if history_length < 1:
        raise ValueError("history_length must be positive")
    required = {"unit_id", "timestamp_s", "target", "target_known", "split",
                "at_risk", "first_red_timestamp_s"}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise ValueError(f"future-red target artifact missing columns: {missing}")
    train_ids = {str(x) for x in split.get("train", [])}
    scaler = fit_train_scaler(features, train_ids, feature_names)
    names = scaler["feature_names"]
    mean = np.asarray(scaler["mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)

    feature_groups = {}
    for uid, group in features.groupby(features.unit_id.astype(str), sort=False):
        ordered = group.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        raw = ordered[names].to_numpy(dtype=np.float64).copy()
        raw[~np.isfinite(raw)] = np.nan
        # Missing sensor values are imputed with training means, hence zero after scaling.
        vals = np.where(np.isfinite(raw), raw, mean)
        gap = ordered.get("gap_before", pd.Series(False, index=ordered.index)).fillna(False).astype(bool).to_numpy()
        feature_groups[str(uid)] = (ordered.timestamp_s.to_numpy(dtype=np.float64),
                                    ((vals - mean) / scale).astype(np.float32), gap)

    output: dict[str, Any] = {"scaler": scaler, "feature_names": names,
                              "history_length": int(history_length)}
    labels = targets.copy()
    labels["unit_id"] = labels.unit_id.astype(str)
    for part in SPLITS:
        unit_ids = {str(x) for x in split.get(part, [])}
        rows = labels.loc[labels.unit_id.isin(unit_ids)].copy()
        seqs: list[np.ndarray] = []
        kept: list[dict[str, Any]] = []
        for row in rows.sort_values(["unit_id", "timestamp_s"], kind="stable").to_dict("records"):
            uid = str(row["unit_id"])
            if uid not in feature_groups:
                continue
            times, values, gap = feature_groups[uid]
            idx = int(np.searchsorted(times, float(row["timestamp_s"]), side="right") - 1)
            if idx < 0 or not np.isclose(times[idx], float(row["timestamp_s"]), rtol=0, atol=1e-7):
                continue
            segment_start = int(np.flatnonzero(gap[: idx + 1])[-1]) if gap[: idx + 1].any() else 0
            begin = max(segment_start, idx - history_length + 1)
            seq = values[begin : idx + 1]
            if len(seq) < history_length:
                seq = np.concatenate([np.repeat(seq[:1], history_length - len(seq), axis=0), seq], axis=0)
            seqs.append(seq)
            row["split"] = part
            row["target"] = float(row["target"]) if pd.notna(row["target"]) else np.nan
            row["target_known"] = bool(row["target_known"])
            kept.append(row)
        output[f"X_{part}"] = np.stack(seqs).astype(np.float32) if seqs else np.empty(
            (0, history_length, len(names)), dtype=np.float32)
        output[f"y_{part}"] = np.asarray([r["target"] for r in kept], dtype=np.float32)
        output[f"rows_{part}"] = pd.DataFrame(kept)
    output["split"] = {key: [str(v) for v in split.get(key, [])] for key in SPLITS}
    return output


def transform_future_red_features(dataset_id: str, features: pd.DataFrame,
                                  split: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Apply the existing causal feature recipe and return only declared inputs."""
    cfg = load_dataset_config(dataset_id)
    recipe = str(cfg.get("feature_recipe", "base_v1"))
    train_ids = {str(x) for x in split.get("train", [])}
    log_cols = list(cfg.get("features", {}).get("log1p_features") or []) if dataset_id == "bearings" else []
    if dataset_id == "filters":
        cats = {"dust": sorted(features.loc[features.unit_id.astype(str).isin(train_ids), "dust"]
                                .astype(str).unique().tolist())}
    else:
        cats = {}
        if recipe in {"multiscale_trend_v1", "multiscale_no_age_v1",
                      "multiscale_trend_v2", "multiscale_no_age_v2"}:
            cats["regime_id"] = sorted(features.loc[features.unit_id.astype(str).isin(train_ids), "regime_id"]
                                       .astype(str).unique().tolist())
    encoded = raw_to_feature_frame(dataset_id, features, cats, log_cols,
                                   raw_features=True, feature_recipe=recipe)
    if dataset_id == "bearings":
        _, names, _ = bearings_feature_frame(encoded, log_cols)
    elif dataset_id == "filters":
        _, names, _ = filters_feature_frame(encoded, None, cats.get("dust", []))
    else:
        raise ValueError(f"Unsupported future-red dataset {dataset_id!r}")
    names = names + recipe_names(dataset_id, recipe)
    names = [name for name in names if name.lower() not in _NON_SENSOR_MODEL_FEATURES]
    forbidden = sorted(set(n.lower() for n in names) & FORBIDDEN_FEATURE_NAMES)
    if forbidden:
        raise ValueError(f"Refusing future-red target leakage features: {forbidden}")
    return encoded, names


def select_validation_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Select a F1-maximizing threshold using validation rows only."""
    y = np.asarray(y_true, dtype=np.int8).reshape(-1)
    p = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if y.size == 0 or y.size != p.size or len(np.unique(y)) < 2:
        return 0.5, {"selection": "default_0.5_insufficient_validation_classes",
                     "selection_unit": "origin_rows"}
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], p)))
    best = None
    for threshold in candidates:
        pred = p >= threshold
        tp = int(np.sum(pred & (y == 1)))
        fp = int(np.sum(pred & (y == 0)))
        fn = int(np.sum(~pred & (y == 1)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        candidate = (f1, precision, recall, float(threshold))
        if best is None or candidate > best:
            best = candidate
    assert best is not None
    return best[3], {"selection": "validation_max_f1", "selection_unit": "origin_rows",
                     "f1": best[0], "precision": best[1], "recall": best[2],
                     "instability_note": "Row-level F1 can be unstable when validation contains few event units."}


def event_level_metrics(rows: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    """Per-unit event detection; units without a known red event remain unknown."""
    if rows.empty:
        return {"event_observed_units": 0, "detected_event_units": 0,
                "missed_event_units": 0, "unknown_censored_units": 0,
                "event_recall": None}
    work = rows.copy()
    if "at_risk" in work:
        mask = work.at_risk.astype(bool).to_numpy()
        work = work.loc[mask].copy()
        probabilities = np.asarray(probabilities, dtype=float)[mask]
    work["probability"] = np.asarray(probabilities, dtype=float)
    work["predicted_red"] = work.probability >= float(threshold)
    outcomes = []
    unknown = 0
    alerted_unknown = 0
    early_unmatched = 0
    early_negative_rows = 0
    for uid, group in work.groupby("unit_id", sort=False):
        event_times = group.first_red_timestamp_s.dropna()
        if event_times.empty:
            unknown += 1
            alerted_unknown += int(bool(group.predicted_red.any()))
            outcomes.append({"unit_id": str(uid), "event_observed": False,
                             "outcome": "unresolved_censored",
                             "detected": None, "warning": bool(group.predicted_red.any()),
                             "lead_time_s": None})
            continue
        event_time = float(event_times.iloc[0])
        positive_target = group.target.eq(1).fillna(False)
        timely = group.loc[group.predicted_red & positive_target]
        detected = not timely.empty
        any_warning = bool(group.predicted_red.any())
        early_negative_rows += int((group.predicted_red & group.target.eq(0)).sum())
        early_unmatched += int(any_warning and not detected)
        first_warning = float(timely.timestamp_s.min()) if detected else None
        outcomes.append({"unit_id": str(uid), "event_observed": True,
                         "outcome": "detected" if detected else "missed_event",
                         "detected": detected, "warning": any_warning,
                         "lead_time_s": event_time - first_warning if detected else None})
    observed = sum(bool(x["event_observed"]) for x in outcomes)
    detected = sum(bool(x["detected"]) for x in outcomes)
    return {"event_observed_units": observed, "detected_event_units": detected,
            "missed_event_units": observed - detected, "unknown_censored_units": unknown,
            "event_recall": detected / observed if observed else None,
            "mean_lead_time_s": float(np.mean([x["lead_time_s"] for x in outcomes if x["lead_time_s"] is not None]))
            if detected else None,
            "alerted_observed_units": sum(bool(x["warning"]) for x in outcomes),
            "early_or_unmatched_warning_units": early_unmatched,
            "early_warning_known_negative_rows": early_negative_rows,
            "alerted_censored_unknown_units": alerted_unknown,
            "alert_burden_fraction": float(work.predicted_red.mean()) if len(work) else None,
            "per_unit": outcomes}


def _binary_metrics(rows: pd.DataFrame, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    y = rows.target.to_numpy(dtype=np.int8) if len(rows) else np.empty(0, dtype=np.int8)
    p = np.asarray(probabilities, dtype=float)
    pred = p >= threshold
    tp, fp, fn, tn = (int(np.sum(pred & (y == 1))), int(np.sum(pred & (y == 0))),
                      int(np.sum(~pred & (y == 1))), int(np.sum(~pred & (y == 0)))
                      ) if len(y) else (0, 0, 0, 0)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    from sklearn.metrics import average_precision_score

    return {"known_rows": int(len(y)), "positive_rows": int(y.sum()), "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision and recall else 0.0,
            "brier_score": float(np.mean((p - y) ** 2)) if len(y) else None,
            "average_precision": float(average_precision_score(y, p)) if len(y) and len(np.unique(y)) > 1 else None,
            "event_level": event_level_metrics(rows, p, threshold)}


def evaluate_future_red_baselines(
    prepared: dict[str, Any],
    baseline_probabilities: dict[str, dict[str, np.ndarray]],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Score transparent baselines on identical per-split origin clocks.

    ``baseline_probabilities[name][split]`` must align one-for-one with the
    supplied ``rows_{split}``, including unknown and out-of-risk origins. The
    caller can therefore provide always-no-entry, current-state persistence,
    and causal trend-to-red forecasts without changing eligibility or using
    test outcomes to tune a rule.
    """
    result: dict[str, Any] = {}
    for name, by_split in baseline_probabilities.items():
        result[name] = {}
        for part in SPLITS:
            rows = prepared[f"rows_{part}"]
            probs = np.asarray(by_split[part], dtype=float).reshape(-1)
            if len(probs) != len(rows):
                raise ValueError(f"Baseline {name!r} {part} predictions are not on the origin clock")
            known = rows.target_known.astype(bool) & rows.at_risk.astype(bool) & rows.target.notna()
            result[name][part] = _binary_metrics(rows.loc[known], probs[known.to_numpy()], threshold)
            if "at_risk" in rows:
                at_risk = rows.at_risk.astype(bool).to_numpy()
                result[name][part]["event_level"] = event_level_metrics(
                    rows.loc[at_risk], probs[at_risk], threshold)
            if part == "test" and str((prepared.get("target_manifest") or {}).get("dataset_id", "")) == "filters":
                result[name][part].pop("event_level", None)
                result[name][part]["test_event_recall"] = "not_reported_by_design"
    return result


def build_future_red_baselines(
    dataset_id: str,
    features: pd.DataFrame,
    prepared: dict[str, Any],
    *,
    horizon_s: float,
    zone_policy: dict[str, Any],
    trend_points: int = 5,
) -> dict[str, dict[str, np.ndarray]]:
    """Create no-entry, current-red persistence, and causal trend forecasts.

    Rules use raw sensor measurements only. Bearing thresholds come from the
    source zone-label policy; filter red pressure is the saved >=600 Pa policy.
    The trailing trend uses at most ``trend_points`` observations since the
    latest causal gap marker and linearly extrapolates to the saved red limit.
    """
    if horizon_s <= 0 or trend_points < 2:
        raise ValueError("horizon_s must be positive and trend_points must be at least 2")
    predictions = {name: {} for name in ("always_no_entry", "current_red_persistence", "trend_to_red")}
    policy = dict(zone_policy or {})
    for part in SPLITS:
        rows = prepared[f"rows_{part}"]
        trend_values = np.zeros(len(rows), dtype=np.float32)
        persistent = np.zeros(len(rows), dtype=np.float32)
        for uid, positions in rows.groupby(rows.unit_id.astype(str), sort=False).groups.items():
            group = features.loc[features.unit_id.astype(str).eq(str(uid))].sort_values(
                "timestamp_s", kind="stable").reset_index(drop=True)
            if group.empty:
                continue
            times = group.timestamp_s.to_numpy(dtype=float)
            if dataset_id == "bearings":
                signal = np.maximum(group.horizontal_rms.to_numpy(dtype=float),
                                     group.vertical_rms.to_numpy(dtype=float))
                baseline_n = int(policy.get("baseline_n_measurements", 5))
                threshold_ratio = float(policy.get("red_threshold_ratio", 2.0))
                threshold = None
            elif dataset_id == "filters":
                signal = group.differential_pressure.to_numpy(dtype=float)
                threshold = float(policy.get("red_limit_pa", 600.0))
            else:
                raise ValueError(f"Unsupported future-red dataset {dataset_id!r}")
            gaps = group.get("gap_before", pd.Series(False, index=group.index)).fillna(False).astype(bool).to_numpy()
            for row_index in positions:
                row = rows.loc[row_index]
                idx = int(np.searchsorted(times, float(row.timestamp_s), side="right") - 1)
                if idx < 0 or not np.isclose(times[idx], float(row.timestamp_s), rtol=0, atol=1e-7):
                    continue
                current = float(signal[idx])
                if dataset_id == "bearings":
                    baseline = float(np.median(signal[: min(idx + 1, baseline_n)]))
                    threshold = baseline * threshold_ratio
                is_red = bool(np.isfinite(current) and current >= threshold)
                if dataset_id == "bearings" and idx < baseline_n:
                    is_red = False
                persistent[rows.index.get_loc(row_index)] = float(is_red)
                start = max(0, idx - trend_points + 1)
                if gaps[: idx + 1].any():
                    start = max(start, int(np.flatnonzero(gaps[: idx + 1])[-1]))
                tx = times[start : idx + 1]
                sy = signal[start : idx + 1]
                finite = np.isfinite(tx) & np.isfinite(sy)
                tx, sy = tx[finite], sy[finite]
                forecast = is_red
                if len(tx) >= 2 and not is_red and np.ptp(tx) > 0:
                    slope = float(np.polyfit(tx - tx[-1], sy, 1)[0])
                    if slope > 0:
                        crossing_s = (threshold - current) / slope
                        forecast = 0 <= crossing_s <= float(horizon_s)
                trend_values[rows.index.get_loc(row_index)] = float(forecast)
        predictions["always_no_entry"][part] = np.zeros(len(rows), dtype=np.float32)
        predictions["current_red_persistence"][part] = persistent
        predictions["trend_to_red"][part] = trend_values
    return predictions


def _write_outputs(prepared: dict[str, Any], probabilities: dict[str, np.ndarray],
                   threshold: float, output_dir: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    prediction_parts = []
    metrics = {}
    for part in SPLITS:
        rows = prepared[f"rows_{part}"].copy()
        probs = np.asarray(probabilities[part], dtype=float)
        if len(rows) != len(probs):
            raise ValueError(f"{part} rows/probabilities length mismatch")
        if len(rows):
            rows["probability"] = probs
            rows["prediction"] = (probs >= threshold).astype(np.int8)
            prediction_parts.append(rows)
        known = rows.target_known.astype(bool) & rows.at_risk.astype(bool) & rows.target.notna()
        metrics[part] = _binary_metrics(rows.loc[known], probs[known.to_numpy()], threshold)
        if "at_risk" in rows:
            at_risk = rows.at_risk.astype(bool).to_numpy()
            metrics[part]["event_level"] = event_level_metrics(rows.loc[at_risk], probs[at_risk], threshold)
        dataset = metadata.get("dataset_id") or (metadata.get("target_manifest") or {}).get("dataset_id")
        if part == "test" and dataset == "filters":
            metrics[part].pop("event_level", None)
            metrics[part]["test_event_recall"] = "not_reported_by_design"
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    predictions.to_parquet(out / "predictions.parquet", index=False)
    per_unit = []
    for part in SPLITS:
        per_unit.extend(metrics[part].get("event_level", {}).get("per_unit", []))
    pd.DataFrame(per_unit).to_csv(out / "per_unit_metrics.csv", index=False)
    manifest = {"schema_version": FUTURE_RED_MODEL_VERSION, "threshold": float(threshold),
                "threshold_policy": "validation_max_f1_or_0.5_if_unavailable",
                "baseline_definitions": prepared.get("baseline_definitions", {}),
                "metrics": metrics, **metadata}
    output_names = ["predictions.parquet", "per_unit_metrics.csv"]
    if prepared.get("baseline_probabilities") is not None:
        manifest["baseline_metrics"] = _persist_baselines(prepared, output_dir)
        output_names.extend(("baseline_predictions.parquet", "baseline_metrics.json"))
    manifest["artifact_hashes"] = {
        **(manifest.get("artifact_hashes") or {}),
        **{name: _file_sha256(out / name) for name in output_names},
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return manifest


def _persist_baselines(prepared: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    baseline_probabilities = prepared["baseline_probabilities"]
    metrics = evaluate_future_red_baselines(prepared, baseline_probabilities)
    frames = []
    for name, by_split in baseline_probabilities.items():
        for part in SPLITS:
            rows = prepared[f"rows_{part}"].copy()
            rows["baseline"] = name
            rows["probability"] = np.asarray(by_split[part], dtype=float)
            frames.append(rows)
    baseline_rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    out = Path(output_dir)
    baseline_rows.to_parquet(out / "baseline_predictions.parquet", index=False)
    (out / "baseline_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return metrics


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_future_red_reservoir_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Verify saved Fly/Random reservoir files against their run manifest."""
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest.get("model_contract") or {}
    architecture = contract.get("architecture")
    if architecture not in {"fly_connectome_reservoir", "random_reservoir"}:
        raise ValueError("Run manifest does not describe a Fly or Random reservoir")
    if contract.get("version") != REPLAY_CONTRACT_VERSION:
        raise ValueError("Unsupported reservoir replay contract")
    expected_names = {"reservoir_weights.npz", "graph.json", "readout.pt"}
    if architecture == "random_reservoir":
        expected_names.add("parent_graph.json")
    hashes = manifest.get("artifact_hashes") or {}
    if set(hashes) != expected_names:
        raise ValueError("Reservoir manifest artifact hash list is incomplete or unexpected")
    for name in sorted(expected_names):
        artifact = root / name
        if not artifact.is_file() or _file_sha256(artifact) != hashes[name]:
            raise ValueError(f"Reservoir artifact hash mismatch: {name}")
    return manifest


def fit_future_red_model(
    dataset_id: str,
    architecture: str,
    *,
    prepared: dict[str, Any],
    output_dir: str | Path,
    seed: int = 42,
    epochs: int = 30,
    batch_size: int = 64,
    hidden_size: int = 64,
    num_layers: int = 1,
    dropout: float = 0.1,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    threshold_policy: str = "validation_f1",
) -> dict[str, Any]:
    """Train GRU/LSTM binary head, checkpoint on validation BCE, then freeze test."""
    arch = str(architecture).lower()
    if arch not in {"gru", "lstm"}:
        raise ValueError("fit_future_red_model supports gru/lstm; use fit_future_red_readout for reservoirs")
    if threshold_policy != "validation_f1":
        raise ValueError("Only validation_f1 threshold selection is supported")
    _seed_everything(int(seed))
    xs = {part: torch.as_tensor(prepared[f"X_{part}"], dtype=torch.float32) for part in SPLITS}
    ys = {part: torch.as_tensor(prepared[f"y_{part}"], dtype=torch.float32) for part in SPLITS}
    known_masks = {part: (prepared[f"rows_{part}"].target_known.astype(bool).to_numpy()
                         & prepared[f"rows_{part}"].at_risk.astype(bool).to_numpy()
                         & np.isfinite(np.asarray(prepared[f"y_{part}"], dtype=float))) for part in SPLITS}
    train_ix, val_ix = np.flatnonzero(known_masks["train"]), np.flatnonzero(known_masks["validation"])
    if not len(train_ix):
        raise ValueError("No known at-risk training targets")
    if not len(val_ix):
        raise ValueError("No known at-risk validation targets")
    encoder = RecurrentEncoder(xs["train"].shape[-1], hidden_size, num_layers, arch)
    head = nn.Sequential(nn.Dropout(float(dropout)), nn.Linear(hidden_size, 1))
    model = nn.ModuleDict({"encoder": encoder, "head": head})
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    best_state = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    rng = np.random.default_rng(seed)
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        order = rng.permutation(train_ix)
        for start in range(0, len(order), max(1, int(batch_size))):
            ix = torch.as_tensor(order[start : start + batch_size], dtype=torch.long)
            logits = head(encoder(xs["train"][ix])).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, ys["train"][ix])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_logits = torch.cat([head(encoder(xs["validation"][i : i + batch_size])).squeeze(-1)
                                    for i in range(0, len(xs["validation"]), batch_size)])
            val_loss = float(F.binary_cross_entropy_with_logits(
                val_logits[val_ix], ys["validation"][val_ix]).item())
        if val_loss < best_loss - 1e-8:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= max(1, int(patience)):
                break
    if best_state is None:
        raise RuntimeError("Training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    probabilities = {}
    with torch.no_grad():
        for part in SPLITS:
            probabilities[part] = torch.cat([
                torch.sigmoid(head(encoder(xs[part][i : i + batch_size])).squeeze(-1))
                for i in range(0, len(xs[part]), batch_size)
            ]).numpy() if len(xs[part]) else np.empty(0)
    threshold, threshold_info = select_validation_threshold(ys["validation"].numpy()[val_ix],
                                                             probabilities["validation"][val_ix])
    model_contract = {
        "version": REPLAY_CONTRACT_VERSION,
        "dataset_id": dataset_id,
        "architecture": arch,
        "input_size": int(xs["train"].shape[-1]),
        "history_length": int(xs["train"].shape[1]),
        "hidden_size": int(hidden_size),
        "num_layers": int(num_layers),
        "dropout": float(dropout),
        "input_feature_names": prepared.get("feature_names", []),
        "feature_recipe": prepared.get("feature_recipe", "base_v1"),
        "feature_pipeline_version": prepared.get("feature_pipeline_version", FEATURE_PIPELINE_VERSION),
        "scaler": prepared.get("scaler", {}),
        "horizon_s": (prepared.get("target_manifest") or {}).get("horizon_s"),
        "target_artifact_id": (prepared.get("target_manifest") or {}).get("artifact_id"),
        "target_artifact_hash": (prepared.get("target_manifest") or {}).get("targets_sha256"),
        "state_policy": "causal fixed window ending at each origin; left-pad the available prefix; reset recurrent state per origin",
    }
    model_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model_contract": model_contract, "state_dict": model.state_dict(), "seed": int(seed)},
               model_dir / "model.pt")
    result = _write_outputs(prepared, probabilities, threshold, output_dir, {
        "dataset_id": dataset_id, "architecture": arch, "seed": int(seed), "best_epoch": best_epoch,
        "validation_bce": best_loss, "threshold_selection": threshold_info,
        "model_contract": model_contract,
        "target_manifest": prepared.get("target_manifest", {}),
        "scaler": prepared.get("scaler", {}), "provenance": prepared.get("provenance", {}),
    })
    result["artifact_hashes"] = {**result.get("artifact_hashes", {}),
                                 "model.pt": _file_sha256(model_dir / "model.pt")}
    (model_dir / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return result


def load_future_red_model(
    checkpoint_path: str | Path,
    *,
    manifest_path: str | Path | None = None,
    device: str = "cpu",
) -> tuple[nn.Module, dict[str, Any]]:
    """Load a saved GRU/LSTM checkpoint and verify its adjacent run manifest."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    contract = payload.get("model_contract") or {}
    if contract.get("version") != REPLAY_CONTRACT_VERSION:
        raise ValueError("Unsupported or missing future-red replay contract")
    manifest_file = Path(manifest_path) if manifest_path else checkpoint_path.parent / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != FUTURE_RED_MODEL_VERSION or manifest.get("model_contract") != contract:
        raise ValueError("Future-red model checkpoint does not match its manifest replay contract")
    expected_hash = (manifest.get("artifact_hashes") or {}).get("model.pt")
    if not expected_hash or _file_sha256(checkpoint_path) != expected_hash:
        raise ValueError("Future-red model checkpoint hash does not match its manifest")
    arch = str(contract.get("architecture", "")).lower()
    if arch not in {"gru", "lstm"}:
        raise ValueError(f"Replay helper does not load architecture {arch!r}")
    model = nn.ModuleDict({
        "encoder": RecurrentEncoder(int(contract["input_size"]), int(contract["hidden_size"]),
                                    int(contract["num_layers"]), arch),
        "head": nn.Sequential(nn.Dropout(float(contract["dropout"])),
                              nn.Linear(int(contract["hidden_size"]), 1)),
    }).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    model._future_red_contract = contract
    return model, contract


def predict_future_red_origins(
    model: nn.Module,
    prepared: dict[str, Any],
    *,
    split: str = "test",
    model_contract: dict[str, Any] | None = None,
    batch_size: int = 64,
    device: str = "cpu",
) -> np.ndarray:
    """Replay probabilities for a prepared split, preserving its origin order."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    model_contract = model_contract or getattr(model, "_future_red_contract", None)
    x = np.asarray(prepared[f"X_{split}"], dtype=np.float32)
    rows = prepared[f"rows_{split}"]
    if len(x) != len(rows):
        raise ValueError(f"{split} input windows and origin rows are misaligned")
    if model_contract:
        expected = (int(model_contract["history_length"]), int(model_contract["input_size"]))
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise ValueError(f"Prepared input shape {x.shape[1:]} does not match replay contract {expected}")
        names = prepared.get("feature_names")
        if names is not None and list(names) != list(model_contract.get("input_feature_names", [])):
            raise ValueError("Prepared input feature names do not match the replay contract")
    model = model.to(device)
    model.eval()
    if not len(x):
        return np.empty(0, dtype=np.float32)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), max(1, int(batch_size))):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            logits = model["head"](model["encoder"](batch)).squeeze(-1)
            outputs.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(outputs).astype(np.float32, copy=False)


def fit_future_red_readout(
    prepared: dict[str, Any], output_dir: str | Path, *, seed: int = 42,
    threshold_policy: str = "validation_f1", epochs: int = 100,
    learning_rate: float = 0.01, weight_decay: float = 1e-3,
) -> dict[str, Any]:
    """Fit a regularized logistic readout on fixed reservoir representations."""
    if threshold_policy != "validation_f1":
        raise ValueError("Only validation_f1 threshold selection is supported")
    _seed_everything(int(seed))
    arrays = {}
    known_masks = {}
    for part in SPLITS:
        x = np.asarray(prepared[f"X_{part}"], dtype=np.float32)
        arrays[part] = x.reshape(len(x), -1) if len(x) else np.empty((0, 0), dtype=np.float32)
        known_masks[part] = (prepared[f"rows_{part}"].target_known.astype(bool).to_numpy()
                             & prepared[f"rows_{part}"].at_risk.astype(bool).to_numpy()
                             & np.isfinite(np.asarray(prepared[f"y_{part}"], dtype=float)))
    train_ix, val_ix = np.flatnonzero(known_masks["train"]), np.flatnonzero(known_masks["validation"])
    if not len(train_ix):
        raise ValueError("No known at-risk training targets")
    if not len(val_ix):
        raise ValueError("No known at-risk validation targets")
    dim = arrays["train"].shape[1]
    model = nn.Linear(dim, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay))
    xtrain_all = torch.from_numpy(arrays["train"])
    ytrain_all = torch.as_tensor(prepared["y_train"], dtype=torch.float32)
    xval_all = torch.from_numpy(arrays["validation"])
    yval_all = torch.as_tensor(prepared["y_validation"], dtype=torch.float32)
    xtrain, ytrain = xtrain_all[train_ix], ytrain_all[train_ix]
    xv, yv = xval_all[val_ix], yval_all[val_ix]
    best, best_state, best_epoch, stale = float("inf"), None, 0, 0
    for epoch in range(1, max(1, int(epochs)) + 1):
        model.train()
        logits = model(xtrain).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, ytrain)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(F.binary_cross_entropy_with_logits(model(xv).squeeze(-1), yv).item())
        if val < best - 1e-8:
            best, best_epoch, stale = val, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 10:
                break
    if best_state is None:
        raise RuntimeError("Readout training failed to produce a checkpoint")
    model.load_state_dict(best_state)
    with torch.no_grad():
        probabilities = {part: torch.sigmoid(model(torch.from_numpy(arrays[part])).squeeze(-1)).numpy()
                         if len(arrays[part]) else np.empty(0) for part in SPLITS}
    threshold, threshold_info = select_validation_threshold(yv.numpy(), probabilities["validation"][val_ix])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    provenance = prepared.get("provenance", {})
    contract = dict(prepared.get("model_contract") or {
        "version": REPLAY_CONTRACT_VERSION,
        "architecture": prepared.get("architecture", "reservoir_readout"),
        "representation_size": int(dim),
        "state_policy": provenance.get("state_policy", "fixed representation supplied by adapter"),
        "history_length": prepared.get("history_length"),
        "horizon_s": (prepared.get("target_manifest") or {}).get("horizon_s"),
        "target_artifact_id": (prepared.get("target_manifest") or {}).get("artifact_id"),
    })
    torch.save({"model_contract": contract, "state_dict": model.state_dict(), "seed": int(seed)},
               out / "readout.pt")
    return _write_outputs(prepared, probabilities, threshold, output_dir, {
        "architecture": prepared.get("architecture", "reservoir_readout"), "seed": int(seed),
        "best_epoch": best_epoch, "validation_bce": best, "threshold_selection": threshold_info,
        "model_contract": contract,
        "target_manifest": prepared.get("target_manifest", {}),
        "scaler": prepared.get("scaler", {}), "provenance": prepared.get("provenance", {}),
    })


def train_future_red_model(
    dataset_id: str,
    architecture: str,
    *,
    targets_dir: str | Path | None = None,
    output_dir: str | Path,
    seed: int = 42,
    history_length: int = 20,
    data: dict[str, Any] | None = None,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Load official targets and fit a recurrent or matched reservoir classifier."""
    from pdm.future_red_targets import build_future_red_targets, load_future_red_targets

    data = data or load_processed(dataset_id)
    target_artifact = build_future_red_targets(dataset_id) if targets_dir is None else None
    artifact_dir = targets_dir or target_artifact["directory"]
    targets, target_manifest = load_future_red_targets(artifact_dir, data=data)
    model_features, feature_names = transform_future_red_features(dataset_id, data["features"], data["split"])
    cfg = load_dataset_config(dataset_id)
    prepared = prepare_future_red_sequences(model_features, targets, data["split"],
                                            history_length=history_length, feature_names=feature_names)
    prepared["feature_recipe"] = str(cfg.get("feature_recipe", "base_v1"))
    prepared["feature_pipeline_version"] = FEATURE_PIPELINE_VERSION
    prepared["target_manifest"] = target_manifest
    prepared["provenance"] = {"dataset_version": data.get("dataset_version"),
                              "dataset_fingerprint": data.get("fingerprint"),
                              "target_artifact_id": target_manifest.get("artifact_id"),
                              "target_artifact_hash": target_manifest.get("artifact_hash")}
    prepared["baseline_probabilities"] = build_future_red_baselines(
        dataset_id, data["features"], prepared,
        horizon_s=float(target_manifest["horizon_s"]),
        zone_policy=target_manifest.get("zone_policy", {}))
    prepared["baseline_definitions"] = {
        "always_no_entry": "Predict no RED entry at every origin.",
        "current_red_persistence": "Predict RED when the current causal sensor measurement already meets the saved RED comparator.",
        "trend_to_red": (f"Fit a straight line to the last {int(target_manifest.get('zone_policy', {}).get('trend_points', 5))} "
                         "observations since the latest gap; predict RED if it crosses the saved threshold within the target horizon."),
        "horizon_s": float(target_manifest["horizon_s"]),
        "threshold_policy": target_manifest.get("zone_policy", {}),
        "probability_encoding": "deterministic 0 or 1",
    }
    arch = str(architecture).lower()
    if arch in {"gru", "lstm"}:
        return fit_future_red_model(dataset_id, arch, prepared=prepared, output_dir=output_dir,
                                    seed=seed, **fit_kwargs)
    if arch not in {"fly_connectome_reservoir", "random_reservoir"}:
        raise ValueError(f"Unsupported future-red architecture: {architecture}")
    return train_future_red_reservoir(dataset_id, arch, prepared=prepared,
                                      output_dir=output_dir, seed=seed, **fit_kwargs)


def train_future_red_reservoir(
    dataset_id: str,
    architecture: str,
    *,
    prepared: dict[str, Any],
    output_dir: str | Path,
    seed: int = 42,
    n_nodes: int = 1000,
    leak: float = 0.2,
    spectral_radius: float = 0.9,
    input_scale: float = 0.1,
    source_path: str | Path | None = None,
    **readout_kwargs: Any,
) -> dict[str, Any]:
    """Fit a fixed Fly or degree-matched Random reservoir plus binary readout.

    ``prepare_run_graph`` can return a synthetic fixture when its requested real
    source is missing. This pathway rejects that fallback so provenance cannot
    imply biological topology that was not used.
    """
    from pdm.connectome.provenance import GRAPH_MODE_REAL
    from pdm.connectome.sampling import prepare_run_graph
    from pdm.models.fly_reservoir import FlyConnectomeReservoir
    from pdm.models.random_reservoir import RandomReservoir

    arch = str(architecture).lower()
    if arch not in {"fly_connectome_reservoir", "random_reservoir"}:
        raise ValueError("architecture must be fly_connectome_reservoir or random_reservoir")
    graph, parent_provenance, resolved_nodes = prepare_run_graph(
        architecture=arch, graph_mode=GRAPH_MODE_REAL, n_nodes=int(n_nodes),
        seed=int(seed), source_path=source_path)
    if (parent_provenance.get("graph_mode") != GRAPH_MODE_REAL
            or parent_provenance.get("is_synthetic")
            or int(resolved_nodes) != int(n_nodes)):
        raise RuntimeError("Real Fly connectome graph unavailable; refusing synthetic fallback")
    model_cls = FlyConnectomeReservoir if arch == "fly_connectome_reservoir" else RandomReservoir
    model_kwargs = {"input_size": int(prepared["X_train"].shape[-1]), "head": "rul",
                    "leak": float(leak), "spectral_radius": float(spectral_radius),
                    "input_scale": float(input_scale), "seed": int(seed),
                    "state_mode": "window_reset", "provenance": parent_provenance,
                    "n_nodes": int(n_nodes)}
    if arch == "random_reservoir":
        model_kwargs["parent_provenance"] = parent_provenance
    model = model_cls(graph, **model_kwargs)
    reps: dict[str, np.ndarray] = {}
    from scipy.sparse import csr_matrix

    recurrent = csr_matrix(model.W_res.cpu().numpy())
    w_in = model.W_in.cpu().numpy()
    bias = model.b_res.cpu().numpy()
    keep, alpha = 1.0 - float(leak), float(leak)
    for part in SPLITS:
        x = np.asarray(prepared[f"X_{part}"], dtype=np.float32)
        encoded = np.empty((len(x), int(n_nodes) + x.shape[-1]), dtype=np.float32)
        for start in range(0, len(x), 64):
            batch = x[start : start + 64]
            state = np.zeros((len(batch), int(n_nodes)), dtype=np.float32)
            for t in range(batch.shape[1]):
                drive = recurrent.dot(state.T).T + batch[:, t, :] @ w_in.T + bias
                state = keep * state + alpha * np.tanh(drive)
            encoded[start : start + len(batch)] = np.concatenate([state, batch[:, -1, :]], axis=1)
        reps[part] = encoded
    model_prepared = {**prepared, **{f"X_{part}": reps[part] for part in SPLITS},
                      "architecture": arch,
                      "provenance": {**prepared.get("provenance", {}),
                                     "graph_mode": model.graph_mode,
                                     "is_synthetic": model.is_synthetic,
                                     "graph_hash": model.graph_hash,
                                     "parent_graph_hash": model.parent_graph_hash,
                                     "graph_provenance": model.provenance,
                                     "n_nodes": model.n_nodes,
                                     "n_edges": graph.number_of_edges(),
                                     "reservoir": {"leak": float(leak),
                                                   "spectral_radius": float(spectral_radius),
                                                   "input_scale": float(input_scale),
                                                   "state_mode": "window_reset"}}}
    model_prepared["model_contract"] = {
        "version": REPLAY_CONTRACT_VERSION, "architecture": arch,
        "state_policy": "window_reset; zero state at each causal history window",
        "history_length": int(prepared["X_train"].shape[1]),
        "input_size": int(prepared["X_train"].shape[2]),
        "n_nodes": int(model.n_nodes), "graph_mode": model.graph_mode,
        "graph_hash": model.graph_hash, "parent_graph_hash": model.parent_graph_hash,
        "horizon_s": (prepared.get("target_manifest") or {}).get("horizon_s"),
        "target_artifact_id": (prepared.get("target_manifest") or {}).get("artifact_id"),
    }
    result = fit_future_red_readout(model_prepared, output_dir, seed=seed, **readout_kwargs)
    out = Path(output_dir)
    np.savez_compressed(out / "reservoir_weights.npz", W_in=model.W_in.cpu().numpy(),
                        W_res=model.W_res.cpu().numpy(), b_res=model.b_res.cpu().numpy())
    from pdm.connectome.graph import graph_to_payload

    (out / "graph.json").write_text(json.dumps(
        graph_to_payload(model.graph, node_order=model.node_order), sort_keys=True) + "\n", encoding="utf-8")
    if arch == "random_reservoir":
        (out / "parent_graph.json").write_text(json.dumps(
            graph_to_payload(graph, node_order=model.node_order), sort_keys=True) + "\n", encoding="utf-8")
    result["model_contract"] = model_prepared["model_contract"]
    result["artifact_hashes"] = {**result.get("artifact_hashes", {}), **{
        name: _file_sha256(out / name)
        for name in ("reservoir_weights.npz", "graph.json", "readout.pt")
    }}
    if arch == "random_reservoir":
        result["artifact_hashes"]["parent_graph.json"] = _file_sha256(out / "parent_graph.json")
    result["provenance"] = model_prepared["provenance"]
    (out / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
                                        encoding="utf-8")
    return result
