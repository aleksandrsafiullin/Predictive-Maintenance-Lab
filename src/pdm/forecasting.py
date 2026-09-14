"""Continuous fly-reservoir forecasting and empirical held-out uncertainty.

The recurrent kernel is shared with ordinary ESN inference. Calibration never sees
current-run outcomes; prediction consumes only timestamps and causal readouts.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from pdm.models.readout import ridge_design_matrix
from pdm.preprocessing import apply_preprocessor

PROFILE_VERSION = 1
DEFAULT_TAU_S = 300.0


def continuous_states(model, inputs, gap_before=None):
    """Shared-kernel full history, reset at explicit acquisition gaps only."""
    values = np.asarray(inputs, dtype=np.float32)
    if len(values) == 0:
        return np.zeros((0, model.n_nodes), dtype=np.float32)
    gaps = np.zeros(len(values), dtype=bool) if gap_before is None else np.asarray(gap_before, dtype=bool)
    if len(gaps) != len(values):
        raise ValueError("gap_before length mismatch")
    starts = np.unique(np.r_[0, np.flatnonzero(gaps)])
    frames = []
    with torch.no_grad():
        for start, end in zip(starts, np.r_[starts[1:], len(values)], strict=True):
            frames.append(model.forward_states(torch.from_numpy(values[start:end].copy())).cpu().numpy())
    return np.concatenate(frames)


def predict_failure_interval(timestamps_s, raw_rul_s, profile, *, gap_before=None):
    """Causal event-date filtering plus empirically calibrated failure interval.

    No actual event time or future measurement is accepted by this API. The
    interval is a held-out empirical range, not a guaranteed probability bound.
    """
    if int(profile.get("version", 0)) != PROFILE_VERSION:
        raise ValueError("Unsupported interval profile version")
    ts = np.asarray(timestamps_s, dtype=float)
    raw = np.asarray(raw_rul_s, dtype=float)
    if len(ts) != len(raw) or not np.isfinite(ts).all() or (len(ts) > 1 and np.any(np.diff(ts) <= 0)):
        raise ValueError("Expected matching, strictly increasing timestamp/readout arrays")
    gaps = np.zeros(len(ts), dtype=bool) if gap_before is None else np.asarray(gap_before, dtype=bool)
    if len(gaps) != len(ts):
        raise ValueError("gap_before length mismatch")
    warmup = int(profile.get("warmup_measurements", 20))
    tau = float(profile.get("smoothing_tau_s", DEFAULT_TAU_S))
    center = np.full(len(ts), np.nan)
    event = None
    n_seen = 0
    previous = None
    for i, (stamp, prediction) in enumerate(zip(ts, raw, strict=True)):
        if gaps[i]:
            event, previous, n_seen = None, None, 0
        n_seen += 1
        if n_seen < warmup or not np.isfinite(prediction):
            continue
        proposal = stamp + max(float(prediction), 0.0)
        if event is None:
            event = proposal
        else:
            gain = 1.0 - np.exp(-(stamp - previous) / max(tau, 1e-6))
            event += gain * (proposal - event)
        center[i] = max(event - stamp, 0.0)
        previous = stamp
    scale = residual_scale(center, profile)
    lower = np.maximum(0.0, center + float(profile.get("residual_q_low", 0.0)) * scale)
    upper = np.maximum(lower, center + float(profile.get("residual_q_high", 0.0)) * scale)
    return pd.DataFrame({"timestamp_s": ts, "raw_rul_s": raw, "predicted_rul_s": center,
                         "lower_rul_s": lower, "upper_rul_s": upper})


def residual_scale(predicted_rul_s, profile):
    params = profile.get("residual_scale") or {}
    intercept = float(params.get("intercept", np.log(60.0)))
    slope = float(params.get("slope", 0.0))
    values = np.asarray(predicted_rul_s, dtype=float)
    return np.maximum(60.0, np.exp(np.clip(intercept + slope * np.log1p(np.maximum(values, 0.0) / 60.0), -10, 20)))


def unit_weights(ids):
    """Each independent bearing has equal total weight regardless of duration."""
    ids = np.asarray(ids).astype(str)
    units, counts = np.unique(ids, return_counts=True)
    count = dict(zip(units, counts, strict=True))
    return np.asarray([1.0 / count[uid] for uid in ids], dtype=float)


def weighted_quantile(values, quantile, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(weights <= 0):
        raise ValueError("Need finite calibration residuals and positive weights")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    return float(values[order][min(np.searchsorted(cumulative, quantile * cumulative[-1]), len(values) - 1)])


def weighted_ridge(z, target, ids, alpha):
    """Equal-bearing mean squared error; intercept remains unpenalized."""
    z, y = np.asarray(z, float), np.asarray(target, float)
    weights = unit_weights(ids)
    weights /= weights.sum()
    gram = (z.T * weights) @ z
    penalty = np.eye(z.shape[1]) * float(alpha)
    penalty[-1, -1] = 0.0
    return np.linalg.solve(gram + penalty, z.T @ (weights * y))


def trajectory_design(model, prep, features, units, ids, warmup=20):
    """Full chronological state features, with row metadata for unitwise scoring."""
    selected = features[features.unit_id.astype(str).isin([str(x) for x in ids])]
    encoded = apply_preprocessor(prep, selected, "bearings")
    endpoints = units.set_index("unit_id")["event_time_s"]
    blocks, rows = [], []
    for uid, group in encoded.groupby("unit_id", sort=True):
        group = group.sort_values("timestamp_s")
        u = group[prep.feature_names].to_numpy(np.float32)
        gaps = group.get("gap_before", pd.Series(False, index=group.index)).fillna(False).to_numpy(bool)
        states = continuous_states(model, u, gaps)
        age = np.arange(len(group))
        start = 0
        eligible = np.zeros(len(group), bool)
        for i in age:
            if gaps[i]:
                start = i
            eligible[i] = i - start + 1 >= warmup
        ts = group.timestamp_s.to_numpy(float)
        if not eligible.any():
            continue
        blocks.append(ridge_design_matrix(states[eligible], u[eligible]))
        rows.append(pd.DataFrame({"unit_id": str(uid), "timestamp_s": ts[eligible],
                                  "target_rul_s": np.maximum(float(endpoints.loc[uid]) - ts[eligible], 0),
                                  "gap_before": np.r_[True, np.diff(np.flatnonzero(eligible)) > 1]}))
    if not blocks:
        raise ValueError("No eligible bearing histories")
    return np.concatenate(blocks), pd.concat(rows, ignore_index=True)


def encode_rul_targets(rul_s, time_scale_s, transform="linear", reference_s=60.0):
    values = np.maximum(np.asarray(rul_s, dtype=float), 0.0)
    if transform == "linear":
        return values / float(time_scale_s)
    if transform == "log1p":
        return np.log1p(values / float(reference_s))
    raise ValueError(f"Unsupported RUL transform: {transform}")


def decode_rul_outputs(raw, time_scale_s, transform="linear", reference_s=60.0):
    values = np.asarray(raw, dtype=float)
    if transform == "linear":
        return np.maximum(values, 0.0) * float(time_scale_s)
    if transform == "log1p":
        return np.expm1(np.clip(values, 0.0, 20.0)) * float(reference_s)
    raise ValueError(f"Unsupported RUL transform: {transform}")


def score_readout(z, rows, weights, time_scale_s, profile):
    """Apply exactly the runtime causal filter, separately for each bearing."""
    scored = rows.copy()
    if "n_nodes" in profile:
        # Match runtime F.linear grouping and float32 arithmetic exactly, including
        # the output transform. A float64 dot can hide readout cancellation.
        from torch.nn import functional as functional

        n = int(profile["n_nodes"])
        values = torch.as_tensor(np.asarray(z, dtype=np.float32))
        coeff = torch.as_tensor(np.asarray(weights, dtype=np.float32))
        raw = functional.linear(values[:, :n], coeff[:n].unsqueeze(0)) + functional.linear(
            values[:, n:-1], coeff[n:-1].unsqueeze(0), coeff[-1:])
        if profile.get("rul_transform", "linear") == "log1p":
            prediction = torch.expm1(torch.clamp(raw, min=0.0, max=20.0)) * float(profile.get("rul_reference_s", 60.0)) / time_scale_s
            prediction = prediction * time_scale_s
        else:
            prediction = functional.relu(raw) * time_scale_s
        scored["raw_rul_s"] = prediction.numpy().reshape(-1)
    else:
        scored["raw_rul_s"] = decode_rul_outputs(z @ weights, time_scale_s,
                                                profile.get("rul_transform", "linear"),
                                                profile.get("rul_reference_s", 60.0))
    pieces = []
    # Design already removed the warmup rows, so first scored row is ready.
    ready_profile = {**profile, "warmup_measurements": 1}
    for _, group in scored.groupby("unit_id", sort=False):
        estimates = predict_failure_interval(group.timestamp_s, group.raw_rul_s, ready_profile,
                                            gap_before=group.gap_before)
        group = group.copy()
        for column in ("predicted_rul_s", "lower_rul_s", "upper_rul_s"):
            group[column] = estimates[column].to_numpy()
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


def fit_interval_profile(oof_rows, calibration_rows, split, source_hashes=None):
    """Learn error scaling on train OOF only; calibrate residual quantiles on val."""
    train_ids = set(map(str, split["train"]))
    cal_ids = set(map(str, split["validation"]))
    test_ids = set(map(str, split["test"]))
    if train_ids & cal_ids or train_ids & test_ids or cal_ids & test_ids:
        raise ValueError("Train/calibration/test bearing leak")
    if not set(oof_rows.unit_id.astype(str)) <= train_ids or not set(calibration_rows.unit_id.astype(str)) <= cal_ids:
        raise ValueError("Residual rows violate split identity")
    x = np.log1p(np.maximum(oof_rows.predicted_rul_s.to_numpy(), 0) / 60.0)
    error = np.abs(oof_rows.target_rul_s.to_numpy() - oof_rows.predicted_rul_s.to_numpy())
    y = np.log(np.maximum(error, 60.0))
    w = unit_weights(oof_rows.unit_id)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    slope = float(np.clip(np.sum(w * (x-xm) * (y-ym)) / max(np.sum(w*(x-xm)**2), 1e-12), 0, 1))
    profile: dict[str, Any] = {
        "version": PROFILE_VERSION, "ready": True, "state_mode": "continuous",
        "warmup_measurements": 20, "smoothing_tau_s": DEFAULT_TAU_S,
        "method": "unit_weighted_empirical_signed_residuals",
        "label": "Empirical forecast interval", "nominal_empirical_mass": 0.90,
        "coverage_guarantee": False,
        "limitation": "Three independent calibration bearings; no distribution-free coverage guarantee.",
        "endpoint_definition": "last_recorded_sample",
        "residual_scale": {"intercept": float(ym-slope*xm), "slope": slope},
        "train_ids": sorted(train_ids), "calibration_ids": sorted(cal_ids), "test_ids": sorted(test_ids),
        "source_hashes": source_hashes or {},
        "n_calibration_bearings": int(calibration_rows.unit_id.nunique()),
        "n_calibration_measurements": len(calibration_rows),
    }
    residual = (calibration_rows.target_rul_s.to_numpy() - calibration_rows.predicted_rul_s.to_numpy()) / residual_scale(calibration_rows.predicted_rul_s, profile)
    cw = unit_weights(calibration_rows.unit_id)
    profile["residual_q_low"] = min(0.0, weighted_quantile(residual, 0.05, cw))
    profile["residual_q_high"] = max(0.0, weighted_quantile(residual, 0.95, cw))
    return profile


def forecast_metrics(rows):
    """Equal-bearing metrics, including actual interval coverage and stability."""
    records = []
    for uid, group in rows.groupby("unit_id"):
        truth = group.target_rul_s.to_numpy()
        center = group.predicted_rul_s.to_numpy()
        raw = group.raw_rul_s.to_numpy()
        lo, hi = group.lower_rul_s.to_numpy(), group.upper_rul_s.to_numpy()
        ts = group.timestamp_s.to_numpy()
        late = truth <= 1800
        records.append({"unit_id": str(uid), "mae_s": float(np.mean(np.abs(truth-center))),
                        "raw_mae_s": float(np.mean(np.abs(truth-raw))),
                        "coverage": float(np.mean((truth >= lo) & (truth <= hi))),
                        "mean_width_s": float(np.mean(hi-lo)),
                        "late_mae_s": float(np.mean(np.abs(truth[late]-center[late]))) if late.any() else None,
                        "late_width_s": float(np.mean((hi-lo)[late])) if late.any() else None,
                        "forecast_date_step_s": float(np.mean(np.abs(np.diff(ts+center)))) if len(ts)>1 else 0,
                        "raw_forecast_date_step_s": float(np.mean(np.abs(np.diff(ts+raw)))) if len(ts)>1 else 0})
    keys = [key for key in records[0] if key != "unit_id"]
    return {"per_bearing": records, "bearing_balanced": {key: float(np.mean([r[key] for r in records if r[key] is not None])) for key in keys}}


def load_interval_profile(run_dir):
    """Load only a profile bound to this exact checkpoint/data/preprocessing."""
    from pathlib import Path

    from pdm.io_util import read_json, sha256_file

    directory = Path(run_dir)
    profile = read_json(directory / "interval_profile.json")
    if int(profile.get("version", 0)) != PROFILE_VERSION or not profile.get("ready"):
        raise ValueError("Forecast interval profile is not ready or has an unsupported version")
    sources = profile.get("source_hashes") or {}
    for key, relative in (("checkpoint_sha256", "best.pt"),
                          ("preprocessing_sha256", "preprocessing.json"),
                          ("graph_sha256", "connectome/graph.json"),
                          ("dataset_fingerprint_sha256", "dataset_fingerprint.json")):
        expected = sources.get(key)
        file = directory / relative
        if not expected or not file.exists() or sha256_file(file) != expected:
            raise ValueError(f"Forecast interval profile mismatch: {relative}")
    split = read_json(directory / "split.json")
    groups = []
    for key, part in (("train_ids", "train"), ("calibration_ids", "validation"), ("test_ids", "test")):
        ids = set(map(str, profile.get(key) or []))
        if ids != set(map(str, split.get(part) or [])):
            raise ValueError(f"Forecast interval profile mismatch: {part} identities")
        groups.append(ids)
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Forecast interval profile contains split leakage")
    return profile
