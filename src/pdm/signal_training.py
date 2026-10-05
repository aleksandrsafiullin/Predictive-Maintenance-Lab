"""Project-scoped, causal multi-horizon numeric signal training."""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from torch.nn import functional as F

from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.models.signal_full_cns import (
    ENGINE_LABEL,
    build_signal_full_cns,
    source_unavailable_reason,
)
from pdm.models.signal_recurrent import SignalRecurrent
from pdm.projects import project_store

ENGINES = ("gru", "lstm", "quantile_boosting", "full_cns")
SCHEMA_VERSION = "project_signal_forecast_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _params(engine_id: str, params: dict | None, features: pd.DataFrame) -> dict:
    if (params or {}).get("forecast_mode") == "bounded_trend_corridor":
        from pdm.trend_corridor import corridor_params
        return corridor_params(engine_id, params, features)
    if (params or {}).get("forecast_mode") == "learned_joint_trajectories":
        from pdm.learned_trajectory import learned_params
        return learned_params(engine_id, params, features)
    if engine_id not in ENGINES:
        raise ValueError(f"Unsupported signal engine: {engine_id}")
    raw = dict(params or {})
    allowed = {"history_length", "horizons_s", "epochs", "hidden_size", "batch_size", "seed", "max_iter", "learning_rate", "residual_forecast", "forecast_mode", "nominal_coverage", "path_samples", "cv_folds", "max_windows_per_unit"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown signal training parameters: {sorted(unknown)}")
    cadence = []
    for _, group in features.groupby("unit_id"):
        dt = np.diff(np.sort(group.timestamp_s.to_numpy(float)))
        cadence.extend(dt[np.isfinite(dt) & (dt > 0)].tolist())
    if not cadence:
        raise ValueError("Signal histories require increasing timestamps")
    step = float(np.median(cadence))
    horizons = [float(x) for x in raw.get("horizons_s", [step, 2 * step, 3 * step])]
    if not horizons or len(horizons) > 24 or any(not np.isfinite(x) or x <= 0 for x in horizons):
        raise ValueError("Horizon grid must contain 1–24 positive finite seconds")
    if len(set(horizons)) != len(horizons) or horizons != sorted(horizons):
        raise ValueError("Horizons must be unique and increasing")
    config = {
        "history_length": int(raw.get("history_length", 8)),
        "horizons_s": horizons,
        "epochs": int(raw.get("epochs", 12)),
        "hidden_size": int(raw.get("hidden_size", 32)),
        "batch_size": int(raw.get("batch_size", 32)),
        "seed": int(raw.get("seed", 42)),
        "max_iter": int(raw.get("max_iter", 40)),
        "learning_rate": float(raw.get("learning_rate", 0.001)),
        "residual_forecast": raw.get("residual_forecast", engine_id in {"gru", "lstm"}),
        "target_tolerance_s": max(0.000001, min(step * 0.01, 0.01)),
    }
    if not (2 <= config["history_length"] <= 256 and 1 <= config["epochs"] <= 500
            and 4 <= config["hidden_size"] <= 512 and 1 <= config["batch_size"] <= 4096
            and 2 <= config["max_iter"] <= 1000 and 0 < config["learning_rate"] <= 0.1):
        raise ValueError("Signal model parameters are outside supported ranges")
    if not isinstance(config["residual_forecast"], bool):
        raise ValueError("Residual forecast must be true or false")
    mode = raw.get("forecast_mode", "legacy")
    if mode not in {"legacy", "joint_residual_paths"}:
        raise ValueError("Unsupported forecast_mode")
    if mode == "joint_residual_paths":
        config.update(forecast_mode=mode, nominal_coverage=float(raw.get("nominal_coverage", 0.9)),
                      path_samples=int(raw.get("path_samples", 256)), cv_folds=int(raw.get("cv_folds", 3)),
                      max_windows_per_unit=raw.get("max_windows_per_unit"))
        cap = config["max_windows_per_unit"]
        if (not 0 < config["nominal_coverage"] < 1 or not 2 <= config["path_samples"] <= 10000
                or not 2 <= config["cv_folds"] <= 100
                or (cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0))):
            raise ValueError("Joint funnel parameters are outside supported ranges")
        return config
    if set(raw) & {"nominal_coverage", "path_samples", "cv_folds", "max_windows_per_unit"}:
        raise ValueError("Joint uncertainty parameters require joint_residual_paths mode")
    if engine_id == "full_cns":
        return {key: config[key] for key in ("history_length", "horizons_s", "seed", "target_tolerance_s")}
    return config


def _validate_snapshot(data: dict) -> None:
    required = {"unit_id", "timestamp_s", "signal", "gap_before"}
    if not required.issubset(data["features"].columns):
        raise ValueError("Prepared snapshot lacks canonical signal columns")
    schema = data["schema"]
    if schema.get("time_unit") != "s" or schema.get("input_columns") != ["signal"]:
        raise ValueError("Signal v1 requires seconds and an allowlisted signal-only input")
    values = data["features"]["signal"].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Admitted signal must be finite")
    groups = [set(map(str, data["split"].get(part, []))) for part in ("train", "validation", "test")]
    if any(not group for group in groups) or any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Train/Validation/Test require disjoint physical units")


def _segments(features: pd.DataFrame, unit_id: str) -> list[pd.DataFrame]:
    unit = features[features.unit_id.astype(str) == str(unit_id)].sort_values("timestamp_s")
    if unit.empty:
        return []
    times = unit.timestamp_s.to_numpy(float)
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"Invalid signal clock for unit {unit_id}")
    gap = unit.gap_before.fillna(False).to_numpy(bool).copy()
    gap[0] = True
    if "segment_id" in unit:
        sid = unit.segment_id.to_numpy()
        gap[1:] |= sid[1:] != sid[:-1]
    starts = np.flatnonzero(gap)
    return [unit.iloc[start:end].reset_index(drop=True) for start, end in zip(starts, [*starts[1:], len(unit)], strict=True)]


def average_training_duration_s(snapshot: dict) -> float:
    """Mean observed continuous duration of each Train unit, excluding gaps and Test."""
    durations = []
    for uid in snapshot["split"]["train"]:
        segments = _segments(snapshot["features"], str(uid))
        durations.append(sum(float(part.timestamp_s.iloc[-1] - part.timestamp_s.iloc[0])
                             for part in segments if len(part) > 1))
    return float(np.mean(durations)) if durations else 0.0


def _windows(features: pd.DataFrame, ids: list[str], params: dict) -> dict[str, Any]:
    history = params["history_length"]
    horizons = params["horizons_s"]
    tolerance = params["target_tolerance_s"]
    x_rows, y_rows, mask_rows, unit_rows, at_rows = [], [], [], [], []
    for uid in ids:
        for segment in _segments(features, str(uid)):
            ts = segment.timestamp_s.to_numpy(float)
            signal = segment.signal.to_numpy(float)
            targets = np.full((len(ts), len(horizons)), np.nan, dtype=np.float32)
            for j, horizon in enumerate(horizons):
                desired = ts + horizon
                left = np.searchsorted(ts, desired - tolerance, side="left")
                right = np.searchsorted(ts, desired + tolerance, side="right")
                matched = (right - left == 1) & (left < len(ts))
                candidates = np.flatnonzero(matched)
                matched[candidates] &= ts[left[candidates]] > ts[candidates]
                targets[matched, j] = signal[left[matched]]
            for end in range(history - 1, len(segment)):
                target = targets[end]
                mask = np.isfinite(target)
                if not mask.any() and not params.get("include_targetless", False):
                    continue
                x_rows.append(signal[end - history + 1:end + 1])
                y_rows.append(np.nan_to_num(target, nan=0.0))
                mask_rows.append(mask)
                unit_rows.append(str(uid))
                at_rows.append(float(ts[end]))
    if not x_rows:
        return {"x": np.empty((0, history, 1), np.float32), "y": np.empty((0, len(horizons)), np.float32),
                "mask": np.empty((0, len(horizons)), bool), "unit_id": [], "as_of_s": []}
    return {"x": np.asarray(x_rows, np.float32)[:, :, None], "y": np.asarray(y_rows, np.float32),
            "mask": np.asarray(mask_rows, bool), "unit_id": unit_rows, "as_of_s": at_rows}


def _weights(ids: list[str]) -> np.ndarray:
    counts = pd.Series(ids).value_counts()
    values = np.asarray([1.0 / counts[uid] for uid in ids], np.float32)
    return values / values.mean()


def _metrics(pred: np.ndarray, data: dict, horizons: list[float]) -> dict:
    mask = data["mask"]
    target = data["y"]
    rows = []
    for j, h in enumerate(horizons):
        active = mask[:, j]
        if active.any():
            error = pred[active, j] - target[active, j]
            rows.append({"horizon_s": h, "known_targets": int(active.sum()),
                         "independent_units": len(set(np.asarray(data["unit_id"])[active])),
                         "mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
                         "persistence_mae": float(np.mean(np.abs(data["x"][active, -1, 0] - target[active, j])))})
        else:
            rows.append({"horizon_s": h, "known_targets": 0, "independent_units": 0,
                         "mae": None, "rmse": None, "persistence_mae": None})
    per_unit = []
    uid_array = np.asarray(data["unit_id"])
    for uid in sorted(set(data["unit_id"])):
        for j, h in enumerate(horizons):
            active = (uid_array == uid) & mask[:, j]
            if active.any():
                per_unit.append({"unit_id": uid, "horizon_s": h, "known_targets": int(active.sum()),
                                 "mae": float(np.mean(np.abs(pred[active, j] - target[active, j])))})
    all_active = mask
    per_unit_mae = []
    for uid in sorted(set(data["unit_id"])):
        active = (uid_array == uid)[:, None] & mask
        if active.any():
            per_unit_mae.append(float(np.mean(np.abs(pred[active] - target[active]))))
    return {"known_targets": int(all_active.sum()),
            "mae": float(np.mean(per_unit_mae)) if per_unit_mae else None,
            "row_mae": float(np.mean(np.abs(pred[all_active] - target[all_active]))) if all_active.any() else None,
            "by_horizon": rows, "by_unit": per_unit}


def _predict(model: Any, engine_id: str, data: dict, scaler: dict, *, should_stop=None) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if not len(data["x"]):
        shape = data["y"].shape
        return np.empty(shape), None, None
    x = (data["x"] - scaler["mean"]) / scaler["std"]
    if engine_id == "full_cns":
        point = model.predict(x, should_stop=should_stop) * scaler["std"] + scaler["mean"]
        if scaler.get("output_domain") == "nonnegative":
            point = np.maximum(point, 0.0)
        return point, None, None
    if engine_id in {"gru", "lstm"}:
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(x.astype(np.float32))).numpy()
        point = pred * scaler["std"] + scaler["mean"]
        if scaler.get("output_domain") == "nonnegative":
            point = np.maximum(point, 0.0)
        return point, None, None
    flat = x.reshape(len(x), -1)
    pred = np.column_stack([entry[0.5].predict(flat) for entry in model])
    lower = np.column_stack([entry[0.05].predict(flat) for entry in model])
    upper = np.column_stack([entry[0.95].predict(flat) for entry in model])
    band = np.sort(np.stack([lower, pred, upper], axis=-1), axis=-1)
    if scaler.get("output_domain") == "nonnegative":
        band = np.maximum(band, 0.0)
    return band[:, :, 1], band[:, :, 0], band[:, :, 2]


def available_signal_engines(project_id: str, snapshot_id: str | None = None) -> list[dict]:
    data = load_snapshot(project_id, snapshot_id)
    _validate_snapshot(data)
    split = data["split"]
    train_features = data["features"][data["features"].unit_id.astype(str).isin(set(map(str, split["train"])))].copy()
    reason = None
    try:
        defaults = _params("gru", {"history_length": 2}, train_features)
        defaults["horizons_s"] = defaults["horizons_s"][:1]
        train = _windows(data["features"], split["train"], defaults)
        validation = _windows(data["features"], split["validation"], defaults)
        if (not len(train["x"]) or not len(validation["x"]) or
                not np.all(train["mask"].any(axis=0)) or not np.all(validation["mask"].any(axis=0))):
            reason = "Train and Validation need known next-step signal targets from continuous histories."
    except ValueError as exc:
        reason = str(exc)
    base = [
        {"engine_id": "gru", "label": "GRU", "available": reason is None, "reason": reason,
         "params": ["history_length", "horizons_s", "epochs", "hidden_size", "batch_size", "seed"]},
        {"engine_id": "lstm", "label": "LSTM", "available": reason is None, "reason": reason,
         "params": ["history_length", "horizons_s", "epochs", "hidden_size", "batch_size", "seed"]},
        {"engine_id": "quantile_boosting", "label": "Quantile boosting", "available": reason is None, "reason": reason,
         "params": ["history_length", "horizons_s", "max_iter", "seed"]},
    ]
    full_cns_reason = reason or source_unavailable_reason()
    base.append({"engine_id": "full_cns", "label": ENGINE_LABEL,
                 "available": full_cns_reason is None, "reason": full_cns_reason,
                 "params": ["history_length", "horizons_s", "seed"]})
    for engine_id, label in (("fly", "Fly reservoir (sampled)"), ("random", "Random reservoir")):
        base.append({"engine_id": engine_id, "label": label, "available": False,
                     "reason": "This research engine has no validated numeric signal head for project replay.",
                     "params": []})
    return base


def _fit_full_cns(train, validation, params, scaler, should_stop, status_cb):
    if should_stop():
        raise InterruptedError("Full MaleCNS signal training cancelled")
    status_cb({"stage": "preparing", "message": "Verifying and loading the full MaleCNS connectome"})
    model = build_signal_full_cns(params["seed"])
    designs = []
    frames = [("Train", train)] + ([("Validation", validation)] if validation is not None else [])
    for label, frame in frames:
        x = (frame["x"] - scaler["mean"]) / scaler["std"]
        def progress(done, total, label=label):
            status_cb({"stage": "training", "message": f"Full MaleCNS · {label} windows {done}/{total}"})
        designs.append(model.transform(x, should_stop=should_stop, status_cb=progress))
    train_design = designs[0]
    val_design = designs[1] if validation is not None else None
    model.design_mean = train_design.mean(axis=0)
    model.design_std = np.maximum(train_design.std(axis=0), 1e-6)
    standardized = (train_design - model.design_mean) / model.design_std
    y = (train["y"] - scaler["mean"]) / scaler["std"]
    best_heads, best_pred, best_mae, scores = None, None, float("inf"), []
    for alpha in ((1.0,) if validation is None else (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)):
        heads = []
        for j in range(len(params["horizons_s"])):
            if should_stop():
                raise InterruptedError("Full MaleCNS signal training cancelled")
            active = train["mask"][:, j]
            weights = _weights(np.asarray(train["unit_id"])[active].tolist())
            head = Ridge(alpha=alpha, solver="cholesky")
            head.fit(standardized[active], y[active, j], sample_weight=weights)
            heads.append(head)
        model.heads = heads
        if validation is None:
            return model, {"criterion": "predeclared_fixed_configuration", "selected_ridge_alpha": 1.0}, None
        pred = model.predict_design(val_design) * scaler["std"] + scaler["mean"]
        if scaler.get("output_domain") == "nonnegative":
            pred = np.maximum(pred, 0.0)
        metric = _metrics(pred, validation, params["horizons_s"])["mae"]
        if metric is None or not np.isfinite(metric):
            raise ValueError("Validation has no finite Full MaleCNS signal score")
        scores.append({"ridge_alpha": alpha, "unit_equal_mae": metric})
        if metric < best_mae:
            best_heads, best_pred, best_mae = heads, pred.copy(), metric
    model.heads = best_heads
    selection = {"criterion": "validation_unit_equal_mae", "candidates": scores,
                 "selected_ridge_alpha": next(row["ridge_alpha"] for row in scores if row["unit_equal_mae"] == best_mae)}
    return model, selection, best_pred


def _fit_recurrent(engine_id: str, train: dict, validation: dict, params: dict, scaler: dict,
                   should_stop: Callable[[], bool], status_cb: Callable[[dict], None]) -> SignalRecurrent:
    torch.manual_seed(params["seed"])
    model = SignalRecurrent(engine_id, 1, params["hidden_size"], len(params["horizons_s"]),
                            residual_forecast=params.get("residual_forecast", False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["learning_rate"])
    x_train = torch.from_numpy(((train["x"] - scaler["mean"]) / scaler["std"]).astype(np.float32))
    y_train = torch.from_numpy(((train["y"] - scaler["mean"]) / scaler["std"]).astype(np.float32))
    m_train = torch.from_numpy(train["mask"])
    w_train = torch.from_numpy(_weights(train["unit_id"]))
    rng = np.random.default_rng(params["seed"])
    best_state, best_val = None, float("inf")
    for epoch in range(params["epochs"]):
        if should_stop():
            raise InterruptedError("Signal training cancelled")
        model.train()
        for indices in np.array_split(rng.permutation(len(x_train)), max(1, int(np.ceil(len(x_train) / params["batch_size"])))):
            if should_stop():
                raise InterruptedError("Signal training cancelled")
            ix = torch.from_numpy(indices)
            prediction = model(x_train[ix])
            active = m_train[ix]
            error = F.mse_loss(prediction, y_train[ix], reduction="none")
            weighted = error * active * w_train[ix, None]
            loss = weighted.sum() / (active * w_train[ix, None]).sum().clamp_min(1e-6)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite signal loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if validation is None:
            status_cb({"stage": "training", "progress": (epoch + 1) / params["epochs"],
                       "message": f"Fixed epoch {epoch + 1}/{params['epochs']} (Train only)"})
            continue
        pred, _, _ = _predict(model, engine_id, validation, scaler)
        metric = _metrics(pred, validation, params["horizons_s"])["mae"]
        if metric is None:
            raise ValueError("Validation has no known signal targets")
        if metric < best_val:
            best_val, best_state = metric, copy.deepcopy(model.state_dict())
        status_cb({"stage": "training", "progress": (epoch + 1) / params["epochs"],
                   "message": f"Epoch {epoch + 1}/{params['epochs']} · validation MAE {metric:.4g}"})
    if validation is None:
        return model
    if best_state is None:
        raise RuntimeError("No selected signal checkpoint")
    model.load_state_dict(best_state)
    return model


def _fit_boosting(train: dict, validation: dict, params: dict, scaler: dict,
                  should_stop: Callable[[], bool], status_cb: Callable[[dict], None]) -> tuple[list[dict], dict]:
    x = ((train["x"] - scaler["mean"]) / scaler["std"]).reshape(len(train["x"]), -1)
    weights = _weights(train["unit_id"])
    candidates = ([params["max_iter"]] if validation is None else
                  sorted({max(2, params["max_iter"] // 2), params["max_iter"]}))
    best_models, best_mae, scores = None, float("inf"), []
    total_fits = len(candidates) * len(params["horizons_s"]) * 3
    completed = 0
    for iterations in candidates:
        models = []
        for j, h in enumerate(params["horizons_s"]):
            mask = train["mask"][:, j]
            if not mask.any():
                raise ValueError(f"No train targets at {h:g} s")
            by_quantile = {}
            for q in (0.05, 0.5, 0.95):
                if should_stop():
                    raise InterruptedError("Signal training cancelled")
                model = HistGradientBoostingRegressor(
                    loss="quantile", quantile=q, max_iter=iterations, max_leaf_nodes=15,
                    min_samples_leaf=max(2, min(20, int(mask.sum()) // 5)), early_stopping=False,
                    random_state=params["seed"],
                )
                model.fit(x[mask], train["y"][mask, j], sample_weight=weights[mask])
                by_quantile[q] = model
                completed += 1
                status_cb({"stage": "training", "progress": completed / total_fits,
                           "message": f"Fitting {h:g} s, quantile {q:g}, {iterations} iterations"})
            models.append(by_quantile)
        if validation is None:
            return models, {"criterion": "predeclared_fixed_configuration", "selected_max_iter": params["max_iter"]}
        point, _, _ = _predict(models, "quantile_boosting", validation, scaler)
        metric = _metrics(point, validation, params["horizons_s"])["mae"]
        if metric is None or not np.isfinite(metric):
            raise ValueError("Validation has no known signal targets")
        scores.append({"max_iter": iterations, "unit_equal_mae": metric})
        if metric < best_mae:
            best_models, best_mae = models, metric
    assert best_models is not None
    return best_models, {"criterion": "validation_unit_equal_mae", "candidates": scores,
                         "selected_max_iter": next(row["max_iter"] for row in scores if row["unit_equal_mae"] == best_mae)}


def _physical_map(data: dict) -> dict[str, str]:
    """Resolve cycle/fragment IDs to equipment IDs and reject partition leakage."""
    frame = data.get("units", data["features"])
    column = "physical_unit_id" if "physical_unit_id" in frame else "unit_id"
    mapping = {}
    for uid, rows in frame.groupby("unit_id"):
        groups = rows[column].dropna().astype(str).unique()
        if len(groups) != 1:
            raise ValueError(f"Unit {uid} needs exactly one physical equipment identity")
        mapping[str(uid)] = str(groups[0])
    partitions = []
    for part in ("train", "validation", "test"):
        ids = list(map(str, data["split"][part]))
        if any(uid not in mapping for uid in ids):
            raise ValueError("Snapshot split has an unknown physical unit")
        partitions.append({mapping[uid] for uid in ids})
    if any(partitions[a] & partitions[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Physical equipment groups overlap Train/Validation/Test")
    return mapping


def _joint_windows(features, ids, config, physical):
    frame = _windows(features, ids, config)
    groups = np.asarray([physical[uid] for uid in frame["unit_id"]], str)
    keep, counts = [], {}
    for group in sorted(set(groups)):
        rows = np.flatnonzero(groups == group)
        # Uniform origin-index sampling per physical equipment, including all its cycles.
        cap = config.get("max_windows_per_unit")
        chosen = rows if cap is None or len(rows) <= cap else rows[np.linspace(0, len(rows)-1, cap, dtype=int)]
        keep.extend(chosen.tolist())
        counts[group] = {"eligible": int(len(rows)), "retained": int(len(chosen))}
    keep = np.asarray(sorted(keep), int)
    result = {key: frame[key][keep] for key in ("x", "y", "mask")}
    result.update({key: [frame[key][i] for i in keep] for key in ("unit_id", "as_of_s")})
    result.update(physical_unit_id=groups[keep].tolist(), window_counts=counts)
    # Match replay's conservative scope gate: one recording per physical group,
    # first causal segment, exactly the first complete input history. Target
    # availability is irrelevant to this identity/time check.
    first_origin = {}
    group_units = {}
    for uid, group in physical.items():
        group_units.setdefault(group, set()).add(uid)
    for uid in map(str, ids):
        segments = _segments(features, uid)
        if len(group_units.get(physical[uid], ())) == 1 and segments and len(segments[0]) >= config["history_length"]:
            first_origin[uid] = float(segments[0].timestamp_s.iloc[config["history_length"]-1])
    result["origin_within_calibration_scope"] = [uid in first_origin and at == first_origin[uid]
                                                 for uid, at in zip(result["unit_id"], result["as_of_s"])]
    return result


def _fit_scaler(features, ids, physical, output_domain):
    values = features.loc[features.unit_id.astype(str).isin(set(map(str, ids))), "signal"].to_numpy(float)
    return {"mean": float(np.mean(values)), "std": float(max(np.std(values), 1e-6)),
            "fit_units": sorted(map(str, ids)), "fit_physical_groups": sorted({physical[str(uid)] for uid in ids}),
            "output_domain": output_domain}


def _fit_fixed(engine_id, train, config, scaler, stop, report):
    if not len(train["x"]) or not np.all(train["mask"].any(axis=0)):
        raise ValueError("Every fixed-fit Train fold needs known targets at every frozen horizon")
    train = {**train, "unit_id": train.get("physical_unit_id", train["unit_id"])}
    # Fixed configuration is declared before any fits. Validation/Test never select it.
    if engine_id == "full_cns":
        return _fit_full_cns(train, None, config, scaler, stop, report)[0]
    if engine_id in {"gru", "lstm"}:
        return _fit_recurrent(engine_id, train, None, config, scaler, stop, report)
    return _fit_boosting(train, None, config, scaler, stop, report)[0]


def _joint_oof_allocation(features, train_ids, config, physical):
    """Balance physical groups using Train observation clocks only."""
    groups = sorted({physical[str(uid)] for uid in train_ids})
    folds = min(config["cv_folds"], len(groups))
    if folds < 2:
        raise ValueError("Grouped OOF uncertainty needs at least two physical Train groups")
    # Remove signals and outcomes before inspecting continuous clock geometry.
    columns = [name for name in ("unit_id", "timestamp_s", "gap_before", "segment_id") if name in features]
    clocks = features.loc[features.unit_id.astype(str).isin(set(map(str, train_ids))), columns]
    geometry = {group: {"units": [], "max_continuous_duration_s": 0.0} for group in groups}
    for uid in sorted(set(map(str, train_ids))):
        segments = _segments(clocks, uid)
        records = [{"start_s": float(part.timestamp_s.iloc[0]),
                    "end_s": float(part.timestamp_s.iloc[-1]),
                    "duration_s": float(part.timestamp_s.iloc[-1] - part.timestamp_s.iloc[0]),
                    "observation_count": len(part)} for part in segments]
        group = geometry[physical[uid]]
        group["units"].append({"unit_id": uid, "continuous_segments": records})
        group["max_continuous_duration_s"] = max(
            group["max_continuous_duration_s"], max((row["duration_s"] for row in records), default=0.0))
    tie_order = np.random.default_rng(config["seed"]).permutation(groups).tolist()
    tie_rank = {group: rank for rank, group in enumerate(tie_order)}
    order = sorted(groups, key=lambda group: (-geometry[group]["max_continuous_duration_s"], tie_rank[group]))
    held_groups = [order[number::folds] for number in range(folds)]
    metadata = {"policy": "train_max_continuous_duration_descending_seeded_ties_round_robin",
                "seed": config["seed"], "clock_columns": columns, "group_clock_geometry": geometry,
                "ordered_physical_groups": order, "held_physical_groups_by_fold": held_groups}
    return held_groups, metadata


def _joint_fit(data, engine_id, config, physical, stop, report):
    from pdm.signal_funnel import build_residual_bank

    features, split = data["features"], data["split"]
    held_folds, allocation = _joint_oof_allocation(features, split["train"], config, physical)
    folds = len(held_folds)
    residuals, masks, bank_groups, fold_records = [], [], [], []
    for number, held in enumerate(held_folds, 1):
        if stop():
            raise InterruptedError("Signal grouped OOF training cancelled")
        held_groups = set(held)
        fit_ids = [str(uid) for uid in split["train"] if physical[str(uid)] not in held_groups]
        held_ids = [str(uid) for uid in split["train"] if physical[str(uid)] in held_groups]
        scaler = _fit_scaler(features, fit_ids, physical, data["schema"].get("output_domain", "real"))
        train = _joint_windows(features, fit_ids, config, physical)
        held_frame = _joint_windows(features, held_ids, config, physical)
        def fold_report(payload, number=number):
            report({**payload, "message": f"OOF {number}/{folds} · {payload.get('message', '')}"})
        if not len(train["x"]) or not np.all(train["mask"].any(axis=0)):
            fold_records.append({"fold": number, "status": "insufficient_fit_horizon_support",
                                 "fit_units": fit_ids, "held_units": held_ids,
                                 "fit_physical_groups": scaler["fit_physical_groups"],
                                 "held_physical_groups": sorted(held_groups), "scaler": scaler,
                                 "fit_target_counts_by_horizon": train["mask"].sum(axis=0).tolist(),
                                 "excluded_held_origin_count": len(held_frame["x"])})
            report({"stage": "training", "message": f"OOF {number}/{folds} lacks frozen horizon support; excluded from residual bank"})
            continue
        model = _fit_fixed(engine_id, train, config, scaler, stop, fold_report)
        pred = _predict(model, engine_id, held_frame, scaler, should_stop=stop)[0]
        residuals.append(held_frame["y"] - pred)
        masks.append(held_frame["mask"])
        bank_groups.extend(held_frame["physical_unit_id"])
        fold_records.append({"fold": number, "fit_units": fit_ids, "held_units": held_ids,
                             "fit_physical_groups": scaler["fit_physical_groups"],
                             "held_physical_groups": sorted(held_groups), "scaler": scaler,
                             "fit_window_counts": train["window_counts"], "held_window_counts": held_frame["window_counts"],
                             "held_origin_count": len(pred), "complete_held_origin_count": int(held_frame["mask"].all(axis=1).sum())})
        del model
    shape = (0, len(config["horizons_s"]))
    bank = build_residual_bank(np.concatenate(residuals) if residuals else np.empty(shape),
                               np.concatenate(masks) if masks else np.empty(shape, bool), bank_groups, config["horizons_s"])
    bank["excluded_unfittable_fold_origin_count"] = sum(row.get("excluded_held_origin_count", 0) for row in fold_records)
    all_masks = np.concatenate(masks) if masks else np.empty(shape, bool)
    all_groups = np.asarray(bank_groups)
    bank["observed_oof_origin_counts_by_horizon"] = all_masks.sum(axis=0).tolist()
    bank["observed_oof_physical_group_counts_by_horizon"] = [len(set(all_groups[all_masks[:, j]])) for j in range(shape[1])]
    train = _joint_windows(features, split["train"], config, physical)
    scaler = _fit_scaler(features, split["train"], physical, data["schema"].get("output_domain", "real"))
    report({"stage": "training", "message": "Final fixed configuration fit on all Train physical groups"})
    model = _fit_fixed(engine_id, train, config, scaler, stop, report)
    selection = {"criterion": "predeclared_fixed_configuration", "validation_role": "calibration_only",
                 "test_role": "frozen_evaluation_only", "epochs": config["epochs"], "max_iter": config["max_iter"],
                 "ridge_alpha": 1.0, "cv_folds_requested": config["cv_folds"], "cv_folds_actual": folds,
                 "folds": fold_records, "oof_allocation": allocation,
                 "window_policy": "uniform_origin_index_per_physical_group",
                 "train_window_counts": train["window_counts"],
                 "configuration_note": "No tuning or early stopping; fixed final epoch/iterations/ridge declared before fitting"}
    return model, scaler, selection, bank


def _joint_paths(bank, pred, frame, config, scaler, stop):
    from pdm.signal_funnel import sample_paths
    if bank["status"] != "available":
        return None
    paths = []
    for i, point in enumerate(pred):
        if stop():
            raise InterruptedError("Signal joint path evaluation cancelled")
        paths.append(sample_paths(point, float(frame["x"][i, -1, 0]), bank,
                                  n_samples=config["path_samples"], seed=config["seed"],
                                  nonnegative=scaler.get("output_domain") == "nonnegative"))
    return np.asarray(paths) if paths else np.empty((0, config["path_samples"], len(config["horizons_s"]) + 1))


def _joint_targets(frame):
    return (np.column_stack((frame["x"][:, -1, 0], frame["y"])),
            np.column_stack((np.ones(len(frame["x"]), bool), frame["mask"])))


def _joint_metrics(paths, frame, config, scaler, calibration):
    from pdm.signal_funnel import apply_calibration, evaluate_paths
    if paths is None or not len(paths):
        return {"status": "insufficient_support", "physical_group_count": len(set(frame["physical_unit_id"])),
                "complete_path_count": int(frame["mask"].all(axis=1).sum()), "unknown_path_count": int((~frame["mask"].all(axis=1)).sum())}
    actual, mask = _joint_targets(frame)
    metrics = evaluate_paths(paths, actual, mask, frame["physical_unit_id"], coverage=config["nominal_coverage"],
                             scale=np.full(actual.shape[1], scaler["std"]))
    complete_groups = np.asarray(frame["physical_unit_id"])[mask.all(axis=1)]
    metrics.update(complete_physical_group_count=len(set(complete_groups)),
                   observed_physical_group_counts_by_horizon=[len(set(np.asarray(frame["physical_unit_id"])[mask[:, j]])) for j in range(mask.shape[1])],
                   status="raw_empirical_unvalidated", saved_calibration_status=calibration["status"],
                   nominal_coverage=config["nominal_coverage"],
                   horizon_grid_s=[0.0, *config["horizons_s"]], interval_kind="raw_empirical_pointwise_band_from_joint_samples")
    lower, upper = np.quantile(paths, [(1-config["nominal_coverage"])/2, (1+config["nominal_coverage"])/2], axis=1)
    origin_scope = np.asarray(frame.get("origin_within_calibration_scope", [False]*len(paths)), bool)
    origin_scope &= calibration.get("guarantee_scope") == "predeclared_earliest_origin_per_physical_group_only"
    calibrated_rows = origin_scope & (calibration["status"] == "calibrated")
    bounds = [apply_calibration(lo, hi, calibration if admitted else
                               {**calibration, "status": "outside_calibration_origin_scope"},
                               nonnegative=scaler.get("output_domain") == "nonnegative")
              for lo, hi, admitted in zip(lower, upper, origin_scope)]
    lower, upper = np.asarray([row[0] for row in bounds]), np.asarray([row[1] for row in bounds])
    valid = mask & np.isfinite(actual)
    complete = valid.all(axis=1)
    groups = np.asarray(frame["physical_unit_id"])
    def balanced(values, admitted):
        rows = [float(values[(groups == group) & admitted].mean()) for group in np.unique(groups)
                if np.any((groups == group) & admitted)]
        return float(np.mean(rows)) if rows else None
    inside = (actual >= lower) & (actual <= upper)
    alpha = 1-config["nominal_coverage"]
    score = upper-lower + 2/alpha*(np.maximum(lower-actual, 0)+np.maximum(actual-upper, 0))
    display_status = ("scope_aware_mixed_bands" if calibrated_rows.any() and not calibrated_rows.all() else
                      "calibrated_earliest_origin_only" if calibrated_rows.all() else "raw_empirical_unvalidated")
    metrics["display_band"] = {"status": display_status,
                               "guarantee_scope": calibration.get("guarantee_scope", "unspecified"),
                               "all_origin_coverage_guarantee": False,
                               "calibrated_origin_count": int(calibrated_rows.sum()),
                               "raw_origin_count": int((~calibrated_rows).sum()),
                               "whole_path_coverage": balanced(inside.all(axis=1), complete),
                               "point_coverage_by_horizon": [balanced(inside[:, j], valid[:, j]) for j in range(actual.shape[1])],
                               "interval_score_by_horizon": [balanced(score[:, j], valid[:, j]) for j in range(actual.shape[1])],
                               "band_width_by_horizon": [balanced((upper-lower)[:, j], valid[:, j]) for j in range(actual.shape[1])]}
    metrics["display_band"]["earliest_origin_diagnostics"] = {
        "status": calibration["status"], "origin_count": int(origin_scope.sum()),
        "complete_origin_count": int((origin_scope & complete).sum()),
        "physical_group_count": len(set(groups[origin_scope])),
        "whole_path_coverage": balanced(inside.all(axis=1), complete & origin_scope),
        "point_coverage_by_horizon": [balanced(inside[:, j], valid[:, j] & origin_scope) for j in range(actual.shape[1])]}
    return metrics


def train_signal_run(project_id: str, snapshot_id: str | None, engine_id: str, params: dict | None,
                     *, should_stop: Callable[[], bool] | None = None,
                     status_cb: Callable[[dict], None] | None = None) -> dict:
    """Fit on train, select on validation, then score frozen model on test once."""
    stop = should_stop or (lambda: False)
    report = status_cb or (lambda _: None)
    data = load_snapshot(project_id, snapshot_id)
    _validate_snapshot(data)
    if engine_id not in ENGINES:
        raise ValueError(f"Engine {engine_id} does not forecast numeric signals")
    if (params or {}).get("forecast_mode") == "bounded_trend_corridor":
        from pdm.trend_corridor import train_corridor_run
        return train_corridor_run(project_id, data, engine_id, params, stop, report)
    if (params or {}).get("forecast_mode") == "learned_joint_trajectories":
        from pdm.learned_trajectory import train_learned_run
        return train_learned_run(project_id, data, engine_id, params, stop, report)
    split = data["split"]
    training_features = data["features"][data["features"].unit_id.astype(str).isin(set(map(str, split["train"])))].copy()
    config = _params(engine_id, params, training_features)
    joint = config.get("forecast_mode") == "joint_residual_paths"
    funnel, bank, calibration = None, None, None
    if joint:
        from pdm.signal_funnel import calibrate_band
        physical = _physical_map(data)
        model, scaler, selection, bank = _joint_fit(data, engine_id, config, physical, stop, report)
        artifact_name = "model.pt" if engine_id in {"gru", "lstm"} else "model.joblib"
        # Validation first enters after OOF residual construction and final fitting.
        val = _joint_windows(data["features"], split["validation"], config, physical)
        val_pred = _predict(model, engine_id, val, scaler, should_stop=stop)[0]
        paths = _joint_paths(bank, val_pred, val, config, scaler, stop)
        # Calibration picks the first full observed history of the first sorted
        # cycle per equipment, even when every future target is unknown. Thus
        # future target availability cannot move a chosen origin later in time.
        cal_frame = _joint_windows(data["features"], sorted(map(str, split["validation"])),
                                   {**config, "max_windows_per_unit": None, "include_targetless": True}, physical)
        chosen = []
        for group in sorted(set(cal_frame["physical_unit_id"])):
            indices = [i for i, value in enumerate(cal_frame["physical_unit_id"]) if value == group]
            chosen.append(min(indices, key=lambda i: (cal_frame["unit_id"][i], cal_frame["as_of_s"][i])))
        chosen = np.asarray(chosen, int)
        cal_frame = {key: value[chosen] if key in {"x", "y", "mask"} else
                     [value[i] for i in chosen] if key in {"unit_id", "as_of_s", "physical_unit_id", "origin_within_calibration_scope"} else value
                     for key, value in cal_frame.items()}
        cal_pred = _predict(model, engine_id, cal_frame, scaler, should_stop=stop)[0]
        cal_paths = _joint_paths(bank, cal_pred, cal_frame, config, scaler, stop)
        if cal_paths is None or not len(cal_paths):
            calibration = {"version": "joint_residual_v1", "status": "insufficient_calibration",
                           "coverage": config["nominal_coverage"], "expansion": None,
                           "reason": "insufficient_complete_oof_or_validation_support"}
        else:
            lower, upper = np.quantile(cal_paths, [(1-config["nominal_coverage"])/2, (1+config["nominal_coverage"])/2], axis=1)
            actual, mask = _joint_targets(cal_frame)
            calibration = calibrate_band(lower, upper, actual, mask, cal_frame["physical_unit_id"], coverage=config["nominal_coverage"])
        calibration["selected_origins"] = [{"physical_unit_id": group, "unit_id": uid, "as_of_s": at}
                                            for group, uid, at in zip(cal_frame["physical_unit_id"], cal_frame["unit_id"], cal_frame["as_of_s"])]
        calibration.update(origin_policy="one_earliest_full_history_origin_per_physical_group_sorted_unit_then_timestamp; targetless_and_incomplete_origins_unknown",
                           engineering_default_only=True, operational_coverage_approved=False,
                           guarantee_scope="predeclared_earliest_origin_per_physical_group_only",
                           arbitrary_replay_origin_guarantee=False)
        val_metrics = _metrics(val_pred, {**val, "unit_id": val["physical_unit_id"]}, config["horizons_s"])
        val_metrics["aggregation"] = "physical_group_balanced"
        val_metrics["funnel"] = _joint_metrics(paths, val, config, scaler, calibration)
        funnel = {"mode": "joint_residual_v1", "artifact": "joint_residual_bank.npz",
                  "calibration": "funnel_calibration.json", "metadata": "joint_residual_metadata.json",
                  "status": bank["status"], "calibration_status": calibration["status"],
                  "nominal_coverage": config["nominal_coverage"], "path_samples": config["path_samples"],
                  "seed": config["seed"], "seed_policy": "fixed_run_seed_for_every_origin",
                  "approximation": bank["approximation"], "probabilistic_architecture_claim": False}
    else:
        train = _windows(data["features"], split["train"], config)
        val = _windows(data["features"], split["validation"], config)
        if not len(train["x"]) or not len(val["x"]):
            raise ValueError("Train and validation each need eligible, unbroken signal windows")
        if not np.all(train["mask"].any(axis=0)) or not np.all(val["mask"].any(axis=0)):
            raise ValueError("Every horizon needs known train and validation targets")
        train_values = training_features.signal.to_numpy(float)
        scaler = {"mean": float(np.mean(train_values)), "std": float(max(np.std(train_values), 1e-6)),
                  "fit_units": sorted(map(str, split["train"])),
                  "output_domain": data["schema"].get("output_domain", "real")}
        val_pred = None
        if engine_id == "full_cns":
            model, selection, val_pred = _fit_full_cns(train, val, config, scaler, stop, report)
            artifact_name = "model.joblib"
        elif engine_id in {"gru", "lstm"}:
            model = _fit_recurrent(engine_id, train, val, config, scaler, stop, report)
            artifact_name = "model.pt"
            selection = {"criterion": "validation_unit_equal_mae", "selected_checkpoint": "best_epoch"}
        else:
            model, selection = _fit_boosting(train, val, config, scaler, stop, report)
            artifact_name = "model.joblib"
        if val_pred is None:
            val_pred, _, _ = _predict(model, engine_id, val, scaler)
        val_metrics = _metrics(val_pred, val, config["horizons_s"])
    if stop():
        raise InterruptedError("Signal training cancelled before test evaluation")
    # The test data are first touched after model selection has finished.
    test = (_joint_windows(data["features"], split["test"], config, physical) if joint else
            _windows(data["features"], split["test"], config))
    report({"stage": "evaluating", "message": "Evaluating the frozen signal model on held-out Test"})
    test_pred, _, _ = _predict(model, engine_id, test, scaler, should_stop=stop)
    test_metrics = _metrics(test_pred, {**test, "unit_id": test["physical_unit_id"]} if joint else test, config["horizons_s"])
    if joint:
        test_metrics["aggregation"] = "physical_group_balanced"
    if joint:
        test_paths = _joint_paths(bank, test_pred, test, config, scaler, stop)
        test_metrics["funnel"] = _joint_metrics(test_paths, test, config, scaler, calibration)
    if stop():
        raise InterruptedError("Signal training cancelled before run publication")
    store = project_store()
    run_id = "signal-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    run_dir = store.run_path(project_id, run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        artifact = run_dir / artifact_name
        if engine_id in {"gru", "lstm"}:
            torch.save(model.state_dict(), artifact)
        else:
            joblib.dump(model, artifact)
        fingerprint = data["fingerprint"]
        contract = {"project_id": project_id, "snapshot_id": data["snapshot_id"],
                    "engine_id": engine_id, "params": config, "schema": data["schema"],
                    "snapshot_fingerprint_sha256": _digest(fingerprint), "scaler": scaler}
        if joint:
            from pdm.signal_funnel import save_sampler
            save_sampler(run_dir / funnel["artifact"], bank)
            atomic_write_json(run_dir / funnel["calibration"], calibration)
            bank_metadata = {key: value for key, value in bank.items() if key not in {"residuals", "groups", "horizons_s"}}
            bank_metadata.update(horizons_s=config["horizons_s"], oof_folds=selection["folds"],
                                 exclusions="Incomplete residual paths excluded; no padded residual suffixes",
                                 energy_scale_source="all_Train_signal_standard_deviation", energy_scale=scaler["std"],
                                 nominal_coverage_operational_approval=False)
            atomic_write_json(run_dir / funnel["metadata"], bank_metadata)
            contract["funnel"] = funnel
        if engine_id == "full_cns":
            contract["connectome"] = model.provenance
        atomic_write_json(run_dir / "training_contract.json", contract)
        manifest = {
            "schema_version": SCHEMA_VERSION, "task": "signal_forecast", "status": "completed",
            "project_id": project_id, "snapshot_id": data["snapshot_id"], "run_id": run_id,
            "engine_id": engine_id, "params": config, "schema": data["schema"],
            "snapshot_fingerprint_sha256": _digest(fingerprint), "scaler": scaler,
            "metrics": {"validation": val_metrics, "test": test_metrics},
            "selection": selection,
            "artifact": artifact_name, "artifacts": {artifact_name: sha256_file(artifact),
                                                    "training_contract.json": sha256_file(run_dir / "training_contract.json")},
            "interval_status": "unvalidated_pointwise_quantiles" if engine_id == "quantile_boosting" else "unavailable",
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        if joint:
            manifest["schema_version"] = "project_signal_forecast_joint_v1"
            manifest["funnel"] = funnel
            manifest["interval_status"] = calibration["status"]
            for name in (funnel["artifact"], funnel["calibration"], funnel["metadata"]):
                manifest["artifacts"][name] = sha256_file(run_dir / name)
        if engine_id == "full_cns":
            manifest["connectome"] = model.provenance
        atomic_write_json(run_dir / "manifest.json", manifest)
        if stop():
            raise InterruptedError("Signal training cancelled before run selection")
        store.update(project_id, selected_run_id=run_id)
        report({"stage": "completed", "progress": 1.0, "message": "Signal run saved"})
        return manifest
    except BaseException:
        # A failed run is not discoverable as a completed signal artifact.
        (run_dir / "manifest.json").unlink(missing_ok=True)
        raise


def list_project_runs(project_id: str) -> list[dict]:
    store = project_store()
    root = store.project_path(project_id) / "runs"
    runs = []
    for path in root.glob("*/manifest.json"):
        try:
            if path.is_symlink():
                continue
            manifest = read_json(path)
            if manifest.get("project_id") == project_id and manifest.get("task") == "signal_forecast" and manifest.get("status") == "completed":
                runs.append(manifest)
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return sorted(runs, key=lambda item: item.get("created_at", ""), reverse=True)


def load_signal_run(project_id: str, run_id: str) -> dict:
    store = project_store()
    directory = store.run_path(project_id, run_id)
    return _load_signal_run_directory(project_id, run_id, directory)


def _load_signal_run_directory(project_id: str, run_id: str, directory: Path) -> dict:
    """Apply the standard run validations to an owned run or trusted import stage."""
    if directory.is_symlink():
        raise ValueError("Signal run directory cannot be a symlink")
    for name in ("manifest.json", "training_contract.json"):
        if (directory / name).is_symlink():
            raise ValueError("Signal run files cannot be symlinks")
    manifest = read_json(directory / "manifest.json")
    if (manifest.get("task") != "signal_forecast" or manifest.get("status") != "completed"
            or manifest.get("project_id") != project_id or manifest.get("run_id") != run_id):
        raise ValueError("Run is not a completed signal forecast for this project")
    data = load_snapshot(project_id, manifest.get("snapshot_id"))
    if _digest(data["fingerprint"]) != manifest.get("snapshot_fingerprint_sha256"):
        raise ValueError("Signal run snapshot fingerprint mismatch")
    if manifest.get("schema") != data["schema"]:
        raise ValueError("Signal run schema differs from bound snapshot")
    for name, digest in manifest.get("artifacts", {}).items():
        if Path(name).name != name:
            raise ValueError("Signal run artifact name is unsafe")
        if (directory / name).is_symlink():
            raise ValueError("Signal run artifact cannot be a symlink")
        if sha256_file(directory / name) != digest:
            raise ValueError("Signal run artifact hash mismatch")
    if not manifest.get("artifacts") or manifest.get("artifact") not in manifest["artifacts"]:
        raise ValueError("Signal run artifact missing")
    contract = read_json(directory / "training_contract.json")
    for key in ("project_id", "snapshot_id", "engine_id", "params", "schema", "snapshot_fingerprint_sha256", "scaler"):
        if contract.get(key) != manifest.get(key):
            raise ValueError(f"Signal run {key} differs from saved training contract")
    if manifest.get("params", {}).get("forecast_mode") == "joint_residual_paths":
        funnel = manifest.get("funnel")
        if not funnel or funnel != contract.get("funnel") or funnel.get("mode") != "joint_residual_v1":
            raise ValueError("Signal joint funnel differs from saved training contract")
        if any(funnel.get(key) not in manifest["artifacts"] for key in ("artifact", "calibration", "metadata")):
            raise ValueError("Signal joint funnel artifact missing")
    if manifest.get("params", {}).get("forecast_mode") == "learned_joint_trajectories":
        if manifest.get("funnel") != contract.get("funnel") or not manifest.get("reload_verified"):
            raise ValueError("Learned trajectory contract or reload verification missing")
        if any(name not in manifest["artifacts"] for name in
               ("checkpoint.pt", "learned_model.json", "objective_trace.json", "model_input_contract.json", "training_contract.json")):
            raise ValueError("Learned distribution artifacts missing")
    if manifest.get("params", {}).get("forecast_mode") == "bounded_trend_corridor":
        from pdm.trend_corridor import corridor_params, corridor_widths
        corridor_widths(manifest.get("corridor_contract"))
        if (manifest.get("corridor_contract") != contract.get("corridor_contract")
                or not manifest.get("reload_verified")):
            raise ValueError("Saved bounded corridor contract mismatch")
        corridor_params(manifest["engine_id"], manifest["params"], None)
        if any(name not in manifest["artifacts"] for name in
               ("corridor.pt", "corridor_model.json", "objective_trace.json", "training_contract.json")):
            raise ValueError("Bounded corridor artifacts missing")
        if manifest["engine_id"] in {"quantile_boosting", "full_cns"} and "corridor_encoder.joblib" not in manifest["artifacts"]:
            raise ValueError("Bounded corridor encoder missing")
    if manifest.get("engine_id") == "full_cns":
        provenance = contract.get("connectome")
        if not provenance or provenance != manifest.get("connectome"):
            raise ValueError("Signal run connectome differs from saved training contract")
        if (provenance.get("graph_mode") != "real_connectome" or provenance.get("is_synthetic") is not False
                or provenance.get("node_sampling") is not False or not provenance.get("graph_hash")):
            raise ValueError("Signal run requires a complete real MaleCNS connectome")
    return {**manifest, "dir": directory, "artifact_path": directory / manifest["artifact"]}
