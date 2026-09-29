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
from torch.nn import functional as F

from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.models.signal_recurrent import SignalRecurrent
from pdm.projects import project_store

ENGINES = ("gru", "lstm", "quantile_boosting")
SCHEMA_VERSION = "project_signal_forecast_v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _params(engine_id: str, params: dict | None, features: pd.DataFrame) -> dict:
    if engine_id not in ENGINES:
        raise ValueError(f"Unsupported signal engine: {engine_id}")
    raw = dict(params or {})
    allowed = {"history_length", "horizons_s", "epochs", "hidden_size", "batch_size", "seed", "max_iter", "learning_rate"}
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
        "target_tolerance_s": max(0.000001, min(step * 0.01, 0.01)),
    }
    if not (2 <= config["history_length"] <= 256 and 1 <= config["epochs"] <= 500
            and 4 <= config["hidden_size"] <= 512 and 1 <= config["batch_size"] <= 4096
            and 2 <= config["max_iter"] <= 1000 and 0 < config["learning_rate"] <= 0.1):
        raise ValueError("Signal model parameters are outside supported ranges")
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


def _windows(features: pd.DataFrame, ids: list[str], params: dict) -> dict[str, Any]:
    history = params["history_length"]
    horizons = params["horizons_s"]
    tolerance = params["target_tolerance_s"]
    x_rows, y_rows, mask_rows, unit_rows, at_rows = [], [], [], [], []
    for uid in ids:
        for segment in _segments(features, str(uid)):
            ts = segment.timestamp_s.to_numpy(float)
            signal = segment.signal.to_numpy(float)
            for end in range(history - 1, len(segment)):
                target = np.full(len(horizons), np.nan, dtype=np.float32)
                for j, horizon in enumerate(horizons):
                    candidates = np.flatnonzero((ts > ts[end]) & (np.abs(ts - (ts[end] + horizon)) <= tolerance))
                    if len(candidates) == 1:
                        target[j] = signal[candidates[0]]
                mask = np.isfinite(target)
                if not mask.any():
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


def _predict(model: Any, engine_id: str, data: dict, scaler: dict) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if not len(data["x"]):
        shape = data["y"].shape
        return np.empty(shape), None, None
    x = (data["x"] - scaler["mean"]) / scaler["std"]
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
    for engine_id, label in (("fly", "Fly reservoir"), ("random", "Random reservoir"),
                             ("full_cns", "Full MaleCNS")):
        base.append({"engine_id": engine_id, "label": label, "available": False,
                     "reason": "This research engine has no validated numeric signal head for project replay.",
                     "params": []})
    return base


def _fit_recurrent(engine_id: str, train: dict, validation: dict, params: dict, scaler: dict,
                   should_stop: Callable[[], bool], status_cb: Callable[[dict], None]) -> SignalRecurrent:
    torch.manual_seed(params["seed"])
    model = SignalRecurrent(engine_id, 1, params["hidden_size"], len(params["horizons_s"]))
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
        pred, _, _ = _predict(model, engine_id, validation, scaler)
        metric = _metrics(pred, validation, params["horizons_s"])["mae"]
        if metric is None:
            raise ValueError("Validation has no known signal targets")
        if metric < best_val:
            best_val, best_state = metric, copy.deepcopy(model.state_dict())
        status_cb({"stage": "training", "progress": (epoch + 1) / params["epochs"],
                   "message": f"Epoch {epoch + 1}/{params['epochs']} · validation MAE {metric:.4g}"})
    if best_state is None:
        raise RuntimeError("No selected signal checkpoint")
    model.load_state_dict(best_state)
    return model


def _fit_boosting(train: dict, validation: dict, params: dict, scaler: dict,
                  should_stop: Callable[[], bool], status_cb: Callable[[dict], None]) -> tuple[list[dict], dict]:
    x = ((train["x"] - scaler["mean"]) / scaler["std"]).reshape(len(train["x"]), -1)
    weights = _weights(train["unit_id"])
    candidates = sorted({max(2, params["max_iter"] // 2), params["max_iter"]})
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
    split = data["split"]
    training_features = data["features"][data["features"].unit_id.astype(str).isin(set(map(str, split["train"])))].copy()
    config = _params(engine_id, params, training_features)
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
    if engine_id in {"gru", "lstm"}:
        model = _fit_recurrent(engine_id, train, val, config, scaler, stop, report)
        artifact_name = "model.pt"
        selection = {"criterion": "validation_unit_equal_mae", "selected_checkpoint": "best_epoch"}
    else:
        model, selection = _fit_boosting(train, val, config, scaler, stop, report)
        artifact_name = "model.joblib"
    val_pred, _, _ = _predict(model, engine_id, val, scaler)
    val_metrics = _metrics(val_pred, val, config["horizons_s"])
    if stop():
        raise InterruptedError("Signal training cancelled before test evaluation")
    # The test data are first touched after model selection has finished.
    test = _windows(data["features"], split["test"], config)
    test_pred, _, _ = _predict(model, engine_id, test, scaler)
    test_metrics = _metrics(test_pred, test, config["horizons_s"])
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
    return {**manifest, "dir": directory, "artifact_path": directory / manifest["artifact"]}
