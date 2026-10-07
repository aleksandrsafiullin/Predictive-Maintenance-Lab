"""Train-only forecasting engines. Predictors consume observed signal values only."""
from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from torch import nn

from .contract import (
    default_config,
    is_bounded_trend,
    is_dense_v2,
    is_reference_balanced,
    reference_origin_policy,
    unit_balanced_weights,
    validate_config,
)

QUANTILES = np.array([0.05, 0.50, 0.95])


class TrainingCancelled(InterruptedError):
    """A stop request never produces a deployable partial checkpoint."""


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _config(config):
    cfg = default_config(**config)
    validate_config(cfg)
    return cfg


def _check_stop(should_stop):
    if should_stop is not None and should_stop():
        raise TrainingCancelled("Training stopped by user")


def _arrays(windows, cfg, split):
    if windows.get("split", split) != split:
        raise ValueError(f"Expected {split} windows; other split values cannot enter fit")
    if windows.get("config", {}).get("horizon_protocol") != cfg.get("horizon_protocol"):
        raise ValueError("Training window configuration mismatch: horizon_protocol")
    if is_bounded_trend(cfg) and windows.get("config") != cfg:
        raise ValueError("Bounded training window configuration must equal the frozen configuration")
    for key in ("target", "unit", "cadence_s", "schema", "history_length", "max_horizon", "positive_domain"):
        if key in windows.get("config", {}) and windows["config"][key] != cfg[key]:
            raise ValueError(f"Training window configuration mismatch: {key}")
    x = np.asarray(windows["x"], dtype=np.float64)
    y = np.asarray(windows["y"], dtype=np.float64)
    mask = np.asarray(windows["mask"], dtype=bool)
    units = np.asarray(windows.get("physical_units", windows["units"])).astype(str)
    if x.ndim != 2 or x.shape[1] != cfg["history_length"]:
        raise ValueError("History shape does not match frozen configuration")
    if y.shape != (len(x), cfg["max_horizon"]) or mask.shape != y.shape or len(units) != len(x):
        raise ValueError("Target/mask/unit shapes do not match direct horizon grid")
    if not len(x) or not mask.any():
        raise ValueError("No supported targets; loss is unavailable")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("Observed histories must be finite nonnegative signals")
    if not np.isfinite(y[mask]).all() or (y[mask] < 0).any():
        raise ValueError("Supported targets must be finite nonnegative signals")
    if is_dense_v2(cfg) and ((x <= 0).any() or (y[mask] <= 0).any()
                           or (np.diff(mask.astype(int), axis=1) > 0).any()):
        raise ValueError("Dense-v2 needs positive observations and continuous prefix target masks")
    # Sanitize BEFORE arithmetic: multiplying NaN by zero does not mask it.
    return x, np.where(mask, y, 0.0), mask, units, unit_balanced_weights(mask, units)


def masked_pinball(y, q, mask, weights):
    """Physical-scale, exactly unit/horizon/origin-balanced objective."""
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    weights = np.asarray(weights, dtype=float)
    if not mask.any():
        raise ValueError("No supported targets")
    safe_y = y[mask]
    safe_q = q[mask]
    if not np.isfinite(safe_y).all() or not np.isfinite(safe_q).all():
        raise ValueError("Nonfinite supported target/prediction")
    errors = safe_y[:, None] - safe_q
    losses = np.maximum(QUANTILES * errors, (QUANTILES - 1) * errors)
    return float(np.sum(weights[mask] * losses.mean(axis=-1)))


def _preprocess(x, y, mask, units):
    # Each physical unit contributes equally to input moments despite overlap.
    z = np.log1p(x)
    group_means = [z[units == u].mean() for u in np.unique(units)]
    group_second = [(z[units == u] ** 2).mean() for u in np.unique(units)]
    mean = float(np.mean(group_means))
    std = max(float(np.sqrt(max(np.mean(group_second) - mean * mean, 0))), 1e-6)
    scale = max(float(np.median(y[mask])), 1e-6)
    return {"input_transform": "log1p_train_standardized", "input_mean": mean,
            "input_std": std, "target_transform": "positive_train_scale", "target_scale": scale,
            "fit_split": "train", "fit_physical_units": sorted(np.unique(units).tolist())}


def _transform(x, prep):
    return ((np.log1p(x) - prep["input_mean"]) / prep["input_std"]).astype(np.float32)


class DirectQuantileGRU(nn.Module):
    """One recurrent encoder with direct ordered-positive marginal heads."""
    def __init__(self, hidden_size, layers, horizon):
        super().__init__()
        self.encoder = nn.GRU(1, hidden_size, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden_size, horizon * 3)
        self.horizon = horizon
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias.reshape(horizon, 3)[:, 1] = 4.0
            self.head.bias.reshape(horizon, 3)[:, 2] = -4.0

    def forward(self, x, last_scaled):
        _, state = self.encoder(x.unsqueeze(-1))
        a, b, c = self.head(state[-1]).reshape(-1, self.horizon, 3).unbind(-1)
        level = last_scaled.clamp_min(1e-8).unsqueeze(-1)
        inverse_softplus = level + torch.log(-torch.expm1(-level))
        median = torch.nn.functional.softplus(inverse_softplus + a)
        low = median * torch.sigmoid(b)
        high = median + torch.nn.functional.softplus(c)
        return torch.stack((low, median, high), -1)


def _relative_history(x, target_scale):
    """Every statistic uses only the supplied causal history, never origin metadata."""
    x = np.asarray(x, dtype=float)
    level = np.median(x[:, -min(5, x.shape[1]):], axis=1)
    relative = np.log(x / level[:, None])
    slopes = []
    for length in (5, 20, 60):
        length = min(length, x.shape[1])
        half = max(1, length // 2)
        values = relative[:, -length:]
        slopes.append((np.median(values[:, -half:], axis=1) - np.median(values[:, :half], axis=1)) / max(1, length - half))
    median = np.median(relative, axis=1)
    noise = np.median(np.abs(relative - median[:, None]), axis=1)
    features = np.column_stack((np.log(level / target_scale), *slopes, noise, relative[:, -1]))
    return relative, features, level


def _trend_preprocess(x, units, prep):
    relative, features, _ = _relative_history(x, prep["target_scale"])
    means = np.array([features[units == u].mean(axis=0) for u in np.unique(units)])
    seconds = np.array([(features[units == u] ** 2).mean(axis=0) for u in np.unique(units)])
    mean = means.mean(axis=0)
    std = np.maximum(np.sqrt(np.maximum(seconds.mean(axis=0) - mean ** 2, 0)), 1e-6)
    shape_mean = float(np.mean([relative[units == u].mean() for u in np.unique(units)]))
    shape_second = float(np.mean([(relative[units == u] ** 2).mean() for u in np.unique(units)]))
    return {**prep, "center_protocol": "relative_shared_lead_v1",
            "feature_names": ["log_robust_level", "slope_5", "slope_20", "slope_60", "relative_mad", "last_relative_offset"],
            "trend_feature_mean": mean.tolist(), "trend_feature_std": std.tolist(),
            "relative_shape_mean": shape_mean,
            "relative_shape_std": max(np.sqrt(max(shape_second - shape_mean ** 2, 0)), 1e-6),
            "feature_information": "supplied_history_only;no_age_or_unit_id_or_future"}


def _trend_design(x, prep):
    relative, features, level = _relative_history(x, prep["target_scale"])
    z = (relative - prep["relative_shape_mean"]) / prep["relative_shape_std"]
    features = (features - prep["trend_feature_mean"]) / prep["trend_feature_std"]
    return z.astype(np.float32), features.astype(np.float32), level.astype(np.float32)


class RelativeTrendGRU(nn.Module):
    """History-conditioned smooth direct curve; neither upward forcing nor stitching."""
    def __init__(self, hidden_size, horizon, basis_size=16):
        super().__init__()
        from scipy.interpolate import CubicSpline
        knots = np.linspace(0, horizon, basis_size)
        spline = CubicSpline(knots, np.eye(basis_size), axis=0, bc_type="natural")
        basis = (spline(np.arange(1, horizon + 1)) - spline(0)).T
        self.register_buffer("basis", torch.as_tensor(basis, dtype=torch.float32))
        self.encoder = nn.GRU(1, hidden_size, batch_first=True)
        self.coefficients = nn.Linear(hidden_size + 6, basis_size)
        nn.init.zeros_(self.coefficients.weight)
        nn.init.zeros_(self.coefficients.bias)
        self.horizon = horizon

    def forward(self, x, features, level):
        _, state = self.encoder(x.unsqueeze(-1))
        coefficients = self.coefficients(torch.cat((state[-1], features), dim=-1))
        delta = coefficients @ self.basis
        # Explicit float32 representability guard, never a width/quality adjustment.
        log_center = (torch.log(level).unsqueeze(-1) + delta).clamp(-80, 80)
        return torch.exp(log_center)


def _trend_predict(network, x, prep, batch_size=512):
    outputs = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            z, features, level = _trend_design(x[start:start + batch_size], prep)
            outputs.append(network(torch.from_numpy(z), torch.from_numpy(features), torch.from_numpy(level)).numpy())
    return np.concatenate(outputs) if outputs else np.empty((0, network.horizon))


def _gru_predict(network, x, prep, batch_size=512):
    outputs = []
    network.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = x[start:start + batch_size]
            z = torch.from_numpy(_transform(batch, prep))
            last = torch.as_tensor(batch[:, -1] / prep["target_scale"], dtype=torch.float32)
            outputs.append(network(z, last).numpy() * prep["target_scale"])
    return np.concatenate(outputs, axis=0) if outputs else np.empty((0, network.horizon, 3))


def _features(x, prep, lead=None, horizon=60):
    z = _transform(x, prep)
    if lead is not None:
        return np.column_stack((z, np.asarray(lead, dtype=np.float32) / horizon)).astype(np.float32)
    return np.column_stack((np.repeat(z, horizon, axis=0), np.tile(np.arange(1, horizon + 1), len(x)) / horizon)).astype(np.float32)


def baseline_predictions(x, engine_id, horizon=60):
    x = np.asarray(x, dtype=float)
    last = x[:, -1]
    if engine_id == "persistence":
        median = np.repeat(last[:, None], horizon, axis=1)
    elif engine_id == "local_trend":
        values = np.log(np.maximum(x[:, -min(20, x.shape[1]):], 1e-12))
        time = np.arange(values.shape[1], dtype=float)
        centered = time - time.mean()
        slope = np.sum(values * centered, axis=1) / np.sum(centered ** 2) if len(time) > 1 else np.zeros(len(x))
        intercept_now = values.mean(axis=1) + slope * (time[-1] - time.mean())
        exponent = intercept_now[:, None] + slope[:, None] * np.arange(1, horizon + 1)
        # Finite representability guard, not an interval-width clipping rule.
        median = np.exp(np.clip(exponent, -700, 700))
    else:
        raise ValueError("Unknown baseline")
    return np.repeat(median[:, :, None], 3, axis=2)


def _model_hash(model):
    identity = {k: model[k] for k in ("engine_id", "config", "preprocessing")}
    identity["training_summary"] = model["training_summary"]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    if model["engine_id"] == "gru":
        for name, value in sorted(model["model"].state_dict().items()):
            array = value.detach().cpu().numpy()
            digest.update(name.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
    elif model["engine_id"] == "quantile_boosting":
        def add_array(array):
            array = np.asarray(array)
            digest.update(str(array.shape).encode())
            if array.dtype.names:
                for field in array.dtype.names:
                    digest.update(field.encode())
                    add_array(array[field])
            else:
                digest.update(str(array.dtype).encode())
                digest.update(np.ascontiguousarray(array).tobytes())
        for estimator in model["model"]:
            digest.update(json.dumps(estimator.get_params(), sort_keys=True).encode())
            add_array(estimator._baseline_prediction)
            for thresholds in estimator._bin_mapper.bin_thresholds_:
                add_array(thresholds)
            for iteration in estimator._predictors:
                for tree in iteration:
                    for name in ("nodes", "binned_left_cat_bitsets", "raw_left_cat_bitsets"):
                        digest.update(name.encode())
                        add_array(getattr(tree, name))
    return digest.hexdigest()


def fit_model(train_windows, validation_windows, config, *, should_stop: Callable | None = None, status_cb: Callable | None = None, checkpoint_directory=None, reference_validation_windows=None, reference_train_windows=None):
    cfg = _config(config)
    _check_stop(should_stop)
    x, y, mask, units, weights = _arrays(train_windows, cfg, "train")
    vx, vy, vm, vu, vw = _arrays(validation_windows, cfg, "validation")
    if is_dense_v2(cfg) and not mask.any(axis=0).all():
        raise ValueError("Dense-v2 cannot train a direct head with zero Train target support")
    source_keys = ("snapshot_id", "dataset_hash", "release_id", "suite", "profile")
    source = {key: train_windows[key] for key in source_keys if key in train_windows}
    for key in source:
        if validation_windows.get(key) != source[key]:
            raise ValueError(f"Train/Validation source provenance mismatch: {key}")
    if set(units) & set(vu):
        raise ValueError("Train and Validation physical units overlap")
    engine = cfg["engine_id"]
    prep = _preprocess(x, y, mask, units)
    summary = {"fit_split": "train", "selection_split": "validation", "train_units": sorted(set(units)),
               "validation_units": sorted(set(vu)), "train_origins": len(x), "validation_origins": len(vx),
               "loss": "unit_horizon_origin_balanced_masked_pinball", "weight_sum": float(weights.sum()),
               "seed": cfg["seed"], "source_binding": source, "epochs": []}
    if is_dense_v2(cfg):
        from .horizons import target_support
        summary["support_by_lead"] = {"train": target_support(mask, units),
                                      "validation": target_support(vm, vu)}
    model = {"engine_id": engine, "config": cfg, "preprocessing": prep, "model": None, "training_summary": summary}
    if is_bounded_trend(cfg):
        return _fit_bounded(model, (x, y, mask, units, weights), (vx, vy, vm, vu, vw),
                            reference_validation_windows, should_stop, status_cb, checkpoint_directory,
                            reference_train_windows, train_windows)
    if engine in ("persistence", "local_trend"):
        score = masked_pinball(vy, baseline_predictions(vx, engine, cfg["max_horizon"]), vm, vw)
        summary.update(initial_validation_loss=score, best_validation_loss=score, best_epoch=0,
                       raw_quantiles_label="technical_uncalibrated_placeholder")
    elif engine == "gru":
        torch.manual_seed(cfg["seed"])
        # CPU avoids hidden platform-dependent device policies; bundle is portable.
        torch.set_num_threads(int(cfg.get("torch_threads", 2)))
        net = DirectQuantileGRU(cfg["hidden_size"], cfg["layers"], cfg["max_horizon"])
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
        initial = masked_pinball(vy, _gru_predict(net, vx, prep), vm, vw)
        best, best_epoch, best_state = initial, 0, copy.deepcopy(net.state_dict())
        rng = np.random.default_rng(cfg["seed"])
        z = torch.from_numpy(_transform(x, prep))
        ty = torch.as_tensor(y / prep["target_scale"], dtype=torch.float32)
        tm = torch.as_tensor(mask)
        tw = torch.as_tensor(weights, dtype=torch.float32)
        last = torch.as_tensor(x[:, -1] / prep["target_scale"], dtype=torch.float32)
        taus = torch.as_tensor(QUANTILES, dtype=torch.float32)
        start_epoch = 0
        checkpoint = Path(checkpoint_directory) / "training_resume.pt" if checkpoint_directory else None
        binding = json_hash({"config": cfg, "train_units": sorted(set(units)), "validation_units": sorted(set(vu)),
                             "train_arrays": hashlib.sha256(x.tobytes() + y.tobytes() + mask.tobytes() + weights.tobytes()).hexdigest(),
                             "validation_arrays": hashlib.sha256(vx.tobytes() + vy.tobytes() + vm.tobytes() + vw.tobytes()).hexdigest()})
        if checkpoint is not None and checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["binding"] != binding:
                raise ValueError("Resume checkpoint does not match exact Train/Validation/config binding")
            net.load_state_dict(saved["state"])
            optimizer.load_state_dict(saved["optimizer"])
            best_state, best, best_epoch = saved["best_state"], saved["best_loss"], saved["best_epoch"]
            summary["epochs"], start_epoch = saved["epochs"], saved["epoch"]
            rng.bit_generator.state = saved["rng_state"]
        for epoch in (range(start_epoch + 1, cfg["max_epochs"] + 1) if start_epoch - best_epoch < cfg["patience"] else []):
            _check_stop(should_stop)
            net.train()
            total = 0.0
            for ids in np.array_split(rng.permutation(len(x)), max(1, int(np.ceil(len(x) / cfg["batch_size"])))):
                _check_stop(should_stop)
                optimizer.zero_grad()
                predicted = net(z[ids], last[ids])
                active = tm[ids]
                residual = ty[ids][active].unsqueeze(-1) - predicted[active]
                loss = (tw[ids][active] * torch.maximum(taus * residual, (taus - 1) * residual).mean(-1)).sum()
                # Global weights are not renormalized separately within each minibatch.
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise RuntimeError("Nonfinite GRU loss or gradient")
                nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip"])
                optimizer.step()
                total += float(loss.detach()) * prep["target_scale"]
            val = masked_pinball(vy, _gru_predict(net, vx, prep), vm, vw)
            record = {"epoch": epoch, "train_loss": total, "validation_loss": val}
            summary["epochs"].append(record)
            if status_cb:
                status_cb(record)
            if val < best - 1e-9:
                best, best_epoch, best_state = val, epoch, copy.deepcopy(net.state_dict())
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({"binding": binding, "state": net.state_dict(), "optimizer": optimizer.state_dict(),
                            "best_state": best_state, "best_loss": best, "best_epoch": best_epoch,
                            "epochs": summary["epochs"], "epoch": epoch, "rng_state": rng.bit_generator.state}, temporary)
                temporary.replace(checkpoint)
            if epoch - best_epoch >= cfg["patience"]:
                break
        _check_stop(should_stop)
        net.load_state_dict(best_state)
        net.eval()
        model["model"] = net
        summary.update(initial_validation_loss=initial, best_validation_loss=best, best_epoch=best_epoch,
                       stopped_epoch=len(summary["epochs"]), head="positive_ordered_residual_current_level")
    elif engine == "quantile_boosting":
        rows, leads = np.where(mask)
        features = _features(x[rows], prep, leads + 1, cfg["max_horizon"])
        targets = y[rows, leads] / prep["target_scale"]
        sample_weights = weights[rows, leads] * len(rows)
        estimators = [HistGradientBoostingRegressor(loss="quantile", quantile=float(tau),
                        max_iter=1, max_leaf_nodes=cfg["max_leaf_nodes"], min_samples_leaf=cfg["min_samples_leaf"],
                        learning_rate=cfg["boosting_learning_rate"], l2_regularization=cfg["l2_regularization"],
                        early_stopping=False, warm_start=True, random_state=cfg["seed"]) for tau in QUANTILES]
        validation_features = _features(vx, prep, horizon=cfg["max_horizon"])
        initial_q = np.stack([np.full(vy.shape, np.quantile(targets, tau)) for tau in QUANTILES], -1) * prep["target_scale"]
        initial = masked_pinball(vy, initial_q, vm, vw)
        best, best_iter, best_estimators = float("inf"), 0, None
        # True held-out Validation selects the shared iteration checkpoint.
        iterations = list(range(10, cfg["boosting_max_iter"] + 1, 10))
        if not iterations or iterations[-1] != cfg["boosting_max_iter"]:
            iterations.append(cfg["boosting_max_iter"])
        for iteration in iterations:
            _check_stop(should_stop)
            for estimator in estimators:
                _check_stop(should_stop)
                estimator.set_params(max_iter=iteration)
                with threadpool_limits(limits=2):
                    estimator.fit(features, targets, sample_weight=sample_weights)
            with threadpool_limits(limits=2):
                raw = np.stack([e.predict(validation_features).reshape(vy.shape) for e in estimators], -1) * prep["target_scale"]
            q = np.sort(np.maximum(raw, 0), axis=-1)
            val = masked_pinball(vy, q, vm, vw)
            record = {"epoch": iteration, "validation_loss": val,
                      "train_loss": None, "raw_crossing_rate": float(np.mean((raw[..., 0] > raw[..., 1]) | (raw[..., 1] > raw[..., 2])))}
            summary["epochs"].append(record)
            if status_cb:
                status_cb(record)
            if val < best:
                best, best_iter, best_estimators = val, iteration, copy.deepcopy(estimators)
        _check_stop(should_stop)
        model["model"] = best_estimators
        summary.update(initial_validation_loss=initial, best_validation_loss=best, best_epoch=best_iter,
                       stopped_epoch=cfg["boosting_max_iter"], raw_quantiles_label="sorted_uncalibrated_quantile_estimates")
    else:
        raise ValueError("Unsupported engine")
    model["model_hash"] = _model_hash(model)
    return model


def _validate_train_reference_evidence(train_windows, reference_windows, reference_arrays, cfg):
    from .windows import causal_anchor_content_hash
    evidence = train_windows.get("admission",{}).get("reference_train_evidence",{})
    rolling_manifest = train_windows.get("origin_manifest",{})
    if (train_windows.get("mode") != "train"
            or rolling_manifest.get("origin_hash") != json_hash({k:v for k,v in rolling_manifest.items() if k != "origin_hash"})
            or evidence.get("evidence_hash") != json_hash({k:v for k,v in evidence.items() if k != "evidence_hash"})
            or evidence.get("origin_policy") != reference_origin_policy(cfg)
            or evidence.get("config_hash") != json_hash(cfg)
            or evidence.get("rolling_origin_hash") != rolling_manifest.get("origin_hash")):
        raise ValueError("Reference Train causal admission evidence integrity/binding mismatch")
    expected = evidence.get("records",[])
    histories,targets,supported,physical_units,_ = reference_arrays
    if len(expected) != len(histories) or {r["physical_unit_id"] for r in expected} != set(physical_units):
        raise ValueError("Reference Train population differs from canonical causal admission")
    identities = [{"unit_id":str(u),"physical_unit_id":str(p),"origin_s":float(o)}
                  for u,p,o in zip(reference_windows["units"],physical_units,reference_windows["origin_s"])]
    if identities != [{k:r[k] for k in ("unit_id","physical_unit_id","origin_s")} for r in expected]:
        raise ValueError("Reference Train earliest anchor differs from canonical causal admission")
    if any(record.get("content_hash") != causal_anchor_content_hash(x,y,m)
           for record,x,y,m in zip(expected,histories,targets,supported)):
        raise ValueError("Reference Train anchor content differs from canonical causal admission")
    return evidence["evidence_hash"]


def _fit_bounded(model, train, validation, reference_windows, should_stop, status_cb, checkpoint_directory,
                 reference_train_windows, train_windows):
    from .corridor import center_objective
    cfg, summary = model["config"], model["training_summary"]
    if reference_windows is None or reference_windows.get("mode") != "reference" or reference_windows.get("origin_policy") != reference_origin_policy(cfg):
        raise ValueError("Bounded trend selection requires explicit earliest-reference Validation windows")
    if reference_windows.get("config") != cfg:
        raise ValueError("Reference Validation configuration mismatch")
    for key, value in summary["source_binding"].items():
        if reference_windows.get(key) != value:
            raise ValueError(f"Reference Validation provenance mismatch: {key}")
    x, y, mask, units, weights = train
    vx, vy, vm, vu, vw = validation
    rx, ry, rm, ru, rw = _arrays(reference_windows, cfg, "validation")
    if len(set(ru)) != len(ru) or not set(ru[rm.any(axis=1)]) <= set(vu) or set(ru) & set(units):
        raise ValueError("Reference Validation must contain independent Validation physical units")
    prep = _trend_preprocess(x, units, model["preprocessing"])
    model["preprocessing"] = prep
    summary["validation_units"] = sorted(set(vu) | set(ru))
    summary.update(output_kind="decision_corridor", coverage_guarantee=False, nominal=None,
                   loss="unit_horizon_origin_balanced_raw_log_center_error_plus_point_excess",
                   objective="unit_balanced_raw_log_center_error_plus_point_excess",
                   center_loss_weight=1.0, point_excess_loss_weight=1.0,
                   selection_components={"rolling_validation": .5, "earliest_reference_validation": .5},
                   reference_validation_units=sorted(set(ru)), reference_validation_origins=len(rx),
                   reference_validation_origin_hash=reference_windows["origin_manifest"]["origin_hash"],
                   center_representability_guard="log physical center clamped to [-80,80];not width clipping")
    balanced = is_reference_balanced(cfg)
    reference_train = None
    fit_x, fit_y, fit_mask, fit_weights = x, y, mask, weights
    if balanced:
        if any(key not in summary["source_binding"] for key in ("snapshot_id", "dataset_hash", "release_id", "suite", "profile")):
            raise ValueError("Balanced training requires explicit Train source provenance")
        if reference_train_windows is None or reference_train_windows.get("mode") != "reference" or reference_train_windows.get("origin_policy") != reference_origin_policy(cfg):
            raise ValueError("Balanced training requires independent earliest-reference Train windows")
        if reference_train_windows.get("config") != cfg:
            raise ValueError("Reference Train configuration mismatch")
        for key, value in summary["source_binding"].items():
            if reference_train_windows.get(key) != value:
                raise ValueError(f"Reference Train provenance mismatch: {key}")
        reference_train = _arrays(reference_train_windows, cfg, "train")
        ax, ay, am, au, aw = reference_train
        if len(set(au)) != len(au) or not set(au[am.any(axis=1)]) <= set(units) or set(au) & (set(vu) | set(ru)):
            raise ValueError("Reference Train requires one independent origin per Train physical unit")
        manifest = reference_train_windows.get("origin_manifest", {})
        records = [{"unit_id": str(u), "physical_unit_id": str(p), "origin_s": float(o)}
                   for u, p, o in zip(reference_train_windows["units"], au, reference_train_windows["origin_s"])]
        if (manifest.get("origin_hash") != json_hash({k: v for k, v in manifest.items() if k != "origin_hash"})
                or manifest.get("origin_policy") != reference_origin_policy(cfg)
                or manifest.get("records") != records or len(records) != len(ax)):
            raise ValueError("Reference Train origin manifest integrity/records mismatch")
        admission_hash = _validate_train_reference_evidence(train_windows,reference_train_windows,reference_train,cfg)
        fit_x = np.concatenate((x, ax))
        fit_y = np.concatenate((y, ay))
        fit_mask = np.concatenate((mask, am))
        fit_weights = np.concatenate((.5 * weights, .5 * aw))
        summary.update(training_population_protocol=cfg["training_population_protocol"],
                       training_components={"rolling_train": .5, "earliest_reference_train": .5},
                       optimization_origins=len(fit_x), reference_train_origins=len(ax),
                       optimization_weight_sum=float(fit_weights.sum()),
                       reference_train_units=sorted(set(au)),
                       reference_train_origin_hash=manifest["origin_hash"],
                       reference_train_admission_hash=admission_hash,
                       reference_train_source_binding={k: reference_train_windows[k] for k in summary["source_binding"]},
                       preprocessing_population="original_rolling_train_only",
                       train_risk_definition="end_of_epoch_fixed_model;separately_unit_lead_origin_normalized")
        summary["train_units"] = sorted(set(units) | set(au))
    elif reference_train_windows is not None:
        raise ValueError("Reference Train optimization requires explicit training_population_protocol")

    def risk(predict_center, arrays):
        histories, targets, supported, _, population_weights = arrays
        total = 0.
        for start in range(0, len(histories), 512):
            sl = slice(start, start + 512)
            if supported[sl].any():
                total += center_objective(targets[sl], predict_center(histories[sl]), supported[sl], population_weights[sl], cfg)
        return total

    def train_components(predict_center):
        rolling_risk, reference_risk = risk(predict_center, train), risk(predict_center, reference_train)
        return {"rolling_train_loss": rolling_risk, "reference_train_loss": reference_risk,
                "train_loss": .5 * rolling_risk + .5 * reference_risk,
                "train_loss_scope": "end_of_epoch_fixed_model_full_population_risk"}

    def score(vc, rc):
        rolling = center_objective(vy, vc, vm, vw, cfg)
        reference = center_objective(ry, rc, rm, rw, cfg)
        return .5 * rolling + .5 * reference, rolling, reference

    engine = model["engine_id"]
    if engine in ("persistence", "local_trend"):
        initial, rolling, reference = score(baseline_predictions(vx, engine, cfg["max_horizon"])[..., 1],
                                            baseline_predictions(rx, engine, cfg["max_horizon"])[..., 1])
        summary.update(initial_validation_loss=initial, best_validation_loss=initial, best_epoch=0,
                       best_rolling_validation_loss=rolling, best_reference_validation_loss=reference,
                       center_kind="technical_fixed_baseline")
    elif engine == "gru":
        torch.manual_seed(cfg["seed"])
        torch.set_num_threads(int(cfg.get("torch_threads", 2)))
        net = RelativeTrendGRU(cfg["hidden_size"], cfg["max_horizon"], cfg["trend_basis_size"])
        optimizer = torch.optim.Adam(net.parameters(), lr=cfg["learning_rate"])
        initial, rolling, reference = score(_trend_predict(net, vx, prep), _trend_predict(net, rx, prep))
        best, best_epoch, best_state = initial, 0, copy.deepcopy(net.state_dict())
        best_components = (rolling, reference)
        rng = np.random.default_rng(cfg["seed"])
        z, features, levels = _trend_design(fit_x, prep)
        tz, tf, tl = map(torch.from_numpy, (z, features, levels))
        target_log = torch.as_tensor(np.log(np.where(fit_mask, fit_y, 1)), dtype=torch.float32)
        tm = torch.as_tensor(fit_mask)
        tw = torch.as_tensor(fit_weights, dtype=torch.float32)
        half = cfg["width_budget"] / 2
        log_low, log_high = np.log1p(-half), np.log1p(half)
        checkpoint = Path(checkpoint_directory) / "training_resume.pt" if checkpoint_directory else None
        def arrays_hash(arrays):
            digest = hashlib.sha256()
            for array in arrays:
                digest.update(np.asarray(array).tobytes())
            return digest.hexdigest()
        binding_payload = {"config": cfg, "preprocessing": prep,
                             "train_arrays": arrays_hash(train), "validation_arrays": arrays_hash(validation),
                             "reference_arrays": arrays_hash((rx, ry, rm, ru, rw)),
                             "reference_origin_hash": reference_windows["origin_manifest"]["origin_hash"]}
        if balanced:
            binding_payload.update(reference_train_arrays=arrays_hash(reference_train),
                                   reference_train_origin_hash=summary["reference_train_origin_hash"],
                                   reference_train_admission_hash=summary["reference_train_admission_hash"],
                                   reference_train_source_binding=summary["reference_train_source_binding"])
        binding = json_hash(binding_payload)
        start_epoch = 0
        if checkpoint is not None and checkpoint.exists():
            saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if saved["binding"] != binding:
                raise ValueError("Resume checkpoint does not match exact bounded Train/Validation/reference/config")
            net.load_state_dict(saved["state"])
            optimizer.load_state_dict(saved["optimizer"])
            best_state, best, best_epoch = saved["best_state"], saved["best_loss"], saved["best_epoch"]
            best_components = tuple(saved["best_components"])
            summary["epochs"], start_epoch = saved["epochs"], saved["epoch"]
            rng.bit_generator.state = saved["rng_state"]
        epochs = range(start_epoch + 1, cfg["max_epochs"] + 1) if start_epoch - best_epoch < cfg["patience"] else []
        for epoch in epochs:
            _check_stop(should_stop)
            net.train()
            total = 0.0
            for ids in np.array_split(rng.permutation(len(fit_x)), max(1, int(np.ceil(len(fit_x) / cfg["batch_size"])))):
                _check_stop(should_stop)
                optimizer.zero_grad()
                log_center = torch.log(net(tz[ids], tf[ids], tl[ids]))[tm[ids]]
                target = target_log[ids][tm[ids]]
                error = torch.abs(log_center - target)
                excess = torch.relu(log_center + log_low - target) + torch.relu(target - log_center - log_high)
                loss = (tw[ids][tm[ids]] * (error + excess)).sum()
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in net.parameters()):
                    raise RuntimeError("Nonfinite bounded trend loss/gradient")
                nn.utils.clip_grad_norm_(net.parameters(), cfg["gradient_clip"])
                optimizer.step()
                total += float(loss.detach())
            val, rolling, reference = score(_trend_predict(net, vx, prep), _trend_predict(net, rx, prep))
            record = {"epoch": epoch, "train_loss": total, "validation_loss": val,
                      "rolling_validation_loss": rolling, "reference_validation_loss": reference}
            if balanced:
                record.update(stochastic_in_epoch_train_loss=total,
                              **train_components(lambda histories: _trend_predict(net, histories, prep)))
            summary["epochs"].append(record)
            if status_cb:
                status_cb(record)
            if val < best - 1e-9:
                best, best_epoch, best_state = val, epoch, copy.deepcopy(net.state_dict())
                best_components = (rolling, reference)
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                torch.save({"binding": binding, "state": net.state_dict(), "optimizer": optimizer.state_dict(),
                            "best_state": best_state, "best_loss": best, "best_epoch": best_epoch,
                            "best_components": best_components, "epochs": summary["epochs"], "epoch": epoch,
                            "rng_state": rng.bit_generator.state}, temporary)
                temporary.replace(checkpoint)
            if epoch - best_epoch >= cfg["patience"]:
                break
        _check_stop(should_stop)
        net.load_state_dict(best_state)
        net.eval()
        model["model"] = net
        summary.update(initial_validation_loss=initial, best_validation_loss=best, best_epoch=best_epoch,
                       stopped_epoch=len(summary["epochs"]), best_rolling_validation_loss=best_components[0],
                       best_reference_validation_loss=best_components[1], center_kind="positive_relative_smooth_basis_gru")
    elif engine == "quantile_boosting":
        # One median center with relative causal features; no interval/calibration fitting.
        def features_for(histories, leads=None):
            z, feature, level = _trend_design(histories, prep)
            base = np.column_stack((z, feature))
            if leads is not None:
                return np.column_stack((base, leads / cfg["max_horizon"])).astype(np.float32), level
            return np.column_stack((np.repeat(base, cfg["max_horizon"], axis=0),
                                    np.tile(np.arange(1, cfg["max_horizon"] + 1), len(histories)) / cfg["max_horizon"])).astype(np.float32), level
        rows, leads = np.where(fit_mask)
        features, level = features_for(fit_x[rows], leads + 1)
        targets = np.log(fit_y[rows, leads] / level)
        estimator = HistGradientBoostingRegressor(loss="quantile", quantile=.5, max_iter=1,
                      max_leaf_nodes=cfg["max_leaf_nodes"], min_samples_leaf=cfg["min_samples_leaf"],
                      learning_rate=cfg["boosting_learning_rate"], l2_regularization=cfg["l2_regularization"],
                      early_stopping=False, warm_start=True, random_state=cfg["seed"])
        vf, vl = features_for(vx)
        rf, rl = features_for(rx)
        initial, _, _ = score(np.repeat(vl[:, None], cfg["max_horizon"], axis=1), np.repeat(rl[:, None], cfg["max_horizon"], axis=1))
        best, best_iter, best_estimator, best_components = float("inf"), 0, None, None
        iterations = list(range(10, cfg["boosting_max_iter"] + 1, 10))
        if not iterations or iterations[-1] != cfg["boosting_max_iter"]:
            iterations.append(cfg["boosting_max_iter"])
        for iteration in iterations:
            _check_stop(should_stop)
            estimator.set_params(max_iter=iteration)
            with threadpool_limits(limits=2):
                estimator.fit(features, targets, sample_weight=fit_weights[rows, leads] * len(rows))
                vc = np.exp(np.clip(estimator.predict(vf).reshape(vy.shape) + np.log(vl[:, None]), -80, 80))
                rc = np.exp(np.clip(estimator.predict(rf).reshape(ry.shape) + np.log(rl[:, None]), -80, 80))
            val, rolling, reference = score(vc, rc)
            record = {"epoch": iteration, "train_loss": None, "validation_loss": val,
                      "rolling_validation_loss": rolling, "reference_validation_loss": reference}
            if balanced:
                def boost_center(histories):
                    bf, bl = features_for(histories)
                    with threadpool_limits(limits=2):
                        return np.exp(np.clip(estimator.predict(bf).reshape(len(histories), cfg["max_horizon"]) + np.log(bl[:, None]), -80, 80))
                record.update(**train_components(boost_center))
            summary["epochs"].append(record)
            if status_cb:
                status_cb(record)
            if val < best:
                best, best_iter, best_estimator = val, iteration, copy.deepcopy(estimator)
                best_components = (rolling, reference)
        _check_stop(should_stop)
        model["model"] = [best_estimator]
        summary.update(initial_validation_loss=initial, best_validation_loss=best, best_epoch=best_iter,
                       stopped_epoch=cfg["boosting_max_iter"], best_rolling_validation_loss=best_components[0],
                       best_reference_validation_loss=best_components[1], center_kind="relative_median_boosting",
                       train_objective_note="median quantile loss;center-plus-point-excess selects Validation checkpoint")
    else:
        raise ValueError("Unsupported bounded center engine")
    if balanced:
        if engine in ("persistence", "local_trend"):
            def predictor(histories):
                return baseline_predictions(histories, engine, cfg["max_horizon"])[..., 1]
        elif engine == "gru":
            def predictor(histories):
                return _trend_predict(model["model"], histories, prep)
        else:
            def predictor(histories):
                bf, bl = features_for(histories)
                with threadpool_limits(limits=2):
                    return np.exp(np.clip(model["model"][0].predict(bf).reshape(len(histories), cfg["max_horizon"]) + np.log(bl[:, None]), -80, 80))
        component = train_components(predictor)
        summary.update(best_rolling_train_loss=component["rolling_train_loss"],
                       best_reference_train_loss=component["reference_train_loss"],
                       best_training_loss=component["train_loss"])
    summary["quality_accepted"] = False
    _check_stop(should_stop)
    model["model_hash"] = _model_hash(model)
    return model


def _history_array(model, observed_history):
    if isinstance(observed_history, dict):
        if observed_history.get("status", "available") != "available":
            raise ValueError(observed_history["status"])
        x = observed_history.get("x", observed_history.get("history"))
    else:
        x = observed_history
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] != model["config"]["history_length"]:
        raise ValueError("insufficient_history or history configuration mismatch")
    if not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("invalid_observation")
    return x


def validate_model(model):
    if model["model_hash"] != _model_hash(model):
        raise ValueError("Frozen model/config/preprocessor was mutated")
    return model


def predict_details(frozen_model, observed_history):
    validate_model(frozen_model)
    x = _history_array(frozen_model, observed_history)
    engine = frozen_model["engine_id"]
    prep = frozen_model["preprocessing"]
    horizon = frozen_model["config"]["max_horizon"]
    if is_bounded_trend(frozen_model["config"]):
        from .corridor import decision_outputs
        if (x <= 0).any():
            raise ValueError("invalid_observation: bounded trend requires positive history")
        if engine in ("persistence", "local_trend"):
            center = baseline_predictions(x, engine, horizon)[..., 1]
        elif engine == "gru":
            center = _trend_predict(frozen_model["model"], x, prep)
        else:
            z, feature, level = _trend_design(x, prep)
            base = np.column_stack((z, feature))
            features = np.column_stack((np.repeat(base, horizon, axis=0), np.tile(np.arange(1, horizon + 1), len(x)) / horizon)).astype(np.float32)
            with threadpool_limits(limits=2):
                log_delta = frozen_model["model"][0].predict(features).reshape(len(x), horizon) if len(x) else np.empty((0, horizon))
            center = np.exp(np.clip(log_delta + np.log(level[:, None]), -80, 80))
        outputs = decision_outputs(center, frozen_model["config"])
        return {"outputs": outputs, "center": center, "lower": outputs[..., 0], "upper": outputs[..., 2],
                "output_kind": "decision_corridor", "coverage_guarantee": False, "nominal": None,
                "representability_guard": "log physical center [-80,80];not width clipping"}
    if engine in ("persistence", "local_trend"):
        raw = baseline_predictions(x, engine, horizon)
    elif engine == "gru":
        raw = _gru_predict(frozen_model["model"], x, prep)
    else:
        features = _features(x, prep, horizon=horizon)
        with threadpool_limits(limits=2):
            raw = np.stack([e.predict(features).reshape(len(x), horizon) for e in frozen_model["model"]], -1) * prep["target_scale"] if len(x) else np.empty((0, horizon, 3))
    crossing = (raw[..., 0] > raw[..., 1]) | (raw[..., 1] > raw[..., 2])
    quantiles = np.sort(np.maximum(raw, 0), axis=-1)
    if not np.isfinite(quantiles).all():
        raise RuntimeError("Nonfinite physical prediction")
    return {"quantiles": quantiles, "raw_quantiles_before_sort": raw,
            "raw_crossing": crossing, "raw_crossing_rate": float(crossing.mean()) if crossing.size else None}


def predict(frozen_model, observed_history):
    details = predict_details(frozen_model, observed_history)
    return details["outputs"] if is_bounded_trend(frozen_model["config"]) else details["quantiles"]


def save_model(model, directory):
    if model["model_hash"] != _model_hash(model):
        raise ValueError("Model was mutated after freezing")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "model_weights.joblib").exists():
        raise FileExistsError("Frozen model artifacts are immutable")
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    payload = buffer.getvalue()
    (directory / "model_weights.joblib").write_bytes(payload)
    (directory / "preprocessing.json").write_text(json.dumps(model["preprocessing"], indent=2))
    manifest = {"model_hash": model["model_hash"], "weights_sha256": hashlib.sha256(payload).hexdigest(),
                "preprocessing_hash": json_hash(model["preprocessing"]), "config": model["config"],
                "engine_id": model["engine_id"], "training_summary": model["training_summary"]}
    (directory / "model.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_model(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "model.json").read_text())
    payload = (directory / "model_weights.joblib").read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["weights_sha256"]:
        raise ValueError("Model artifact checksum mismatch")
    model = joblib.load(io.BytesIO(payload))
    preprocessing = json.loads((directory / "preprocessing.json").read_text())
    if json_hash(preprocessing) != manifest["preprocessing_hash"] or preprocessing != model["preprocessing"]:
        raise ValueError("Preprocessing checksum mismatch")
    if _model_hash(model) != manifest["model_hash"] or model["model_hash"] != manifest["model_hash"] or model["config"] != manifest["config"]:
        raise ValueError("Frozen model/config hash mismatch")
    return model
