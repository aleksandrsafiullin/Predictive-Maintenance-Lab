"""Causal multiscale trend + learned residual, with a frozen full direct grid.

Protocol v1 is separate from historical free-curve GRU artifacts. Width is a
decision budget, never a confidence claim. Training consumes Train/Validation.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from torch import nn

from pdm.long_forecast_data import HISTORY, MIN_HISTORY, digest

PROTOCOL = "causal_multiscale_residual_v1"
ENGINES = ("robust_trend", "trend_boosting", "trend_gru")


def config(horizon, cadence=60., **overrides):
    result = dict(protocol=PROTOCOL, horizon=int(horizon), cadence_s=float(cadence),
                  history_capacity=HISTORY, minimum_history=MIN_HISTORY, history_scales=[60, 120, 240],
                  width=.30, seed=21, hidden_size=64, epochs=40, patience=8, batch_size=64,
                  learning_rate=.001, threads=2, stride=30, correction_penalty=.002,
                  baseline_family="linear_v1",
                  selection="physical_unit_then_observed_regime_then_horizon_quarter_balanced_log_interval_and_shape_error",
                  future_inputs=False, coverage_guarantee=False)
    if overrides.keys() - result.keys():
        raise ValueError("Unknown long forecast configuration")
    result.update(overrides)
    if result["protocol"] != PROTOCOL or not 1 <= result["horizon"] <= 4096 or result["width"] != .30:
        raise ValueError("Unsupported long forecast contract")
    if result["history_capacity"] != 240 or result["minimum_history"] != 60 or result["history_scales"] != [60, 120, 240]:
        raise ValueError("Unsupported causal history contract")
    for key in ("seed", "hidden_size", "epochs", "patience", "batch_size", "threads", "stride"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(result["cadence_s"]) or result["cadence_s"] <= 0:
        raise ValueError("Invalid cadence")
    if result["baseline_family"] not in {"linear_v1", "acceleration_v2"}:
        raise ValueError("Unsupported baseline family")
    return result


def _robust_fit(values, degree=1):
    # Medians in disjoint 5-sample blocks suppress spikes without future access.
    groups = np.array_split(np.arange(len(values)), max(6, len(values) // 5))
    t = np.array([g.mean() - len(values) + 1 for g in groups]) / 60
    y = np.array([np.median(values[g]) for g in groups])
    design = np.column_stack([t ** k for k in range(degree + 1)])
    weights = np.ones(len(y))
    for _ in range(4):
        coef = np.linalg.lstsq(design * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
        error = y - design @ coef
        scale = max(1e-7, np.median(np.abs(error - np.median(error))) * 1.4826)
        weights = np.minimum(1., 1.5 * scale / np.maximum(np.abs(error), 1e-12))
    return coef, float(np.median(np.abs(y - design @ coef)))


def history_design(x, lengths, horizon, baseline_family="linear_v1"):
    """Pure function of admitted past values, availability and requested leads."""
    x, lengths = np.asarray(x), np.asarray(lengths)
    if x.ndim != 2 or x.shape[1] != HISTORY or len(lengths) != len(x):
        raise ValueError("History shape mismatch")
    leads = np.arange(1, horizon + 1) / 60
    features, bases, sequences = [], [], []
    for row, length in zip(x, lengths, strict=True):
        if not MIN_HISTORY <= int(length) <= HISTORY:
            raise ValueError("Invalid history availability")
        observed = row[-int(length):].astype(float)
        if not np.isfinite(observed).all() or (observed <= 0).any():
            raise ValueError("Invalid positive observations")
        level = float(np.median(observed[-5:]))
        log = np.log(observed / level)
        noise = max(1e-4, np.median(np.abs(np.diff(log) - np.median(np.diff(log)))) / np.sqrt(2))
        feature = [np.log(level), noise, observed[-1] / level - 1, length / HISTORY]
        candidates = []
        for scale in (60, 120, 240):
            available = min(scale, len(log))
            tail = log[-available:]
            lin, error = _robust_fit(tail)
            quad, qerror = _robust_fit(tail, 2)
            physical, physical_error = _robust_fit(observed[-available:] / level)
            feature.extend([lin[0], lin[1], quad[2], physical[1], error, qerror, available / scale])
            if scale > len(log):
                continue
            # Backtest exclusively within supplied history, ending at Now.
            ntest = min(20, available // 3)
            previous = tail[:-ntest]
            plog, _ = _robust_fit(previous)
            pphys, _ = _robust_fit(np.exp(previous))
            ahead = np.arange(1, ntest + 1) / 60
            log_error = np.median(np.abs(plog[0] + plog[1] * ahead - tail[-ntest:]))
            physical_path = physical[0] + physical[1] * leads
            past_phys = pphys[0] + pphys[1] * ahead
            phys_error = np.median(np.abs(np.log(np.maximum(past_phys, 1e-8)) - tail[-ntest:]))
            candidates.append((log_error, np.log(level) + lin[0] + lin[1] * leads))
            candidates.append((phys_error, np.log(level) + np.log(np.maximum(physical_path, 1e-8))))
            if baseline_family == "acceleration_v2":
                # Admit acceleration only when it is visible in past values and
                # improves both their fit and a withheld suffix of this history.
                quadratic, quadratic_error = _robust_fit(observed[-available:] / level, 2)
                past_quadratic, _ = _robust_fit(np.exp(previous), 2)
                check = past_quadratic[0] + past_quadratic[1] * ahead + past_quadratic[2] * ahead**2
                check_error = np.median(np.abs(np.log(np.maximum(check, 1e-8)) - tail[-ntest:]))
                if (quadratic[1] > 0 and quadratic[2] > 0 and
                        quadratic_error < .8 * physical_error and
                        check_error < .8 * min(log_error, phys_error)):
                    path = quadratic[0] + quadratic[1] * leads + quadratic[2] * leads**2
                    candidates.append((check_error, np.log(level) + np.log(np.maximum(path, 1e-8))))
        # Stable histories naturally prefer persistence, without forced upward motion.
        flat_error = np.median(np.abs(log[-20:] - np.median(log[-40:-20])))
        candidates.append((flat_error, np.full(horizon, np.log(level))))
        chosen = min(candidates, key=lambda item: item[0])[1]
        features.append(feature)
        bases.append(chosen)
        # 60 bins cover up to 240 past samples; availability is supplied separately.
        padded = np.zeros(HISTORY)
        padded[-length:] = log
        sequences.append(padded.reshape(60, 4).mean(1))
    return np.asarray(features, np.float32), np.asarray(bases, np.float32), np.asarray(sequences, np.float32)


def regime(features):
    # Declared from robust history slope relative to observed short-term noise.
    return features[:, 5] > np.maximum(.015, features[:, 1] * .5)


def target_weights(data, features):
    mask = data["mask"]
    groups = data["physical"]
    rising = regime(features)
    weights = np.zeros(mask.shape, np.float32)
    # Equal equipment, equal available observed regimes, equal supported quarters.
    for group in np.unique(groups):
        members = groups == group
        states = [state for state in (False, True) if (members & (rising == state)).any()]
        for state in states:
            selected = members & (rising == state)
            chunks = [chunk for chunk in np.array_split(np.arange(mask.shape[1]), 4) if len(chunk) and mask[selected][:, chunk].any()]
            for chunk in chunks:
                active = mask[selected][:, chunk]
                counts = active.sum(0)
                supported = counts > 0
                block = active / np.maximum(counts, 1) / supported.sum() / len(chunks) / len(states) / len(np.unique(groups))
                weights[np.ix_(selected, chunk)] = block
    return weights


class ResidualGRU(nn.Module):
    def __init__(self, hidden, feature_count, horizon):
        super().__init__()
        self.encoder = nn.GRU(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden + feature_count, 16)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        from scipy.interpolate import CubicSpline
        spline = CubicSpline(np.linspace(0, 1, 16), np.eye(16), bc_type="natural")
        self.register_buffer("basis", torch.as_tensor((spline(np.arange(1, horizon + 1) / horizon) - spline(0)).T, dtype=torch.float32))

    def forward(self, sequence, features, baseline):
        _, state = self.encoder(sequence.unsqueeze(-1))
        coefficients = self.head(torch.cat([state[-1], features], -1))
        residual = coefficients @ self.basis
        return baseline + residual, coefficients


def _log_predict(model, design):
    features, base, seq = design
    if model["engine"] == "robust_trend":
        return base
    features = (features - model["mean"]) / model["std"]
    if model["engine"] == "trend_boosting":
        h = base.shape[1]
        leads = np.tile(np.arange(1, h + 1) / h, len(base))
        inputs = np.column_stack([np.repeat(features, h, axis=0), leads, base.ravel()])
        with threadpool_limits(limits=2):
            return base + model["body"].predict(inputs).reshape(base.shape)
    result = []
    model["body"].eval()
    with torch.inference_mode():
        for start in range(0, len(base), 256):
            sl = slice(start, start + 256)
            values, _ = model["body"](torch.from_numpy(seq[sl]), torch.from_numpy(features[sl].astype(np.float32)), torch.from_numpy(base[sl]))
            result.append(values.numpy())
    return np.concatenate(result)


def predict(model, x, lengths, horizon=None):
    cfg = model["config"]
    horizon = cfg["horizon"] if horizon is None else horizon
    if not isinstance(horizon, int) or not 1 <= horizon <= cfg["horizon"]:
        raise ValueError("Requested horizon exceeds the frozen direct grid")
    log = _log_predict(model, history_design(x, lengths, cfg["horizon"], cfg.get("baseline_family", "linear_v1")))
    # Float32 finite-domain guard only; neither thresholds nor future facts clip it.
    center = np.exp(np.clip(log.astype(float), -80, 80))[:, :horizon]
    return np.stack([center * .85, center, center * 1.15], -1)


def objective(log_center, data, weights):
    target = np.log(np.maximum(data["y"], 1e-12))
    delta = log_center - target
    excess = np.maximum(delta + np.log(.85), 0) + np.maximum(-delta - np.log(1.15), 0)
    value = np.sum(weights * (np.abs(delta) + excess))
    # Compare smoothed consecutive 30-step blocks, rather than noisy derivatives.
    for start in range(0, data["horizon"] - 59, 30):
        a, b = slice(start, start + 30), slice(start + 30, start + 60)
        valid = data["mask"][:, b].all(1)
        if valid.any():
            shape = (log_center[:, b].mean(1) - log_center[:, a].mean(1)) - (target[:, b].mean(1) - target[:, a].mean(1))
            value += .2 * np.sum(weights[:, b].sum(1) * valid * np.abs(shape))
    return float(value)


def fit(train, validation, cfg, engine="trend_gru", report=None, stop=None):
    cfg = config(**cfg)
    if engine not in ENGINES or train.get("split") != "train" or validation.get("split") != "validation":
        raise ValueError("Training requires explicitly separated Train and Validation")
    if set(train["physical"]) & set(validation["physical"]):
        raise ValueError("Physical equipment overlaps Train/Validation")
    for data in (train, validation):
        if data["horizon"] != cfg["horizon"] or data["cadence_s"] != cfg["cadence_s"]:
            raise ValueError("Frozen horizon/cadence mismatch")
        if data["y"].shape != (len(data["x"]), cfg["horizon"]) or data["mask"].shape != data["y"].shape:
            raise ValueError("Target/mask shape mismatch")
        if (np.diff(data["mask"].astype(int), axis=1) > 0).any():
            raise ValueError("Target support must be a continuous prefix")
        known = data["y"][data["mask"]]
        if not len(known) or not np.isfinite(known).all() or (known <= 0).any():
            raise ValueError("Invalid supported training targets")
    if not train["mask"].any(0).all():
        raise ValueError("Every frozen horizon lead requires Train support")
    report = report or (lambda value: None)
    stop = stop or (lambda: False)
    td = history_design(train["x"], train["lengths"], cfg["horizon"], cfg["baseline_family"])
    vd = history_design(validation["x"], validation["lengths"], cfg["horizon"], cfg["baseline_family"])
    tw, vw = target_weights(train, td[0]), target_weights(validation, vd[0])
    model = dict(protocol=PROTOCOL, config=cfg, engine=engine, mean=td[0].mean(0), std=np.maximum(td[0].std(0), .001))
    baseline_loss = objective(vd[1], validation, vw)
    records = []
    correction_selected = False
    if engine == "robust_trend":
        model["body"] = None
    elif engine == "trend_boosting":
        # Sample leads deterministically with physical/regime/quarter weights.
        rng = np.random.default_rng(cfg["seed"])
        indices = rng.choice(tw.size, min(150000, np.count_nonzero(tw) * 2), p=tw.ravel() / tw.sum())
        rows, leads = np.unravel_index(indices, tw.shape)
        z = (td[0] - model["mean"]) / model["std"]
        inputs = np.column_stack([z[rows], (leads + 1) / cfg["horizon"], td[1][rows, leads]])
        target = np.log(train["y"][rows, leads]) - td[1][rows, leads]
        best, body = baseline_loss, None
        estimator = HistGradientBoostingRegressor(loss="absolute_error", max_iter=40, warm_start=True,
                                                  max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=2,
                                                  learning_rate=.075, early_stopping=False, random_state=cfg["seed"])
        for count in (40, 80, 120, 160):
            if stop():
                raise InterruptedError("Training cancelled")
            estimator.set_params(max_iter=count)
            with threadpool_limits(limits=cfg["threads"]):
                estimator.fit(inputs, target)
            model["body"] = estimator
            value = objective(_log_predict(model, vd), validation, vw)
            record = dict(iterations=count, validation_objective=value)
            records.append(record)
            report(record)
            if value < best:
                best, body = value, copy.deepcopy(estimator)
        if body is None:
            # A correction that cannot beat the baseline is never attached.
            model["engine"], model["body"] = "robust_trend", None
        else:
            model["body"] = body
            correction_selected = True
    else:
        torch.manual_seed(cfg["seed"])
        torch.set_num_threads(cfg["threads"])
        network = ResidualGRU(cfg["hidden_size"], td[0].shape[1], cfg["horizon"])
        optimizer = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"])
        z = torch.from_numpy(((td[0] - model["mean"]) / model["std"]).astype(np.float32))
        sequence, base = torch.from_numpy(td[2]), torch.from_numpy(td[1])
        target = torch.from_numpy(np.log(np.maximum(train["y"], 1e-12)))
        weights, mask = torch.from_numpy(tw), torch.from_numpy(train["mask"])
        rng = np.random.default_rng(cfg["seed"])
        best, best_epoch, best_state = baseline_loss, 0, copy.deepcopy(network.state_dict())
        model["body"] = network
        for epoch in range(1, cfg["epochs"] + 1):
            network.train()
            for ids in np.array_split(rng.permutation(len(z)), max(1, int(np.ceil(len(z) / cfg["batch_size"])))):
                if stop():
                    raise InterruptedError("Training cancelled")
                optimizer.zero_grad()
                log, coefficients = network(sequence[ids], z[ids], base[ids])
                delta = log - target[ids]
                excess = torch.relu(delta + np.log(.85)) + torch.relu(-delta - np.log(1.15))
                loss = (weights[ids] * (delta.abs() + excess)).sum()
                # Match the objective's blockwise future shape; only targets enter here.
                for start in range(0, cfg["horizon"] - 59, 30):
                    a, b = slice(start, start + 30), slice(start + 30, start + 60)
                    valid = mask[ids, b].all(1)
                    shape = (log[:, b].mean(1) - log[:, a].mean(1)) - (target[ids, b].mean(1) - target[ids, a].mean(1))
                    loss += .2 * (weights[ids, b].sum(1) * valid * shape.abs()).sum()
                loss += cfg["correction_penalty"] * coefficients.square().mean() * float(weights[ids].sum())
                loss.backward()
                if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                    raise FloatingPointError("Nonfinite residual training")
                nn.utils.clip_grad_norm_(network.parameters(), 1)
                optimizer.step()
            value = objective(_log_predict(model, vd), validation, vw)
            record = dict(epoch=epoch, validation_objective=value)
            records.append(record)
            report(record)
            if value < best - 1e-5:
                best, best_epoch, best_state = value, epoch, copy.deepcopy(network.state_dict())
            if epoch - best_epoch >= cfg["patience"]:
                break
        network.load_state_dict(best_state)
        network.eval()
        correction_selected = best_epoch > 0
    model["training"] = dict(baseline_validation_objective=baseline_loss,
                             validation_objective=objective(_log_predict(model, vd), validation, vw),
                             records=records, train_windows=len(train["x"]), validation_windows=len(validation["x"]),
                             learned_correction_selected=correction_selected,
                             fit_split="train", selection_split="validation", test_used=False)
    return model


def metrics(outputs, data, red=None):
    center, lo, hi = outputs[..., 1], outputs[..., 0], outputs[..., 2]
    y, mask = data["y"], data["mask"]
    features = history_design(data["x"], data["lengths"], data["horizon"])[0]
    weights = target_weights(data, features)
    inside = (y >= lo) & (y <= hi)
    error = np.abs(center - y)
    def aggregate(active):
        w = weights * active
        if not w.sum():
            return dict(status="unknown", known_targets=0)
        w /= w.sum()
        return dict(status="measured", point_coverage=float((w * inside).sum()),
                    center_mae=float((w * error).sum()), relative_bias=float((w * (center - y) / np.maximum(y, .001)).sum()),
                    known_targets=int(active.sum()), supported_units=len(set(data["physical"][active.any(1)])))
    result = aggregate(mask)
    result["aggregation"] = "equal physical equipment, observed regime and supported horizon quarter"
    result["unweighted_point_coverage"] = float(inside[mask].mean())
    result["quarters"] = []
    for j, chunk in enumerate(np.array_split(np.arange(data["horizon"]), 4)):
        active = np.zeros_like(mask)
        active[:, chunk] = mask[:, chunk]
        result["quarters"].append(dict(quarter=j + 1, **aggregate(active)))
    result["observed_growing"] = aggregate(mask & regime(features)[:, None])
    result["observed_stable_or_falling"] = aggregate(mask & ~regime(features)[:, None])
    result["red"] = aggregate(mask & (y >= red)) if red is not None else dict(status="unknown", known_targets=0)
    complete = mask.all(1)
    result["complete_paths"] = int(complete.sum())
    result["whole_path_coverage"] = float(inside[complete].all(1).mean()) if complete.any() else None
    result["whole_supported_path_coverage"] = float(((inside | ~mask).all(1)).mean())
    result["paths_with_90_percent_points"] = float((np.sum(inside & mask, 1) / mask.sum(1) >= .9).mean())
    result["relative_full_width"] = .30
    result["quality_accepted"] = False
    result["coverage_guarantee"] = False
    return result


def save(model, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, directory / "model.joblib")
    (directory / "config.json").write_text(json.dumps(model["config"], indent=2) + "\n")
    return digest(model["config"])
