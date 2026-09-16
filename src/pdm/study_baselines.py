"""Source-bound current-feature baselines, evaluated on the model cohort clock."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from pdm.data.prepare import load_processed
from pdm.forecasting import encode_rul_targets
from pdm.io_util import atomic_write_json
from pdm.losses import weibull_median_rul, weibull_nll_seconds
from pdm.preprocessing import fit_preprocessor
from pdm.training_protocol import window_weights
from pdm.training_readout import ALPHAS, solve_readout, training_folds
from pdm.windows import build_windows, filter_gap_params


def _design(encoded, windows):
    by_unit = {uid: g.sort_values("timestamp_s").reset_index(drop=True) for uid, g in encoded.groupby("unit_id")}
    rows = [by_unit[row.unit_id].iloc[int(row.end_index)] for row in windows.itertuples()]
    return pd.DataFrame(rows).reset_index(drop=True)


def fit_baseline(dataset_id, root):
    root.mkdir(parents=True, exist_ok=True)
    saved = root / "model.json"
    previous = None
    if saved.exists():
        previous = json.loads(saved.read_text())
        optimization = previous.get("optimization", {})
        if dataset_id != "filters" or optimization.get("success") or optimization.get("maxiter", 300) >= 2000:
            return previous
    from pdm.config import load_dataset_config

    bundle = load_processed(dataset_id)
    features, units, split = bundle["features"], bundle["units"], bundle["split"]
    cfg = load_dataset_config(dataset_id)
    gap = {}
    if dataset_id == "filters":
        k, interval = filter_gap_params(cfg)
        gap = {"gap_multiplier": k, "sampling_interval_s": interval}
    windows = build_windows(features, units, 20, dataset_id, **gap)
    train = windows[windows.unit_id.isin(split["train"])].reset_index(drop=True)
    prep, encoded = fit_preprocessor(dataset_id, features, units, split, cfg)
    rows = _design(encoded, train)
    z = np.column_stack([rows[prep.feature_names].to_numpy(float), np.ones(len(rows))])
    index = [(str(w.unit_id), 0, 0, float(w.target_rul_s), 0, 0) for w in train.itertuples()]
    weights = window_weights(index)
    result = {"dataset_id": dataset_id, "dataset_version": bundle["dataset_version"], "preprocessing": prep.to_dict(),
              "training_ids": split["train"], "feature_recipe": "base_v1", "history_length": 20}
    if dataset_id == "bearings":
        records = []
        for fold in training_folds(dataset_id, units, split["train"]):
            fp, fe = fit_preprocessor(dataset_id, features, units, {**split, **fold}, cfg)
            fit = train[train.unit_id.isin(fold["train"])]
            hold = train[train.unit_id.isin(fold["validation"])]
            xf, xh = _design(fe, fit), _design(fe, hold)
            zf = np.column_stack([xf[fp.feature_names], np.ones(len(xf))])
            zh = np.column_stack([xh[fp.feature_names], np.ones(len(xh))])
            fw = window_weights([(u, 0, 0, t, 0, 0) for u, t in zip(fit.unit_id, fit.target_rul_s, strict=True)])
            for transform in ("linear", "log1p"):
                for alpha in ALPHAS:
                    coef = solve_readout(zf, encode_rul_targets(fit.target_rul_s, fp.time_scale_s, transform), fw, alpha)
                    raw = zh @ coef
                    pred = np.maximum(raw, 0) * fp.time_scale_s if transform == "linear" else np.expm1(np.clip(raw, 0, 20)) * 60
                    score = pd.DataFrame({"unit_id": hold.unit_id.to_numpy(), "target": hold.target_rul_s.to_numpy(),
                                          "error": abs(pred - hold.target_rul_s.to_numpy())})
                    score = score[score.target.between(0, 1800, inclusive="right")]
                    records.append({"transform": transform, "alpha": alpha, "score": score.groupby("unit_id").error.mean().mean()})
        scores = pd.DataFrame(records).groupby(["transform", "alpha"]).score.mean()
        transform, alpha = scores.idxmin()
        coef = solve_readout(z, encode_rul_targets(train.target_rul_s, prep.time_scale_s, transform), weights, alpha)
        result.update(name="current_feature_ridge", transform=transform, alpha=float(alpha), coefficients=coef.tolist(), cv=records)
    else:
        x = torch.as_tensor(z, dtype=torch.float64)
        duration = torch.as_tensor(train.duration_s.to_numpy(copy=True), dtype=torch.float64)
        event = torch.as_tensor(train.event.to_numpy(copy=True), dtype=torch.float64)
        w = torch.as_tensor(weights / weights.sum(), dtype=torch.float64)

        def objective(vector):
            coef = torch.tensor(vector.reshape(z.shape[1], 2), dtype=torch.float64, requires_grad=True)
            raw = x @ coef
            parameters = torch.nn.functional.softplus(raw) + 1e-6
            per_window = weibull_nll_seconds(duration, event, parameters[:, 0] * prep.time_scale_s, parameters[:, 1])
            invalid = ~torch.isfinite(per_window)
            if invalid.any():
                row = train.iloc[int(torch.nonzero(invalid)[0, 0])]
                raise FloatingPointError(f"Nonfinite linear Weibull loss: {row.unit_id}, window ending at {row.timestamp_s}")
            loss = (per_window * w).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite linear Weibull baseline objective")
            loss.backward()
            gradient = coef.grad.detach().numpy().ravel()
            if not np.isfinite(gradient).all():
                raise FloatingPointError("Nonfinite linear Weibull baseline gradient")
            return float(loss.detach()), gradient

        initial = np.asarray(previous["coefficients"]).ravel() if previous else np.zeros(z.shape[1] * 2)
        fitted = minimize(objective, initial, method="L-BFGS-B", jac=True,
                          bounds=[(-5., 5.)] * (z.shape[1] * 2), options={"maxiter": 2000, "ftol": 1e-10})
        result.update(name="current_feature_weibull", coefficients=fitted.x.reshape(z.shape[1], 2).tolist(),
                      optimization={"success": bool(fitted.success), "message": str(fitted.message), "iterations": int(fitted.nit),
                                    "maxiter": 2000, "objective": float(fitted.fun)})
        if previous:
            result["previous_attempts"] = [*previous.get("previous_attempts", []),
                                            {"optimization": previous["optimization"], "coefficients": previous["coefficients"]}]
    atomic_write_json(saved, result)
    return result


def evaluate_baseline(model, reference_predictions, output_path):
    from pdm.benchmark import add_survival_scores
    from pdm.preprocessing import Preprocessor, apply_preprocessor

    ds = model["dataset_id"]
    bundle = load_processed(ds, model["dataset_version"])
    prep = Preprocessor.from_dict(model["preprocessing"])
    encoded = apply_preprocessor(prep, bundle["features"])
    reference = reference_predictions.copy()
    x = align_cohort_features(reference, encoded)
    if x[prep.feature_names].isna().any().any():
        raise ValueError("Baseline has missing model-cohort features")
    z = np.column_stack([x[prep.feature_names], np.ones(len(x))])
    raw = z @ np.asarray(model["coefficients"])
    if ds == "bearings":
        pred = np.maximum(raw, 0) * prep.time_scale_s if model["transform"] == "linear" else np.expm1(np.clip(raw, 0, 20)) * 60
    else:
        parameters = torch.nn.functional.softplus(torch.from_numpy(raw)) + 1e-6
        pred = weibull_median_rul(parameters[:, 0], parameters[:, 1], prep.time_scale_s).numpy()
        reference["weibull_scale_s"] = parameters[:, 0].numpy() * prep.time_scale_s
        reference["weibull_shape"] = parameters[:, 1].numpy()
    ready = ~reference.prediction_status.eq("Collecting history")
    if not np.isfinite(pred[ready]).all():
        raise FloatingPointError(f"Nonfinite {model['name']} prediction")
    reference["predicted_rul_s"] = np.where(ready, pred, np.nan)
    reference.loc[ready, "prediction_status"] = "ok"
    reference["source_reference_run_id"] = reference.get("run_id")
    reference["run_id"] = "baseline_" + model["name"]
    reference["raw_rul_s"] = reference.predicted_rul_s
    reference["absolute_error_s"] = abs(reference.predicted_rul_s - reference.actual_rul_s)
    reference["model"] = model["name"]
    for col in ("lower_rul_s", "upper_rul_s", "interval_method"):
        if col in reference:
            reference = reference.drop(columns=col)
    if ds == "filters":
        reference = add_survival_scores(reference, bundle["units"])
    reference.to_csv(output_path, index=False)
    return reference


def align_cohort_features(reference, encoded):
    """Join the same measurements despite CSV float round-trip noise.

    Microsecond keys are only a lookup aid. Ambiguous keys fail, and the actual
    timestamps must agree within 0.1 microseconds; this never fills a lost row
    with another measurement from the six-second (or longer) source clock.
    """
    left = reference[["unit_id", "timestamp_s"]].copy()
    right = encoded.rename(columns={"timestamp_s": "source_timestamp_s"}).copy()
    left["_time_key"] = left.timestamp_s.round(6)
    right["_time_key"] = right.source_timestamp_s.round(6)
    joined = left.merge(right, on=["unit_id", "_time_key"], how="left", validate="one_to_one")
    same_measurement = (joined.timestamp_s - joined.source_timestamp_s).abs() <= 1e-7
    if not same_measurement.all():
        row = joined.loc[~same_measurement].iloc[0]
        raise ValueError(f"Baseline has no exact source measurement: {row.unit_id}, {row.timestamp_s}")
    return joined.drop(columns=["_time_key", "source_timestamp_s"])
