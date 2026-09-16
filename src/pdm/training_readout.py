"""Train-only grouped readout selection with equal equipment weights."""
from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from pdm.forecasting import encode_rul_targets
from pdm.io_util import atomic_write_json, dump_yaml, load_yaml
from pdm.paths import runs_root
from pdm.preprocessing import fit_preprocessor
from pdm.training_protocol import window_weights
from pdm.windows import build_windows

ALPHAS = (0.0001, 0.001, 0.01, 0.1, 1.0)


def training_folds(dataset_id, units, train_ids):
    selected = units[units.unit_id.isin(train_ids)].copy()
    groups = []
    if dataset_id == "bearings":
        for instance in sorted(selected.instance.unique()):
            groups.append(sorted(selected.loc[selected.instance == instance, "unit_id"].tolist()))
    else:
        events = sorted(selected.loc[selected.event_observed.eq(1), "unit_id"].tolist())
        if len(events) < 2:
            raise ValueError("Need at least two independent train failures for grouped CV")
        groups = [[uid] for uid in events]
        censored = selected[selected.event_observed.eq(0)]
        # Balance each dust group, then total fold size, with deterministic ties.
        for _, cohort in censored.groupby("dust", sort=True):
            counts = [0] * len(groups)
            for uid in sorted(cohort.unit_id):
                index = min(range(len(groups)), key=lambda i: (counts[i], len(groups[i]), i))
                groups[index].append(uid)
                counts[index] += 1
    return [{"train": sorted(set(train_ids) - set(hold)), "validation": sorted(hold)} for hold in groups]


def solve_readout(z, target, weights, alpha):
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    matrix = np.asarray(z, float)
    gram = matrix.T @ (matrix * weights[:, None])
    penalty = np.eye(matrix.shape[1]) * alpha
    penalty[-1, -1] = 0
    return np.linalg.solve(gram + penalty, matrix.T @ (weights * target))


def design(dataset, states):
    inputs = np.stack([dataset.arrays[row[0]][row[2]] for row in dataset.index])
    return np.column_stack([states.numpy(), inputs, np.ones(len(dataset))])


def fit_window_readout(*, model, prep, cfg, processed, split, config, train_ds, val_ds,
                       caches, root, binding, rmeta, emit, log, should_stop):
    from pdm.train import UnitWindowDataset, _collate, _save_ckpt, _sync_run_checkpoint_hash
    from pdm.training_engine import score_model, state_cache

    device = next(model.parameters()).device
    candidates = [(transform, alpha) for transform in ("linear", "log1p") for alpha in ALPHAS]
    folds = training_folds("bearings", processed["units"], split["train"])
    records = []
    windows = build_windows(processed["features"], processed["units"], 20, "bearings")
    for number, fold in enumerate(folds):
        if should_stop and should_stop():
            emit("cancelled", stop_reason="user_stop")
            return {"status": "cancelled", "run_id": root.name, "dir": str(root)}
        fold_prep, encoded = fit_preprocessor("bearings", processed["features"], processed["units"], {**split, **fold}, cfg)
        datasets = [UnitWindowDataset(encoded[encoded.unit_id.isin(fold[part])], windows[windows.unit_id.isin(fold[part])],
                                     prep.feature_names, "bearings", fold_prep.time_scale_s) for part in ("train", "validation")]
        cached = [state_cache(model, ds, fold_prep, binding, runs_root() / "fixed_reservoir_cache", should_stop) for ds in datasets]
        z = design(datasets[0], cached[0])
        targets = np.array([row[3] for row in datasets[0].index])
        weights = window_weights(datasets[0].index, config["near_weight"])
        model.time_scale_s = fold_prep.time_scale_s
        for transform, alpha in candidates:
            model.rul_transform = transform
            w = solve_readout(z, encode_rul_targets(targets, fold_prep.time_scale_s, transform), weights, alpha)
            model.readout.load_ridge_vector(w)
            score = score_model(model, DataLoader(datasets[1], batch_size=32, collate_fn=_collate), "bearings", device, cached[1])
            records.append({"fold": number, "transform": transform, "alpha": alpha,
                            "near_30m_mae_s": score["selection_metric"], "held_out_units": fold["validation"]})
        log(f"Readout CV fold {number + 1}/{len(folds)} complete")
        emit("training", stage="readout_cv", fold=number + 1, folds=len(folds))
    scores = pd.DataFrame(records).groupby(["transform", "alpha"]).near_30m_mae_s.mean()
    transform, alpha = min(candidates, key=lambda item: (scores.loc[item], -item[1], item[0]))
    model.time_scale_s = prep.time_scale_s
    model.rul_transform = transform
    targets = np.asarray([row[3] for row in train_ds.index])
    w = solve_readout(design(train_ds, caches[0]), encode_rul_targets(targets, prep.time_scale_s, transform),
                      window_weights(train_ds.index, config["near_weight"]), alpha)
    model.readout.load_ridge_vector(w)
    train = score_model(model, DataLoader(train_ds, batch_size=32, collate_fn=_collate), "bearings", device, caches[0])
    val = score_model(model, DataLoader(val_ds, batch_size=32, collate_fn=_collate), "bearings", device, caches[1])
    metric = val["selection_metric"]
    cfg["model"]["reservoir"].update(ridge_alpha=float(alpha), rul_transform=transform, rul_reference_s=60.0)
    saved_cfg = load_yaml(root / "config.yaml")
    saved_cfg["model"] = cfg["model"]
    dump_yaml(root / "config.yaml", saved_cfg)
    for name in ("best.pt", "last.pt"):
        _save_ckpt(root / name, model, None, 1, 1, metric, cfg["model"], prep, split,
                   "bearings", "rul", False, fingerprint=binding, reservoir_meta=rmeta)
    _sync_run_checkpoint_hash(root)
    atomic_write_json(root / "readout_selection.json", {"folds": folds, "candidates": records,
                     "selected_transform": transform, "selected_alpha": float(alpha),
                     "metric": "near_30m_mae_s", "test_used": False})
    metrics = {"best_epoch": 1, "best_metric": metric, "selection_metric_name": "near_30m_mae_s",
               "selection_metric_unit": "seconds", "selection_metric_label": "near_30m_mae_s",
               "last": val, "last_train": train, "n_train_windows": len(train_ds), "n_val_windows": len(val_ds),
               "training_protocol": config, "smoke": False}
    atomic_write_json(root / "validation_metrics.json", metrics)
    pd.DataFrame([{"epoch": 1, "train_metric": train["selection_metric"], "val_metric": metric,
                   "negative_raw_fraction": val["negative_raw_fraction"], "zero_fraction": val["zero_fraction"],
                   "stop_reason": "closed_form_train_cv", "n_eligible_windows": len(train_ds),
                   "n_unique_sampled_windows": len(train_ds)}]).to_csv(root / "training_history.csv", index=False)
    emit("completed", epoch=1, best_epoch=1, best_metric=metric, stop_reason="closed_form_train_cv")
    return {"status": "completed", "run_id": root.name, "dir": str(root), "best_metric": metric}
