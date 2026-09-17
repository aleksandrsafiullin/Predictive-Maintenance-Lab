"""Versioned optimization rules; checkpoint selection and stopping are separate."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import torch

VERSION = "training_v2"


def protocol(dataset_id, *, mode="adaptive", learning_rate=0.001,
             sampling="unit_replacement", near_weight=0.0, feature_recipe="base_v1"):
    if mode not in {"adaptive", "diagnostic"}:
        raise ValueError("Unknown training mode")
    result = {"version": VERSION, "mode": mode, "max_epochs": 100,
            "min_epochs": 20 if mode == "adaptive" else 100,
            "patience": 20, "min_delta": 1.0 if dataset_id == "bearings" else 0.0001,
            "learning_rate": float(learning_rate), "sampling": sampling,
            "near_weight": float(near_weight), "feature_recipe": feature_recipe,
            "selection_metric": "near_30m_mae_s" if dataset_id == "bearings" else "survival_nll",
            # PyTorch reduces after bad_epochs > patience: 4 means the fifth
            # consecutive non-improving epoch triggers a reduction.
            "scheduler": {"factor": 0.5, "patience": 4 if mode == "adaptive" else 5, "min_lr": 0.00001}}
    if sampling == "full_pass":
        result["window_loss_reduction"] = "fixed_batch_denominator"
    return result


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def protocol_artifact(recipe, dataset_id, architecture, seed, *, training_identity=None):
    closed_form = dataset_id == "bearings" and architecture not in {"gru", "lstm"}
    objective = ("weighted_ridge_on_selected_linear_or_log1p_target" if closed_form else
                 "smooth_l1_on_time_scale_normalized_rul" if dataset_id == "bearings" else
                 "weibull_survival_nll_in_internal_seconds_float64")
    payload = {"protocol": recipe, "hash": fingerprint(recipe), "seed": int(seed), "objective": objective,
               "optimizer": ({"name": "closed_form_ridge", "selection": "train_only_grouped_cv"} if closed_form else
                             {"name": "AdamW", "batch_size": 32, "dropout": .1, "weight_decay": .0001, "gradient_norm_limit": 1.})}
    if training_identity:
        payload["training_identity"] = training_identity
    payload["artifact_hash"] = fingerprint(payload)
    return payload


def validate_protocol(value, dataset_id):
    result = dict(value)
    expected = protocol(dataset_id)
    if result.get("version") != VERSION or result.get("selection_metric") != expected["selection_metric"]:
        raise ValueError("Incompatible training protocol")
    if result.get("mode") not in {"diagnostic", "adaptive"}:
        raise ValueError("Unknown optimization mode")
    if result.get("sampling") not in {"unit_replacement", "full_pass"}:
        raise ValueError("Unknown sampling policy")
    if not 0 <= result.get("near_weight", 0) <= 1 or (dataset_id != "bearings" and result.get("near_weight", 0)):
        raise ValueError("Invalid near-event training weight")
    if result.get("feature_recipe") not in {"base_v1", "degradation_v1", "multiscale_trend_v1", "multiscale_no_age_v1", "multiscale_trend_v2", "multiscale_no_age_v2"}:
        raise ValueError("Unknown feature recipe")
    if not 0 < result["learning_rate"] or not 1 <= result["min_epochs"] <= result["max_epochs"]:
        raise ValueError("Invalid optimization limits")
    return result


@dataclass
class TrainingControl:
    reference_metric: float = float("inf")
    bad_epochs: int = 0
    last_epoch: int = 0
    stop_reason: str = "running"

    def update(self, metric, epoch, config):
        if not np.isfinite(metric):
            raise FloatingPointError("Nonfinite selection metric; checkpoint is ineligible")
        if metric < self.reference_metric - config["min_delta"]:
            self.reference_metric = float(metric)
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        self.last_epoch = int(epoch)
        if epoch >= config["max_epochs"]:
            self.stop_reason = "max_epochs"
        elif config["mode"] != "diagnostic" and epoch >= config["min_epochs"] and self.bad_epochs >= config["patience"]:
            self.stop_reason = "early_stopping"
        return self.stop_reason != "running"

    def state_dict(self):
        return asdict(self)


class FullPassSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, seed):
        self.n = len(dataset)
        self.seed = int(seed)
        self.epoch = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n

    def __iter__(self):
        if self.epoch is None:
            raise RuntimeError("set_epoch is required")
        return iter(np.random.RandomState(self.seed + self.epoch).permutation(self.n).tolist())


def window_weights(index, near_weight=0.0):
    """Mean weight is one; each unit contributes equally to the objective."""
    units = np.asarray([row[0] for row in index])
    targets = np.asarray([row[3] for row in index], dtype=float)
    weights = np.zeros(len(index), dtype=np.float64)
    ids = np.unique(units)
    for uid in ids:
        mask = units == uid
        close = mask & (targets > 0) & (targets <= 1800)
        mix = float(near_weight) if close.any() else 0.0
        weights[mask] = (1 - mix) / mask.sum()
        if mix:
            weights[close] += mix / close.sum()
    return weights * len(index) / max(len(ids), 1)


def weighted_batch_loss(losses, weights, config, batch_size=32):
    weighted = losses * weights
    if config.get("window_loss_reduction") == "fixed_batch_denominator":
        # Do not multiply the final short batch's window weights by 32 / len(batch).
        return weighted.sum() / batch_size
    return weighted.mean()
