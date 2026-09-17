"""Train-only candidate healthy references; independent from the training scaler."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.history import contiguous_history
from pdm.training_protocol import fingerprint


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def regime_key(row, dataset_id):
    if dataset_id == "bearings":
        return f"rpm={_number(row.get('rpm', np.nan)):g}|load={_number(row.get('load_kn', np.nan)):g}"
    # Saved engineering bins in raw source flow/feed units, no fitted test quantiles.
    return f"dust={row.get('dust', 'unknown')}|flow={np.floor(_number(row.get('flow_rate', np.nan)) / 50):g}|feed={np.floor(_number(row.get('dust_feed', np.nan)) / 100):g}"


def reference_features(dataset_id):
    if dataset_id == "bearings":
        return {"horizontal_rms": ("vibration_level", .001, "increase"),
                "vertical_rms": ("vibration_level", .001, "increase"),
                "horizontal_kurtosis": ("impulsiveness", .1, "increase"),
                "vertical_kurtosis": ("impulsiveness", .1, "increase")}
    return {"differential_pressure": ("pressure_level", 1., "increase")}


def fit_reference(features, train_ids, dataset_id, *, initial_count=20, max_relative_mad=.5):
    """Fixed initial contiguous stable prefix, never a fraction of eventual life."""
    spec = reference_features(dataset_id)
    references, selected, rejected = {}, [], []
    train = features[features.unit_id.isin(train_ids)]
    for uid, group in train.groupby("unit_id", sort=True):
        part = group.sort_values("timestamp_s").iloc[:initial_count]
        part = contiguous_history(part, dataset_id)
        if dataset_id == "filters" and (part.flow_rate.le(0).any() or part.differential_pressure.le(0).any()):
            rejected.append({"unit_id": str(uid), "reason": "startup_or_zero_flow_not_a_healthy_reference"})
            continue
        if len(part) != initial_count:
            rejected.append({"unit_id": str(uid), "reason": "initial_prefix_gap_or_short"})
            continue
        values = part[list(spec)].to_numpy(float)
        median = np.median(values, axis=0)
        relative = np.median(np.abs(values - median), axis=0) / np.maximum(np.abs(median), [v[1] for v in spec.values()])
        keys = [regime_key(r, dataset_id) for _, r in part.iterrows()]
        if not np.isfinite(values).all() or max(relative) > max_relative_mad or len(set(keys)) != 1:
            rejected.append({"unit_id": str(uid), "reason": "initial_prefix_not_stable_in_one_regime"})
            continue
        selected.append({"unit_id": str(uid), "start": float(part.timestamp_s.iloc[0]),
                         "end": float(part.timestamp_s.iloc[-1]), "count": len(part), "label": "candidate_healthy"})
        references.setdefault(keys[0], []).append(part)
    regimes = {}
    for key, parts in references.items():
        # Equal rows per object here by construction; never take test statistics.
        data = pd.concat(parts)
        regimes[key] = {name: {"expected": float(data[name].median()),
                        "scale": float(max(1.4826 * (data[name] - data[name].median()).abs().median(), eps)),
                        "epsilon": eps, "group": group, "direction": direction}
                        for name, (group, eps, direction) in spec.items()}
    known_regimes = sorted({regime_key(row, dataset_id) for _, row in train.iterrows()})
    manifest = {"model_regimes": known_regimes, "version": "robust_regime_reference_v1", "dataset_id": dataset_id,
                "status": "provisional" if regimes else "unavailable", "regimes": regimes,
                "selected_segments": selected, "rejected": rejected,
                "method": "fixed_initial_stable_train_prefix", "initial_count": initial_count,
                "max_relative_mad": max_relative_mad, "automatic_adaptation": False,
                "regime_convention": "rpm/load exact; filters dust + floor(flow/50) + floor(feed/100)",
                "fit_units": sorted(str(u) for u in train_ids)}
    manifest["reference_id"] = fingerprint(manifest)
    return manifest


def assess_normality(row, reference):
    key = regime_key(row, reference["dataset_id"])
    fitted = reference.get("regimes", {}).get(key)
    if not fitted:
        return {"model_applicability": "in_domain" if key in reference.get("model_regimes", []) else "out_of_domain" if reference.get("model_regimes", reference.get("regimes")) else "unverified",
                "reference_status": reference["status"], "score": None, "residuals": {},
                "reason_codes": ["normality_unavailable" if key in reference.get("model_regimes", []) or not reference.get("regimes") else "unknown_operating_regime"]}
    residuals, groups = {}, {}
    for name, stats in fitted.items():
        value = _number(row.get(name))
        if not np.isfinite(value):
            continue
        residual = float(value) - stats["expected"]
        z = residual / stats["scale"]
        score = max(z, 0.) if stats["direction"] == "increase" else abs(z)
        residuals[name] = {**stats, "value": float(value), "residual": residual, "z": z}
        groups[stats["group"]] = max(groups.get(stats["group"], 0.), score)
    return {"model_applicability": "in_domain", "reference_status": reference["status"],
            "score": max(groups.values()) if groups else None, "groups": groups,
            "residuals": residuals, "regime_key": key, "reason_codes": []}
