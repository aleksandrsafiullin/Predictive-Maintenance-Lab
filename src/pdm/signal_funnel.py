"""Joint whole-vector OOF residual approximation and honest path diagnostics."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

VERSION = "joint_residual_v1"


def build_residual_bank(residuals, mask, groups, horizons_s):
    residuals = np.asarray(residuals, float)
    mask = np.asarray(mask, bool)
    groups = np.asarray(groups, str)
    horizons = np.asarray(horizons_s, float)
    if residuals.ndim != 2 or mask.shape != residuals.shape or len(groups) != len(residuals) or residuals.shape[1] != len(horizons):
        raise ValueError("Residual bank dimensions differ")
    if not len(horizons) or not np.isfinite(horizons).all() or np.any(horizons <= 0) or np.any(np.diff(horizons) <= 0):
        raise ValueError("Horizons must be positive and strictly increasing")
    complete = mask.all(axis=1) & np.isfinite(residuals).all(axis=1)
    retained = residuals[complete]
    physical = groups[complete]
    return {"residuals": retained, "groups": physical, "horizons_s": horizons,
            "version": VERSION, "status": "available" if len(np.unique(physical)) >= 2 else "insufficient_support",
            "physical_group_count": int(len(np.unique(physical))), "complete_vector_count": int(complete.sum()),
            "excluded_incomplete_count": int((~complete).sum()),
            "approximation": "unconditional_whole_oof_residual_vectors", "sampling": "physical_group_balanced"}


def save_sampler(path, bank):
    np.savez_compressed(path, residuals=bank["residuals"], groups=np.asarray(bank["groups"], str),
                        horizons_s=bank["horizons_s"], version=np.asarray(VERSION),
                        excluded_incomplete_count=np.asarray(bank.get("excluded_incomplete_count", 0)))
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_sampler(path, expected_sha256=None):
    raw = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Joint sampler artifact hash mismatch")
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["version"]) != VERSION:
            raise ValueError("Unsupported joint sampler version")
        bank = build_residual_bank(saved["residuals"], np.ones_like(saved["residuals"], bool), saved["groups"], saved["horizons_s"])
        bank["excluded_incomplete_count"] = int(saved["excluded_incomplete_count"])
        return bank


def sample_paths(point, current, bank, n_samples=256, seed=0, nonnegative=False):
    if bank["status"] != "available":
        raise ValueError("Joint sampler needs at least two physical Train groups with complete OOF vectors")
    point = np.asarray(point, float)
    if point.shape != (len(bank["horizons_s"]),) or not np.isfinite(point).all() or not np.isfinite(current):
        raise ValueError("Invalid forecast anchor or point vector")
    rng = np.random.default_rng(seed)
    groups = np.unique(bank["groups"])
    chosen = rng.choice(groups, size=n_samples)
    indices = [rng.choice(np.flatnonzero(bank["groups"] == g)) for g in chosen]
    future = point[None, :] + bank["residuals"][indices]
    if nonnegative:
        future = np.maximum(future, 0)
    return np.column_stack((np.full(n_samples, float(current)), future))


def empirical_band(paths, coverage=0.9):
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one")
    return tuple(np.quantile(np.asarray(paths, float), [(1-coverage)/2, (1+coverage)/2], axis=0))


def calibrate_band(lower, upper, actual, mask, groups, coverage=0.9):
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one")
    lower, upper, actual = [np.asarray(x, float) for x in (lower, upper, actual)]
    mask, groups = np.asarray(mask, bool), np.asarray(groups, str)
    if lower.shape != actual.shape or upper.shape != actual.shape or mask.shape != actual.shape or len(groups) != len(actual):
        raise ValueError("Calibration dimensions differ")
    complete = mask.all(axis=1) & np.isfinite(actual).all(axis=1) & np.isfinite(lower).all(axis=1) & np.isfinite(upper).all(axis=1)
    scores = {}
    # Every chosen origin of a physical group must be fully observed; otherwise
    # its group maximum is unknown and cannot enter finite-sample calibration.
    for group in np.unique(groups):
        rows = groups == group
        if np.all(complete[rows]):
            scores[str(group)] = float(np.maximum(0, np.maximum(lower[rows]-actual[rows], actual[rows]-upper[rows])).max())
    count = len(scores)
    rank = math.ceil((count+1)*coverage)
    available = count > 0 and rank <= count
    return {"version": VERSION, "status": "calibrated" if available else "insufficient_calibration",
            "coverage": float(coverage), "physical_group_count": count, "chosen_physical_group_count": len(np.unique(groups)),
            "unknown_group_count": len(np.unique(groups))-count, "complete_origin_count": int(complete.sum()),
            "unknown_origin_count": int((~complete).sum()), "finite_sample_rank": rank,
            "expansion": float(np.sort(list(scores.values()))[rank-1]) if available else None,
            "scores_by_physical_group": scores, "method": "group_max_over_origins_and_horizons",
            "coverage_guarantee": "requires_exchangeable_physical_groups" if available else False}


def apply_calibration(lower, upper, calibration, *, nonnegative=False):
    lower, upper = np.asarray(lower, float).copy(), np.asarray(upper, float).copy()
    if calibration.get("status") == "calibrated":
        lower[1:] -= float(calibration["expansion"])
        upper[1:] += float(calibration["expansion"])
    if nonnegative:
        lower[1:] = np.maximum(lower[1:], 0)
        upper[1:] = np.maximum(upper[1:], 0)
    return lower, upper


def red_corridor(paths, horizons_s, threshold, issued=0.0, coverage=0.9):
    paths = np.asarray(paths, float)
    times = np.r_[0., np.asarray(horizons_s, float)] + issued
    base = {"target": "first_forecast_grid_red_entry", "kind": "empirical_conditional_crossing",
            "calibration_status": "unvalidated", "coverage": coverage, "supported_through_s": float(times[-1]),
            "earliest_s": None, "latest_s": None, "conditional_bounds_s": None,
            "unconditional_bounds_s": None, "continuous_time_entry": "unresolved_between_grid_points",
            "grid_limitation": "Sparse forecast grid can miss crossings that return before the next grid point"}
    if threshold.get("status") != "available":
        return {**base, "status": "unknown_threshold", "probability_within_horizon": None, "no_entry_probability": None}
    hit = paths >= threshold["red"] if threshold["direction"] == "above" else paths <= threshold["red"]
    if hit[:, 0].all():
        return {**base, "status": "already_red", "probability_within_horizon": 1., "no_entry_probability": 0.,
                "earliest_s": issued, "latest_s": issued, "conditional_bounds_s": [issued, issued], "unconditional_bounds_s": [issued, issued]}
    valid = np.isfinite(paths).all(axis=1)
    if not valid.all():
        return {**base, "status": "insufficient_support", "probability_within_horizon": None, "no_entry_probability": None}
    crossed = hit.any(axis=1)
    probability = float(crossed.mean())
    indices = hit[crossed].argmax(axis=1)
    brackets = [[float(times[max(i-1, 0)]), float(times[i])] for i in indices]
    base.update(probability_within_horizon=probability, no_entry_probability=1-probability,
                crossing_brackets_s=brackets, sample_count=len(paths))
    if not brackets:
        return {**base, "status": "none_within_horizon", "unconditional_status": "open_upper_bound"}
    alpha = 1-coverage
    brackets = np.asarray(brackets)
    lo = float(np.quantile(brackets[:, 0], alpha/2, method="lower"))
    hi = float(np.quantile(brackets[:, 1], 1-alpha/2, method="higher"))
    base.update(status="empirical_conditional", earliest_s=lo, latest_s=hi, conditional_bounds_s=[lo, hi],
                conditioning="entry_within_supported_horizon")
    # Include no-entry mass at infinity without silently renormalizing.
    if 1-probability > alpha/2 + 1e-12:
        base["unconditional_status"] = "open_upper_bound"
    else:
        lows = np.r_[brackets[:, 0], np.full(len(paths)-len(brackets), np.inf)]
        highs = np.r_[brackets[:, 1], np.full(len(paths)-len(brackets), np.inf)]
        unconditional = [float(np.quantile(lows, alpha/2, method="lower")), float(np.quantile(highs, 1-alpha/2, method="higher"))]
        base.update(unconditional_bounds_s=unconditional if np.isfinite(unconditional).all() else None,
                    unconditional_status="empirical_unvalidated" if np.isfinite(unconditional).all() else "open_upper_bound")
    return base


def evaluate_paths(paths, actual, mask, groups, coverage=0.9, scale=None):
    """Group-balanced metrics; complete-path scores never fabricate targets."""
    if not 0 < coverage < 1:
        raise ValueError("Coverage must be between zero and one")
    paths, actual, mask, groups = np.asarray(paths, float), np.asarray(actual, float), np.asarray(mask, bool), np.asarray(groups, str)
    lower, upper = np.quantile(paths, [(1-coverage)/2, (1+coverage)/2], axis=1)
    valid = mask & np.isfinite(actual) & np.isfinite(lower) & np.isfinite(upper)
    complete = valid.all(axis=1)
    width = upper-lower
    score = width + 2/(1-coverage)*(np.maximum(lower-actual, 0)+np.maximum(actual-upper, 0))
    inside = (actual >= lower) & (actual <= upper)
    def balanced(values, admitted):
        means = [float(np.mean(values[(groups == g) & admitted])) for g in np.unique(groups) if np.any((groups == g) & admitted)]
        return float(np.mean(means)) if means else None
    per_horizon = [balanced(inside[:, h], valid[:, h]) for h in range(actual.shape[1])]
    interval = [balanced(score[:, h], valid[:, h]) for h in range(actual.shape[1])]
    widths = [balanced(width[:, h], valid[:, h]) for h in range(actual.shape[1])]
    energy = np.full(len(actual), np.nan)
    scale = np.ones(actual.shape[1]) if scale is None else np.asarray(scale, float)
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Energy scales must be finite positive Train-derived values")
    for i in np.flatnonzero(complete):
        x, y = paths[i]/scale, actual[i]/scale
        n = len(x)
        if n < 2:
            raise ValueError("Energy score needs at least two joint samples")
        pair_sum = sum(np.linalg.norm(chunk[:, None, :] - x[None, :, :], axis=2).sum()
                       for chunk in np.array_split(x, max(1, math.ceil(n / 32))))
        energy[i] = np.linalg.norm(x-y, axis=1).mean() - .5*pair_sum/(n*(n-1))
    return {"point_coverage_by_horizon": per_horizon, "interval_score_by_horizon": interval,
            "band_width_by_horizon": widths, "whole_path_coverage": balanced(inside.all(axis=1), complete),
            "joint_energy_score": balanced(energy, complete), "complete_path_count": int(complete.sum()),
            "unknown_path_count": int((~complete).sum()), "observed_target_count_by_horizon": valid.sum(axis=0).tolist(),
            "unknown_target_count_by_horizon": (~valid).sum(axis=0).tolist(), "physical_group_count": len(np.unique(groups)),
            "complete_path_physical_group_count": len(np.unique(groups[complete])),
            "observed_physical_group_count_by_horizon": [len(np.unique(groups[valid[:, h]])) for h in range(actual.shape[1])],
            "aggregation": "physical_group_balanced", "joint_score_population": "complete_observed_vectors_only"}
