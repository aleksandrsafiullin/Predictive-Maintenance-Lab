"""Saved learned joint trajectories and RED-entry mixture, selected on Validation.

Trees and the complete connectome remain causal encoders. Their probabilistic
readout is optimized on the same joint path/event objective as recurrent models.
Validation selects weights; this is not a distribution-free coverage guarantee.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from numbers import Integral, Real
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from pdm.io_util import atomic_write_json, sha256_file
from pdm.models.signal_distribution import SignalDistribution
from pdm.models.signal_full_cns import build_signal_full_cns

MODE = "learned_joint_trajectories"


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _check(stop):
    if stop and stop():
        raise InterruptedError("Learned trajectory training cancelled")


def _post_factor_smoothness_weight(config):
    value = config.get("post_factor_smoothness_weight", 0.0)
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError("post_factor_smoothness_weight must be finite and nonnegative")
    if value > 0 and (
        config.get("phase_covariance", "shared") != "separate"
        or config.get("path_distribution", "single") != "single"
    ):
        raise ValueError("Positive post_factor_smoothness_weight requires separate covariance and single paths")
    return float(value)


def _path_band_geometry(config):
    value = config.get("path_band_geometry", "observed_prefix")
    if not isinstance(value, str) or value not in {"observed_prefix", "issued_prefix"}:
        raise ValueError("Invalid path_band_geometry")
    return value


def _red_corridor_objective_settings(config):
    """Validate the optional finite-S decision prior and its physical time scale."""
    values = {}
    for key, default in (("red_corridor_width_weight", 0.0),
                         ("red_corridor_miss_weight", 0.0),
                         ("red_corridor_scale_s", 1800.0)):
        value = config.get(key, default)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0
                or (key == "red_corridor_scale_s" and value == 0)):
            raise ValueError(f"{key} must be finite, nonboolean and "
                             + ("positive" if key == "red_corridor_scale_s" else "nonnegative"))
        values[key] = float(value)
    values["enabled"] = any(values[k] > 0 for k in (
        "red_corridor_width_weight", "red_corridor_miss_weight"))
    if values["enabled"]:
        grid = np.asarray(config.get("horizons_s", []), dtype=float)
        if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
                or grid[0] <= 0
                or not np.allclose(grid, np.arange(1, len(grid) + 1) * grid[0])):
            raise ValueError("RED corridor objective requires a dense positive saved cadence grid")
        samples = config.get("training_samples", 32)
        if (isinstance(samples, (bool, np.bool_)) or not isinstance(samples, Integral)
                or samples < 2):
            raise ValueError("RED corridor objective requires integer training_samples>=2")
        values["cadence_s"] = float(grid[0])
        values["scale_steps"] = values["red_corridor_scale_s"] / grid[0]
        if not np.isfinite(values["scale_steps"]) or values["scale_steps"] <= 0:
            raise ValueError("RED corridor objective step scale must be finite and positive")
    return values


def _red_finite_bound_weight(config):
    """Validate the optional positive-only finite-upper confidence prior."""
    value = config.get("red_finite_bound_weight", 0.0)
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
            or not np.isfinite(value) or value < 0):
        raise ValueError("red_finite_bound_weight must be finite, nonboolean and nonnegative")
    if value > 0:
        coverage = config.get("nominal_coverage", .9)
        if (isinstance(coverage, (bool, np.bool_)) or not isinstance(coverage, Real)
                or not np.isfinite(coverage) or not 0 < coverage < 1
                or (1 - coverage) / 2 <= 2e-8):
            raise ValueError("Positive finite-bound weight requires coverage with alpha>2e-8")
    return float(value)


def _event_cdf_objective_settings(config):
    """Validate the optional known-CDF auxiliary and fixed physical time scale."""
    values = {}
    for key, default in (("event_cdf_weight", 0.0), ("event_cdf_scale_s", 1800.0)):
        value = config.get(key, default)
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                or not np.isfinite(value) or value < 0
                or (key == "event_cdf_scale_s" and value == 0)):
            raise ValueError(f"{key} must be finite, nonboolean and "
                             + ("positive" if key == "event_cdf_scale_s" else "nonnegative"))
        values[key] = float(value)
    values["enabled"] = values["event_cdf_weight"] > 0
    if values["enabled"]:
        grid = np.asarray(config.get("horizons_s", []), dtype=float)
        if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
                or grid[0] <= 0
                or not np.allclose(grid, np.arange(1, len(grid) + 1) * grid[0])):
            raise ValueError("Known-CDF objective requires a dense positive saved cadence grid")
        values["cadence_s"] = float(grid[0])
        values["scale_steps"] = values["event_cdf_scale_s"] / grid[0]
        if not np.isfinite(values["scale_steps"]) or values["scale_steps"] <= 0:
            raise ValueError("Known-CDF objective step scale must be finite and positive")
    return values


def _path_objective_prefix_lengths(config, key="path_objective_horizons_s"):
    """Resolve frozen seconds to exact prefixes of the saved dense path grid."""
    requested = config.get(key)
    if requested is None:
        return None
    if not isinstance(requested, (list, tuple)) or not requested:
        raise ValueError(f"{key} must be a nonempty ordered list or None")
    grid = np.asarray(config["horizons_s"], dtype=float)
    lengths = []
    previous = 0.0
    for value in requested:
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value <= previous
        ):
            raise ValueError(f"{key} must be finite, positive, unique and increasing")
        match = np.flatnonzero(grid == value)
        if len(match) != 1:
            raise ValueError(f"{key} must exactly match the saved dense horizon grid")
        lengths.append(int(match[0]) + 1)
        previous = value
    return tuple(lengths)


def learned_params(engine_id, params, features):
    if engine_id not in {"gru", "lstm", "quantile_boosting", "full_cns"}:
        raise ValueError("Unsupported learned trajectory engine")
    raw = dict(params or {})
    defaults = dict(
        history_length=8,
        epochs=150,
        hidden_size=128,
        num_layers=2,
        rank=6,
        batch_size=16,
        batch_sampling="row_permutation",
        seed=42,
        learning_rate=0.001,
        event_head_learning_rate_multiplier=1.0,
        weight_decay=0.0001,
        patience=20,
        min_delta=0.00001,
        dropout=0.1,
        nominal_coverage=0.9,
        path_samples=256,
        training_samples=32,
        max_iter=100,
        max_windows_per_unit=None,
        energy_weight=1.0,
        width_weight=0.05,
        miss_weight=2.0,
        event_weight=1.0,
        phase_weight=0.5,
        post_factor_smoothness_weight=0.0,
        path_band_geometry="observed_prefix",
        red_corridor_width_weight=0.0,
        red_corridor_miss_weight=0.0,
        red_corridor_scale_s=1800.0,
        red_finite_bound_weight=0.0,
        event_cdf_weight=0.0,
        event_cdf_scale_s=1800.0,
        tree_horizons=8,
        cpu_threads=4,
        event_location="log",
        event_distribution="lognormal_cure",
        phase_covariance="shared",
        path_distribution="single",
        recurrent_context_mode="fixed",
        max_history_length=None,
        sensor_feature_mode="absolute",
        path_objective_horizons_s=None,
        event_objective_horizons_s=None,
    )
    allowed = set(defaults) | {
        "horizons_s",
        "forecast_mode",
        "residual_forecast",
        "cv_folds",
        "cadence_s",
        "target_tolerance_s",
    }
    if set(raw) - allowed:
        raise ValueError(f"Unknown learned parameters: {sorted(set(raw) - allowed)}")
    out = {key: raw.get(key, val) for key, val in defaults.items()}
    out["event_head_learning_rate_multiplier"] = _event_head_rate_multiplier(out)
    if not isinstance(out["batch_sampling"], str) or out["batch_sampling"] not in {
        "row_permutation", "physical_group", "physical_event_stratified"
    }:
        raise ValueError("Invalid batch_sampling")
    if out["event_location"] not in {"softplus", "log"}:
        raise ValueError("Invalid event_location")
    if not isinstance(out["event_distribution"], str) or out["event_distribution"] not in {
        "lognormal_cure", "finite_horizon_mixture"
    }:
        raise ValueError("Invalid event_distribution")
    if not isinstance(out["phase_covariance"], str) or out["phase_covariance"] not in {"shared", "separate"}:
        raise ValueError("Invalid phase_covariance")
    if not isinstance(out["path_distribution"], str) or out["path_distribution"] not in {
        "single", "coupled_timing_mixture"
    }:
        raise ValueError("Invalid path_distribution")
    if (out["path_distribution"] == "coupled_timing_mixture"
            and out["event_distribution"] != "finite_horizon_mixture"):
        raise ValueError("Coupled path distribution requires finite_horizon_mixture")
    out["post_factor_smoothness_weight"] = _post_factor_smoothness_weight(out)
    out["path_band_geometry"] = _path_band_geometry(out)
    if not isinstance(out["recurrent_context_mode"], str) or out["recurrent_context_mode"] not in {
        "fixed", "variable_causal"
    }:
        raise ValueError("Invalid recurrent_context_mode")
    if out["recurrent_context_mode"] == "variable_causal" and engine_id not in {"gru", "lstm"}:
        raise ValueError("Variable causal context supports GRU/LSTM only")
    if out["sensor_feature_mode"] not in {"absolute", "baseline_relative", "combined"}:
        raise ValueError("Invalid sensor_feature_mode")
    cadence = []
    durations = []
    for _, unit in features.groupby("unit_id"):
        unit = unit.sort_values("timestamp_s")
        dt = np.diff(unit.timestamp_s.to_numpy(float))
        gap = unit.gap_before.fillna(False).to_numpy(bool)[1:]
        good = np.isfinite(dt) & (dt > 0) & ~gap
        cadence.extend(dt[good].tolist())
        durations.append(float(dt[good].sum()))
    if not cadence:
        raise ValueError("Training clocks have no continuous cadence")
    step = float(np.median(cadence))
    horizon = np.arange(1, max(1, int(np.ceil(np.mean(durations) / step - 1e-9))) + 1) * step
    out["horizons_s"] = [float(h) for h in raw.get("horizons_s", horizon)]
    hs = np.asarray(out["horizons_s"])
    if (
        not len(hs)
        or len(hs) > 4096
        or not np.isfinite(hs).all()
        or not np.allclose(hs, np.arange(1, len(hs) + 1) * step)
    ):
        raise ValueError(
            "Learned trajectories require a dense Training-cadence horizon grid of 1–4096 steps"
        )
    red_settings = _red_corridor_objective_settings(out)
    for key in ("red_corridor_width_weight", "red_corridor_miss_weight", "red_corridor_scale_s"):
        out[key] = red_settings[key]
    out["red_finite_bound_weight"] = _red_finite_bound_weight(out)
    cdf_settings = _event_cdf_objective_settings(out)
    for key in ("event_cdf_weight", "event_cdf_scale_s"):
        out[key] = cdf_settings[key]
    _path_objective_prefix_lengths(out)
    if out["path_objective_horizons_s"] is not None:
        out["path_objective_horizons_s"] = [float(h) for h in out["path_objective_horizons_s"]]
    _path_objective_prefix_lengths(out, "event_objective_horizons_s")
    if out["event_objective_horizons_s"] is not None:
        out["event_objective_horizons_s"] = [float(h) for h in out["event_objective_horizons_s"]]
    _event_stratified_sampling_settings(out)
    for key, lo, hi in (
        ("history_length", 2, 256),
        ("epochs", 1, 2000),
        ("hidden_size", 16, 1024),
        ("num_layers", 1, 8),
        ("rank", 1, 64),
        ("batch_size", 1, 4096),
        ("patience", 1, 500),
        ("path_samples", 16, 10000),
        ("training_samples", 4, 1024),
        ("max_iter", 2, 2000),
        ("tree_horizons", 1, 32),
        ("cpu_threads", 1, 32),
    ):
        value = out[key]
        if isinstance(value, bool) or int(value) != value or not lo <= int(value) <= hi:
            raise ValueError(f"Invalid {key}")
        out[key] = int(value)
    maximum = out["max_history_length"]
    if out["recurrent_context_mode"] == "fixed":
        if maximum is not None:
            raise ValueError("Fixed recurrent context requires max_history_length=None")
    elif (isinstance(maximum, (bool, np.bool_)) or not isinstance(maximum, Integral)
          or not out["history_length"] <= maximum <= 256):
        raise ValueError("Variable causal context requires integer max_history_length>=history_length")
    else:
        out["max_history_length"] = int(maximum)
    for key in (
        "energy_weight",
        "width_weight",
        "miss_weight",
        "event_weight",
        "phase_weight",
        "learning_rate",
    ):
        out[key] = float(out[key])
        if not np.isfinite(out[key]) or out[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    for key in ("weight_decay", "min_delta", "dropout", "nominal_coverage"):
        out[key] = float(out[key])
        if not np.isfinite(out[key]):
            raise ValueError(f"Invalid {key}")
    if (
        not 0 <= out["dropout"] < 1
        or not 0 < out["nominal_coverage"] < 1
        or out["weight_decay"] < 0
        or out["min_delta"] < 0
    ):
        raise ValueError("Invalid optimizer or coverage parameter")
    cap = out["max_windows_per_unit"]
    if cap is not None and (isinstance(cap, bool) or int(cap) != cap or cap < 1):
        raise ValueError("Window cap must be positive integer or None")
    out.update(forecast_mode=MODE, target_tolerance_s=min(step * 0.01, 0.01), cadence_s=step)
    _optimizer_rate_metadata(out)
    return out



_EVENT_SAMPLING_STRATA = (
    "known_entry_inside_prefix", "known_no_entry_through_prefix",
    "partial_or_unknown_at_risk", "already_red",
)


def _event_stratified_sampling_settings(config):
    """Static, reloadable policy; never consult outcome frames or consume RNG."""
    if config.get("batch_sampling", "row_permutation") != "physical_event_stratified":
        return None
    grid = np.asarray(config.get("horizons_s", []), dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or grid[0] <= 0
            or not np.allclose(grid, np.arange(1, len(grid) + 1) * grid[0])):
        raise ValueError("Event stratified sampling requires a dense finite positive saved grid")
    prefixes = _path_objective_prefix_lengths(config, "event_objective_horizons_s")
    if prefixes is None:
        raise ValueError("Event stratified sampling requires explicit event_objective_horizons_s")
    prefix = prefixes[0]
    return {
        "mode": "physical_event_stratified", "version": 1,
        "prefix_rule": "first_explicit_resolved_event_objective_prefix",
        "prefix_length": prefix, "prefix_horizon_s": float(grid[prefix - 1]),
        "cadence_s": float(grid[0]), "horizons_s": grid.tolist(),
        "grid_sha256": _digest(grid.tolist()), "strata": list(_EVENT_SAMPLING_STRATA),
        "canonicalization": "known_cdf_timing_loss_A_known_v1_observed_priority_contiguous_censor",
        "partition": "Train_only_once_no_Validation_or_Test_plan",
        "replacement": True,
        "group_slot_policy": "randomized_balanced_uniform_remainder_then_slot_permutation",
        "stratum_slot_policy": "conditional_group_randomized_balanced_nonempty_uniform_remainder_then_slot_permutation",
        "row_policy": "uniform_with_replacement_within_group_stratum",
        "row_probability": "pi_i=1/(G*C_g*n_gs)",
        "term_reduction": "batch_mean(original_full_frame_term_weight_i/pi_i*row_loss_i)",
        "gradient_scale": 1.0, "epoch_report_scale": "B/N",
        "draw_budget": "N_draws_ceil(N/batch_size)_steps_array_split_batch_sizes",
        "mc_seed_policy": "seed+epoch*10000+int(idx[0])_unchanged",
        "term_populations": "unchanged_full_frame_objective_weights_at_risk_weights_and_optional_all_origin_weights",
        "independent_events": "origins_and_recrossings_are_not_independent_physical_events",
        "stochastic_gradient_equality_proven": False, "variance_reduction_proven": False,
    }


def _event_stratified_plan(frame, prefix_length):
    """Canonical optimizer-only evidence. Every Train origin has exactly one stratum."""
    from pdm.trajectory_objectives import known_cdf_timing_loss

    x = np.asarray(frame["x"])
    if x.ndim != 3 or not len(x):
        raise ValueError("Event stratified sampling requires nonempty [N,T,F] x")
    n = len(x)
    mask = np.asarray(frame["mask"])
    if mask.ndim != 2 or len(mask) != n or mask.shape[1] < 1:
        raise ValueError("Event stratified sampling requires [N,H] mask")
    h = mask.shape[1]
    for key in ("current", "red_threshold", "event_observed", "no_entry_prefix", "physical_unit_id"):
        if np.asarray(frame[key]).shape != (n,):
            raise ValueError(f"Event stratified sampling invalid {key} dimension")
    current, threshold = np.asarray(frame["current"], float), np.asarray(frame["red_threshold"], float)
    if not np.isfinite(current).all() or not np.isfinite(threshold).all() or (threshold <= 0).any():
        raise ValueError("Event stratified sampling requires finite current and positive finite thresholds")
    evidence = known_cdf_timing_loss(
        torch.zeros((n, h + 1)), prefix_length=prefix_length,
        already_red=torch.as_tensor(current >= threshold),
        event_allowed_mask=torch.as_tensor(np.asarray(frame["event_allowed"])),
        event_observed_mask=torch.as_tensor(np.asarray(frame["event_observed"])),
        no_entry_prefix=torch.as_tensor(np.asarray(frame["no_entry_prefix"])),
    )
    allowed, known = evidence["A"].numpy(), evidence["known"].numpy()
    risk = current < threshold
    row_stratum = np.full(n, 2, dtype=int)
    row_stratum[risk & known & ~allowed[:, prefix_length:].any(1)] = 0
    row_stratum[risk & known & ~allowed[:, :prefix_length].any(1)] = 1
    row_stratum[~risk] = 3
    groups, membership = [], {}
    row_group = np.empty(n, dtype=int)
    for i, name in enumerate(frame["physical_unit_id"]):
        if name is None or not str(name) or (isinstance(name, Real) and not np.isfinite(name)):
            raise ValueError("Event stratified sampling invalid physical membership")
        name = str(name)
        if name not in membership:
            membership[name] = len(groups)
            groups.append(name)
        row_group[i] = membership[name]
    members, counts = [], []
    pi = np.empty(n, dtype=np.float64)
    for g, name in enumerate(groups):
        strata = [np.flatnonzero((row_group == g) & (row_stratum == k)) for k in range(4)]
        nonempty = [rows for rows in strata if len(rows)]
        members.append(nonempty)
        for rows in nonempty:
            pi[rows] = 1.0 / (len(groups) * len(nonempty) * len(rows))
        counts.append({"physical_unit_id": name, "origin_count": int((row_group == g).sum()),
                       "stratum_counts": {key: len(rows) for key, rows in zip(_EVENT_SAMPLING_STRATA, strata)}})
    if not np.isfinite(pi).all() or not (pi > 0).all() or not np.isclose(pi.sum(), 1):
        raise ValueError("Event stratified sampling invalid probabilities")
    return {"n_rows": n, "prefix_length": int(prefix_length), "strata": list(_EVENT_SAMPLING_STRATA),
            "row_stratum": row_stratum, "row_group": row_group, "groups": groups,
            "members": members, "probabilities": pi,
            "counts": {"groups": counts, "total_origin_count": n, "physical_group_count": len(groups),
                       "stratum_origin_counts": {key: int((row_stratum == k).sum())
                                                 for k, key in enumerate(_EVENT_SAMPLING_STRATA)}}}



def _validate_sampling_train_metadata(value, config=None, scaler=None):
    """Check saved diagnostics without reading Train data at reload."""
    def count(v):
        return isinstance(v, int) and not isinstance(v, bool) and v >= 0
    if not isinstance(value, dict) or set(value) != {
        "groups", "total_origin_count", "physical_group_count", "stratum_origin_counts", "term_weight_sha256"
    }:
        raise ValueError("Learned optimizer sampling Train metadata missing or malformed")
    groups = value["groups"]
    totals = value["stratum_origin_counts"]
    if (not isinstance(groups, list) or not groups or not count(value["total_origin_count"])
            or value["total_origin_count"] < 1 or not count(value["physical_group_count"])
            or value["physical_group_count"] != len(groups)
            or not isinstance(totals, dict) or set(totals) != set(_EVENT_SAMPLING_STRATA)
            or not all(count(v) for v in totals.values())):
        raise ValueError("Learned optimizer sampling Train counts invalid")
    names, sums = set(), {k: 0 for k in _EVENT_SAMPLING_STRATA}
    for group in groups:
        if not isinstance(group, dict) or set(group) != {"physical_unit_id", "origin_count", "stratum_counts"}:
            raise ValueError("Learned optimizer sampling Train group malformed")
        name, counts = group["physical_unit_id"], group["stratum_counts"]
        if (not isinstance(name, str) or not name or name in names
                or not count(group["origin_count"]) or group["origin_count"] < 1
                or not isinstance(counts, dict) or set(counts) != set(sums)
                or not all(count(v) for v in counts.values())
                or sum(counts.values()) != group["origin_count"]):
            raise ValueError("Learned optimizer sampling Train group counts invalid")
        names.add(name)
        for key in sums:
            sums[key] += counts[key]
    hashes = value["term_weight_sha256"]
    if scaler is not None and sorted(names) != sorted(str(name) for name in scaler["fit_physical_groups"]):
        raise ValueError("Learned optimizer sampling physical groups differ from scaler")
    if config is not None:
        expected_terms = {"energy", "width", "miss", "event", "phase_energy"}
        if _post_factor_smoothness_weight(config) > 0:
            expected_terms.add("post_factor_smoothness")
        if _red_corridor_objective_settings(config)["enabled"]:
            expected_terms.update(("red_corridor_width", "red_corridor_miss"))
        if _red_finite_bound_weight(config) > 0:
            expected_terms.add("red_finite_bound")
        if _event_cdf_objective_settings(config)["enabled"]:
            expected_terms.add("event_cdf")
        if not isinstance(hashes, dict) or set(hashes) != expected_terms:
            raise ValueError("Learned optimizer sampling term weight hash populations differ from config")
    if sums != totals or sum(sums.values()) != value["total_origin_count"]:
        raise ValueError("Learned optimizer sampling Train totals differ")
    if (not isinstance(hashes, dict) or not {"energy", "width", "miss", "event", "phase_energy"}.issubset(hashes)
            or not all(isinstance(k, str) and isinstance(v, str) and len(v) == 64
                       and all(c in "0123456789abcdef" for c in v) for k, v in hashes.items())):
        raise ValueError("Learned optimizer sampling term weight hashes invalid")

def _event_stratified_batches(frame, batch_size, rng, plan):
    n = len(frame["x"])
    if plan is None or plan["n_rows"] != n:
        raise ValueError("Event stratified sampling requires its matching Train plan")
    rows, pi = plan["members"], plan["probabilities"]
    if (not rows or any(not group or any(len(r) == 0 for r in group) for group in rows)
            or len(rows) != len(plan["groups"]) or plan["strata"] != list(_EVENT_SAMPLING_STRATA)
            or np.asarray(plan["row_group"]).shape != (n,)
            or np.asarray(plan["row_stratum"]).shape != (n,)):
        raise ValueError("Event stratified sampling invalid plan groups or strata")
    flattened = np.concatenate([r for group in rows for r in group])
    if (len(flattened) != n or not np.array_equal(np.sort(flattened), np.arange(n))
            or np.asarray(pi).shape != (n,) or not np.isfinite(pi).all() or not (pi > 0).all()):
        raise ValueError("Event stratified sampling invalid plan membership or probabilities")
    expected_pi = np.empty(n, dtype=np.float64)
    expected_counts = []
    frame_groups = np.asarray(frame["physical_unit_id"], str)
    for g, strata in enumerate(rows):
        previous_stratum = -1
        counts = {key: 0 for key in _EVENT_SAMPLING_STRATA}
        for origins in strata:
            memberships = plan["row_stratum"][origins]
            k = int(memberships[0])
            if (k <= previous_stratum or k >= len(_EVENT_SAMPLING_STRATA)
                    or not (memberships == k).all()
                    or not (plan["row_group"][origins] == g).all()
                    or not (frame_groups[origins] == plan["groups"][g]).all()):
                raise ValueError("Event stratified sampling inconsistent plan memberships")
            previous_stratum = k
            counts[_EVENT_SAMPLING_STRATA[k]] = len(origins)
            expected_pi[origins] = 1.0 / (len(rows) * len(strata) * len(origins))
        expected_counts.append({"physical_unit_id": plan["groups"][g],
                                "origin_count": sum(counts.values()), "stratum_counts": counts})
    if (not np.array_equal(expected_pi, pi) or not np.isclose(np.sum(pi), 1.0)
            or plan["counts"] != {"groups": expected_counts, "total_origin_count": n,
                "physical_group_count": len(rows), "stratum_origin_counts": {
                    key: sum(group["stratum_counts"][key] for group in expected_counts)
                    for key in _EVENT_SAMPLING_STRATA}}):
        raise ValueError("Event stratified sampling inconsistent plan probabilities or counts")
    def slots(size, count):
        base = np.repeat(np.arange(count), size // count)
        remainder = rng.choice(count, size % count, replace=False)
        return rng.permutation(np.concatenate((base, remainder)))
    for budget in np.array_split(np.arange(n), int(np.ceil(n / batch_size))):
        groups = slots(len(budget), len(rows))
        idx = np.empty(len(budget), dtype=int)
        for g, strata in enumerate(rows):
            positions = np.flatnonzero(groups == g)
            stratum_slots = slots(len(positions), len(strata))
            for k, origins in enumerate(strata):
                destination = positions[stratum_slots == k]
                idx[destination] = rng.choice(origins, len(destination), replace=True)
        yield idx, pi[idx]

def _event_head_rate_multiplier(config):
    value = config.get("event_head_learning_rate_multiplier", 1.0)
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError("event_head_learning_rate_multiplier must be a finite positive real")
    return float(value)


def _optimizer_rate_metadata(config):
    multiplier = _event_head_rate_multiplier(config)
    base_rate = config["learning_rate"]
    event_rate = base_rate * multiplier
    if not np.isfinite(event_rate):
        raise ValueError("Resolved event_head learning rate must be finite")
    return {
        "optimizer": "AdamW",
        "parameter_group_policy": (
            "legacy_single_group" if multiplier == 1.0 else "all_other_parameters_then_event_head"
        ),
        "event_head_learning_rate_multiplier": multiplier,
        "base_learning_rate": base_rate,
        "event_head_learning_rate": event_rate,
        "weight_decay": config["weight_decay"],
        "base_decay_per_step": base_rate * config["weight_decay"],
        "event_head_decay_per_step": event_rate * config["weight_decay"],
        "decay_policy": "same_weight_decay_in_all_groups_effective_decay_scales_with_learning_rate",
        "global_gradient_clip_norm": 5.0,
    }


def _make_optimizer(model, config):
    rates = _optimizer_rate_metadata(config)
    if rates["event_head_learning_rate_multiplier"] == 1.0:
        # Preserve legacy parameter traversal, group structure and AdamW defaults.
        return torch.optim.AdamW(
            model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
        )
    head_ids = {id(parameter) for parameter in model.event_head.parameters()}
    parameters = list(model.parameters())
    other = [parameter for parameter in parameters if id(parameter) not in head_ids]
    head = [parameter for parameter in parameters if id(parameter) in head_ids]
    if not head or len(head) != len(head_ids):
        raise ValueError("Event head parameters must belong to the learned model")
    # AdamW applies (1 - lr * weight_decay) per active parameter at each step.
    # Head decay therefore scales with this ablation's rate; it is not compensated.
    return torch.optim.AdamW(
        [
            {"params": other, "lr": config["learning_rate"]},
            {"params": head, "lr": rates["event_head_learning_rate"]},
        ],
        lr=config["learning_rate"], weight_decay=config["weight_decay"],
    )


def _weights(frame):
    groups = np.asarray(frame["physical_unit_id"], str)
    counts = {uid: int((groups == uid).sum()) for uid in set(groups)}
    return np.asarray([1 / counts[uid] / len(counts) for uid in groups], np.float32)


def _objective_weights(frame):
    """Each objective term balances only units with evidence for that term."""
    path_known = np.asarray(frame["mask"]).any(1)
    event_known = (
        np.asarray(frame["event_observed"]) | (np.asarray(frame["no_entry_prefix"]) > 0)
    ) & (np.asarray(frame["current"]) < np.asarray(frame["red_threshold"]))
    result = {}
    for key, known in (
        ("energy", path_known),
        ("width", path_known),
        ("miss", path_known),
        ("event", event_known),
        (
            "phase_energy",
            path_known
            & (event_known | (np.asarray(frame["current"]) >= np.asarray(frame["red_threshold"]))),
        ),
    ):
        out = np.zeros(len(known), np.float32)
        if known.any():
            subset = {
                "physical_unit_id": [g for g, k in zip(frame["physical_unit_id"], known) if k]
            }
            out[known] = _weights(subset)
        result[key] = out
    return result


def _at_risk_weights(frame):
    """Balance the width prior using current state, without future availability."""
    applicable = np.asarray(frame["current"]) < np.asarray(frame["red_threshold"])
    result = np.zeros(len(applicable), np.float32)
    if applicable.any():
        result[applicable] = _weights({"physical_unit_id": [
            g for g, keep in zip(frame["physical_unit_id"], applicable) if keep
        ]})
    return result


def _optimizer_batches(frame, batch_size, rng, sampling="row_permutation", *, event_stratified_plan=None):
    """Draw only from Train membership; preserve the legacy permutation RNG path.

    Group draws are stratified within each batch: every group receives floor(B/G)
    slots, and a uniform subset receives the remainder. Shuffling these slots
    makes every slot group-uniform. Rows within each group are drawn uniformly
    with replacement, hence their marginal probability is 1/(G * group_size).
    Group order follows first appearance, so identifier spelling has no effect.
    """
    if sampling == "physical_event_stratified":
        yield from _event_stratified_batches(frame, batch_size, rng, event_stratified_plan)
        return
    n = len(frame["x"])
    batch_count = int(np.ceil(n / batch_size))
    if sampling == "row_permutation":
        for idx in np.array_split(rng.permutation(n), batch_count):
            yield idx, None
        return
    if sampling != "physical_group":
        raise ValueError("Invalid batch_sampling")
    members = {}
    for row, group in enumerate(frame["physical_unit_id"]):
        members.setdefault(str(group), []).append(row)
    rows = [np.asarray(indices, dtype=int) for indices in members.values()]
    group_count = len(rows)
    # Same row budget, batch count and batch sizes as legacy np.array_split.
    for slots in np.array_split(np.arange(n), batch_count):
        size = len(slots)
        groups = np.repeat(np.arange(group_count), size // group_count)
        remainder = rng.choice(group_count, size % group_count, replace=False)
        groups = rng.permutation(np.concatenate((groups, remainder)))
        idx = np.empty(size, dtype=int)
        probabilities = np.empty(size, dtype=np.float32)
        for group in range(group_count):
            positions = np.flatnonzero(groups == group)
            idx[positions] = rng.choice(rows[group], len(positions), replace=True)
            probabilities[positions] = 1.0 / (group_count * len(rows[group]))
        yield idx, probabilities


def _optimizer_term(values, weights, indices, probabilities):
    """Original weighted term, or its unbiased inverse-probability batch mean."""
    row_weights = torch.as_tensor(weights[indices])
    if probabilities is None:
        return (values * row_weights).sum()
    return (values * (row_weights / torch.as_tensor(probabilities))).mean()


def _subset(frame, idx):
    n = len(frame["x"])
    return {
        key: value[idx]
        if isinstance(value, np.ndarray) and value.ndim and len(value) == n
        else [value[i] for i in idx]
        if isinstance(value, list) and len(value) == n and key != "feature_names"
        else value
        for key, value in frame.items()
    }


def _normalized(frame, scaler):
    mean, std = np.asarray(scaler["mean"], np.float32), np.asarray(scaler["std"], np.float32)
    if "history_mask" in frame:
        mask = np.asarray(frame["history_mask"])
        if mask.dtype != bool or mask.shape != frame["x"].shape[:2]:
            raise ValueError("Invalid recurrent history mask for normalization")
        # Ignore padding before arithmetic, including arbitrary NaN/Inf pads.
        return (np.where(mask[..., None], frame["x"], mean) - mean) / std
    return (frame["x"] - mean) / std


def _context_model_kwargs(config, frame):
    if config.get("recurrent_context_mode", "fixed") == "fixed":
        if "history_lengths" in frame or "history_mask" in frame:
            raise ValueError("Variable history frame supplied to a fixed-context model")
        return {}
    if "history_lengths" not in frame or "history_mask" not in frame:
        raise ValueError("Variable context requires history lengths and mask")
    lengths, mask = np.asarray(frame["history_lengths"]), np.asarray(frame["history_mask"])
    x = np.asarray(frame["x"])
    maximum, minimum = config["max_history_length"], config["history_length"]
    if (x.ndim != 3 or x.shape[1] != maximum or lengths.shape != (len(x),)
            or not np.issubdtype(lengths.dtype, np.integer) or mask.dtype != bool
            or mask.shape != x.shape[:2] or (lengths < minimum).any() or (lengths > maximum).any()
            or not np.array_equal(mask, np.arange(maximum)[None] < lengths[:, None])):
        raise ValueError("Variable context history lengths/mask differ from saved layout")
    return {"history_lengths": torch.as_tensor(lengths), "history_mask": torch.as_tensor(mask)}


def _normalization_reference(frame, config):
    _context_model_kwargs(config, frame)
    if config.get("recurrent_context_mode", "fixed") == "fixed":
        return frame["x"]
    if "scaler_x" not in frame:
        raise ValueError("Variable context requires trailing real-history scaler anchors")
    minimum = config["history_length"]
    expected = np.stack([frame["x"][i, length - minimum:length]
                         for i, length in enumerate(frame["history_lengths"])])
    anchor = np.asarray(frame["scaler_x"])
    if not np.array_equal(anchor, expected) or not np.isfinite(anchor).all():
        raise ValueError("Variable context scaler anchors differ from trailing real history")
    return anchor


def _prediction_identity(config, frame, index):
    if config.get("recurrent_context_mode", "fixed") == "fixed":
        return frame["x"][index].tobytes() + str(config["seed"]).encode()
    length = int(frame["history_lengths"][index])
    minimum = config["history_length"]
    # Seed couples random draws across shared origins; it is not a forecast
    # cache key. Longer real history still changes the packed model moments.
    return frame["x"][index, length - minimum:length].tobytes() + str(config["seed"]).encode()


def _tree_event_horizons(horizons_s):
    """Nearest supported dense-grid cutoffs, including the saved full horizon."""
    grid = np.asarray(horizons_s, float)
    return sorted({int(np.argmin(np.abs(grid - minutes * 60)))
                   for minutes in (10, 20, 30, 60, 120)} | {len(grid) - 1})


TREE_WEIGHT_POLICY = "physical_group_equal_over_known_rows_per_horizon_mean_one_v2"


def _tree_weights(groups):
    """Balance admitted groups without shrinking the tree optimizer's weight scale."""
    return _weights({"physical_unit_id": groups}) * len(groups)


def _tree_event_targets(frame, horizon_index):
    """First recorded-grid entry by cutoff; partial survival remains unknown."""
    observed = np.asarray(frame["event_observed"], bool)
    allowed = np.asarray(frame["event_allowed"], bool)[:, :-1]
    if np.any(observed & (allowed.sum(1) != 1)):
        raise ValueError("Observed tree event needs exactly one recorded-grid entry")
    at_risk = np.asarray(frame["current"]) < np.asarray(frame["red_threshold"])
    known = at_risk & (observed | (np.asarray(frame["no_entry_prefix"]) >= horizon_index + 1))
    labels = (observed & (np.argmax(allowed, axis=1) <= horizon_index)).astype(np.int64)
    return labels, known


def _fit_tree_event_heads(train, flat, config, stop=None):
    heads = []
    for h in _tree_event_horizons(config["horizons_s"]):
        _check(stop)
        labels, known = _tree_event_targets(train, h)
        groups = np.asarray(train["physical_unit_id"], str)[known].tolist()
        entry = {"horizon_index": h, "horizon_s": float(config["horizons_s"][h]),
                 "known_rows": int(known.sum()), "positive_rows": int(labels[known].sum()),
                 "fit_physical_groups": sorted(set(groups))}
        classes = np.unique(labels[known])
        if len(classes) < 2:
            entry.update(kind="constant", probability=float(classes[0]) if len(classes) else 0.5,
                         reason="single_train_class" if len(classes) else "no_known_train_evidence")
        else:
            head = HistGradientBoostingClassifier(
                max_iter=config["max_iter"], max_leaf_nodes=15, min_samples_leaf=5,
                early_stopping=False, random_state=config["seed"])
            head.fit(flat[known], labels[known],
                     sample_weight=_tree_weights(groups))
            entry.update(kind="classifier", model=head)
        heads.append(entry)
    return heads


def _tree_fit_fingerprint(train):
    digest = hashlib.sha256()
    digest.update(TREE_WEIGHT_POLICY.encode())
    for key in ("x", "y", "mask", "current", "red_threshold", "event_allowed",
                "event_observed", "no_entry_prefix"):
        value = np.asarray(train[key])
        digest.update(_digest([key, value.shape, str(value.dtype)]).encode())
        digest.update(value.tobytes())
    digest.update(_digest(list(map(str, train["physical_unit_id"]))).encode())
    return digest.hexdigest()


def _external(encoder, frame, scaler, stop=None, status_cb=None):
    if encoder is None:
        return None
    values = _normalized(frame, scaler)
    if encoder["kind"] == "full_cns":
        raw = np.log1p(frame["raw_x"]) / encoder["signal_scale"]
        return np.concatenate(
            [encoder["body"].transform(raw, should_stop=stop, status_cb=status_cb), values[:, -1, :]], axis=1
        )
    flat = values.reshape(len(values), -1)
    preds = np.column_stack([head.predict(flat) for head in encoder["heads"]])
    columns = [preds, values[:, -1, :]]
    for entry in encoder.get("event_heads", []):
        _check(stop)
        if entry["kind"] == "constant":
            probability = np.full(len(flat), entry["probability"])
        else:
            head = entry["model"]
            positive = np.flatnonzero(np.asarray(head.classes_) == 1)
            if len(positive) != 1:
                raise ValueError("Tree event classifier has no unique positive class")
            probability = head.predict_proba(flat)[:, int(positive[0])]
        if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
            raise ValueError("Tree event probabilities must be finite and bounded")
        columns.append(probability[:, None])
    return np.concatenate(columns, axis=1).astype(np.float32)


def _make_model(engine, config, input_size, external_size):
    _post_factor_smoothness_weight(config)
    _path_band_geometry(config)
    _red_corridor_objective_settings(config)
    _red_finite_bound_weight(config)
    _event_cdf_objective_settings(config)
    context = {}
    mode = config.get("recurrent_context_mode", "fixed")
    if not isinstance(mode, str) or mode not in {"fixed", "variable_causal"}:
        raise ValueError("Invalid recurrent_context_mode")
    if mode == "fixed" and config.get("max_history_length") is not None:
        raise ValueError("Fixed recurrent context requires max_history_length=None")
    if mode == "variable_causal":
        if engine not in {"gru", "lstm"}:
            raise ValueError("Variable causal context supports GRU/LSTM only")
        context = {"recurrent_context_mode": "variable_causal",
                   "min_history_length": config["history_length"],
                   "max_history_length": config["max_history_length"]}
    return SignalDistribution(
        engine if engine in {"gru", "lstm"} else "gru",
        input_size,
        len(config["horizons_s"]),
        config["hidden_size"],
        config["num_layers"],
        config["rank"],
        external_feature_size=external_size,
        dropout=config["dropout"],
        event_location=config.get("event_location", "softplus"),
        event_distribution=config.get("event_distribution", "lognormal_cure"),
        phase_covariance=config.get("phase_covariance", "shared"),
        path_distribution=config.get("path_distribution", "single"),
        **context,
    )


def fit_learned_model(engine, train, validation, config, *, stop=None, report=None, cache_dir=None):
    """Fit Train only and restore the checkpoint with best Validation joint score."""
    report = report or (lambda _: None)
    _optimizer_rate_metadata(config)
    regularization_weight = _post_factor_smoothness_weight(config)
    band_geometry = _path_band_geometry(config)
    red_settings = _red_corridor_objective_settings(config)
    finite_bound_weight = _red_finite_bound_weight(config)
    cdf_settings = _event_cdf_objective_settings(config)
    sampling_contract = _event_stratified_sampling_settings(config)
    sampling_plan = (_event_stratified_plan(train, sampling_contract["prefix_length"])
                     if sampling_contract is not None else None)
    if sampling_plan is not None and train["mask"].shape[1] != len(config["horizons_s"]):
        raise ValueError("Event stratified Train horizon differs from saved grid")
    if not len(train["x"]) or not len(validation["x"]):
        raise ValueError("Train and Validation need causal trajectory windows")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    weights = _weights(train)
    reference = _normalization_reference(train, config)
    mean = (reference * weights[:, None, None]).sum((0, 1)) / reference.shape[1]
    var = ((reference - mean) ** 2 * weights[:, None, None]).sum((0, 1)) / reference.shape[1]
    scaler = {
        "mean": mean.tolist(),
        "std": np.maximum(np.sqrt(var), 1e-5).tolist(),
        "fit_physical_groups": sorted(set(train["physical_unit_id"])),
        "output_domain": "nonnegative",
    }
    encoder = None
    if engine == "full_cns":
        report({"stage": "encoding", "message": "Building complete MaleCNS encoder"})
        body = build_signal_full_cns(config["seed"])
        body.provenance["readout_policy"] = (
            "learned conditional low-rank path distribution and RED-entry mixture"
        )
        encoder = {
            "kind": "full_cns",
            "body": body,
            "signal_scale": float(max(np.std(np.log1p(train["raw_x"])), 1e-5)),
        }
    elif engine == "quantile_boosting":
        flat = _normalized(train, scaler).reshape(len(train["x"]), -1)
        candidates = np.flatnonzero(train["mask"].any(0))
        selected = candidates[
            np.unique(
                np.linspace(
                    0, len(candidates) - 1, min(config["tree_horizons"], len(candidates)), dtype=int
                )
            )
        ]
        if not len(selected):
            raise ValueError("Tree encoder has no observed Train horizons")
        heads = []
        for h in selected:
            valid = train["mask"][:, h]
            for q in (0.1, 0.5, 0.9):
                _check(stop)
                head = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=q,
                    max_iter=config["max_iter"],
                    max_leaf_nodes=15,
                    min_samples_leaf=5,
                    early_stopping=False,
                    random_state=config["seed"],
                )
                groups = np.asarray(train["physical_unit_id"], str)[valid].tolist()
                head.fit(flat[valid], np.log1p(train["y"][valid, h]),
                         sample_weight=_tree_weights(groups))
                heads.append(head)
        encoder = {
            "kind": "quantile_boosting",
            "heads": heads,
            "horizons": selected.tolist(),
            "tree_weight_policy": TREE_WEIGHT_POLICY,
            "event_heads": _fit_tree_event_heads(train, flat, config, stop),
            "risk_head_policy": {
                "target": "first_future_recorded_grid_RED_entry_by_cutoff",
                "fit_split": "Train", "calibration": "none",
                "sample_weight": TREE_WEIGHT_POLICY,
                "unknown_partial_survival": "excluded", "already_RED": "excluded",
                "observed_entry_after_cutoff": "known_negative",
                "inputs": "normalized_causal_history_without_physical_identifiers",
                "validation_or_test_tree_fit": False,
            },
            "readout_policy": "Train-fitted quantile and event-probability trees supply fixed features to a neural joint path/event readout; neural gradients do not train trees",
            "fit_fingerprint": _tree_fit_fingerprint(train),
        }

    def design(frame, partition):
        if encoder is None:
            return None
        key = _digest(
            {
                "encoder": encoder["kind"],
                "config": config,
                "scaler": scaler,
                "graph": getattr(encoder.get("body"), "provenance", None),
                "fit_fingerprint": encoder.get("fit_fingerprint"),
                "windows": hashlib.sha256(
                    frame["x"].tobytes() + frame["raw_x"].tobytes()
                ).hexdigest(),
            }
        )
        path = Path(cache_dir) / f"{key}.npy" if cache_dir else None
        if path and path.is_file():
            return np.load(path, allow_pickle=False)
        def encoding_progress(done, total):
            if done == total or done % 128 == 0:
                report({"stage": "encoding", "message": f"MaleCNS {partition} windows: {done}/{total}"})
        result = _external(encoder, frame, scaler, stop, encoding_progress)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, result, allow_pickle=False)
        return result

    train_ext, val_ext = design(train, "Train"), design(validation, "Validation")
    ext_scaler = None
    if train_ext is not None:
        em = (train_ext * weights[:, None]).sum(0)
        es = np.maximum(np.sqrt(((train_ext - em) ** 2 * weights[:, None]).sum(0)), 1e-5)
        ext_scaler = {"mean": em.tolist(), "std": es.tolist()}
        train_ext = (train_ext - em) / es
        val_ext = (val_ext - em) / es
    model = _make_model(
        engine, config, train["x"].shape[-1], None if train_ext is None else train_ext.shape[-1]
    )
    optimizer = _make_optimizer(model, config)
    path_prefix_lengths = _path_objective_prefix_lengths(config)
    event_prefix_lengths = _path_objective_prefix_lengths(config, "event_objective_horizons_s")

    def terms(frame, indices, external, seed):
        part = _subset(frame, indices)
        generator = torch.Generator().manual_seed(seed)
        moments = model(
            torch.as_tensor(_normalized(part, scaler)),
            torch.as_tensor(part["current"]),
            None if external is None else torch.as_tensor(external[indices]),
            **_context_model_kwargs(config, part),
        )
        return model.objective(
            moments,
            torch.as_tensor(part["y"]),
            torch.as_tensor(part["mask"]),
            torch.as_tensor(part["red_threshold"]),
            event_allowed_mask=torch.as_tensor(part["event_allowed"]),
            event_observed_mask=torch.as_tensor(part["event_observed"]),
            no_entry_prefix=torch.as_tensor(part["no_entry_prefix"]),
            n_samples=config["training_samples"],
            generator=generator,
            coverage=config["nominal_coverage"],
            width_weight=config["width_weight"],
            miss_weight=config["miss_weight"],
            event_weight=config["event_weight"],
            energy_weight=config["energy_weight"],
            phase_weight=config["phase_weight"],
            path_objective_prefix_lengths=path_prefix_lengths,
            event_objective_prefix_lengths=event_prefix_lengths,
            **({"post_factor_smoothness_weight": regularization_weight}
               if regularization_weight > 0 else {}),
            **({"path_band_geometry": band_geometry}
               if band_geometry == "issued_prefix" else {}),
            **({"red_corridor_width_weight": red_settings["red_corridor_width_weight"],
                "red_corridor_miss_weight": red_settings["red_corridor_miss_weight"],
                "red_corridor_scale_steps": red_settings["scale_steps"]}
               if red_settings["enabled"] else {}),
            **({"red_finite_bound_weight": finite_bound_weight}
               if finite_bound_weight > 0 else {}),
            **({"event_cdf_weight": cdf_settings["event_cdf_weight"],
                "event_cdf_scale_steps": cdf_settings["scale_steps"]}
               if cdf_settings["enabled"] else {}),
        )

    rng = np.random.default_rng(config["seed"])
    best_score = float("inf")
    best = None
    trace = []
    stale = 0
    tw = _objective_weights(train)
    vw = _objective_weights(validation)
    coefficients = {key: config[key + "_weight"] for key in ("energy", "width", "miss", "event")}
    coefficients["phase_energy"] = config["phase_weight"]
    if regularization_weight > 0:
        # This model-only prior balances all causal origins independently of
        # future availability. Existing evidence-specific weights stay intact.
        coefficients["post_factor_smoothness"] = regularization_weight
        tw["post_factor_smoothness"] = _weights(train)
        vw["post_factor_smoothness"] = _weights(validation)
    if red_settings["enabled"]:
        for key in ("red_corridor_width", "red_corridor_miss"):
            coefficients[key] = red_settings[key + "_weight"]
        tw["red_corridor_width"] = _at_risk_weights(train)
        vw["red_corridor_width"] = _at_risk_weights(validation)
        tw["red_corridor_miss"] = tw["event"].copy()
        vw["red_corridor_miss"] = vw["event"].copy()
    if finite_bound_weight > 0:
        coefficients["red_finite_bound"] = finite_bound_weight
        tw["red_finite_bound"] = tw["event"].copy()
        vw["red_finite_bound"] = vw["event"].copy()
    if cdf_settings["enabled"]:
        coefficients["event_cdf"] = cdf_settings["event_cdf_weight"]
        tw["event_cdf"] = tw["event"].copy()
        vw["event_cdf"] = vw["event"].copy()
    for epoch in range(config["epochs"]):
        model.train()
        train_tot = {key: 0.0 for key in ("total", *coefficients)}
        gradient_norms = []
        if sampling_plan is not None:
            sampled_origins, batch_estimates = [], {key: [] for key in coefficients}
        for idx, probabilities in _optimizer_batches(
            train, config["batch_size"], rng, config.get("batch_sampling", "row_permutation"),
            **({"event_stratified_plan": sampling_plan} if sampling_plan is not None else {})
        ):
            _check(stop)
            if sampling_plan is not None:
                sampled_origins.extend(idx.tolist())
            optimizer.zero_grad()
            loss = terms(train, idx, train_ext, config["seed"] + epoch * 10000 + int(idx[0]))
            if sampling_plan is not None and any(not bool(torch.isfinite(loss[key]).all()) for key in coefficients):
                raise FloatingPointError("Nonfinite event stratified row losses before weighting")
            total = sum(
                coefficients[key] * _optimizer_term(loss[key], tw[key], idx, probabilities)
                for key in coefficients
            )
            # Both modes target the original objective before gradient clipping.
            gradient_scale = len(train["x"]) / len(idx) if probabilities is None else 1.0
            (total * gradient_scale).backward()
            gradient_norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 5)))
            if sampling_plan is not None and not np.isfinite(gradient_norms[-1]):
                raise FloatingPointError("Nonfinite event stratified preclip gradient norm")
            optimizer.step()
            report_scale = 1.0 if probabilities is None else len(idx) / len(train["x"])
            for key in coefficients:
                estimate = _optimizer_term(loss[key].detach(), tw[key], idx, probabilities)
                if sampling_plan is not None:
                    if not bool(torch.isfinite(estimate)):
                        raise FloatingPointError("Nonfinite event stratified batch term estimate")
                    batch_estimates[key].append(float(estimate))
                train_tot[key] += float(estimate) * report_scale
            train_tot["total"] += float(total.detach()) * report_scale
        model.eval()
        val_tot = {key: 0.0 for key in train_tot}
        with torch.no_grad():
            for start in range(0, len(validation["x"]), config["batch_size"]):
                _check(stop)
                idx = np.arange(start, min(start + config["batch_size"], len(validation["x"])))
                loss = terms(validation, idx, val_ext, config["seed"] + 900000 + start)
                for key in coefficients:
                    val_tot[key] += float((loss[key] * torch.as_tensor(vw[key][idx])).sum())
        val_tot["total"] = sum(coefficients[key] * val_tot[key] for key in coefficients)
        score = val_tot["total"]
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite Validation joint objective")
        trace.append({"epoch": epoch + 1, "train": train_tot, "validation": val_tot,
                      "gradient_norm_mean": float(np.mean(gradient_norms)),
                      "clipped_batch_fraction": float(np.mean(np.asarray(gradient_norms) > 5))})
        if sampling_plan is not None:
            draws = np.asarray(sampled_origins, dtype=int)
            trace[-1]["optimizer_sampling"] = {
                "draw_count": len(draws), "distinct_origin_count": len(np.unique(draws)),
                "duplicate_origin_draw_count": len(draws) - len(np.unique(draws)),
                "group_stratum_draw_counts": [
                    {"physical_unit_id": name, "draw_count": int((sampling_plan["row_group"][draws] == g).sum()),
                     "stratum_counts": {key: int(((sampling_plan["row_group"][draws] == g) &
                                                   (sampling_plan["row_stratum"][draws] == k)).sum())
                                        for k, key in enumerate(_EVENT_SAMPLING_STRATA)}}
                    for g, name in enumerate(sampling_plan["groups"])],
                "term_batch_estimates": {key: {"count": len(values), "mean": float(np.mean(values)),
                    "std": float(np.std(values)), "min": float(np.min(values)), "max": float(np.max(values))}
                    for key, values in batch_estimates.items()},
                "interpretation": "moving_parameter_batch_estimates_no_variance_reduction_proof",
            }
        report(
            {
                "stage": "training",
                "progress": (epoch + 1) / config["epochs"],
                "message": f"Learned paths epoch {epoch + 1}, Validation joint score {score:.5f}",
            }
        )
        if score < best_score - config["min_delta"]:
            best_score = score
            best = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= config["patience"]:
            break
    model.load_state_dict(best)
    model.eval()
    return {
        "model": model,
        "encoder": encoder,
        "scaler": scaler,
        "external_scaler": ext_scaler,
        "config": config,
        "trace": trace,
        "selection": {
            "optimizer_rates": _optimizer_rate_metadata(config),
            "criterion": "validation_physical_group_equal_joint_objective",
            "batch_sampling": config.get("batch_sampling", "row_permutation"),
            **({"optimizer_sampling_contract": sampling_contract,
                 "optimizer_sampling_train": {**sampling_plan["counts"],
                    "term_weight_sha256": {key: hashlib.sha256(value.tobytes()).hexdigest()
                                           for key, value in tw.items()}}}
               if sampling_plan is not None else {}),
            "optimizer_sampling_policy": (
                "physical_event_stratified_inverse_probability_term_means"
                if sampling_plan is not None else
                "group_stratified_uniform_with_replacement_inverse_probability_term_means"
                if config.get("batch_sampling", "row_permutation") == "physical_group"
                else "legacy_row_permutation_global_term_weights"
            ),
            "path_objective_horizons_s": (
                [config["horizons_s"][-1]] if path_prefix_lengths is None
                else list(config["path_objective_horizons_s"])
            ),
            "event_objective_horizons_s": (
                [config["horizons_s"][-1]] if event_prefix_lengths is None
                else list(config["event_objective_horizons_s"])
            ),
            "best_epoch": best_epoch,
            "best_score": best_score,
            "restored_best_checkpoint": True,
            "test_feedback": False,
            "coverage_guarantee": False,
            **({"post_factor_smoothness_contract":
                _event_distribution_metadata(model, config)["post_factor_smoothness_contract"]}
               if regularization_weight > 0 else {}),
            **({"path_band_geometry_contract":
                _event_distribution_metadata(model, config)["path_band_geometry_contract"]}
               if band_geometry == "issued_prefix" else {}),
            **({"red_corridor_objective_contract":
                _event_distribution_metadata(model, config)["red_corridor_objective_contract"]}
               if red_settings["enabled"] else {}),
            **({"red_finite_bound_contract":
                _event_distribution_metadata(model, config)["red_finite_bound_contract"]}
               if finite_bound_weight > 0 else {}),
            **({"event_cdf_objective_contract":
                _event_distribution_metadata(model, config)["event_cdf_objective_contract"]}
               if cdf_settings["enabled"] else {}),
        },
        "feature_names": train["feature_names"],
    }


def predict_learned(bundle, frame, *, stop=None, samples=None):
    _check(stop)
    if frame["feature_names"] != bundle["feature_names"]:
        raise ValueError("Causal input feature contract differs from saved learned model")
    model = bundle["model"]
    model.eval()
    config = bundle["config"]
    ext = frame.get("_external_features")
    if ext is None:
        ext = _external(bundle["encoder"], frame, bundle["scaler"], stop)
    if ext is not None:
        es = bundle["external_scaler"]
        ext = (ext - np.asarray(es["mean"], np.float32)) / np.asarray(es["std"], np.float32)
    outputs = []
    with torch.no_grad():
        normalized = _normalized(frame, bundle["scaler"])
        for i in range(len(frame["x"])):
            _check(stop)
            moments = model(
                torch.as_tensor(normalized[i : i + 1]),
                torch.as_tensor(frame["current"][i : i + 1]),
                None if ext is None else torch.as_tensor(ext[i : i + 1]),
                **_context_model_kwargs(config, _subset(frame, np.asarray([i]))),
            )
            identity = hashlib.sha256(
                _prediction_identity(config, frame, i)
            ).digest()
            seed = int.from_bytes(identity[:8], "little") % (2**63 - 1)
            paths = model.sample(
                moments,
                torch.as_tensor(frame["red_threshold"][i : i + 1]),
                samples or config["path_samples"],
                torch.Generator().manual_seed(seed),
            )
            lower, upper = model.simultaneous_band(paths, config["nominal_coverage"])
            outputs.append(
                {
                    "paths": paths.numpy(),
                    "lower": lower.numpy(),
                    "upper": upper.numpy(),
                    "mean": paths.median(0).values.numpy(),
                    "event_probabilities": moments["event_logits"].softmax(-1).numpy(),
                }
            )
    return {
        key: np.concatenate([item[key] for item in outputs], axis=1 if key == "paths" else 0)
        for key in outputs[0]
    }


def prepare_prediction_frame(bundle, frame, cache_dir, stop=None):
    if bundle["encoder"] is None:
        return frame
    encoder = bundle["encoder"]
    key = _digest(
        {
            "kind": encoder["kind"],
            "scaler": bundle["scaler"],
            "graph": getattr(encoder.get("body"), "provenance", None),
            "signal_scale": encoder.get("signal_scale"),
            "fit_fingerprint": encoder.get("fit_fingerprint"),
            "windows": hashlib.sha256(frame["x"].tobytes() + frame["raw_x"].tobytes()).hexdigest(),
        }
    )
    path = Path(cache_dir) / f"prediction-{key}.npy"
    if path.is_file():
        external = np.load(path, allow_pickle=False)
    else:
        external = _external(encoder, frame, bundle["scaler"], stop)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, external, allow_pickle=False)
    if len(external) != len(frame["x"]) or not np.isfinite(external).all():
        raise ValueError("Invalid causal prediction cache")
    return {**frame, "_external_features": external}


def _event_distribution_metadata(model, config):
    """Bind optional event/path families to the saved grid, without RNG use."""
    fields = {
        "event_distribution": model.event_distribution,
        "effective_event_location": model.effective_event_location,
    }
    sampling_contract = _event_stratified_sampling_settings(config)
    if sampling_contract is not None:
        fields["optimizer_sampling_contract"] = sampling_contract
    if model.event_distribution == "finite_horizon_mixture":
        fields["event_distribution_contract"] = {
            "horizon_steps": model.n_horizons,
            "horizons_s": list(config["horizons_s"]),
            "median_lower_steps": 0.05,
            "median_upper_steps": model.n_horizons,
            "component_truncation": "per component at saved horizon",
            "survival_semantics": "no recorded entry through saved horizon",
        }
    if model.path_distribution == "coupled_timing_mixture":
        fields.update({
            "path_distribution": model.path_distribution,
            "path_components": model.path_components,
            "path_distribution_contract": {
                "horizon_steps": model.n_horizons,
                "horizons_s": list(config["horizons_s"]),
                "component_count": 3,
                "component_association": "path head index equals timing component index",
                "head_layout": "batch_component_horizon_parameters",
                "shared_hidden_stem": True,
                "path_head_outputs": model.path_head[-1].out_features,
                "parameters_per_component_horizon": model.path_parameters_per_horizon,
                "phase_covariance": model.phase_covariance,
                "latent_scope": "one component and entry bucket per whole saved-grid path",
                "joint_law": "issued event marginal times conditional timing-component responsibility",
                "survival_component_law": "unconditional timing mixture weights",
                "already_red_component_law": "unconditional timing mixture weights; entry inapplicable",
                "conditional_phase_score": "joint compatible-entry conditional score with live normalizer",
            },
        })
    if getattr(model, "recurrent_context_mode", "fixed") == "variable_causal":
        fields.update({
            "recurrent_context_mode": "variable_causal",
            "recurrent_context_contract": {
                "minimum_real_observations": model.min_history_length,
                "maximum_real_observations": model.max_history_length,
                "tensor_layout": "batch_time_feature",
                "chronological_order": "oldest to newest",
                "padding": "contiguous real prefix followed by right padding",
                "lengths_dtype": "integer",
                "mask_dtype": "boolean",
                "encoder_architecture": "gru" if isinstance(model.encoder, torch.nn.GRU) else "lstm",
                "encoder_policy": "packed_sequence_final_hidden_restored_original_order",
                "gap_policy": "reset context at continuous segment boundary",
                "normalization_reference": "trailing minimum real history on unchanged physical-group-balanced Train origins",
                "sensor_baseline": "causal initial8 log-feature baseline remains fixed after8",
                "feature_history": "causal_features computed over completed segment before truncation",
                "prediction_seed_policy": "legacy trailing minimum real feature bytes plus saved seed; common random draws across fixed/variable contexts; padding excluded; not a forecast cache key",
                "minimum_eligibility_policy": "unchanged history_length",
            },
        })
    regularization_weight = _post_factor_smoothness_weight(config)
    if regularization_weight > 0:
        fields["post_factor_smoothness_contract"] = {
            "weight": regularization_weight,
            "phase": "post only; pre and entry factors excluded",
            "factor_layout": "batch_horizon_rank",
            "factor_coordinate": "absolute saved-grid forecast lead",
            "horizons_s": list(config["horizons_s"]),
            "objective_prefix_horizons_s": (
                [config["horizons_s"][-1]]
                if config.get("path_objective_horizons_s") is None
                else list(config["path_objective_horizons_s"])
            ),
            "formula": "equal prefix mean of adjacent post-factor squared L2 differences divided by squared RED threshold; one-lead prefix zero",
            "units": "raw post softplus-input squared units / squared RED physical signal threshold per recorded-grid adjacent pair",
            "origin_weighting": "all causal origins with equal physical-unit total weight; independent of future evidence",
            "data_dependency": "issued post factors, fixed RED threshold and saved prefixes only",
            "selection_policy": "same regularizer included in Validation joint checkpoint criterion",
            "sampler_change": False,
        }
    if _path_band_geometry(config) == "issued_prefix":
        fields.update({
            "path_band_geometry": "issued_prefix",
            "path_band_geometry_contract": {
                "geometry": "issued_prefix",
                "horizons_s": list(config["horizons_s"]),
                "objective_prefix_horizons_s": (
                    [config["horizons_s"][-1]]
                    if config.get("path_objective_horizons_s") is None
                    else list(config["path_objective_horizons_s"])
                ),
                "nominal_coverage": config["nominal_coverage"],
                "band_source": "unmasked generated joint paths over each full declared objective prefix",
                "band_formula": "median plus or minus nominal quantile of maximum standardized path excursion; lower clipped at zero",
                "observation_mask": "score width and worst miss only at known leads; does not change band geometry",
                "energy_mask": "unchanged known-coordinate joint distance and independent pair score",
                "categorical_costs": "same issued geometry for full sample and every leave-group-out recomputation",
                "unknown_targets": "masked before target arithmetic; no invented suffix values",
                "unknown_rows": "zero evidence-driven path terms",
                "generated_paths": "full prefix finite and nonnegative including unobserved target leads",
                "selection_policy": "same issued-prefix geometry in Validation joint checkpoint criterion",
                "inference_change": False,
                "parameter_count_change": False,
            },
        })
    red_settings = _red_corridor_objective_settings(config)
    if red_settings["enabled"]:
        fields["red_corridor_objective_contract"] = {
            "kind": "finite_sample_empirical_compatible_bracket_decision_prior",
            "width_weight": red_settings["red_corridor_width_weight"],
            "miss_weight": red_settings["red_corridor_miss_weight"],
            "scale_s": red_settings["red_corridor_scale_s"],
            "cadence_s": red_settings["cadence_s"],
            "scale_steps": red_settings["scale_steps"],
            "horizons_s": list(config["horizons_s"]),
            "objective_prefix_horizons_s": (
                [config["horizons_s"][-1]]
                if config.get("event_objective_horizons_s") is None
                else list(config["event_objective_horizons_s"])
            ),
            "prefix_aggregation": "equal mean of fixed event objective prefixes",
            "nominal_coverage": config["nominal_coverage"],
            "sample_count": config["training_samples"],
            "sample_law": "existing unconditional IID full-grid event marginal; reuse draws; no evidence conditioning or new RNG",
            "empirical_interval": "lower-interpolated alpha quantile of bracket left edges; higher-interpolated 1-alpha quantile of right edges; alpha=(1-coverage)/2",
            "brackets": "finite class j has [j,j+1]; tail collapses to survival; survival [L+1,L+1] is numeric scoring only",
            "width_formula": "max(upper-lower,0) divided by scale_steps",
            "miss_formula": "minimum over compatible brackets of (lower-left)+ plus (right-upper)+, divided by scale_steps",
            "evidence": "recorded event priority; contiguous no-entry prefix admits unknown suffix; tail probability summed and admissibility OR collapsed",
            "width_population": "currently at-risk causal origins, including unknown future; equal physical-unit total weight independently of future availability",
            "miss_population": "same equal physical-unit known at-risk population as compatible event NLL; unknown and alreadyRED miss zero",
            "already_RED": "both prior terms and gradients zero",
            "categorical_gradient": "full marginal float64 log-softmax E score; disjoint leave-group-out same-row baselines; sum scores without division by S; S2/3 no baseline",
            "rounded_sampler_caveat": "real-arithmetic IID common-law theorem; ordinary dtype/joint marginal rounding is not bitwise equality",
            "probability_anchor": "existing compatible/censored first-entry NLL unchanged",
            "selection_policy": "same two prior terms and population weights included in Validation joint checkpoint criterion",
            "interpretation": "optimistic compatible whole-bracket lower bound for the same empirical interval plus model-only width prior; not continuous-point or exact-runtime interval risk",
            "survival_collapse_risk": "all-survival draws have zero width and may zero compatible miss under censoring; lower training loss alone does not establish warning quality",
            "proper_or_calibrated_population_risk": False,
            "runtime_exact_CDF_decoder_changed": False,
            "parameter_count_change": False,
        }
    finite_bound_weight = _red_finite_bound_weight(config)
    if finite_bound_weight > 0:
        alpha = (1 - config["nominal_coverage"]) / 2
        fields["red_finite_bound_contract"] = {
            "kind": "positive_only_finite_upper_confidence_prior",
            "weight": finite_bound_weight,
            "nominal_coverage": config["nominal_coverage"],
            "alpha": alpha,
            "confidence_margin": alpha / 2,
            "horizons_s": list(config["horizons_s"]),
            "objective_prefix_horizons_s": (
                [config["horizons_s"][-1]]
                if config.get("event_objective_horizons_s") is None
                else list(config["event_objective_horizons_s"])
            ),
            "prefix_aggregation": "equal mean of fixed event objective prefixes",
            "formula": "relu(logsumexp(z[L:])-logsumexp(z[:L])-logit(alpha/2))",
            "arithmetic": "float64 shared detached max centering, then detached per-group centering for each logsumexp; reject nonfinite derived values",
            "law": "full unconditional event marginal, not sampled or evidence-conditioned; no new RNG",
            "eligibility": "known currently-at-risk evidence with every admitted original class strictly inside L",
            "evidence": "observed priority; contiguous no-entry prefix; prefix admissibility OR tail, probability SUM tail; uninterrupted recorded-grid evidence upstream",
            "population": "same equal physical-unit known at-risk weights as compatible event NLL; excluded prefixes zero",
            "exclusions": "unknown, censored/tail-compatible, straddling, event outside prefix and alreadyRED cost and logit gradients zero",
            "margin": "interior confidence prior separate from unchanged runtime availability threshold; alpha>2e-8",
            "probability_anchor": "compatible and censored first-entry NLL unchanged",
            "selection_policy": "same term and event population weights in Validation joint checkpoint criterion",
            "limitations": "not proper or calibrated; no timing/width/lead guarantee; may increase false alerts; cannot repair saturated upstream Jacobians or falsely gap-spanning labels; tiny coordinates may vanish",
            "proper_or_calibrated_population_risk": False,
            "runtime_exact_CDF_decoder_changed": False,
            "parameter_count_change": False,
        }
    cdf_settings = _event_cdf_objective_settings(config)
    if cdf_settings["enabled"]:
        fields["event_cdf_objective_contract"] = {
            "kind": "deterministic_known_cdf_timing_auxiliary",
            "weight": cdf_settings["event_cdf_weight"],
            "scale_s": cdf_settings["event_cdf_scale_s"],
            "cadence_s": cdf_settings["cadence_s"],
            "scale_steps": cdf_settings["scale_steps"],
            "horizons_s": list(config["horizons_s"]),
            "objective_prefix_horizons_s": (
                [config["horizons_s"][-1]]
                if config.get("event_objective_horizons_s") is None
                else list(config["event_objective_horizons_s"])
            ),
            "prefix_aggregation": "equal mean of fixed event objective prefixes",
            "formula": "sum_j known_j*(F_j-label_j)^2/fixed_scale_steps",
            "law": "full unconditional event marginal; F_j=sum(q[:j+1]); original L..H classes summed as tail; never conditional renormalization",
            "arithmetic": "float64 softmax after shared detached max centering; reject nonfinite derived values; no floors or clamps",
            "targets": "zero if every admitted class exceeds j; one if every admitted class is <=j; otherwise unknown",
            "evidence": "observed priority; positive uninterrupted no-entry count admits c..H; c0 unknown ignores supplied bits; upstream recorded-grid continuity required",
            "exclusions": "unknown bits and alreadyRED sanitized before subtraction; cost and logit gradients zero",
            "normalization": "fixed physical time scale only; no known-count, prefix-length, outcome or follow-up denominator",
            "population": "same equal physical-unit known at-risk weights as compatible event NLL; no eligible-only renormalization",
            "probability_anchor": "compatible and censored first-entry NLL, MC RED width/miss and finite-bound prior unchanged",
            "selection_policy": "same auxiliary and event population weights included in Validation joint checkpoint criterion",
            "limitations": "outcome/censor-dependent masking is not automatically proper or calibrated; softmax underflow and upstream saturation can erase timing gradients; no warning usefulness guarantee",
            "proper_or_calibrated_population_risk": False,
            "new_random_draws": False,
            "runtime_exact_CDF_decoder_changed": False,
            "parameter_count_change": False,
        }
    return fields


def _validate_event_distribution_metadata(fields, expected):
    mandatory = (expected["event_distribution"] == "finite_horizon_mixture"
                 or expected.get("recurrent_context_mode") == "variable_causal"
                 or "post_factor_smoothness_contract" in expected
                 or "path_band_geometry_contract" in expected
                 or "red_corridor_objective_contract" in expected
                 or "red_finite_bound_contract" in expected
                 or "event_cdf_objective_contract" in expected
                 or "optimizer_sampling_contract" in expected)
    for key, value in expected.items():
        if (mandatory and key not in fields) or (key in fields and fields[key] != value):
            label = key.replace("_", " ")
            raise ValueError(f"Learned {label} missing or differs from saved config")
    if ("optimizer_sampling_contract" not in expected
            and "optimizer_sampling_contract" in fields):
        raise ValueError("Unexpected learned optimizer sampling contract")
    # Older single-path bundles lack these optional fields. A contradictory
    # coupled label must still fail even when the config reconstructs single.
    if "path_distribution" not in expected:
        if fields.get("path_distribution", "single") != "single":
            raise ValueError("Learned path distribution differs from saved config")
        if fields.get("path_components", 1) != 1 or "path_distribution_contract" in fields:
            raise ValueError("Learned path components differ from saved config")
    else:
        contract = expected["path_distribution_contract"]
        for key in ("path_head_outputs", "phase_covariance"):
            if key in fields and fields[key] != contract[key]:
                raise ValueError(f"Learned {key.replace('_', ' ')} differs from path contract")
    if "recurrent_context_mode" not in expected:
        if (fields.get("recurrent_context_mode", "fixed") != "fixed"
                or "recurrent_context_contract" in fields):
            raise ValueError("Learned recurrent context differs from saved config")
    if ("post_factor_smoothness_contract" not in expected
            and "post_factor_smoothness_contract" in fields):
        raise ValueError("Learned post factor smoothness differs from saved config")
    if "path_band_geometry_contract" not in expected:
        if (fields.get("path_band_geometry", "observed_prefix") != "observed_prefix"
                or "path_band_geometry_contract" in fields):
            raise ValueError("Learned path band geometry differs from saved config")
    if ("red_corridor_objective_contract" not in expected
            and "red_corridor_objective_contract" in fields):
        raise ValueError("Learned RED corridor objective differs from saved config")
    if ("red_finite_bound_contract" not in expected
            and "red_finite_bound_contract" in fields):
        raise ValueError("Learned RED finite-bound objective differs from saved config")
    if ("event_cdf_objective_contract" not in expected
            and "event_cdf_objective_contract" in fields):
        raise ValueError("Learned known-CDF objective differs from saved config")


def save_learned_bundle(bundle, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(bundle["model"].state_dict(), directory / "checkpoint.pt")
    metadata = {
        key: bundle[key]
        for key in ("config", "scaler", "external_scaler", "selection", "feature_names")
    }
    metadata["input_size"] = (
        bundle["model"].encoder.input_size
        if hasattr(bundle["model"].encoder, "input_size")
        else len(bundle["feature_names"])
    )
    metadata["external_feature_size"] = bundle["model"].external_feature_size
    metadata["phase_covariance"] = bundle["model"].phase_covariance
    metadata.update(_event_distribution_metadata(bundle["model"], bundle["config"]))
    metadata["path_head_outputs"] = bundle["model"].path_head[-1].out_features
    if bundle["encoder"] is not None:
        joblib.dump(bundle["encoder"], directory / "encoder.joblib")
    atomic_write_json(directory / "learned_model.json", metadata)
    atomic_write_json(directory / "objective_trace.json", bundle["trace"])
    names = ["checkpoint.pt", "learned_model.json", "objective_trace.json"] + (
        ["encoder.joblib"] if bundle["encoder"] else []
    )
    return {name: sha256_file(directory / name) for name in names}


def load_learned_bundle(run):
    directory = Path(run["dir"])
    required = {"checkpoint.pt", "learned_model.json", "objective_trace.json"}
    if not required.issubset(run["artifacts"]):
        raise ValueError("Learned artifact hashes missing")
    for name, digest in run["artifacts"].items():
        if sha256_file(directory / name) != digest:
            raise ValueError(f"Learned artifact hash mismatch: {name}")
    metadata = json.loads((directory / "learned_model.json").read_text())
    if metadata["config"] != run["params"]:
        raise ValueError("Learned config differs from manifest")
    saved_rates = metadata["selection"].get("optimizer_rates")
    if saved_rates is not None and saved_rates != _optimizer_rate_metadata(metadata["config"]):
        raise ValueError("Learned optimizer rates differ from saved config")
    if "scaler" in run and metadata["scaler"] != run["scaler"]:
        raise ValueError("Learned scaler differs from manifest")
    external_size = metadata["external_feature_size"]
    has_encoder = (directory / "encoder.joblib").is_file()
    if external_size is not None and ("encoder.joblib" not in run["artifacts"] or not has_encoder):
        raise ValueError("External learned encoder requires a verified artifact")
    if external_size is None and has_encoder:
        raise ValueError("Unexpected external encoder for recurrent learned model")
    model = _make_model(
        run["engine_id"],
        metadata["config"],
        metadata["input_size"],
        metadata["external_feature_size"],
    )
    # Optional provenance fields keep older bundles loadable. New bundles bind
    # the saved head size and covariance mode to their reconstruction config.
    if metadata.get("phase_covariance", model.phase_covariance) != model.phase_covariance:
        raise ValueError("Learned phase covariance differs from saved config")
    event_metadata = _event_distribution_metadata(model, metadata["config"])
    _validate_event_distribution_metadata(metadata, event_metadata)
    sampling_contract = event_metadata.get("optimizer_sampling_contract")
    if sampling_contract is not None:
        selection = metadata["selection"]
        if selection.get("optimizer_sampling_contract") != sampling_contract:
            raise ValueError("Learned optimizer sampling selection differs from saved config")
        if selection.get("batch_sampling") != "physical_event_stratified" or selection.get("optimizer_sampling_policy") != "physical_event_stratified_inverse_probability_term_means":
            raise ValueError("Learned optimizer sampling policy differs from saved config")
        _validate_sampling_train_metadata(selection.get("optimizer_sampling_train"), metadata["config"], metadata["scaler"])
        if ("schema_version" in run or "selection" in run) and run.get("selection") != selection:
            raise ValueError("Learned optimizer sampling full-run selection differs from checkpoint")
        if "schema_version" in run and not {"training_contract.json", "model_input_contract.json"}.issubset(run["artifacts"]):
            raise ValueError("Learned optimizer sampling full run requires both verified contracts")
    elif any(key in metadata["selection"] for key in ("optimizer_sampling_contract", "optimizer_sampling_train")):
        raise ValueError("Unexpected learned optimizer sampling selection")
    regularization_contract = event_metadata.get("post_factor_smoothness_contract")
    if regularization_contract is not None:
        if metadata["selection"].get("post_factor_smoothness_contract") != regularization_contract:
            raise ValueError("Learned post factor smoothness checkpoint criterion differs from saved config")
        if ("schema_version" in run or "selection" in run) and run.get("selection") != metadata["selection"]:
            raise ValueError("Learned post factor smoothness full-run selection differs from saved checkpoint")
    elif "post_factor_smoothness_contract" in metadata["selection"]:
        raise ValueError("Unexpected learned post factor smoothness checkpoint criterion")
    band_contract = event_metadata.get("path_band_geometry_contract")
    if band_contract is not None:
        if "schema_version" in run and not {
            "training_contract.json", "model_input_contract.json"
        }.issubset(run["artifacts"]):
            raise ValueError("Learned path band geometry full run requires both verified contracts")
        if metadata["selection"].get("path_band_geometry_contract") != band_contract:
            raise ValueError("Learned path band geometry checkpoint criterion differs from saved config")
        if ("schema_version" in run or "selection" in run) and run.get("selection") != metadata["selection"]:
            raise ValueError("Learned path band geometry full-run selection differs from saved checkpoint")
    elif "path_band_geometry_contract" in metadata["selection"]:
        raise ValueError("Unexpected learned path band geometry checkpoint criterion")
    red_contract = event_metadata.get("red_corridor_objective_contract")
    if red_contract is not None:
        if "schema_version" in run and not {
            "training_contract.json", "model_input_contract.json"
        }.issubset(run["artifacts"]):
            raise ValueError("Learned RED corridor objective full run requires both verified contracts")
        if metadata["selection"].get("red_corridor_objective_contract") != red_contract:
            raise ValueError("Learned RED corridor objective checkpoint criterion differs from saved config")
        if ("schema_version" in run or "selection" in run) and run.get("selection") != metadata["selection"]:
            raise ValueError("Learned RED corridor objective full-run selection differs from saved checkpoint")
    elif "red_corridor_objective_contract" in metadata["selection"]:
        raise ValueError("Unexpected learned RED corridor objective checkpoint criterion")
    finite_contract = event_metadata.get("red_finite_bound_contract")
    if finite_contract is not None:
        if "schema_version" in run and not {
            "training_contract.json", "model_input_contract.json"
        }.issubset(run["artifacts"]):
            raise ValueError("Learned RED finite-bound full run requires both verified contracts")
        if metadata["selection"].get("red_finite_bound_contract") != finite_contract:
            raise ValueError("Learned RED finite-bound checkpoint criterion differs from saved config")
        if ("schema_version" in run or "selection" in run) and run.get("selection") != metadata["selection"]:
            raise ValueError("Learned RED finite-bound full-run selection differs from saved checkpoint")
    elif "red_finite_bound_contract" in metadata["selection"]:
        raise ValueError("Unexpected learned RED finite-bound checkpoint criterion")
    cdf_contract = event_metadata.get("event_cdf_objective_contract")
    if cdf_contract is not None:
        if "schema_version" in run and not {
            "training_contract.json", "model_input_contract.json"
        }.issubset(run["artifacts"]):
            raise ValueError("Learned known-CDF full run requires both verified contracts")
        if metadata["selection"].get("event_cdf_objective_contract") != cdf_contract:
            raise ValueError("Learned known-CDF checkpoint criterion differs from saved config")
        if ("schema_version" in run or "selection" in run) and run.get("selection") != metadata["selection"]:
            raise ValueError("Learned known-CDF full-run selection differs from saved checkpoint")
    elif "event_cdf_objective_contract" in metadata["selection"]:
        raise ValueError("Unexpected learned known-CDF checkpoint criterion")
    # Full run manifests and verified contracts carry the same interpretation.
    # Minimal saved bundles need only their mandatory learned-model metadata.
    if any(key in run for key in (
        "schema_version", "event_distribution", "path_distribution", "path_components",
        "path_distribution_contract", "recurrent_context_mode", "recurrent_context_contract",
        "post_factor_smoothness_contract",
        "path_band_geometry", "path_band_geometry_contract",
        "red_corridor_objective_contract",
        "red_finite_bound_contract",
        "event_cdf_objective_contract", "optimizer_sampling_contract",
    )):
        _validate_event_distribution_metadata(run, event_metadata)
    for name in ("training_contract.json", "model_input_contract.json"):
        if name in run["artifacts"]:
            saved_contract = json.loads((directory / name).read_text())
            _validate_event_distribution_metadata(saved_contract, event_metadata)
            if sampling_contract is not None and "params" in saved_contract and saved_contract["params"] != metadata["config"]:
                raise ValueError("Learned optimizer sampling contract parameters differ from saved config")
            if (regularization_contract is not None and "params" in saved_contract
                    and saved_contract["params"] != metadata["config"]):
                raise ValueError("Learned post factor smoothness contract parameters differ from saved config")
            if (band_contract is not None and "params" in saved_contract
                    and saved_contract["params"] != metadata["config"]):
                raise ValueError("Learned path band geometry contract parameters differ from saved config")
            if (red_contract is not None and "params" in saved_contract
                    and saved_contract["params"] != metadata["config"]):
                raise ValueError("Learned RED corridor objective contract parameters differ from saved config")
            if (finite_contract is not None and "params" in saved_contract
                    and saved_contract["params"] != metadata["config"]):
                raise ValueError("Learned RED finite-bound contract parameters differ from saved config")
            if (cdf_contract is not None and "params" in saved_contract
                    and saved_contract["params"] != metadata["config"]):
                raise ValueError("Learned known-CDF contract parameters differ from saved config")
    if metadata.get("path_head_outputs", model.path_head[-1].out_features) != model.path_head[-1].out_features:
        raise ValueError("Learned path head size differs from saved config")
    if model.path_distribution == "coupled_timing_mixture" and "path_head_outputs" not in metadata:
        raise ValueError("Learned path head size missing from coupled metadata")
    model.load_state_dict(
        torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    )
    model.eval()
    encoder = (
        joblib.load(directory / "encoder.joblib")
        if has_encoder
        else None
    )
    expected_kind = {"full_cns": "full_cns", "quantile_boosting": "quantile_boosting"}.get(run["engine_id"])
    if (encoder is not None and encoder.get("kind") != expected_kind) or (encoder is None and expected_kind is not None):
        raise ValueError("Saved external encoder differs from requested engine")
    if (
        encoder
        and encoder["kind"] == "full_cns"
        and encoder["body"].provenance != run["connectome"]
    ):
        raise ValueError("Full connectome provenance differs from manifest")
    return {**metadata, "model": model, "encoder": encoder, "trace": []}


def train_learned_run(project_id, data, engine_id, params, stop=None, report=None):
    from pdm.projects import project_store
    from pdm.signal_training import _physical_map
    from pdm.trajectory_data import build_trajectory_frame
    from pdm.trajectory_evaluation import evaluate_trajectory_model

    report = report or (lambda _: None)
    _physical_map(data)
    training = data["features"][
        data["features"].unit_id.astype(str).isin(map(str, data["split"]["train"]))
    ]
    config = learned_params(engine_id, params, training)
    train = build_trajectory_frame(
        data, data["split"]["train"], config, cap=config["max_windows_per_unit"]
    )
    validation = build_trajectory_frame(
        data, data["split"]["validation"], config, cap=config["max_windows_per_unit"]
    )
    store = project_store()
    bundle = fit_learned_model(
        engine_id,
        train,
        validation,
        config,
        stop=stop,
        report=report,
        cache_dir=store.root / "learned_trajectory_cache",
    )
    _check(stop)
    run_id = "signal-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    directory = store.run_path(project_id, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    artifacts = save_learned_bundle(bundle, directory)
    contract = {
        "project_id": project_id,
        "snapshot_id": data["snapshot_id"],
        "engine_id": engine_id,
        "params": config,
        "schema": data["schema"],
        "snapshot_fingerprint_sha256": _digest(data["fingerprint"]),
        "scaler": bundle["scaler"],
        **_event_distribution_metadata(bundle["model"], config),
    }
    manifest = {
        **contract,
        "schema_version": "project_signal_forecast_learned_v1",
        "task": "signal_forecast",
        "status": "completed",
        "run_id": run_id,
        "artifact": "checkpoint.pt",
        "artifacts": artifacts,
        "selection": bundle["selection"],
        "interval_status": "learned_uncalibrated_simultaneous_band",
        "funnel": {
            "mode": MODE,
            "probabilistic_architecture_claim": True,
            "nominal_coverage": config["nominal_coverage"],
            "path_samples": config["path_samples"],
            "operational_coverage_approved": False,
        },
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    contract["funnel"] = manifest["funnel"]
    if engine_id == "full_cns":
        manifest["connectome"] = bundle["encoder"]["body"].provenance
        contract["connectome"] = manifest["connectome"]
    atomic_write_json(directory / "training_contract.json", contract)
    artifacts["training_contract.json"] = sha256_file(directory / "training_contract.json")
    atomic_write_json(
        directory / "model_input_contract.json",
        {
            "features": bundle["feature_names"],
            "fit_partition": "Train",
            "architecture": engine_id,
            "encoder_adapter": bundle["encoder"]["readout_policy"]
            if bundle["encoder"] and engine_id == "quantile_boosting"
            else "complete source connectome plus learned probabilistic readout"
            if engine_id == "full_cns"
            else "recurrent learned probabilistic encoder",
            "targets": "joint future nonnegative RMS path and first recorded-grid RED entry",
            "selection_partition": "Validation",
            "test_feedback": False,
            "unknown_suffix_policy": "masked path targets and censored RED likelihood; never negative padding",
            "normalization": "Train-only physical group balanced",
            "warning_distribution": "first entry mixture with horizon survival class",
            "phase_covariance": bundle["model"].phase_covariance,
            **_event_distribution_metadata(bundle["model"], config),
            "path_head_outputs": bundle["model"].path_head[-1].out_features,
            "coverage_guarantee": False,
        },
    )
    artifacts["model_input_contract.json"] = sha256_file(directory / "model_input_contract.json")
    loaded = load_learned_bundle({**manifest, "dir": directory})
    audit = _subset(validation, np.arange(min(3, len(validation["x"]))))
    before = predict_learned(bundle, audit, stop=stop)
    after = predict_learned(loaded, audit, stop=stop)
    if any(not np.array_equal(before[key], after[key]) for key in before):
        raise AssertionError("Saved learned distribution reload differs")
    manifest["reload_verified"] = True
    report({"stage": "evaluating", "message": "Scoring restored model; Test cannot select weights"})
    validation_all = build_trajectory_frame(data, data["split"]["validation"], config)
    validation_all = prepare_prediction_frame(
        loaded, validation_all, store.root / "learned_trajectory_cache", stop
    )
    val_metrics = evaluate_trajectory_model(
        lambda batch: predict_learned(loaded, batch, stop=stop), validation_all, config,
        partition="Validation",
    )
    _check(stop)
    test = build_trajectory_frame(data, data["split"]["test"], config)
    test = prepare_prediction_frame(loaded, test, store.root / "learned_trajectory_cache", stop)
    test_metrics = evaluate_trajectory_model(
        lambda batch: predict_learned(loaded, batch, stop=stop), test, config,
        partition="Test",
    )
    manifest["metrics"] = {"validation": val_metrics, "test": test_metrics}
    _check(stop)
    atomic_write_json(directory / "manifest.json", manifest)
    store.update(project_id, selected_run_id=run_id)
    report(
        {
            "stage": "completed",
            "progress": 1.0,
            "message": "Learned trajectory run saved and reload verified",
        }
    )
    return manifest


def forecast_learned_prefix(run, data, prefix, unit_id, stop=None, *, prediction_horizon_s=None):
    from pdm.project_zones import resolve_thresholds
    from pdm.trajectory_data import build_trajectory_prefix

    config = run["params"]
    saved_horizons = np.asarray(config["horizons_s"], float)
    count = len(saved_horizons)
    if prediction_horizon_s is not None:
        requested = float(prediction_horizon_s)
        if not np.isfinite(requested) or requested < saved_horizons[0] or requested > saved_horizons[-1]:
            raise ValueError("Prediction span must be within the saved dense horizon")
        count = int(np.searchsorted(saved_horizons, requested, side="right"))
    issued = float(prefix.timestamp_s.iloc[-1]) if len(prefix) else None
    result = {
        "as_of_s": issued,
        "points": [],
        "sampled_paths": [],
        "project_id": run["project_id"],
        "run_id": run["run_id"],
        "snapshot_id": run["snapshot_id"],
        "status": "unavailable",
        "reason": None,
        "observed_prefix": [
            {"timestamp_s": float(t), "signal": float(v)}
            for t, v in zip(prefix.timestamp_s, prefix.signal)
        ],
        "thresholds": resolve_thresholds(run["schema"], prefix),
        "crossing": {"status": "unavailable", "time_s": None},
        "calibration_status": "learned_uncalibrated_simultaneous_band",
        "red_entry_corridor": {"status": "unavailable", "reason": "Insufficient observed history"},
    }
    try:
        frame = build_trajectory_prefix(data, prefix, config)
    except ValueError as exc:
        if "observations" not in str(exc).lower() and "empty" not in str(exc).lower():
            raise
        result["reason"] = str(exc)
        return result
    if not len(frame["x"]):
        result["reason"] = f"Need {config['history_length']} observations since last gap"
        return result
    output = predict_learned(load_learned_bundle(run), frame, stop=stop)
    hs = saved_horizons[:count]
    if count < len(saved_horizons):
        # Issue a band for this declared horizon, matching the prefix-horizon
        # evaluation. The model still samples its immutable full joint grid.
        output["paths"] = output["paths"][:, :, :count]
        lower, upper = SignalDistribution.simultaneous_band(
            torch.as_tensor(output["paths"]), config["nominal_coverage"]
        )
        output["lower"], output["upper"] = lower.numpy(), upper.numpy()
        output["mean"] = output["mean"][:, :count]
        probabilities = output["event_probabilities"]
        output["event_probabilities"] = np.column_stack(
            [probabilities[:, :count], probabilities[:, count:].sum(1)]
        )
    current = float(prefix.signal.iloc[-1])
    prob = output["event_probabilities"][0]
    points = [
        {
            "target_time_s": issued,
            "value": current,
            "lower": current,
            "upper": current,
            "kind": "anchor",
        }
    ]
    points += [
        {
            "target_time_s": issued + float(h),
            "value": float(output["mean"][0, j]),
            "lower": float(output["lower"][0, j]),
            "upper": float(output["upper"][0, j]),
            "kind": "direct",
        }
        for j, h in enumerate(hs)
    ]
    from pdm.trajectory_evaluation import entry_corridor

    already = current >= float(frame["red_threshold"][0])
    corridor = entry_corridor(prob, hs, config["nominal_coverage"], issued, already)
    if not already and (prefix.signal >= float(frame["red_threshold"][0])).any():
        corridor["status"] = "previously_red"
    hit = next(
        (p["target_time_s"] for p in points[1:] if p["value"] >= frame["red_threshold"][0]), None
    )
    result.update(
        status="available",
        points=points,
        sampled_paths=np.concatenate(
            [np.full((len(output["paths"]), 1), current), output["paths"][:, 0, :]], axis=1
        ).tolist(),
        red_entry_corridor=corridor,
        crossing={
            "status": "already_red"
            if already
            else "predicted"
            if hit is not None
            else "none_within_horizon",
            "time_s": issued if already else hit,
        },
        funnel={
            "mode": MODE,
            "nominal_coverage": config["nominal_coverage"],
            "simultaneous": True,
            "coverage_guarantee": False,
            "issued_horizon_s": float(hs[-1]),
            "band_scope": "full_saved_horizon" if count == len(saved_horizons) else "declared_prefix_horizon",
        },
    )
    return result
