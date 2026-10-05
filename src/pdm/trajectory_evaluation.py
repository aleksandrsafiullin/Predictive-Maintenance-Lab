"""Dense recorded-path and RED-bracket evaluation, balanced by physical equipment."""

from __future__ import annotations

import numpy as np
import torch

from pdm.trajectory_data import slice_frame
from pdm.trajectory_objectives import simultaneous_band


def _macro(values, admitted, groups):
    means = [
        float(np.mean(values[admitted & (groups == group)]))
        for group in np.unique(groups)
        if np.any(admitted & (groups == group))
    ]
    return float(np.mean(means)) if means else None


def _band(paths, coverage):
    # Use the exact same dtype, median convention and envelope as saved replay.
    lower, upper = simultaneous_band(torch.as_tensor(paths, dtype=torch.float32), coverage)
    return lower.numpy(), upper.numpy()


def _corridor(probabilities, horizons, coverage):
    mass = probabilities.sum(axis=1)
    conditional = np.divide(
        probabilities, mass[:, None], out=np.zeros_like(probabilities), where=mass[:, None] > 0
    )
    cdf = np.cumsum(conditional, axis=1)
    alpha = (1 - coverage) / 2
    left = np.argmax(cdf >= alpha, axis=1)
    right = np.argmax(cdf >= 1 - alpha - 1e-7, axis=1)
    edges = np.r_[0.0, horizons]
    return edges[left], horizons[right], mass


def entry_corridor(probabilities, horizons, coverage=0.9, issued=0.0, already_red=False):
    probabilities = np.asarray(probabilities, float)
    horizons = np.asarray(horizons, float)
    if probabilities.shape != (len(horizons) + 1,) or not np.isclose(
        probabilities.sum(), 1, atol=1e-5
    ):
        raise ValueError("Entry distribution must include horizon survival")
    cdf = probabilities.cumsum()
    alpha = (1 - coverage) / 2
    left, right = (int(np.searchsorted(cdf, q)) for q in (alpha, 1 - alpha))
    if probabilities[-1] >= alpha-1e-8:
        right=len(horizons)
    edges = np.r_[0.0, horizons]
    lower = float(issued + edges[left]) if left < len(horizons) else None
    upper = float(issued + horizons[right]) if right < len(horizons) else None
    conditional_left, conditional_right, mass = _corridor(
        probabilities[None, :-1], horizons, coverage
    )
    return {
        "status": "already_red" if already_red else "learned",
        "earliest_s": float(issued) if already_red else lower,
        "latest_s": float(issued) if already_red else upper,
        "probability_within_horizon": None if already_red else float(mass[0]),
        "event_probability": None if already_red else float(mass[0]),
        "no_entry_probability": None if already_red else float(probabilities[-1]),
        "right_censored": bool(upper is None),
        "nominal_coverage": coverage,
        "conditional_earliest_s": float(issued + conditional_left[0]),
        "conditional_latest_s": float(issued + conditional_right[0]),
        "calibration_status": "learned_uncalibrated",
        "conditioning": "unconditional_including_no_entry",
        "time_definition": "first future recorded-grid entry bracket",
        "observed_future_used": False,
    }


def evaluate_trajectory_model(predict_fn, frame, config, *, partition="unspecified"):
    """Forecast first, then inspect dense targets; incomplete paths are never successes.

    Fixed prefix-band summaries explicitly recalculate the simultaneous envelope
    for that declared prediction horizon. The full saved-grid envelope is also
    evaluated. Neither makes a coverage guarantee from three reused units.
    """
    if partition not in {"Train", "Validation", "Test", "unspecified"}:
        raise ValueError("Unknown evaluation partition")
    horizons = np.asarray(config["horizons_s"], float)
    n, h = frame["y"].shape
    groups = np.asarray(frame["physical_unit_id"], str)
    cover = float(config["nominal_coverage"])
    cutoffs = sorted(
        set(
            [
                min(h, int(np.searchsorted(horizons, minute * 60, side="right")))
                for minute in (10, 20, 30, 60, 120)
            ]
            + [h]
        )
        - {0}
    )
    records = {
        k: {
            name: np.zeros(n)
            for name in (
                "whole",
                "point",
                "width",
                "score",
                "mae",
                "baseline_mae",
                "baseline_whole",
                "baseline_score",
                "complete",
                "known",
                "event_known",
                "event_positive",
                "event_brier",
                "event_log",
                "corridor_contains",
                "corridor_overlap",
                "corridor_width",
                "probability",
                "alert",
                "useful_warning",
                "already_red",
                "energy",
                "energy_physical",
                "corridor_finite",
                "conditional_contains",
                "conditional_width",
            )
        }
        for k in cutoffs
    }
    per_horizon_known = frame["mask"].sum(axis=0).astype(int)
    event_target = np.where(frame["event_observed"], np.argmax(frame["event_allowed"], axis=1), h)
    warning = np.asarray(frame["warning_eligible"], bool)
    batch_size = 16
    for start in range(0, n, batch_size):
        indexes = np.arange(start, min(start + batch_size, n))
        batch = slice_frame(frame, indexes)
        output = predict_fn(batch)
        paths = np.asarray(output["paths"], float)
        probabilities = np.asarray(output["event_probabilities"], float)
        if paths.ndim != 3 or paths.shape[1:] != batch["y"].shape or not np.isfinite(paths).all():
            raise ValueError("Invalid saved joint-path predictions")
        if probabilities.shape != (len(indexes), h + 1) or not np.allclose(
            probabilities.sum(1), 1, atol=1e-5
        ):
            raise ValueError("Entry probabilities must retain a normalized no-entry class")
        for k in cutoffs:
            row = records[k]
            sampled = paths[:, :, :k]
            if k == h:
                lo, hi = np.asarray(output["lower"]), np.asarray(output["upper"])
            else:
                lo, hi = _band(sampled, cover)
            mask = batch["mask"][:, :k]
            target = np.where(mask, batch["y"][:, :k], 0)
            if not np.isfinite(target).all() or (target < 0).any():
                raise ValueError("Observed trajectory evaluation targets must be finite and nonnegative")
            count = np.maximum(mask.sum(1), 1)
            complete, known = mask.all(1), mask.any(1)
            inside = (target >= lo) & (target <= hi)
            width = hi - lo
            score = width + 2 / (1 - cover) * (
                np.maximum(lo - target, 0) + np.maximum(target - hi, 0)
            )
            median = np.median(sampled, axis=0)
            current = np.asarray(batch["current"])[:, None]
            # Predeclared useful-width baseline: last value +/-0.75 g (1.5 g full width).
            base_lo, base_hi = np.maximum(current - 0.75, 0), current + 0.75
            base_score = (
                base_hi
                - base_lo
                + 2
                / (1 - cover)
                * (np.maximum(base_lo - target, 0) + np.maximum(target - base_hi, 0))
            )
            lp, truth = np.log1p(sampled), np.log1p(np.where(mask, target, 0))
            d = np.sqrt(np.sum((lp - truth) ** 2 * mask, axis=-1) / count)
            half = len(lp) // 2
            pair = np.sqrt(np.sum((lp[:half] - lp[half : 2 * half]) ** 2 * mask, axis=-1) / count)
            energy = d.mean(0) - 0.5 * pair.mean(0)
            red_scale = np.asarray(batch["red_threshold"], float)[None, :, None]
            scaled_paths = sampled / red_scale
            scaled_truth = target / red_scale[0]
            d_physical = np.sqrt(np.sum((scaled_paths - scaled_truth) ** 2 * mask, axis=-1) / count)
            pair_physical = np.sqrt(np.sum((scaled_paths[:half] - scaled_paths[half:2 * half]) ** 2 * mask, axis=-1) / count)
            assignments = {
                "complete": complete,
                "known": known,
                "whole": inside.all(1),
                "point": (inside * mask).sum(1) / count,
                "width": (width * mask).sum(1) / count,
                "score": (score * mask).sum(1) / count,
                "mae": (np.abs(median - target) * mask).sum(1) / count,
                "baseline_mae": (np.abs(current - target) * mask).sum(1) / count,
                "baseline_whole": ((target >= base_lo) & (target <= base_hi)).all(1),
                "baseline_score": (base_score * mask).sum(1) / count,
                "energy": energy,
                "energy_physical": d_physical.mean(0) - 0.5 * pair_physical.mean(0),
                "already_red": np.asarray(batch["current"]) >= np.asarray(batch["red_threshold"]),
            }
            for name, value in assignments.items():
                row[name][indexes] = value
            conditional_left, conditional_right, mass = _corridor(
                probabilities[:, :k], horizons[:k], cover
            )
            prefix_prob = np.column_stack([probabilities[:, :k], probabilities[:, k:].sum(1)])
            issued_corridors = [entry_corridor(p, horizons[:k], cover) for p in prefix_prob]
            left = np.asarray(
                [
                    c["earliest_s"] if c["earliest_s"] is not None else np.nan
                    for c in issued_corridors
                ]
            )
            right = np.asarray(
                [c["latest_s"] if c["latest_s"] is not None else np.nan for c in issued_corridors]
            )
            finite = np.isfinite(left) & np.isfinite(right)
            actual_index = event_target[indexes]
            positive = np.asarray(batch["event_observed"]) & (actual_index < k) & warning[indexes]
            event_known = warning[indexes] & (positive | complete)
            actual_left = np.r_[0.0, horizons][np.minimum(actual_index, h)]
            actual_right = horizons[np.minimum(actual_index, h - 1)]
            contained = finite & (left <= actual_left + 1e-7) & (right >= actual_right - 1e-7)
            overlap = (left < actual_right) & (right >= actual_left)
            brier = (mass - positive) ** 2
            observed_prob = np.where(positive, mass, 1 - mass)
            alert = (mass >= 0.5) & ((right - left) <= 900) & warning[indexes]
            useful = alert & contained & positive & (actual_left >= 600)
            for name, value in {
                "event_known": event_known,
                "event_positive": positive,
                "event_brier": brier,
                "event_log": -np.log(np.maximum(observed_prob, 1e-9)),
                "probability": mass,
                "corridor_contains": contained,
                "corridor_overlap": overlap,
                "corridor_width": np.where(finite, right - left, 0),
                "corridor_finite": finite,
                "conditional_contains": (conditional_left <= actual_left + 1e-7)
                & (conditional_right >= actual_right - 1e-7),
                "conditional_width": conditional_right - conditional_left,
                "alert": alert,
                "useful_warning": useful,
            }.items():
                row[name][indexes] = value
    summaries = []
    for k, row in records.items():
        complete, known = row["complete"].astype(bool), row["known"].astype(bool)
        event_known, positive = row["event_known"].astype(bool), row["event_positive"].astype(bool)
        unit_rows = []
        for group in np.unique(groups):
            group_rows = groups == group
            unit_rows.append(
                {
                    "physical_unit_id": group,
                    "origins": int(group_rows.sum()),
                    "complete_paths": int((complete & group_rows).sum()),
                    "whole_path_coverage": _macro(row["whole"], complete & group_rows, groups),
                    "mean_width_g": _macro(row["width"], known & group_rows, groups),
                    "event_positive_origins": int((positive & group_rows).sum()),
                    "event_has_useful_warning": bool(np.any(row["useful_warning"][group_rows])),
                    "corridor_bracket_coverage": _macro(
                        row["corridor_contains"], positive & group_rows, groups
                    ),
                }
            )
        signal_score = _macro(row["score"], known, groups)
        baseline_score = _macro(row["baseline_score"], known, groups)
        false_alerts = row["alert"].astype(bool) & event_known & ~positive
        false_episodes = 0
        at = np.asarray(frame["as_of_s"], float)
        units = np.asarray(frame["unit_id"], str)
        for unit in np.unique(units):
            indexes = np.flatnonzero(units == unit)
            indexes = indexes[np.argsort(at[indexes])]
            flags = false_alerts[indexes]
            contiguous = np.r_[
                False, np.isclose(np.diff(at[indexes]), config.get("cadence_s", horizons[0]))
            ]
            previous = np.r_[False, flags[:-1]] & contiguous
            false_episodes += int((flags & ~previous).sum())
        monitored_hours = float(event_known.sum() * config.get("cadence_s", horizons[0]) / 3600)
        summaries.append(
            {
                "horizon_s": float(horizons[k - 1]),
                "recorded_grid_points": k,
                "band_scope": "full_saved_horizon" if k == h else "declared_prefix_horizon",
                "complete_paths": int(complete.sum()),
                "incomplete_paths": int((~complete).sum()),
                "complete_physical_groups": int(len(np.unique(groups[complete]))),
                "whole_path_coverage": _macro(row["whole"], complete, groups),
                "point_coverage": _macro(row["point"], known, groups),
                "mean_width_g": _macro(row["width"], known, groups),
                "pre_red_mean_width_g": _macro(row["width"], known & warning, groups),
                "pre_red_whole_path_coverage": _macro(row["whole"], complete & warning, groups),
                "mean_width_red_span": _macro(row["width"], known, groups) / 1.5
                if known.any()
                else None,
                "interval_score_g": signal_score,
                "energy_log_signal": _macro(row["energy"], known, groups),
                "energy_signal_red_scaled": _macro(row["energy_physical"], known, groups),
                "point_mae": _macro(row["mae"], known, groups),
                "persistence_mae": _macro(row["baseline_mae"], known, groups),
                "baseline_whole_coverage": _macro(row["baseline_whole"], complete, groups),
                "baseline_interval_score_g": baseline_score,
                "interval_score_skill": 1 - signal_score / baseline_score
                if baseline_score
                else None,
                "red_event_positive_origins": int(positive.sum()),
                "red_event_known_origins": int(event_known.sum()),
                "red_event_unknown_origins": int((warning & ~event_known).sum()),
                "red_brier": _macro(row["event_brier"], event_known, groups),
                "red_log_score": _macro(row["event_log"], event_known, groups),
                "red_bracket_containment": _macro(row["corridor_contains"], positive, groups),
                "red_bracket_overlap": _macro(row["corridor_overlap"], positive, groups),
                "red_mean_corridor_width_s": _macro(
                    row["corridor_width"], positive & row["corridor_finite"].astype(bool), groups
                ),
                "red_finite_corridor_fraction": _macro(row["corridor_finite"], positive, groups),
                "conditional_red_bracket_coverage": _macro(
                    row["conditional_contains"], positive, groups
                ),
                "conditional_red_mean_width_s": _macro(row["conditional_width"], positive, groups),
                "independent_red_events": int(len(np.unique(groups[positive]))),
                "events_with_useful_warning": int(
                    sum(item["event_has_useful_warning"] for item in unit_rows)
                ),
                "by_physical_unit": unit_rows,
                "false_alert_episodes": false_episodes,
                "monitored_pre_red_hours": monitored_hours,
                "false_episodes_per_100_hours": 100 * false_episodes / monitored_hours
                if monitored_hours
                else None,
                "alert_time_fraction": float(row["alert"][event_known].mean())
                if event_known.any()
                else None,
            }
        )
    return {
        "status": "exploratory_reused_test" if partition == "Test" else "exploratory",
        "evaluation_partition": partition,
        "aggregation": "physical_equipment_equal_weight",
        "physical_group_count": int(len(np.unique(groups))),
        "origins": n,
        "nominal_coverage": cover,
        "coverage_guarantee": False,
        "dense_recorded_targets_by_horizon": per_horizon_known.tolist(),
        "warning_policy": {
            "probability_threshold": 0.5,
            "maximum_corridor_width_s": 900,
            "minimum_conservative_lead_s": 600,
            "policy_source": "predeclared_engineering_comparison",
        },
        "baseline": {"kind": "last_value_fixed_1.5g_band", "width_cap_g": 1.5},
        "horizons": summaries,
    }
