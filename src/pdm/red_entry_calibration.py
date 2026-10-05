"""Grouped hazard calibration and supported event-time quantiles for RED v2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pdm.red_entry_protocol import validate_horizons


def _hazards(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Hazards must be finite [N,K] conditional probabilities")
    return values


def hazard_to_cdf(hazards):
    """Conditional bin hazards produce a finite monotone cumulative distribution."""
    return 1.0 - np.cumprod(1.0 - _hazards(hazards), axis=1)


def event_quantiles(cdf, horizons_s, quantiles=(0.05, 0.5, 0.95)):
    """Conservative bin right-edge quantiles; no extrapolation outside support."""
    grid = validate_horizons(horizons_s)
    values = _hazards(cdf)
    if values.shape[1] != len(grid) or (np.diff(values, axis=1) < -1e-12).any():
        raise ValueError("CDF must match horizons and be monotone")
    if any(not 0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must be strictly between zero and one")
    return [[float(grid[np.flatnonzero(row >= q)[0]]) if (row >= q).any() else None
             for q in quantiles] for row in values]


def apply_hazard_calibrator(hazards, artifact):
    values = _hazards(hazards)
    temperature = float(artifact.get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Calibration temperature must be positive")
    if temperature == 1.0:
        return values.copy()
    clipped = np.clip(values, 1e-9, 1 - 1e-9)
    logits = np.log(clipped) - np.log1p(-clipped)
    calibrated = 1 / (1 + np.exp(-np.clip(logits / temperature, -700, 700)))
    return np.where(values == 0, 0, np.where(values == 1, 1, calibrated))


def fit_hazard_calibrator(hazards, targets, mask, physical_unit_ids, *, requirements=None,
                          calibration_role="dedicated_grouped_calibration",
                          selection_physical_ids=(), train_physical_ids=()):
    """Fit only disjoint calibration equipment, equal weighting each equipment.

    Independent event support is an explicit operational requirement. Missing
    requirements yield identity calibration, not a guessed production minimum.
    Temperature fit alone never validates interval coverage.
    """
    values = _hazards(hazards)
    y, known = np.asarray(targets, float), np.asarray(mask, bool)
    ids = np.asarray(physical_unit_ids, str)
    if y.shape != values.shape or known.shape != values.shape or ids.shape != (len(values),):
        raise ValueError("Calibration arrays must align")
    if not np.isfinite(y[known]).all() or not np.isin(y[known], [0, 1]).all():
        raise ValueError("Known hazard targets must be binary")
    if any(not uid.strip() or uid in {"None", "nan"} for uid in ids):
        raise ValueError("Calibration requires explicit physical equipment identities")
    groups = sorted(set(ids))
    if set(groups).intersection(
            set(map(str, selection_physical_ids)) | set(map(str, train_physical_ids))):
        raise ValueError("Calibration equipment must be dedicated and disjoint from fitting/selection")
    events = len(set(ids[np.any((y == 1) & known, axis=1)]))
    minimum = (requirements or {}).get("minimum_independent_event_evidence")
    if minimum is not None and (isinstance(minimum, bool) or not isinstance(minimum, int) or minimum < 1):
        raise ValueError("Minimum independent event evidence must be an explicit positive integer")
    artifact = {"version": "red_entry_hazard_temperature_v1", "status": "insufficient_event_evidence",
                "temperature": 1.0, "physical_unit_ids": groups,
                "independent_event_units": events, "minimum_independent_event_evidence": minimum,
                "calibration_role": calibration_role, "interval_support_validated": False,
                "weighting": "physical_unit_equal_masked_bin_log_loss"}
    if calibration_role != "dedicated_grouped_calibration" or minimum is None or events == 0 or events < minimum or not known.any():
        return artifact
    losses = []
    candidates = np.exp(np.linspace(np.log(0.1), np.log(10.0), 161))
    for temperature in candidates:
        p = np.clip(apply_hazard_calibrator(values, {"temperature": temperature}), 1e-9, 1 - 1e-9)
        loss = -(y * np.log(p) + (1-y) * np.log1p(-p))
        losses.append(np.mean([loss[(ids == uid)[:, None] & known].mean() for uid in groups
                               if known[ids == uid].any()]))
    artifact.update(status="fitted", temperature=float(candidates[int(np.argmin(losses))]))
    return artifact


def save_calibrator(artifact, path):
    Path(path).write_text(json.dumps(artifact, indent=2, allow_nan=False), encoding="utf-8")


def load_calibrator(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if artifact.get("version") != "red_entry_hazard_temperature_v1":
        raise ValueError("Unsupported calibration artifact")
    apply_hazard_calibrator(np.array([[0.5]]), artifact)
    return artifact


def event_corridor(cdf, horizons_s, issued_at_s, *, calibration=None, requirements=None):
    requirements, calibration = requirements or {}, calibration or {}
    coverage = requirements.get("nominal_interval_coverage")
    if coverage is None or not 0 < coverage < 1 or not calibration.get("interval_support_validated"):
        return {"status": "unavailable", "earliest_s": None, "latest_s": None,
                "calibration_status": calibration.get("status", "insufficient_event_evidence"),
                "reason": "interval_calibration_support_not_validated"}
    evidence = requirements.get("interval_coverage_evidence_requirement")
    if evidence is None or calibration.get("interval_independent_event_units", 0) < evidence:
        return {"status": "unavailable", "earliest_s": None, "latest_s": None,
                "reason": "insufficient_interval_event_evidence"}
    bounds = event_quantiles(np.asarray(cdf).reshape(1, -1), horizons_s,
                             ((1-coverage)/2, (1+coverage)/2))[0]
    return {"status": "available" if all(v is not None for v in bounds) else "unbounded",
            "earliest_s": None if bounds[0] is None else float(issued_at_s + bounds[0]),
            "latest_s": None if bounds[1] is None else float(issued_at_s + bounds[1]),
            "remaining_earliest_s": bounds[0], "remaining_latest_s": bounds[1],
            "nominal_coverage": coverage, "calibration_status": calibration.get("status")}
