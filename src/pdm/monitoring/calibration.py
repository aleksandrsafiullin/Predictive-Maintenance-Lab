"""Research probability outputs and strict saved-identity calibration gates."""
from __future__ import annotations

import numpy as np

from pdm.training_protocol import fingerprint


def weibull_probabilities(scale, shape, horizons):
    h = np.asarray(horizons, float)
    if scale <= 0 or shape <= 0 or not np.isfinite([scale, shape]).all() or np.any(np.diff(h) <= 0) or np.any(h <= 0):
        raise ValueError("Invalid remaining-duration Weibull or horizons")
    cdf = -np.expm1(-np.power(h / scale, shape))
    return [{"horizon": float(t), "cumulative": float(p), "bin_probability": float(d)}
            for t, p, d in zip(h, cdf, np.diff(np.r_[0., cdf]), strict=True)]


def calibration_report(identity, calibration_units, events, *, role="validation_diagnostics"):
    return {"version": "calibration_v1", "identity": identity, "identity_hash": fingerprint(identity),
            "role": role, "independent_units": len(set(calibration_units)), "independent_events": int(events),
            "status": "insufficient_independent_calibration_data", "operator_probabilities_allowed": False,
            "coverage_guarantee": False, "timing_validated": False,
            "reason": "No independently validated reliability policy for this small laboratory cohort"}


def verify_calibration(report, identity):
    if report["identity_hash"] != fingerprint(identity) or report["identity"] != identity:
        raise ValueError("Calibration checkpoint/recipe/history/data identity mismatch")
    return report
