"""Runtime quality is independent of health and never fills a required channel."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pdm.history import contiguous_history
from pdm.monitoring.contracts import observed_prefix
from pdm.windows import raw_numeric_columns


def quality_policy(profile):
    return {"version": "runtime_quality_v2", "stale_after": 3 * profile["nominal_interval"],
            "minimum_history": 20, "sensor_ranges": {"differential_pressure": [0., 2500.]} if profile["dataset_id"] == "filters" else {}, "stuck_count": 5,
            "stuck_action": "diagnostic_only", "time_basis": profile["time_basis"]}


def assess_quality(frame, profile, policy, as_of):
    prefix = observed_prefix(frame, profile["dataset_id"], as_of)
    result = {"data_quality_status": "invalid", "reason_codes": [], "channel_valid": {},
              "last_measurement_at": None, "available_measurements": 0, "duration": None,
              "measurement_usable": False, "nonfinite_fraction": None}
    if prefix.empty:
        result["reason_codes"] = ["no_observations"]
        return result
    ts = pd.to_numeric(prefix.timestamp_s, errors="coerce").to_numpy(float)
    if not np.isfinite(ts).all() or np.any(np.diff(ts) <= 0):
        result["reason_codes"] = ["invalid_source_clock_or_duplicate"]
        return result
    result["last_measurement_at"] = float(ts[-1])
    last = prefix.iloc[-1]
    required = raw_numeric_columns(profile["dataset_id"])
    for name in required:
        value = pd.to_numeric(pd.Series([last.get(name)]), errors="coerce").iloc[0]
        ok = bool(np.isfinite(value))
        bounds = policy.get("sensor_ranges", {}).get(name)
        if bounds and ok and not bounds[0] <= value <= bounds[1]:
            ok = False
        result["channel_valid"][name] = ok
    values = prefix.reindex(columns=required).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    result["nonfinite_fraction"] = float((~np.isfinite(values)).mean())
    row_valid = np.isfinite(values).all(axis=1)
    for name, bounds in policy.get("sensor_ranges", {}).items():
        if name in prefix:
            row_valid &= pd.to_numeric(prefix[name], errors="coerce").between(*bounds).to_numpy()
    if "n_nan" in prefix:
        row_valid &= pd.to_numeric(prefix.n_nan, errors="coerce").eq(0).to_numpy()
    if "sample_count_ok" in prefix:
        row_valid &= prefix.sample_count_ok.fillna(False).to_numpy(bool)
    bad_fragment = ("n_nan" in last and last.n_nan > 0) or ("sample_count_ok" in last and not last.sample_count_ok)
    if bad_fragment:
        result["channel_valid"] = {k: False for k in result["channel_valid"]}
        result["reason_codes"].append("invalid_raw_fragment")
    stale = as_of - ts[-1] > policy["stale_after"]
    if stale:
        result["data_quality_status"] = "stale"
        result["reason_codes"].append("stale_observation")
    elif bad_fragment or not all(result["channel_valid"].values()):
        result["reason_codes"].append("invalid_required_channel")
    else:
        result["measurement_usable"] = True
        # Invalid earlier measurements are boundaries, not imputed history.
        clean = prefix.copy()
        invalid = np.flatnonzero(~row_valid)
        if len(invalid):
            clean = clean.iloc[int(invalid[-1]) + 1:]
        segment = contiguous_history(clean, profile["dataset_id"])
        n = len(segment)
        result.update(available_measurements=n,
                      duration=float(segment.timestamp_s.iloc[-1] - segment.timestamp_s.iloc[0]) if n else None)
        result["data_quality_status"] = "valid" if n >= policy["minimum_history"] else "insufficient_history"
        if n < policy["minimum_history"]:
            result["reason_codes"].append("collecting_contiguous_history")
        signal = profile["signal_name"]
        count = policy["stuck_count"]
        if n >= count and segment[signal].iloc[-count:].nunique() == 1:
            result["reason_codes"].append("constant_signal_check_sensor")
            if result["data_quality_status"] == "valid":
                result["data_quality_status"] = "degraded"
    # Separate critical-channel validity: another covariate cannot suppress a
    # usable hard limit, but stale/invalid raw fragment must suppress it.
    result["critical_channel_usable"] = bool(result["channel_valid"].get(profile["signal_name"], False) and not stale and not bad_fragment)
    return result
