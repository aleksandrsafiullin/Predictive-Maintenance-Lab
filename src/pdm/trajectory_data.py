"""Causal dense trajectory training data; identifiers never enter model inputs."""

from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "log_signal",
    "log_mean_8",
    "log_mean_32",
    "log_mean_128",
    "log_std_32",
    "log_change_8",
    "log_change_32",
    "log_known_age",
    "age_known",
    "rpm_scaled",
    "rpm_known",
    "load_scaled",
    "load_known",
]

# Acquisition-local features only. Archive identifiers and target metadata must
# never be selected by a numeric-column heuristic.
SENSOR_FEATURE_NAMES = tuple(
    f"{axis}_{name}"
    for axis in ("horizontal", "vertical")
    for name in ("rms", "std", "abs_peak", "peak_to_peak", "crest_factor", "kurtosis",
                 "band_0", "band_1", "band_2", "band_3")
)
SENSOR_AVAILABILITY = "timestamp_s_is_completed_acquisition_issue_time"
SENSOR_FEATURE_MODES = ("absolute", "baseline_relative", "combined")


def _sensor_feature_names(sensor_features, mode):
    if mode not in SENSOR_FEATURE_MODES:
        raise ValueError(f"Invalid sensor_feature_mode: {mode!r}")
    if not sensor_features and mode != "absolute":
        raise ValueError("Nonabsolute sensor_feature_mode requires declared trajectory sensor features")
    absolute = [f"log1p_{name}" for name in sensor_features]
    relative = [f"log1p_relative_initial8_{name}" for name in sensor_features]
    return absolute if mode == "absolute" else relative if mode == "baseline_relative" else absolute + relative


def trajectory_sensor_features(schema):
    declaration = schema.get("trajectory_sensor_features")
    if declaration is None:
        return ()
    if (not isinstance(declaration, dict)
            or declaration.get("columns") != list(SENSOR_FEATURE_NAMES)
            or declaration.get("transform") != "log1p"
            or declaration.get("availability") != SENSOR_AVAILABILITY
            or declaration.get("schema_version") != 1):
        raise ValueError("Invalid trajectory sensor feature declaration")
    return SENSOR_FEATURE_NAMES


def causal_features(segment: pd.DataFrame, sensor_features=(), sensor_feature_mode="absolute") -> np.ndarray:
    """Every row uses only measurements and admitted context available by that row."""
    _sensor_feature_names(sensor_features, sensor_feature_mode)
    values = segment.signal.astype(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Learned RMS trajectories require finite nonnegative measurements")
    log = np.log1p(values)
    columns = [log]
    for size in (8, 32, 128):
        columns.append(np.log1p(values.rolling(size, min_periods=1).mean()))
    columns.append(np.log1p(values.rolling(32, min_periods=1).std(ddof=0)))
    for size in (8, 32):
        columns.append(log - log.shift(size - 1).fillna(float(log.iloc[0])))
    age = pd.to_numeric(
        segment.get("operating_age_s", pd.Series(np.nan, index=segment.index)), errors="coerce"
    )
    known = age.notna() & np.isfinite(age) & (age >= 0)
    if "operating_age_known" in segment:
        known &= segment.operating_age_known.fillna(False).astype(bool)
    columns.extend([np.log1p(age.where(known, 0)) / 10, known.astype(float)])
    for name, denominator in (("rpm", 1000.0), ("load_kn", 10.0)):
        value = pd.to_numeric(
            segment.get(name, pd.Series(np.nan, index=segment.index)), errors="coerce"
        )
        valid = value.notna() & np.isfinite(value)
        if name + "_known" in segment:
            valid &= segment[name + "_known"].fillna(False).astype(bool)
        columns.extend([value.where(valid, 0) / denominator, valid.astype(float)])
    if sensor_features:
        if tuple(sensor_features) != SENSOR_FEATURE_NAMES:
            raise ValueError("Invalid trajectory sensor allowlist")
        missing = set(sensor_features) - set(segment.columns)
        if missing:
            raise ValueError(f"Missing declared trajectory sensor features: {sorted(missing)}")
        sensors = segment.loc[:, list(sensor_features)].to_numpy(dtype=float)
        if not np.isfinite(sensors).all() or (sensors < 0).any():
            raise ValueError("Trajectory sensor features must be finite and nonnegative")
        canonical = np.maximum(segment.horizontal_rms, segment.vertical_rms).to_numpy(float)
        if not np.array_equal(canonical, values.to_numpy(float)):
            raise ValueError("Trajectory sensors do not match canonical max-axis RMS")
        sensor_log = np.log1p(sensors)
        if sensor_feature_mode in ("absolute", "combined"):
            columns.extend(sensor_log.T)
        if sensor_feature_mode in ("baseline_relative", "combined"):
            # The caller supplies one continuous segment. Each early row uses
            # its expanding initial mean; after eight completed acquisitions
            # the baseline stays fixed. Never use a whole-segment mean.
            initial = sensor_log[:8]
            baseline = np.cumsum(initial, axis=0) / np.arange(1, len(initial) + 1)[:, None]
            baseline = baseline[np.minimum(np.arange(len(sensor_log)), len(initial) - 1)]
            columns.extend((sensor_log - baseline).T)
    return np.column_stack(columns).astype(np.float32)


def _threshold(data):
    rule = data["schema"].get("thresholds", {})
    if rule.get("mode") != "absolute" or rule.get("direction") != "above":
        raise ValueError("Learned trajectory v2 currently requires an absolute upper RED rule")
    red = float(rule["red"])
    if not np.isfinite(red) or red <= 0:
        raise ValueError("RED must be finite and positive")
    return red


def build_trajectory_frame(data, ids, config, cap=None):
    from pdm.signal_training import _segments

    history = int(config["history_length"])
    context_mode = config.get("recurrent_context_mode", "fixed")
    if not isinstance(context_mode, str) or context_mode not in {"fixed", "variable_causal"}:
        raise ValueError("Unsupported recurrent context mode")
    variable = context_mode == "variable_causal"
    max_history = history
    if variable:
        max_history = config.get("max_history_length")
        if (isinstance(max_history, bool) or not isinstance(max_history, Integral)
                or max_history < history or history < 1
                or config.get("min_history_length", history) != history):
            raise ValueError("Variable causal context requires declared max >= minimum history_length")
    horizons = np.asarray(config["horizons_s"], float)
    red = _threshold(data)
    sensors = trajectory_sensor_features(data["schema"])
    sensor_feature_mode = config.get("sensor_feature_mode", "absolute")
    feature_names = FEATURE_NAMES + _sensor_feature_names(sensors, sensor_feature_mode)
    missing_sensors = set(sensors) - set(data["features"].columns)
    if missing_sensors:
        raise ValueError(f"Missing declared trajectory sensor features: {sorted(missing_sensors)}")
    rows = {
        key: []
        for key in (
            "x",
            "raw_x",
            "current",
            "y",
            "mask",
            "unit_id",
            "physical_unit_id",
            "as_of_s",
            "event_allowed",
            "event_observed",
            "no_entry_prefix",
            "warning_eligible",
            "red_threshold",
        )
    }
    if variable:
        rows.update({"history_lengths": [], "history_mask": [], "scaler_x": []})
    for uid in map(str, ids):
        for segment_number, segment in enumerate(_segments(data["features"], uid)):
            if len(segment) < history:
                continue
            signal = segment.signal.to_numpy(float)
            times = segment.timestamp_s.to_numpy(float)
            features = causal_features(segment, sensors, sensor_feature_mode)
            physical = (
                str(segment.physical_unit_id.iloc[0]) if "physical_unit_id" in segment else uid
            )
            ends = np.arange(history - 1, len(signal))
            for end in ends:
                desired = times[end] + horizons
                indexes = np.searchsorted(times, desired)
                admitted = indexes < len(times)
                admitted[admitted] &= np.isclose(
                    times[indexes[admitted]],
                    desired[admitted],
                    atol=config.get("target_tolerance_s", 0.01),
                    rtol=0,
                )
                target = np.zeros(len(horizons), np.float32)
                target[admitted] = signal[indexes[admitted]]
                # Only the uninterrupted observed grid prefix admits entry/survival evidence.
                missing = np.flatnonzero(~admitted)
                count = int(missing[0]) if len(missing) else len(horizons)
                entries = np.flatnonzero(target[:count] >= red)
                allowed = np.zeros(len(horizons) + 1, bool)
                at_risk = signal[end] < red
                event_observed = bool(at_risk and len(entries))
                if event_observed:
                    allowed[int(entries[0])] = True
                elif at_risk and count:
                    allowed[count:] = True
                if variable:
                    length = min(max_history, end + 1)
                    chronological = features[end - length + 1:end + 1]
                    raw_chronological = signal[end - length + 1:end + 1, None]
                    rows["x"].append(np.pad(chronological, ((0, max_history - length), (0, 0))))
                    rows["raw_x"].append(np.pad(raw_chronological, ((0, max_history - length), (0, 0))))
                    rows["history_lengths"].append(length)
                    rows["history_mask"].append(np.arange(max_history) < length)
                    rows["scaler_x"].append(features[end - history + 1:end + 1])
                else:
                    rows["x"].append(features[end - history + 1 : end + 1])
                    rows["raw_x"].append(signal[end - history + 1 : end + 1, None])
                rows["current"].append(signal[end])
                rows["y"].append(target)
                rows["mask"].append(admitted)
                rows["unit_id"].append(uid)
                rows["physical_unit_id"].append(physical)
                rows["as_of_s"].append(times[end])
                rows["event_allowed"].append(allowed)
                rows["event_observed"].append(event_observed)
                rows["no_entry_prefix"].append(count if at_risk and not len(entries) else 0)
                rows["warning_eligible"].append(
                    segment_number == 0 and not bool(np.any(signal[: end + 1] >= red))
                )
                rows["red_threshold"].append(red)
    result = {}
    list_keys = {"unit_id", "physical_unit_id", "as_of_s"}
    bool_keys = {"mask", "event_allowed", "event_observed", "warning_eligible"}
    if variable:
        bool_keys.add("history_mask")
    for key, values in rows.items():
        result[key] = (
            values
            if key in list_keys
            else np.asarray(
                values,
                dtype=bool
                if key in bool_keys
                else np.int64
                if key in {"no_entry_prefix", "history_lengths"}
                else np.float32,
            )
        )
    if not len(rows["x"]):
        for key, shape in {
            "x": (0, max_history, len(feature_names)),
            "raw_x": (0, max_history, 1),
            "y": (0, len(horizons)),
            "mask": (0, len(horizons)),
            "event_allowed": (0, len(horizons) + 1),
        }.items():
            result[key] = np.empty(shape, bool if key in bool_keys else np.float32)
        if variable:
            result["history_mask"] = np.empty((0, max_history), dtype=bool)
            result["scaler_x"] = np.empty((0, history, len(feature_names)), dtype=np.float32)
    result["feature_names"] = feature_names
    if cap and len(result["x"]):
        selected = []
        physical = np.asarray(result["physical_unit_id"])
        for group in np.unique(physical):
            candidates = np.flatnonzero(physical == group)
            if len(candidates) > cap:
                candidates = candidates[
                    np.unique(np.rint(np.linspace(0, len(candidates) - 1, cap)).astype(int))
                ]
            selected.extend(candidates.tolist())
        result = slice_frame(result, np.asarray(sorted(selected)))
    return result


def build_trajectory_prefix(data, prefix, config):
    from pdm.signal_training import _segments

    if prefix.empty:
        raise ValueError("Prefix is empty")
    uid = str(prefix.unit_id.iloc[-1])
    last_segment = _segments(prefix, uid)[-1]
    frame = build_trajectory_frame({**data, "features": last_segment}, [uid], config)
    if not len(frame["x"]):
        raise ValueError("Insufficient observations since the last gap")
    return slice_frame(frame, np.asarray([len(frame["x"]) - 1]))


def slice_frame(frame, indexes):
    indexes = np.asarray(indexes, int)
    n = len(frame["x"])
    return {
        key: value[indexes]
        if isinstance(value, np.ndarray) and len(value) == n
        else [value[int(i)] for i in indexes]
        if key in {"unit_id", "physical_unit_id", "as_of_s"}
        else value
        for key, value in frame.items()
    }
