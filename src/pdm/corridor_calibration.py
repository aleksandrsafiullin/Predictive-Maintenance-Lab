"""Bounded empirical widths fitted on independent Calibration observations only.

The center and model remain frozen. One width is used for the full trajectory;
limits are expressed as full width relative to its positive forecast center.
"""

from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone

import numpy as np

from pdm import stable_forecast as stable
from pdm.io_util import atomic_write_json, read_json
from pdm.long_forecast_data import digest, read_part, source_for
from pdm.projects import _safe_id, project_store

PROTOCOL = "bounded_empirical_calibration_v1"
DEFAULTS = dict(min_width=0.20, max_width=0.45, target_coverage=0.90,
                scope="points", origin_stride=60)


def settings(values=None):
    supplied = values or {}
    if supplied.keys() - DEFAULTS.keys():
        raise ValueError("Unknown corridor calibration setting")
    result = {**DEFAULTS, **supplied}
    for field in ("min_width", "max_width", "target_coverage"):
        value = result[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{field} must be a finite number")
    if not 0 < result["min_width"] <= result["max_width"] < 2:
        raise ValueError("Minimum width must be positive and no greater than maximum; maximum must be below 200%.")
    if not 0 < result["target_coverage"] <= 1:
        raise ValueError("Target coverage must be between 0 and 100%.")
    if result["scope"] not in {"points", "whole_paths"}:
        raise ValueError("Select point coverage or complete trajectory coverage")
    stride = result["origin_stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError("Calibration origin step must be a positive number of observations")
    return result


def _quantile(values, weights, probability):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(len(order) - 1, np.searchsorted(cumulative, probability * cumulative[-1], side="left"))
    return float(values[order[index]])


def _point_weights(mask, physical):
    """Each unit has equal total weight, each origin equal weight within its unit."""
    weights = np.zeros(mask.shape, float)
    for uid in sorted(set(physical)):
        rows = np.flatnonzero((physical == uid) & mask.any(axis=1))
        for row in rows:
            weights[row, mask[row]] = 1.0 / (len(rows) * mask[row].sum())
    return weights / weights.sum()


def fit_width(outputs, windows, policy=None):
    """Fit a single bounded width; correlated rolling origins are empirical evidence."""
    policy = settings(policy)
    if windows.get("split") != "calibration":
        raise ValueError("Only Calibration data may determine corridor width")
    raw = np.asarray(outputs, float)
    y, mask = np.asarray(windows["y"], float), np.asarray(windows["mask"], bool)
    physical = np.asarray(windows["physical"]).astype(str)
    if (raw.shape != (*y.shape, 3) or mask.shape != y.shape or len(physical) != len(y)
            or not np.isfinite(raw).all() or (raw <= 0).any()
            or (np.diff(raw, axis=-1) < 0).any()
            or not mask.any() or not np.isfinite(y[mask]).all() or (y[mask] <= 0).any()):
        raise ValueError("Calibration requires ordered positive forecasts and supported positive observations")
    if (np.diff(mask.astype(int), axis=1) > 0).any():
        raise ValueError("Supported calibration targets must form continuous prefixes")
    center = raw[..., 1]
    required = 2 * np.abs(np.where(mask, y, center) - center) / center
    weights = _point_weights(mask, physical)
    complete = mask.all(axis=1)
    if policy["scope"] == "whole_paths":
        if not complete.any():
            raise ValueError("No complete Calibration trajectories support this horizon; whole-path coverage is unknown.")
        scores = required[complete].max(axis=1)
        complete_ids = physical[complete]
        population_weights = np.array([1. / np.sum(complete_ids == uid) for uid in complete_ids])
    else:
        scores, population_weights = required[mask], weights[mask]
    required_width = _quantile(scores, population_weights, policy["target_coverage"])
    width = float(np.clip(required_width, policy["min_width"], policy["max_width"]))
    inside = mask & (required <= width + 1e-12)
    complete_coverage = None
    if complete.any():
        complete_ids = physical[complete]
        complete_coverage = float(np.mean([
            inside[complete][complete_ids == uid].all(axis=1).mean()
            for uid in sorted(set(complete_ids))]))
    point_coverage = float(weights[inside].sum())
    measured = point_coverage if policy["scope"] == "points" else complete_coverage
    return dict(settings=policy, width=width, required_width=required_width,
                upper_limit_reached=required_width > policy["max_width"] + 1e-12,
                target_met=measured >= policy["target_coverage"] - 1e-12,
                point_coverage=point_coverage, whole_path_coverage=complete_coverage,
                physical_units=len(set(physical[mask.any(axis=1)])), origins=int(mask.any(axis=1).sum()),
                complete_origins=int(complete.sum()), supported_points=int(mask.sum()),
                support_by_lead=mask.sum(axis=0).tolist(),
                weighting="equal_physical_unit_then_equal_origin_then_equal_supported_point",
                coverage_guarantee=False)


def apply_width(outputs, calibration):
    policy = settings(calibration["settings"])
    width = calibration["width"]
    if not math.isfinite(width) or not policy["min_width"] <= width <= policy["max_width"]:
        raise ValueError("Saved calibration width violates its limits")
    raw = np.asarray(outputs, float)
    if raw.shape[-1] != 3 or not np.isfinite(raw).all() or (raw[..., 1] <= 0).any():
        raise ValueError("Calibration requires a finite positive forecast center")
    center = raw[..., 1]
    return np.stack((center * (1 - width / 2), center, center * (1 + width / 2)), axis=-1)


def _draft_path(project_id, snapshot_id):
    return project_store().project_path(project_id) / "calibration" / _safe_id(snapshot_id, "snapshot ID") / "settings.json"


def load_settings(project_id, snapshot_id):
    path = _draft_path(project_id, snapshot_id)
    if not path.exists():
        return settings()
    row = read_json(path)
    if (row.get("project_id"), row.get("snapshot_id")) != (project_id, snapshot_id):
        raise ValueError("Calibration settings belong to another snapshot")
    return settings(row["settings"])


def save_settings(project_id, snapshot_id, policy):
    value = settings(policy)
    store = project_store()
    path = _draft_path(project_id, snapshot_id)
    with store.launch_lock():
        if store._entry(store._load(), project_id)["active_snapshot_id"] != snapshot_id:
            raise ValueError("Data changed. Reload Calibration Data before saving settings.")
        atomic_write_json(path, dict(
            project_id=project_id, snapshot_id=snapshot_id, settings=value))


def _directory(project_id, run_id):
    directory = project_store().run_path(project_id, run_id) / "corridor_calibrations"
    if directory.is_symlink():
        raise ValueError("Calibration storage is unsafe")
    return directory


def load_active(project_id, run_id, manifest):
    directory = _directory(project_id, run_id)
    path = directory / "active.json"
    if not path.exists():
        return None
    pointer = read_json(path)
    identifier = _safe_id(pointer["calibration_id"], "calibration ID")
    payload_path = directory / (identifier + ".json")
    if payload_path.is_symlink():
        raise ValueError("Calibration payload is unsafe")
    payload = read_json(payload_path)
    unsigned = {key: value for key, value in payload.items() if key != "calibration_hash"}
    if payload.get("calibration_hash") != digest(unsigned) or pointer.get("calibration_hash") != payload["calibration_hash"]:
        raise ValueError("Saved calibration integrity mismatch")
    if payload.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported corridor calibration protocol")
    for key in ("project_id", "run_id", "snapshot_id", "model_hash", "dataset_hash", "config"):
        if payload.get(key) != manifest.get(key):
            raise ValueError(f"Calibration/model binding mismatch: {key}")
    settings(payload["settings"])
    if not payload["settings"]["min_width"] <= payload["width"] <= payload["settings"]["max_width"]:
        raise ValueError("Calibration width violates saved limits")
    return payload


def calibrate(project_id, run_id, policy=None, *, progress=None, publish=True):
    from pdm.long_forecast_run import alive, load_run, read_status
    from pdm.worker import heavy_job_active

    store = project_store()
    policy = settings(policy)
    if heavy_job_active() or alive(read_status(project_id)):
        raise RuntimeError("Wait for the active job to finish before calibrating")
    source = source_for(project_id)
    manifest, model = load_run(project_id, run_id)
    if manifest.get("protocol") not in stable.PROTOCOLS:
        raise ValueError("Bounded width calibration currently requires a saved GRU, LSTM or MaleCNS observed-trend model")
    if manifest["snapshot_id"] != source["snapshot_id"] or manifest["dataset_hash"] != source["dataset_hash"]:
        raise ValueError("Model and Calibration data belong to different snapshots")
    fitted = read_json(store.run_path(project_id, run_id) / "origins.json")
    forbidden = set(fitted["train"]["physical"]) | set(fitted["validation"]["physical"])
    frame = read_part(source, "calibration")
    if frame.empty:
        raise ValueError("This snapshot has no Calibration measurements")
    if forbidden & set(frame.physical_unit_id.astype(str)):
        raise ValueError("Calibration physical units overlap Train/Validation")
    horizon = model["config"]["horizon"]
    windows = stable.full_windows(frame, horizon, source["cadence_s"], policy["origin_stride"])
    chunks = []
    for start in range(0, len(windows["x"]), 64):
        stop = min(start + 64, len(windows["x"]))
        chunks.append(stable.predict(model, windows["x"][start:stop], windows["lengths"][start:stop]))
        if progress:
            progress(stop / len(windows["x"]))
    result = fit_width(np.concatenate(chunks), windows, policy)
    identifier = "cal-" + uuid.uuid4().hex
    identity = {key: manifest[key] for key in ("project_id", "run_id", "snapshot_id", "model_hash", "dataset_hash", "config")}
    result.update(identity, protocol=PROTOCOL, calibration_id=identifier,
                  created_at=datetime.now(timezone.utc).isoformat(),
                  calibration_units=sorted(set(windows["physical"].astype(str))),
                  origins_hash=digest(dict(units=windows["physical"].tolist(), origins=windows["origins"].tolist())),
                  source_split="calibration", horizon=horizon)
    result["calibration_hash"] = digest(result)
    if publish:
        directory = _directory(project_id, run_id)
        draft = _draft_path(project_id, source["snapshot_id"])
        with store.launch_lock():
            if store._entry(store._load(), project_id)["active_snapshot_id"] != source["snapshot_id"]:
                raise ValueError("Data changed during Calibration; result was not activated")
            if heavy_job_active():
                raise RuntimeError("A job started during Calibration; result was not activated")
            atomic_write_json(directory / (identifier + ".json"), result)
            atomic_write_json(directory / "active.json", dict(
                calibration_id=identifier, calibration_hash=result["calibration_hash"]))
            atomic_write_json(draft, dict(
                project_id=project_id, snapshot_id=source["snapshot_id"], settings=policy))
    return result
