"""Observed-only inputs for full-horizon experiments and replay.

Old project snapshots remain immutable. Synthetic snapshots expose separate
numeric split files; project snapshots use their existing physical split.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.io_util import read_json, sha256_file
from pdm.projects import project_store

HISTORY = 240
MIN_HISTORY = 60


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def source_for(project_id):
    store = project_store()
    project = store.get(project_id)
    if project["source_kind"] == "hse_filters":
        raise ValueError("Long forecast experiments are currently restricted to Bearings and synthetic vibration data")
    sid = project["active_snapshot_id"]
    if not sid:
        raise ValueError("Prepare a project snapshot first")
    mapped = (project.get("source_manifest") or {}).get("snapshots", {}).get(sid)
    directory = Path(mapped).resolve() if mapped else store.snapshot_path(project_id, sid)
    if not directory.resolve().is_relative_to(store.project_path(project_id).resolve()):
        raise ValueError("Snapshot escapes its project")
    path = directory / "snapshot.json"
    if path.exists():
        snapshot = read_json(path)
        unsigned = {k: v for k, v in snapshot.items() if k != "snapshot_hash"}
        if digest(unsigned) != snapshot.get("snapshot_hash") or snapshot["snapshot_id"] != sid:
            raise ValueError("Sensor snapshot integrity mismatch")
        cfg = snapshot["config"]
        from pdm.data.project_prepare import read_zone_limits

        limits = read_zone_limits(directory) or dict(yellow=cfg.get("yellow", .55), red=cfg.get("red", .8), direction=cfg.get("threshold_direction", "above"))
        return dict(project_id=project_id, snapshot_id=sid, directory=str(directory), kind="sensor",
                    target=cfg["target"], unit=cfg["unit"], label="Vibration RMS", cadence_s=cfg["cadence_s"],
                    yellow=limits["yellow"], red=limits["red"], direction=limits["direction"],
                    dataset_hash=snapshot["dataset_hash"], release_id=snapshot["release_id"], snapshot=snapshot)
    from pdm.data.project_prepare import load_snapshot, read_zone_limits
    data = load_snapshot(project_id, sid)
    schema = data["schema"]
    limits = read_zone_limits(directory) or schema.get("thresholds", {})
    train_features = data["features"].loc[data["features"].unit_id.astype(str).isin(map(str, data["split"]["train"]))]
    times = train_features.groupby("unit_id").timestamp_s.diff()
    cadence = float(times[times > 0].median())
    return dict(project_id=project_id, snapshot_id=sid, directory=str(directory), kind="project",
                target="signal", unit=schema.get("signal_unit", "g"), label=schema.get("signal_label", "Signal"),
                cadence_s=cadence, yellow=limits.get("yellow"), red=limits.get("red"), direction=limits.get("direction", "above"),
                dataset_hash=data["fingerprint"], data=data)


def read_part(source, part):
    if part not in {"train", "validation", "calibration", "test"}:
        raise ValueError("Unknown data role")
    if source["kind"] == "sensor":
        snapshot = source["snapshot"]
        record = snapshot["split_manifest"].get(part)
        if record is None:
            return pd.DataFrame(columns=["unit_id", "physical_unit_id", "timestamp_s", "signal"])
        directory = Path(source["directory"])
        path = directory / record["path"]
        if path.is_symlink() or not path.resolve().is_relative_to(directory) or sha256_file(path) != record["sha256"]:
            raise ValueError("Numeric split integrity mismatch")
        frame = pd.read_csv(path, dtype={"unit_id": str})
        if list(frame.columns) != snapshot["config"]["schema"]:
            raise ValueError("Observed numeric schema mismatch")
        frame = frame.rename(columns={source["target"]: "signal"})
        physical = snapshot["dataset_manifest"].get("physical_unit_map")
        frame["physical_unit_id"] = (frame.unit_id.map(physical) if physical else source["release_id"] + ":" + frame.unit_id)
    else:
        data = source["data"]
        frame = data["features"].loc[data["features"].unit_id.astype(str).isin(map(str, data["split"].get(part, [])))].copy()
        if "physical_unit_id" not in frame:
            frame["physical_unit_id"] = frame.unit_id.astype(str)
    if frame.empty:
        return frame
    if frame.physical_unit_id.isna().any() or frame.groupby("unit_id").physical_unit_id.nunique().gt(1).any():
        raise ValueError("Ambiguous physical identity")
    frame.attrs.update(split=part, dataset_hash=source["dataset_hash"])
    return frame


def segments(frame, cadence):
    for uid, group in frame.groupby("unit_id", sort=True):
        t = group.timestamp_s.to_numpy(float)
        y = group.signal.to_numpy(float)
        if (np.diff(t[np.isfinite(t)]) <= 0).any():
            raise ValueError("Duplicate or unordered observation times")
        valid = np.isfinite(t) & np.isfinite(y) & (y > 0)
        boundary = np.ones(len(group), bool)
        boundary[1:] = ~valid[:-1] | ~valid[1:] | ~np.isclose(np.diff(t), cadence, rtol=0, atol=1e-6)
        if "gap_before" in group:
            boundary |= group.gap_before.to_numpy(bool)
        if "component_cycle_id" in group:
            cycle = group.component_cycle_id.astype(str).to_numpy()
            boundary[1:] |= cycle[1:] != cycle[:-1]
        start = 0
        for end in [*np.flatnonzero(boundary)[1:], len(group)]:
            part = group.iloc[start:end]
            if valid[start:end].all() and len(part):
                yield str(uid), part
            start = end


def capacity(frame, cadence):
    n = max((len(s) for _, s in segments(frame, cadence)), default=0) - MIN_HISTORY
    if n < 1:
        raise ValueError("No continuous training targets after 60 measurements")
    return min(n, 4096)


def history_array(values):
    values = np.asarray(values, float)[-HISTORY:]
    if len(values) < MIN_HISTORY or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("60 continuous positive observations are required")
    result = np.full(HISTORY, np.median(values[-5:]), np.float32)
    result[-len(values):] = values
    return result, len(values)


def windows(frame, horizon, cadence, stride=30):
    xs, lengths, ys, masks, units, physical, origins = [], [], [], [], [], [], []
    for uid, segment in segments(frame, cadence):
        values = segment.signal.to_numpy(float)
        times = segment.timestamp_s.to_numpy(float)
        for i in range(MIN_HISTORY - 1, len(values) - 1, stride):
            x, n = history_array(values[max(0, i - HISTORY + 1):i + 1])
            future = values[i + 1:i + 1 + horizon]
            y = np.zeros(horizon, np.float32)
            mask = np.zeros(horizon, bool)
            y[:len(future)], mask[:len(future)] = future, True
            xs.append(x)
            lengths.append(n)
            ys.append(y)
            masks.append(mask)
            units.append(uid)
            physical.append(str(segment.physical_unit_id.iloc[0]))
            origins.append(float(times[i]))
    if not xs:
        raise ValueError("No admitted causal windows")
    return dict(x=np.asarray(xs), lengths=np.asarray(lengths), y=np.asarray(ys), mask=np.asarray(masks),
                units=np.asarray(units), physical=np.asarray(physical), origins=np.asarray(origins),
                split=frame.attrs.get("split"), horizon=horizon, cadence_s=cadence)


def at_origin(frame, uid, origin, cadence):
    # Restrict the stream before segmentation: future gaps/events never affect history.
    prefix = frame.loc[frame.unit_id.eq(uid) & frame.timestamp_s.le(origin)]
    candidates = list(segments(prefix, cadence))
    if not candidates or candidates[-1][1].timestamp_s.iloc[-1] != origin:
        raise ValueError("Now must be an admitted measurement")
    return history_array(candidates[-1][1].signal.to_numpy())
