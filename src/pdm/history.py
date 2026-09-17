"""Saved model memory policy; legacy checkpoints retain their exact fixed window."""
from __future__ import annotations

import numpy as np

from pdm.windows import _unit_gap_flags, build_windows

HISTORY_MODES = ("fixed_20", "fixed_40", "fixed_60", "variable_20_60")


def history_policy(mode):
    if mode not in HISTORY_MODES:
        raise ValueError(f"Unknown history mode: {mode}")
    variable = mode == "variable_20_60"
    size = 60 if variable else int(mode.split("_")[1])
    return {"version": "history_v1", "mode": mode, "min": 20 if variable else size,
            "max": size, "memory_mode": "window_reset",
            "training_lengths": "deterministic_endpoint_cycle_20_60" if variable else "fixed"}


def contiguous_history(frame, dataset_id):
    if frame.empty:
        return frame.copy()
    ordered, gaps = _unit_gap_flags(frame, dataset_id)
    gaps = gaps.copy()
    # Explicit maintenance/reset markers are independent of acquisition gaps.
    for name in ("maintenance_reset", "segment_reset"):
        if name in ordered:
            gaps |= ordered[name].fillna(False).to_numpy(bool)
    if "segment_id" in ordered:
        gaps[1:] |= ordered.segment_id.to_numpy()[1:] != ordered.segment_id.to_numpy()[:-1]
    starts = np.flatnonzero(gaps)
    return ordered.iloc[int(starts[-1]) if len(starts) else 0:].copy()


def resolved_length(frame, prep, legacy_length):
    policy = getattr(prep, "history_policy", None)
    if not policy:
        return int(legacy_length)
    if policy["mode"] != "variable_20_60":
        return int(policy["max"])
    n = len(contiguous_history(frame, prep.dataset_id))
    return max(int(policy["min"]), min(n, int(policy["max"])))


def history_windows(features, units, dataset_id, policy, *, training=False, **gap_kw):
    windows = build_windows(features, units, policy["min"], dataset_id, **gap_kw)
    if windows.empty:
        return windows
    starts = {}
    for uid, group in features.groupby("unit_id", sort=False):
        ordered, gaps = _unit_gap_flags(group, dataset_id, **gap_kw)
        gaps = gaps.copy()
        for name in ("maintenance_reset", "segment_reset"):
            if name in ordered:
                gaps |= ordered[name].fillna(False).to_numpy(bool)
        if "segment_id" in ordered:
            gaps[1:] |= ordered.segment_id.to_numpy()[1:] != ordered.segment_id.to_numpy()[:-1]
        starts[str(uid)] = np.maximum.accumulate(np.where(gaps, np.arange(len(gaps)), 0))
    keep = []
    for i, row in windows.iterrows():
        end = int(row.end_index)
        segment_start = int(starts[str(row.unit_id)][end])
        available = end - segment_start + 1
        if available < policy["min"]:
            continue
        length = min(available, policy["max"])
        if training and policy["mode"] == "variable_20_60":
            length = min(length, policy["min"] + end % (policy["max"] - policy["min"] + 1))
        windows.at[i, "start_index"] = end - length + 1
        keep.append(i)
    return windows.loc[keep].reset_index(drop=True)
