"""Train-only direct horizon capacity and observed target support."""

from __future__ import annotations

import numpy as np

from .contract import default_config
from .windows import _segments


def train_horizon_profile(frame, cfg: dict) -> dict:
    """Capacity uses admitted continuous Train samples, never another split."""
    cfg = default_config(**cfg)
    if frame.attrs.get("split") != "train" or frame.attrs.get("external"):
        raise ValueError("Horizon capacity requires an explicit internal Train split")
    if not set(cfg["schema"]).issubset(frame.columns):
        raise ValueError("Missing observed signal schema for Train horizon capacity")
    source_cfg = frame.attrs.get("config")
    if source_cfg and any(source_cfg[k] != cfg[k] for k in (
        "schema", "target", "unit", "cadence_s", "positive_domain", "clock_tolerance_s"
    )):
        raise ValueError("Train horizon source physical contract mismatch")
    lengths = [len(segment) for _, group in frame.groupby("unit_id", sort=True)
               for segment in _segments(group, cfg)]
    maximum = max(lengths, default=0)
    capacity = maximum - cfg["common_history_length"]
    if capacity <= 0:
        raise ValueError("Train has no target after a continuous common-history anchor")
    horizon = min(4096, capacity)
    return {"horizon": horizon, "max_horizon": horizon, "max_supported_leads": capacity,
            "span_s": horizon * cfg["cadence_s"], "cadence_s": cfg["cadence_s"],
            "common_history_length": cfg["common_history_length"], "maximum_segment_length": maximum,
            "capped": capacity > 4096, "continuous_segments": len(lengths),
            "source_binding": {key: frame.attrs[key] for key in (
                "split", "snapshot_id", "dataset_hash", "release_id", "suite", "profile"
            ) if key in frame.attrs},
            "policy": "dense-v2:train-only:max-continuous-segment-minus-common-history:cap-4096"}


def target_support(mask, physical_units) -> dict:
    mask = np.asarray(mask, dtype=bool)
    units = np.asarray(physical_units, dtype=str)
    if mask.ndim != 2 or units.shape != (len(mask),):
        raise ValueError("Expected a target mask and one physical identity per origin")
    return {"target_count": mask.sum(axis=0).astype(int).tolist(),
            "physical_unit_count": [int(len(np.unique(units[mask[:, lead]])))
                                    for lead in range(mask.shape[1])]}
