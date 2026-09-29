"""Admit long-table sensor CSVs without inferring failure labels."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = frozenset({"unit_id", "timestamp_s"})


def read_generic_csv(
    grouped_files: dict[str, list[dict[str, Any]]], signal_column: str
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return finite canonical rows, physical units, and quality observations.

    Invalid signal rows are removed only after marking the next accepted row as
    a segment boundary. A training/replay window must never cross gap_before.
    """
    if not signal_column or signal_column in {"unit_id", "timestamp_s"}:
        raise ValueError("Choose a numeric signal column distinct from unit_id and timestamp_s")
    raw_frames: list[pd.DataFrame] = []
    file_counts: dict[str, int] = {}
    for group, files in grouped_files.items():
        file_counts[group] = len(files)
        for rec in files:
            path = Path(rec["path"])
            try:
                frame = pd.read_csv(path)
            except (OSError, pd.errors.ParserError, UnicodeError) as exc:
                raise ValueError(f"Cannot read CSV {rec['relative_path']}: {exc}") from exc
            missing = (REQUIRED_COLUMNS | {signal_column}) - set(frame.columns)
            if missing:
                raise ValueError(f"CSV {rec['relative_path']} is missing columns: {sorted(missing)}")
            if frame.empty:
                raise ValueError(f"CSV {rec['relative_path']} is empty")
            frame = frame[["unit_id", "timestamp_s", signal_column]].copy()
            frame["_group"] = group
            frame["_relative_path"] = rec["relative_path"]
            raw_frames.append(frame)
    if not raw_frames:
        raise ValueError("No CSV files were supplied")
    raw = pd.concat(raw_frames, ignore_index=True)
    raw["unit_id"] = raw["unit_id"].astype("string").str.strip()
    if raw["unit_id"].isna().any() or (raw["unit_id"] == "").any():
        raise ValueError("Every generic CSV row needs a nonempty unit_id")
    raw["timestamp_s"] = pd.to_numeric(raw["timestamp_s"], errors="coerce")
    if not np.isfinite(raw["timestamp_s"].to_numpy(dtype=float)).all():
        raise ValueError("Every generic CSV row needs a finite timestamp_s")
    raw["_signal"] = pd.to_numeric(raw[signal_column], errors="coerce")
    raw["_valid_signal"] = np.isfinite(raw["_signal"].to_numpy(dtype=float))
    raw["unit_id"] = raw["unit_id"].astype(str)

    owner_count = raw.groupby("unit_id")["_group"].nunique()
    duplicate_ids = owner_count[owner_count > 1].index.tolist()
    if duplicate_ids:
        raise ValueError(f"Physical unit_id appears in multiple sets: {duplicate_ids[:10]}")

    accepted: list[pd.DataFrame] = []
    unit_records: list[dict[str, Any]] = []
    history_hashes: dict[str, str] = {}
    quality = {"rejected_signal_rows": 0, "gap_boundaries": 0, "files_by_source": file_counts}
    for unit_id, unit in raw.groupby("unit_id", sort=True):
        unit = unit.sort_values("timestamp_s", kind="stable").reset_index(drop=True)
        times = unit["timestamp_s"].to_numpy(dtype=float)
        if (np.diff(times) <= 0).any():
            raise ValueError(f"Unit {unit_id} has repeated timestamp_s values")
        source_group = str(unit["_group"].iloc[0])
        valid = unit["_valid_signal"].to_numpy(dtype=bool)
        quality["rejected_signal_rows"] += int((~valid).sum())
        if valid.sum() < 2:
            raise ValueError(f"Unit {unit_id} needs at least two finite signal rows")
        # Hash physical observations without unit_id/file name to catch renamed
        # copies across manually supplied folders or different source files.
        content = unit[["timestamp_s", "_signal"]].to_csv(index=False, float_format="%.17g")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest in history_hashes:
            raise ValueError(f"Duplicate physical history: {unit_id} and {history_hashes[digest]}")
        history_hashes[digest] = unit_id

        selected = unit.loc[valid, ["unit_id", "timestamp_s", "_signal"]].copy()
        selected = selected.rename(columns={"_signal": "signal"})
        original_index = np.flatnonzero(valid)
        gaps = np.zeros(len(selected), dtype=bool)
        gaps[0] = True
        accepted_times = selected["timestamp_s"].to_numpy(dtype=float)
        prior_deltas: list[float] = []
        for i in range(1, len(selected)):
            delta = float(accepted_times[i] - accepted_times[i - 1])
            missing_between = original_index[i] != original_index[i - 1] + 1
            cadence_break = bool(prior_deltas) and delta > 3.0 * float(np.median(prior_deltas))
            gaps[i] = missing_between or cadence_break
            if gaps[i]:
                prior_deltas = []
            else:
                prior_deltas.append(delta)
        selected["gap_before"] = gaps
        quality["gap_boundaries"] += int(gaps[1:].sum())
        accepted.append(selected)
        unit_records.append(
            {
                "unit_id": unit_id,
                "origin_unit_id": unit_id,
                "source_group": source_group,
                "n_samples": len(selected),
                "observation_end_s": float(selected["timestamp_s"].iloc[-1]),
                "event_observed": False,
                "history_hash": digest,
                "rejected_signal_rows": int((~valid).sum()),
                "gap_boundaries": int(gaps[1:].sum()),
            }
        )
    features = pd.concat(accepted, ignore_index=True)
    features["signal"] = features["signal"].astype(float)
    if not all(math.isfinite(float(x)) for x in features["signal"]):
        raise ValueError("Admitted signal values must be finite")
    units = pd.DataFrame(unit_records)
    return features, units, quality
