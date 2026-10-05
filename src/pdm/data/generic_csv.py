"""Admit long-table sensor CSVs without inferring failure labels."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from pdm.red_entry_context import ENDPOINT_TIMESTAMPS, add_context, validate_context_mapping
from pdm.red_entry_targets import ENDPOINT_FIELDS

REQUIRED_COLUMNS = frozenset({"unit_id", "timestamp_s"})


def read_generic_csv(
    grouped_files: dict[str, list[dict[str, Any]]], signal_column: str, *,
    context_mapping: Mapping[str, str] | None = None,
    context_units: Mapping[str, str] | None = None, age_source: str = "counter",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return finite canonical rows, physical units, and quality observations.

    Invalid signal rows are removed only after marking the next accepted row as
    a segment boundary. A training/replay window must never cross gap_before.
    """
    if not signal_column or signal_column in {"unit_id", "timestamp_s"}:
        raise ValueError("Choose a numeric signal column distinct from unit_id and timestamp_s")
    mapping = validate_context_mapping(context_mapping)
    age_units = dict(context_units or {})
    for name in ENDPOINT_TIMESTAMPS:
        if name in age_units and age_units[name] not in {"s", "seconds"}:
            raise ValueError(f"{name} must use seconds on the same clock as timestamp_s")
    for name in ("operating_age_s", "operating_time_since_component_install_s", "operating_time_since_service_s"):
        if name in age_units and age_units[name] not in {"s", "seconds", "min", "h", "hours"}:
            raise ValueError(f"Unsupported age unit: {age_units[name]}")
    raw_frames: list[pd.DataFrame] = []
    file_counts: dict[str, int] = {}
    for group, files in grouped_files.items():
        file_counts[group] = len(files)
        for rec in files:
            path = Path(rec["path"])
            try:
                # Preserve float64 measurements and clocks from lossless CSV
                # exports rather than rounding their final significant digits.
                frame = pd.read_csv(path, float_precision="round_trip")
            except (OSError, pd.errors.ParserError, UnicodeError) as exc:
                raise ValueError(f"Cannot read CSV {rec['relative_path']}: {exc}") from exc
            missing = (REQUIRED_COLUMNS | {signal_column} | set(mapping.values())) - set(frame.columns)
            if missing:
                raise ValueError(f"CSV {rec['relative_path']} is missing columns: {sorted(missing)}")
            if frame.empty:
                raise ValueError(f"CSV {rec['relative_path']} is empty")
            context = pd.DataFrame({name: frame[column] for name, column in mapping.items()})
            frame = frame[["unit_id", "timestamp_s", signal_column]].copy()
            for name in context:
                frame[name] = context[name]
                if name in {"operating_age_s", "operating_time_since_component_install_s", "operating_time_since_service_s"}:
                    factor = {"s": 1., "seconds": 1., "min": 60., "h": 3600., "hours": 3600.}.get(age_units.get(name, "s"), 1.)
                    frame[name] = pd.to_numeric(frame[name], errors="coerce") * factor
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
    quality = {"rejected_signal_rows": 0, "gap_boundaries": 0, "files_by_source": file_counts,
               "endpoint_records": []}
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

        # Preserve cycle changes even when their signal row is rejected.
        raw_gaps = np.zeros(len(unit), dtype=bool)
        raw_gaps[0] = True
        prior_deltas = []
        for i in range(1, len(unit)):
            delta = float(times[i] - times[i - 1])
            raw_gaps[i] = bool(prior_deltas) and delta > 3.0 * float(np.median(prior_deltas))
            if raw_gaps[i]:
                prior_deltas = []
            else:
                prior_deltas.append(delta)
        unit["gap_before"] = raw_gaps
        contextual = add_context(unit[["unit_id", "timestamp_s", "_signal", "gap_before", *mapping]],
                                 age_source=age_source if "operating_age_s" in mapping else "unknown")
        # Endpoint evidence survives rejected telemetry, but never becomes a
        # signal origin. Raw episode bounds also retain entirely unobserved
        # episodes without assigning their records to a neighbouring cycle.
        cycles = contextual["component_cycle_id"].astype(str).to_numpy()
        replaced = contextual.get("component_replaced", pd.Series(False, index=contextual.index)).fillna(False).to_numpy(bool)
        boundaries = np.r_[True, (cycles[1:] != cycles[:-1]) | replaced[1:]]
        episode_numbers = np.cumsum(boundaries) - 1
        starts = np.flatnonzero(boundaries)
        for index in np.flatnonzero(~valid):
            row = contextual.iloc[index]
            fields = {name: row[name] for name in ENDPOINT_FIELDS if name in contextual and pd.notna(row[name])}
            if fields:
                number = int(episode_numbers[index])
                record = {"unit_id": unit_id, "physical_unit_id": str(row.physical_unit_id),
                          "component_cycle_id": str(row.component_cycle_id),
                          "timestamp_s": float(row.timestamp_s),
                          "episode_start_timestamp_s": float(times[starts[number]]),
                          "episode_end_timestamp_s": float(times[starts[number + 1]]) if number + 1 < len(starts) else None,
                          **fields}
                # Pandas extension booleans/numbers must serialize as JSON scalars.
                quality["endpoint_records"].append({key: value.item() if isinstance(value, np.generic) else value
                                                    for key, value in record.items()})
        context_names = [name for name in contextual if name not in {"unit_id", "timestamp_s", "_signal", "gap_before"} and not name.startswith("_")]
        selected = contextual.loc[valid, ["unit_id", "timestamp_s", "_signal", *context_names]].copy()
        selected = selected.rename(columns={"_signal": "signal"})
        original_index = np.flatnonzero(valid)
        if "component_replaced" in selected:
            for i in range(1, len(selected)):
                previous, current = original_index[i - 1:i + 1]
                if episode_numbers[current] != episode_numbers[previous] and cycles[current] == cycles[previous]:
                    selected.loc[selected.index[i], "component_replaced"] = True
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
                "physical_unit_id": str(selected["physical_unit_id"].iloc[0]),
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
