"""Versioned causal context and separately marked, supplied outcome records."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from pdm.red_entry_targets import ENDPOINT_FIELDS

ENDPOINT_FLAGS = frozenset({"confirmed_failure", "emergency_stop", "planned_maintenance", "component_replaced"})
ENDPOINT_TIMESTAMPS = frozenset(ENDPOINT_FIELDS) - ENDPOINT_FLAGS

CONTEXT_FIELDS = frozenset({
    "physical_unit_id", "component_cycle_id", "component_replaced", "is_running",
    "operating_age_s", "operating_age_known", "operating_age_source",
    "operating_time_since_component_install_s", "operating_time_since_service_s",
    "rpm", "load_kn", "flow_rate", "dust_feed", "dust", "temperature",
})
CONTEXT_UNITS = {
    "operating_age_s": "s", "observed_elapsed_s": "s", "delta_t_s": "s",
    "operating_time_since_component_install_s": "s", "operating_time_since_service_s": "s",
    "rpm": "rpm", "load_kn": "kN", "flow_rate": "source_unspecified",
    "dust_feed": "source_unspecified", "temperature": "source_unspecified",
}
AGE_SOURCES = {"unknown", "counter", "laboratory_proxy", "running_clock"}


def validate_context_mapping(mapping: Mapping[str, str] | None) -> dict[str, str]:
    result = dict(mapping or {})
    unknown = set(result) - CONTEXT_FIELDS - set(ENDPOINT_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported context roles: {sorted(unknown)}")
    if any(not isinstance(v, str) or not v.strip() for v in result.values()):
        raise ValueError("Context mappings need nonempty source column names")
    return result


def _boolean(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.lower().str.strip()
    invalid = series.notna() & ~values.isin(["true", "false", "1", "0", "1.0", "0.0"])
    if invalid.any():
        raise ValueError("Boolean context must be true/false or 1/0")
    return values.map({"true": True, "false": False, "1": True, "0": False,
                       "1.0": True, "0.0": False}).astype("boolean")


def add_context(frame: pd.DataFrame, *, age_source: str = "unknown") -> pd.DataFrame:
    """Preserve counters and derive only past context, never from record end.

    running_clock integrates the previously observed running state over known
    consecutive intervals. Missing telemetry loses certainty, never resets age.
    An explicit replacement starts a component cycle; service does not reset it.
    Laboratory age is retained as a proxy, separate from physical counter age.
    """
    if age_source not in AGE_SOURCES:
        raise ValueError(f"Unsupported age source: {age_source}")
    out = frame.copy()
    if "physical_unit_id" not in out:
        out["physical_unit_id"] = out["unit_id"].astype(str)
    if out["physical_unit_id"].isna().any() or out["physical_unit_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Physical unit identity must be nonempty")
    for name in {"is_running", "operating_age_known", *ENDPOINT_FLAGS}:
        if name in out:
            out[name] = _boolean(out[name])
    for name in ENDPOINT_TIMESTAMPS:
        if name in out:
            supplied = out[name]
            numeric = pd.to_numeric(supplied, errors="coerce")
            if (supplied.notna() & ~np.isfinite(numeric)).any():
                raise ValueError(f"{name} must contain finite timestamps in seconds or missing values")
            out[name] = numeric.astype(float)
    for name in CONTEXT_UNITS:
        if name in out:
            out[name] = pd.to_numeric(out[name], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if "operating_age_s" not in out:
        out["operating_age_s"] = np.nan
    if (out["operating_age_s"].dropna() < 0).any():
        raise ValueError("Operating age cannot be negative")
    for name in ("rpm", "load_kn", "flow_rate", "dust_feed", "temperature"):
        if name in out:
            out[name + "_known"] = np.isfinite(out[name].to_numpy(float))
    if "dust" in out:
        out["dust_known"] = out["dust"].notna() & out["dust"].astype(str).str.strip().ne("")
    parts = []
    for _, group in out.groupby("unit_id", sort=False):
        group = group.copy()
        if group["physical_unit_id"].astype(str).nunique() != 1:
            raise ValueError("A unit history must have one physical unit identity")
        times = group["timestamp_s"].to_numpy(float)
        group["observed_elapsed_s"] = times - times[0]
        group["delta_t_s"] = np.r_[0., np.diff(times)]
        replacement = group.get("component_replaced", pd.Series(False, index=group.index)).fillna(False).to_numpy(bool)
        if "component_cycle_id" not in group:
            group["component_cycle_id"] = np.cumsum(replacement).astype(str)
        elif group["component_cycle_id"].isna().any():
            raise ValueError("Component cycle identity must be nonempty")
        if group["component_cycle_id"].astype(str).str.strip().eq("").any():
            raise ValueError("Component cycle identity must be nonempty")
        cycles = group["component_cycle_id"].astype(str).to_numpy()
        age = group["operating_age_s"].to_numpy(float).copy()
        supplied_known = group.get("operating_age_known", pd.Series(True, index=group.index)).fillna(False).to_numpy(bool)
        age[~supplied_known] = np.nan
        running = group.get("is_running", pd.Series(pd.NA, index=group.index, dtype="boolean"))
        gaps = group.get("gap_before", pd.Series(False, index=group.index)).to_numpy(bool)
        for i in range(len(group)):
            new_cycle = i > 0 and cycles[i] != cycles[i - 1]
            if replacement[i] and not np.isfinite(age[i]):
                age[i] = 0.
            if i and np.isfinite(age[i]) and np.isfinite(age[i - 1]) and age[i] < age[i - 1] and not (replacement[i] or new_cycle):
                raise ValueError("Operating counter decreased without an explicit replacement cycle")
            if age_source == "running_clock" and i and not np.isfinite(age[i]) and not new_cycle and not gaps[i]:
                if np.isfinite(age[i - 1]) and pd.notna(running.iloc[i - 1]):
                    age[i] = age[i - 1] + (times[i] - times[i - 1]) * bool(running.iloc[i - 1])
        known = np.isfinite(age)
        if age_source == "unknown":
            known[:] = False
            age[:] = np.nan
        group["operating_age_s"] = age
        group["operating_age_known"] = known
        if "operating_age_source" in group:
            sources = group["operating_age_source"].fillna("unknown").astype(str)
            if not sources.isin(AGE_SOURCES).all():
                raise ValueError("Unsupported row age source")
            known &= sources.ne("unknown").to_numpy()
            group["operating_age_known"] = known
            group["operating_age_s"] = np.where(known, age, np.nan)
            group["operating_age_source"] = np.where(known, sources, "unknown")
        else:
            group["operating_age_source"] = np.where(known, age_source, "unknown")
        # Installation-relative running age requires explicit import evidence.
        for name in ("operating_time_since_component_install_s", "operating_time_since_service_s"):
            if name in group:
                if (group[name].dropna() < 0).any():
                    raise ValueError(f"{name} cannot be negative")
                group[name + "_known"] = np.isfinite(group[name].to_numpy(float))
        parts.append(group)
    return pd.concat(parts).sort_index()


def context_schema(frame: pd.DataFrame, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Retain v1 signal inputs; expose v2 context with explicit semantic roles."""
    units = {**CONTEXT_UNITS, **dict(schema.get("context_units") or {})}
    for name in ("operating_age_s", "operating_time_since_component_install_s", "operating_time_since_service_s"):
        units[name] = "s"
    for name in ENDPOINT_TIMESTAMPS:
        units[name] = "s"
    columns = {}
    for name in frame.columns:
        role = ("target_only" if name in ENDPOINT_FIELDS else "sensor" if name == "signal" else "known_age" if name in {
                    "operating_age_s", "operating_time_since_component_install_s", "operating_time_since_service_s"}
                else "quality" if name in {"gap_before", "delta_t_s", "context_quality_errors"} or name.endswith("_known")
                else "operating_context" if name in {"rpm", "load_kn", "flow_rate", "dust_feed", "dust", "temperature", "is_running"}
                else "metadata")
        columns[name] = {"role": role, "unit": schema.get("signal_unit") if name == "signal" else "s" if name == "timestamp_s" else units.get(name, "dimensionless"),
                         "provenance": {"source_kind": schema.get("source_kind", "unknown"),
                                        "source_column": schema.get("context_mapping", {}).get(name, name),
                                        "method": "past_elapsed" if name in {"observed_elapsed_s", "delta_t_s"} else "imported_or_masked"},
                         "available_at": "target_only" if name in ENDPOINT_FIELDS else "timestamp_s"}
    return {**schema, "schema_version": 2, "context_version": "red_entry_context_v2",
            "input_columns": [name for name in schema.get("input_columns", ["signal"]) if name not in ENDPOINT_FIELDS],
            "columns": columns,
            "red_entry_input_columns": [name for name, spec in columns.items() if spec["role"] in {"sensor", "known_age", "operating_context", "quality"}],
            "context_columns": [name for name in frame if name not in {"unit_id", "timestamp_s", "signal", "gap_before"}],
            "age_semantics": "laboratory_proxy is not confirmed industrial component running age"}


def assert_physical_split(units: pd.DataFrame, split: Mapping[str, Any]) -> None:
    if "physical_unit_id" not in units:
        return
    owner = {str(uid): name for name in ("train", "validation", "test") for uid in split[name]}
    assigned = units.copy()
    assigned["_set"] = assigned["unit_id"].astype(str).map(owner)
    if assigned.groupby("physical_unit_id")["_set"].nunique().gt(1).any():
        raise ValueError("Split leak: cycles of one physical unit cross data sets")
