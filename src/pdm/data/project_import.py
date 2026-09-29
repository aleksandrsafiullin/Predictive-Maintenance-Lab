"""Stage and validate project source files before publishing a source manifest."""

from __future__ import annotations

import hashlib
import math
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from pdm.data.generic_csv import read_generic_csv
from pdm.io_util import atomic_write_json
from pdm.projects import ProjectStore, project_store

MAX_SOURCE_FILES = 100_000
MAX_SOURCE_BYTES = 100 * 1024**3


def _relative_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\\" in raw or any(ord(c) < 32 for c in raw):
        raise ValueError("Invalid relative source path")
    if any(part in {"", ".", ".."} for part in raw.split("/")):
        raise ValueError(f"Unsafe relative source path: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe relative source path: {raw!r}")
    return str(path)


def _iter_source_files(spec: Mapping[str, Any]) -> list[tuple[Path, str]]:
    mode = spec.get("mode")
    if mode == "folder":
        root = Path(str(spec.get("path", ""))).expanduser()
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Source folder does not exist or is a symlink: {root}")
        files = [(p, _relative_path(p.relative_to(root).as_posix())) for p in root.rglob("*") if p.is_file()]
    elif mode == "files":
        supplied = spec.get("files")
        if not isinstance(supplied, list):
            raise ValueError("files mode requires a files list")
        files = []
        for item in supplied:
            if not isinstance(item, Mapping):
                raise ValueError("Each uploaded file needs path and relative_path")
            p = Path(str(item.get("path", ""))).expanduser()
            files.append((p, _relative_path(item.get("relative_path"))))
    else:
        raise ValueError("Source mode must be folder or files")
    files = sorted(
        [
            (path, rel) for path, rel in files
            if not any(part.startswith(".") or part == "__MACOSX" for part in PurePosixPath(rel).parts)
            and Path(rel).suffix.lower() not in {".md", ".txt", ".pdf"}
        ],
        key=lambda row: row[1],
    )
    if not files or len(files) > MAX_SOURCE_FILES:
        raise ValueError(f"Source needs 1–{MAX_SOURCE_FILES} files")
    names = [name.casefold() for _, name in files]
    if len(names) != len(set(names)):
        raise ValueError("Source has duplicate relative paths")
    for path, _ in files:
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Source file does not exist or is a symlink: {path}")
    return files


XJTU_BASELINE_THRESHOLDS: dict[str, Any] = {
    "mode": "initial_baseline_multiple", "direction": "above", "yellow": None, "red": None,
    "baseline_n": 5, "onset_sigma": 3.0, "onset_ratio": 1.25, "red_ratio": 2.0,
    "baseline_statistic": "median of first five causal max-axis RMS measurements",
    "yellow_formula": "max(median + onset_sigma * population_sd, onset_ratio * median)",
    "red_formula": "red_ratio * median",
}


def _baseline_thresholds(source_kind: str, thresholds: Mapping[str, Any]) -> dict[str, Any]:
    if source_kind != "xjtu_bearings":
        raise ValueError("Only XJTU-SY bearings imports can use the initial-baseline rule")
    if thresholds.get("direction", "above") != "above":
        raise ValueError("The initial-baseline rule only supports a rising signal")
    try:
        baseline_n = thresholds.get("baseline_n", 5)
        if isinstance(baseline_n, bool) or int(baseline_n) != baseline_n or int(baseline_n) < 1:
            raise ValueError
        params = {key: float(thresholds.get(key, XJTU_BASELINE_THRESHOLDS[key]))
                  for key in ("onset_sigma", "onset_ratio", "red_ratio")}
    except (TypeError, ValueError) as exc:
        raise ValueError("Baseline rule needs a positive integer baseline_n and numeric ratios") from exc
    if not all(math.isfinite(v) and v > 0 for v in params.values()):
        raise ValueError("Baseline rule ratios must be finite and positive")
    rule = {**XJTU_BASELINE_THRESHOLDS, "baseline_n": int(baseline_n), **params}
    if rule["baseline_n"] != XJTU_BASELINE_THRESHOLDS["baseline_n"]:
        rule["baseline_statistic"] = f"median of first {rule['baseline_n']} causal max-axis RMS measurements"
    return rule


def validate_thresholds(source_kind: str, thresholds: Any) -> dict[str, Any]:
    """Canonical saved zone rule: explicit absolute limits, the XJTU initial-baseline rule,
    or no rule at all for a generic CSV whose limits are set later on Data Quality."""
    if source_kind == "generic_sensor_csv" and not thresholds:
        return {}
    if isinstance(thresholds, Mapping) and thresholds.get("mode") == "initial_baseline_multiple":
        return _baseline_thresholds(source_kind, thresholds)
    return validate_absolute_thresholds(thresholds)


def validate_absolute_thresholds(thresholds: Any) -> dict[str, Any]:
    if not isinstance(thresholds, Mapping) or thresholds.get("mode") != "absolute":
        raise ValueError("Imported sources need explicitly declared absolute thresholds")
    direction = thresholds.get("direction")
    if direction not in {"above", "below"}:
        raise ValueError("Threshold direction must be above or below")
    try:
        yellow, red = float(thresholds["yellow"]), float(thresholds["red"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Yellow and red limits must be numeric") from exc
    if not all(math.isfinite(x) for x in (yellow, red)):
        raise ValueError("Yellow and red limits must be finite")
    if (direction == "above" and yellow >= red) or (direction == "below" and yellow <= red):
        raise ValueError("Red limit must be more severe than yellow limit")
    return {"mode": "absolute", "direction": direction, "yellow": yellow, "red": red}


def _signal_schema(source_kind: str, source: Mapping[str, Any]) -> dict[str, Any]:
    defaults = {
        "xjtu_bearings": ("combined_rms", "Combined max-axis RMS", "g"),
        "hse_filters": ("differential_pressure", "Differential pressure", "Pa"),
    }
    default_col, default_label, default_unit = defaults.get(source_kind, ("", "", ""))
    column = str(source.get("signal_column") or default_col).strip()
    label = str(source.get("signal_label") or default_label or column).strip()
    unit = str(source.get("signal_unit") or default_unit).strip()
    if not column or not label or not unit:
        raise ValueError("Signal column, label, and unit are required")
    reserved = {
        "unit_id", "timestamp_s", "split", "rul", "rul_s", "failure_label",
        "event_time_s", "event_observed", "observation_end_s", "life_fraction",
        "origin_unit_id", "author_split", "official_rul", "remaining_useful_life",
    }
    if column.casefold() in reserved:
        raise ValueError(f"{column} is an identity, outcome, or evaluation field, not a sensor signal")
    return {
        "source_kind": source_kind,
        "signal_column": column,
        "signal_label": label,
        "signal_unit": unit,
        "time_unit": "s",
        "input_columns": ["signal"],
        "output_domain": "real" if source_kind == "generic_sensor_csv" else "nonnegative",
        "thresholds": validate_thresholds(source_kind, source.get("thresholds")),
    }


def _split_request(source: Mapping[str, Any]) -> dict[str, Any]:
    modes = {group: source.get(f"{group}_mode", "auto") for group in ("validation", "test")}
    if any(mode not in {"auto", "folder"} for mode in modes.values()):
        raise ValueError("Validation and Test modes must be auto or folder")
    seed = source.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed > 2**32 - 1:
        raise ValueError("Split seed must be a nonnegative 32-bit integer")
    given = source.get("weights") or {"train": 0.7, "validation": 0.15, "test": 0.15}
    if not isinstance(given, Mapping) or set(given) != {"train", "validation", "test"}:
        raise ValueError("weights must contain train, validation, and test")
    try:
        weights = {key: float(given[key]) for key in ("train", "validation", "test")}
    except (TypeError, ValueError) as exc:
        raise ValueError("Split weights must be numeric") from exc
    if not all(math.isfinite(v) and v > 0 for v in weights.values()):
        raise ValueError("Split weights must be finite and positive")
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-8):
        raise ValueError("Split weights must sum to 1")
    for group, mode in modes.items():
        if mode == "folder" and not isinstance(source.get(group), Mapping):
            raise ValueError(f"{group} folder mode needs a source")
        if mode == "auto" and source.get(group) is not None:
            raise ValueError(f"{group} source supplied while mode is auto")
    return {"validation_mode": modes["validation"], "test_mode": modes["test"], "seed": seed, "weights": weights}


def import_project(
    project_id: str, source: Mapping[str, Any], *, store: ProjectStore | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Copy source files by stream, validate, and activate one complete manifest."""
    store = store or project_store()
    project = store.get(project_id)
    if project["storage_mode"] != "owned":
        raise ValueError("Linked legacy data is wrapped by prepare_project; it is not imported")
    if not isinstance(source, Mapping) or not isinstance(source.get("primary"), Mapping):
        raise ValueError("Primary source is required")
    split_request = _split_request(source)
    schema = _signal_schema(project["source_kind"], source)
    groups = {"primary": source["primary"]}
    for group in ("validation", "test"):
        if split_request[f"{group}_mode"] == "folder":
            groups[group] = source[group]

    parent = store.project_path(project_id) / "source"
    staging = Path(tempfile.mkdtemp(prefix=".import-", dir=parent))
    manifest_id = uuid.uuid4().hex
    destination = parent / manifest_id
    try:
        if should_stop and should_stop():
            raise InterruptedError("Project job cancelled")
        records: list[dict[str, Any]] = []
        staged_groups: dict[str, list[dict[str, Any]]] = {}
        total_size = 0
        for group, spec in groups.items():
            staged_groups[group] = []
            for original, rel in _iter_source_files(spec):
                if should_stop and should_stop():
                    raise InterruptedError("Project job cancelled")
                target = staging / group / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                hasher = hashlib.sha256()
                size = 0
                with original.open("rb") as src, target.open("xb") as dst:
                    while chunk := src.read(1024 * 1024):
                        if should_stop and should_stop():
                            raise InterruptedError("Project job cancelled")
                        size += len(chunk)
                        total_size += len(chunk)
                        if total_size > MAX_SOURCE_BYTES:
                            raise ValueError("Source exceeds maximum supported size")
                        hasher.update(chunk)
                        dst.write(chunk)
                rec = {"group": group, "relative_path": rel, "sha256": hasher.hexdigest(), "bytes": size}
                records.append(rec)
                if len(records) > MAX_SOURCE_FILES:
                    raise ValueError("Source has too many files")
                staged_groups[group].append({"path": str(target), "relative_path": rel})
        kind = project["source_kind"]
        if kind == "generic_sensor_csv":
            non_csv = [rec["relative_path"] for rec in records if not rec["relative_path"].lower().endswith(".csv")]
            if non_csv:
                raise ValueError(f"Generic source contains non-CSV files: {non_csv[:5]}")
            from pdm.data.project_prepare import allocate_project_split

            _, units, _ = read_generic_csv(staged_groups, schema["signal_column"])
            allocate_project_split(units, split_request)
        elif kind == "xjtu_bearings":
            from pdm.data.project_prepare import _owned_adapted, allocate_project_split

            zips = [rec for rec in records if rec["relative_path"].lower().endswith(".zip")]
            csvs = [rec for rec in records if rec["relative_path"].lower().endswith(".csv")]
            if zips and (len(zips) != 1 or csvs or zips[0]["group"] != "primary"):
                raise ValueError("XJTU ZIP import supports one primary archive; use extracted folders for multiple sets")
            _, units, _ = _owned_adapted(project, staging, {}, should_stop=should_stop)
            allocate_project_split(units, split_request)
        else:
            from pdm.data.project_prepare import _owned_adapted, allocate_project_split

            if any(rec["relative_path"].lower().endswith(".zip") for rec in records):
                raise ValueError("HSE ZIP import is not supported here; choose the extracted CSV folder")
            _, units, _ = _owned_adapted(project, staging, {}, should_stop=should_stop)
            allocate_project_split(units, split_request)
        if should_stop and should_stop():
            raise InterruptedError("Project job cancelled")
        digest = hashlib.sha256(
            "".join(f"{r['group']}:{r['relative_path']}:{r['sha256']}\n" for r in records).encode()
        ).hexdigest()
        manifest = {
            "manifest_id": manifest_id,
            "project_id": project_id,
            "source_kind": kind,
            "source_digest": digest,
            "files": records,
            "groups": {group: len(files) for group, files in staged_groups.items()},
            "split_request": split_request,
            "signal_schema": schema,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_write_json(staging / "source_manifest.json", manifest)
        if should_stop and should_stop():
            raise InterruptedError("Project job cancelled")
        staging.rename(destination)
        try:
            if should_stop and should_stop():
                raise InterruptedError("Project job cancelled")
            # Keep a working project's source/snapshot/run bound together until
            # prepare_project validates and activates the replacement snapshot.
            if not project["active_snapshot_id"]:
                store.update(project_id, source_manifest=manifest, state="imported")
        except Exception:
            shutil.rmtree(destination)
            raise
        return manifest
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
