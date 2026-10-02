"""Build and verify immutable, project-scoped signal snapshots."""

from __future__ import annotations

import hashlib
import math
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal, Mapping

import numpy as np
import pandas as pd

from pdm.data.generic_csv import read_generic_csv
from pdm.data.project_import import XJTU_BASELINE_THRESHOLDS, validate_absolute_thresholds
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.projects import ProjectStore, _safe_id, project_store
from pdm.splits import assert_split_coverage, split_hash

SNAPSHOT_FILES = (
    "features.parquet",
    "units.parquet",
    "split.json",
    "data_report.json",
    "feature_schema.json",
)
ZONE_LIMITS_FILE = "zone_limits.json"
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_LABELS = {"train": "Training Data", "validation": "Validation Data", "test": "Testing Data"}
MANUAL_SPLIT_PROTOCOL = "whole_unit_project_v1_manual"
LINKED_LEGACY_MOVE_ERROR = ("This project uses the published split of its source dataset. "
                            "Create a new project to change the split.")
JOB_ACTIVE_MOVE_ERROR = "Wait for the current job to finish before changing sets."
STALE_SNAPSHOT_ERROR = "The data changed since this page loaded. Reload Data Quality and try again."


def allocate_project_split(units: pd.DataFrame, request: Mapping[str, Any]) -> dict[str, Any]:
    """Allocate whole physical units, renormalizing weights over auto groups."""
    modes = {name: request.get(f"{name}_mode", "auto") for name in ("validation", "test")}
    weights = dict(request.get("weights") or {"train": .7, "validation": .15, "test": .15})
    seed = int(request.get("seed", 42))
    if units["unit_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate physical unit IDs")
    indexed_units = units.set_index("unit_id")
    groups = indexed_units.get("source_group")
    if groups is None:
        groups = pd.Series("primary", index=units["unit_id"])
    manual = {
        name: sorted(groups.index[groups.eq(name)].astype(str).tolist())
        for name in ("validation", "test")
    }
    # Official HSE test histories retain their protected split group, but
    # only those imported from the selected Testing source can fill it.
    source_folders = indexed_units.get("source_folder", groups)
    folder_author_test = groups.eq("author_test") & source_folders.eq("test")
    for name in ("validation", "test"):
        has_folder_units = bool(manual[name]) or (name == "test" and folder_author_test.any())
        if modes[name] == "folder" and not has_folder_units:
            raise ValueError(f"Separate {name} folder has no admitted units")
        if modes[name] == "auto" and has_folder_units:
            raise ValueError(f"Unexpected manual {name} units")
    primary = sorted(groups.index[groups.eq("primary")].astype(str).tolist())
    # HSE's official author-test histories remain test-only, regardless of
    # requested custom ratios for the author-train pool.
    author_test = sorted(groups.index[groups.eq("author_test")].astype(str).tolist())
    if not primary:
        raise ValueError("Primary Training pool has no admitted units")
    automatic = ["train"] + [
        name for name in ("validation", "test")
        if modes[name] == "auto" and not (name == "test" and author_test)
    ]
    if len(primary) < len(automatic):
        raise ValueError(
            f"Primary pool needs at least {len(automatic)} physical units for automatic groups; found {len(primary)}"
        )
    total_weight = sum(float(weights[name]) for name in automatic)
    proportions = {name: float(weights[name]) / total_weight for name in automatic}
    # Whole-unit largest-remainder allocation, then enforce one per automatic
    # group without changing the requested relative weights more than needed.
    quotas = {name: len(primary) * proportions[name] for name in automatic}
    counts = {name: int(math.floor(quotas[name])) for name in automatic}
    leftover = len(primary) - sum(counts.values())
    order = sorted(automatic, key=lambda name: (-(quotas[name] % 1), automatic.index(name)))
    for name in order[:leftover]:
        counts[name] += 1
    for name in automatic:
        if counts[name] == 0:
            donor = max((candidate for candidate in automatic if counts[candidate] > 1),
                        key=lambda candidate: (counts[candidate] - quotas[candidate], counts[candidate]),
                        default=None)
            if donor is None:
                raise ValueError("Too few primary units for the requested automatic groups")
            counts[donor] -= 1
            counts[name] = 1
    permutation = np.random.default_rng(seed).permutation(primary).tolist()
    split = {"train": [], "validation": manual["validation"], "test": manual["test"] + author_test}
    cursor = 0
    for name in automatic:
        split[name] += sorted(permutation[cursor : cursor + counts[name]])
        cursor += counts[name]
    split.update(
        protocol="whole_unit_project_v1",
        seed=seed,
        desired_weights=weights,
        realized_counts={name: len(split[name]) for name in ("train", "validation", "test")},
        manual_modes=modes,
    )
    assert_split_coverage(units.copy(), split)
    if not all(split[name] for name in ("train", "validation", "test")):
        raise ValueError("Train, Validation, and Test each need at least one physical unit")
    return split


def _source_files(root: Path, manifest: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    if root.is_symlink():
        raise ValueError("Imported source root is a symlink")
    for rec in manifest["files"]:
        path = root / rec["group"] / rec["relative_path"]
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or any(part.is_symlink() for part in (path, *path.parents))
            or sha256_file(path) != rec["sha256"]
        ):
            raise ValueError(f"Imported source changed: {rec['relative_path']}")
        grouped.setdefault(rec["group"], []).append(
            {"path": str(path), "relative_path": rec["relative_path"]}
        )
    digest = hashlib.sha256(
        "".join(f"{r['group']}:{r['relative_path']}:{r['sha256']}\n" for r in manifest["files"]).encode()
    ).hexdigest()
    if digest != manifest.get("source_digest"):
        raise ValueError("Source manifest digest mismatch")
    return grouped


def _canonicalize_adapted(
    features: pd.DataFrame, units: pd.DataFrame, *, signal: pd.Series,
    source_groups: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    context_quality_rows = 0
    if "_quality_errors" in features:
        errors = features["_quality_errors"].fillna("").astype(str)
        bad_clock = errors.str.contains("nonmonotonic_source_time", regex=False)
        if bad_clock.any():
            affected = sorted(features.loc[bad_clock, "unit_id"].astype(str).unique().tolist())
            raise ValueError(f"Nonmonotonic source time in physical units {affected[:10]}; correct source clock before import")
        context_quality_rows = int(errors.str.contains("missing_dust", regex=False).sum())
    frame = pd.DataFrame({
        "unit_id": features["unit_id"].astype(str),
        "timestamp_s": pd.to_numeric(features["timestamp_s"], errors="coerce"),
        "signal": pd.to_numeric(signal, errors="coerce"),
        "gap_before": features["gap_before"].astype(bool) if "gap_before" in features else False,
    }).sort_values(["unit_id", "timestamp_s"], kind="stable")
    groups = []
    rejected = 0
    rejected_by_unit: dict[str, int] = {}
    physical_histories: dict[str, str] = {}
    for unit_id, group in frame.groupby("unit_id", sort=True):
        group = group.copy()
        good = np.isfinite(group["timestamp_s"].to_numpy(float)) & np.isfinite(group["signal"].to_numpy(float))
        rejected += int((~good).sum())
        rejected_by_unit[str(unit_id)] = int((~good).sum())
        selected = group.loc[good].copy()
        if len(selected) < 2:
            raise ValueError(f"Unit {unit_id} has fewer than two finite sensor points")
        if np.any(np.diff(selected["timestamp_s"].to_numpy(float)) <= 0):
            raise ValueError(f"Unit {unit_id} has duplicate or nonincreasing signal timestamps")
        indexes = np.flatnonzero(good)
        gaps = selected["gap_before"].to_numpy(bool).copy()
        gaps[0] = True
        gaps[1:] |= np.diff(indexes) > 1
        selected["gap_before"] = gaps
        content = selected[["timestamp_s", "signal"]].to_csv(index=False, float_format="%.17g")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest in physical_histories:
            raise ValueError(f"Duplicate physical history: {unit_id} and {physical_histories[digest]}")
        physical_histories[digest] = str(unit_id)
        groups.append(selected)
    canonical = pd.concat(groups, ignore_index=True)
    if units["unit_id"].astype(str).duplicated().any():
        raise ValueError("Adapter returned duplicate physical units")
    new_units = units.copy()
    new_units["unit_id"] = new_units["unit_id"].astype(str)
    if "origin_unit_id" not in new_units:
        new_units["origin_unit_id"] = new_units["unit_id"]
    new_units["source_group"] = new_units["unit_id"].map(source_groups or {}).fillna("primary")
    counts = canonical.groupby("unit_id").size()
    new_units["n_samples"] = new_units["unit_id"].map(counts).astype(int)
    interior_gaps = canonical.groupby("unit_id")["gap_before"].sum() - 1
    new_units["gap_boundaries"] = new_units["unit_id"].map(interior_gaps).astype(int)
    new_units["rejected_signal_rows"] = new_units["unit_id"].map(rejected_by_unit).astype(int)
    return canonical, new_units, {
        "rejected_signal_rows": rejected,
        "gap_boundaries": int(new_units["gap_boundaries"].sum()),
        "descriptive_context_quality_rows": context_quality_rows,
        "descriptive_context_note": "Dust/flow/feed are excluded from signal-only model inputs",
    }


def _owned_adapted(
    project: Mapping[str, Any], source_root: Path, manifest: Mapping[str, Any],
    *, should_stop: Callable[[], bool] | None = None,
):
    from pdm.config import load_dataset_config

    kind = project["source_kind"]
    if should_stop and should_stop():
        raise InterruptedError("Project job cancelled")
    if kind == "xjtu_bearings":
        from pdm.data.bearings import build_bearing_units, extract_bearings_features

        cfg = load_dataset_config("bearings")
        def progress(_stage: str, _info: dict[str, Any]) -> None:
            if should_stop and should_stop():
                raise InterruptedError("Project job cancelled")

        features = extract_bearings_features(cfg, raw_dir=source_root, progress=progress)
        progress("done", {})
        units = build_bearing_units(features, cfg)
        source_groups: dict[str, str] = {}
        single_primary_zip = len(list(source_root.rglob("*.zip"))) == 1 and not list(source_root.rglob("*.csv"))
        for uid, group in features.groupby("unit_id"):
            rels = group["relpath"].astype(str)
            group_names = {"primary"} if single_primary_zip else {rel.split("/", 1)[0] for rel in rels}
            if len(group_names) != 1:
                raise ValueError(f"Bearing {uid} spans source groups")
            source_groups[str(uid)] = group_names.pop()
        signal = np.maximum(features["horizontal_rms"], features["vertical_rms"])
        return _canonicalize_adapted(features, units, signal=signal, source_groups=source_groups)
    from pdm.data.filters import (
        FILTER_TIME_TO_SECONDS,
        PRESSURE_LIMIT_PA,
        _measurements_from_csv,
        _require_columns,
        _test_unit_table,
        _train_unit_table,
    )
    from pdm.windows import filter_gap_params

    cfg = load_dataset_config("filters")
    factor = float(cfg.get("time_to_seconds", FILTER_TIME_TO_SECONDS))
    limit = float(cfg.get("pressure_limit_pa", PRESSURE_LIMIT_PA))
    gap_multiplier, sampling_interval_s = filter_gap_params(cfg)
    feature_parts = []
    unit_parts = []
    source_groups = {}
    seen_primary_train = False
    for group_name in ("primary", "validation", "test"):
        if should_stop and should_stop():
            raise InterruptedError("Project job cancelled")
        group_root = source_root / group_name
        if not group_root.is_dir():
            continue
        for basename, author in (("Train_Data_CSV.csv", "author_train"),
                                 ("Test_Data_CSV.csv", "author_test")):
            if should_stop and should_stop():
                raise InterruptedError("Project job cancelled")
            matches = list(group_root.rglob(basename))
            if len(matches) > 1:
                raise ValueError(f"Multiple {basename} files in {group_name} source")
            if not matches:
                continue
            path = matches[0]
            raw = pd.read_csv(path)
            _require_columns(
                raw,
                ["Data_No", "Differential_pressure", "Flow_rate", "Time", "Dust_feed", "Dust"],
                path,
            )
            feat = _measurements_from_csv(
                raw, split_source=author, time_factor=factor,
                gap_multiplier=gap_multiplier, sampling_interval_s=sampling_interval_s,
            )
            if author == "author_train":
                unit_table = _train_unit_table(feat, limit)
                seen_primary_train |= group_name == "primary"
            else:
                unit_table = _test_unit_table(raw, feat, limit, factor)
                if group_name == "validation":
                    raise ValueError("Official HSE Test_Data_CSV.csv cannot be supplied as Validation")
            # Author Data_No is local to each source file. Namespace the
            # manual folders, while retaining their original identifiers for
            # provenance; duplicate physical readings are checked below.
            prefix = "" if group_name == "primary" else f"{group_name}_"
            feat["unit_id"] = prefix + feat["unit_id"].astype(str)
            unit_table["unit_id"] = prefix + unit_table["unit_id"].astype(str)
            unit_table["origin_unit_id"] = unit_table["unit_id"]
            unit_table["source_folder"] = group_name
            group_for_split = "author_test" if author == "author_test" else group_name
            source_groups.update({str(uid): group_for_split for uid in unit_table["unit_id"]})
            feature_parts.append(feat)
            unit_parts.append(unit_table)
    if not seen_primary_train:
        raise ValueError("HSE primary source needs Train_Data_CSV.csv")
    features = pd.concat(feature_parts, ignore_index=True)
    units = pd.concat(unit_parts, ignore_index=True)
    return _canonicalize_adapted(
        features, units, signal=features["differential_pressure"], source_groups=source_groups
    )


def _legacy_adapted(project: Mapping[str, Any]):
    from pdm.data.prepare import load_processed

    dataset_id = project["source_manifest"]["legacy_dataset_id"]
    legacy = load_processed(dataset_id)
    features, units = legacy["features"], legacy["units"]
    if dataset_id == "bearings":
        signal = np.maximum(features["horizontal_rms"], features["vertical_rms"])
        schema = {
            "source_kind": "xjtu_bearings", "signal_column": "combined_rms",
            "signal_label": "Combined max-axis RMS", "signal_unit": "g", "time_unit": "s",
            "input_columns": ["signal"], "output_domain": "nonnegative",
            "thresholds": dict(XJTU_BASELINE_THRESHOLDS),
        }
    else:
        signal = features["differential_pressure"]
        schema = {
            "source_kind": "hse_filters", "signal_column": "differential_pressure",
            "signal_label": "Differential pressure", "signal_unit": "Pa", "time_unit": "s",
            "input_columns": ["signal"], "output_domain": "nonnegative",
            "thresholds": {"mode": "absolute", "direction": "above", "yellow": 300.0, "red": 600.0,
                           "note": "Provisional laboratory bands; not confirmed industrial fault limits"},
        }
    canonical, normalized_units, quality = _canonicalize_adapted(features, units, signal=signal)
    return canonical, normalized_units, quality, legacy["split"], schema, legacy["fingerprint"]


def prepare_project(
    project_id: str, source_manifest_id: str, *, store: ProjectStore | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    store = store or project_store()
    project = store.get(project_id)
    _safe_id(source_manifest_id, "source manifest ID")
    manifest = project["source_manifest"]
    if project["storage_mode"] == "owned":
        manifest_path = store._owned_path(project_id, "source", source_manifest_id, "source_manifest.json")
        if not manifest_path.is_file() or manifest_path.is_symlink():
            raise ValueError("Source manifest is missing")
        manifest = read_json(manifest_path)
        if (manifest.get("manifest_id") != source_manifest_id
                or manifest.get("project_id") != project_id
                or manifest.get("source_kind") != project["source_kind"]):
            raise ValueError("Source manifest binding does not match this project")
    elif not manifest or source_manifest_id != manifest.get("manifest_id"):
        raise ValueError("Source manifest ID does not match this linked project")
    if should_stop and should_stop():
        raise InterruptedError("Project job cancelled")
    if project["storage_mode"] == "linked_legacy":
        features, units, quality, source_split, schema, legacy_fp = _legacy_adapted(project)
        split = dict(source_split)
        split.update(protocol=source_split.get("protocol", "linked_legacy_prepared"),
                     seed=None, desired_weights=None,
                     realized_counts={name: len(split[name]) for name in ("train", "validation", "test")})
        source_digest = str(legacy_fp.get("split_hash") or legacy_fp.get("dataset_version") or "legacy")
    else:
        source_root = store.project_path(project_id) / "source" / source_manifest_id
        grouped = _source_files(source_root, manifest)
        schema = dict(manifest["signal_schema"])
        if project["source_kind"] == "generic_sensor_csv":
            features, units, quality = read_generic_csv(grouped, schema["signal_column"])
        else:
            features, units, quality = _owned_adapted(
                project, source_root, manifest, should_stop=should_stop
            )
        split = allocate_project_split(units, manifest["split_request"])
        source_digest = manifest["source_digest"]
    assert_split_coverage(units.copy(), split)
    if should_stop and should_stop():
        raise InterruptedError("Project job cancelled")

    def write_data(staging: Path) -> None:
        features.to_parquet(staging / "features.parquet", index=False)
        units.to_parquet(staging / "units.parquet", index=False)
        atomic_write_json(staging / "feature_schema.json", schema)

    report = {
        "project_id": project_id, "snapshot_id": None,
        "source_kind": project["source_kind"], "source_digest": source_digest,
        "n_units": len(units), "n_rows": len(features),
        "split_counts": split["realized_counts"], "by_split": _by_split(features, units, split),
        "quality": quality,
        "outcome_semantics": "unlabelled_observations" if project["source_kind"] == "generic_sensor_csv" else "source_specific",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    base = store.project_path(project_id) / "snapshots"
    staging, snapshot_id, payload = _stage_snapshot(
        base, project_id, write_data=write_data, split=split, report=report,
        fingerprint={"source_digest": source_digest, "source_manifest_id": source_manifest_id},
    )
    try:
        if should_stop and should_stop():
            raise InterruptedError("Project job cancelled")
        destination = store.snapshot_path(project_id, snapshot_id)
    except Exception:
        shutil.rmtree(staging)
        raise

    def activate() -> None:
        if should_stop and should_stop():
            raise InterruptedError("Project job cancelled")
        store.update(project_id, source_manifest=manifest, active_snapshot_id=snapshot_id,
                     selected_run_id=None, state="ready")

    _publish_snapshot(staging, destination, activate)
    return {"project_id": project_id, "snapshot_id": snapshot_id, "dir": destination,
            "split": split, "report": payload["report"], "schema": schema,
            "fingerprint": payload["fingerprint"]}


def _by_split(features: pd.DataFrame, units: pd.DataFrame, split: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    by_split = {}
    for group_name in SPLIT_NAMES:
        selected = units[units["unit_id"].astype(str).isin(split[group_name])]
        by_split[group_name] = {
            "units": len(selected),
            "rows": int(features["unit_id"].astype(str).isin(split[group_name]).sum()),
            "rejected_signal_rows": int(selected.get("rejected_signal_rows", pd.Series(dtype=int)).sum()),
            "gap_boundaries": int(selected.get("gap_boundaries", pd.Series(dtype=int)).sum()),
        }
    return by_split


def _stage_snapshot(
    base: Path, project_id: str, *, write_data: Callable[[Path], None],
    split: Mapping[str, Any], report: Mapping[str, Any], fingerprint: Mapping[str, Any],
) -> tuple[Path, str, dict[str, Any]]:
    """Write a complete snapshot into a private staging dir under ``base``.

    Takes no registry lock. On failure the staging dir is removed.
    ``write_data`` writes features.parquet, units.parquet and feature_schema.json.
    """
    snapshot_id = uuid.uuid4().hex
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=base))
    try:
        write_data(staging)
        atomic_write_json(staging / "split.json", split)
        full_report = {**report, "snapshot_id": snapshot_id}
        atomic_write_json(staging / "data_report.json", full_report)
        hashes = {filename: sha256_file(staging / filename) for filename in SNAPSHOT_FILES}
        full_fingerprint = {
            "schema_version": 1, "project_id": project_id, "snapshot_id": snapshot_id,
            **fingerprint, "split_hash": split_hash(split), "file_hashes": hashes,
        }
        atomic_write_json(staging / "processed_fingerprint.json", full_fingerprint)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return staging, snapshot_id, {"report": full_report, "fingerprint": full_fingerprint}


def _publish_snapshot(staging: Path, destination: Path, activate: Callable[[], Any]) -> None:
    """Rename a staged snapshot into place, then run ``activate``.

    Never takes or assumes the registry lock; ``activate`` decides how the
    project record is written. Any failure leaves neither dir behind.
    """
    try:
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    try:
        activate()
    except BaseException:
        shutil.rmtree(destination)
        raise


def _copy_snapshot_file(source: Path, target: Path) -> None:
    shutil.copyfile(source, target)


def _fixed_test_units(snapshot: Mapping[str, Any]) -> list[str]:
    units = snapshot.get("units")
    if units is None or "source_group" not in getattr(units, "columns", ()):
        return []
    return sorted(units.loc[units["source_group"].astype(str).eq("author_test"), "unit_id"].astype(str).tolist())


def preview_move(snapshot: Mapping[str, Any], unit_ids, destination: str) -> dict[str, Any]:
    """Validate a whole-unit move against a snapshot without any I/O."""
    if destination not in SPLIT_NAMES:
        raise ValueError(f"Unknown set: {destination}")
    split = snapshot["split"]
    current = {uid: name for name in SPLIT_NAMES for uid in map(str, split[name])}
    counts = {name: len(split[name]) for name in SPLIT_NAMES}
    fixed = _fixed_test_units(snapshot)
    moved = list(dict.fromkeys(map(str, unit_ids or ())))

    def result(problem: str | None, projected: dict[str, int] | None = None) -> dict[str, Any]:
        return {"counts": projected or counts, "problem": problem, "fixed_units": fixed}

    if not moved:
        return result("Choose at least one unit.")
    unknown = next((uid for uid in moved if uid not in current), None)
    if unknown is not None:
        return result(f"Unknown unit: {unknown}.")
    already = next((uid for uid in moved if current[uid] == destination), None)
    if already is not None:
        return result(f"{already} is already in {SPLIT_LABELS[destination]}.")
    if destination != "test" and set(moved) & set(fixed):
        return result("Official HSE test units stay in Testing Data.")
    projected = dict(counts)
    for uid in moved:
        projected[current[uid]] -= 1
        projected[destination] += 1
    empty = next((name for name in SPLIT_NAMES if projected[name] == 0), None)
    if empty is not None:
        return result(f"{SPLIT_LABELS[empty]} would have no units. Keep at least one unit in each set.", projected)
    return result(None, projected)


def _swap_unit_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


def preview_swap(snapshot: Mapping[str, Any], unit_a, unit_b) -> dict[str, Any]:
    """Validate exchanging two whole units between sets without any I/O."""
    split = snapshot["split"]
    current = {uid: name for name in SPLIT_NAMES for uid in map(str, split[name])}
    counts = {name: len(split[name]) for name in SPLIT_NAMES}
    fixed = _fixed_test_units(snapshot)

    def result(
        problem: str | None,
        projected: dict[str, int] | None = None,
        origin: dict[str, str] | None = None,
        destination: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        return {
            "counts": projected or counts,
            "problem": problem,
            "fixed_units": fixed,
            "from": origin,
            "to": destination,
        }

    left, right = _swap_unit_text(unit_a), _swap_unit_text(unit_b)
    if left is None or right is None or left == right:
        return result("Choose two different units.")
    if left not in current:
        return result(f"Unknown unit: {left}.")
    if right not in current:
        return result(f"Unknown unit: {right}.")
    set_a, set_b = current[left], current[right]
    if set_a == set_b:
        return result("Units must be in different sets.")
    protected = set(fixed)
    for uid, origin_set, dest_set in ((left, set_a, set_b), (right, set_b, set_a)):
        if uid in protected and origin_set == "test" and dest_set != "test":
            return result("Official HSE test units stay in Testing Data.")
    projected = dict(counts)
    empty = next((name for name in SPLIT_NAMES if projected[name] == 0), None)
    if empty is not None:
        return result(
            f"{SPLIT_LABELS[empty]} would have no units. Keep at least one unit in each set.",
            projected,
        )
    return result(None, projected, {left: set_a, right: set_b}, {left: set_b, right: set_a})


def _publish_manual_snapshot(
    store: ProjectStore,
    project_id: str,
    parent: Mapping[str, Any],
    split: dict[str, Any],
    report: Mapping[str, Any],
    *,
    expected_snapshot_id: str,
) -> dict[str, Any]:
    """Stage one manual snapshot and activate it under the registry lock.

    The destination path is resolved before ``launch_lock``. That lock is not
    re-entrant: the critical section must not call ``store.get``,
    ``load_snapshot``, ``project_path``, or ``snapshot_path``.
    """
    from pdm.worker import heavy_job_active

    old_fingerprint = parent["fingerprint"]
    copied = ("features.parquet", "units.parquet", "feature_schema.json")

    def write_data(staging: Path) -> None:
        for filename in copied:
            _copy_snapshot_file(parent["dir"] / filename, staging / filename)

    base = store.project_path(project_id) / "snapshots"
    staging, snapshot_id, payload = _stage_snapshot(
        base, project_id, write_data=write_data, split=split, report=report,
        fingerprint={"source_digest": old_fingerprint.get("source_digest"),
                     "source_manifest_id": old_fingerprint.get("source_manifest_id"),
                     "parent_snapshot_id": expected_snapshot_id},
    )
    try:
        new_hashes = payload["fingerprint"]["file_hashes"]
        if any(new_hashes[name] != old_fingerprint["file_hashes"][name] for name in copied):
            raise ValueError("Snapshot copy does not match its parent")
        destination = store.snapshot_path(project_id, snapshot_id)
        with store.launch_lock():
            if heavy_job_active():
                raise RuntimeError(JOB_ACTIVE_MOVE_ERROR)
            registry = store._load()
            record = store._entry(registry, project_id)
            if record.get("storage_mode") != "owned":
                raise ValueError(LINKED_LEGACY_MOVE_ERROR)
            if record.get("active_snapshot_id") != expected_snapshot_id:
                raise ValueError(STALE_SNAPSHOT_ERROR)
            limits = read_zone_limits(parent["dir"])
            if limits is not None:
                atomic_write_json(staging / ZONE_LIMITS_FILE, limits)
            _publish_snapshot(staging, destination,
                              lambda: store._activate_snapshot_locked(registry, project_id, snapshot_id))
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"project_id": project_id, "snapshot_id": snapshot_id,
            "parent_snapshot_id": expected_snapshot_id, "dir": destination, "split": split,
            "report": payload["report"], "schema": parent["schema"],
            "fingerprint": payload["fingerprint"]}


def move_units(
    project_id: str, unit_ids, destination: Literal["train", "validation", "test"], *,
    expected_snapshot_id: str, store: ProjectStore | None = None,
) -> dict[str, Any]:
    """Move whole physical units by publishing a new immutable snapshot.

    The parent snapshot and its runs are left untouched. Staging happens
    outside the registry lock; the lock covers only the job re-check, the
    stale-snapshot compare, the rename and the registry write.
    """
    from pdm.worker import heavy_job_active

    store = store or project_store()
    _safe_id(expected_snapshot_id, "snapshot ID")
    project = store.get(project_id)
    if project["storage_mode"] != "owned":
        raise ValueError(LINKED_LEGACY_MOVE_ERROR)
    if project["active_snapshot_id"] != expected_snapshot_id:
        raise ValueError(STALE_SNAPSHOT_ERROR)
    if heavy_job_active():
        raise RuntimeError(JOB_ACTIVE_MOVE_ERROR)
    parent = load_snapshot(project_id, expected_snapshot_id, store=store)
    problem = preview_move(parent, unit_ids, destination)["problem"]
    if problem:
        raise ValueError(problem)

    moved = sorted(set(map(str, unit_ids)))
    old_split = parent["split"]
    origin = {uid: name for name in SPLIT_NAMES for uid in map(str, old_split[name]) if uid in moved}
    split = dict(old_split)
    for name in SPLIT_NAMES:
        kept = [uid for uid in map(str, old_split[name]) if uid not in origin]
        split[name] = sorted(kept + moved) if name == destination else sorted(kept)
    now = datetime.now(timezone.utc).isoformat()
    split.update(
        protocol=MANUAL_SPLIT_PROTOCOL,
        parent_snapshot_id=expected_snapshot_id,
        realized_counts={name: len(split[name]) for name in SPLIT_NAMES},
        manual_moves=[*old_split.get("manual_moves", []),
                      {"unit_ids": moved, "from": {uid: origin[uid] for uid in moved},
                       "to": destination, "at": now}],
    )
    assert_split_coverage(parent["units"].copy(), split)
    report = {
        **parent["report"], "snapshot_id": None, "parent_snapshot_id": expected_snapshot_id,
        "split_counts": split["realized_counts"],
        "by_split": _by_split(parent["features"], parent["units"], split),
        "created_at": now,
    }
    return _publish_manual_snapshot(
        store, project_id, parent, split, report, expected_snapshot_id=expected_snapshot_id,
    )


def swap_units(
    project_id: str, unit_a, unit_b, *,
    expected_snapshot_id: str, store: ProjectStore | None = None,
) -> dict[str, Any]:
    """Exchange two whole units in one new snapshot.

    Set counts stay equal to the parent, so two one-unit sets can trade places.
    The parent snapshot and its runs are left untouched.
    """
    from pdm.worker import heavy_job_active

    store = store or project_store()
    _safe_id(expected_snapshot_id, "snapshot ID")
    project = store.get(project_id)
    if project["storage_mode"] != "owned":
        raise ValueError(LINKED_LEGACY_MOVE_ERROR)
    if project["active_snapshot_id"] != expected_snapshot_id:
        raise ValueError(STALE_SNAPSHOT_ERROR)
    if heavy_job_active():
        raise RuntimeError(JOB_ACTIVE_MOVE_ERROR)
    parent = load_snapshot(project_id, expected_snapshot_id, store=store)
    problem = preview_swap(parent, unit_a, unit_b)["problem"]
    if problem:
        raise ValueError(problem)

    left, right = str(unit_a), str(unit_b)
    old_split = parent["split"]
    current = {uid: name for name in SPLIT_NAMES for uid in map(str, old_split[name])}
    set_a, set_b = current[left], current[right]
    split = dict(old_split)
    exchanged = {left, right}
    destinations = {left: set_b, right: set_a}
    for name in SPLIT_NAMES:
        kept = [uid for uid in map(str, old_split[name]) if uid not in exchanged]
        arrived = [uid for uid, dest in destinations.items() if dest == name]
        split[name] = sorted(kept + arrived)
    now = datetime.now(timezone.utc).isoformat()
    split.update(
        protocol=MANUAL_SPLIT_PROTOCOL,
        parent_snapshot_id=expected_snapshot_id,
        realized_counts={name: len(split[name]) for name in SPLIT_NAMES},
        manual_moves=[
            *old_split.get("manual_moves", []),
            {
                "kind": "swap",
                "unit_ids": sorted((left, right)),
                "from": {left: set_a, right: set_b},
                "to": {left: set_b, right: set_a},
                "at": now,
            },
        ],
    )
    assert_split_coverage(parent["units"].copy(), split)
    report = {
        **parent["report"], "snapshot_id": None, "parent_snapshot_id": expected_snapshot_id,
        "split_counts": split["realized_counts"],
        "by_split": _by_split(parent["features"], parent["units"], split),
        "created_at": now,
    }
    return _publish_manual_snapshot(
        store, project_id, parent, split, report, expected_snapshot_id=expected_snapshot_id,
    )


JOB_ACTIVE_LIMITS_ERROR = ("A background job is running, so this edit is not saved. "
                           "Press Save after the job finishes.")


def read_zone_limits(directory: Path) -> dict[str, Any] | None:
    """Display limits saved beside a snapshot; outside SNAPSHOT_FILES and its fingerprint.

    A sidecar that reads but fails validation is removed; a read error leaves it in place.
    """
    path = Path(directory) / ZONE_LIMITS_FILE
    try:
        if path.is_symlink() or not path.is_file():
            return None
        content = read_json(path)
    except OSError:
        return None
    except ValueError:
        pass
    else:
        try:
            return validate_absolute_thresholds(content)
        except (ValueError, TypeError):
            pass
    try:
        if not path.is_symlink():
            path.unlink(missing_ok=True)
    except OSError:
        pass
    return None


def load_zone_limits(
    project_id: str, snapshot_id: str, *, store: ProjectStore | None = None
) -> dict[str, Any] | None:
    store = store or project_store()
    try:
        directory = store.snapshot_path(project_id, snapshot_id)
    except (OSError, ValueError, KeyError):
        return None
    return read_zone_limits(directory)


def save_zone_limits(
    project_id: str, thresholds: Mapping[str, Any], *, expected_snapshot_id: str,
    store: ProjectStore | None = None,
) -> dict[str, Any]:
    """Save display limits for the active snapshot (same snapshot ID, registry untouched).

    Training, runs and ``load_snapshot`` keep the import-time schema.
    """
    from pdm.worker import heavy_job_active

    store = store or project_store()
    rule = validate_absolute_thresholds(thresholds)
    project = store.get(project_id)
    if project["active_snapshot_id"] != expected_snapshot_id:
        raise ValueError(STALE_SNAPSHOT_ERROR)
    directory = store.snapshot_path(project_id, expected_snapshot_id)
    if not directory.is_dir():
        raise ValueError("Snapshot is missing")
    if heavy_job_active():
        raise RuntimeError(JOB_ACTIVE_LIMITS_ERROR)
    with store.launch_lock():
        if heavy_job_active():
            raise RuntimeError(JOB_ACTIVE_LIMITS_ERROR)
        record = store._entry(store._load(), project_id)
        if record.get("active_snapshot_id") != expected_snapshot_id:
            raise ValueError(STALE_SNAPSHOT_ERROR)
        if read_zone_limits(directory) != rule:
            atomic_write_json(directory / ZONE_LIMITS_FILE, rule)
    return rule


def load_snapshot(
    project_id: str, snapshot_id: str | None = None, *, store: ProjectStore | None = None
) -> dict[str, Any]:
    store = store or project_store()
    project = store.get(project_id)
    sid = snapshot_id or project["active_snapshot_id"]
    if not sid:
        raise ValueError("Project has no prepared snapshot")
    directory = store.snapshot_path(project_id, sid)
    fingerprint = read_json(directory / "processed_fingerprint.json")
    if fingerprint.get("project_id") != project_id or fingerprint.get("snapshot_id") != sid:
        raise ValueError("Snapshot binding does not match project")
    for filename, expected in fingerprint.get("file_hashes", {}).items():
        artifact = directory / filename
        if filename not in SNAPSHOT_FILES or artifact.is_symlink() or sha256_file(artifact) != expected:
            raise ValueError(f"Snapshot artifact hash mismatch: {filename}")
    if set(fingerprint.get("file_hashes", {})) != set(SNAPSHOT_FILES):
        raise ValueError("Snapshot is incomplete")
    features = pd.read_parquet(directory / "features.parquet")
    units = pd.read_parquet(directory / "units.parquet")
    split = read_json(directory / "split.json")
    assert_split_coverage(units.copy(), split)
    schema = read_json(directory / "feature_schema.json")
    report = read_json(directory / "data_report.json")
    if not {"unit_id", "timestamp_s", "signal", "gap_before"}.issubset(features.columns):
        raise ValueError("Snapshot lacks canonical signal columns")
    return {"project_id": project_id, "snapshot_id": sid, "dir": directory,
            "features": features, "units": units, "split": split, "report": report,
            "schema": schema, "fingerprint": fingerprint}
