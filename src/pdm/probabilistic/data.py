"""Verified, immutable sensor snapshots with explicit independent split access."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.io_util import atomic_write_json, sha256_file

from .contract import SPLITS, canonical_hash, config_hash, default_config


def _canonical_offsets(times: np.ndarray, config: dict) -> list:
    """Translation-invariant timing identity at the declared clock resolution."""
    finite = times[np.isfinite(times)]
    offsets = times - finite[0] if len(finite) else times
    cadence, tolerance = config["cadence_s"], config["clock_tolerance_s"]
    result = []
    for offset in offsets:
        if not np.isfinite(offset):
            result.append(None)  # Never erase an invalid source-row barrier.
            continue
        grid = int(np.rint(offset / cadence))
        if abs(offset - grid * cadence) <= tolerance:
            result.append(["cadence_grid", grid])
        elif tolerance > 0:
            result.append(["clock_ticks", int(np.rint(offset / tolerance))])
        else:
            result.append(["exact", float(offset)])
    return result


def _read_sensor(path: Path, config: dict, release_id: str, *, barriers=None, physical_map=None) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(path, dtype={"unit_id": "string"},float_precision="round_trip" if config.get("observed_profile") else None)
    if list(frame.columns) != config["schema"]:
        raise ValueError(f"Sensor schema mismatch: {path}")
    if frame.empty or frame["unit_id"].isna().any() or (frame["unit_id"].str.strip() == "").any():
        raise ValueError(f"Empty sensor file or missing physical unit ID: {path}")
    frame["unit_id"] = frame["unit_id"].astype(str)
    for column in ("timestamp_s", config["target"]):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite_time = np.isfinite(frame["timestamp_s"].to_numpy(dtype=float))
    finite_signal = np.isfinite(frame[config["target"]].to_numpy(dtype=float))
    frame["valid"] = finite_time & finite_signal & (frame[config["target"]] > 0)
    frame["physical_unit_id"] = frame["unit_id"].map(physical_map) if physical_map else release_id + ":" + frame["unit_id"]
    if frame["physical_unit_id"].isna().any():
        raise ValueError("Observed physical identity is missing")
    frame["segment_id"] = -1
    gaps = 0
    eligible, eligible_reference, units_without = 0, 0, []
    trajectory_hashes = {}
    for uid, group in frame.groupby("unit_id", sort=True):
        times = group["timestamp_s"].to_numpy(dtype=float)
        values = group[config["target"]].to_numpy(dtype=float)
        valid = group["valid"].to_numpy(dtype=bool)
        finite = times[np.isfinite(times)]
        if len(np.unique(finite)) != len(finite):
            raise ValueError(f"Duplicate physical unit timestamp: {uid} in {path}")
        if (np.diff(finite) <= 0).any():
            raise ValueError(f"Out-of-order physical unit timestamps: {uid} in {path}")
        segments = np.full(len(group), -1, dtype=int)
        forced = set((barriers or {}).get(uid, []))
        segment, previous_valid = -1, False
        for index in range(len(group)):
            continuous = (
                previous_valid and valid[index]
                and index not in forced
                and abs(times[index] - times[index - 1] - config["cadence_s"])
                <= config["clock_tolerance_s"]
            )
            if valid[index]:
                if not continuous:
                    segment += 1
                    if index > 0:
                        gaps += 1
                segments[index] = segment
            previous_valid = bool(valid[index])
        frame.loc[group.index, "segment_id"] = segments
        lengths = [int((segments == s).sum()) for s in range(segment + 1)]
        usable = sum(max(0, n - config["history_length"] + 1) for n in lengths)
        common = sum(max(0, n - 60 + 1) for n in lengths)
        reference = sum(max(0, n - 119) for n in lengths)
        eligible += common
        eligible_reference += reference
        if not usable:
            units_without.append(uid)
        # IDs and absolute age cannot disguise a byte-identical numeric trajectory.
        payload = {
            "timestamp_offsets": _canonical_offsets(times, config),
            "signal": [float(y) if np.isfinite(y) else None for y in values],
        }
        trajectory_hashes[uid] = canonical_hash(payload)
    report = {
        "rows": len(frame), "accepted_rows": int(frame["valid"].sum()),
        "rejected_rows": int((~frame["valid"]).sum()), "units": int(frame["unit_id"].nunique()),
        "gaps": gaps, "units_without_usable_windows": units_without,
        "common_history_origins": eligible, "full_reference_candidates": eligible_reference,
        "trajectory_hashes": trajectory_hashes,
    }
    return frame, report


def _source_root(source_root: str | Path) -> Path:
    root = Path(source_root).expanduser().resolve()
    if (root / "data").is_dir():
        root = root / "data"
    if not root.is_dir():
        raise FileNotFoundError(root)
    return root


def _load_manifest(root: Path, manifest_path: str | Path | None, config: dict) -> dict:
    path = Path(manifest_path).expanduser().resolve() if manifest_path else root / "dataset_manifest.json"
    if not path.is_file():
        raise FileNotFoundError("Original dataset_manifest.json is required; supply manifest_path")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != config["schema"]
        or manifest.get("target") != config["target"]
        or manifest.get("cadence_s") != config["cadence_s"]
        or not manifest.get("version") or "seed" not in manifest
        or not isinstance(manifest.get("splits"), dict)
        or not isinstance(manifest.get("file_sha256"), dict)
    ):
        raise ValueError("Dataset manifest schema/target/cadence/release contract mismatch")
    # This initial sensor profile has a known physical unit; never infer it from values.
    physical_unit = str(manifest.get("physical_unit", ""))
    if physical_unit != config["unit"] and not physical_unit.startswith(config["unit"] + " ("):
        raise ValueError("Dataset manifest physical unit mismatch")
    return manifest


def _prepare(source_root, suite, output_dir, manifest_path, config, external):
    config = default_config(**(config or {}))
    root = _source_root(source_root)
    manifest = _load_manifest(root, manifest_path, config)
    if suite not in manifest["splits"]:
        raise ValueError("Suite is absent from the original dataset manifest")
    declared_splits = manifest["splits"][suite]
    required = ("test",) if external else SPLITS
    if not external and set(declared_splits) != set(SPLITS):
        raise ValueError("Training requires four explicit independent sources; use external evaluation")
    release_id = str(manifest.get("release_id") or f"{manifest['version']}:seed={manifest['seed']}")
    expected_files, reports, physical_ids, trajectory_ids = {}, {}, {}, {}
    # Structural source audit may inspect all supplied sensor sources, never latent metadata.
    for source_suite, source_splits in manifest["splits"].items():
        for split, count in source_splits.items():
            if split not in SPLITS:
                raise ValueError(f"Unsupported declared split: {split}")
            relative = f"{source_suite}/sensor_csv/{split}/measurements.csv"
            path = root / relative
            if not path.is_file():
                if source_suite == suite and split in required:
                    raise FileNotFoundError(path)
                continue
            expected = manifest["file_sha256"].get(relative)
            if not expected or sha256_file(path) != expected:
                raise ValueError(f"Original sensor SHA256 mismatch: {relative}")
            frame, report = _read_sensor(path, config, release_id,barriers=manifest.get("observed_barriers",{}).get(split),physical_map=manifest.get("physical_unit_map"))
            if report["units"] != int(count):
                raise ValueError(f"Manifest unit count mismatch: {relative}")
            reports[relative] = report
            expected_files[relative] = expected
            for uid, digest in report["trajectory_hashes"].items():
                physical = manifest.get("physical_unit_map",{}).get(uid,release_id + ":" + uid)
                if physical in physical_ids and physical_ids[physical] != {"suite": source_suite, "split": split}:
                    raise ValueError(f"Physical unit appears in different sources/splits: {uid}")
                if digest in trajectory_ids:
                    raise ValueError(f"Renamed duplicate trajectory: {uid} and {trajectory_ids[digest]}")
                physical_ids[physical] = {"suite": source_suite, "split": split}
                trajectory_ids[digest] = uid
    if set(expected_files) == {
        f"{s}/sensor_csv/{p}/measurements.csv" for s, splits in manifest["splits"].items()
        for p in splits
    }:
        if manifest.get("total_units") is not None and len(physical_ids) != manifest["total_units"]:
            raise ValueError("Dataset total units mismatch")
        if manifest.get("total_rows") is not None and sum(r["rows"] for r in reports.values()) != manifest["total_rows"]:
            raise ValueError("Dataset total rows mismatch")
    selected = {
        split: f"{suite}/sensor_csv/{split}/measurements.csv" for split in required
    }
    identity = {
        "release_id": release_id, "suite": suite,
        "files": {split: expected_files[path] for split, path in selected.items()},
        "profile": {key: config[key] for key in (
            "target", "unit", "schema", "cadence_s", "clock_tolerance_s", "positive_domain"
        )}, "external": external,
    }
    assignment = manifest.get("assignment_provenance")
    if assignment is not None:
        if not isinstance(assignment, dict):
            raise ValueError("Development assignment provenance must be a mapping")
        claimed = assignment.get("assignment_hash")
        if claimed != canonical_hash({k: v for k, v in assignment.items() if k != "assignment_hash"}):
            raise ValueError("Development assignment provenance integrity mismatch")
        # Original reference identities retain their exact historical payload.
        # Development identity includes the request and parent even when the
        # realized memberships happen to be identical.
        identity["assignment_hash"] = claimed
    context = manifest.get("descriptive_context",{})
    if not isinstance(context,dict):
        raise ValueError("Descriptive context descriptors must be a recorded-schema mapping")
    if context:
        if set(context) != set(required):
            raise ValueError("Descriptive context must be bound to the explicit snapshot roles")
        for part,item in context.items():
            _validate_context_descriptor(manifest,config,part,item)
            if sha256_file(root/item["path"]) != item.get("sha256"):
                raise ValueError("Descriptive context integrity mismatch")
            rows = json.loads((root/item["path"]).read_text())
            _validate_context_rows(rows,item,set(reports[f"{suite}/sensor_csv/{part}/measurements.csv"]["trajectory_hashes"]))
        identity["descriptive_context_sha256"] = {part:item["sha256"] for part,item in context.items()}
    dataset_hash = canonical_hash(identity)
    directory = Path(output_dir).expanduser().resolve()
    if (directory / "snapshot.json").exists():
        existing = load_snapshot(directory)
        if existing["dataset_hash"] != dataset_hash or existing["config_hash"] != config_hash(config):
            raise FileExistsError("Immutable snapshot already contains different sources/config")
        return existing
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Snapshot output directory must be empty; existing files are preserved")
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".sensor-snapshot-", dir=directory.parent))
    try:
        split_manifest, admission, split_files = {}, {}, {}
        for split, relative in selected.items():
            target = staging / "sensor_csv" / split / "measurements.csv"
            target.parent.mkdir(parents=True)
            shutil.copyfile(root / relative, target)
            if sha256_file(target) != expected_files[relative]:
                raise ValueError("Source changed during immutable snapshot copy")
            report = reports[relative]
            split_manifest[split] = {
                "path": f"sensor_csv/{split}/measurements.csv", "sha256": expected_files[relative],
                "units": sorted(report["trajectory_hashes"]),
                "physical_units": sorted({manifest.get("physical_unit_map",{}).get(u,release_id + ":" + u) for u in report["trajectory_hashes"]}),
                "unit_count": report["units"], "rows": report["rows"],
            }
            admission[split] = {k: v for k, v in report.items() if k != "trajectory_hashes"}
            split_files[split] = str(directory / split_manifest[split]["path"])
        for item in context.values():
            path = staging/item["path"]
            path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(root/item["path"],path)
            if sha256_file(path) != item["sha256"]:
                raise ValueError("Descriptive source context changed during snapshot copy")
        status = manifest.get("evaluation_status", "reference_exposed")
        if manifest["version"] == "pdm-synthetic-corridor-1.0.0" and manifest["seed"] == 20261005:
            status = "reference_exposed"
        snapshot = {
            "snapshot_id": "sensor-" + dataset_hash[:24], "dataset_hash": dataset_hash,
            "release_id": release_id, "suite": suite, "profile": manifest.get("profile","synthetic_single_channel"),
            "split_files": split_files, "split_manifest": split_manifest,
            "dataset_manifest": manifest, "admission": admission, "directory": str(directory),
            "evaluation_status": status, "external": external, "config": config,
            "config_hash": config_hash(config), "structural_audit": {
                "files_verified": len(expected_files), "units": len(physical_ids),
                "rows": sum(r["rows"] for r in reports.values()),
                "evaluation_only_accessed": False,
            },
        }
        if manifest.get("profile") == "observed_single_channel":
            snapshot["task"] = config["task"]
        snapshot["snapshot_hash"] = canonical_hash(snapshot)
        atomic_write_json(staging / "snapshot.json", snapshot)
        atomic_write_json(staging / "dataset_manifest.json", manifest)
        atomic_write_json(staging / "split_manifest.json", split_manifest)
        atomic_write_json(staging / "admission_report.json", admission)
        # A read-only byte copy plus load-time integrity checks, never a rewritten CSV.
        for path in staging.rglob("*"):
            if path.is_file():
                path.chmod(0o444)
        if directory.exists():
            directory.rmdir()  # Only the already checked empty directory.
        staging.rename(directory)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return load_snapshot(directory)


def prepare_dataset(source_root, suite, output_dir, *, manifest_path=None, config=None) -> dict:
    return _prepare(source_root, suite, output_dir, manifest_path, config, False)


def prepare_external(source_root, suite, output_dir, *, manifest_path=None, config=None) -> dict:
    """Create a separate Test-only source without replacing the training snapshot."""
    return _prepare(source_root, suite, output_dir, manifest_path, config, True)


def _validate_context_descriptor(manifest,config,part,item):
    from pdm.red_entry_context import CONTEXT_FIELDS
    from pdm.red_entry_targets import ENDPOINT_FIELDS
    allowed = {"unit_id","timestamp_s","signal_admitted","episode_start_timestamp_s","episode_end_timestamp_s",*CONTEXT_FIELDS,*ENDPOINT_FIELDS}
    if (manifest.get("profile") != "observed_single_channel" or config.get("observed_profile") is not True
            or not isinstance(manifest.get("assignment_provenance"),dict)):
        raise ValueError("Descriptive context requires a verified observed profile; original reference sidecars are unsupported")
    if not isinstance(item,dict):
        raise ValueError("Descriptive context descriptor must declare its recorded schema")
    columns = item.get("columns")
    if (part not in SPLITS or item.get("path") != f"descriptive_context/{part}.json"
            or not isinstance(columns,list) or any(not isinstance(name,str) for name in columns)
            or len(set(columns)) != len(columns) or not {"unit_id","timestamp_s"}.issubset(columns)
            or not set(columns).issubset(allowed)):
        raise ValueError("Descriptive context must contain only declared recorded context/endpoint fields")


def _validate_context_rows(rows,item,unit_ids):
    columns = set(item["columns"])
    if not isinstance(rows,list) or any(not isinstance(row,dict) or set(row) != columns for row in rows):
        raise ValueError("Descriptive context content differs from its declared recorded schema")
    if any(not isinstance(row["unit_id"],str) or row["unit_id"] not in unit_ids for row in rows):
        raise ValueError("Descriptive context unit membership mismatch")
    if any(any(value is not None and (not isinstance(value,(str,int,float,bool))
                                      or isinstance(value,float) and not np.isfinite(value))
               for value in row.values()) for row in rows):
        raise ValueError("Descriptive context values must be recorded finite scalars or unknown")


def prepare_observations(features, units, split, output_dir, *, schema, source_manifest, parent=None, should_stop=None, endpoint_records=()):
    """Publish decoded observations without changing values, cadence or physical IDs."""
    cadence = float(schema["cadence_s"])
    target, unit = schema["signal_column"], schema["signal_unit"]
    values = features["signal"].to_numpy(float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Calibrated forecasting requires finite strictly positive observations; signed data is unsupported")
    physical_map = units.set_index("unit_id")["physical_unit_id"].astype(str).to_dict()
    from pdm.red_entry_targets import ENDPOINT_FIELDS
    endpoint_fields = {"unit_id","physical_unit_id","component_cycle_id","timestamp_s","episode_start_timestamp_s","episode_end_timestamp_s",*ENDPOINT_FIELDS}
    for record in endpoint_records:
        if (not isinstance(record,dict) or not set(record).issubset(endpoint_fields)
                or record.get("unit_id") not in physical_map
                or record.get("physical_unit_id") != physical_map[record["unit_id"]]
                or not np.isfinite(float(record.get("timestamp_s",np.nan)))):
            raise ValueError("Rejected endpoint evidence must retain a declared physical unit and finite observed clock")
    if any(not split.get(part) for part in SPLITS):
        raise ValueError("Import four nonempty disjoint roles before calibrated training")
    from pdm.red_entry_context import assert_physical_split
    assert_physical_split(units, split)
    for uid, rows in features.groupby("unit_id", sort=False):
        times = rows.timestamp_s.to_numpy(float)
        delta = np.diff(times)
        if not np.isfinite(times).all() or (delta <= 0).any() or not np.allclose(delta / cadence, np.rint(delta / cadence), atol=1e-6 / cadence, rtol=0):
            raise ValueError(f"{uid}: observations do not follow the declared cadence; no resampling is performed")
    train_values = features.loc[features.unit_id.isin(split["train"]), "signal"].to_numpy(float)
    floor = float(schema.get("scale_floor") or .02 * np.median(train_values))
    rule = schema.get("thresholds") or {}
    overrides = dict(target=target,unit=unit,cadence_s=cadence,scale_floor=floor,observed_profile=True,
                     signal_label=schema["signal_label"],scale_floor_policy="declared" if schema.get("scale_floor") else "0.02 * Training median",
                     imported_rule=rule)
    if rule.get("mode") == "absolute":
        overrides.update(yellow=rule["yellow"],red=rule["red"],threshold_direction=rule.get("direction","above"))
    config = default_config(**overrides)
    assignment = {"policy":"observed-four-role-whole-physical-unit-v1","source_digest":source_manifest["source_digest"],
                  "source_files":source_manifest["files"],"request":source_manifest["split_request"],
                  "mapping":{part:sorted(map(str,split[part])) for part in SPLITS},"physical_unit_map":physical_map,
                  "parent":parent,"signal_schema":schema,"realized_counts":split["realized_counts"],
                  "unit_source_groups":units.set_index("unit_id").get("source_group",pd.Series(dtype=str)).to_dict(),
                  "protected_test_units":units.loc[units.get("source_group",pd.Series(index=units.index,dtype=str)).eq("author_test"),"unit_id"].astype(str).tolist()}
    exposure = source_manifest.get("verified_exposure")
    if exposure:
        if (target,unit,cadence) != (exposure["target"],exposure["unit"],exposure["cadence_s"]):
            raise ValueError("Declared physical profile differs from the verified exposed release")
        physical_map = {uid:exposure["release_id"]+":"+uid for uid in physical_map}
        assignment.update(physical_unit_map=physical_map,verified_exposure=exposure)
    with tempfile.TemporaryDirectory(prefix="pdm-observations-") as temp:
        root = Path(temp)
        barriers, hashes, context = {}, {}, {}
        from pdm.red_entry_context import CONTEXT_FIELDS
        context_columns = [name for name in features.columns if name in {"unit_id","timestamp_s",*CONTEXT_FIELDS,*ENDPOINT_FIELDS}]
        if endpoint_records:
            context_columns = list(dict.fromkeys([*context_columns,"signal_admitted",*[name for record in endpoint_records for name in record]]))
        for part in SPLITS:
            if should_stop and should_stop():
                raise InterruptedError("Project import cancelled")
            rows = features.loc[features.unit_id.isin(split[part])].copy()
            barriers[part] = {str(uid):np.flatnonzero(group.gap_before.to_numpy(bool)).tolist() for uid,group in rows.groupby("unit_id",sort=False)}
            relative = f"observed/sensor_csv/{part}/measurements.csv"
            path = root / relative
            path.parent.mkdir(parents=True)
            rows[["unit_id","timestamp_s","signal"]].rename(columns={"signal":target}).to_csv(path,index=False,float_format="%.17g")
            hashes[relative] = sha256_file(path)
            if set(context_columns)-{"unit_id","timestamp_s"}:
                context_path = root/f"descriptive_context/{part}.json"
                context_path.parent.mkdir(parents=True,exist_ok=True)
                descriptive = rows.reindex(columns=context_columns).copy()
                if endpoint_records:
                    descriptive["signal_admitted"] = True
                    rejected = [{**record,"signal_admitted":False} for record in endpoint_records if record["unit_id"] in split[part]]
                    if rejected:
                        descriptive = pd.concat([descriptive,pd.DataFrame(rejected).reindex(columns=context_columns)],ignore_index=True).sort_values(["unit_id","timestamp_s"],kind="stable")
                descriptive = descriptive.astype(object)
                atomic_write_json(context_path,descriptive.where(pd.notna(descriptive),None).to_dict(orient="records"))
                context[part] = {"path":str(context_path.relative_to(root)),"sha256":sha256_file(context_path),
                                 "columns":context_columns,"usage":"recorded descriptive context only; excluded from model inputs"}
        assignment["gap_barriers"] = barriers
        if context:
            assignment["descriptive_context"] = context
        assignment["assignment_hash"] = canonical_hash(assignment)
        manifest = {"version":"observed-single-channel-v1","profile":"observed_single_channel","seed":split["seed"],
                    "release_id":exposure["release_id"] if exposure else "user-observed:"+source_manifest["source_digest"],"schema":config["schema"],"target":target,
                    "physical_unit":unit,"cadence_s":cadence,"splits":{"observed":{part:len(split[part]) for part in SPLITS}},
                    "file_sha256":hashes,"physical_unit_map":physical_map,"observed_barriers":barriers,
                    "assignment_provenance":assignment,"evaluation_status":"development_reference_exposed" if exposure else "development_user_supplied"}
        if context:
            manifest["descriptive_context"] = context
        atomic_write_json(root / "dataset_manifest.json", manifest)
        return _prepare(root,"observed",output_dir,None,config,False)


def load_snapshot(directory: str | Path) -> dict:
    directory = Path(directory).expanduser().resolve()
    snapshot = json.loads((directory / "snapshot.json").read_text())
    digest = snapshot.pop("snapshot_hash", None)
    if not digest or canonical_hash(snapshot) != digest:
        raise ValueError("Immutable snapshot manifest integrity mismatch")
    snapshot["snapshot_hash"] = digest
    if snapshot.get("config_hash") != config_hash(snapshot["config"]):
        raise ValueError("Immutable snapshot configuration binding mismatch")
    if str(directory) != snapshot["directory"]:
        raise ValueError("Snapshot directory binding mismatch")
    for name, content in (
        ("dataset_manifest.json", snapshot["dataset_manifest"]),
        ("split_manifest.json", snapshot["split_manifest"]),
        ("admission_report.json", snapshot["admission"]),
    ):
        if json.loads((directory / name).read_text()) != content:
            raise ValueError(f"Immutable companion manifest mismatch: {name}")
    context = snapshot["dataset_manifest"].get("descriptive_context",{})
    if not isinstance(context,dict):
        raise ValueError("Descriptive context descriptors must be a recorded-schema mapping")
    for part,item in context.items():
        _validate_context_descriptor(snapshot["dataset_manifest"],snapshot["config"],part,item)
        if part not in snapshot["split_manifest"]:
            raise ValueError("Descriptive context role/path binding mismatch")
        path = directory/item["path"]
        if path.is_symlink() or path.parent.is_symlink() or sha256_file(path) != item["sha256"]:
            raise ValueError("Immutable descriptive context SHA256 mismatch")
    for split, record in snapshot["split_manifest"].items():
        path = directory / record["path"]
        if path.resolve().parent.parent.parent != directory or snapshot["split_files"][split] != str(path):
            raise ValueError("Snapshot sensor path escapes immutable directory")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Immutable snapshot sensor SHA256 mismatch: {split}")
    return snapshot


def load_split(snapshot: dict, split: str) -> pd.DataFrame:
    """Explicit split access; no automatic fallback or concatenation."""
    # Public callers may hold mutable dicts: trust only the verified on-disk snapshot.
    authoritative = load_snapshot(snapshot["directory"])
    if snapshot != authoritative:
        raise ValueError("Immutable snapshot provenance was changed after preparation")
    snapshot = authoritative
    if split not in SPLITS or split not in snapshot["split_files"]:
        raise ValueError(f"Split is not available in this snapshot: {split}")
    if snapshot.get("config_hash") != config_hash(snapshot["config"]):
        raise ValueError("Snapshot configuration was changed after preparation")
    path = Path(snapshot["split_files"][split])
    if sha256_file(path) != snapshot["split_manifest"][split]["sha256"]:
        raise ValueError(f"Immutable split hash mismatch: {split}")
    manifest = snapshot["dataset_manifest"]
    frame, _ = _read_sensor(path, snapshot["config"], snapshot["release_id"],barriers=manifest.get("observed_barriers",{}).get(split),physical_map=manifest.get("physical_unit_map"))
    frame.attrs.update({key: snapshot[key] for key in (
        "snapshot_id", "dataset_hash", "release_id", "suite", "profile", "evaluation_status",
    )})
    frame.attrs.update(split=split, config=snapshot["config"], external=snapshot["external"])
    return frame


def load_training_splits(snapshot: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The training worker's only numeric input path: Train and Validation."""
    if snapshot.get("external"):
        raise ValueError("An external evaluator source cannot be used for training")
    return load_split(snapshot, "train"), load_split(snapshot, "validation")
