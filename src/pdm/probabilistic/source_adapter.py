"""Verified sensor acquisition and configurable development split assignments.

The original prepare_dataset entry point and its reference manifests are intact.
Only selected, manifest-verified observed sensor sources enter a development pool.
"""
from __future__ import annotations

import csv
import math
import shutil
import tempfile
from pathlib import Path

from pdm.data.project_import import _iter_source_files
from pdm.data.unit_allocation import allocate_whole_units
from pdm.io_util import atomic_write_json, sha256_file
from pdm.probabilistic.contract import SPLITS, canonical_hash, config_hash, default_config
from pdm.probabilistic.data import _load_manifest, _prepare, _read_sensor, load_snapshot

ASSIGNMENT_POLICY = "whole_physical_unit_development_v1:sorted:largest_remainder:min_one:seeded_permutation"
DEVELOPMENT_STATUS = "development_reference_exposed"


def validate_source_plan(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("primary"), dict):
        raise ValueError("Primary Training source is required")
    modes = {part:plan.get(f"{part}_mode", "auto") for part in SPLITS[1:]}
    if any(mode not in {"auto","folder"} for mode in modes.values()):
        raise ValueError("Holdout mode must be auto or folder")
    seed = plan.get("seed",42)
    if isinstance(seed,bool) or not isinstance(seed,int) or not 0 <= seed <= 2**32-1:
        raise ValueError("Split seed must be a nonnegative 32-bit integer")
    weights = plan.get("weights") or dict(zip(SPLITS,(.55,.15,.15,.15),strict=True))
    if not isinstance(weights,dict) or set(weights) != set(SPLITS):
        raise ValueError("Split weights must contain all four roles")
    try:
        weights = {part:float(weights[part]) for part in SPLITS}
    except (ValueError,TypeError) as exc:
        raise ValueError("Split weights must be numeric") from exc
    if not all(math.isfinite(w) and w > 0 for w in weights.values()) or not math.isclose(sum(weights.values()),1,abs_tol=1e-8):
        raise ValueError("Positive split weights must sum to 1")
    for part,mode in modes.items():
        if mode == "folder" and not isinstance(plan.get(part),dict):
            raise ValueError(f"Separate {part} folder source is required")
        if mode == "auto" and plan.get(part) is not None:
            raise ValueError(f"Unexpected manual {part} source in automatic mode")
    return {"modes":modes,"weights":weights,"seed":seed}


def _stop(should_stop):
    if should_stop and should_stop():
        raise InterruptedError("Project import cancelled")


def _manifest_location(plan):
    if plan.get("manifest_path"):
        return Path(plan["manifest_path"]).expanduser().resolve()
    primary = plan["primary"]
    if primary.get("mode") == "folder":
        root = Path(primary["path"]).expanduser().resolve()
        for directory in (root,*list(root.parents)[:5]):
            for candidate in (directory/"dataset_manifest.json",directory/"data"/"dataset_manifest.json"):
                if candidate.is_file():
                    return candidate
    else:
        manifests = [Path(rec["path"]) for rec in primary.get("files",[]) if Path(rec.get("relative_path","")).name == "dataset_manifest.json"]
        if len(manifests) == 1:
            return manifests[0]
    raise ValueError("The original dataset manifest is required to verify the selected sensor sources")


def prepare_source_plan(plan, suite, output_dir, *, config=None, parent=None, existing_directories=(), should_stop=None, receipt_path=None):
    """Create/reuse one immutable snapshot from the common Import source plan.

    Auto allocation uses exactly the selected primary pool. Manual source overlap
    is rejected. All admitted files must match this suite's declared sensor SHA.
    """
    request = validate_source_plan(plan)
    cfg = default_config(**(config or {}))
    manifest_path = _manifest_location(plan)
    manifest_digest = sha256_file(manifest_path)
    manifest = _load_manifest(manifest_path.parent,manifest_path,cfg)
    if suite not in manifest["splits"] or set(manifest["splits"][suite]) != set(SPLITS):
        raise ValueError("This project's suite needs four declared sensor sources")
    release = str(manifest.get("release_id") or f"{manifest['version']}:seed={manifest['seed']}")
    declared = {part:f"{suite}/sensor_csv/{part}/measurements.csv" for part in SPLITS}
    expected = {part:manifest["file_sha256"].get(relative) for part,relative in declared.items()}
    if not all(expected.values()):
        raise ValueError("The suite's original sensor checksums are missing")
    by_hash = {digest:part for part,digest in expected.items()}
    if len(by_hash) != 4:
        raise ValueError("Duplicate sensor sources across original splits")
    sources = {"primary":plan["primary"],**{part:plan[part] for part in SPLITS[1:] if request["modes"][part] == "folder"}}
    records, assignments, identity, trajectories = [],[],{},{}
    raw_rows = {}
    for group,spec in sources.items():
        admitted = 0
        for path,relative in _iter_source_files(spec):
            _stop(should_stop)
            if path.name != "measurements.csv":
                continue  # Manifest and evaluator-only/other CSV material is never a feature.
            foreign = set(manifest["splits"]) - {suite}
            if any(name in Path(relative).parts or name in path.parts for name in foreign):
                continue
            digest = sha256_file(path)
            original_part = by_hash.get(digest)
            if original_part is None:
                raise ValueError("Sensor source checksum or suite mismatch")
            frame,report = _read_sensor(path,cfg,release)
            if report["units"] != int(manifest["splits"][suite][original_part]):
                raise ValueError("Original manifest unit count mismatch")
            record = {"group":group,"relative_path":relative,"sha256":digest,"origin_role":original_part,"origin_path":declared[original_part]}
            records.append(record)
            with path.open(newline="") as handle:
                reader = csv.DictReader(handle)
                rows = list(reader)
            if sha256_file(path) != digest:
                raise ValueError("Source changed while acquiring sensor rows")
            for unit,digest_unit in report["trajectory_hashes"].items():
                physical = release+":"+unit
                if physical in identity:
                    raise ValueError("Duplicate physical unit or overlap between manual and automatic sources")
                if digest_unit in trajectories:
                    raise ValueError("Renamed duplicate physical trajectory")
                identity[physical] = unit
                trajectories[digest_unit] = physical
                raw_rows[physical] = [row for row in rows if row["unit_id"] == unit]
                assignments.append({"physical_unit_id":physical,"unit_id":unit,"source_group":group,"original_role":original_part,"source_sha256":digest})
            admitted += len(frame)
        if not admitted:
            raise ValueError(f"Source {group} has no permitted suite sensor rows (foreign suite or Stress source)")
    primary = [row["physical_unit_id"] for row in assignments if row["source_group"] == "primary"]
    automatic = ["train",*[part for part in SPLITS[1:] if request["modes"][part] == "auto"]]
    allocated = allocate_whole_units(primary,automatic,request["weights"],request["seed"])
    for part in SPLITS[1:]:
        if request["modes"][part] == "folder":
            allocated[part] = sorted(row["physical_unit_id"] for row in assignments if row["source_group"] == part)
    if set(allocated) != set(SPLITS) or not all(allocated[part] for part in SPLITS):
        raise ValueError("All four splits need at least one physical unit")
    destination = {physical:part for part,ids in allocated.items() for physical in ids}
    for row in assignments:
        row["destination_role"] = destination[row["physical_unit_id"]]
    reference = (len(assignments) == sum(manifest["splits"][suite].values())
                 and all(row["original_role"] == row["destination_role"] for row in assignments)
                 and {row["sha256"] for row in records} == set(expected.values()))
    provenance = {"policy":ASSIGNMENT_POLICY,"modes":request["modes"],"weights":request["weights"],"seed":request["seed"],
                  "source_manifest_sha256":manifest_digest,"source_release_id":release,"suite":suite,"sources":records,
                  "unit_mapping":sorted(assignments,key=lambda row:row["physical_unit_id"]),
                  "realized_counts":{part:len(allocated[part]) for part in SPLITS},"parent":parent,
                  "reference_equivalent":reference,"evaluation_only_accessed":False}
    provenance["assignment_hash"] = canonical_hash(provenance)
    def record(snapshot):
        if receipt_path:
            path = Path(receipt_path)
            if path.exists():
                raise FileExistsError("Import receipt is immutable")
            _stop(should_stop)
            atomic_write_json(path,{"assignment_provenance":provenance,"snapshot_id":snapshot["snapshot_id"],"dataset_hash":snapshot["dataset_hash"]})
            path.chmod(0o444)
        return snapshot
    _stop(should_stop)
    if reference:
        for directory in existing_directories:
            old = load_snapshot(directory)
            if (old["suite"] == suite and old["release_id"] == release and old["config_hash"] == config_hash(cfg)
                    and old["dataset_manifest"] == manifest
                    and all(old["split_manifest"][part]["sha256"] == expected[part] for part in SPLITS)):
                return record(old)
    output = Path(output_dir).expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".source-plan-",dir=output.parent) as temporary:
        root = Path(temporary)
        derived_manifest = dict(manifest)
        if not reference:
            derived_manifest.update(version=manifest["version"]+":development-v1",release_id=release,
                                    evaluation_status=DEVELOPMENT_STATUS,assignment_provenance=provenance,
                                    splits={suite:{part:len(allocated[part]) for part in SPLITS}},
                                    total_units=len(assignments),total_rows=sum(len(rows) for rows in raw_rows.values()),file_sha256={})
        for part in SPLITS:
            _stop(should_stop)
            relative = declared[part]
            target = root/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            matching = [row for row in records if row["origin_role"] == part]
            if reference:
                # Find the verified source again; preserve the original CSV byte for byte.
                candidate = next(path for group,spec in sources.items() for path,_ in _iter_source_files(spec)
                                 if path.name == "measurements.csv" and sha256_file(path) == matching[0]["sha256"])
                shutil.copyfile(candidate,target)
            else:
                with target.open("w",newline="") as handle:
                    writer = csv.DictWriter(handle,fieldnames=cfg["schema"],lineterminator="\n")
                    writer.writeheader()
                    for physical in allocated[part]:
                        writer.writerows(raw_rows[physical])
                derived_manifest["file_sha256"][relative] = sha256_file(target)
        if sha256_file(manifest_path) != manifest_digest:
            raise ValueError("Source manifest changed during import")
        atomic_write_json(root/"dataset_manifest.json",derived_manifest)
        snapshot = _prepare(root,suite,output,root/"dataset_manifest.json",cfg,False)
    return record(snapshot)
