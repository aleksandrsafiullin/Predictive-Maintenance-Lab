"""Owned snapshot storage and display adapters independent of raw-source labels."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pdm.projects import _safe_id, project_store

ROLES = ("train","validation","calibration","test")


def snapshot_directory(project_id, snapshot_id=None, *, store=None):
    store = store or project_store()
    project = store.get(project_id)
    sid = snapshot_id or project.get("active_snapshot_id")
    _safe_id(sid,"snapshot ID")
    directory = Path(((project.get("source_manifest") or {}).get("snapshots") or {}).get(sid,store.snapshot_path(project_id,sid)))
    base = store.project_path(project_id) / "snapshots"
    if directory.is_symlink() or not directory.resolve().is_relative_to(base.resolve()) or any(p.is_symlink() for p in directory.parents if p != base.parent):
        raise ValueError("Snapshot storage escapes its owned project")
    return directory


def project_snapshot(project_id, snapshot_id=None, *, store=None):
    store = store or project_store()
    sid = snapshot_id or store.get(project_id).get("active_snapshot_id")
    directory = snapshot_directory(project_id,sid,store=store)
    if (directory / "snapshot.json").exists():
        from pdm.probabilistic.data import load_snapshot
        raw = load_snapshot(directory)
        if raw["snapshot_id"] != sid:
            raise ValueError("Snapshot ID does not match its registered storage")
        cfg = raw["config"]
        rule = cfg.get("imported_rule",{"mode":"absolute","direction":cfg.get("threshold_direction","above"),"yellow":cfg["yellow"],"red":cfg["red"]})
        assignment = raw["dataset_manifest"].get("assignment_provenance",{})
        declaration = assignment.get("signal_schema",{})
        identities = raw["dataset_manifest"].get("physical_unit_map",{})
        ids = [uid for part in ROLES for uid in raw["split_manifest"].get(part,{}).get("units",[])]
        units = pd.DataFrame({"unit_id":ids,"physical_unit_id":[identities.get(uid,raw["release_id"]+":"+uid) for uid in ids],
                              "source_group":["author_test" if uid in assignment.get("protected_test_units",[]) else assignment.get("unit_source_groups",{}).get(uid,"primary") for uid in ids]})
        return {"project_id":project_id,"snapshot_id":sid,"dir":directory,"task":cfg["task"],"sensor_snapshot":raw,"units":units,
                "split":{part:raw["split_manifest"].get(part,{}).get("units",[]) for part in ROLES},
                "schema":{**{key:declaration[key] for key in ("context_mapping","context_units","age_source") if key in declaration},
                          "signal_column":cfg["target"],"signal_label":cfg.get("signal_label",{"vibration_rms_g":"Vibration RMS"}.get(cfg["target"],cfg["target"])),"signal_unit":cfg["unit"],"cadence_s":cfg["cadence_s"],"thresholds":rule},
                "report":{"by_split":raw["admission"]},"config":cfg}
    from pdm.data.project_prepare import load_snapshot
    raw = load_snapshot(project_id,sid,store=store)
    return {**raw,"task":None,"split":{**raw["split"],"calibration":raw["split"].get("calibration",[])}}


def role_frame(snapshot, part, unit_id=None):
    if "sensor_snapshot" in snapshot:
        from pdm.probabilistic.data import load_split
        raw = load_split(snapshot["sensor_snapshot"],part)
        one = raw.loc[raw.unit_id == unit_id].copy() if unit_id is not None else raw.copy()
        one["signal"] = one[snapshot["schema"]["signal_column"]]
        one["gap_before"] = ~one.valid | ~one.groupby("unit_id").valid.shift(fill_value=False) | one.segment_id.ne(one.groupby("unit_id").segment_id.shift())
        one.loc[~one.valid,"signal"] = float("nan")
        return one
    frame = snapshot["features"]
    ids = snapshot["split"].get(part,[])
    return frame.loc[frame.unit_id.astype(str).eq(str(unit_id)) if unit_id is not None else frame.unit_id.astype(str).isin(ids)].copy()


def descriptive_frame(snapshot,part,unit_id=None):
    """Read only the selected role's immutable context; never a model feature loader."""
    if "sensor_snapshot" not in snapshot:
        return role_frame(snapshot,part,unit_id)
    from pdm.probabilistic.data import _validate_context_rows, load_snapshot
    raw = load_snapshot(snapshot["sensor_snapshot"]["directory"])
    if raw != snapshot["sensor_snapshot"]:
        raise ValueError("Descriptive context parent binding mismatch")
    descriptor = raw["dataset_manifest"].get("descriptive_context",{}).get(part)
    if not descriptor:
        return pd.DataFrame()
    from pdm.io_util import read_json
    body = read_json(Path(raw["directory"])/descriptor["path"])
    _validate_context_rows(body,descriptor,set(raw["split_manifest"][part]["units"]))
    rows = pd.DataFrame(body,columns=descriptor["columns"])
    if set(rows.columns) != set(descriptor["columns"]) or not set(rows.unit_id).issubset(raw["split_manifest"][part]["units"]):
        raise ValueError("Descriptive context schema/unit binding mismatch")
    rows = rows[descriptor["columns"]]
    return rows.loc[rows.unit_id.eq(unit_id)].copy() if unit_id is not None else rows


def limits_features(project_id,snapshot_id=None):
    """Only Training values are decoded for a limit suggestion."""
    return role_frame(project_snapshot(project_id,snapshot_id),"train")


def load_project_limits(project_id,snapshot_id):
    from pdm.data.project_prepare import load_zone_limits
    return load_zone_limits(project_id,snapshot_id)


def save_project_limits(project_id,snapshot_id,rule):
    from pdm.data.project_prepare import save_zone_limits
    return save_zone_limits(project_id,rule,expected_snapshot_id=snapshot_id)


def _prepare_sensor_assignment(sensor, split, directory, operation):
    """Reassign verified parent rows, including rejected telemetry, without decoding."""
    import csv
    import tempfile

    from pdm.io_util import atomic_write_json, sha256_file
    from pdm.probabilistic.contract import canonical_hash
    from pdm.probabilistic.data import _prepare, load_snapshot
    if load_snapshot(sensor["directory"]) != sensor:
        raise ValueError("Assignment parent differs from its verified sensor snapshot")
    source_ids = {uid for item in sensor["split_manifest"].values() for uid in item["units"]}
    assigned = [uid for part in ROLES for uid in split[part]]
    if set(assigned) != source_ids or len(assigned) != len(source_ids) or any(not split[part] for part in ROLES):
        raise ValueError("Assignment must preserve every parent unit in four nonempty disjoint roles")
    cfg = sensor["config"]
    raw_rows, source_files = {}, []
    for part, item in sensor["split_manifest"].items():
        path = Path(sensor["split_files"][part])
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != cfg["schema"]:
                raise ValueError("Assignment parent sensor schema changed")
            for row in reader:
                raw_rows.setdefault(row["unit_id"],[]).append(row)
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Assignment parent sensor bytes changed")
        source_files.append({"group":part,"relative_path":item["path"],"sha256":item["sha256"]})
    original = sensor["dataset_manifest"]
    physical = {uid:original.get("physical_unit_map",{}).get(uid,sensor["release_id"]+":"+uid) for uid in source_ids}
    old_barriers = {uid:indices for group in original.get("observed_barriers",{}).values() for uid,indices in group.items()}
    barriers = {part:{uid:old_barriers[uid] for uid in split[part] if uid in old_barriers} for part in ROLES}
    old_assignment = original.get("assignment_provenance",{})
    assignment = {**old_assignment,"policy":"verified-parent-sensor-assignment-v1:preserve-all-rows-and-barriers",
                  "source_digest":sensor["dataset_hash"],"source_files":source_files,
                  "mapping":{part:sorted(split[part]) for part in ROLES},"physical_unit_map":physical,
                  "parent":{"snapshot_id":sensor["snapshot_id"],"dataset_hash":sensor["dataset_hash"]},
                  "operation":operation,"operation_hash":canonical_hash(operation),"gap_barriers":barriers,
                  "realized_counts":{part:len(split[part]) for part in ROLES},"reference_equivalent":False}
    assignment.pop("assignment_hash",None)
    assignment["assignment_hash"] = canonical_hash(assignment)
    suite = sensor["suite"]
    manifest = {"version":"verified-parent-sensor-assignment-v1","profile":sensor["profile"],
                "seed":original["seed"],"release_id":sensor["release_id"],"schema":cfg["schema"],
                "target":cfg["target"],"physical_unit":cfg["unit"],"cadence_s":cfg["cadence_s"],
                "splits":{suite:{part:len(split[part]) for part in ROLES}},"physical_unit_map":physical,
                "observed_barriers":barriers,"assignment_provenance":assignment,"file_sha256":{},
                "total_units":len(set(physical.values())),"total_rows":sum(map(len,raw_rows.values())),
                "evaluation_status":"development_user_supplied" if sensor["evaluation_status"] == "development_user_supplied" else "development_reference_exposed"}
    with tempfile.TemporaryDirectory(prefix="pdm-sensor-assignment-") as temporary:
        root = Path(temporary)
        context_rows = {}
        from pdm.io_util import read_json
        for item in original.get("descriptive_context",{}).values():
            for row in read_json(Path(sensor["directory"])/item["path"]):
                context_rows.setdefault(row["unit_id"],[]).append(row)
        context = {}
        for part in ROLES:
            relative = f"{suite}/sensor_csv/{part}/measurements.csv"
            path = root/relative
            path.parent.mkdir(parents=True)
            with path.open("w",newline="") as handle:
                writer = csv.DictWriter(handle,fieldnames=cfg["schema"],lineterminator="\n")
                writer.writeheader()
                for uid in sorted(split[part]):
                    writer.writerows(raw_rows[uid])
            manifest["file_sha256"][relative] = sha256_file(path)
            if context_rows:
                relative_context = f"descriptive_context/{part}.json"
                target = root/relative_context
                target.parent.mkdir(parents=True,exist_ok=True)
                rows = [row for uid in sorted(split[part]) for row in context_rows.get(uid,[])]
                atomic_write_json(target,rows)
                context[part] = {"path":relative_context,"sha256":sha256_file(target),"columns":next(iter(original["descriptive_context"].values()))["columns"],
                                 "usage":"recorded descriptive context only; excluded from model inputs"}
        if context:
            manifest["descriptive_context"] = context
            assignment["descriptive_context"] = context
            assignment.pop("assignment_hash")
            assignment["assignment_hash"] = canonical_hash(assignment)
        atomic_write_json(root/"dataset_manifest.json",manifest)
        return _prepare(root,suite,directory,None,cfg,False)


def publish_assignment(project_id, parent, split, *, operation, store=None):
    """Publish a derived observed snapshot; neither source nor old bands are relabelled."""
    import shutil
    import uuid

    from pdm.worker import heavy_job_active
    store = store or project_store()
    sensor = parent["sensor_snapshot"]
    limits = load_project_limits(project_id,parent["snapshot_id"])
    directory = store.project_path(project_id)/"snapshots"/("snapshot-"+uuid.uuid4().hex)
    snapshot = _prepare_sensor_assignment(sensor,split,directory,operation)
    try:
        if limits:
            from pdm.data.project_prepare import ZONE_LIMITS_FILE
            from pdm.io_util import atomic_write_json
            atomic_write_json(directory/ZONE_LIMITS_FILE,limits)
        with store.launch_lock():
            if heavy_job_active():
                raise RuntimeError("Wait for the current job to finish before changing sets.")
            registry = store._load()
            project = store._entry(registry,project_id)
            if project["active_snapshot_id"] != parent["snapshot_id"]:
                raise ValueError("The data changed since this page loaded. Reload Data Quality and try again.")
            previous = project.get("source_manifest") or {}
            project["source_manifest"] = {**previous,"snapshots":{**previous.get("snapshots",{}),snapshot["snapshot_id"]:snapshot["directory"]}}
            store._activate_snapshot_locked(registry,project_id,snapshot["snapshot_id"])
    except BaseException:
        shutil.rmtree(directory)
        raise
    return project_snapshot(project_id,snapshot["snapshot_id"],store=store)
