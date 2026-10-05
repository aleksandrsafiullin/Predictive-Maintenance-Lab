"""Create a hash-bound, nonactivated sensor snapshot; never read Test targets for selection."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.data.project_prepare import SNAPSHOT_FILES, load_snapshot
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.projects import project_store
from pdm.signal_training import _validate_snapshot
from pdm.trajectory_data import SENSOR_AVAILABILITY, SENSOR_FEATURE_NAMES, causal_features

KEYS = ["unit_id", "timestamp_s"]


def align_sensors(parent, source):
    """Join by acquisition key and admit only the audited local sensor allowlist."""
    if set(SENSOR_FEATURE_NAMES) & set(parent.columns):
        raise ValueError("Parent already contains trajectory sensor columns")
    if parent[KEYS].isna().any().any() or source[KEYS].isna().any().any():
        raise ValueError("Acquisition keys must be nonmissing")
    if parent.duplicated(KEYS).any() or source.duplicated(KEYS).any():
        raise ValueError("Duplicate acquisition keys")
    a = pd.MultiIndex.from_frame(parent[KEYS])
    b = pd.MultiIndex.from_frame(source[KEYS])
    if len(a) != len(b) or len(a.difference(b)) or len(b.difference(a)):
        raise ValueError("Missing or extra acquisition keys")
    aligned = source.set_index(KEYS).loc[a].reset_index()
    if not np.array_equal(aligned.timestamp_s.to_numpy(float), aligned.file_index.to_numpy(float) * 60):
        raise ValueError("Archive timestamp does not match acquisition file index")
    if not ((aligned.n_samples == 32768) & aligned.sample_count_ok & (aligned.n_nan == 0)).all():
        raise ValueError("Archive acquisition quality admission failed")
    for column in ("operating_age_s", "rpm", "load_kn"):
        if column in parent and not np.array_equal(parent[column].to_numpy(float), aligned[column].to_numpy(float), equal_nan=True):
            raise ValueError(f"Archive context mismatch: {column}")
    result = parent.copy()
    for column in SENSOR_FEATURE_NAMES:
        result[column] = aligned[column].to_numpy()
    # This checks finite/nonnegative values and exact max-axis RMS identity.
    causal_features(result, SENSOR_FEATURE_NAMES)
    pd.testing.assert_frame_equal(result.loc[:, parent.columns], parent)
    return result


def prepare_sensor_snapshot(project_id, parent_snapshot_id, archive_version, audit_evidence, *, store=None):
    store = store or project_store()
    data = load_snapshot(project_id, parent_snapshot_id, store=store)
    _validate_snapshot(data)
    parent = data["dir"]
    parent_hashes = {name: sha256_file(parent / name) for name in (*SNAPSHOT_FILES, "processed_fingerprint.json")}
    archive_version, audit_evidence = Path(archive_version).resolve(), Path(audit_evidence).resolve()
    evidence = read_json(audit_evidence)
    if evidence.get("allowlist") != list(SENSOR_FEATURE_NAMES):
        raise ValueError("Audit sensor allowlist does not match implementation")
    audit_hashes = {str(Path(rec["path"]).resolve()): rec["sha256"] for rec in evidence["files"]}
    source_hashes = {}
    for name in ("features.parquet", "feature_schema.json", "processed_fingerprint.json"):
        path = archive_version / name
        digest = sha256_file(path)
        if audit_hashes.get(str(path)) != digest:
            raise ValueError(f"Archive hash differs from audit: {name}")
        source_hashes[name] = digest
    for name, digest in parent_hashes.items():
        expected = audit_hashes.get(str((parent / name).resolve()))
        if expected is not None and expected != digest:
            raise ValueError(f"Parent hash differs from audit: {name}")
    source_fingerprint = read_json(archive_version / "processed_fingerprint.json")
    # Historical archive split and labels are never copied.
    if source_fingerprint.get("features_hash") != source_hashes["features.parquet"]:
        raise ValueError("Archive fingerprint feature hash mismatch")
    features = align_sensors(data["features"], pd.read_parquet(archive_version / "features.parquet"))
    key_digest = hashlib.sha256(json.dumps(features[KEYS].values.tolist(), separators=(",", ":")).encode()).hexdigest()
    provenance = {
        "schema_version": 1, "parent_project_id": project_id, "parent_snapshot_id": parent_snapshot_id,
        "parent_file_hashes": parent_hashes, "source_archive_version": archive_version.name,
        "source_archive_path": str(archive_version), "source_file_hashes": source_hashes,
        "audit_evidence_sha256": sha256_file(audit_evidence), "ordered_acquisition_key_sha256": key_digest,
        "sensor_allowlist": list(SENSOR_FEATURE_NAMES), "pipeline_version": "bearings_acquisition_local_enrichment_v1",
        "formulas": "pdm.features.time_domain_features and spectral_band_energy; no future filtering",
        "raw_verification_scope": {"zip": evidence.get("zip"), "sampled_members": evidence.get("raw_samples")},
        "integrity": {"rows": len(features), "missing_keys": 0, "extra_keys": 0,
                      "duplicate_keys": 0, "nonfinite_sensors": 0, "canonical_rms_max_abs_error": 0.0,
                      "parent_columns_exact": True, "split_unchanged": True, "test_target_diagnostics": False},
    }
    schema = {**data["schema"], "trajectory_sensor_features": {
        "schema_version": 1, "columns": list(SENSOR_FEATURE_NAMES), "transform": "log1p",
        "availability": SENSOR_AVAILABILITY, "acquisition_duration_s": 1.28,
        "units": {name: "FFT energy sum abs(rfft)^2/n" if "_band_" in name else
                  "dimensionless" if name.endswith(("crest_factor", "kurtosis")) else "g"
                  for name in SENSOR_FEATURE_NAMES},
        "scaler_fit": "Train only; persisted learned_trajectory scaler",
    }}
    sid = uuid.uuid4().hex
    destination = store.snapshot_path(project_id, sid)
    staging = Path(tempfile.mkdtemp(prefix=".trajectory-snapshot-", dir=parent.parent))
    published = False
    try:
        features.to_parquet(staging / "features.parquet", index=False)
        for name in ("units.parquet", "split.json"):
            shutil.copyfile(parent / name, staging / name)
        atomic_write_json(staging / "feature_schema.json", schema)
        atomic_write_json(staging / "data_report.json", {
            **data["report"], "snapshot_id": sid, "parent_snapshot_id": parent_snapshot_id,
            "created_at": datetime.now(timezone.utc).isoformat(), "trajectory_sensor_enrichment": provenance,
        })
        fingerprint = {**data["fingerprint"], "snapshot_id": sid,
                       "trajectory_sensor_enrichment": provenance,
                       "file_hashes": {name: sha256_file(staging / name) for name in SNAPSHOT_FILES}}
        atomic_write_json(staging / "processed_fingerprint.json", fingerprint)
        if parent_hashes != {name: sha256_file(parent / name) for name in parent_hashes}:
            raise ValueError("Parent changed during enrichment")
        for name, digest in source_hashes.items():
            if sha256_file(archive_version / name) != digest:
                raise ValueError("Archive changed during enrichment")
        staging.rename(destination)
        published = True
        loaded = load_snapshot(project_id, sid, store=store)
        _validate_snapshot(loaded)
        pd.testing.assert_frame_equal(loaded["features"].loc[:, data["features"].columns], data["features"])
        if sha256_file(destination / "split.json") != parent_hashes["split.json"]:
            raise ValueError("Split copy changed")
        return loaded
    except BaseException:
        shutil.rmtree(destination if published else staging, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--parent-snapshot-id", required=True)
    parser.add_argument("--archive-version", required=True)
    parser.add_argument("--audit-evidence", required=True)
    args = parser.parse_args()
    data = prepare_sensor_snapshot(args.project_id, args.parent_snapshot_id, args.archive_version, args.audit_evidence)
    print(json.dumps({"snapshot_id": data["snapshot_id"], "dir": str(data["dir"]),
                      "fingerprint": data["fingerprint"]}, indent=2))


if __name__ == "__main__":
    main()
