"""Fixed-horizon entry targets derived from saved sensor-zone labels."""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from pdm.io_util import atomic_write_json, sha256_file
from pdm.paths import runs_root

DEFAULT_HORIZONS_S = {"bearings": 1800.0, "filters": 20.0}
SCHEMA_VERSION = "future_sensor_red_entry_targets_v1"


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fingerprint_digest(value: object) -> str:
    # Match the canonical serialization used by the current zone-label exporters.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode("utf-8")).hexdigest()


def _load_current_labels(dataset_id: str, data: dict) -> tuple[pd.DataFrame, dict, Path]:
    """Select a hash-valid label artifact for the exact prepared snapshot."""
    if dataset_id == "filters":
        from pdm.filter_health_zones_ui import _read_current_filter_zone_labels

        try:
            labels, manifest, directory = _read_current_filter_zone_labels(data)
        except FileNotFoundError:
            from pdm.monitoring.filter_zones import export_filter_zone_labels

            export_filter_zone_labels(dataset_version=data.get("dataset_version"))
            labels, manifest, directory = _read_current_filter_zone_labels(data)
        return labels, manifest, directory

    root = runs_root() / "_zones" / "bearings" / "label_artifacts"
    expected = _fingerprint_digest(data.get("fingerprint") or {})
    candidates = []
    for mp in root.glob("*/manifest.json"):
        try:
            manifest = json.loads(mp.read_text(encoding="utf-8"))
            lp = mp.parent / manifest.get("labels_file", "labels.csv")
            if (manifest.get("dataset_id") == "bearings"
                    and manifest.get("dataset_version") == data.get("dataset_version")
                    and manifest.get("dataset_fingerprint_sha256") == expected
                    and manifest.get("zone_definition") == "causal_vibration_bands_v1"
                    and sha256_file(lp) == manifest.get("labels_sha256")):
                labels = pd.read_csv(lp)
                required = {"unit_id", "split", "timestamp_s", "true_zone", "zone_name"}
                if required.issubset(labels.columns) and len(labels) == int(manifest["row_count"]):
                    candidates.append((str(manifest.get("created_at", "")), labels, manifest, mp.parent))
        except (OSError, ValueError, KeyError, json.JSONDecodeError, pd.errors.ParserError):
            continue
    if not candidates:
        from pdm.health_zones import export_zone_labels

        export_zone_labels("bearings")
        return _load_current_labels(dataset_id, data)
    _, labels, manifest, directory = max(candidates, key=lambda item: item[0])
    expected_splits = {str(uid): name for name in ("train", "validation", "test")
                       for uid in (data.get("split") or {}).get(name, [])}
    for uid, group in labels.groupby(labels.unit_id.astype(str)):
        if uid not in expected_splits or set(group.split.astype(str)) != {expected_splits[uid]}:
            raise ValueError(f"Bearing zone label split mismatch for unit {uid}")
    return labels.sort_values(["unit_id", "timestamp_s"], kind="stable"), manifest, directory


def _targets(labels: pd.DataFrame, dataset_id: str, horizon_s: float) -> tuple[pd.DataFrame, dict]:
    output = []
    first_red_map = {}
    masked = known_zero = known_one = outside = 0
    masked_quality = masked_gap = masked_censor = 0
    for uid, idx in labels.groupby(labels.unit_id.astype(str), sort=True).groups.items():
        group = labels.loc[idx].copy()
        group["_t"] = pd.to_numeric(group.timestamp_s, errors="coerce")
        group = group.sort_values("_t", kind="stable").reset_index(drop=True)
        if group._t.isna().any() or group._t.duplicated().any():
            raise ValueError(f"Invalid or duplicate timestamps in zone labels for unit {uid}")
        if dataset_id == "bearings":
            code_to_name = {-1: "unknown", 0: "green", 1: "yellow", 2: "red"}
            codes = pd.to_numeric(group["true_zone"], errors="coerce")
            names = group["zone_name"].astype(str)
            expected_names = codes.map(code_to_name)
            if codes.isna().any() or not names.eq(expected_names).all():
                raise ValueError(f"Bearing numeric zone codes and saved zone names disagree for unit {uid}")
            z = names.to_numpy()
        else:
            z = group["display_zone"].astype(str).to_numpy()
        ts = group["_t"].to_numpy(float)
        ok = np.isin(z, ["green", "yellow", "red"])
        reds = np.flatnonzero(z == "red")
        first_idx = int(reds[0]) if len(reds) else None
        first_t = float(ts[first_idx]) if first_idx is not None else None
        first_red_map[str(uid)] = first_t
        diffs = np.diff(ts)
        typical = float(pd.Series(diffs[diffs > 0]).mode().iloc[0]) if np.any(diffs > 0) else math.inf
        for i, t in enumerate(ts):
            risk = first_t is None or t < first_t
            target = None
            status = "out_of_risk_set"
            mask_reason = None
            if risk:
                end = t + horizon_s
                future_red = np.flatnonzero((ts > t) & (ts <= end) & (z == "red"))
                # Every recorded row in the interval must be usable; excessive gaps are unknown.
                window = np.flatnonzero((ts > t) & (ts <= end))
                evidence_end = int(future_red[0]) if future_red.size else None
                quality_rows = window[window <= evidence_end] if evidence_end is not None else window
                quality_ok = bool(ok[i]) and bool(ok[quality_rows].all())
                prev = i
                gap_ok = True
                next_after = np.flatnonzero(ts > end)
                checked = list(window[window <= evidence_end]) if evidence_end is not None else list(window)
                if evidence_end is None and next_after.size:
                    checked.append(int(next_after[0]))
                for j in checked:
                    if ts[j] - ts[prev] > typical * 1.5 + 1e-9:
                        gap_ok = False
                    prev = int(j)
                complete = end <= ts[-1] + 1e-9
                if future_red.size and quality_ok and gap_ok:
                    target, status = 1, "positive"
                    known_one += 1
                elif complete and quality_ok and gap_ok:
                    target, status = 0, "negative"
                    known_zero += 1
                else:
                    status = "masked"
                    masked += 1
                    if not quality_ok:
                        masked_quality += 1
                        mask_reason = "quality"
                    elif not gap_ok:
                        masked_gap += 1
                        mask_reason = "gap"
                    else:
                        masked_censor += 1
                        mask_reason = "censor"
            else:
                outside += 1
            output.append({"unit_id": str(uid), "split": str(group.split.iloc[0]), "timestamp_s": float(t),
                           "first_red_timestamp_s": first_t, "at_risk": bool(risk), "target": target,
                           "target_known": target is not None, "target_status": status,
                           "mask_reason": mask_reason})
    result = pd.DataFrame(output)
    result["target"] = pd.array(result["target"], dtype="Int64")
    return result, {"first_red_timestamp_s_by_unit": first_red_map,
                    "known_positive_count": known_one, "known_negative_count": known_zero,
                    "masked_count": masked, "masked_quality_count": masked_quality,
                    "masked_gap_count": masked_gap, "masked_censor_count": masked_censor,
                    "out_of_risk_set_count": outside}


def build_future_red_targets(dataset_id: str, horizon_s: float | None = None) -> dict:
    """Export entry-in-red targets from the current saved bearing/filter zone labels."""
    if dataset_id not in DEFAULT_HORIZONS_S:
        raise ValueError("dataset_id must be 'bearings' or 'filters'")
    horizon = float(DEFAULT_HORIZONS_S[dataset_id] if horizon_s is None else horizon_s)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon_s must be a finite positive number")
    from pdm.data.prepare import load_processed

    data = load_processed(dataset_id)
    if dataset_id == "bearings":
        from pdm.health_zones import _verify_zone_dataset_snapshot

        _verify_zone_dataset_snapshot(data)
    else:
        from pdm.monitoring.filter_zones import export_filter_zone_labels

        # This verifies every fingerprinted prepared input before labels are read.
        export_filter_zone_labels(dataset_version=data.get("dataset_version"))
    labels, label_manifest, label_dir = _load_current_labels(dataset_id, data)
    targets, counts = _targets(labels, dataset_id, horizon)
    fp = data.get("fingerprint") or {}
    fp_sha = _fingerprint_digest(fp)
    zone_policy = label_manifest.get("label_policy", label_manifest.get("policy", {}))
    comparator = ("first saved bearing zone_name == red (numeric true_zone code 2 validated)" if dataset_id == "bearings"
                  else "first saved filter display_zone == red (differential pressure >= 600 Pa)")
    identity = {"dataset_id": dataset_id, "dataset_version": data.get("dataset_version"),
                "dataset_fingerprint_sha256": fp_sha, "source_label_artifact_id": label_manifest.get("artifact_id"),
                "source_label_sha256": label_manifest.get("labels_sha256"), "horizon_s": horizon,
                "comparator": comparator, "zone_policy_sha256": _digest(zone_policy)}
    artifact_id = "future_red_entry_v1_" + _digest(identity)[:12]
    out_dir = runs_root() / "_future_red" / dataset_id / "targets" / artifact_id
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "targets.csv"
    targets.to_csv(target_path, index=False, float_format="%.10g")
    split_counts = {}
    for split, group in targets.groupby("split", sort=True):
        split_counts[str(split)] = {"known_positive": int((group.target == 1).sum()),
                                    "known_negative": int((group.target == 0).sum()),
                                    "masked": int(group.target_status.eq("masked").sum()),
                                    "masked_quality": int(group.mask_reason.eq("quality").sum()),
                                    "masked_gap": int(group.mask_reason.eq("gap").sum()),
                                    "masked_censor": int(group.mask_reason.eq("censor").sum()),
                                    "out_of_risk_set": int(group.target_status.eq("out_of_risk_set").sum())}
    manifest = {"schema_version": SCHEMA_VERSION, "artifact_id": artifact_id,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **identity, "dataset_fingerprint": fp, "zone_policy": zone_policy,
                "zone_policy_sha256": _digest(zone_policy), "source_zone_label_directory": str(label_dir),
                "source_zone_label_file_sha256": label_manifest.get("labels_sha256"),
                "source_zone_label_manifest_sha256": sha256_file(label_dir / "manifest.json"),
                "target_definition": ("1 iff first saved bearing zone_name == red occurs in (origin, origin+horizon]; "
                                      "0 iff complete usable observed horizon contains no red; otherwise masked"
                                      if dataset_id == "bearings" else
                                      "1 iff first saved filter display_zone == red occurs in (origin, origin+horizon]; "
                                      "0 iff complete usable observed horizon contains no red; otherwise masked"),
                "first_red_timestamp_s_by_unit": counts.pop("first_red_timestamp_s_by_unit"),
                "comparator": comparator, "horizon_s": horizon, **counts,
                "known_count": counts["known_positive_count"] + counts["known_negative_count"],
                "row_count": int(len(targets)), "unit_count": int(targets.unit_id.nunique()),
                "split_counts": split_counts, "targets_file": target_path.name,
                "targets_sha256": sha256_file(target_path)}
    atomic_write_json(out_dir / "manifest.json", manifest)
    return {"artifact_id": artifact_id, "directory": str(out_dir), "targets_path": str(target_path),
            "manifest_path": str(out_dir / "manifest.json"), "manifest": manifest}


def load_future_red_targets(artifact_dir: str | Path, data: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Load and verify a target artifact, optionally against a current prepared snapshot."""
    directory = Path(artifact_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    path = directory / manifest["targets_file"]
    if manifest.get("schema_version") != SCHEMA_VERSION or sha256_file(path) != manifest.get("targets_sha256"):
        raise ValueError("Future red target artifact schema or file hash is invalid")
    source_dir = Path(manifest["source_zone_label_directory"])
    source_manifest_path = source_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_labels_path = source_dir / source_manifest["labels_file"]
    if (sha256_file(source_manifest_path) != manifest.get("source_zone_label_manifest_sha256")
            or sha256_file(source_labels_path) != manifest.get("source_zone_label_file_sha256")
            or source_manifest.get("artifact_id") != manifest.get("source_label_artifact_id")
            or _digest(manifest.get("zone_policy", {})) != manifest.get("zone_policy_sha256")):
        raise ValueError("Future red target source zone-label artifact or policy hash is invalid")
    if data is not None:
        if data.get("dataset_version") != manifest.get("dataset_version") or _fingerprint_digest(data.get("fingerprint") or {}) != manifest.get("dataset_fingerprint_sha256"):
            raise ValueError("Future red target artifact does not match the supplied prepared snapshot")
    frame = pd.read_csv(path)
    required = {"unit_id", "split", "timestamp_s", "first_red_timestamp_s", "at_risk", "target", "target_known", "target_status"}
    if not required.issubset(frame.columns) or len(frame) != int(manifest.get("row_count", -1)):
        raise ValueError("Future red target table has an invalid schema or row count")
    if frame.duplicated(["unit_id", "timestamp_s"]).any():
        raise ValueError("Future red target table has duplicate unit/timestamp origins")
    return frame, manifest
