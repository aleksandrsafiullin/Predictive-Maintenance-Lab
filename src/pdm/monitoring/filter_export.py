"""Validate exported real tables; full-history import remains fail-closed."""
from pathlib import Path

import pandas as pd

from pdm.io_util import read_json, sha256_file


def validate_filter_export(directory):
    directory = Path(directory)
    if not (directory / "manifest.json").exists():
        return {"mode": "filters_full_history_v1", "status": "blocked_external_dependency",
                "reasons": ["MATLAB verified table export and origin mapping are required"]}
    manifest = read_json(directory / "manifest.json")
    if manifest.get("schema_version") != "filter_table_export_v1":
        raise ValueError("Unsupported MATLAB export schema")
    checks = []
    for entry in manifest["exported_tables"]:
        path = (directory / entry["file"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Invalid table path")
        frame = pd.read_csv(path)
        if len(frame) != entry["rows"] or list(frame.columns) != entry["columns"]:
            raise ValueError("Exported row/column mismatch")
        if not entry.get("numeric_roundtrip_verified"):
            raise ValueError("MATLAB numeric roundtrip was not verified")
        checks.append({"file": entry["file"], "sha256": sha256_file(path), "rows": len(frame)})
    return {"mode": "filters_full_history_v1", "status": "blocked_external_dependency", "checks": checks,
            "reasons": ["Independent origin/prefix-content matching, endpoint and Time/RUL unit verification required; import disabled"]}
