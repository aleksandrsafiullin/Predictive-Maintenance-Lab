"""Append-only human feedback; acknowledgement never resolves or retrains."""
from __future__ import annotations

import json
from pathlib import Path

from pdm.training_protocol import fingerprint

FEEDBACK = {"unreviewed", "acknowledged", "inspection_completed", "fault_confirmed",
            "regime_explained", "sensor_issue", "maintenance_completed", "unresolved"}


def append_feedback(path, record):
    required = {"episode_id", "unit_id", "at", "status", "source", "verification_status"}
    if required - set(record) or record["status"] not in FEEDBACK:
        raise ValueError("Incomplete feedback record")
    payload = {"action": None, "confirmed_cause": None, "replaced_part": None,
               "post_maintenance_condition": None, **record, "automatic_retraining": False}
    payload["feedback_id"] = fingerprint(payload)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if not any(r["feedback_id"] == payload["feedback_id"] for r in existing):
        with path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    return payload
