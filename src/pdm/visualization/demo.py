"""Demo scenario hooks. Synthetic fixture only — never presented as biology."""

from __future__ import annotations

from typing import Any

from pdm.connectome.provenance import GRAPH_MODE_SYNTHETIC, SYNTHETIC_DISCLAIMER

DEMO_TRAIN_COMMAND = (
    "pdm train --dataset bearings --arch fly_connectome_reservoir --smoke --n-nodes 8"
)


def get_demo_instructions() -> str:
    """Return the documented smoke-train command for a local synthetic demo run."""
    return DEMO_TRAIN_COMMAND


def find_demo_run(dataset_id: str) -> dict | None:
    """Local fly_connectome_reservoir run with graph_mode=synthetic_fixture, or None."""
    from pdm.experiments import list_runs

    want = str(dataset_id)
    for rec in list_runs(want):
        if str(rec.get("dataset_id") or want) != want:
            continue
        arch = str(rec.get("architecture") or "").strip().lower()
        mode = str(rec.get("graph_mode") or "").strip().lower()
        if arch != "fly_connectome_reservoir":
            continue
        if mode == GRAPH_MODE_SYNTHETIC:
            return dict(rec)
        if bool(rec.get("is_synthetic")) and not mode:
            return dict(rec)
    return None


def load_demo_scenario(dataset_id: str) -> dict[str, Any] | None:
    """Select a synthetic-fixture reservoir run. Does not invent states or biology."""
    rec = find_demo_run(dataset_id)
    if rec is None:
        return None
    return {
        "run": rec,
        "run_id": rec.get("run_id"),
        "dataset_id": rec.get("dataset_id") or dataset_id,
        "architecture": rec.get("architecture"),
        "graph_mode": GRAPH_MODE_SYNTHETIC,
        "is_synthetic": True,
        "disclaimer": SYNTHETIC_DISCLAIMER,
        "instructions": get_demo_instructions(),
    }
