"""Train-group audit of provisional reference/policy; no labels or policy tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pdm.data.prepare import load_processed
from pdm.io_util import atomic_write_json, read_json
from pdm.monitoring.contracts import dataset_profile
from pdm.monitoring.normality import fit_reference
from pdm.monitoring.quality import quality_policy
from pdm.monitoring.runtime import replay_monitoring


def main(root):
    root = Path(root)
    study = read_json(root / "study_manifest.json")
    ds = study["config"]["dataset_id"]
    data = load_processed(ds, study["data_binding"]["dataset_version"])
    profile = dataset_profile(ds)
    policy = read_json(root / "state_policy.json")
    rows = []
    for fold_id, fold in enumerate(study["folds"][:2]):
        reference = fit_reference(data["features"], fold["train"], ds)
        for uid in fold["validation"]:
            # Initial fixed prefixes only: candidate-healthy status is itself unverified.
            frame = data["features"].loc[lambda f: f.unit_id.eq(uid)].head(60)
            records, state = replay_monitoring(
                frame,
                profile=profile,
                reference=reference,
                quality_policy=quality_policy(profile),
                state_policy=policy,
                bundle_id="train_fold_policy_audit",
            )
            rows.append(
                {
                    "fold": fold_id,
                    "unit_id": uid,
                    "measurements": len(records),
                    "reference_status": reference["status"],
                    "episodes": len(state["episodes"]) if state else 0,
                    "unknown_fraction": sum(
                        r["condition"]["health_state"] == "unknown" for r in records
                    )
                    / len(records),
                    "healthy_ground_truth": False,
                    "false_episode_rate": None,
                }
            )
    pd.DataFrame(rows).to_csv(root / "policy_train_fold_audit.csv", index=False)
    atomic_write_json(
        root / "policy_calibration_status.json",
        {
            "status": "provisional_not_false_alarm_calibrated",
            "reason": "Initial train prefixes are candidate healthy, not confirmed negative operating periods. No industrial false-episode target or verified healthy follow-up is available.",
            "threshold_selection": "fixed train candidate-healthy score quantiles; no test tuning",
            "audit": "two grouped train reference folds; no independent false-alarm accuracy claim",
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study_dir")
    main(p.parse_args().study_dir)
