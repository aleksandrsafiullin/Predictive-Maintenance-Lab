"""Train/Validation-only candidates; Test is never accessed by this script."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pdm.data.project_prepare import load_snapshot
from pdm.io_util import atomic_write_json
from pdm.learned_trajectory import (
    _event_distribution_metadata,
    fit_learned_model,
    learned_params,
    load_learned_bundle,
    predict_learned,
    prepare_prediction_frame,
    save_learned_bundle,
)
from pdm.trajectory_data import build_trajectory_frame, slice_frame
from pdm.trajectory_evaluation import evaluate_trajectory_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.config).read_text())
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=False)
    data = load_snapshot(raw["project_id"], raw["snapshot_id"], feature_partitions=("train", "validation"))
    # Test feature rows are excluded by Parquet filtering before decoding.
    train_features = data["features"][data["features"].unit_id.isin(data["split"]["train"])]
    config = learned_params(raw["engine_id"], raw["params"], train_features)
    sources = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path("src/pdm").rglob("*.py")
    }
    runner = Path(__file__).resolve()
    sources[str(runner)] = hashlib.sha256(runner.read_bytes()).hexdigest()
    train = build_trajectory_frame(
        data, data["split"]["train"], config, cap=config["max_windows_per_unit"]
    )
    validation = build_trajectory_frame(data, data["split"]["validation"], config,
                                      cap=raw.get("selection_validation_cap", 128))

    def frame_receipt(frame):
        groups = np.asarray(frame["physical_unit_id"], str)
        return {
            "origins": len(frame["x"]),
            "input_shape": list(frame["x"].shape),
            "target_shape": list(frame["y"].shape),
            "observed_targets": int(frame["mask"].sum()),
            "by_physical_unit": {
                group: int((groups == group).sum()) for group in np.unique(groups)
            },
        }

    atomic_write_json(
        directory / "frozen_candidate.json",
        {
            "config": config,
            "project_id": raw["project_id"],
            "snapshot_id": raw["snapshot_id"],
            "engine_id": raw["engine_id"],
            "fingerprint": data["fingerprint"],
            "source_hashes": sources,
            "test_access": False,
            "feature_read_scope": data["loaded_feature_partitions"],
            "selection_validation_cap": raw.get("selection_validation_cap", 128),
            "fit_frame_receipts": {"train": frame_receipt(train),
                                   "validation": frame_receipt(validation)},
            "quality_target": {"whole_path_coverage": 0.9, "mean_30min_width_g": 1.5},
        },
    )
    def report(status):
        atomic_write_json(directory / "status.json", status)
        print(status.get("message", status), flush=True)

    bundle = fit_learned_model(
        raw["engine_id"],
        train,
        validation,
        config,
        report=report,
        cache_dir=Path("output/bearings-learned-funnel-20261002/encoder-cache"),
    )
    artifacts = save_learned_bundle(bundle, directory)
    saved_run = {
        "dir": str(directory),
        "engine_id": raw["engine_id"],
        "params": config,
        "scaler": bundle["scaler"],
        "artifacts": artifacts,
        **_event_distribution_metadata(bundle["model"], config),
    }
    if raw["engine_id"] == "full_cns":
        saved_run["connectome"] = bundle["encoder"]["body"].provenance
    atomic_write_json(directory / "candidate_artifact_contract.json", saved_run)
    loaded = load_learned_bundle(saved_run)
    audit = slice_frame(validation, np.arange(min(3, len(validation["x"]))))
    before = predict_learned(bundle, audit)
    after = predict_learned(loaded, audit)
    if any(not np.array_equal(before[key], after[key]) for key in before):
        raise AssertionError("Saved candidate reload differs from fitted joint distribution")
    atomic_write_json(directory / "reload_verification.json", {
        "passed": True, "audit_partition": "Validation", "audit_origins": len(audit["x"]),
        "exact_joint_outputs": list(before), "test_access": False,
    })
    full_validation = build_trajectory_frame(data, data["split"]["validation"], config)
    full_validation = prepare_prediction_frame(
        loaded, full_validation, Path("output/bearings-learned-funnel-20261002/encoder-cache")
    )
    metrics = evaluate_trajectory_model(
        lambda batch: predict_learned(loaded, batch), full_validation, config, partition="Validation"
    )
    atomic_write_json(directory / "validation_metrics.json", metrics)
    report(
        {
            "stage": "completed",
            "message": f"Candidate saved: {bundle['selection']}",
            "selection": bundle["selection"],
        }
    )
    print(
        json.dumps(
            [
                {
                    k: r[k]
                    for k in (
                        "horizon_s",
                        "whole_path_coverage",
                        "mean_width_g",
                        "interval_score_skill",
                        "red_bracket_containment",
                        "red_mean_corridor_width_s",
                        "events_with_useful_warning",
                    )
                }
                for r in metrics["horizons"]
            ],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
