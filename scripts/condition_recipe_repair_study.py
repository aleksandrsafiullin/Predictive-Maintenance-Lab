"""Recorded v2 replacement diagnostics within the combined 48-fit cap; train only."""

from __future__ import annotations

import argparse
from pathlib import Path

from pdm.io_util import atomic_write_json, read_json
from pdm.paths import dataset_runs
from pdm.train import run_training
from pdm.training_protocol import protocol


def reserved_attempts():
    total = 0
    roots = [
        Path("runs/condition_studies") / sid
        for sid in ("20260916T170147Z_bfd686", "20260916T171150Z_9177ed")
    ]
    for root in roots:
        m = read_json(root / "study_manifest.json")
        total += sum(j.get("fit_attempts", 1) for j in m["fit_jobs"])
        for extra in ("residual_ablation/fit_ledger.json", "recipe_repair_ledger.json"):
            path = root / extra
            if path.exists():
                item = read_json(path)
                total += sum(j.get("fit_attempts", 1) for j in item["fits"])
    return total


def main(root):
    root = Path(root)
    m = read_json(root / "study_manifest.json")
    ds = m["config"]["dataset_id"]
    path = root / "recipe_repair_ledger.json"
    if path.exists():
        raise ValueError("Already attempted; explicit ledger-based resume is required")
    if reserved_attempts() + 2 > 48:
        raise ValueError("Combined fit budget exhausted: pending original fits are reserved")
    base = next(j for j in m["fit_jobs"] if j["id"] == "event_final")
    mode = base.get("resolved_history") or ("variable_20_60" if ds == "bearings" else "fixed_60")
    ledger = {
        "status": "running",
        "role": "train_fold_recipe_repair_diagnostics_no_promotion",
        "test_access": False,
        "replaces_invalid_comparisons": ["multiscale_trend_v1", "multiscale_no_age_v1"],
        "fits": [
            {"recipe": r, "status": "pending", "fit_attempts": 1}
            for r in ("multiscale_trend_v2", "multiscale_no_age_v2")
        ],
    }
    atomic_write_json(path, ledger)
    for entry in ledger["fits"]:
        recipe = entry["recipe"]
        rid = "condition_" + m["study_id"] + "_" + recipe
        entry.update(status="running", run_id=rid)
        atomic_write_json(path, ledger)
        config = protocol(
            ds,
            learning_rate=0.0003 if ds == "filters" else 0.001,
            sampling="full_pass",
            feature_recipe=recipe,
        )
        if ds == "filters":
            config["selection_history_min"] = 60
        result = run_training(
            ds,
            architecture="gru",
            training_protocol=config,
            history_mode=mode,
            seed=42,
            device_pref="cpu",
            split_override=m["folds"][0],
            run_id_override=rid,
            log=lambda text: print(text, flush=True),
        )
        entry.update(
            status=result["status"],
            history_mode=mode,
            best_metric=read_json(dataset_runs(ds) / rid / "validation_metrics.json")[
                "best_metric"
            ],
        )
        atomic_write_json(path, ledger)
    ledger.update(status="completed", reserved_combined_attempts=reserved_attempts())
    atomic_write_json(path, ledger)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study_dir")
    main(p.parse_args().study_dir)
