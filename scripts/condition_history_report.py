"""Post-fit common-clock and end-to-end history diagnostics; never selects on test."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from pdm.data.prepare import load_processed
from pdm.history import history_policy, history_windows
from pdm.io_util import read_json
from pdm.paths import dataset_runs
from pdm.preprocessing import apply_preprocessor
from pdm.train import UnitWindowDataset, _collate, load_trained_model
from pdm.training_engine import batch_forecast


def main(study_dir):
    torch.set_num_threads(1)
    root = Path(study_dir)
    manifest = read_json(root / "study_manifest.json")
    ds = manifest["config"]["dataset_id"]
    data = load_processed(ds, manifest["data_binding"]["dataset_version"])
    result = []
    jobs = list(manifest["fit_jobs"])
    repair = root / "recipe_repair_ledger.json"
    if repair.exists():
        jobs += [{**j, "id": j["recipe"], "kind": "event", "fold": 0} for j in read_json(repair)["fits"]]
    for job in jobs:
        if job["kind"] != "event" or job["status"] != "completed":
            continue
        model, prep, meta = load_trained_model(dataset_runs(ds) / job["run_id"])
        split = read_json(dataset_runs(ds) / job["run_id"] / "split.json")
        ids = split["validation"]
        f = data["features"][data["features"].unit_id.isin(ids)]
        encoded = apply_preprocessor(prep, f, ds)
        memory = prep.history_policy or history_policy("fixed_20")
        windows = history_windows(f, data["units"], ds, memory)
        dataset = UnitWindowDataset(encoded, windows, prep.feature_names, ds, prep.time_scale_s)
        predictions = []
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=32, collate_fn=_collate):
                output = batch_forecast(model, batch, None, "cpu")
                predictions.extend(output["point"].cpu().numpy().tolist())
        windows = windows.copy()
        windows["prediction"] = predictions
        common = history_windows(f, data["units"], ds, history_policy("fixed_60"))
        keys = set(zip(common.unit_id, common.end_index, strict=True))
        for uid in ids:
            unit = f[f.unit_id == uid]
            w = windows[windows.unit_id == uid]
            for clock in ("end_to_end", "common_clock_60"):
                selected = w if clock == "end_to_end" else w.loc[[key in keys for key in zip(w.unit_id, w.end_index, strict=True)]]
                expected = len(unit) if clock == "end_to_end" else int(common.unit_id.eq(uid).sum())
                errors = (selected.prediction - selected.target_rul_s).abs()
                near = selected.target_rul_s.between(0, 1800, inclusive="right")
                result.append({"candidate_id": job["id"], "run_id": job["run_id"], "unit_id": uid,
                    "role": "train_fold" if job["fold"] is not None else "validation_stopping_diagnostics",
                    "clock": clock, "expected": expected, "available": len(selected),
                    "coverage": len(selected) / expected if expected else None,
                    "mae": float(errors.mean()) if errors.notna().any() else None,
                    "near_1800_saved_time_units_mae": float(errors[near].mean()) if near.any() else None,
                    "quality_eligible": job.get("quality_eligible", True),
                    "independent_event": int(data["units"].set_index("unit_id").loc[uid, "event_observed"])})
    pd.DataFrame(result).to_csv(root / "history_per_unit_results.csv", index=False)
    table = pd.DataFrame(result).groupby(["candidate_id", "clock", "role"])[["mae", "near_1800_saved_time_units_mae", "coverage"]].mean().reset_index()
    table.to_csv(root / "history_comparison.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("study_dir")
    main(parser.parse_args().study_dir)
