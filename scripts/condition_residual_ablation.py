"""Two extra median fits, train-fold only, no model/policy promotion or test access."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

try:
    from condition_recipe_repair_study import reserved_attempts
except ModuleNotFoundError:
    from scripts.condition_recipe_repair_study import reserved_attempts

from pdm.data.prepare import load_processed
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.monitoring.signal_forecast import supervised_targets


@threadpool_limits.wrap(limits=1)
def main(root):
    root = Path(root)
    study = read_json(root / "study_manifest.json")
    if study["status"] != "completed":
        raise ValueError("Complete the base study before spending its remaining fit budget")
    used = sum(j.get("fit_attempts", 1) for j in study["fit_jobs"])
    model = joblib.load(root / "sensor_screen.joblib")
    cfg = model["config"]
    if used + len(cfg["horizons"]) > study["config"]["fit_job_budget"]:
        raise ValueError("Ablation would exceed the original per-dataset attempt budget")
    if reserved_attempts() + len(cfg["horizons"]) > 48:
        raise ValueError("Combined fit budget exhausted: pending original fits are reserved")
    output = root / "residual_ablation"
    output.mkdir(exist_ok=True)
    if (output / "fit_ledger.json").exists():
        raise ValueError("Ablation already attempted; do not silently refit outside budget")
    ledger = {
        "status": "running",
        "base_attempts": used,
        "planned_extra_fits": len(cfg["horizons"]),
        "role": "train_fold_diagnostic_ablation_only_no_promotion",
        "test_access": False,
        "fits": [],
    }
    atomic_write_json(output / "fit_ledger.json", ledger)
    data = load_processed(cfg["profile"]["dataset_id"], study["data_binding"]["dataset_version"])
    fold = study["folds"][0]
    arguments = dict(
        tolerance=cfg["tolerance"],
        history_min=cfg["history_min"],
        history_max=cfg["history_max"],
        include_age=cfg["include_age"],
        multiscale=cfg["multiscale"],
        reference=model["reference"],
    )
    x, y, meta = supervised_targets(
        data["features"].loc[lambda f: f.unit_id.isin(fold["train"])],
        cfg["profile"],
        cfg["horizons"],
        **arguments,
    )
    vx, vy, vm = supervised_targets(
        data["features"].loc[lambda f: f.unit_id.isin(fold["validation"])],
        cfg["profile"],
        cfg["horizons"],
        **arguments,
    )
    names = [c for c in x.columns if c not in {"normality_score", "normality_available"}]
    rows = []
    for j, h in enumerate(cfg["horizons"]):
        entry = {"horizon": h, "quantile": 0.5, "status": "started", "fit_attempts": 1}
        ledger["fits"].append(entry)
        atomic_write_json(output / "fit_ledger.json", ledger)
        valid = np.isfinite(y[:, j])
        counts = meta.loc[valid].groupby("unit_id").size()
        weights = meta.loc[valid, "unit_id"].map(1 / counts).to_numpy()
        control = HistGradientBoostingRegressor(**model["models"][h][0.5].get_params())
        control.fit(x.loc[valid, names], y[valid, j], sample_weight=weights / weights.mean())
        artifact = output / f"raw_without_normality_{h}.joblib"
        joblib.dump(control, artifact)
        entry.update(status="completed", sha256=sha256_file(artifact))
        atomic_write_json(output / "fit_ledger.json", ledger)
        for name, prediction in [
            ("raw_plus_norm", model["models"][h][0.5].predict(vx)),
            ("raw_without_norm", control.predict(vx[names])),
        ]:
            prediction = np.maximum(0.0, prediction)
            for uid, idx in vm.groupby("unit_id").groups.items():
                idx = np.asarray(idx)
                idx = idx[np.isfinite(vy[idx, j])]
                rows.append(
                    {
                        "method": name,
                        "horizon": h,
                        "unit_id": uid,
                        "targets": len(idx),
                        "mae": float(np.abs(prediction[idx] - vy[idx, j]).mean())
                        if len(idx)
                        else None,
                        "evaluation_transform": "nonnegative_unrepaired_median_both_controls",
                    }
                )
    pd.DataFrame(rows).to_csv(output / "per_unit.csv", index=False)
    pd.DataFrame(rows).groupby(["method", "horizon"]).mae.mean().to_csv(output / "comparison.csv")
    ledger.update(
        status="completed",
        total_attempts=used + len(ledger["fits"]),
        interpretation="Ablates score and availability only; regime context and all other features held fixed. Not full quantile-curve ranking.",
    )
    atomic_write_json(output / "fit_ledger.json", ledger)
    print(pd.read_csv(output / "comparison.csv").to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("study_dir")
    main(p.parse_args().study_dir)
