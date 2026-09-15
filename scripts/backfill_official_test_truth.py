"""Publish corrected official filter truth annotations without rerunning inference.

Only touches the named study manifest and creates new evaluation directories.
Source evaluation CSVs, checkpoints and original metrics remain immutable.
"""
from __future__ import annotations

import argparse
import shutil

import pandas as pd

from pdm.benchmark import load_evaluation
from pdm.config import load_dataset_config
from pdm.evaluate import (
    allocate_eval_staging,
    bind_evaluation_to_run,
    build_rul_metrics,
    filter_prefix_backtest_table,
    publish_eval_dir,
    resolve_run_task_config,
    run_alert_evaluation,
)
from pdm.experiments import run_dir
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.paths import runs_root
from pdm.worker import worker_alive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    args = parser.parse_args()
    if worker_alive():
        raise SystemExit("Stop the worker before updating its study manifest")
    manifest_path = runs_root() / "batches" / args.batch_id / "manifest.json"
    manifest = read_json(manifest_path)
    for task in manifest["tasks"]:
        if task["dataset_id"] != "filters" or not task.get("test_eval_id"):
            continue
        rid, old_id = task["run_id"], task["test_eval_id"]
        ev = load_evaluation("filters", rid, old_id)
        if ev["config"].get("truth_annotation_method") == "official_prefix_end_plus_elapsed":
            continue
        rdir = run_dir("filters", rid)
        bound = bind_evaluation_to_run(rdir, "filters")
        cfg, _ = resolve_run_task_config(rdir, load_dataset_config("filters"))
        pred = filter_prefix_backtest_table(ev["predictions"], bound["units"])
        pd.testing.assert_frame_equal(pred.drop(columns="actual_rul_s"), ev["predictions"].drop(columns="actual_rul_s"))
        assert pred.actual_rul_s.notna().all()
        eid, staging, dest = allocate_eval_staging(rdir)
        try:
            config = {**ev["config"], "eval_id": eid, "source_evaluation_id": old_id,
                      "truth_annotation_method": "official_prefix_end_plus_elapsed",
                      "inference_reused_unchanged": True}
            pred.to_csv(staging / "predictions.csv", index=False)
            config["predictions_sha256"] = sha256_file(staging / "predictions.csv")
            atomic_write_json(staging / "evaluation_config.json", config)
            metrics, by_unit = build_rul_metrics(pred, dataset_id="filters", units=bound["units"], cfg=cfg,
                                                eval_id=eid, run_id=rid, split=bound["split"],
                                                test_ids=config["evaluate_mask"]["unit_ids"], bound=bound, rdir=rdir)
            metrics.update(evaluation_status="exploratory_reused_holdout", source_evaluation_id=old_id,
                           inference_reused_unchanged=True, truth_annotation_method=config["truth_annotation_method"])
            atomic_write_json(staging / "metrics.json", metrics)
            by_unit.to_csv(staging / "metrics_by_unit.csv", index=False)
            run_alert_evaluation(pred, config["alert_policy"], units=bound["units"], eval_dir=staging,
                                 run_id=rid, pressure_limit_pa=config["pressure_limit_pa"])
            publish_eval_dir(staging, dest)
        except Exception:
            shutil.rmtree(staging)
            raise
        task["test_eval_id"] = eid
        task["superseded_test_eval_id"] = old_id
        atomic_write_json(manifest_path, manifest)
        print(f"{rid}: {old_id} → {eid}; {len(pred)} unchanged forecasts", flush=True)


if __name__ == "__main__":
    main()
