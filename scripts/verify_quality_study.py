"""Verify real saved study models against their frozen prediction CSVs.

Run after the worker has stopped. Each model loads and releases sequentially;
this is an integration check of saved artifacts, not another training run.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pdm.benchmark import compare_evaluations, load_evaluation
from pdm.experiments import run_dir
from pdm.forecasting import load_interval_profile, predict_failure_interval
from pdm.io_util import atomic_write_json, read_json
from pdm.paths import runs_root
from pdm.predict import Predictor
from pdm.replay import bind_replay_to_run
from pdm.train import load_trained_model
from pdm.visualization.simulation import simulate_step
from pdm.worker import worker_alive


def verify_task(task):
    ds, rid = task["dataset_id"], task["run_id"]
    rdir = run_dir(ds, rid)
    bound = bind_replay_to_run(ds, rid)
    split = "test" if task["role"] == "main" else "validation"
    ev = load_evaluation(ds, rid, task[split + "_eval_id"])
    summary = compare_evaluations([ev])["table"].iloc[0]
    if "primary_score" in ev["metrics"]:
        np.testing.assert_allclose(ev["metrics"]["primary_score"], summary.primary_score,
                                   rtol=1e-10, atol=1e-8, err_msg=rid)
    uid = ev["config"]["evaluate_mask"]["unit_ids"][0]
    source = bound["features"].loc[bound["features"].unit_id == uid].sort_values("timestamp_s").reset_index(drop=True)
    model, prep, meta = load_trained_model(rdir, device="cpu", which="best")
    history = int(meta["history_length"])
    index = min(history + 4, len(source) - 1)
    now = float(source.iloc[index].timestamp_s)
    prefix = source.iloc[:index + 1].copy()
    predictor = Predictor(model, prep, history)
    plain = predictor.predict_from_history(prefix)
    traced = predictor.predict_from_history(prefix, with_trace=True)
    assert plain["ready"] and traced["ready"], rid
    np.testing.assert_allclose(plain["predicted_rul_s"], traced["predicted_rul_s"], rtol=1e-6, atol=1e-5, err_msg=rid)
    trace = simulate_step(source, uid, now, model, prep, history)
    rewind = simulate_step(source, uid, float(source.iloc[history].timestamp_s), model, prep, history, previous_trace=trace)
    repeated = simulate_step(source, uid, now, model, prep, history, previous_trace=rewind)
    np.testing.assert_allclose(trace["states"], repeated["states"], rtol=1e-6, atol=1e-6, err_msg=rid)
    np.testing.assert_allclose(trace["predicted_rul_s"], repeated["predicted_rul_s"], rtol=1e-6, atol=1e-5, err_msg=rid)
    perturbed = source.copy()
    sensor = "horizontal_rms" if ds == "bearings" else "differential_pressure"
    perturbed.loc[perturbed.timestamp_s > now, sensor] = 1e9
    perturbed["event_time_s"] = -1e9
    perturbed["RUL"] = 1e12
    unchanged = simulate_step(perturbed, uid, now, model, prep, history)
    np.testing.assert_allclose(trace["predicted_rul_s"], unchanged["predicted_rul_s"], rtol=0, atol=0, err_msg=rid)
    final = plain["predicted_rul_s"]
    lower, upper = plain.get("lower_rul_s"), plain.get("upper_rul_s")
    if getattr(model, "state_mode", None) == "continuous":
        profile = load_interval_profile(rdir)
        point = predict_failure_interval(trace["timestamps_s"], trace["raw_rul_s"], profile).iloc[-1]
        final, lower, upper = point.predicted_rul_s, point.lower_rul_s, point.upper_rul_s
    saved = ev["predictions"].loc[(ev["predictions"].unit_id == uid) & (ev["predictions"].timestamp_s == now)].iloc[0]
    np.testing.assert_allclose(final, saved.predicted_rul_s, rtol=1e-6, atol=1e-4, err_msg=rid)
    if lower is not None:
        np.testing.assert_allclose([lower, upper], [saved.lower_rul_s, saved.upper_rul_s], rtol=1e-6, atol=1e-4, err_msg=rid)
    result = {"dataset_id": ds, "run_id": rid, "eval_id": ev["config"]["eval_id"], "unit_id": uid,
              "timestamp_s": now, "architecture": model.architecture, "nodes": model.n_nodes,
              "state_mode": model.state_mode, "states_shape": list(traced["states"].shape),
              "cell_states_shape": list(traced["cell_states"].shape) if "cell_states" in traced else None,
              "raw_rul_s": plain["predicted_rul_s"], "displayed_rul_s": float(final),
              "saved_rul_s": float(saved.predicted_rul_s), "absolute_difference_s": abs(float(final) - float(saved.predicted_rul_s)),
              "trace_matches_plain": True, "seek_deterministic": True, "future_and_truth_independent": True,
              "evaluation_and_comparison_primary_match": "primary_score" in ev["metrics"]}
    del predictor, model, trace, traced, repeated, unchanged, rewind
    gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_id")
    parser.add_argument("--training-study", action="store_true", help="Verify a v2 improvement-study manifest")
    parser.add_argument("--scope", choices=["main", "all"], default="all")
    parser.add_argument("--output", type=Path, default=Path("output/quality-study-verification.json"))
    args = parser.parse_args()
    if worker_alive():
        raise SystemExit("Wait for the heavy worker to stop before loading verification models.")
    torch.set_num_threads(1)
    manifest = read_json(runs_root() / ("training_studies" if args.training_study else "batches") / args.batch_id / "manifest.json")
    if args.training_study:
        audited = 0
        for stage, task in manifest["tasks"].items():
            root = run_dir(task["dataset_id"], task["run_id"])
            history = pd.read_csv(root / "training_history.csv")
            metrics = read_json(root / "validation_metrics.json")
            if task["architecture"] in ("gru", "lstm") or task["dataset_id"] == "filters":
                selected = history.loc[history.val_metric.idxmin()]
                assert metrics["best_epoch"] == selected.epoch, stage
                np.testing.assert_allclose(metrics["best_metric"], selected.val_metric, rtol=1e-10, atol=1e-8)
                if task["recipe"]["mode"] == "diagnostic":
                    assert len(history) == 100, stage
                else:
                    assert 20 <= len(history) <= 100, stage
                if task["recipe"]["sampling"] == "full_pass":
                    assert history.n_unique_sampled_windows.eq(history.n_eligible_windows).all(), stage
            audited += 1
        print(json.dumps({"audited_training_histories": audited}), flush=True)
        tasks = [{**t, "role": "main", "test_eval_id": t.get("test_evaluation"),
                  "validation_eval_id": t.get("validation_evaluation")}
                 for key, t in manifest["tasks"].items() if ":main:" in key or (args.scope == "all" and ":confirmation:" in key)]
    else:
        tasks = [t for t in manifest["tasks"] if args.scope == "all" or t["role"] == "main"]
    if not all(t.get("status") == "completed" for t in tasks):
        raise SystemExit("Selected study tasks are not complete")
    rows = []
    for task in tasks:
        row = verify_task(task)
        rows.append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output, {"batch_id": args.batch_id, "verified_models": len(rows), "models": rows})
    comparisons = []
    for directory in manifest.get("comparisons", []):
        root = Path(directory)
        composition = read_json(root / "comparison.json")
        result = compare_evaluations([load_evaluation(composition["dataset_id"], item["run_id"], item["eval_id"])
                                      for item in composition["selections"]])
        keys = ["run_id", "unit_id", "timestamp_s"]
        saved = pd.read_csv(root / "predictions.csv").sort_values(keys).reset_index(drop=True)
        expected = result["predictions"].sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(saved[keys], expected[keys], check_dtype=False)
        for col in ("predicted_rul_s", "raw_rul_s", "actual_rul_s", "lower_rul_s", "upper_rul_s", "absolute_error_s"):
            if col in expected:
                np.testing.assert_allclose(saved[col], expected[col], equal_nan=True, rtol=1e-10, atol=1e-8)
        table = result["table"].set_index("run_id").sort_index()
        for exported in (pd.read_csv(root / "table.csv"), pd.DataFrame(composition["table"])):
            values = exported.set_index("run_id").sort_index()
            for col in ("primary_score", "rank", "prediction_coverage", "units", "interval_coverage"):
                np.testing.assert_allclose(pd.to_numeric(values[col]), pd.to_numeric(table[col]), equal_nan=True, rtol=1e-9, atol=1e-9)
        comparisons.append({"directory": directory, "dataset_id": composition["dataset_id"], "split": composition["split"],
                            "models": len(table), "prediction_rows": len(saved), "csv_json_match_saved_evaluations": True})
    atomic_write_json(args.output, {"batch_id": args.batch_id, "verified_models": len(rows), "models": rows, "comparisons": comparisons})
    print(json.dumps({"verified_comparisons": comparisons}), flush=True)


if __name__ == "__main__":
    main()
