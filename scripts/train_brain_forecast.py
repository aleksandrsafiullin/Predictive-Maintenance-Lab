#!/usr/bin/env python3
"""Train a new anatomically complete continuous fly model; never mutate old runs.

Run with .venv/bin/python scripts/train_brain_forecast.py. Fits three grouped
train-only CV folds, reserves validation for empirical interval calibration,
and reports test bearings as an exploratory reused holdout.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pdm.config import load_dataset_config, model_defaults  # noqa: E402
from pdm.data.prepare import dataset_fingerprint_for_run, load_processed  # noqa: E402
from pdm.forecasting import (  # noqa: E402
    DEFAULT_TAU_S,
    encode_rul_targets,
    fit_interval_profile,
    forecast_metrics,
    score_readout,
    trajectory_design,
    weighted_ridge,
)
from pdm.io_util import atomic_write_json, checkpoint_hash, dump_yaml, sha256_file  # noqa: E402
from pdm.paths import dataset_runs  # noqa: E402
from pdm.preprocessing import fit_preprocessor  # noqa: E402
from pdm.train import (  # noqa: E402
    _build_reservoir_model,
    _experiment_snapshot,
    _reservoir_meta_from_model,
    _save_ckpt,
    _write_connectome_artifacts,
    load_trained_model,
    new_run_id,
)
from pdm.worker import clear_stop, pid_path, stop_path, worker_alive, write_status  # noqa: E402


def _check_stop():
    if stop_path().exists():
        raise InterruptedError("Brain forecast training cancelled")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-path")
    parser.add_argument("--graph-run", type=Path, help="Reuse immutable saved graph and reservoir matrices")
    parser.add_argument("--n-nodes", type=int, default=1000)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    cfg = load_dataset_config("bearings")
    mcfg = model_defaults(cfg)
    mcfg["architecture"] = "fly_connectome_reservoir"
    mcfg["history_length"] = 20
    mcfg["reservoir"].update(n_nodes=args.n_nodes, graph_mode="real_connectome", state_mode="continuous", readout="ridge")
    if args.source_path:
        mcfg["reservoir"]["source_path"] = args.source_path
    processed = load_processed("bearings")
    features, units, split = processed["features"], processed["units"], processed["split"]
    prep, _ = fit_preprocessor("bearings", features, units, split, cfg)
    if args.graph_run:
        print(f"Reusing frozen graph from {args.graph_run}", flush=True)
        model, _, old_meta = load_trained_model(args.graph_run)
        if model.input_size != len(prep.feature_names) or model.graph_mode != "real_connectome":
            raise ValueError("Reusable graph must be a real connectome with matching input width")
        model.state_mode = "continuous"
        model.time_scale_s = prep.time_scale_s
        mcfg["reservoir"].update({key: old_meta[key] for key in ("n_nodes", "leak", "spectral_radius", "input_scale", "seed")})
        rmeta = _reservoir_meta_from_model(model, mcfg)
    else:
        print("Building soma-complete real connectome", flush=True)
        model, rmeta = _build_reservoir_model(mcfg, input_size=len(prep.feature_names), head="rul", time_scale_s=prep.time_scale_s)
    if model.provenance.get("sampling_method") != "seeded_bfs_soma_xyz":
        raise RuntimeError("Training requires a soma-complete graph, not fallback selection")
    _check_stop()
    base = {"version": 1, "warmup_measurements": 20, "smoothing_tau_s": DEFAULT_TAU_S, "n_nodes": model.n_nodes}
    # Alpha applies to equal-bearing mean square loss, not a row-summed loss.
    alphas = (0.0001, 0.001, 0.01, 0.1, 1.0)
    candidates = [(transform, alpha) for transform in ("linear", "log1p") for alpha in alphas]
    scores = {candidate: [] for candidate in candidates}
    relative_scores = {candidate: [] for candidate in candidates}
    oof = {candidate: [] for candidate in candidates}
    train_ids = split["train"]
    train_units = units[units.unit_id.isin(train_ids)]
    folds = []
    for instance in sorted(train_units.instance.unique()):
        _check_stop()
        hold = sorted(train_units.loc[train_units.instance == instance, "unit_id"].tolist())
        fit_ids = sorted(set(train_ids)-set(hold))
        folds.append({"fit_ids": fit_ids, "held_out_ids": hold})
        fold_split = {**split, "train": fit_ids}
        fold_prep, _ = fit_preprocessor("bearings", features, units, fold_split, cfg)
        z_fit, rows_fit = trajectory_design(model, fold_prep, features, units, fit_ids)
        z_hold, rows_hold = trajectory_design(model, fold_prep, features, units, hold)
        for transform, alpha in candidates:
            targets = encode_rul_targets(rows_fit.target_rul_s, fold_prep.time_scale_s, transform)
            weights = weighted_ridge(z_fit, targets, rows_fit.unit_id, alpha)
            predictions = score_readout(z_hold, rows_hold, weights, fold_prep.time_scale_s, {**base, "rul_transform": transform})
            metrics = forecast_metrics(predictions)
            scores[(transform, alpha)].append(metrics["bearing_balanced"]["mae_s"])
            relative_scores[(transform, alpha)].append(float(np.mean([
                np.mean(np.abs(group.target_rul_s-group.predicted_rul_s)) / max(float((group.timestamp_s+group.target_rul_s).iloc[0]), 60.0)
                for _, group in predictions.groupby("unit_id")
            ])))
            oof[(transform, alpha)].append(predictions)
        print(f"CV fold {instance}: held out {hold}", flush=True)
    candidate = min(candidates, key=lambda candidate: (np.mean(relative_scores[candidate]), -candidate[1]))
    transform, alpha = candidate
    print(f"Selected transform={transform}, alpha={alpha}; grouped CV MAE={np.mean(scores[candidate]):.1f}s", flush=True)
    print(json.dumps({f"{t}:{a}": float(np.mean(scores[(t, a)])) for t, a in candidates}), flush=True)
    base.update(rul_transform=transform, rul_reference_s=60.0)
    model.rul_transform = transform
    model.rul_reference_s = 60.0
    z_train, rows_train = trajectory_design(model, prep, features, units, train_ids)
    weights = weighted_ridge(z_train, encode_rul_targets(rows_train.target_rul_s, prep.time_scale_s, transform), rows_train.unit_id, alpha)
    model.readout.load_ridge_vector(weights)
    # Evaluate and calibrate using the actual float32 serialized readout values.
    weights = np.r_[model.readout.W_x.detach().numpy().ravel(), model.readout.W_u.detach().numpy().ravel(), model.readout.b.detach().numpy().ravel()]
    _check_stop()
    z_cal, rows_cal = trajectory_design(model, prep, features, units, split["validation"])
    cal = score_readout(z_cal, rows_cal, weights, prep.time_scale_s, base)
    fingerprint = dataset_fingerprint_for_run(processed, split)
    source_hashes = {key: fingerprint.get(key) for key in ("dataset_version", "features_hash", "units_hash", "split_hash")}
    source_hashes.update(graph_hash=model.graph_hash, soma_file_hash=model.provenance.get("soma_file_hash"), weights_source_hash=model.provenance.get("file_hash"))
    profile = fit_interval_profile(pd.concat(oof[candidate], ignore_index=True), cal, split, source_hashes)
    profile.update(rul_transform=transform, rul_reference_s=60.0, n_nodes=model.n_nodes, evaluation_status="exploratory_reused_holdout")
    mcfg["reservoir"].update(rul_transform=transform, rul_reference_s=60.0)
    mcfg["reservoir"]["ridge_alpha"] = float(alpha)
    profile["model_selection"] = {"protocol": "three grouped folds by train instance; preprocessing fitted within fold", "folds": folds,
                                  "alpha_candidates": list(alphas), "selected_alpha": float(alpha),
                                  "selected_transform": transform, "target_candidates": ["linear", "log1p"],
                                  "selection_metric": "bearing_balanced_MAE_divided_by_observed_lifetime; train_CV_only",
                                  "cv_relative_mae": {f"{t}:{a}": float(np.mean(relative_scores[(t, a)])) for t, a in candidates},
                                  "cv_mae_s": {f"{t}:{a}": float(np.mean(scores[(t, a)])) for t, a in candidates}}
    cal = score_readout(z_cal, rows_cal, weights, prep.time_scale_s, profile)
    train = score_readout(z_train, rows_train, weights, prep.time_scale_s, profile)
    # This holdout was inspected in earlier iterations; report as exploratory.
    z_test, rows_test = trajectory_design(model, prep, features, units, split["test"])
    test = score_readout(z_test, rows_test, weights, prep.time_scale_s, profile)
    report = {"training": forecast_metrics(train), "calibration": forecast_metrics(cal), "test": forecast_metrics(test),
              "endpoint_definition": "last_recorded_sample", "test_used_for_selection": False,
              "evaluation_status": "exploratory_reused_holdout",
              "limitation": "Test outcomes were viewed in earlier iterations; this is not a fresh blind evaluation."}
    _check_stop()
    run_id = new_run_id("bearings", "fly_connectome_reservoir", False)
    rdir = dataset_runs("bearings") / run_id
    rdir.mkdir(parents=True, exist_ok=False)
    print(f"Saving {rdir}", flush=True)
    atomic_write_json(rdir / "status.json", {"status": "running", "run_id": run_id})
    atomic_write_json(rdir / "preprocessing.json", prep.to_dict())
    atomic_write_json(rdir / "split.json", split)
    atomic_write_json(rdir / "feature_schema.json", {"feature_names": prep.feature_names})
    _write_connectome_artifacts(rdir, model)
    metric = report["calibration"]["bearing_balanced"]["mae_s"]
    rmeta["ridge_alpha"] = float(alpha)
    for name in ("best.pt", "last.pt"):
        _save_ckpt(rdir / name, model, None, 1, 1, metric, mcfg, prep, split, "bearings", "rul", False, fingerprint, rmeta)
    fingerprint["checkpoint_hash"] = checkpoint_hash(rdir / "best.pt")
    profile["source_hashes"]["checkpoint_hash"] = fingerprint["checkpoint_hash"]
    atomic_write_json(rdir / "dataset_fingerprint.json", fingerprint)
    profile["source_hashes"].update(
        checkpoint_sha256=sha256_file(rdir / "best.pt"),
        preprocessing_sha256=sha256_file(rdir / "preprocessing.json"),
        graph_sha256=sha256_file(rdir / "connectome" / "graph.json"),
        dataset_fingerprint_sha256=sha256_file(rdir / "dataset_fingerprint.json"),
    )
    atomic_write_json(rdir / "interval_profile.json", profile)
    atomic_write_json(rdir / "forecast_evaluation.json", report)
    for name, frame in (("calibration", cal), ("test", test)):
        frame.to_parquet(rdir / f"{name}_forecast.parquet", index=False)
    full_cfg = copy.deepcopy(cfg)
    full_cfg.update(model=mcfg, smoke=False, mode="Full", head="rul", max_windows_per_unit=None,
                    n_train_windows=len(train), n_val_windows=len(cal))
    dump_yaml(rdir / "config.yaml", full_cfg)
    atomic_write_json(rdir / "experiment_snapshot.json", _experiment_snapshot(cfg, mcfg, prep, rdir, "bearings"))
    metrics = {"best_epoch": 1, "best_metric": metric, "selection_metric_name": "val MAE", "selection_metric_unit": "seconds",
               "selection_metric_label": "val MAE (seconds)", "last": {"selection_metric": metric},
               "last_train": {"selection_metric": report["training"]["bearing_balanced"]["mae_s"]},
               "n_train_windows": len(train), "n_val_windows": len(cal), "smoke": False, "mode": "Full"}
    atomic_write_json(rdir / "validation_metrics.json", metrics)
    pd.DataFrame([{"epoch": 1, "train_loss": 0, "val_loss": 0, "train_metric": metrics["last_train"]["selection_metric"], "val_metric": metric}]).to_csv(rdir / "training_history.csv", index=False)
    atomic_write_json(rdir / "status.json", {**metrics, "status": "completed", "run_id": run_id,
                                           "dataset_id": "bearings", "architecture": mcfg["architecture"],
                                           "state_mode": "continuous", "graph_mode": "real_connectome", "n_nodes": model.n_nodes, "rul_transform": transform})
    print(json.dumps({"run_id": run_id, "run_path": str(rdir), "evaluation": report}, indent=2), flush=True)


if __name__ == "__main__":
    if worker_alive():
        raise RuntimeError("A heavy job is already running. Please wait.")
    pid_path().parent.mkdir(parents=True, exist_ok=True)
    pid_path().write_text(str(os.getpid()), encoding="utf-8")
    clear_stop()
    write_status({"status": "training", "kind": "brain_forecast", "dataset_id": "bearings"})
    try:
        main()
        write_status({"status": "completed", "kind": "brain_forecast", "dataset_id": "bearings"})
    except InterruptedError:
        write_status({"status": "cancelled", "kind": "brain_forecast", "dataset_id": "bearings"})
    except Exception as exc:
        write_status({"status": "failed", "kind": "brain_forecast", "dataset_id": "bearings", "error": str(exc)})
        raise
    finally:
        if pid_path().exists() and pid_path().read_text().strip() == str(os.getpid()):
            pid_path().unlink()

