"""Reproducible training v2. Legacy checkpoints continue to use their saved path."""
from __future__ import annotations

import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pdm.architectures import is_reservoir
from pdm.config import load_dataset_config, model_defaults
from pdm.data.prepare import dataset_fingerprint_for_run, load_processed
from pdm.data.quality import training_admission
from pdm.device import resolve_device
from pdm.io_util import atomic_write_json, dump_yaml, load_yaml, read_json, sha256_file
from pdm.losses import weibull_nll_seconds
from pdm.models import build_model
from pdm.paths import dataset_runs, runs_root
from pdm.predict import forecast_tensors
from pdm.preprocessing import fit_preprocessor
from pdm.training_protocol import (
    FullPassSampler,
    TrainingControl,
    fingerprint,
    validate_protocol,
    weighted_batch_loss,
    window_weights,
)
from pdm.windows import build_windows, filter_gap_params


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])


def batch_identity(batch):
    return [{"unit_id": uid, "window_index": i} for uid, i in zip(batch["unit_id"], batch["window_index"], strict=True)]


def checked(values, label, batch):
    if not torch.isfinite(values).all():
        raise FloatingPointError(f"Nonfinite {label}: {batch_identity(batch)}")
    return values


@torch.no_grad()
def state_cache(model, dataset, prep, binding, root, should_stop=None):
    """Cache fixed states only; no readout parameters participate in identity."""
    from pdm.train import _collate

    def tensor_hash(value):
        return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()

    identity = {"version": 2, "binding": binding, "preprocessing": prep.to_dict(),
                "windows": [list(row[:3]) for row in dataset.index],
                "graph": tensor_hash(model.W_res), "state_mode": model.state_mode,
                "leak": model.alpha, "input_weights": tensor_hash(model.W_in),
                "bias": tensor_hash(model.b_res)}
    key = fingerprint(identity)
    directory = Path(root) / "state_cache"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (key + ".npy")
    manifest = directory / (key + ".json")
    if path.exists() and manifest.exists() and read_json(manifest).get("sha256") == sha256_file(path):
        values = np.load(path)
        if values.shape == (len(dataset), model.n_nodes) and np.isfinite(values).all():
            return torch.from_numpy(values)
    result = []
    model.eval()
    for batch in DataLoader(dataset, batch_size=32, collate_fn=_collate):
        if should_stop and should_stop():
            raise InterruptedError("Stopped while computing reservoir cache")
        x = batch["x"].to(next(model.parameters()).device)
        result.append(checked(model.forward_states(x)[:, -1, :], "reservoir state", batch).cpu())
    values = torch.cat(result).contiguous()
    np.save(path, values.numpy())
    atomic_write_json(manifest, {"identity": identity, "sha256": sha256_file(path)})
    return values


def batch_forecast(model, batch, cache, device):
    x = batch["x"].to(device)
    states = cache[batch["window_index"]].to(device) if cache is not None else None
    output = forecast_tensors(model, x, states)
    if output["raw"] is not None:
        checked(output["raw"], "raw readout output", batch)
    checked(output["point"], "forecast", batch)
    if (output["point"] < 0).any():
        raise FloatingPointError(f"Negative forecast: {batch_identity(batch)}")
    return output


def per_window_loss(model, batch, output, dataset_id, device):
    if dataset_id == "bearings":
        target = batch["target"].to(device) / model.time_scale_s
        return torch.nn.functional.smooth_l1_loss(output["point"] / model.time_scale_s, target, reduction="none")
    loss = weibull_nll_seconds(batch["duration"].to(device), batch["event"].to(device),
                               output["scale_s"], output["shape"])
    return checked(loss, "survival loss", batch)


@torch.no_grad()
def score_model(model, loader, dataset_id, device, cache=None):
    model.eval()
    rows = []
    for batch in loader:
        output = batch_forecast(model, batch, cache, device)
        loss = per_window_loss(model, batch, output, dataset_id, device)
        points = output["point"].cpu().numpy()
        raw = output["raw"].cpu().numpy().reshape(len(points), -1) if output["raw"] is not None else None
        for i, uid in enumerate(batch["unit_id"]):
            target = float(batch["target"][i])
            row = {"unit_id": uid, "loss": float(loss[i]), "target": target,
                   "prediction": float(points[i]), "error": abs(float(points[i]) - target),
                   "over": max(float(points[i]) - target, 0), "event": float(batch["event"][i]),
                   "zero": float(points[i]) == 0, "negative_raw": bool(raw[i, 0] < 0) if raw is not None else False}
            rows.append(row)
    data = pd.DataFrame(rows)
    expected = {row[0] for row in loader.dataset.index}
    if len(data) != len(loader.dataset) or set(data.unit_id) != expected:
        raise ValueError("Incomplete epoch prediction coverage")
    by_unit = data.groupby("unit_id")
    all_mae = float(by_unit.error.mean().mean())
    if dataset_id == "bearings":
        near = data[data.target.between(0, 1800, inclusive="right")]
        if set(near.unit_id) != expected:
            raise ValueError("An equipment unit has no eligible near-30-minute score")
        metric = float(near.groupby("unit_id").error.mean().mean())
    else:
        metric = float(by_unit.loss.mean().mean())
    events = data[data.event > 0.5]
    return {"selection_metric": metric, "val_loss": float(by_unit.loss.mean().mean()),
            "all_history_mae_s": all_mae if dataset_id == "bearings" else None,
            "overestimation_s": float(by_unit.over.mean().mean()) if dataset_id == "bearings" else None,
            "val_mae_events": float(events.groupby("unit_id").error.mean().mean()) if len(events) else None,
            "n_val_event_units": int(events.unit_id.nunique()), "prediction_coverage": 1.0,
            "zero_fraction": float(by_unit.zero.mean().mean()),
            "negative_raw_fraction": float(by_unit.negative_raw.mean().mean())}


def train_v2(dataset_id, *, architecture, training_protocol, seed=None, n_nodes=None,
             graph_mode=None, readout=None, source_path=None, split_override=None,
             resume_run_id=None, run_id_override=None, device_pref="cpu", log=None,
             should_stop=None, status_cb=None):
    from pdm.train import (
        UnitBalancedSampler,
        UnitWindowDataset,
        _build_reservoir_model,
        _collate,
        _environment,
        _experiment_snapshot,
        _save_ckpt,
        _sync_run_checkpoint_hash,
        _write_connectome_artifacts,
        new_run_id,
        set_seeds,
        windows_per_unit_summary,
    )

    config = validate_protocol(training_protocol, dataset_id)
    log = log or (lambda message: None)
    if resume_run_id:
        saved_root = dataset_runs(dataset_id) / resume_run_id
        saved = load_yaml(saved_root / "config.yaml")
        saved_model = saved["model"]
        if saved_model.get("reservoir", {}).get("graph_scope") == "whole_classified_cns":
            raise ValueError("Resume Full CNS through its study; window training cannot continue a continuous model")
        if architecture != saved_model["architecture"]:
            raise ValueError("Checkpoint architecture changed")
        seed = saved_model["seed"] if seed is None else seed
        saved_res = saved_model.get("reservoir", {})
        n_nodes = saved_res.get("n_nodes") if n_nodes is None else n_nodes
        graph_mode = saved_res.get("graph_mode") if graph_mode is None else graph_mode
        readout = saved_res.get("readout") if readout is None else readout
        source_path = saved_res.get("source_path") if source_path is None else source_path
        if split_override is None:
            saved_split = read_json(saved_root / "split.json")
            if saved_split.get("protocol") == "training_v2_internal_cv":
                split_override = {k: saved_split[k] for k in ("train", "validation")}
    cfg = load_dataset_config(dataset_id)
    mcfg = model_defaults(cfg)
    mcfg.update(architecture=architecture, seed=int(seed if seed is not None else 42),
                history_length=20, max_epochs=config["max_epochs"], learning_rate=config["learning_rate"],
                batch_size=32, dropout=.1, weight_decay=.0001, gradient_clip_norm=1.)
    cfg.update(feature_recipe=config["feature_recipe"], training_protocol=config)
    cfg["model"] = mcfg
    res = mcfg.setdefault("reservoir", {})
    res.update(n_nodes=int(n_nodes or 1000), graph_mode=graph_mode or "real_connectome",
               readout=readout or ("ridge" if dataset_id == "bearings" else "gradient"))
    if source_path:
        res["source_path"] = source_path
    processed = load_processed(dataset_id)
    features, units = processed["features"], processed["units"]
    split, counts = training_admission(processed, cfg, 20)
    if split_override:
        fit, hold = set(split_override["train"]), set(split_override["validation"])
        if not fit or not hold or fit & hold or not fit | hold <= set(split["train"]):
            raise ValueError("Internal CV must partition original train objects only")
        split = {**split, **split_override, "test": [], "protocol": "training_v2_internal_cv"}
    cfg["admission_counts"] = counts
    seed = mcfg["seed"]
    set_seeds(seed)
    gap_kw = {}
    if dataset_id == "filters":
        k, interval = filter_gap_params(cfg)
        gap_kw = {"gap_multiplier": k, "sampling_interval_s": interval}
    windows = build_windows(features, units, 20, dataset_id, **gap_kw)
    train_windows = windows[windows.unit_id.isin(split["train"])]
    val_windows = windows[windows.unit_id.isin(split["validation"])]
    if train_windows.empty or val_windows.empty:
        raise ValueError("Empty admitted training or validation windows")
    missing_validation = set(split["validation"]) - set(val_windows.unit_id)
    if missing_validation:
        raise ValueError(f"Validation objects have no eligible windows: {sorted(missing_validation)}")
    prep, encoded = fit_preprocessor(dataset_id, features, units, split, cfg)
    head = "rul" if dataset_id == "bearings" else "weibull"
    # This study's reproducibility contract is CPU. MPS cannot evaluate float64
    # likelihoods; moving a resumed run across devices changes its trajectory.
    device_info = resolve_device("cpu")
    if device_pref != "cpu":
        device_info.fallback_reason = "Training v2 uses CPU for float64 likelihoods and reproducible resume"
        log(device_info.fallback_reason)
    device = device_info.torch_device
    torch.set_num_threads(1)
    rmeta = None
    if is_reservoir(architecture):
        model, rmeta = _build_reservoir_model(mcfg, input_size=len(prep.feature_names), head=head, time_scale_s=prep.time_scale_s)
    else:
        model = build_model(architecture=architecture, input_size=len(prep.feature_names),
                            hidden_size=mcfg["hidden_size"], num_layers=mcfg["recurrent_layers"], head=head, dropout=0.1, time_scale_s=prep.time_scale_s)
    model = model.to(device)
    run_id = resume_run_id or run_id_override or new_run_id(dataset_id, architecture, False)
    if Path(run_id).name != run_id:
        raise ValueError("Invalid run identifier")
    root = dataset_runs(dataset_id) / run_id
    root.mkdir(parents=True, exist_ok=True)
    binding = dataset_fingerprint_for_run(processed, split)
    identity = fingerprint({"protocol": config, "preprocessing": prep.to_dict(), "split": split,
                            "architecture": architecture, "seed": seed, "binding": binding,
                            "reservoir": rmeta})
    blob = None
    if resume_run_id:
        blob = torch.load(root / "last.pt", map_location="cpu", weights_only=False)
        if blob.get("training_state", {}).get("identity") != identity:
            raise ValueError("Checkpoint training protocol/data/features changed; start a new run")
        model.load_state_dict(blob["model_state_dict"], strict=True)
    elif (root / "config.yaml").exists():
        raise ValueError("Run exists; explicit resume is required")
    train_ds = UnitWindowDataset(encoded[encoded.unit_id.isin(split["train"])], train_windows, prep.feature_names, dataset_id, prep.time_scale_s)
    val_ds = UnitWindowDataset(encoded[encoded.unit_id.isin(split["validation"])], val_windows, prep.feature_names, dataset_id, prep.time_scale_s)
    full_pass = config["sampling"] == "full_pass"
    sampler = FullPassSampler(train_ds, seed) if full_pass else UnitBalancedSampler(train_ds, len(train_ds), seed)
    loader = DataLoader(train_ds, batch_size=32, sampler=sampler, collate_fn=_collate)
    train_eval = DataLoader(train_ds, batch_size=32, collate_fn=_collate)
    val_eval = DataLoader(val_ds, batch_size=32, collate_fn=_collate)
    weights = window_weights(train_ds.index, config["near_weight"]) if full_pass else np.ones(len(train_ds))
    if config["near_weight"] and not full_pass:
        raise ValueError("Near weighting requires the explicit full-pass objective")
    twpu, vwpu = windows_per_unit_summary(train_ds), windows_per_unit_summary(val_ds)
    selection = {"name": config["selection_metric"], "unit": "seconds" if dataset_id == "bearings" else "NLL", "label": config["selection_metric"]}
    run_cfg = {**cfg, "model": mcfg, "head": head, "dataset_id": dataset_id, "smoke": False,
               "mode": "Full", "max_windows_per_unit": None, "selection_metric": selection,
               "n_train_windows": len(train_ds), "n_val_windows": len(val_ds),
               "train_windows_per_unit": twpu, "val_windows_per_unit": vwpu}
    if not resume_run_id:
        dump_yaml(root / "config.yaml", run_cfg)
        from pdm.training_protocol import protocol_artifact

        atomic_write_json(root / "training_protocol.json", protocol_artifact(config, dataset_id, architecture, seed, training_identity=identity))
        atomic_write_json(root / "preprocessing.json", prep.to_dict())
        atomic_write_json(root / "split.json", split)
        atomic_write_json(root / "feature_schema.json", {"feature_names": prep.feature_names, "order": prep.feature_names})
        atomic_write_json(root / "dataset_fingerprint.json", binding)
        atomic_write_json(root / "environment.json", _environment(device_info))
        if rmeta:
            _write_connectome_artifacts(root, model)
        atomic_write_json(root / "experiment_snapshot.json", _experiment_snapshot(cfg, mcfg, prep, root, dataset_id))

    def emit(status, **extra):
        payload = {"status": status, "run_id": run_id, "dataset_id": dataset_id, "architecture": architecture,
                   "mode": "Full", "smoke": False, "training_protocol": config, "feature_recipe": prep.feature_recipe,
                   "max_epochs": config["max_epochs"], "n_train_windows": len(train_ds), "n_val_windows": len(val_ds),
                   "selection_metric_name": selection["name"], "selection_metric_label": selection["label"],
                   "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra}
        atomic_write_json(root / "status.json", payload)
        if status_cb:
            status_cb(payload)
        return payload

    emit("training", stage="initializing")
    caches = [None, None]
    if rmeta:
        try:
            caches = [state_cache(model, ds, prep, binding, runs_root() / "fixed_reservoir_cache", should_stop) for ds in (train_ds, val_ds)]
        except InterruptedError:
            emit("cancelled", stop_reason="user_stop")
            return {"status": "cancelled", "run_id": run_id, "dir": str(root)}
    if rmeta and dataset_id == "bearings":
        from pdm.training_readout import fit_window_readout

        try:
            return fit_window_readout(model=model, prep=prep, cfg=cfg, processed=processed,
                                      split=split, config=config, train_ds=train_ds, val_ds=val_ds,
                                      caches=caches, root=root, binding=binding, rmeta=rmeta,
                                      emit=emit, log=log, should_stop=should_stop)
        except InterruptedError:
            emit("cancelled", stop_reason="user_stop")
            return {"status": "cancelled", "run_id": run_id, "dir": str(root)}
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", threshold_mode="abs",
                  threshold=config["min_delta"], **config["scheduler"]) if config["mode"] == "adaptive" else None
    control = TrainingControl()
    start, best_epoch, best_metric = 1, 0, float("inf")
    history = []
    if blob:
        opt.load_state_dict(blob["optimizer_state_dict"])
        state = blob["training_state"]
        control = TrainingControl(**state["control"])
        if control.stop_reason in {"max_epochs", "early_stopping"}:
            emit("completed", **control.state_dict(), best_epoch=blob["meta"]["best_epoch"])
            return {"status": "completed", "run_id": run_id, "dir": str(root)}
        if scheduler:
            scheduler.load_state_dict(state["scheduler"])
        start = blob["meta"]["epoch"] + 1
        best_epoch, best_metric = blob["meta"]["best_epoch"], blob["meta"]["best_metric"]
        history = pd.read_csv(root / "training_history.csv").to_dict("records") if (root / "training_history.csv").exists() else []
        history = [row for row in history if row["epoch"] < start]
        restore_rng(state["rng"])

    def save(name, epoch):
        state = {"identity": identity, "protocol": config, "control": control.state_dict(),
                 "rng": rng_state(), "scheduler": scheduler.state_dict() if scheduler else None,
                 "sampler": {"policy": config["sampling"], "seed": seed, "next_epoch": epoch + 1}}
        _save_ckpt(root / name, model, opt, epoch, best_epoch, best_metric, mcfg, prep, split,
                   dataset_id, head, False, fingerprint=binding, reservoir_meta=rmeta, training_state=state)

    if not blob:
        save("last.pt", 0)
    started = time.monotonic()
    elapsed_before = float(history[-1].get("elapsed_s", 0)) if history else 0.
    last_epoch = start - 1
    try:
        for epoch in range(start, config["max_epochs"] + 1):
            if should_stop and should_stop():
                raise InterruptedError("Stopped before epoch")
            sampler.set_epoch(epoch)
            model.train()
            draws, losses, grads, shapes, scales = [], [], [], [], []
            weighted_loss_sum = 0.
            lr = opt.param_groups[0]["lr"]
            for batch in loader:
                if should_stop and should_stop():
                    raise InterruptedError("Stopped inside epoch; resume restarts this epoch")
                opt.zero_grad(set_to_none=True)
                output = batch_forecast(model, batch, caches[0], device)
                per_loss = checked(per_window_loss(model, batch, output, dataset_id, device), "loss", batch)
                w = torch.as_tensor(weights[batch["window_index"]], device=device)
                loss = weighted_batch_loss(per_loss, w, config)
                weighted_loss_sum += float((per_loss * w).detach().sum().cpu())
                loss.backward()
                try:
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
                except RuntimeError as exc:
                    raise FloatingPointError(f"Invalid gradients: {batch_identity(batch)}") from exc
                checked(torch.as_tensor(norm), "gradient norm", batch)
                opt.step()
                draws.extend(batch["window_index"])
                losses.extend(per_loss.detach().cpu().tolist())
                grads.append(float(norm))
                if "shape" in output:
                    shapes.extend(output["shape"].detach().cpu().tolist())
                    scales.extend(output["scale_s"].detach().cpu().tolist())
            train_stats = score_model(model, train_eval, dataset_id, device, caches[0])
            val_stats = score_model(model, val_eval, dataset_id, device, caches[1])
            metric = val_stats["selection_metric"]
            improved = metric < best_metric
            if improved:
                best_epoch, best_metric = epoch, metric
            stop = control.update(metric, epoch, config)
            if scheduler:
                scheduler.step(metric)
            last_epoch = epoch
            row = {"epoch": epoch, "train_loss": weighted_loss_sum / len(draws), "unweighted_train_loss": float(np.mean(losses)), "val_loss": val_stats["val_loss"],
                   "train_metric": train_stats["selection_metric"], "val_metric": metric,
                   "train_all_history_mae_s": train_stats["all_history_mae_s"],
                   "val_all_history_mae_s": val_stats["all_history_mae_s"],
                   "val_mae_events": val_stats["val_mae_events"], "n_val_event_units": val_stats["n_val_event_units"],
                   "val_overestimation_s": val_stats["overestimation_s"], "prediction_coverage": val_stats["prediction_coverage"],
                   "learning_rate": lr, "next_learning_rate": opt.param_groups[0]["lr"],
                   "best_epoch": best_epoch, "bad_epochs": control.bad_epochs, "stop_reason": control.stop_reason,
                   "n_eligible_windows": len(train_ds), "n_gradient_draws": len(draws), "n_unique_sampled_windows": len(set(draws)),
                   "loss_p50": float(np.median(losses)), "loss_p95": float(np.quantile(losses, .95)), "loss_max": max(losses),
                   "gradient_norm_p95": float(np.quantile(grads, .95)), "gradient_norm_max": max(grads),
                   "shape_min": min(shapes) if shapes else None, "shape_max": max(shapes) if shapes else None,
                   "scale_min_s": min(scales) if scales else None, "scale_max_s": max(scales) if scales else None,
                   "zero_fraction": val_stats["zero_fraction"], "negative_raw_fraction": val_stats["negative_raw_fraction"],
                   "elapsed_s": elapsed_before + time.monotonic() - started}
            history.append(row)
            pd.DataFrame(history).to_csv(root / "training_history.csv", index=False)
            save("last.pt", epoch)
            if improved:
                save("best.pt", epoch)
            _sync_run_checkpoint_hash(root)
            atomic_write_json(root / "validation_metrics.json", {"best_epoch": best_epoch, "best_metric": best_metric,
                              "selection_metric_name": selection["name"], "selection_metric_unit": selection["unit"],
                              "selection_metric_label": selection["label"], "last": val_stats, "last_train": train_stats,
                              "n_train_windows": len(train_ds), "n_val_windows": len(val_ds),
                              "train_windows_per_unit": twpu, "val_windows_per_unit": vwpu, "smoke": False,
                              "training_protocol": config})
            message = f"{run_id} epoch {epoch}/{config['max_epochs']} val={metric:.6f} best={best_epoch} lr={lr:g} unique={len(set(draws))}/{len(train_ds)}"
            log(message)
            with (root / "train.log").open("a") as handle:
                handle.write(message + "\n")
            emit("training", **row)
            if stop:
                break
    except InterruptedError as exc:
        emit("cancelled", epoch=last_epoch, best_epoch=best_epoch, stop_reason="user_stop", message=str(exc))
        return {"status": "cancelled", "run_id": run_id, "dir": str(root)}
    except Exception as exc:
        emit("failed", epoch=last_epoch, best_epoch=best_epoch, error=str(exc))
        raise
    emit("completed", epoch=last_epoch, best_epoch=best_epoch, best_metric=best_metric,
         stop_reason=control.stop_reason, learning_rate=opt.param_groups[0]["lr"])
    return {"status": "completed", "run_id": run_id, "dir": str(root), "best_epoch": best_epoch, "best_metric": best_metric}
