from __future__ import annotations

import json
import random
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from pdm.config import load_dataset_config, model_defaults
from pdm.data.prepare import load_processed
from pdm.device import resolve_device
from pdm.io_util import append_line, atomic_write_json, dump_yaml
from pdm.losses import smooth_l1, weibull_nll
from pdm.models import PDMNet
from pdm.paths import dataset_runs, project_root
from pdm.preprocessing import Preprocessor, fit_preprocessor
from pdm.windows import build_windows

LogFn = Callable[[str], None]
StopFn = Callable[[], bool]


class UnitWindowDataset(Dataset):
    def __init__(
        self,
        features: pd.DataFrame,
        windows: pd.DataFrame,
        feature_names: list[str],
        dataset_id: str,
        time_scale_s: float,
        max_windows_per_unit: int | None = None,
        seed: int = 42,
    ) -> None:
        self.dataset_id = dataset_id
        self.feature_names = feature_names
        self.time_scale_s = time_scale_s
        self.arrays: dict[str, np.ndarray] = {}
        self.index: list[tuple[str, int, int, float, float, int]] = []
        rng = np.random.RandomState(seed)
        by_unit = defaultdict(list)
        feat_sorted = {
            uid: g.sort_values("timestamp_s").reset_index(drop=True)
            for uid, g in features.groupby("unit_id")
        }
        for uid, g in feat_sorted.items():
            arr = g[feature_names].to_numpy(dtype=np.float32)
            self.arrays[str(uid)] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        for _, w in windows.iterrows():
            uid = str(w["unit_id"])
            duration = float(w.get("duration_s", w.get("target_rul_s", np.nan)))
            event = int(w.get("event", 1))
            target = float(w["target_rul_s"]) if pd.notna(w.get("target_rul_s")) else float("nan")
            by_unit[uid].append((uid, int(w["start_index"]), int(w["end_index"]), target, duration, event))
        for uid, items in by_unit.items():
            if max_windows_per_unit is not None and len(items) > max_windows_per_unit:
                pick = rng.choice(len(items), size=max_windows_per_unit, replace=False)
                items = [items[i] for i in sorted(pick)]
            self.index.extend(items)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        uid, start, end, target, duration, event = self.index[i]
        x = self.arrays[uid][start : end + 1]
        return {
            "x": torch.from_numpy(x),
            "target": torch.tensor(target, dtype=torch.float32),
            "duration": torch.tensor(duration, dtype=torch.float32),
            "event": torch.tensor(event, dtype=torch.float32),
            "unit_id": uid,
        }


class UnitBalancedSampler(Sampler[int]):
    def __init__(self, dataset: UnitWindowDataset, n_draws: int, seed: int) -> None:
        self.n_draws = n_draws
        self.seed = seed
        self.by_unit: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(dataset.index):
            self.by_unit[item[0]].append(i)
        self.units = list(self.by_unit.keys())

    def __len__(self) -> int:
        return self.n_draws

    def __iter__(self):
        rng = np.random.RandomState(self.seed)
        for _ in range(self.n_draws):
            u = self.units[int(rng.randint(0, len(self.units)))]
            opts = self.by_unit[u]
            yield int(opts[int(rng.randint(0, len(opts)))])


def _collate(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    return {
        "x": x,
        "target": torch.stack([b["target"] for b in batch]),
        "duration": torch.stack([b["duration"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
        "unit_id": [b["unit_id"] for b in batch],
    }


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_run_id(dataset_id: str, architecture: str, smoke: bool) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    prefix = "smoke_" if smoke else ""
    return f"{prefix}{dataset_id}_{architecture}_{stamp}_{short}"


def split_fingerprint(split: dict) -> str:
    payload = json.dumps(
        {k: split.get(k) for k in ("train", "validation", "test", "protocol")},
        sort_keys=True,
    )
    import hashlib

    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def compatibility_dict(cfg_model: dict, prep: Preprocessor, split: dict, dataset_id: str, head: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id,
        "architecture": cfg_model["architecture"],
        "history_length": int(cfg_model["history_length"]),
        "hidden_size": int(cfg_model["hidden_size"]),
        "recurrent_layers": int(cfg_model["recurrent_layers"]),
        "head": head,
        "feature_names": list(prep.feature_names),
        "time_scale_s": prep.time_scale_s,
        "split_fingerprint": split_fingerprint(split),
    }


def checkpoints_compatible(saved: dict, current: dict) -> bool:
    keys = [
        "dataset_id",
        "architecture",
        "history_length",
        "hidden_size",
        "recurrent_layers",
        "head",
        "feature_names",
        "time_scale_s",
        "split_fingerprint",
    ]
    for k in keys:
        if saved.get(k) != current.get(k):
            return False
    return True


def run_training(
    dataset_id: str,
    *,
    architecture: str = "gru",
    max_epochs: int | None = None,
    history_length: int | None = None,
    smoke: bool = False,
    resume_run_id: str | None = None,
    device_pref: str = "auto",
    log: LogFn | None = None,
    should_stop: StopFn | None = None,
    status_cb: Callable[[dict[str, Any]], None] | None = None,
    max_windows_per_unit: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    def _log(msg: str) -> None:
        if log:
            log(msg)

    cfg = load_dataset_config(dataset_id)
    mcfg = model_defaults(cfg)
    mcfg["architecture"] = architecture.lower()
    if max_epochs is not None:
        mcfg["max_epochs"] = int(max_epochs)
    if history_length is not None:
        mcfg["history_length"] = int(history_length)
    if seed is not None:
        mcfg["seed"] = int(seed)
    if smoke:
        mcfg["max_epochs"] = min(int(mcfg["max_epochs"]), 5)
        if max_windows_per_unit is None:
            max_windows_per_unit = 48
        _log("Smoke test — not a quality benchmark")

    processed = load_processed(dataset_id)
    features = processed["features"]
    units = processed["units"]
    split = processed["split"]
    head = "rul" if dataset_id == "bearings" else "weibull"
    hist = int(mcfg["history_length"])
    set_seeds(int(mcfg["seed"]))

    windows = build_windows(features, units, hist, dataset_id)
    train_w = windows[windows["unit_id"].isin(split["train"])]
    val_w = windows[windows["unit_id"].isin(split["validation"])]
    if train_w.empty:
        raise RuntimeError("No training windows. Reduce history_length or prepare data.")

    prep, feat_t = fit_preprocessor(dataset_id, features, units, split, cfg, train_w)
    device_info = resolve_device(device_pref)
    device = device_info.torch_device
    if device_info.fallback_reason:
        _log(device_info.fallback_reason)

    if resume_run_id:
        run_id = resume_run_id
        rdir = dataset_runs(dataset_id) / run_id
        last_path = rdir / "last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"No last.pt to resume: {last_path}")
        blob = torch.load(last_path, map_location="cpu", weights_only=False)
        current = compatibility_dict(mcfg, prep, split, dataset_id, head)
        if not checkpoints_compatible(blob.get("compat") or {}, current):
            raise RuntimeError(
                "Checkpoint is not compatible with current dataset/split/features/architecture. "
                "Start a new experiment."
            )
        start_epoch = int(blob["meta"]["epoch"]) + 1
        best_epoch = int(blob["meta"]["best_epoch"])
        best_metric = float(blob["meta"]["best_metric"])
        _log(f"Resuming {run_id} from epoch {start_epoch}")
    else:
        run_id = new_run_id(dataset_id, mcfg["architecture"], smoke)
        rdir = dataset_runs(dataset_id) / run_id
        rdir.mkdir(parents=True, exist_ok=True)
        start_epoch = 1
        best_epoch = 0
        best_metric = float("inf")
        blob = None

    train_ds = UnitWindowDataset(
        feat_t[feat_t["unit_id"].isin(split["train"])],
        train_w,
        prep.feature_names,
        dataset_id,
        prep.time_scale_s,
        max_windows_per_unit=max_windows_per_unit,
        seed=int(mcfg["seed"]),
    )
    val_ds = UnitWindowDataset(
        feat_t[feat_t["unit_id"].isin(split["validation"])],
        val_w,
        prep.feature_names,
        dataset_id,
        prep.time_scale_s,
        max_windows_per_unit=None,
        seed=int(mcfg["seed"]) + 1,
    )
    n_draws = max(len(train_ds), 1)
    sampler = UnitBalancedSampler(train_ds, n_draws=n_draws, seed=int(mcfg["seed"]))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(mcfg["batch_size"]),
        sampler=sampler,
        num_workers=int(mcfg["num_workers"]),
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(mcfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )

    model = PDMNet(
        input_size=len(prep.feature_names),
        hidden_size=int(mcfg["hidden_size"]),
        num_layers=int(mcfg["recurrent_layers"]),
        architecture=mcfg["architecture"],
        head=head,
        dropout=float(mcfg["dropout"]),
        time_scale_s=prep.time_scale_s,
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(mcfg["learning_rate"]),
        weight_decay=float(mcfg["weight_decay"]),
    )
    if blob is not None:
        model.load_state_dict(blob["model_state_dict"])
        opt.load_state_dict(blob["optimizer_state_dict"])

    env = _environment(device_info)
    dump_yaml(rdir / "config.yaml", {"dataset_id": dataset_id, "model": mcfg, "smoke": smoke, "head": head})
    atomic_write_json(rdir / "split.json", split)
    atomic_write_json(rdir / "feature_schema.json", {"feature_names": prep.feature_names, "order": prep.feature_names})
    atomic_write_json(rdir / "preprocessing.json", prep.to_dict())
    atomic_write_json(rdir / "environment.json", env)
    manifest_src = project_root() / "data" / "manifest.json"
    if manifest_src.exists():
        atomic_write_json(rdir / "data_manifest.json", json.loads(manifest_src.read_text()))
    log_path = rdir / "train.log"
    hist_path = rdir / "training_history.csv"
    if not hist_path.exists():
        hist_path.write_text("epoch,train_loss,val_loss,val_metric,val_mae_events,n_val_event_units\n", encoding="utf-8")

    def emit(status: str, **extra):
        payload = {
            "status": status,
            "dataset_id": dataset_id,
            "run_id": run_id,
            "architecture": mcfg["architecture"],
            "head": head,
            "smoke": smoke,
            "device": device_info.name,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **extra,
        }
        atomic_write_json(rdir / "status.json", payload)
        if status_cb:
            status_cb(payload)
        return payload

    emit("training", epoch=start_epoch - 1, max_epochs=int(mcfg["max_epochs"]), best_epoch=best_epoch, best_metric=best_metric)
    patience = int(mcfg["early_stopping_patience"])
    bad = 0
    last_status = "training"
    try:
        for epoch in range(start_epoch, int(mcfg["max_epochs"]) + 1):
            if should_stop and should_stop():
                last_status = "stopped"
                _log(f"Stop requested at epoch {epoch}")
                break
            tr_loss = _run_epoch(model, opt, train_loader, dataset_id, device, mcfg, train=True)
            val_stats = _validate(model, val_loader, dataset_id, device, units, split)
            val_metric = val_stats["selection_metric"]
            improved = val_metric < best_metric - 1e-8
            if improved:
                best_metric = val_metric
                best_epoch = epoch
                bad = 0
                _save_ckpt(rdir / "best.pt", model, opt, epoch, best_epoch, best_metric, mcfg, prep, split, dataset_id, head, smoke)
            else:
                bad += 1
            _save_ckpt(rdir / "last.pt", model, opt, epoch, best_epoch, best_metric, mcfg, prep, split, dataset_id, head, smoke)
            line = (
                f"{epoch},{tr_loss:.6f},{val_stats['val_loss']:.6f},{val_metric:.6f},"
                f"{val_stats.get('val_mae_events')},{val_stats.get('n_val_event_units')}"
            )
            append_line(hist_path, line)
            msg = (
                f"epoch {epoch}/{mcfg['max_epochs']} train_loss={tr_loss:.4f} "
                f"val_loss={val_stats['val_loss']:.4f} val_metric={val_metric:.4f} best_epoch={best_epoch}"
            )
            _log(msg)
            append_line(log_path, msg)
            emit(
                "training",
                epoch=epoch,
                max_epochs=int(mcfg["max_epochs"]),
                train_loss=tr_loss,
                val_loss=val_stats["val_loss"],
                val_metric=val_metric,
                best_epoch=best_epoch,
                best_metric=best_metric,
                message=msg,
            )
            atomic_write_json(rdir / "validation_metrics.json", {"best_epoch": best_epoch, "best_metric": best_metric, "last": val_stats})
            if bad >= patience:
                _log(f"Early stopping at epoch {epoch}, best_epoch={best_epoch}")
                break
        if last_status != "stopped":
            last_status = "completed"
    except Exception as exc:  # noqa: BLE001
        last_status = "failed"
        _log(f"Training failed: {exc}")
        emit("failed", error=str(exc), best_epoch=best_epoch, best_metric=best_metric)
        raise
    emit(last_status, epoch=best_epoch, best_epoch=best_epoch, best_metric=best_metric, message=last_status)
    return {"run_id": run_id, "dir": str(rdir), "status": last_status, "best_epoch": best_epoch, "best_metric": best_metric}


def _run_epoch(model, opt, loader, dataset_id, device, mcfg, train: bool) -> float:
    model.train(train)
    losses = []
    for batch in loader:
        x = batch["x"].to(device)
        if train:
            opt.zero_grad(set_to_none=True)
        if dataset_id == "bearings":
            pred_norm = model(x)
            target_norm = (batch["target"].to(device) / max(model.time_scale_s, 1e-8))
            loss = smooth_l1(pred_norm, target_norm)
        else:
            lam, k = model(x)
            nll = weibull_nll(batch["duration"].to(device), batch["event"].to(device), lam, k, model.time_scale_s)
            loss = nll.mean()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(mcfg["grad_clip"]))
            opt.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def _validate(model, loader, dataset_id, device, units, split) -> dict[str, Any]:
    model.eval()
    losses = []
    by_unit_err: dict[str, list[float]] = defaultdict(list)
    by_unit_nll: dict[str, list[float]] = defaultdict(list)
    event_mae_units: dict[str, list[float]] = defaultdict(list)
    for batch in loader:
        x = batch["x"].to(device)
        if dataset_id == "bearings":
            pred_norm = model(x)
            target_norm = batch["target"].to(device) / max(model.time_scale_s, 1e-8)
            loss = torch.nn.functional.smooth_l1_loss(pred_norm, target_norm, reduction="none")
            pred_s = pred_norm * model.time_scale_s
            tgt_s = batch["target"].to(device)
            err = torch.abs(pred_s - tgt_s)
            for i, uid in enumerate(batch["unit_id"]):
                losses.append(float(loss[i].cpu()))
                by_unit_err[uid].append(float(err[i].cpu()))
        else:
            lam, k = model(x)
            nll = weibull_nll(batch["duration"].to(device), batch["event"].to(device), lam, k, model.time_scale_s)
            pred_s = model.predicted_rul_s(x)
            tgt = batch["target"].to(device)
            ev = batch["event"].to(device)
            for i, uid in enumerate(batch["unit_id"]):
                losses.append(float(nll[i].cpu()))
                by_unit_nll[uid].append(float(nll[i].cpu()))
                if float(ev[i]) > 0.5 and torch.isfinite(tgt[i]):
                    event_mae_units[uid].append(float(torch.abs(pred_s[i] - tgt[i]).cpu()))
    if dataset_id == "bearings":
        unit_mae = [float(np.mean(v)) for v in by_unit_err.values() if v]
        sel = float(np.mean(unit_mae)) if unit_mae else float(np.mean(losses) if losses else np.nan)
        return {"val_loss": float(np.mean(losses) if losses else np.nan), "selection_metric": sel, "val_mae_events": sel, "n_val_event_units": len(unit_mae)}
    unit_nll = [float(np.mean(v)) for v in by_unit_nll.values() if v]
    sel = float(np.mean(unit_nll)) if unit_nll else float(np.mean(losses) if losses else np.nan)
    ev_mae = [float(np.mean(v)) for v in event_mae_units.values() if v]
    return {
        "val_loss": float(np.mean(losses) if losses else np.nan),
        "selection_metric": sel,
        "val_mae_events": float(np.mean(ev_mae)) if ev_mae else None,
        "n_val_event_units": len(ev_mae),
        "note": "MAE reported only on validation units with observed 600 Pa events",
    }


def _save_ckpt(path, model, opt, epoch, best_epoch, best_metric, mcfg, prep, split, dataset_id, head, smoke):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "compat": compatibility_dict(mcfg, prep, split, dataset_id, head),
            "meta": {
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "smoke": smoke,
                "time_scale_s": prep.time_scale_s,
                "head": head,
                "architecture": mcfg["architecture"],
                "history_length": int(mcfg["history_length"]),
                "hidden_size": int(mcfg["hidden_size"]),
                "recurrent_layers": int(mcfg["recurrent_layers"]),
                "dropout": float(mcfg["dropout"]),
                "input_size": len(prep.feature_names),
                "feature_names": list(prep.feature_names),
                "dataset_id": dataset_id,
            },
        },
        path,
    )


def load_trained_model(run_path: Path, device: str = "cpu", which: str = "best") -> tuple[PDMNet, Preprocessor, dict]:
    ckpt_file = run_path / f"{which}.pt"
    if not ckpt_file.exists():
        ckpt_file = run_path / "last.pt"
    blob = torch.load(ckpt_file, map_location=device, weights_only=False)
    meta = blob["meta"]
    prep = Preprocessor.from_dict(json.loads((run_path / "preprocessing.json").read_text()))
    model = PDMNet(
        input_size=int(meta["input_size"]),
        hidden_size=int(meta["hidden_size"]),
        num_layers=int(meta["recurrent_layers"]),
        architecture=meta["architecture"],
        head=meta["head"],
        dropout=float(meta.get("dropout", 0.1)),
        time_scale_s=float(meta["time_scale_s"]),
    )
    model.load_state_dict(blob["model_state_dict"])
    model.to(device)
    model.eval()
    return model, prep, meta


def _environment(device_info) -> dict[str, Any]:
    import platform
    import sys

    import sklearn
    import torch

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "device": device_info.name,
        "fallback_reason": device_info.fallback_reason,
    }
