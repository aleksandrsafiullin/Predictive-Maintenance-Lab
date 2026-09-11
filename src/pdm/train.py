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
from pdm.data.prepare import dataset_fingerprint_for_run, load_processed
from pdm.device import resolve_device
from pdm.io_util import (
    append_line,
    atomic_write_json,
    checkpoint_hash,
    dump_yaml,
    load_yaml,
    read_json,
    sha256_file,
)
from pdm.losses import smooth_l1, weibull_nll
from pdm.models import PDMNet
from pdm.paths import dataset_runs, project_root
from pdm.preprocessing import (
    Preprocessor,
    apply_preprocessor,
    categorical_maps_fingerprint,
    fit_preprocessor,
    preprocessor_resume_mismatches,
)
from pdm.splits import resolve_split_hash, split_hash
from pdm.windows import build_windows, filter_gap_params

# Deprecated alias so `from pdm.train import split_fingerprint` still works.
split_fingerprint = split_hash

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
            self.arrays[str(uid)] = g[feature_names].to_numpy(dtype=np.float32)
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
            "window_index": int(i),
        }


class UnitBalancedSampler(Sampler[int]):
    def __init__(self, dataset: UnitWindowDataset, n_draws: int, seed: int) -> None:
        self.n_draws = int(n_draws)
        self.seed = int(seed)
        self.epoch: int | None = None
        self.by_unit: dict[str, list[int]] = defaultdict(list)
        for i, item in enumerate(dataset.index):
            self.by_unit[item[0]].append(i)
        self.units = list(self.by_unit.keys())

    def set_epoch(self, epoch: int) -> None:
        """Bind the 1-based training epoch. Resume uses last.pt epoch + 1, never 0."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_draws

    def __iter__(self):
        if self.epoch is None:
            raise RuntimeError("UnitBalancedSampler.set_epoch(epoch) must be called before iterating")
        # seed+0 is not epoch 1; training epochs are 1-based.
        rng = np.random.RandomState(int(self.seed) + int(self.epoch))
        n_units = len(self.units)
        for _ in range(self.n_draws):
            u = self.units[int(rng.randint(0, n_units))]
            opts = self.by_unit[u]
            yield int(opts[int(rng.randint(0, len(opts)))])

    @staticmethod
    def draw_stats(n_eligible: int, drawn: list[int]) -> dict[str, int]:
        """Per-epoch replacement diagnostics from an already-yielded index stream."""
        return {
            "n_eligible_windows": int(n_eligible),
            "n_gradient_draws": int(len(drawn)),
            "n_unique_sampled_windows": int(len(set(drawn))),
        }


def _collate(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)
    return {
        "x": x,
        "target": torch.stack([b["target"] for b in batch]),
        "duration": torch.stack([b["duration"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
        "unit_id": [b["unit_id"] for b in batch],
        "window_index": [int(b["window_index"]) for b in batch],
    }


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


SMOKE_MAX_WINDOWS_PER_UNIT = 32
SMOKE_MAX_EPOCHS = 5
HIST_COLS = [
    "epoch",
    "train_loss",
    "val_loss",
    "train_metric",
    "val_metric",
    "val_mae_events",
    "n_val_event_units",
    "n_eligible_windows",
    "n_gradient_draws",
    "n_unique_sampled_windows",
]


def new_run_id(dataset_id: str, architecture: str, smoke: bool) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    prefix = "smoke_" if smoke else ""
    return f"{prefix}{dataset_id}_{architecture}_{stamp}_{short}"


def resolve_max_windows_per_unit(
    value: object | None,
    *,
    smoke: bool,
    smoke_cap: int = SMOKE_MAX_WINDOWS_PER_UNIT,
) -> int | None:
    """None/omitted → smoke cap when smoke, else unlimited. Explicit 0 → unlimited."""
    if value is None or value == "":
        return int(smoke_cap) if smoke else None
    n = int(value)
    if n <= 0:
        return None
    return n


def _coerce_saved_windows(value: object | None) -> int | None:
    """Saved cap as stored: 0 / empty / null → unlimited. Does not apply a smoke preset."""
    if value is None or value == "":
        return None
    n = int(value)
    return None if n <= 0 else n


def load_saved_train_settings(run_path: Path, run_id: str = "") -> dict[str, Any] | None:
    """Read smoke / windows / epochs from an existing run. None if the dir is empty."""
    run_path = Path(run_path)
    cfg_path = run_path / "config.yaml"
    status_path = run_path / "status.json"
    saved: dict[str, Any] = {}
    if cfg_path.exists():
        raw = load_yaml(cfg_path)
        if "smoke" in raw:
            saved["smoke"] = bool(raw["smoke"])
        if "max_windows_per_unit" in raw:
            saved["max_windows_per_unit"] = _coerce_saved_windows(raw.get("max_windows_per_unit"))
        model = raw.get("model") or {}
        if model.get("max_epochs") is not None:
            saved["max_epochs"] = int(model["max_epochs"])
    if status_path.exists() and ("smoke" not in saved or "max_windows_per_unit" not in saved or "max_epochs" not in saved):
        try:
            st = read_json(status_path)
        except Exception:
            st = {}
        if "smoke" not in saved and "smoke" in st:
            saved["smoke"] = bool(st["smoke"])
        if "max_windows_per_unit" not in saved and "max_windows_per_unit" in st:
            saved["max_windows_per_unit"] = _coerce_saved_windows(st.get("max_windows_per_unit"))
        if "max_epochs" not in saved and st.get("max_epochs") is not None:
            saved["max_epochs"] = int(st["max_epochs"])
    if not saved and not (run_path / "last.pt").exists() and not cfg_path.exists() and not status_path.exists():
        return None
    if "smoke" not in saved:
        saved["smoke"] = str(run_id or run_path.name).startswith("smoke_")
    if "max_windows_per_unit" not in saved:
        saved["max_windows_per_unit"] = resolve_max_windows_per_unit(None, smoke=bool(saved["smoke"]))
    return saved


def resolve_run_train_args(
    *,
    resume_settings: dict[str, Any] | None,
    smoke: bool,
    max_windows_per_unit: object | None,
    max_epochs: int | None,
    smoke_cap: int = SMOKE_MAX_WINDOWS_PER_UNIT,
    smoke_epochs: int = SMOKE_MAX_EPOCHS,
    dataset_max_epochs: int = 30,
) -> dict[str, Any]:
    """Final smoke / windows / epochs. A resume snapshot beats form or CLI defaults."""
    if resume_settings is not None:
        smoke_r = bool(resume_settings.get("smoke", False))
        if "max_windows_per_unit" in resume_settings:
            mw = _coerce_saved_windows(resume_settings.get("max_windows_per_unit"))
        else:
            mw = resolve_max_windows_per_unit(None, smoke=smoke_r, smoke_cap=smoke_cap)
        if resume_settings.get("max_epochs") is not None:
            epochs = int(resume_settings["max_epochs"])
        else:
            epochs = int(dataset_max_epochs if max_epochs is None else max_epochs)
            if smoke_r:
                epochs = min(epochs, int(smoke_epochs))
        return {"smoke": smoke_r, "max_windows_per_unit": mw, "max_epochs": epochs}
    epochs = int(dataset_max_epochs if max_epochs is None else max_epochs)
    if smoke:
        epochs = min(epochs, int(smoke_epochs))
    mw = resolve_max_windows_per_unit(max_windows_per_unit, smoke=smoke, smoke_cap=smoke_cap)
    return {"smoke": bool(smoke), "max_windows_per_unit": mw, "max_epochs": epochs}


def next_max_windows_on_mode_change(
    *,
    prev_smoke: bool | None,
    smoke: bool,
    current_cap: int,
    smoke_cap: int = SMOKE_MAX_WINDOWS_PER_UNIT,
) -> int:
    """Drop the smoke preset when turning Full unless the user overrode it."""
    current = int(current_cap)
    cap = int(smoke_cap)
    if prev_smoke is None:
        return cap if smoke else 0
    if bool(prev_smoke) == bool(smoke):
        return current
    old_preset = cap if prev_smoke else 0
    if current == old_preset:
        return cap if smoke else 0
    return current


def training_mode_label(smoke: object, run_id: str = "") -> str:
    if smoke is True:
        return "Smoke"
    if smoke is False:
        return "Full"
    if isinstance(smoke, str):
        low = smoke.strip().lower()
        if low in {"true", "smoke"}:
            return "Smoke"
        if low in {"false", "full"}:
            return "Full"
    return "Smoke" if str(run_id).startswith("smoke_") else "Full"


def selection_metric_spec(dataset_id: str) -> dict[str, str]:
    if dataset_id == "bearings":
        return {"name": "val MAE", "unit": "seconds", "label": "val MAE (seconds)"}
    return {"name": "val NLL", "unit": "NLL", "label": "val NLL"}


def windows_per_unit_summary(dataset: UnitWindowDataset) -> dict[str, float | int]:
    counts: dict[str, int] = defaultdict(int)
    for item in dataset.index:
        counts[str(item[0])] += 1
    vals = list(counts.values())
    if not vals:
        return {"n_units": 0, "n_windows": 0, "min": 0, "max": 0, "mean": 0.0}
    return {
        "n_units": len(vals),
        "n_windows": int(sum(vals)),
        "min": int(min(vals)),
        "max": int(max(vals)),
        "mean": float(np.mean(vals)),
    }


def _history_columns(path: Path) -> list[str]:
    if not path.exists() or path.stat().st_size == 0:
        path.write_text(",".join(HIST_COLS) + "\n", encoding="utf-8")
        return list(HIST_COLS)
    header = path.read_text(encoding="utf-8").splitlines()[0]
    cols = [c.strip() for c in header.split(",") if c.strip()]
    return cols or list(HIST_COLS)


def compatibility_dict(
    cfg_model: dict,
    prep: Preprocessor,
    split: dict,
    dataset_id: str,
    head: str,
    fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "dataset_id": dataset_id,
        "architecture": cfg_model["architecture"],
        "history_length": int(cfg_model["history_length"]),
        "hidden_size": int(cfg_model["hidden_size"]),
        "recurrent_layers": int(cfg_model["recurrent_layers"]),
        "head": head,
        "feature_names": list(prep.feature_names),
        "time_scale_s": prep.time_scale_s,
        "split_hash": split_hash(split),
        "feature_pipeline_version": prep.feature_pipeline_version,
        "categorical_maps_fingerprint": categorical_maps_fingerprint(prep.categorical_maps),
    }
    fp = fingerprint or {}
    for k in ("dataset_version", "features_hash", "units_hash"):
        if fp.get(k) is not None:
            out[k] = fp[k]
    return out


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
        "feature_pipeline_version",
        "categorical_maps_fingerprint",
    ]
    for k in keys:
        if saved.get(k) != current.get(k):
            return False
    for k in ("dataset_version", "features_hash", "units_hash"):
        if k in saved and k in current and saved.get(k) != current.get(k):
            return False
    return resolve_split_hash(saved) == resolve_split_hash(current)


def _load_saved_run_task_config(rdir: Path) -> dict[str, Any] | None:
    """Experiment snapshot first, else run `config.yaml`. None if the run is legacy."""
    from pdm.evaluate import load_experiment_snapshot

    snap = load_experiment_snapshot(rdir)
    if snap:
        nested = snap.get("config")
        if isinstance(nested, dict):
            merged = dict(nested)
            for key, value in snap.items():
                if key != "config":
                    merged[key] = value
            return merged
        return dict(snap)
    cfg_path = Path(rdir) / "config.yaml"
    if cfg_path.exists():
        return load_yaml(cfg_path)
    return None


def resume_task_mismatches(
    live_cfg: dict[str, Any],
    live_mcfg: dict[str, Any],
    saved_cfg: dict[str, Any],
) -> list[str]:
    """Live YAML/CLI vs saved snapshot for gap / history_length / architecture."""
    differing: list[str] = []
    saved_model = dict(saved_cfg.get("model") or {})
    saved_arch = str(saved_model.get("architecture") or "").strip().lower()
    live_arch = str(live_mcfg.get("architecture") or "").strip().lower()
    if saved_arch and live_arch and saved_arch != live_arch:
        differing.append("architecture")
    if saved_model.get("history_length") is not None and live_mcfg.get("history_length") is not None:
        if int(saved_model["history_length"]) != int(live_mcfg["history_length"]):
            differing.append("history_length")
    saved_gap = dict(saved_cfg.get("gap") or {})
    live_gap = dict(live_cfg.get("gap") or {})
    if any(k in saved_gap or k in live_gap for k in ("gap_multiplier", "sampling_interval_s")):
        live_k, live_s = filter_gap_params(live_cfg)
        saved_k, saved_s = filter_gap_params(saved_cfg)
        if not np.isclose(float(live_k), float(saved_k), rtol=0.0, atol=1e-9):
            differing.append("gap_multiplier")
        if not np.isclose(float(live_s), float(saved_s), rtol=0.0, atol=1e-9):
            differing.append("sampling_interval_s")
    return differing


def _apply_saved_model_to_mcfg(mcfg: dict[str, Any], saved_cfg: dict[str, Any]) -> None:
    saved_model = dict(saved_cfg.get("model") or {})
    if saved_model.get("architecture"):
        mcfg["architecture"] = str(saved_model["architecture"]).lower()
    if saved_model.get("history_length") is not None:
        mcfg["history_length"] = int(saved_model["history_length"])


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
    if history_length is not None:
        mcfg["history_length"] = int(history_length)
    if seed is not None:
        mcfg["seed"] = int(seed)
    smoke_cap = int(mcfg.get("smoke_max_windows_per_unit", SMOKE_MAX_WINDOWS_PER_UNIT))
    smoke_epochs = int(mcfg.get("smoke_max_epochs", SMOKE_MAX_EPOCHS))
    resume_settings = None
    if resume_run_id:
        resume_settings = load_saved_train_settings(dataset_runs(dataset_id) / resume_run_id, resume_run_id)
    launch = resolve_run_train_args(
        resume_settings=resume_settings,
        smoke=smoke,
        max_windows_per_unit=max_windows_per_unit,
        max_epochs=max_epochs,
        smoke_cap=smoke_cap,
        smoke_epochs=smoke_epochs,
        dataset_max_epochs=int(mcfg["max_epochs"]),
    )
    smoke = bool(launch["smoke"])
    max_windows_per_unit = launch["max_windows_per_unit"]
    mcfg["max_epochs"] = int(launch["max_epochs"])
    if smoke:
        _log("Smoke test — not a quality benchmark")
    if resume_settings is not None:
        _log(
            f"Resume keeps saved {training_mode_label(smoke)} "
            f"max_epochs={mcfg['max_epochs']} "
            f"max_windows_per_unit={max_windows_per_unit if max_windows_per_unit is not None else 'all'}"
        )

    processed = load_processed(dataset_id)
    from pdm.evaluate import IncompatibleDataError, assert_gap_rule_current

    assert_gap_rule_current(processed.get("fingerprint"))
    features = processed["features"]
    units = processed["units"]
    split = processed["split"]
    head = "rul" if dataset_id == "bearings" else "weibull"

    if resume_run_id:
        rdir_resume = dataset_runs(dataset_id) / resume_run_id
        saved_task = _load_saved_run_task_config(rdir_resume)
        if saved_task:
            task_diff = resume_task_mismatches(cfg, mcfg, saved_task)
            if task_diff:
                raise IncompatibleDataError(
                    task_diff,
                    detail=(
                        "Live YAML/CLI gap, history_length, or architecture disagrees with the "
                        "run snapshot; not rebuilding windows from live YAML"
                    ),
                )
            cfg = saved_task
            _apply_saved_model_to_mcfg(mcfg, saved_task)
            mcfg["max_epochs"] = int(launch["max_epochs"])

    hist = int(mcfg["history_length"])
    set_seeds(int(mcfg["seed"]))

    gap_kw: dict[str, float] = {}
    if dataset_id == "filters":
        k, samp = filter_gap_params(cfg)
        gap_kw = {"gap_multiplier": k, "sampling_interval_s": samp}
    windows = build_windows(features, units, hist, dataset_id, **gap_kw)
    train_w = windows[windows["unit_id"].isin(split["train"])]
    val_w = windows[windows["unit_id"].isin(split["validation"])]
    if train_w.empty:
        raise RuntimeError("No training windows. Reduce history_length or prepare data.")

    rewrite_preprocessing = True
    saved_prep_path = dataset_runs(dataset_id) / resume_run_id / "preprocessing.json" if resume_run_id else None
    if saved_prep_path is not None and saved_prep_path.exists():
        saved_prep = Preprocessor.from_dict(read_json(saved_prep_path))
        live_prep, _live_feat = fit_preprocessor(dataset_id, features, units, split, cfg, train_w)
        mismatches = preprocessor_resume_mismatches(saved_prep, live_prep)
        if mismatches:
            raise IncompatibleDataError(
                mismatches,
                detail=(
                    "Saved preprocessing.json disagrees with a live fit from current YAML/data "
                    "on scaler/maps/time_scale_s/feature_pipeline_version; not replacing"
                ),
            )
        prep = saved_prep
        feat_t = apply_preprocessor(prep, features, dataset_id=dataset_id)
        rewrite_preprocessing = False
        current_prep = live_prep
    else:
        prep, feat_t = fit_preprocessor(dataset_id, features, units, split, cfg, train_w)
        current_prep = prep
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
        current = compatibility_dict(
            mcfg,
            current_prep,
            split,
            dataset_id,
            head,
            fingerprint=dataset_fingerprint_for_run(processed, split),
        )
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
    train_wpu = windows_per_unit_summary(train_ds)
    val_wpu = windows_per_unit_summary(val_ds)
    n_train_windows = int(train_wpu["n_windows"])
    n_val_windows = int(val_wpu["n_windows"])
    _log(
        f"windows train={n_train_windows} val={n_val_windows} "
        f"max_windows_per_unit={max_windows_per_unit if max_windows_per_unit is not None else 'all'}"
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
    # Fixed-mask diagnostics: every kept window once, then equal-weight units.
    train_diag_loader = DataLoader(
        train_ds,
        batch_size=int(mcfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(mcfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate,
    )
    sel_spec = selection_metric_spec(dataset_id)

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
    run_cfg: dict[str, Any] = {
        "dataset_id": dataset_id,
        "model": mcfg,
        "smoke": smoke,
        "mode": training_mode_label(smoke),
        "head": head,
        "max_windows_per_unit": max_windows_per_unit,
        "n_train_windows": n_train_windows,
        "n_val_windows": n_val_windows,
        "train_windows_per_unit": train_wpu,
        "val_windows_per_unit": val_wpu,
        "selection_metric": sel_spec,
    }
    if dataset_id == "filters":
        run_cfg["gap"] = {
            "gap_multiplier": prep.gap_multiplier,
            "sampling_interval_s": prep.sampling_interval_s,
        }
    if not resume_run_id or not (rdir / "config.yaml").exists():
        dump_yaml(rdir / "config.yaml", run_cfg)
    atomic_write_json(rdir / "split.json", split)
    run_fp = dataset_fingerprint_for_run(processed, split)
    fp_path = rdir / "dataset_fingerprint.json"
    if not fp_path.exists():
        atomic_write_json(fp_path, run_fp)
    else:
        run_fp = json.loads(fp_path.read_text(encoding="utf-8"))
    atomic_write_json(rdir / "feature_schema.json", {"feature_names": prep.feature_names, "order": prep.feature_names})
    if rewrite_preprocessing:
        atomic_write_json(rdir / "preprocessing.json", prep.to_dict())
    if not resume_run_id:
        atomic_write_json(rdir / "experiment_snapshot.json", _experiment_snapshot(cfg, mcfg, prep, rdir, dataset_id))
    atomic_write_json(rdir / "environment.json", env)
    manifest_src = project_root() / "data" / "manifest.json"
    if manifest_src.exists():
        atomic_write_json(rdir / "data_manifest.json", json.loads(manifest_src.read_text()))
    log_path = rdir / "train.log"
    hist_path = rdir / "training_history.csv"
    hist_cols = _history_columns(hist_path)

    def emit(status: str, **extra):
        payload = {
            "status": status,
            "dataset_id": dataset_id,
            "run_id": run_id,
            "architecture": mcfg["architecture"],
            "head": head,
            "smoke": smoke,
            "mode": training_mode_label(smoke),
            "device": device_info.name,
            "n_train_windows": n_train_windows,
            "n_val_windows": n_val_windows,
            "max_windows_per_unit": max_windows_per_unit,
            "train_windows_per_unit": train_wpu,
            "val_windows_per_unit": val_wpu,
            "selection_metric_name": sel_spec["name"],
            "selection_metric_unit": sel_spec["unit"],
            "selection_metric_label": sel_spec["label"],
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
            sampler.set_epoch(epoch)
            sampled_idx: list[int] = []
            tr_loss = _run_epoch(
                model,
                opt,
                train_loader,
                dataset_id,
                device,
                mcfg,
                train=True,
                sampled_indices=sampled_idx,
            )
            samp_diag = UnitBalancedSampler.draw_stats(len(train_ds), sampled_idx)
            n_eligible_windows = int(samp_diag["n_eligible_windows"])
            n_gradient_draws = int(samp_diag["n_gradient_draws"])
            n_unique_sampled_windows = int(samp_diag["n_unique_sampled_windows"])
            train_stats = _eval_unit_weighted(model, train_diag_loader, dataset_id, device)
            val_stats = _eval_unit_weighted(model, val_loader, dataset_id, device)
            val_metric = val_stats["selection_metric"]
            train_metric = train_stats["selection_metric"]
            improved = val_metric < best_metric - 1e-8
            if improved:
                best_metric = val_metric
                best_epoch = epoch
                bad = 0
                _save_ckpt(
                    rdir / "best.pt",
                    model,
                    opt,
                    epoch,
                    best_epoch,
                    best_metric,
                    mcfg,
                    prep,
                    split,
                    dataset_id,
                    head,
                    smoke,
                    fingerprint=run_fp,
                )
            else:
                bad += 1
            _save_ckpt(
                rdir / "last.pt",
                model,
                opt,
                epoch,
                best_epoch,
                best_metric,
                mcfg,
                prep,
                split,
                dataset_id,
                head,
                smoke,
                fingerprint=run_fp,
            )
            _sync_run_checkpoint_hash(rdir)
            hist_row = {
                "epoch": epoch,
                "train_loss": f"{tr_loss:.6f}",
                "val_loss": f"{val_stats['val_loss']:.6f}",
                "train_metric": f"{train_metric:.6f}",
                "val_metric": f"{val_metric:.6f}",
                "val_mae_events": val_stats.get("val_mae_events"),
                "n_val_event_units": val_stats.get("n_val_event_units"),
                "n_eligible_windows": n_eligible_windows,
                "n_gradient_draws": n_gradient_draws,
                "n_unique_sampled_windows": n_unique_sampled_windows,
            }
            append_line(
                hist_path,
                ",".join("" if hist_row.get(c) is None else str(hist_row.get(c, "")) for c in hist_cols),
            )
            msg = (
                f"epoch {epoch}/{mcfg['max_epochs']} train_loss={tr_loss:.4f} "
                f"val_loss={val_stats['val_loss']:.4f} train_metric={train_metric:.4f} "
                f"val_metric={val_metric:.4f} ({sel_spec['label']}) best_epoch={best_epoch} "
                f"sampled_unique={n_unique_sampled_windows}/{n_eligible_windows} "
                f"draws={n_gradient_draws}"
            )
            _log(msg)
            append_line(log_path, msg)
            emit(
                "training",
                epoch=epoch,
                max_epochs=int(mcfg["max_epochs"]),
                train_loss=tr_loss,
                val_loss=val_stats["val_loss"],
                train_metric=train_metric,
                val_metric=val_metric,
                best_epoch=best_epoch,
                best_metric=best_metric,
                n_eligible_windows=n_eligible_windows,
                n_gradient_draws=n_gradient_draws,
                n_unique_sampled_windows=n_unique_sampled_windows,
                message=msg,
            )
            atomic_write_json(
                rdir / "validation_metrics.json",
                {
                    "best_epoch": best_epoch,
                    "best_metric": best_metric,
                    "selection_metric_name": sel_spec["name"],
                    "selection_metric_unit": sel_spec["unit"],
                    "selection_metric_label": sel_spec["label"],
                    "last": val_stats,
                    "last_train": train_stats,
                    "n_train_windows": n_train_windows,
                    "n_val_windows": n_val_windows,
                    "max_windows_per_unit": max_windows_per_unit,
                    "train_windows_per_unit": train_wpu,
                    "val_windows_per_unit": val_wpu,
                    "smoke": smoke,
                    "mode": training_mode_label(smoke),
                },
            )
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


def _run_epoch(
    model,
    opt,
    loader,
    dataset_id,
    device,
    mcfg,
    train: bool,
    sampled_indices: list[int] | None = None,
) -> float:
    model.train(train)
    losses = []
    for batch in loader:
        if sampled_indices is not None:
            sampled_indices.extend(int(i) for i in batch["window_index"])
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
def _eval_unit_weighted(model, loader, dataset_id, device) -> dict[str, Any]:
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


def _sync_run_checkpoint_hash(rdir: Path) -> None:
    ckpt = rdir / "best.pt"
    if not ckpt.exists():
        ckpt = rdir / "last.pt"
    if not ckpt.exists():
        return
    fp_path = rdir / "dataset_fingerprint.json"
    if not fp_path.exists():
        return
    fp = json.loads(fp_path.read_text(encoding="utf-8"))
    fp["checkpoint_hash"] = checkpoint_hash(ckpt)
    atomic_write_json(fp_path, fp)


def _save_ckpt(
    path,
    model,
    opt,
    epoch,
    best_epoch,
    best_metric,
    mcfg,
    prep,
    split,
    dataset_id,
    head,
    smoke,
    fingerprint=None,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "compat": compatibility_dict(mcfg, prep, split, dataset_id, head, fingerprint=fingerprint),
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


def load_trained_model(
    run_path: Path,
    device: str = "cpu",
    which: str = "best",
    *,
    verify_checkpoint_hash: bool = True,
) -> tuple[PDMNet, Preprocessor, dict]:
    ckpt_file = run_path / f"{which}.pt"
    if not ckpt_file.exists():
        ckpt_file = run_path / "last.pt"
    blob = torch.load(ckpt_file, map_location=device, weights_only=False)
    meta = blob["meta"]
    _verify_loaded_checkpoint(
        run_path, ckpt_file, blob, verify_checkpoint_hash=verify_checkpoint_hash
    )
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


def _verify_loaded_checkpoint(
    run_path: Path,
    ckpt_file: Path,
    blob: dict,
    *,
    verify_checkpoint_hash: bool = True,
) -> None:
    """Abort if best.pt bytes or split identity drifted from the run snapshot."""
    from pdm.evaluate import IncompatibleDataError
    from pdm.io_util import read_json

    fp_path = run_path / "dataset_fingerprint.json"
    if verify_checkpoint_hash and fp_path.exists() and ckpt_file.name.startswith("best"):
        stored = read_json(fp_path).get("checkpoint_hash")
        if stored and stored != checkpoint_hash(ckpt_file):
            raise IncompatibleDataError(
                ["checkpoint_hash"],
                detail="best.pt does not match the run fingerprint",
            )
    split_path = run_path / "split.json"
    if split_path.exists():
        snap_hash = split_hash(read_json(split_path))
        compat_hash = resolve_split_hash(blob.get("compat") or {})
        if compat_hash and compat_hash != snap_hash:
            raise IncompatibleDataError(
                ["split_hash"],
                detail="checkpoint compat split_hash does not match run split.json",
            )


def _source_commit() -> str | None:
    """Best-effort `git rev-parse HEAD`. Never writes git config; null if unavailable."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(project_root()),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    sha = (proc.stdout or "").strip()
    return sha or None


def _experiment_snapshot(
    cfg: dict[str, Any],
    mcfg: dict[str, Any],
    prep: Preprocessor,
    rdir: Path,
    dataset_id: str,
) -> dict[str, Any]:
    from copy import deepcopy

    from pdm.evaluate import METRICS_VERSION
    from pdm.preprocessing import FEATURE_PIPELINE_VERSION
    from pdm.windows import GAP_RULE_VERSION

    snap = deepcopy(cfg)
    snap["dataset_id"] = dataset_id
    snap["model"] = dict(mcfg)
    if dataset_id == "filters":
        gap = dict(cfg.get("gap") or {})
        if prep.gap_multiplier is not None:
            gap["gap_multiplier"] = prep.gap_multiplier
        if prep.sampling_interval_s is not None:
            gap["sampling_interval_s"] = prep.sampling_interval_s
        snap["gap"] = gap
    snap["preprocessing_hash"] = sha256_file(rdir / "preprocessing.json")
    snap["source_commit"] = _source_commit()
    snap["metrics_version"] = METRICS_VERSION
    snap["gap_rule_version"] = GAP_RULE_VERSION
    snap["feature_pipeline_version"] = prep.feature_pipeline_version or FEATURE_PIPELINE_VERSION
    return snap


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
        "source_commit": _source_commit(),
    }
