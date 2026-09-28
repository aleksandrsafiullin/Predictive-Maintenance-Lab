"""Bearing health zones: green (normal) / yellow (degrading) / red (urgent review).

Retrospective labels come from each bearing's vibration history and the final
recorded sample, which is an experiment-end proxy rather than a confirmed failure:

* red    — the last ``red_minutes`` before the recording ends;
* yellow — from degradation onset: the first run of ``onset_persist`` consecutive
  measurements whose max(horizontal, vertical) RMS stays above
  ``max(median + 3*std, onset_ratio * median)`` of the first ``baseline_n``
  measurements (the bearing's own healthy baseline);
* green  — everything before onset.

The classifier is a GRU over the last ``history`` measurements. Inputs are causal:
log ratios of every vibration feature to the bearing's own baseline (the first
``baseline_n`` measurements, available ``baseline_n`` minutes after start) plus
operating condition. Scalers are fit on train units only. Model selection uses the
validation split only; test is reported separately.

Displayed zones pass through a hysteresis filter: escalation needs ``escalate_n``
consecutive higher predictions, de-escalation needs ``deescalate_n`` consecutive
lower ones. Laboratory result on 15 run-to-failure bearings, not a protection system.

Leave-bearings-out cross-validation over the 12 train+validation bearings (4 folds,
one bearing per operating condition held out) chose 5 epochs and the trend features;
longer training overfits. In that study the calibrated rule baseline
(``rule_baseline``: confirmed onset -> yellow, RMS >= ratio x baseline -> red, ratio
fitted on train bearings) scored as well as or better than every learned variant, so
the UI offers it as the recommended zone engine and the GRU for comparison.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from pdm.io_util import atomic_write_json, read_json
from pdm.paths import runs_root

ZONES = ("green", "yellow", "red")
ZONE_LABELS = {
    "green": "Normal",
    "yellow": "Something is not right",
    "red": "Urgent review",
}
ZONE_ACTIONS = {
    "green": "Continue normal operation and monitoring.",
    "yellow": "Degradation detected. Plan inspection and prepare a replacement.",
    "red": "Inspect promptly and follow site procedures for an operating decision.",
}
CHANNELS = ("horizontal", "vertical")
SIGNALS = ("rms", "std", "abs_peak", "peak_to_peak", "crest_factor", "kurtosis",
           "band_0", "band_1", "band_2", "band_3")
EPS = 1e-6

DEFAULT_CONFIG: dict[str, Any] = {
    "red_minutes": 30.0,
    "baseline_n": 5,
    "onset_ratio": 1.25,
    "onset_sigma": 3.0,
    "onset_persist": 5,
    "history": 20,
    "hidden_size": 32,
    "epochs": 5,
    "batch_size": 256,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "escalate_n": 2,
    "deescalate_n": 10,
    "seed": 42,
    "feature_set": "trend_v1",
}


def zones_root(dataset_id: str = "bearings") -> Path:
    return runs_root() / "_zones" / dataset_id


# ---------------------------------------------------------------- labels

def combined_rms(unit: pd.DataFrame) -> np.ndarray:
    return np.maximum(unit["horizontal_rms"].to_numpy(float), unit["vertical_rms"].to_numpy(float))


def onset_threshold(rms: np.ndarray, cfg: dict[str, Any]) -> float:
    base = rms[: int(cfg["baseline_n"])]
    med = float(np.median(base))
    return max(med + float(cfg["onset_sigma"]) * float(np.std(base)), float(cfg["onset_ratio"]) * med)


def onset_index(rms: np.ndarray, cfg: dict[str, Any]) -> int | None:
    """First index of a persistent rise above the healthy baseline, or None."""
    n0, k = int(cfg["baseline_n"]), int(cfg["onset_persist"])
    if len(rms) < n0 + k:
        return None
    above = rms > onset_threshold(rms, cfg)
    for i in range(n0, len(rms) - k + 1):
        if above[i:i + k].all():
            return i
    return None


def label_unit(unit: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Return unit rows sorted by time with rul_s, onset flag and true zone (0/1/2)."""
    u = unit.sort_values("timestamp_s").reset_index(drop=True)
    t = u["timestamp_s"].to_numpy(float)
    rul = t[-1] - t
    zone = np.zeros(len(u), dtype=int)
    on = onset_index(combined_rms(u), cfg)
    if on is not None:
        zone[on:] = 1
    zone[rul <= float(cfg["red_minutes"]) * 60.0] = 2
    return u.assign(rul_s=rul, true_zone=zone)


# -------------------------------------------------------------- features

TREND_FEATURES = ["log_ratio_rms_running_max", "log_ratio_rms_slope_10", "onset_detected", "log_minutes_since_onset"]


def feature_names(cfg: dict[str, Any] | None = None) -> list[str]:
    names = [f"log_ratio_{c}_{s}" for c in CHANNELS for s in SIGNALS] + ["rpm", "load_kn"]
    if (cfg or DEFAULT_CONFIG).get("feature_set", "trend_v1") == "trend_v1":
        names += TREND_FEATURES
    return names


def causal_onset_flags(rms: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
    """1 from the measurement at which an onset is *confirmed* (end of the persistent run)."""
    n0, k = int(cfg["baseline_n"]), int(cfg["onset_persist"])
    flags = np.zeros(len(rms))
    if len(rms) <= n0:
        return flags
    above = rms > onset_threshold(rms[:n0], cfg)
    for i in range(n0 + k - 1, len(rms)):
        if above[i - k + 1:i + 1].all():
            flags[i:] = 1.0
            break
    return flags


def unit_features(unit: pd.DataFrame, cfg: dict[str, Any]) -> np.ndarray:
    """Causal per-row features for one time-sorted unit.

    The baseline uses the first ``baseline_n`` rows. Rows before the baseline is
    complete use the rows seen so far, so no row ever depends on a later one.
    """
    n0 = int(cfg["baseline_n"])
    cols = [f"{c}_{s}" for c in CHANNELS for s in SIGNALS]
    raw = np.abs(unit[cols].to_numpy(float)) + EPS
    base = np.empty_like(raw)
    for i in range(len(raw)):
        base[i] = np.median(raw[: min(i + 1, n0)], axis=0)
    ratio = np.log(raw / base)
    ops = unit[["rpm", "load_kn"]].to_numpy(float)
    parts = [ratio, ops]
    if cfg.get("feature_set", "trend_v1") == "trend_v1":
        rms = combined_rms(unit)
        rms_ratio = np.log((rms + EPS) / (np.array([np.median(rms[: min(i + 1, n0)]) for i in range(len(rms))]) + EPS))
        running_max = np.maximum.accumulate(rms_ratio)
        slope = rms_ratio - np.concatenate([np.repeat(rms_ratio[:1], 10), rms_ratio])[: len(rms_ratio)]
        onset = causal_onset_flags(rms, cfg)
        t_min = (unit["timestamp_s"].to_numpy(float) - float(unit["timestamp_s"].iloc[0])) / 60.0
        first = np.flatnonzero(onset)
        since = np.where(onset > 0, t_min - (t_min[first[0]] if len(first) else 0.0), 0.0)
        parts.append(np.stack([running_max, slope, onset, np.log1p(since)], axis=1))
    return np.concatenate(parts, axis=1)


def windows(x: np.ndarray, history: int) -> np.ndarray:
    """(n, f) -> (n, history, f); early rows are left-padded with the first row."""
    pad = np.repeat(x[:1], history - 1, axis=0)
    xp = np.concatenate([pad, x], axis=0)
    idx = np.arange(len(x))[:, None] + np.arange(history)[None, :]
    return xp[idx]


class Scaler:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean, self.std = mean, std

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std < 1e-8, 1.0, std))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict[str, list[float]]) -> "Scaler":
        return cls(np.asarray(d["mean"], float), np.asarray(d["std"], float))


# ----------------------------------------------------------------- model

class ZoneGRU(nn.Module):
    def __init__(self, n_features: int, hidden_size: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, len(ZONES)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(x)
        return self.head(h[-1])


def smooth_zones(raw: np.ndarray, escalate_n: int, deescalate_n: int) -> np.ndarray:
    """Hysteresis: escalate after ``escalate_n`` consecutive higher predictions
    (to the lowest level among them), de-escalate after ``deescalate_n`` lower ones."""
    out = np.zeros(len(raw), dtype=int)
    cur, hi_run, lo_run = 0, [], []
    for i, z in enumerate(raw.astype(int)):
        hi_run = hi_run + [z] if z > cur else []
        lo_run = lo_run + [z] if z < cur else []
        if len(hi_run) >= escalate_n:
            cur, hi_run = min(hi_run[-escalate_n:]), []
        elif len(lo_run) >= deescalate_n:
            cur, lo_run = max(lo_run[-deescalate_n:]), []
        out[i] = cur
    return out


# ------------------------------------------------------------ data prep

def _prepared(features: pd.DataFrame, units: list[str], cfg: dict[str, Any]):
    rows, xs = [], []
    for uid in units:
        u = label_unit(features[features["unit_id"].astype(str) == str(uid)], cfg)
        if u.empty:
            continue
        rows.append(u[["unit_id", "timestamp_s", "rul_s", "true_zone"]].assign(
            combined_rms=combined_rms(u)))
        xs.append(unit_features(u, cfg))
    return rows, xs


def _sample_weights(rows: list[pd.DataFrame]) -> np.ndarray:
    """Each (unit, zone) pair present in train gets equal total weight."""
    w = []
    for r in rows:
        counts = r["true_zone"].value_counts()
        w.append(r["true_zone"].map(lambda z: 1.0 / counts[z]).to_numpy(float))
    w = np.concatenate(w)
    return w * (len(w) / w.sum())


def _predict(model: ZoneGRU, xw: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        out = [torch.softmax(model(torch.from_numpy(xw[i:i + 2048]).float()), dim=-1).numpy()
               for i in range(0, len(xw), 2048)]
    return np.concatenate(out) if out else np.zeros((0, len(ZONES)))


def _predict_units(model, scaler, rows, xs, cfg) -> pd.DataFrame:
    frames = []
    for r, x in zip(rows, xs):
        prob = _predict(model, windows(scaler(x), int(cfg["history"])).astype(np.float32))
        raw = prob.argmax(axis=1)
        frames.append(r.assign(
            raw_zone=raw,
            zone=smooth_zones(raw, int(cfg["escalate_n"]), int(cfg["deescalate_n"])),
            p_green=prob[:, 0], p_yellow=prob[:, 1], p_red=prob[:, 2]))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -------------------------------------------------------------- metrics

def zone_metrics(pred: pd.DataFrame, column: str = "zone", red_minutes: float = 30.0) -> dict[str, Any]:
    """Unit-balanced zone metrics plus red-alarm timing for each bearing."""
    per_unit = {}
    for uid, g in pred.groupby("unit_id", sort=True):
        t, z, p = g["true_zone"].to_numpy(), g[column].to_numpy(), g["rul_s"].to_numpy()
        recalls = {ZONES[k]: float((z[t == k] == k).mean()) for k in range(3) if (t == k).any()}
        red_idx = np.flatnonzero(z == 2)
        first_red_min = float(p[red_idx[0]] / 60.0) if len(red_idx) else None
        early = first_red_min is not None and first_red_min > 2 * red_minutes
        per_unit[str(uid)] = {
            "n": int(len(g)),
            "life_min": float(p[0] / 60.0),
            "accuracy": float((z == t).mean()),
            "balanced_accuracy": float(np.mean(list(recalls.values()))),
            "recall": recalls,
            "first_red_min_before_failure": first_red_min,
            "red_raised": first_red_min is not None,
            "red_too_early": bool(early),
            "minutes_green_while_red": int(((z == 0) & (t == 2)).sum()),
            "minutes_red_while_green": int(((z == 2) & (t == 0)).sum()),
        }
    units = list(per_unit.values())
    conf = np.zeros((3, 3), dtype=int)
    for a, b in zip(pred["true_zone"], pred[column]):
        conf[int(a), int(b)] += 1

    def mean(key):
        vals = [u[key] for u in units if u[key] is not None]
        return float(np.mean(vals)) if vals else None

    def zone_recall(k):
        vals = [u["recall"][k] for u in units if k in u["recall"]]
        return float(np.mean(vals)) if vals else None

    return {
        "n_units": len(units),
        "unit_balanced_accuracy": mean("accuracy"),
        "unit_balanced_balanced_accuracy": mean("balanced_accuracy"),
        "unit_balanced_recall": {k: zone_recall(k) for k in ZONES},
        "units_red_raised": int(sum(u["red_raised"] for u in units)),
        "units_red_too_early": int(sum(u["red_too_early"] for u in units)),
        "median_first_red_min_before_failure": (
            float(np.median([u["first_red_min_before_failure"] for u in units if u["red_raised"]]))
            if any(u["red_raised"] for u in units) else None),
        "confusion_true_rows_pred_cols": conf.tolist(),
        "per_unit": per_unit,
    }


# ------------------------------------------------------- rule baseline

def rule_baseline(features: pd.DataFrame, units: list[str], cfg: dict[str, Any], red_ratio: float) -> pd.DataFrame:
    """Causal rules without learning: yellow after a confirmed onset (latched),
    red when combined RMS reaches ``red_ratio`` x baseline median (latched)."""
    frames = []
    k, n0 = int(cfg["onset_persist"]), int(cfg["baseline_n"])
    for uid in units:
        u = label_unit(features[features["unit_id"].astype(str) == str(uid)], cfg)
        rms = combined_rms(u)
        z = np.zeros(len(u), dtype=int)
        for i in range(len(u)):
            if i < n0:
                continue
            base = rms[:n0]
            thr = max(np.median(base) + cfg["onset_sigma"] * np.std(base), cfg["onset_ratio"] * np.median(base))
            level = z[i - 1]
            if i >= n0 + k - 1 and (rms[i - k + 1:i + 1] > thr).all():
                level = max(level, 1)
            if rms[i] >= red_ratio * np.median(base):
                level = 2
            z[i] = level
        frames.append(u[["unit_id", "timestamp_s", "rul_s", "true_zone"]].assign(combined_rms=rms, zone=z))
    return pd.concat(frames, ignore_index=True)


def fit_rule_red_ratio(features, train_units, cfg) -> float:
    best, best_score = 3.0, -1.0
    for ratio in np.round(np.arange(1.5, 8.01, 0.25), 2):
        score = zone_metrics(rule_baseline(features, train_units, cfg, float(ratio)),
                             red_minutes=cfg["red_minutes"])["unit_balanced_balanced_accuracy"]
        if score > best_score:
            best, best_score = float(ratio), score
    return best


# -------------------------------------------------------------- training

def train_zone_model(
    dataset_id: str = "bearings",
    config: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    from pdm.data.prepare import load_processed

    if dataset_id != "bearings":
        raise ValueError("Health zones are defined for bearings (run-to-failure vibration data)")
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    data = load_processed(dataset_id)
    feats, split = data["features"], data["split"]
    tr_rows, tr_x = _prepared(feats, split["train"], cfg)
    va_rows, va_x = _prepared(feats, split["validation"], cfg)
    te_rows, te_x = _prepared(feats, split["test"], cfg)

    scaler = Scaler.fit(np.concatenate(tr_x))
    h = int(cfg["history"])
    xw = np.concatenate([windows(scaler(x), h) for x in tr_x]).astype(np.float32)
    y = np.concatenate([r["true_zone"].to_numpy() for r in tr_rows])
    w = _sample_weights(tr_rows).astype(np.float32)

    model = ZoneGRU(xw.shape[-1], int(cfg["hidden_size"]), float(cfg["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                            weight_decay=float(cfg["weight_decay"]))
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    xt, yt, wt = torch.from_numpy(xw), torch.from_numpy(y).long(), torch.from_numpy(w)
    gen = torch.Generator().manual_seed(int(cfg["seed"]))
    best = {"score": -1.0, "epoch": 0, "state": None}
    history = []
    bs = int(cfg["batch_size"])
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        order = torch.randperm(len(xt), generator=gen)
        total = 0.0
        for i in range(0, len(order), bs):
            idx = order[i:i + bs]
            loss = (loss_fn(model(xt[idx]), yt[idx]) * wt[idx]).sum() / wt[idx].sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        val = zone_metrics(_predict_units(model, scaler, va_rows, va_x, cfg), red_minutes=cfg["red_minutes"])
        score = val["unit_balanced_balanced_accuracy"]
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_balanced_accuracy": score})
        if score > best["score"]:
            best = {"score": score, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if epoch == 1 or epoch % 10 == 0:
            log(f"epoch {epoch}/{cfg['epochs']} train_loss={total / len(order):.4f} "
                f"val_balanced_acc={score:.4f} best_epoch={best['epoch']}")
    model.load_state_dict(best["state"])

    run_id = time.strftime("zones_%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    rdir = zones_root(dataset_id) / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), rdir / "model.pt")

    red_ratio = fit_rule_red_ratio(feats, split["train"], cfg)
    metrics: dict[str, Any] = {"selection": "validation unit-balanced balanced accuracy (smoothed zones)",
                               "best_epoch": best["epoch"], "rule_baseline_red_ratio": red_ratio}
    for name, rows, xs in (("validation", va_rows, va_x), ("test", te_rows, te_x)):
        pred = _predict_units(model, scaler, rows, xs, cfg)
        pred.to_csv(rdir / f"predictions_{name}.csv", index=False)
        rule = rule_baseline(feats, split[name], cfg, red_ratio)
        metrics[name] = {
            "model": zone_metrics(pred, red_minutes=cfg["red_minutes"]),
            "model_raw": zone_metrics(pred, "raw_zone", red_minutes=cfg["red_minutes"]),
            "rule_baseline": zone_metrics(rule, red_minutes=cfg["red_minutes"]),
        }
    meta = {
        "run_id": run_id, "dataset_id": dataset_id, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": data.get("dataset_version"), "config": cfg, "feature_names": feature_names(cfg),
        "scaler": scaler.to_dict(), "split": {k: list(split[k]) for k in ("train", "validation", "test")},
        "zones": list(ZONES), "history": history,
    }
    atomic_write_json(rdir / "meta.json", meta)
    atomic_write_json(rdir / "metrics.json", metrics)
    return {"run_id": run_id, "dir": str(rdir), "best_epoch": best["epoch"], "val_score": best["score"]}


# ------------------------------------------------------------ inference

def list_zone_runs(dataset_id: str = "bearings") -> list[dict[str, Any]]:
    root = zones_root(dataset_id)
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir(), reverse=True):
        if (d / "meta.json").exists() and (d / "model.pt").exists():
            out.append({"run_id": d.name, "dir": str(d)})
    return out


def load_zone_run(rdir: Path) -> tuple[ZoneGRU, Scaler, dict[str, Any], dict[str, Any]]:
    meta = read_json(rdir / "meta.json")
    cfg = meta["config"]
    model = ZoneGRU(len(meta["feature_names"]), int(cfg["hidden_size"]), float(cfg["dropout"]))
    model.load_state_dict(torch.load(rdir / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    metrics = read_json(rdir / "metrics.json") if (rdir / "metrics.json").exists() else {}
    return model, Scaler.from_dict(meta["scaler"]), meta, metrics


def predict_unit(model: ZoneGRU, scaler: Scaler, cfg: dict[str, Any], unit: pd.DataFrame) -> pd.DataFrame:
    """Zone predictions for every row of one unit (each row uses only earlier rows)."""
    rows, xs = [], []
    u = label_unit(unit, cfg)
    rows.append(u[["unit_id", "timestamp_s", "rul_s", "true_zone"]].assign(combined_rms=combined_rms(u)))
    xs.append(unit_features(u, cfg))
    return _predict_units(model, scaler, rows, xs, cfg)


def summary_json(metrics: dict[str, Any]) -> str:
    keep = {}
    for split in ("validation", "test"):
        if split in metrics:
            keep[split] = {k: {m: v for m, v in metrics[split][k].items() if m != "per_unit"}
                           for k in ("model", "rule_baseline")}
    return json.dumps(keep, indent=2)
