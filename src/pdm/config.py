from __future__ import annotations

from copy import deepcopy
from typing import Any

from pdm.io_util import load_yaml
from pdm.paths import configs_root


def load_dataset_config(dataset_id: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    path = configs_root() / f"{dataset_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing config {path}")
    cfg = load_yaml(path)
    if overrides:
        cfg = deepcopy(cfg)
        _deep_update(cfg, overrides)
    return cfg


def _deep_update(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


def model_defaults(cfg: dict[str, Any]) -> dict[str, Any]:
    m = dict(cfg.get("model") or {})
    m.setdefault("architecture", "gru")
    m.setdefault("history_length", 20)
    m.setdefault("hidden_size", 64)
    m.setdefault("recurrent_layers", 1)
    m.setdefault("dropout", 0.1)
    m.setdefault("batch_size", 32)
    m.setdefault("optimizer", "adamw")
    m.setdefault("learning_rate", 0.001)
    m.setdefault("weight_decay", 0.0001)
    m.setdefault("max_epochs", 30)
    m.setdefault("early_stopping_patience", 5)
    m.setdefault("grad_clip", 1.0)
    m.setdefault("seed", 42)
    m.setdefault("num_workers", 0)
    m.setdefault("smoke_max_epochs", 5)
    m.setdefault("smoke_max_windows_per_unit", 32)
    reservoir = dict(m.get("reservoir") or {})
    reservoir.setdefault("n_nodes", 1000)
    reservoir.setdefault("leak", 0.2)
    reservoir.setdefault("spectral_radius", 0.9)
    reservoir.setdefault("input_scale", 0.1)
    reservoir.setdefault("ridge_alpha", 0.001)
    reservoir.setdefault("seed", 42)
    reservoir.setdefault("state_mode", "window_reset")
    reservoir.setdefault("graph_mode", "synthetic_fixture")
    dataset_id = str(cfg.get("dataset_id") or "").strip().lower()
    explicit_readout = reservoir.get("readout")
    readout_norm = str(explicit_readout).strip().lower() if explicit_readout is not None else None
    if dataset_id == "filters" and readout_norm == "ridge":
        raise ValueError(
            "readout=ridge is not allowed for filters; censored units require a gradient Weibull readout"
        )
    if dataset_id == "filters":
        reservoir["readout"] = readout_norm or "gradient"
    elif dataset_id == "bearings":
        reservoir.setdefault("readout", "ridge")
    m["reservoir"] = reservoir
    return m
