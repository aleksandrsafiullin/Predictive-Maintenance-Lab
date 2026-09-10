from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def assert_disjoint_splits(split: dict[str, list[str]]) -> None:
    sets = {k: set(v) for k, v in split.items() if k in {"train", "validation", "test"}}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        inter = sets.get(a, set()) & sets.get(b, set())
        if inter:
            raise ValueError(f"Split leak: {a} ∩ {b} = {sorted(inter)}")


def bearings_split(units: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict:
    cfg = cfg or {}
    spec = cfg.get("split") or {}
    train_i = set(spec.get("train_instances") or [1, 2, 3])
    val_i = set(spec.get("val_instances") or [4])
    test_i = set(spec.get("test_instances") or [5])
    mapping = {"train": [], "validation": [], "test": []}
    unknown = []
    for _, row in units.iterrows():
        inst = int(row["instance"])
        uid = str(row["unit_id"])
        if inst in train_i:
            mapping["train"].append(uid)
        elif inst in val_i:
            mapping["validation"].append(uid)
        elif inst in test_i:
            mapping["test"].append(uid)
        else:
            unknown.append(uid)
    for k in mapping:
        mapping[k] = sorted(mapping[k])
    assert_disjoint_splits(mapping)
    return {
        "dataset_id": "bearings",
        "protocol": "per_regime_instances_1-3_train_4_val_5_test",
        "train": mapping["train"],
        "validation": mapping["validation"],
        "test": mapping["test"],
        "unassigned": unknown,
        "n_train": len(mapping["train"]),
        "n_validation": len(mapping["validation"]),
        "n_test": len(mapping["test"]),
    }


def filters_split(units: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict:
    """Keep author test units untouched. Split author train 80/20 by unit, seed 42."""
    from sklearn.model_selection import StratifiedShuffleSplit

    cfg = cfg or {}
    spec = cfg.get("split") or {}
    seed = int(spec.get("seed", 42))
    val_frac = float(spec.get("val_fraction", 0.2))
    author_train = units[units["author_split"] == "author_train"].copy()
    author_test = units[units["author_split"] == "author_test"].copy()
    test_ids = sorted(author_test["unit_id"].astype(str).tolist())
    train_pool = author_train.copy()
    events = train_pool["event_observed"].astype(int).to_numpy()
    ids = train_pool["unit_id"].astype(str).to_numpy()
    warning = None
    n_events = int(events.sum())
    if n_events < 4:
        warning = (
            f"Only {n_events} observed 600 Pa events in the author training units. "
            "Validation MAE on events will be weak; test RUL is not used for fitting."
        )
    if len(ids) < 2:
        raise ValueError("Not enough author training units to split.")
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        tr_idx, va_idx = next(sss.split(ids, events))
    except ValueError:
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(ids))
        n_val = max(1, int(round(val_frac * len(ids))))
        va_idx, tr_idx = order[:n_val], order[n_val:]
        warning = (warning or "") + " Stratified split failed; used random unit split."
    train_ids = sorted(ids[tr_idx].tolist())
    val_ids = sorted(ids[va_idx].tolist())
    mapping = {"train": train_ids, "validation": val_ids, "test": test_ids}
    assert_disjoint_splits(mapping)
    def _event_count(subset):
        return int(train_pool[train_pool["unit_id"].isin(subset)]["event_observed"].sum())

    return {
        "dataset_id": "filters",
        "protocol": "author_test_held_out; author_train_80_20_units_seed42_stratified_events",
        "mode": cfg.get("mode", "filters_censored"),
        "train": mapping["train"],
        "validation": mapping["validation"],
        "test": mapping["test"],
        "n_train": len(mapping["train"]),
        "n_validation": len(mapping["validation"]),
        "n_test": len(mapping["test"]),
        "n_train_events": _event_count(mapping["train"]),
        "n_validation_events": _event_count(mapping["validation"]),
        "warning": warning,
        "seed": seed,
    }
