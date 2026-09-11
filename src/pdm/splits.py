from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

SPLIT_GROUPS = ("train", "validation", "test")


def split_hash(split: dict[str, Any]) -> str:
    """Stable identity of train/val/test unit lists plus protocol name."""
    payload = json.dumps(
        {k: split.get(k) for k in ("train", "validation", "test", "protocol")},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def split_fingerprint(split: dict[str, Any]) -> str:
    """Deprecated alias of split_hash; kept for old run dirs and imports."""
    return split_hash(split)


def resolve_split_hash(d: dict[str, Any] | None) -> str | None:
    """Read canonical split_hash, else legacy split_fingerprint."""
    if not d:
        return None
    if "split_hash" in d and d["split_hash"] is not None:
        return str(d["split_hash"])
    if "split_fingerprint" in d and d["split_fingerprint"] is not None:
        return str(d["split_fingerprint"])
    return None


def populate_origin_unit_id(units: pd.DataFrame) -> pd.DataFrame:
    """Set origin_unit_id if missing. Mutates ``units`` in place.

    Bearings: one folder per unit → ``origin_unit_id = unit_id``.
    Filters: ``author_data_no`` (HSE ``Data_No``), unique only within an author
    file. Train_Data_CSV and Test_Data_CSV reuse 1–50 for *different*
    experiments; namespaced ``unit_id`` (``Train_*`` / ``Test_*``) is the split key.
    """
    if "origin_unit_id" not in units.columns:
        if "author_data_no" in units.columns:
            units["origin_unit_id"] = units["author_data_no"]
        else:
            units["origin_unit_id"] = units["unit_id"].astype(str)
    missing = units["origin_unit_id"].isna()
    if missing.any():
        if "author_data_no" in units.columns:
            units.loc[missing, "origin_unit_id"] = units.loc[missing, "author_data_no"]
            missing = units["origin_unit_id"].isna()
        if missing.any():
            units.loc[missing, "origin_unit_id"] = units.loc[missing, "unit_id"].astype(str)
    return units


def _group_ids(split: dict[str, Any], group: str) -> list[str]:
    return [str(x) for x in (split.get(group) or [])]


def assert_disjoint_splits(split: dict[str, Any]) -> None:
    sets: dict[str, set[str]] = {}
    for k in SPLIT_GROUPS:
        ids = _group_ids(split, k)
        dups = sorted(uid for uid, n in Counter(ids).items() if n > 1)
        if dups:
            raise ValueError(f"Duplicate unit_id in {k}: {dups}")
        sets[k] = set(ids)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        inter = sets[a] & sets[b]
        if inter:
            raise ValueError(f"Split leak: {a} ∩ {b} = {sorted(inter)}")


def _protocol_expects_full_assignment(split: dict[str, Any]) -> bool:
    dataset_id = str(split.get("dataset_id") or "")
    if dataset_id == "bearings":
        return False
    if dataset_id == "filters":
        return True
    protocol = str(split.get("protocol") or "")
    if "per_regime_instances" in protocol:
        return False
    return True


def _physical_history_key(row: pd.Series) -> str:
    """Identity of one physical run for origin leakage.

    HSE author train/test files reuse ``Data_No`` for held-out *different*
    experiments (CSV lengths and Δp series do not match). Bare number overlap
    across ``author_split`` is not leakage. Collision *is* leakage when the
    same physical history appears twice in one group or in two groups
    (same ``author_split`` + ``origin_unit_id``, or same origin with no file
    namespace — bearings ``unit_id``).
    """
    origin = str(row["origin_unit_id"])
    uid = str(row["unit_id"])
    author_split = ""
    if "author_split" in row.index:
        val = row["author_split"]
        if pd.notna(val) and str(val).strip():
            author_split = str(val)
    if author_split:
        return f"{author_split}:{origin}"
    if uid.startswith("Train_") or uid.startswith("Test_"):
        return f"{uid.split('_', 1)[0]}:{origin}"
    return origin


def assert_split_coverage(units: pd.DataFrame, split: dict[str, Any]) -> None:
    """Fail on duplicates, unassigned units (when required), and origin leaks."""
    populate_origin_unit_id(units)
    assert_disjoint_splits(split)

    unit_ids = units["unit_id"].astype(str)
    table_dups = sorted(uid for uid, n in Counter(unit_ids).items() if n > 1)
    if table_dups:
        raise ValueError(f"Duplicate unit_id in units table: {table_dups}")

    assigned: list[str] = []
    for group in SPLIT_GROUPS:
        assigned.extend(_group_ids(split, group))
    assigned_set = set(assigned)
    unit_set = set(unit_ids)
    recorded_unassigned = {str(x) for x in (split.get("unassigned") or [])}

    missing = sorted(assigned_set - unit_set)
    if missing:
        raise ValueError(f"Split lists units missing from units table: {missing}")

    orphans = sorted(unit_set - assigned_set - recorded_unassigned)
    if orphans:
        raise ValueError(
            f"Units not assigned to train/validation/test/unassigned: {orphans}"
        )

    leftover = sorted((unit_set - assigned_set) | recorded_unassigned)
    if leftover and _protocol_expects_full_assignment(split):
        raise ValueError(f"Unassigned units (protocol expects full assignment): {leftover}")

    if units["origin_unit_id"].isna().any():
        raise ValueError("origin_unit_id has nulls")

    key_by_uid = {str(row["unit_id"]): _physical_history_key(row) for _, row in units.iterrows()}
    origin_sets: dict[str, set[str]] = {}
    for group in SPLIT_GROUPS:
        keys = [key_by_uid[uid] for uid in _group_ids(split, group)]
        same_group = sorted(k for k, n in Counter(keys).items() if n > 1)
        if same_group:
            raise ValueError(
                f"Split leak: duplicate origin_unit_id in {group}: {same_group} "
                "(same physical history twice in one group)"
            )
        origin_sets[group] = set(keys)
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        inter = origin_sets[a] & origin_sets[b]
        if inter:
            raise ValueError(
                f"Split leak: origin_unit_id {a} ∩ {b} = {sorted(inter)} "
                "(same physical history in two groups)"
            )


def bearings_split(units: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict:
    populate_origin_unit_id(units)
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
    unknown = sorted(unknown)
    assert_disjoint_splits(mapping)
    warning = None
    if unknown:
        warning = (
            f"Bearings split left {len(unknown)} unassigned unit(s): {unknown}. "
            "Official 9/3/3 protocol uses instances 1–3 train, 4 val, 5 test."
        )
        warnings.warn(warning, UserWarning, stacklevel=2)
    result = {
        "dataset_id": "bearings",
        "protocol": "per_regime_instances_1-3_train_4_val_5_test",
        "train": mapping["train"],
        "validation": mapping["validation"],
        "test": mapping["test"],
        "unassigned": unknown,
        "n_train": len(mapping["train"]),
        "n_validation": len(mapping["validation"]),
        "n_test": len(mapping["test"]),
        "warning": warning,
    }
    assert_split_coverage(units, result)
    return result


def filters_split(units: pd.DataFrame, cfg: dict[str, Any] | None = None) -> dict:
    """Keep author test units untouched. Split author train 80/20 by unit, seed 42."""
    from sklearn.model_selection import StratifiedShuffleSplit

    populate_origin_unit_id(units)
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

    result = {
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
    assert_split_coverage(units, result)
    return result
