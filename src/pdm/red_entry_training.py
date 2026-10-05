"""Separate v2 RED-event training, equal physical-unit likelihood and frozen runs."""

from __future__ import annotations

import copy
import importlib
import json
import time
import uuid
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.nn import functional as F

from pdm.data.project_prepare import freeze_red_entry_exposure, load_snapshot, load_zone_limits
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.models.red_entry_recurrent import RedEntryRecurrent
from pdm.project_tasks import RED_ENTRY_ENGINES as ENGINES
from pdm.projects import project_store
from pdm.red_entry_features import build_windows, fit_feature_state, transform_prefix
from pdm.red_entry_protocol import (
    assert_run_compatible,
    build_run_contract,
    canonical_json_hash,
    load_protocol,
)
from pdm.red_entry_targets import build_red_entry_targets

BASELINES = ("kaplan_meier", "always_no_entry", "trend_to_red")


def masked_unit_hazard_loss(logits, targets, mask, physical_unit_ids):
    """Mean summed observed-bin NLL per origin, then mean per physical unit.

    Origins without any proven bin contribute nothing, including no negative
    evidence. A physical unit with many overlapping origins retains weight one.
    """
    if logits.shape != targets.shape or mask.shape != logits.shape:
        raise ValueError("Hazard arrays must have identical [origin,bin] shapes")
    if len(physical_unit_ids) != len(logits):
        raise ValueError("Physical unit IDs must align with origins")
    known = mask.bool()
    if (
        not torch.isfinite(logits[known]).all()
        or not torch.isfinite(targets[known]).all()
        or not torch.all((targets[known] == 0) | (targets[known] == 1))
    ):
        raise ValueError("Observed hazard bins require finite logits and binary targets")
    valid = known.any(dim=1)
    safe_logits = torch.where(known, logits, torch.zeros_like(logits))
    safe_targets = torch.where(known, targets, torch.zeros_like(targets)).float()
    per_origin = (
        F.binary_cross_entropy_with_logits(safe_logits, safe_targets, reduction="none") * known
    ).sum(dim=1)
    means = []
    for uid in sorted(set(map(str, physical_unit_ids))):
        indices = (
            torch.tensor([str(x) == uid for x in physical_unit_ids], device=logits.device) & valid
        )
        if indices.any():
            means.append(per_origin[indices].mean())
    return torch.stack(means).mean() if means else safe_logits.sum() * 0


def unit_equal_nll(hazard, batch):
    p = np.asarray(hazard)
    batch = {**batch, "mask": np.asarray(batch["mask"]) & np.isfinite(p)}
    p = np.clip(np.nan_to_num(p, nan=0.5), 1e-7, 1 - 1e-7)
    return float(
        masked_unit_hazard_loss(
            torch.tensor(np.log(p / (1 - p))),
            torch.tensor(batch["y"]),
            torch.tensor(batch["mask"]),
            batch["physical_unit_ids"],
        )
    )


def sample_origin_indices(origins, maximum=64, history_length=16):
    """Outcome-blind fixed cap per physical equipment, preserving early prefixes."""
    if maximum < 1:
        raise ValueError("Origin cap must be positive")
    selected = []
    for _, rows in origins.groupby("physical_unit_id", sort=True):
        rows = rows.sort_values(["unit_id", "timestamp_s"], kind="stable")
        ordered = rows.index.to_numpy()
        if len(ordered) <= maximum:
            selected.extend(ordered.tolist())
            continue
        early = []
        for _, unit in rows.groupby("unit_id", sort=True):
            early.extend(unit.index[:history_length].tolist())
        # Round-robin early histories across cycles instead of starving later IDs.
        groups = [u.index[:history_length].tolist() for _, u in rows.groupby("unit_id", sort=True)]
        early = [g[i] for i in range(history_length) for g in groups if i < len(g)]
        chosen = early[:maximum]
        remaining = [i for i in ordered if i not in set(chosen)]
        slots = maximum - len(chosen)
        if slots:
            chosen.extend(
                np.asarray(remaining)[np.linspace(0, len(remaining) - 1, slots, dtype=int)].tolist()
            )
        selected.extend(chosen)
    return np.asarray(sorted(selected), dtype=np.int64)


def _cancel(should_stop):
    if should_stop and should_stop():
        raise InterruptedError("RED-entry training cancelled")


def fit_model(engine_id, train, validation, config, *, should_stop=None, status_cb=None):
    if engine_id in BASELINES:
        from pdm.red_entry_baselines import fit_baseline_model

        return fit_baseline_model(
            engine_id, train, validation, config, should_stop=should_stop, status_cb=status_cb
        )
    if engine_id not in ("gru", "lstm"):
        module = importlib.import_module(
            "pdm.models.red_entry_" + ("boosting" if engine_id == "hazard_boosting" else "full_cns")
        )
        return module.fit_model(
            train, validation, config, should_stop=should_stop, status_cb=status_cb
        )
    if not train["mask"].any() or not validation["mask"].any():
        raise ValueError(
            "Train and Validation need observed survival bins for checkpoint selection"
        )
    torch.manual_seed(config["seed"])
    model = RedEntryRecurrent(
        engine_id, train["x"].shape[-1], config["hidden_size"], train["y"].shape[-1]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, best_loss, stale, epoch_best = None, float("inf"), 0, None
    # Full bounded origin batch preserves exact unit weighting across all origins.
    x, lengths, y, mask = (torch.as_tensor(train[k]) for k in ("x", "lengths", "y", "mask"))
    for epoch in range(config["epochs"]):
        _cancel(should_stop)
        model.train()
        optimizer.zero_grad()
        loss = masked_unit_hazard_loss(model(x, lengths), y, mask, train["physical_unit_ids"])
        loss.backward()
        optimizer.step()
        score = unit_equal_nll(predict_hazard(model, validation), validation)
        if not np.isfinite(score):
            raise ValueError("Nonfinite Validation NLL")
        if score < best_loss:
            best_loss, best, stale, epoch_best = (
                score,
                copy.deepcopy(model.state_dict()),
                0,
                epoch + 1,
            )
        else:
            stale += 1
        if status_cb:
            status_cb({"epoch": epoch + 1, "validation_unit_equal_nll": score})
        if stale >= config["patience"]:
            break
    model.load_state_dict(best)
    model.eval()
    return model, {
        "part": "validation",
        "metric": "unit_equal_masked_survival_nll",
        "value": best_loss,
        "selected_epoch": epoch_best,
        "completed_epochs": epoch + 1,
        "test_used": False,
    }


def predict_hazard(model, batch, *, should_stop=None, status_cb=None):
    _cancel(should_stop)
    if isinstance(model, dict) and model.get("engine_id") in BASELINES:
        from pdm.red_entry_baselines import predict_baseline_hazard

        return predict_baseline_hazard(model, batch)
    if isinstance(model, RedEntryRecurrent):
        model.eval()
        with torch.no_grad():
            return torch.sigmoid(
                model(torch.as_tensor(batch["x"]), torch.as_tensor(batch["lengths"]))
            ).numpy()
    module_name = getattr(model, "red_entry_adapter_module", None)
    if module_name:
        module = importlib.import_module(module_name)
        if module_name.endswith("full_cns"):
            return module.predict_hazard(model, batch, should_stop=should_stop, status_cb=status_cb)
        return module.predict_hazard(model, batch)
    if hasattr(model, "predict_hazard"):
        return model.predict_hazard(batch)
    raise ValueError("Model lacks RED hazard adapter identity")


def _effective_schema(project_id, data):
    limits = load_zone_limits(project_id, data["snapshot_id"])
    return {**data["schema"], "thresholds": limits} if limits else copy.deepcopy(data["schema"])


def _config(params):
    protocol = load_protocol()
    t = protocol["training"]
    r = t["recurrent"]
    cfg = dict(
        epochs=r["max_epochs"],
        patience=r["early_stopping_patience"],
        history_length=r["history_length"],
        hidden_size=r["hidden_size"],
        seed=t["seed"],
        learning_rate=0.001,
        maximum_origins_per_physical_unit=t["origin_sampling"]["maximum_origins_per_physical_unit"],
        input_mode="hybrid",
        horizons_s=None,
    )
    unknown = set(params or {}) - set(cfg)
    if unknown:
        raise ValueError(f"Unknown RED-entry parameters: {sorted(unknown)}")
    cfg.update(params or {})
    for k in (
        "epochs",
        "patience",
        "history_length",
        "hidden_size",
        "maximum_origins_per_physical_unit",
    ):
        if isinstance(cfg[k], bool) or not isinstance(cfg[k], int) or cfg[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    if not 0 < cfg["learning_rate"] <= 0.1:
        raise ValueError("learning_rate outside supported range")
    return cfg


def _batch(data, targets, state, schema, indices, config):
    b = build_windows(
        data, targets["origins"].iloc[indices], state, config["history_length"], schema=schema
    )
    b.update(y=targets["hazard_targets"][indices], mask=targets["hazard_mask"][indices].copy())
    b["mask"] &= b["availability"][:, None] & b["supported_regime"][:, None]
    b["prefixes"] = [
        data["features"]
        .loc[
            (data["features"].unit_id.astype(str) == str(o.unit_id))
            & (data["features"].timestamp_s <= float(o.timestamp_s))
        ]
        .sort_values("timestamp_s", kind="stable")
        .copy()
        for _, o in b["origins"].iterrows()
    ]
    b["schema"] = schema
    return b


def calibrated_hazard(hazards, calibration):
    """Calibrate supported cells while preserving unknown conditional hazards."""
    from pdm.red_entry_calibration import apply_hazard_calibrator
    values = np.asarray(hazards, float).copy()
    known = np.isfinite(values)
    if known.any():
        values[known] = apply_hazard_calibrator(values[known].reshape(-1, 1), calibration).ravel()
    return values


def chronological_evaluation(data, targets, batch, indices, probability, support, part):
    """Keep every measurement; omitted predictions are explicitly unknown."""
    all_origins = targets["origins"].copy()
    flags = [key for key in ("gap_before", "is_running", "quality_ok", "quality_status")
             if key in data["features"] and key not in all_origins]
    if flags:
        all_origins = all_origins.merge(data["features"][["unit_id", "timestamp_s", *flags]],
                                        on=["unit_id", "timestamp_s"], how="left", validate="one_to_one")
    probabilities = np.full((len(all_origins), len(support)), np.nan)
    probabilities[indices] = np.where(np.asarray(support)[None, :], probability, np.nan)
    all_origins["context_known"] = False
    all_origins.loc[indices, "context_known"] = batch["availability"] & batch["supported_regime"]
    selected = np.flatnonzero(all_origins["split"].to_numpy() == part)
    return all_origins.iloc[selected].reset_index(drop=True), probabilities[selected]


def train_red_entry_run(
    project_id, snapshot_id, engine_id, params=None, *, should_stop=None, status_cb=None
):
    if engine_id not in ENGINES:
        raise ValueError(f"Unsupported RED-entry engine: {engine_id}")
    cfg = _config(params)
    data = load_snapshot(project_id, snapshot_id)
    schema = _effective_schema(project_id, data)
    contract = build_run_contract(
        data,
        input_mode=cfg["input_mode"],
        effective_thresholds=schema["thresholds"],
        horizons_s=cfg["horizons_s"],
        split_provenance=freeze_red_entry_exposure(project_id, data, store=project_store()),
    )
    cfg["horizons_s"] = contract["horizons_s"]
    state = fit_feature_state(data, cfg["input_mode"], schema=schema)
    # Targets for Test are deliberately constructed only after checkpoint freeze.
    development = {
        **data,
        "features": data["features"][
            data["features"]
            .unit_id.astype(str)
            .isin(
                set(map(str, data["split"].get("train", []) + data["split"].get("validation", [])))
            )
        ].copy(),
    }
    dev_targets = build_red_entry_targets(development, cfg["horizons_s"], schema=schema)
    schedule = sample_origin_indices(
        dev_targets["origins"], cfg["maximum_origins_per_physical_unit"], cfg["history_length"]
    )
    batches = {
        p: _batch(
            development,
            dev_targets,
            state,
            schema,
            np.asarray(
                [i for i in schedule if dev_targets["origins"].iloc[i]["split"] == p], dtype=int
            ),
            cfg,
        )
        for p in ("train", "validation")
    }
    fit_cfg = {
        **cfg,
        "events": dev_targets["events"].loc[dev_targets["events"]["split"] == "train"].copy(),
        "train_unit_ids": data["split"]["train"],
        "schema": schema,
    }
    model, selection = fit_model(
        engine_id,
        batches["train"],
        batches["validation"],
        fit_cfg,
        should_stop=should_stop,
        status_cb=status_cb,
    )
    _cancel(should_stop)
    # Calibration and policy are frozen using development before any Test targets.
    support = batches["train"]["mask"].any(axis=0)
    if hasattr(model, "support_by_bin"):
        support &= np.asarray(model.support_by_bin, bool)
    support = np.logical_and.accumulate(support).tolist()
    from pdm.red_entry_calibration import fit_hazard_calibrator
    from pdm.red_entry_evaluation import evaluate_red_entry
    from pdm.red_entry_policy import select_alert_policy

    val_hazards = predict_hazard(model, batches["validation"], should_stop=should_stop, status_cb=status_cb)
    # Unknown tails never remove known early bins from calibration evidence.
    finite = np.isfinite(val_hazards)
    calibration = fit_hazard_calibrator(
        np.where(finite, val_hazards, 0.0), batches["validation"]["y"],
        batches["validation"]["mask"] & finite,
        batches["validation"]["physical_unit_ids"],
        requirements=contract["protocol"]["operational_requirements"],
        calibration_role="shared_validation_exploratory")
    calibration["selection_reused"] = True
    val_indices = np.asarray([i for i in schedule if dev_targets["origins"].iloc[i]["split"] == "validation"], int)
    val_cdf = 1 - np.cumprod(1 - calibrated_hazard(val_hazards, calibration), axis=1)
    val_origins, val_probability = chronological_evaluation(
        development, dev_targets, batches["validation"], val_indices, val_cdf, support, "validation")
    policy = select_alert_policy(val_origins,
        dev_targets["events"].loc[dev_targets["events"]["split"] == "validation"],
        val_probability, cfg["horizons_s"], requirements=contract["protocol"]["operational_requirements"])
    _cancel(should_stop)
    targets = build_red_entry_targets(data, cfg["horizons_s"], schema=schema)
    indices = sample_origin_indices(targets["origins"], cfg["maximum_origins_per_physical_unit"], cfg["history_length"])
    batch = _batch(data, targets, state, schema, indices, cfg)
    hazards = predict_hazard(model, batch, should_stop=should_stop, status_cb=status_cb)
    if (engine_id not in BASELINES and not np.isfinite(hazards).all()) or np.any(
        np.isfinite(hazards) & ((hazards < 0) | (hazards > 1))):
        raise ValueError("Hazard adapter returned invalid probabilities")
    calibrated = calibrated_hazard(hazards, calibration)
    cdf = 1 - np.cumprod(1 - calibrated, axis=1)
    metrics = {}
    for part in ("validation", "test"):
        origins, probabilities = chronological_evaluation(data, targets, batch, indices, cdf, support, part)
        metrics[part] = evaluate_red_entry(origins,
            targets["events"].loc[targets["events"]["split"] == part], probabilities,
            cfg["horizons_s"], policy=policy["config"],
            requirements=contract["protocol"]["operational_requirements"],
            provenance=contract["split_provenance"], model_frozen=True, policy_frozen=True)
        metrics[part]["evaluation_role"] = "shared_validation_exploratory" if part == "validation" else "historical_test_exploratory"
        metrics[part]["policy_selection_status"] = policy["status"]
    run_id = "red_entry_" + uuid.uuid4().hex[:12]
    directory = project_store().run_path(project_id, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    contract.update(
        engine_id=engine_id,
        params=cfg,
        snapshot_fingerprint_hash=canonical_json_hash(data.get("fingerprint")),
        feature_names=state["feature_names"],
        feature_state_hash=state["state_hash"],
        source_features_hash=canonical_json_hash(
            json.loads(data["features"].to_json(orient="records", double_precision=15))
        ),
        selection=selection,
        origin_sampling={"version": "physical_early_then_uniform_v1", "indices": indices.tolist()},
        calibration=calibration,
        support_by_horizon=support,
        policy=policy,
        scenario=contract["event"]["future_operating_scenario"],
        age_source="explicit_known_only",
        model_provenance=getattr(model, "provenance", None),
    )
    contract.pop("contract_hash", None)
    contract["contract_hash"] = canonical_json_hash(contract)
    atomic_write_json(directory / "training_contract.json", contract)
    atomic_write_json(directory / "feature_state.json", state)
    atomic_write_json(directory / "target_identity.json", targets["identity"])
    atomic_write_json(directory / "calibration.json", calibration)
    atomic_write_json(directory / "policy.json", policy)
    atomic_write_json(directory / "metrics.json", metrics)
    joblib.dump(targets, directory / "targets.joblib")
    np.savez_compressed(
        directory / "predictions.npz",
        hazard=hazards,
        calibrated_hazard=calibrated,
        probability=cdf,
        origin_indices=indices,
    )
    if engine_id in ("gru", "lstm"):
        artifact = "weights.pt"
        torch.save(model.state_dict(), directory / artifact)
    else:
        artifact = "model.joblib"
        joblib.dump(model, directory / artifact)
    manifest = dict(
        project_id=project_id,
        run_id=run_id,
        snapshot_id=data["snapshot_id"],
        task="red_entry",
        task_version=contract["task_version"],
        status="completed",
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        engine_id=engine_id,
        params=cfg,
        artifact=artifact,
        contract_hash=contract["contract_hash"],
        feature_names=state["feature_names"],
        support_by_horizon=support,
        selection=selection,
        metrics=metrics,
        calibration_status=calibration["status"],
        quality_gate_status=contract["quality_gate"]["status"],
        artifacts={p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()},
    )
    _cancel(should_stop)
    atomic_write_json(directory / "manifest.json", manifest)
    try:
        _cancel(should_stop)
        store = project_store()
        if store.get(project_id).get("active_snapshot_id") == data["snapshot_id"]:
            store.update(project_id, selected_run_id=run_id)
    except BaseException:
        (directory / "manifest.json").unlink(missing_ok=True)
        raise
    return manifest


def list_red_entry_runs(project_id):
    found = []
    for path in (project_store().project_path(project_id) / "runs").glob("*/manifest.json"):
        if path.is_symlink():
            continue
        try:
            m = read_json(path)
            if (
                m.get("task") == "red_entry"
                and m.get("status") == "completed"
                and m.get("project_id") == project_id
            ):
                found.append(m)
        except (OSError, ValueError):
            pass
    return sorted(found, key=lambda m: m["created_at"], reverse=True)


def load_red_entry_run(project_id, run_id):
    directory = project_store().run_path(project_id, run_id)
    if directory.is_symlink() or (directory / "manifest.json").is_symlink():
        raise ValueError("Unsafe run path")
    m = read_json(directory / "manifest.json")
    if (
        m.get("task") != "red_entry"
        or m.get("status") != "completed"
        or m.get("project_id") != project_id
        or m.get("run_id") != run_id
    ):
        raise ValueError("Not a completed RED-entry run for this project")
    required = {
        "training_contract.json",
        "feature_state.json",
        "target_identity.json",
        "calibration.json",
        "metrics.json",
        "targets.joblib",
        "predictions.npz",
        m["artifact"],
    }
    if not required.issubset(m.get("artifacts", {})):
        raise ValueError("Required RED-entry artifact absent")
    for name, digest in m["artifacts"].items():
        path = directory / name
        if Path(name).name != name or path.is_symlink() or sha256_file(path) != digest:
            raise ValueError("RED-entry artifact integrity mismatch")
    contract = read_json(directory / "training_contract.json")
    state = read_json(directory / "feature_state.json")
    if (
        canonical_json_hash({k: v for k, v in contract.items() if k != "contract_hash"})
        != contract["contract_hash"]
        or contract["contract_hash"] != m["contract_hash"]
    ):
        raise ValueError("Run contract hash mismatch")
    if (
        m["support_by_horizon"] != contract["support_by_horizon"]
        or m["calibration_status"] != contract["calibration"]["status"]
    ):
        raise ValueError("Manifest support/calibration mismatch")
    if contract.get("policy", {}).get("frozen"):
        if "policy.json" not in m["artifacts"] or read_json(directory / "policy.json") != contract["policy"]:
            raise ValueError("Frozen policy artifact mismatch")
    data = load_snapshot(project_id, m["snapshot_id"])
    schema = _effective_schema(project_id, data)
    assert_run_compatible(contract, data, effective_thresholds=schema["thresholds"])
    if contract["snapshot_fingerprint_hash"] != canonical_json_hash(data.get("fingerprint")):
        raise ValueError("Snapshot fingerprint changed")
    if contract["source_features_hash"] != canonical_json_hash(
        json.loads(data["features"].to_json(orient="records", double_precision=15))
    ):
        raise ValueError("Snapshot feature content changed")
    if (
        state["state_hash"] != contract["feature_state_hash"]
        or state["feature_names"] != contract["feature_names"]
        or m["feature_names"] != state["feature_names"]
    ):
        raise ValueError("Feature schema/order mismatch")
    transform_prefix(data["features"].iloc[:0], schema, state)
    if m["params"] != contract["params"] or m["engine_id"] != contract["engine_id"]:
        raise ValueError("Manifest model configuration mismatch")
    if m["engine_id"] in ("gru", "lstm"):
        model = RedEntryRecurrent(
            m["engine_id"],
            len(state["feature_names"]),
            m["params"]["hidden_size"],
            len(contract["horizons_s"]),
        )
        model.load_state_dict(
            torch.load(directory / m["artifact"], map_location="cpu", weights_only=True)
        )
        model.eval()
    else:
        model = joblib.load(directory / m["artifact"])
        if getattr(model, "provenance", None) != contract["model_provenance"]:
            raise ValueError("Saved model provenance mismatch")
    return {
        **m,
        "dir": directory,
        "contract": contract,
        "feature_state": state,
        "model": model,
        "snapshot": data,
        "effective_schema": schema,
    }


def available_red_entry_engines(project_id, snapshot_id=None):
    data = load_snapshot(project_id, snapshot_id)
    engines = []
    for engine in ENGINES:
        reason = None
        if engine == "full_cns":
            try:
                from pdm.models.red_entry_full_cns import source_unavailable_reason

                reason = source_unavailable_reason()
            except (ImportError, OSError, ValueError) as exc:
                reason = str(exc)
        if engine == "kaplan_meier":
            train = data["features"][
                data["features"].unit_id.astype(str).isin(set(map(str, data["split"]["train"])))
            ]
            if (
                "operating_age_known" not in train
                or not train.operating_age_known.fillna(False).any()
            ):
                reason = "Known operating age unavailable in Train"
        engines.append(
            {
                "engine_id": engine,
                "label": engine.replace("_", " ").title(),
                "available": reason is None,
                "reason": reason,
            }
        )
    return engines
