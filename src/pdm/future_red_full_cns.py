"""Full MaleCNS feature adapter for fixed-horizon future-red forecasting.

The connectome and leaky dynamics stay fixed. Every classified neuron is advanced
for each causal sensor row; anatomical groups are pooled only after the full
state update. The trainable future-red readout consumes that pool plus the
current scaled sensor features.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from pdm.config import load_dataset_config
from pdm.data.prepare import load_processed
from pdm.feature_recipes import recipe_names
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.models.full_cns import build_full_cns
from pdm.preprocessing import (
    Preprocessor,
    apply_preprocessor,
    bearings_feature_frame,
    raw_to_feature_frame,
)

ADAPTER_VERSION = "full_malecns_future_red_v1"
DEFAULT_HORIZON_S = 1800.0
MODEL_POPULATION = "all_classified_neurons"
ROW_COLUMNS = (
    "unit_id", "split", "timestamp_s", "target", "target_known",
    "first_red_timestamp_s", "at_risk",
)


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return sha256_file(path)


def _reservoir_parameter_digest(model) -> str:
    """Fingerprint the actual fixed operator and projection, not just its graph."""
    digest = hashlib.sha256()
    for name in ("seed", "alpha", "spectral_radius", "input_scale", "graph_hash"):
        digest.update(name.encode("utf-8"))
        digest.update(repr(getattr(model, name, None)).encode("utf-8"))
    for name in ("W_in", "b_res", "W_res"):
        digest.update(name.encode("utf-8"))
        tensor = getattr(model, name, None)
        if tensor is None:
            raise ValueError(f"Full CNS model is missing frozen parameter {name}")
        if hasattr(tensor, "detach"):
            tensor = tensor.detach().cpu()
            if getattr(tensor, "is_sparse_csr", False):
                arrays = (tensor.crow_indices().contiguous().numpy(),
                          tensor.col_indices().contiguous().numpy(),
                          tensor.values().contiguous().numpy())
            else:
                arrays = (tensor.contiguous().numpy(),)
        else:
            array = np.ascontiguousarray(tensor)
            arrays = (array,)
        for array in arrays:
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _cached_trajectory(model, inputs, gaps, unit_id, cache_dir, binding, should_stop,
                       parameter_sha256):
    """Get one unit's pooled trajectory, with a hash-checked, atomic cache."""
    identity = {
        "version": 1,
        "adapter": ADAPTER_VERSION,
        "binding": binding,
        "unit_id": str(unit_id),
        "input_sha256": hashlib.sha256(inputs.tobytes() + gaps.tobytes()).hexdigest(),
        "graph_hash": model.provenance.get("graph_hash"),
        "leak": float(model.alpha),
        "seed": int(getattr(model, "seed", 0)),
        "reservoir_parameter_sha256": parameter_sha256,
        "pool_index_sha256": hashlib.sha256(np.asarray(model.pool_index).tobytes()).hexdigest(),
    }
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _json_hash(identity)
    array_path = cache_dir / f"{key}.npy"
    meta_path = cache_dir / f"{key}.json"
    if array_path.is_file() and meta_path.is_file():
        meta = read_json(meta_path)
        if meta.get("identity") == identity and meta.get("sha256") == sha256_file(array_path):
            candidate = np.load(array_path, mmap_mode="r", allow_pickle=False)
            if candidate.shape == (len(inputs), model.n_readout_features) and np.isfinite(candidate).all():
                return np.asarray(candidate, dtype=np.float32)
    states = model.pooled_trajectory(inputs, gaps, should_stop=should_stop)
    states = np.asarray(states, dtype=np.float32)
    if states.shape != (len(inputs), model.n_readout_features) or not np.isfinite(states).all():
        raise ValueError(f"Invalid full CNS pooled trajectory for unit {unit_id}")
    # Write to a sibling temporary file and atomically replace; interrupted jobs
    # never leave a cache entry that can be mistaken for a completed trajectory.
    temporary = array_path.with_name(f".{array_path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, states, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, array_path)
    finally:
        temporary.unlink(missing_ok=True)
    atomic_write_json(meta_path, {"identity": identity, "sha256": sha256_file(array_path)})
    return states


def _validate_targets(targets: pd.DataFrame) -> pd.DataFrame:
    missing = set(ROW_COLUMNS) - set(targets.columns)
    if missing:
        raise ValueError(f"Future-red target table is missing columns: {sorted(missing)}")
    out = targets.loc[:, ROW_COLUMNS].copy()
    out["unit_id"] = out.unit_id.astype(str)
    out["split"] = out.split.astype(str)
    out["timestamp_s"] = pd.to_numeric(out.timestamp_s, errors="coerce")
    if out[["timestamp_s"]].isna().any().any() or not np.isfinite(out.timestamp_s).all():
        raise ValueError("Future-red target timestamps must be finite")
    if out.duplicated(["unit_id", "timestamp_s"]).any():
        raise ValueError("Future-red targets contain duplicate unit/timestamp origins")
    known = out.target_known.astype(bool)
    values = pd.to_numeric(out.target, errors="coerce")
    if values[known].isna().any() or not values[known].isin([0, 1]).all():
        raise ValueError("Known future-red targets must be binary")
    if values[~known].notna().any():
        raise ValueError("Unknown future-red targets must have a null target")
    out["target"] = values.astype(float)
    out["target_known"] = known
    out["at_risk"] = out.at_risk.astype(bool)
    out["first_red_timestamp_s"] = pd.to_numeric(out.first_red_timestamp_s, errors="coerce")
    return out.sort_values(["unit_id", "timestamp_s"], kind="stable").reset_index(drop=True)


def _fit_sensor_preprocessor(features: pd.DataFrame, train_ids: set[str], config: dict[str, Any]):
    """Fit the existing causal bearing feature pipeline without endpoint fields."""
    recipe = str(config.get("feature_recipe", "base_v1"))
    log1p_columns = list(config.get("features", {}).get("log1p_features") or [])
    categorical_maps: dict[str, list[str]] = {}
    if recipe in {"multiscale_trend_v1", "multiscale_no_age_v1", "multiscale_trend_v2", "multiscale_no_age_v2"}:
        train = features.loc[features.unit_id.astype(str).isin(train_ids)]
        if "regime_id" not in train:
            raise ValueError("The selected causal feature recipe requires regime_id")
        categorical_maps["regime_id"] = sorted(train.regime_id.astype(str).unique().tolist())
    encoded = raw_to_feature_frame("bearings", features, categorical_maps, log1p_columns,
                                   raw_features=True, feature_recipe=recipe)
    _, names, log1p_columns = bearings_feature_frame(encoded, log1p_columns)
    names += recipe_names("bearings", recipe)
    if recipe in {"multiscale_no_age_v1", "multiscale_no_age_v2"}:
        names = [name for name in names if name != "operating_age_s"]
    train = encoded.loc[encoded.unit_id.astype(str).isin(train_ids), names]
    if train.empty:
        raise ValueError("No training sensor features available for scaler fitting")
    values = train.to_numpy(dtype=np.float64).copy()
    values[~np.isfinite(values)] = np.nan
    fills: dict[str, float] = {}
    for index, name in enumerate(names):
        finite = values[:, index][np.isfinite(values[:, index])]
        fills[name] = float(np.median(finite)) if len(finite) else 0.0
        values[:, index] = np.where(np.isfinite(values[:, index]), values[:, index], fills[name])
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[~np.isfinite(scale) | (scale < 1e-12)] = 1.0
    prep = Preprocessor(
        feature_names=names, log1p_features=log1p_columns,
        scaler_mean=mean.tolist(), scaler_scale=scale.tolist(), time_scale_s=1.0,
        categorical_maps=categorical_maps, fill_values=fills, dataset_id="bearings",
        feature_recipe=recipe,
    )
    return prep


def _add_full_cns_baselines(prepared: dict[str, Any], features: pd.DataFrame) -> None:
    """Attach transparent baseline scores on the same saved target origins."""
    from pdm.future_red_models import build_future_red_baselines

    target_manifest = prepared.get("target_manifest") or {}
    horizon_s = float(target_manifest.get("horizon_s", DEFAULT_HORIZON_S))
    zone_policy = target_manifest.get("zone_policy") or {}
    prepared["baseline_probabilities"] = build_future_red_baselines(
        "bearings", features, prepared, horizon_s=horizon_s, zone_policy=zone_policy,
    )
    prepared["baseline_definitions"] = {
        "always_no_entry": "Predict no RED entry at every origin.",
        "current_red_persistence": "Predict RED when the current causal sensor measurement already meets the saved RED comparator.",
        "trend_to_red": (
            f"Fit a straight line to the last {int(zone_policy.get('trend_points', 5))} "
            "observations since the latest gap; predict RED if it crosses the saved threshold within the target horizon."
        ),
        "horizon_s": horizon_s,
        "threshold_policy": zone_policy,
        "probability_encoding": "deterministic 0 or 1",
        "origin_clock": "exact target unit_id and timestamp_s rows; unknown labels remain prediction origins",
    }


def prepare_full_cns_future_red(
    data: dict[str, Any] | None = None,
    targets: pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]] | None = None,
    *,
    target_artifact_dir: str | Path | None = None,
    target_manifest: dict[str, Any] | None = None,
    source_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
    config: dict[str, Any] | None = None,
    model=None,
    model_factory: Callable[..., Any] | None = None,
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Prepare Full MaleCNS pooled features and binary targets for the readout.

    `data` is the normal `load_processed("bearings")` result. Tests may pass a
    miniature model implementing `pooled_trajectory`; production always builds
    the complete real-connectome model and has no synthetic fallback.

    Return `X_{train,validation,test}` as float32 `[rows, pooled_groups + sensors]`,
    `y_*` with NaN for unsupervised rows, and aligned `rows_*` tables. The generic
    trainer can fit known labels only and still emit a causal probability for each
    observed test origin.
    """
    if data is None:
        data = load_processed("bearings")
    if targets is None:
        if target_artifact_dir is None:
            raise ValueError("Pass targets or target_artifact_dir from the saved future-red target artifact")
        from pdm.future_red_targets import load_future_red_targets

        targets, loaded_manifest = load_future_red_targets(target_artifact_dir, data)
        target_manifest = target_manifest or loaded_manifest
    elif isinstance(targets, tuple):
        targets, loaded_manifest = targets
        target_manifest = target_manifest or loaded_manifest
    targets = _validate_targets(targets)
    if str((data.get("split") or {}).get("dataset_id", "bearings")) != "bearings":
        raise ValueError("Full MaleCNS future-red adapter supports bearings only")
    split = data.get("split") or {}
    split_ids = {name: set(map(str, split.get(name, []))) for name in ("train", "validation", "test")}
    if not split_ids["train"] or not split_ids["validation"] or not split_ids["test"]:
        raise ValueError("Bearing split must contain train, validation, and test units")
    expected_split = {uid: name for name, ids in split_ids.items() for uid in ids}
    for uid, group in targets.groupby("unit_id", sort=False):
        if uid not in expected_split or set(group.split) != {expected_split[uid]}:
            raise ValueError(f"Target split does not match prepared split for unit {uid}")

    cfg = config or load_dataset_config("bearings")
    # This forecast does not read event_time_s, RUL, or endpoint-derived time scales.
    prep = _fit_sensor_preprocessor(data["features"], split_ids["train"], cfg)
    encoded = apply_preprocessor(prep, data["features"], "bearings")
    # Age is a derived clock. Use physical vibration, operating regime, and
    # causal past-only features as sensor projections into the full reservoir.
    sensor_names = [name for name in prep.feature_names if name != "operating_age_s"]
    if not sensor_names:
        raise ValueError("No causal bearing sensor features remain after removing operating age")
    using_test_fixture = model is not None
    if model is None:
        factory = model_factory or build_full_cns
        model = factory(
            len(sensor_names),
            1.0,
            source_path=source_path,
            seed=seed,
            head="rul",  # existing frozen reservoir container; future-red readout is separate
        )
    if model.is_synthetic or model.provenance.get("graph_mode") != "real_connectome":
        raise ValueError("Future-red Full CNS requires the real MaleCNS connectome")
    if model.n_nodes != int(model.provenance.get("n_nodes", model.n_nodes)):
        raise ValueError("Full CNS model size does not match its provenance")
    if not using_test_fixture and (
        int(model.n_nodes) != 166_700
        or int(model.provenance.get("n_edges", -1)) != 25_582_938
        or model.provenance.get("sampling_method") != MODEL_POPULATION
    ):
        raise ValueError("Full MaleCNS future-red requires all 166,700 neurons and 25,582,938 connectome edges")
    if int(getattr(model, "input_size", len(sensor_names))) != len(sensor_names):
        raise ValueError("Full CNS sensor projection width does not match the train-only feature schema")
    binding = {
        "dataset_version": data.get("dataset_version"),
        "dataset_fingerprint": data.get("fingerprint") or {},
        "preprocessor": prep.to_dict(),
        "target_artifact_id": (target_manifest or {}).get("artifact_id"),
        "target_sha256": (target_manifest or {}).get("targets_sha256"),
    }
    if cache_dir is None and output_dir is not None:
        cache_dir = Path(output_dir) / "state_cache"
    parameter_sha256 = _reservoir_parameter_digest(model)
    xs: dict[str, list[np.ndarray]] = {name: [] for name in split_ids}
    ys: dict[str, list[float]] = {name: [] for name in split_ids}
    row_frames: dict[str, list[pd.DataFrame]] = {name: [] for name in split_ids}
    target_groups = {str(uid): group for uid, group in targets.groupby("unit_id", sort=False)}
    for uid, target_rows in target_groups.items():
        part = expected_split[uid]
        if should_stop and should_stop():
            raise InterruptedError("Full CNS future-red feature preparation cancelled")
        group = encoded.loc[encoded.unit_id.astype(str).eq(uid)].sort_values("timestamp_s", kind="stable")
        if group.empty:
            raise ValueError(f"No sensor measurements found for target unit {uid}")
        if group.timestamp_s.duplicated().any() or not np.isfinite(group.timestamp_s.to_numpy(float)).all():
            raise ValueError(f"Invalid sensor timestamps for unit {uid}")
        inputs = group[sensor_names].to_numpy(np.float32)
        gaps = group.get("gap_before", pd.Series(False, index=group.index)).fillna(False).to_numpy(bool)
        representation = (
            _cached_trajectory(model, inputs, gaps, uid, cache_dir, binding, should_stop,
                               parameter_sha256)
            if cache_dir is not None
            else np.asarray(model.pooled_trajectory(inputs, gaps, should_stop=should_stop), dtype=np.float32)
        )
        if representation.shape != (len(group), model.n_readout_features):
            raise ValueError(f"Full CNS trajectory shape mismatch for unit {uid}")
        # Join on the exact measurement timestamp; no interpolation or future row.
        positions = pd.Series(np.arange(len(group), dtype=np.int64), index=group.timestamp_s.to_numpy(float))
        target_rows = target_rows.sort_values("timestamp_s", kind="stable").copy()
        matched = target_rows.timestamp_s.map(positions)
        if matched.isna().any():
            raise ValueError(f"Future-red origins do not exactly match sensor rows for unit {uid}")
        index = matched.to_numpy(np.int64)
        state_and_sensor = np.concatenate([representation, inputs], axis=1)
        x = state_and_sensor[index]
        target_rows = target_rows.loc[:, ROW_COLUMNS].reset_index(drop=True)
        target_rows["target"] = target_rows.target.astype(float)
        xs[part].append(x.astype(np.float32, copy=False))
        ys[part].extend(target_rows.target.where(target_rows.target_known, np.nan).to_numpy(float).tolist())
        row_frames[part].append(target_rows)
        if progress:
            progress(f"Full MaleCNS {uid}: {len(group)} causal measurements pooled across {model.n_nodes:,} neurons")

    prepared: dict[str, Any] = {}
    for part in split_ids:
        if not xs[part]:
            raise ValueError(f"No target rows for {part} split")
        prepared[f"X_{part}"] = np.concatenate(xs[part]).astype(np.float32, copy=False)
        prepared[f"y_{part}"] = np.asarray(ys[part], dtype=np.float32)
        prepared[f"rows_{part}"] = pd.concat(row_frames[part], ignore_index=True)
    feature_positions = [prep.feature_names.index(name) for name in sensor_names]
    prepared["scaler"] = {"mean": [prep.scaler_mean[i] for i in feature_positions],
                          "scale": [prep.scaler_scale[i] for i in feature_positions],
                          "feature_names": sensor_names, "preprocessor": prep.to_dict(),
                          "fit_unit_ids": sorted(split_ids["train"])}
    prepared["target_manifest"] = target_manifest or {}
    prepared["architecture"] = ADAPTER_VERSION
    prepared["provenance"] = {
        "adapter": ADAPTER_VERSION,
        "dataset_id": "bearings",
        "dataset_version": data.get("dataset_version"),
        "dataset_fingerprint": data.get("fingerprint") or {},
        "target_artifact_id": (target_manifest or {}).get("artifact_id"),
        "target_sha256": (target_manifest or {}).get("targets_sha256"),
        "horizon_s": float((target_manifest or {}).get("horizon_s", DEFAULT_HORIZON_S)),
        "model_population": MODEL_POPULATION,
        "n_neurons_computed": int(model.n_nodes),
        "n_pool_groups": int(model.n_readout_features),
        "n_sensor_inputs": len(sensor_names),
        "feature_names": sensor_names,
        "state_policy": "continuous per bearing; reset at marked acquisition gaps; all neurons compute before anatomical pooling",
        "history_contract": "use all available causal rows from the current acquisition episode start through the exact target origin; no fixed lookback truncation",
        "readout_input": "pooled full-brain state plus current scaled causal sensor features",
        "checkpoint_replay_requirement": "the saved model contains frozen reservoir parameters and binary readout, not live neuron state; reload must replay the preprocessed prefix from the episode start (or restore a separately serialized state bound to this graph, seed, and scaler) before scoring",
        "scaler_fit_unit_ids": sorted(split_ids["train"]),
        "connectome": dict(model.provenance),
    }
    from pdm.future_red_models import REPLAY_CONTRACT_VERSION

    prepared["model_contract"] = {
        "version": REPLAY_CONTRACT_VERSION,
        "architecture": ADAPTER_VERSION,
        "dataset_id": "bearings",
        "state_policy": prepared["provenance"]["state_policy"],
        "history_contract": prepared["provenance"]["history_contract"],
        "graph_hash": str(model.graph_hash),
        "seed": int(getattr(model, "seed", seed)),
        "leak": float(model.alpha),
        "n_nodes": int(model.n_nodes),
        "n_pool_groups": int(model.n_readout_features),
        "feature_names": list(sensor_names),
        "preprocessor": prep.to_dict(),
        "target_horizon_s": float((target_manifest or {}).get("horizon_s", DEFAULT_HORIZON_S)),
        "target_artifact_id": (target_manifest or {}).get("artifact_id"),
        "readout_shape": [int(prepared["X_train"].shape[1])],
        "readout_input": "pooled full-brain state plus current scaled causal sensor features",
        "target": "future_red_entry_binary",
        "uses_rul_target": False,
    }
    if output_dir is not None and hasattr(model, "write_artifacts"):
        artifact_path = Path(output_dir) / "full_cns"
        model.write_artifacts(artifact_path)
        atomic_write_json(artifact_path / "model_meta.json", {
            "n_nodes": int(model.n_nodes), "graph_hash": model.graph_hash,
            "leak": float(model.alpha), "time_scale_s": 1.0, "head": "rul",
            "readout_head": "separate_binary_logistic", "seed": int(seed),
            "state_mode": "continuous",
            "history_contract": prepared["provenance"]["history_contract"],
            "checkpoint_replay_requirement": prepared["provenance"]["checkpoint_replay_requirement"],
        })
        prepared["provenance"]["full_cns_artifact_dir"] = str(artifact_path)
        prepared["provenance"]["full_cns_artifact_hashes"] = dict(
            read_json(artifact_path / "provenance.json").get("artifact_hashes", {})
        )
        prepared["model_contract"]["full_cns_artifact_hashes"] = dict(
            prepared["provenance"]["full_cns_artifact_hashes"]
        )
        atomic_write_json(Path(output_dir) / "full_cns_adapter.json", prepared["provenance"])
    return prepared


def fit_full_cns_future_red(
    *,
    targets: pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]] | None = None,
    target_artifact_dir: str | Path | None = None,
    output_dir: str | Path,
    data: dict[str, Any] | None = None,
    source_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
    seed: int = 42,
    config: dict[str, Any] | None = None,
    epochs: int = 100,
    threshold_policy: str = "validation_f1",
    should_stop: Callable[[], bool] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Prepare Full CNS representations and fit the shared binary readout."""
    if data is None:
        data = load_processed("bearings")
    if targets is None and target_artifact_dir is None:
        from pdm.future_red_targets import build_future_red_targets

        target_artifact_dir = build_future_red_targets("bearings")["directory"]
    prepared = prepare_full_cns_future_red(
        data, targets, target_artifact_dir=target_artifact_dir, source_path=source_path,
        cache_dir=cache_dir, output_dir=output_dir, seed=seed, config=config,
        should_stop=should_stop, progress=progress,
    )
    _add_full_cns_baselines(prepared, data["features"])
    from pdm.future_red_models import fit_future_red_readout

    result = fit_future_red_readout(prepared, output_dir, seed=seed, epochs=epochs,
                                    threshold_policy=threshold_policy)
    result["architecture"] = ADAPTER_VERSION
    _finalize_full_cns_run_manifest(output_dir, result)
    return result


def _finalize_full_cns_run_manifest(output_dir: str | Path, result: dict[str, Any]) -> dict[str, Any]:
    """Bind the logistic readout and saved Full CNS body into one replay contract."""
    out = Path(output_dir)
    adapter = read_json(out / "full_cns_adapter.json")
    meta = read_json(out / "full_cns" / "model_meta.json")
    body_provenance = read_json(out / "full_cns" / "provenance.json")
    contract = dict(result.get("model_contract") or {})
    if not contract:
        raise ValueError("Full CNS future-red training did not produce a replay contract")
    contract["full_cns_artifact_hashes"] = dict(body_provenance.get("artifact_hashes") or {})
    contract["full_cns_meta"] = dict(meta)
    hashes = {
        "readout.pt": _file_sha256(out / "readout.pt"),
        **{f"full_cns/{name}": digest for name, digest in contract["full_cns_artifact_hashes"].items()},
    }
    result["model_contract"] = contract
    result["artifact_hashes"] = {**result.get("artifact_hashes", {}), **hashes}
    result["provenance"] = {**result.get("provenance", {}),
                             "full_cns_artifact_dir": str(out / "full_cns"),
                             "full_cns_artifact_hashes": contract["full_cns_artifact_hashes"]}
    payload = torch.load(out / "readout.pt", map_location="cpu", weights_only=True)
    payload["model_contract"] = contract
    torch.save(payload, out / "readout.pt")
    # The readout hash belongs in the external manifest to avoid a self-hash.
    hashes["readout.pt"] = _file_sha256(out / "readout.pt")
    result["artifact_hashes"] = {**result.get("artifact_hashes", {}), **hashes}
    (out / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    atomic_write_json(out / "full_cns_adapter.json", {**adapter, "model_contract": contract})
    return result


def verify_full_cns_future_red_artifacts(run_dir: str | Path) -> dict[str, Any]:
    """Verify the saved readout contract and every Full CNS body artifact."""
    root = Path(run_dir)
    manifest = read_json(root / "manifest.json")
    contract = manifest.get("model_contract") or {}
    from pdm.future_red_models import REPLAY_CONTRACT_VERSION

    if contract.get("version") != REPLAY_CONTRACT_VERSION or contract.get("architecture") != ADAPTER_VERSION:
        raise ValueError("Run manifest does not describe a replayable Full CNS future-red model")
    if contract.get("uses_rul_target") is not False or contract.get("target") != "future_red_entry_binary":
        raise ValueError("Full CNS replay contract must use only the binary future-red target")
    hashes = manifest.get("artifact_hashes") or {}
    body_hashes = contract.get("full_cns_artifact_hashes") or {}
    result_files = {"predictions.parquet", "per_unit_metrics.csv",
                    "baseline_predictions.parquet", "baseline_metrics.json"}
    core_hashes = {"readout.pt": hashes.get("readout.pt"),
                   **{f"full_cns/{name}": digest for name, digest in body_hashes.items()}}
    allowed = set(core_hashes) | result_files
    if (not body_hashes or not all(hashes.get(name) == digest and digest for name, digest in core_hashes.items())
            or not set(hashes).issubset(allowed)):
        raise ValueError("Full CNS replay artifact hash list is incomplete or unexpected")
    for name, digest in sorted(hashes.items()):
        path = root / name
        if not path.is_file() or _file_sha256(path) != digest:
            raise ValueError(f"Full CNS future-red artifact hash mismatch: {name}")
    payload = torch.load(root / "readout.pt", map_location="cpu", weights_only=True)
    if payload.get("model_contract") != contract:
        raise ValueError("Full CNS readout checkpoint does not match its replay contract")
    return manifest


def load_full_cns_future_red_model(run_dir: str | Path) -> dict[str, Any]:
    """Load and verify a saved Full CNS future-red readout and fixed connectome."""
    root = Path(run_dir)
    manifest = verify_full_cns_future_red_artifacts(root)
    contract = manifest["model_contract"]
    meta = read_json(root / "full_cns" / "model_meta.json")
    if meta != contract.get("full_cns_meta"):
        raise ValueError("Full CNS model metadata does not match the future-red replay contract")
    from pdm.models.full_cns import load_full_cns

    body = load_full_cns(root / "full_cns", meta)
    if (str(body.graph_hash) != contract["graph_hash"]
            or int(body.seed) != int(contract["seed"])
            or not np.isclose(float(body.alpha), float(contract["leak"]))
            or not np.isclose(float(body.time_scale_s), float(meta["time_scale_s"]))
            or str(body.head_type) != str(meta["head"])
            or int(body.n_readout_features) != int(contract["n_pool_groups"])):
        raise ValueError("Full CNS loaded body does not match the future-red replay contract")
    payload = torch.load(root / "readout.pt", map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict") or {}
    weight, bias = state_dict.get("weight"), state_dict.get("bias")
    expected_shape = tuple(map(int, contract.get("readout_shape", [])))
    if weight is None or bias is None or tuple(weight.shape) != (1, *expected_shape):
        raise ValueError("Full CNS readout shape does not match its replay contract")
    return {"body": body, "readout_weight": weight.detach().cpu().numpy().reshape(-1),
            "readout_bias": float(bias.detach().cpu().numpy().reshape(-1)[0]),
            "contract": contract, "manifest": manifest}


def predict_full_cns_future_red_origins(
    loaded: dict[str, Any], data: pd.DataFrame | dict[str, Any], *,
    should_stop: Callable[[], bool] | None = None,
) -> pd.DataFrame:
    """Replay future-red probabilities for every supplied causal bearing origin.

    `data` may be the raw bearing feature frame or the standard load_processed
    mapping containing it. Each unit is replayed from its episode start, with
    state resets at marked gaps, so saved live neuron states are not required.
    """
    contract = loaded["contract"]
    features = data.get("features") if isinstance(data, dict) else data
    if not isinstance(features, pd.DataFrame) or not {"unit_id", "timestamp_s"}.issubset(features.columns):
        raise ValueError("Full CNS replay expects bearing feature rows with unit_id and timestamp_s")
    prep = Preprocessor.from_dict(contract["preprocessor"])
    encoded = apply_preprocessor(prep, features, "bearings")
    names = list(contract["feature_names"])
    if names != [name for name in prep.feature_names if name != "operating_age_s"]:
        raise ValueError("Full CNS replay sensor features do not match the saved scaler contract")
    body = loaded["body"]
    weight = np.asarray(loaded["readout_weight"], dtype=np.float32)
    expected = int(contract["n_pool_groups"]) + len(names)
    if weight.shape != (expected,) or int(body.input_size) != len(names):
        raise ValueError("Full CNS replay readout or sensor width does not match its contract")
    output = []
    for uid, original in features.groupby(features.unit_id.astype(str), sort=False):
        if should_stop and should_stop():
            raise InterruptedError("Full CNS future-red replay cancelled")
        group = encoded.loc[encoded.unit_id.astype(str).eq(str(uid))].sort_values(
            "timestamp_s", kind="stable")
        if group.timestamp_s.duplicated().any() or not np.isfinite(group.timestamp_s.to_numpy(float)).all():
            raise ValueError(f"Invalid sensor timestamps for unit {uid}")
        inputs = group[names].to_numpy(np.float32)
        gaps = group.get("gap_before", pd.Series(False, index=group.index)).fillna(False).to_numpy(bool)
        pooled = np.asarray(body.pooled_trajectory(inputs, gaps, should_stop=should_stop), dtype=np.float32)
        if pooled.shape != (len(group), int(contract["n_pool_groups"])):
            raise ValueError(f"Full CNS replay state shape mismatch for unit {uid}")
        logits = np.concatenate([pooled, inputs], axis=1) @ weight + float(loaded["readout_bias"])
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
        part = original.sort_values("timestamp_s", kind="stable").copy()
        part["probability"] = probabilities.astype(np.float32)
        output.append(part)
    return pd.concat(output, ignore_index=True) if output else features.assign(probability=np.empty(0))
