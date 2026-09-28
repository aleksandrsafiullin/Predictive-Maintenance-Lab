#!/usr/bin/env python3
"""Run a reproducible future-RED model matrix on frozen sensor-zone targets.

Example:
  python scripts/run_future_red_matrix.py --datasets bearings --architectures gru,lstm \
      --output-dir runs/_future_red/matrix/bearings_temporal_v1

Use ``--architectures all`` to include Full MaleCNS. Its per-unit state cache
survives interruption, and rerunning the same command with ``--resume`` reuses
completed work. A new output directory is required unless --resume is explicit.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from pdm.data.prepare import load_processed  # noqa: E402
from pdm.future_red_models import train_future_red_model  # noqa: E402
from pdm.future_red_targets import (  # noqa: E402
    DEFAULT_HORIZONS_S,
    build_future_red_targets,
    load_future_red_targets,
)
from pdm.io_util import sha256_file  # noqa: E402

ARCHITECTURES = ("gru", "lstm", "fly", "random", "full_cns")
DATASETS = ("bearings", "filters")
ARCH_ALIASES = {
    "gru": "gru", "lstm": "lstm", "fly": "fly_connectome_reservoir",
    "random": "random_reservoir", "full_cns": "full_cns",
}
RUN_SCHEMA = "future_red_matrix_v1"


def _json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _parse_list(value: str, allowed: tuple[str, ...], label: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(allowed)
    values = [part.strip().lower() for part in value.split(",") if part.strip()]
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"Unknown {label}: {', '.join(invalid)} (choose from {', '.join(allowed)} or all)")
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{label} must contain unique, non-empty names")
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="all", help="Comma-separated bearings,filters or all (default: all)")
    parser.add_argument("--architectures", default="gru,lstm,fly,random",
                        help="Comma-separated gru,lstm,fly,random,full_cns or all; all includes expensive Full CNS")
    parser.add_argument("--output-dir", type=Path, help="New matrix run directory; defaults to runs/_future_red/matrix/<UTC timestamp>")
    parser.add_argument("--resume", action="store_true", help="Resume only when run configuration and frozen targets match")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30, help="Temporal model maximum epochs; Full CNS readout uses --readout-epochs")
    parser.add_argument("--readout-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fly-nodes", type=int, default=1000)
    parser.add_argument("--fly-source", type=Path, help="Optional real connectome source for Fly and degree-matched Random")
    parser.add_argument("--full-cns-source", type=Path, help="Optional MaleCNS source path (defaults to model resolver)")
    parser.add_argument("--full-cns-cache-dir", type=Path, help="Shared resumable per-unit trajectory cache; defaults inside run directory")
    parser.add_argument("--full-cns-config", type=Path, help="Optional JSON config used by Full CNS preprocessing")
    return parser


def _metrics_rows(dataset: str, architecture: str, result: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for split, metrics in (result.get("metrics") or {}).items():
        flat = {"dataset_id": dataset, "architecture": architecture, "split": split}
        for name in ("known_rows", "positive_rows", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "brier_score", "average_precision"):
            flat[name] = metrics.get(name)
        event = metrics.get("event_level") or {}
        for name in ("event_observed_units", "detected_event_units", "missed_event_units", "unknown_censored_units", "event_recall", "mean_lead_time_s", "alert_burden_fraction"):
            flat[name] = event.get(name)
        rows.append(flat)
    for base, by_split in (result.get("baseline_metrics") or {}).items():
        for split, metrics in by_split.items():
            flat = {"dataset_id": dataset, "architecture": f"baseline:{base}", "split": split}
            for name in ("known_rows", "positive_rows", "tp", "fp", "fn", "tn", "precision", "recall", "f1", "brier_score", "average_precision"):
                flat[name] = metrics.get(name)
            event = metrics.get("event_level") or {}
            for name in ("event_observed_units", "detected_event_units", "missed_event_units", "unknown_censored_units", "event_recall", "mean_lead_time_s", "alert_burden_fraction"):
                flat[name] = event.get(name)
            rows.append(flat)
    return rows


def _write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["dataset_id", "architecture", "split", "known_rows", "positive_rows", "tp", "fp", "fn", "tn",
              "precision", "recall", "f1", "brier_score", "average_precision", "event_observed_units",
              "detected_event_units", "missed_event_units", "unknown_censored_units", "event_recall",
              "mean_lead_time_s", "alert_burden_fraction"]
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _summary(run_dir: Path, run_manifest: dict[str, Any], states: dict[str, Any]) -> None:
    entries = []
    metrics_rows = []
    for key, state in states.items():
        entry = {k: state.get(k) for k in ("dataset_id", "architecture", "status", "output_dir", "error")}
        entries.append(entry)
        if state.get("status") == "completed":
            model_manifest_path = Path(state["output_dir"]) / "manifest.json"
            if model_manifest_path.is_file():
                model_result = json.loads(model_manifest_path.read_text(encoding="utf-8"))
                metrics_rows.extend(_metrics_rows(state["dataset_id"], state["architecture"], model_result))
    _json_write(run_dir / "summary.json", {"run_id": run_manifest["run_id"], "run_config_sha256": run_manifest["run_config_sha256"], "models": entries})
    _write_metrics(run_dir / "metrics.csv", metrics_rows)


def _completed_output_is_valid(model_dir: Path, architecture: str) -> bool:
    """Only resume past a complete, hash-valid model artifact."""
    manifest_path = model_dir / "manifest.json"
    required = ("predictions.parquet", "baseline_predictions.parquet",
                "per_unit_metrics.csv", "baseline_metrics.json")
    if not manifest_path.is_file() or any(
        not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
        for name in required
    ):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        hashes = manifest.get("artifact_hashes") or {}
        expected = {
            "gru": {"model.pt"}, "lstm": {"model.pt"},
            "fly": {"reservoir_weights.npz", "graph.json", "readout.pt"},
            "random": {"reservoir_weights.npz", "graph.json", "parent_graph.json", "readout.pt"},
            "full_cns": {"readout.pt"},
        }[architecture]
        expected |= set(required)
        if not expected.issubset(hashes):
            return False
        if any(sha256_file(model_dir / name) != hashes[name] for name in expected):
            return False
        if architecture == "full_cns":
            core = (manifest.get("provenance") or {}).get("full_cns_artifact_hashes") or {}
            required_core = {"recurrent.npz", "inputs.npz", "graph.json", "layout.json"}
            if not required_core.issubset(core) or any(
                sha256_file(model_dir / "full_cns" / name) != core[name] for name in required_core
            ):
                return False
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    return True


def run(args: argparse.Namespace) -> Path:
    datasets = _parse_list(args.datasets, DATASETS, "datasets")
    architectures = _parse_list(args.architectures, ARCHITECTURES, "architectures")
    if args.history_length < 1 or args.epochs < 1 or args.readout_epochs < 1 or args.fly_nodes < 1:
        raise ValueError("history length, epochs, readout epochs, and Fly node count must be positive")
    pairs = [(d, a) for d in datasets for a in architectures if not (a == "full_cns" and d != "bearings")]
    if not pairs:
        raise ValueError("No valid dataset/architecture pairs selected; Full CNS supports bearings only")
    if args.full_cns_config:
        full_config = json.loads(args.full_cns_config.read_text(encoding="utf-8"))
        if not isinstance(full_config, dict):
            raise ValueError("Full CNS config must contain a JSON object")
    else:
        full_config = None

    targets_by_dataset = {}
    data_by_dataset = {}
    target_dirs = {}
    target_manifests = {}
    for dataset in datasets:
        expected = DEFAULT_HORIZONS_S[dataset]
        data = load_processed(dataset)
        artifact = build_future_red_targets(dataset)
        targets, target_manifest = load_future_red_targets(artifact["directory"], data=data)
        if float(target_manifest.get("horizon_s", -1)) != float(expected):
            raise ValueError(f"Frozen target horizon mismatch for {dataset}: expected {expected}, got {target_manifest.get('horizon_s')}")
        data_by_dataset[dataset] = data
        targets_by_dataset[dataset] = (targets, target_manifest)
        target_dirs[dataset] = artifact["directory"]
        target_manifests[dataset] = {
            "artifact_id": target_manifest.get("artifact_id"),
            "targets_sha256": target_manifest.get("targets_sha256"),
            "dataset_version": target_manifest.get("dataset_version"),
            "dataset_fingerprint_sha256": target_manifest.get("dataset_fingerprint_sha256"),
            "horizon_s": expected,
        }

    run_config = {
        "schema_version": RUN_SCHEMA,
        "datasets": datasets,
        "architectures": architectures,
        "pairs": pairs,
        "seed": int(args.seed),
        "history_length": int(args.history_length),
        "epochs": int(args.epochs),
        "readout_epochs": int(args.readout_epochs),
        "batch_size": int(args.batch_size),
        "hidden_size": int(args.hidden_size),
        "num_layers": int(args.num_layers),
        "patience": int(args.patience),
        "fly_nodes": int(args.fly_nodes),
        "fly_source": str(args.fly_source.resolve()) if args.fly_source else None,
        "full_cns_source": str(args.full_cns_source.resolve()) if args.full_cns_source else None,
        "full_cns_config": full_config,
        "target_policy": "frozen saved-zone entry targets; bearings horizon 1800 seconds; filters horizon 20 seconds; no test tuning",
        "target_artifacts": target_manifests,
    }
    # Compare the same JSON-normalized shape that is persisted in run_manifest.
    run_config = json.loads(json.dumps(run_config, sort_keys=True, default=str))
    config_hash = _digest(run_config)
    output_dir = args.output_dir or (REPO_ROOT / "runs" / "_future_red" / "matrix" / dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output_dir = output_dir.expanduser().resolve()
    run_manifest_path = output_dir / "run_manifest.json"
    if output_dir.exists():
        if not args.resume:
            raise FileExistsError(f"Output directory already exists; choose a new directory or pass --resume: {output_dir}")
        if not run_manifest_path.is_file():
            raise ValueError("Cannot resume: output directory has no run_manifest.json")
        old_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if old_manifest.get("run_config_sha256") != config_hash or old_manifest.get("run_config") != run_config:
            raise ValueError("Cannot resume: command configuration or frozen target artifact differs from run manifest")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    run_id = output_dir.name
    run_manifest = {"schema_version": RUN_SCHEMA, "run_id": run_id,
                    "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "run_config_sha256": config_hash, "run_config": run_config,
                    "output_policy": "one directory per matrix run; model outputs are isolated by dataset and architecture",
                    "test_policy": "validation selects checkpoints and thresholds; test split is evaluated once after model freeze"}
    if not run_manifest_path.exists():
        _json_write(run_manifest_path, run_manifest)
    states_path = output_dir / "status.json"
    states = json.loads(states_path.read_text(encoding="utf-8")) if states_path.exists() else {}
    _summary(output_dir, run_manifest, states)

    for dataset, architecture in pairs:
        key = f"{dataset}/{architecture}"
        model_dir = output_dir / dataset / architecture
        prior = states.get(key, {})
        if prior.get("status") == "completed" and _completed_output_is_valid(model_dir, architecture):
            print(f"[{key}] already complete; skipping")
            continue
        if prior.get("status") == "completed":
            print(f"[{key}] saved output is incomplete or failed hash verification; rerunning", flush=True)
        state = {"dataset_id": dataset, "architecture": architecture, "status": "running",
                 "output_dir": str(model_dir), "started_at": dt.datetime.now(dt.timezone.utc).isoformat()}
        states[key] = state
        _json_write(states_path, states)
        _summary(output_dir, run_manifest, states)
        print(f"[{key}] starting", flush=True)
        try:
            data = data_by_dataset[dataset]
            artifact_dir = target_dirs[dataset]
            if architecture == "full_cns":
                from pdm.future_red_full_cns import fit_full_cns_future_red

                cache_dir = args.full_cns_cache_dir or (model_dir / "state_cache")
                fit_full_cns_future_red(
                    targets=targets_by_dataset[dataset], output_dir=model_dir, data=data,
                    target_artifact_dir=artifact_dir, source_path=args.full_cns_source,
                    cache_dir=cache_dir, seed=args.seed, config=full_config,
                    epochs=args.readout_epochs,
                    progress=lambda message: print(f"[{key}] {message}", flush=True),
                )
            else:
                if architecture in {"fly", "random"}:
                    fit_kwargs = {
                        "epochs": args.readout_epochs,
                        "n_nodes": args.fly_nodes,
                        "source_path": args.fly_source,
                    }
                else:
                    fit_kwargs = {
                        "epochs": args.epochs, "batch_size": args.batch_size,
                        "hidden_size": args.hidden_size, "num_layers": args.num_layers,
                        "patience": args.patience,
                    }
                train_future_red_model(
                    dataset, ARCH_ALIASES[architecture], targets_dir=artifact_dir, output_dir=model_dir,
                    seed=args.seed, history_length=args.history_length, data=data, **fit_kwargs,
                )
            state.update({"status": "completed", "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            print(f"[{key}] completed", flush=True)
        except KeyboardInterrupt:
            state.update({"status": "interrupted", "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            states[key] = state
            _json_write(states_path, states)
            _summary(output_dir, run_manifest, states)
            raise
        except Exception as exc:
            state.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                          "updated_at": dt.datetime.now(dt.timezone.utc).isoformat()})
            print(f"[{key}] failed: {state['error']}", file=sys.stderr, flush=True)
        states[key] = state
        _json_write(states_path, states)
        _summary(output_dir, run_manifest, states)
    print(f"Matrix outputs: {output_dir}")
    return output_dir


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    try:
        output_dir = run(args)
    except (ValueError, FileExistsError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    states = json.loads((output_dir / "status.json").read_text(encoding="utf-8"))
    return 1 if any(state.get("status") == "failed" for state in states.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
