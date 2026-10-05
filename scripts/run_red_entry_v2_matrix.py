#!/usr/bin/env python3
"""Frozen bounded RED-entry matrix. No training is performed at import time."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import itertools
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import joblib
import numpy as np

from pdm.data.project_prepare import _by_split, _stage_snapshot, load_snapshot, load_zone_limits
from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.projects import project_store
from pdm.red_entry_evaluation import _outcome
from pdm.red_entry_features import _flag, _starts
from pdm.red_entry_policy import reliable_origin
from pdm.red_entry_protocol import (
    INPUT_MODES,
    build_run_contract,
    canonical_json_hash,
    load_protocol,
    validate_physical_unit_split,
)
from pdm.red_entry_training import ENGINES, _batch, train_red_entry_run
from pdm.signal_training import average_training_duration_s

ROOT = Path(__file__).resolve().parents[1]
BOUNDS = dict(
    epochs=20,
    patience=5,
    maximum_origins_per_physical_unit=64,
    history_length=16,
    hidden_size=32,
    learning_rate=0.001,
)


def full_graph_freeze(engines):
    if "full_cns" not in engines:
        return None
    from pdm.connectome.anatomy import SOMA_ALLOWLIST
    from pdm.connectome.sources import default_malemcns_path

    source = default_malemcns_path()
    return {
        str(p.resolve()): sha256_file(p) if p.is_file() else None
        for p in (source, source.with_name(SOMA_ALLOWLIST[0]))
    }


def csv_values(value):
    return [x.strip() for x in value.split(",") if x.strip()]


def file_hashes(directory):
    return {p.name: sha256_file(p) for p in sorted(Path(directory).iterdir()) if p.is_file()}


def implementation_hashes(root=ROOT):
    """Freeze the complete implementation, including indirect dependencies."""
    root = Path(root)
    sources = set((root / "src/pdm").rglob("*.py"))
    sources.update(root / name for name in (
        "scripts/run_red_entry_v2_matrix.py", "configs/red_entry_protocol.yaml",
        "pyproject.toml", "requirements-lock.txt",
    ))
    return {str(path.relative_to(root)): sha256_file(path) for path in sorted(sources)}


def runtime_versions():
    """Read installed runtime identity locally; ordering is deterministic."""
    packages = sorted((dist.metadata["Name"], dist.version)
                      for dist in importlib.metadata.distributions() if dist.metadata["Name"])
    return {"python": sys.version, "implementation": sys.implementation.name,
            "packages": [list(package) for package in packages]}


def verify_hashes(directory, hashes):
    for name, expected in hashes.items():
        path = Path(directory) / name
        if (
            Path(name).name != name
            or path.is_symlink()
            or not path.is_file()
            or sha256_file(path) != expected
        ):
            raise ValueError(f"Artifact integrity mismatch: {path}")


def freeze_or_resume(directory, proposed, resume):
    directory = Path(directory)
    path = directory / "frozen_contract.json"
    digest = canonical_json_hash(proposed)
    if path.exists():
        if not resume:
            raise ValueError("Output already frozen; use --resume or a new output directory")
        saved = read_json(path)
        if (
            saved.get("contract_hash") != canonical_json_hash(saved["contract"])
            or saved["contract_hash"] != digest
        ):
            raise ValueError("Changed or tampered matrix freeze; create a new output directory")
        return saved
    if resume:
        raise ValueError("Cannot resume without frozen_contract.json")
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Fresh matrix output directory must be empty")
    frozen = {"contract": proposed, "contract_hash": digest}
    atomic_write_json(path, frozen)
    return frozen


def _project_freeze(project_id, record):
    data = load_snapshot(project_id, record["snapshot_id"])
    rule = load_zone_limits(project_id, data["snapshot_id"])
    if (
        canonical_json_hash(data["fingerprint"]) != record["fingerprint_hash"]
        or rule != record["rule"]
    ):
        raise ValueError(f"Research source freeze changed: {project_id}")
    contract = build_run_contract(data, effective_thresholds=rule)
    return {
        "project_id": project_id,
        "snapshot_id": data["snapshot_id"],
        "files": file_hashes(data["dir"]),
        "contract": contract,
        "folds": record["grouped_development_folds"],
    }, data


def fold_split(data, assignment):
    split = {
        **data["split"],
        "train": assignment["train"],
        "validation": assignment["validation"],
        "protocol": "frozen_grouped_development_v2",
    }
    if set(split["train"]) | set(split["validation"]) != set(data["split"]["train"]) | set(
        data["split"]["validation"]
    ):
        raise ValueError("Development fold must cover exactly original Train + Validation")
    validate_physical_unit_split(data["units"], split)
    # Membership is frozen above; descriptive metadata must describe this fold,
    # rather than the parent split. Outcomes never influence the assignment.
    parts = ("train", "validation", "test", "holdout")
    assigned = {str(uid) for part in parts for uid in split.get(part, [])}
    split["unassigned"] = sorted(set(data["units"].unit_id.astype(str)) - assigned)
    counts = {part: len(split.get(part, [])) for part in (*parts, "unassigned")}
    for part, count in counts.items():
        if part in ("train", "validation", "test") or f"n_{part}" in split:
            split[f"n_{part}"] = count
        event_key = f"n_{part}_events"
        if event_key in split:
            if "event_observed" not in data["units"]:
                raise ValueError(f"Cannot recompute {event_key} without unit event_observed")
            selected = data["units"].unit_id.astype(str).isin(split.get(part, []))
            split[event_key] = int(data["units"].loc[selected, "event_observed"].sum())
    realized_parts = set(data["split"].get("realized_counts", {})) | {
        "train", "validation", "test"
    }
    split["realized_counts"] = {part: counts[part] for part in sorted(realized_parts)}
    return split


def stage_fold(project_id, data, assignment, fold_index):
    """Publish an independent immutable snapshot without activating it."""
    store = project_store()
    split = fold_split(data, assignment)

    def write_data(path):
        for name in ("features.parquet", "units.parquet", "feature_schema.json"):
            shutil.copyfile(data["dir"] / name, path / name)

    fingerprint = {
        k: v
        for k, v in data["fingerprint"].items()
        if k not in ("snapshot_id", "file_hashes", "split_hash", "project_id")
    }
    fingerprint.update(parent_snapshot_id=data["snapshot_id"], development_fold=fold_index)
    staged, sid, _ = _stage_snapshot(
        store.project_path(project_id) / "snapshots",
        project_id,
        write_data=write_data,
        split=split,
        report={
            **data["report"],
            "split_counts": split["realized_counts"],
            "by_split": _by_split(data["features"], data["units"], split),
            "evaluation_role": "grouped_development_exploratory",
        },
        fingerprint=fingerprint,
    )
    if (data["dir"] / "zone_limits.json").exists():
        shutil.copyfile(data["dir"] / "zone_limits.json", staged / "zone_limits.json")
    destination = store.snapshot_path(project_id, sid)
    staged.rename(destination)
    return {
        "project_id": project_id,
        "snapshot_id": sid,
        "fold": fold_index,
        "parent_snapshot_id": data["snapshot_id"],
        "files": file_hashes(destination),
    }


def task_key(task):
    return canonical_json_hash(task)[:24]


def save_manifest(path, manifest):
    manifest.pop("manifest_hash", None)
    manifest["manifest_hash"] = canonical_json_hash(manifest)
    atomic_write_json(path, manifest)


def execute_tasks(tasks, output, manifest, trainer=None):
    """Persist each transition; verified completed tasks alone are skipped."""
    trainer = trainer or train_red_entry_run
    store = project_store()
    for task in tasks:
        key = task_key(task)
        previous = manifest["tasks"].get(key)
        if previous and previous.get("task") != task:
            raise ValueError("Manifest task differs from frozen request")
        if previous and previous["status"] == "completed":
            verify_hashes(
                store.run_path(task["project_id"], previous["run_id"]), previous["artifacts"]
            )
            continue
        result = {"task": task, "status": "running", "started_at": time.time()}
        manifest["tasks"][key] = result
        save_manifest(Path(output) / "matrix_manifest.json", manifest)
        start = time.monotonic()
        try:
            run = trainer(
                task["project_id"], task["snapshot_id"], task["engine_id"], task["params"]
            )
            directory = store.run_path(task["project_id"], run["run_id"])
            verify_hashes(directory, run["artifacts"])
            result.update(
                status="completed",
                run_id=run["run_id"],
                artifacts=file_hashes(directory),
                selection=run["selection"],
                support_by_horizon=run["support_by_horizon"],
            )
        except (FileNotFoundError, ImportError) as error:
            result.update(status="unavailable", reason=f"{type(error).__name__}: {error}")
        except Exception as error:
            result.update(status="failed", reason=f"{type(error).__name__}: {error}")
        result["elapsed_s"] = time.monotonic() - start
        save_manifest(Path(output) / "matrix_manifest.json", manifest)
        print(
            f"{key} {task['project_id']} {task['engine_id']} {task['params']['input_mode']} seed={task['params']['seed']} {result['status']}",
            flush=True,
        )
    return manifest


def unit_equal_score(cells):
    by_unit = {}
    for uid, loss in cells.values():
        by_unit.setdefault(uid, []).append(loss)
    means = {u: float(np.mean(v)) for u, v in by_unit.items()}
    return {
        "brier": float(np.mean(list(means.values()))) if means else None,
        "known_supported_origins": len(cells),
        "physical_units": len(means),
        "per_unit": means,
    }


def supported_cells(origins, probabilities, grid, support):
    """NaN tails, unknown horizons and unreliable origins never become negatives."""
    result = []
    for column, horizon in enumerate(grid):
        cells = {}
        for index, row in origins.iterrows():
            outcome = _outcome(row, horizon)
            if (
                outcome is None
                or not support[column]
                or not np.isfinite(probabilities[index, column])
            ):
                continue
            key = (
                str(row.unit_id),
                str(row.get("episode_id", row.unit_id)),
                float(row.timestamp_s),
            )
            uid = str(row.get("physical_unit_id", row.unit_id))
            cells[key] = (uid, float((probabilities[index, column] - outcome) ** 2))
        result.append(cells)
    return result


def summarize_common(method_cells):
    keys = set.intersection(*(set(c) for c in method_cells.values())) if method_cells else set()
    return {
        method: unit_equal_score({k: cells[k] for k in keys})
        for method, cells in method_cells.items()
    }


def summarize_pairwise(method_cells):
    """Declared pairs retain overlap even if a third method has no support."""
    return [{"task_keys": [left, right],
             "common": summarize_common({left: method_cells[left], right: method_cells[right]})}
            for left, right in itertools.combinations(sorted(method_cells), 2)]


def last_value_signal_cells(data, origins, grid):
    """Persistence input is the recorded origin only; future rows score outcomes.

    Nearest recorded timestamp must equal origin+horizon within 1e-6 seconds
    (floating-point clock tolerance). No cadence rounding or interpolation.
    Entire origin-to-target path must have usable quality and no admitted gap,
    segment/cycle change or replacement. Event occurrence is not a signal head.
    """
    units = {}
    for uid, frame in data["features"].groupby("unit_id", sort=False):
        frame = frame.sort_values("timestamp_s").reset_index(drop=True)
        times = frame.timestamp_s.to_numpy(float)
        if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError("Signal evaluation requires a finite strictly increasing clock")
        good = np.isfinite(frame.signal.to_numpy(float))
        for name in ("quality_ok", "usable"):
            if name in frame:
                good &= frame[name].fillna(True).map(_flag).to_numpy(bool)
        if "quality_status" in frame:
            good &= frame.quality_status.fillna("ok").astype(str).str.lower().isin(
                {"good", "ok", "usable", "valid"}
            ).to_numpy()
        breaks = np.asarray(_starts(frame)) == np.arange(len(frame))
        if "segment_id" in frame:
            breaks[1:] |= frame.segment_id.to_numpy()[1:] != frame.segment_id.to_numpy()[:-1]
        units[str(uid)] = (frame.signal.to_numpy(float), times, good,
                           np.cumsum(~good), np.cumsum(breaks))
    result = [dict() for _ in grid]
    for _, row in origins.iterrows():
        # Baseline needs no fitted model's context/regime support.
        if not reliable_origin(row.drop(labels=["context_known"], errors="ignore")):
            continue
        signal, times, good, bad_count, break_count = units[str(row.unit_id)]
        start = int(np.searchsorted(times, float(row.timestamp_s)))
        if start == len(times) or abs(times[start] - float(row.timestamp_s)) > 1e-6 or not good[start]:
            continue
        prediction = float(signal[start])  # Never read a future input to predict.
        for column, horizon in enumerate(grid):
            desired = float(row.timestamp_s) + horizon
            right = int(np.searchsorted(times, desired))
            candidates = [i for i in (right - 1, right) if start < i < len(times)]
            if not candidates:
                continue
            end = min(candidates, key=lambda i: abs(times[i] - desired))
            if (abs(times[end] - desired) > 1e-6 or bad_count[end] != bad_count[start]
                    or break_count[end] != break_count[start]):
                continue
            key = (str(row.unit_id), str(row.get("episode_id", row.unit_id)), float(row.timestamp_s))
            result[column][key] = (str(row.get("physical_unit_id", row.unit_id)),
                                   abs(prediction - float(signal[end])))
    return result


def signal_score(cells, total):
    score = unit_equal_score(cells)
    return {"mae": score["brier"], "known_recorded_origins": score["known_supported_origins"],
            "physical_units": score["physical_units"], "per_unit": score["per_unit"],
            "total_sampled_origins": total,
            "coverage_fraction": len(cells) / total if total else None,
            "excluded_origins": total - len(cells)}


def diagnostic_reference(data, origins, sampled, protocol, events=None):
    """Data reachability is retrospective evidence, independent of action lead."""
    mean = average_training_duration_s(data)
    reference = mean / 3.0

    def reachability(frame, part):
        known, unknown_history, eligible, reachable = {}, {}, {}, {}
        if events is not None:
            for _, row in events.loc[events.split == part].iterrows():
                event_time = row.get("first_red_timestamp_s")
                if event_time is None or not np.isfinite(float(event_time)):
                    continue
                event = (str(row.unit_id), str(row.get("episode_id", row.unit_id)), float(event_time))
                uid = str(row.get("physical_unit_id", row.unit_id))
                if _flag(row.get("first_event_verified", False)):
                    known[event] = uid
                else:
                    unknown_history[event] = uid
        known_origins, reachable_origins = 0, 0
        for _, row in frame.iterrows():
            if (not reliable_origin(row.drop(labels=["context_known"], errors="ignore"))
                    or not bool(row.get("event_observed", False))):
                continue
            lead = float(row.event_time_s) - float(row.timestamp_s)
            if not np.isfinite(lead) or lead <= 0 or float(row.followup_duration_s) < lead:
                continue
            event = (str(row.unit_id), str(row.get("episode_id", row.unit_id)), float(row.event_time_s))
            uid = str(row.get("physical_unit_id", row.unit_id))
            if events is None:  # Compatibility for older origin-only fixtures.
                known[event] = uid
            if event not in known:
                continue
            eligible[event] = uid
            known_origins += 1
            if reference > 0 and lead >= reference:
                reachable[event] = uid
                reachable_origins += 1
        return {"known_reliable_events": len(known), "reachable_events": len(reachable),
                "unknown_history_first_recorded_red_events": len(unknown_history),
                "events_with_eligible_pre_event_origin": len(eligible),
                "no_eligible_pre_event_origin_events": len(known) - len(eligible),
                "eligible_but_short_lead_events": len(eligible) - len(reachable),
                "known_reliable_pre_event_origins": known_origins,
                "reachable_pre_event_origins": reachable_origins,
                "reachable_physical_units": len(set(reachable.values())),
                "reachable_event_ids": [list(k) for k in sorted(reachable)],
                "unreachable_or_no_eligible_origin_events": len(known) - len(reachable)}

    return {"average_training_duration_s": mean, "target_lead_s": reference,
            "status": "available" if reference > 0 else "unavailable_zero_train_duration",
            "definition": "one third of mean summed continuous Train-unit duration; gaps excluded; no Test contribution",
            "minimum_action_lead_s": protocol["operational_requirements"]["minimum_action_lead_s"],
            "interpretation": "retrospective data reachability, not model quality or operational admission",
            "by_part": {part: {"all_data": reachability(origins.loc[origins.split == part], part),
                               "sampled": reachability(sampled.loc[sampled.split == part], part)}
                        for part in ("train", "validation", "test")}}


def build_comparison(manifest):
    rows, grouped, diagnostics = [], {}, {}
    store = project_store()
    for key, result in manifest["tasks"].items():
        if result["status"] != "completed":
            continue
        task = result["task"]
        directory = store.run_path(task["project_id"], result["run_id"])
        verify_hashes(directory, result["artifacts"])
        contract = read_json(directory / "training_contract.json")
        state = read_json(directory / "feature_state.json")
        targets = joblib.load(directory / "targets.joblib")
        with np.load(directory / "predictions.npz") as saved:
            probabilities = saved["probability"].copy()
            indices = saved["origin_indices"].copy()
        data = load_snapshot(task["project_id"], task["snapshot_id"])
        schema = {**data["schema"], "thresholds": contract["red_rule"]["effective_thresholds"]}
        batch = _batch(data, targets, state, schema, indices, task["params"])
        origins = batch["origins"].reset_index(drop=True).copy()
        origins["context_known"] = batch["availability"] & batch["supported_regime"]
        snapshot_key = task["project_id"] + "|" + task["snapshot_id"]
        diagnostics.setdefault(snapshot_key, {
            "project_id": task["project_id"], "snapshot_id": task["snapshot_id"], "fold": task["fold"],
            **diagnostic_reference(data, targets["origins"], origins, contract["protocol"], targets["events"]),
        })
        for part in ("validation", "test"):
            selected = np.flatnonzero(origins["split"].to_numpy() == part)
            frame = origins.iloc[selected].reset_index(drop=True)
            cells = supported_cells(
                frame, probabilities[selected], contract["horizons_s"], result["support_by_horizon"]
            )
            signal_cells = last_value_signal_cells(data, frame, contract["horizons_s"])
            for h, native, signal in zip(contract["horizons_s"], cells, signal_cells):
                row = {
                    "task_key": key,
                    "project_id": task["project_id"],
                    "snapshot_id": task["snapshot_id"],
                    "fold": task["fold"],
                    "part": part,
                    "engine_id": task["engine_id"],
                    "input_mode": task["params"]["input_mode"],
                    "seed": task["params"]["seed"],
                    "horizon_s": h,
                    "total_sampled_origins": len(frame),
                    "native": unit_equal_score(native),
                    "last_value_signal": signal_score(signal, len(frame)),
                    "signal_unit": data["schema"].get("signal_unit", "unspecified_native_unit"),
                    "diagnostic_target_lead_s": diagnostics[snapshot_key]["target_lead_s"],
                    "diagnostic_reachable_events": diagnostics[snapshot_key]["by_part"][part]["sampled"]["reachable_events"],
                    "minimum_action_lead_s": diagnostics[snapshot_key]["minimum_action_lead_s"],
                }
                rows.append(row)
                group = (task["project_id"], task["snapshot_id"], part, h)
                grouped.setdefault(group, {})[key] = native
    for row in rows:
        group = (row["project_id"], row["snapshot_id"], row["part"], row["horizon_s"])
        row["common"] = summarize_common(grouped[group])[row["task_key"]]
    spread = {}
    for row in rows:
        group = "|".join(
            map(
                str,
                (
                    row["project_id"],
                    row["fold"],
                    row["part"],
                    row["engine_id"],
                    row["input_mode"],
                    row["horizon_s"],
                ),
            )
        )
        spread.setdefault(group, []).append(
            {
                "seed": row["seed"],
                "brier": row["common"]["brier"],
                "per_unit": row["common"]["per_unit"],
            }
        )
    return {
        "version": "red_entry_matrix_comparison_v1",
        "rows": rows,
        "pairwise_common": [
            {"project_id": group[0], "snapshot_id": group[1], "part": group[2],
             "horizon_s": group[3], **pair,
             "task_identities": {
                 key: {"engine_id": manifest["tasks"][key]["task"]["engine_id"],
                       "input_mode": manifest["tasks"][key]["task"]["params"]["input_mode"],
                       "seed": manifest["tasks"][key]["task"]["params"]["seed"],
                       "fold": manifest["tasks"][key]["task"]["fold"]}
                 for key in pair["task_keys"]}}
            for group, methods in grouped.items() for pair in summarize_pairwise(methods)
        ],
        "pairwise_scope": "all declared pairs of completed tasks on identical snapshot/part/horizon; known supported intersection only; no score-driven method filtering",
        "seed_unit_spread": spread,
        "status_counts": {
            s: sum(r["status"] == s for r in manifest["tasks"].values())
            for s in ("completed", "failed", "unavailable", "running")
        },
        "common_scope": "completed methods/modes/seeds within identical snapshot, part and horizon; inspect status_counts for missing methods",
        "metric": "physical_unit_equal_known_outcome_brier_not_IPCW",
        "historical_test_status": "explored; descriptive only; never used for selection",
        "signal_mae": {
            "status": "evaluated" if rows else "no_completed_tasks",
            "method": "last_value",
            "metric": "physical_unit_equal_MAE_in_native_signal_units",
            "prediction": "recorded signal at origin only; identical sampled origins and horizons to event comparison",
            "outcome_matching": "nearest recorded origin+horizon timestamp within 1e-6 seconds only; no interpolation or cadence rounding",
            "coverage": "excludes missing exact-step target, incomplete horizon, bad quality, gaps and cycle/segment changes; coverage per row",
            "interpretation": "separate numerical baseline; not event-time quality; hazard adapters have no fitted signal head",
        },
        "diagnostic_reference": list(diagnostics.values()),
        "quality_gate": {"can_pass": False, "status": "requirements_unset"},
    }


def save_comparison(output, comparison):
    output = Path(output)
    atomic_write_json(output / "comparison.json", comparison)
    fields = [
        "task_key",
        "project_id",
        "snapshot_id",
        "fold",
        "part",
        "engine_id",
        "input_mode",
        "seed",
        "horizon_s",
        "total_sampled_origins",
        "native_brier",
        "native_origins",
        "native_units",
        "common_brier",
        "common_origins",
        "common_units",
        "last_value_mae", "last_value_origins", "last_value_units", "last_value_coverage_fraction",
        "signal_unit", "diagnostic_target_lead_s", "diagnostic_reachable_events", "minimum_action_lead_s",
    ]
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for row in comparison["rows"]:
            record = {k: row[k] for k in fields if k in row}
            for scope in ("native", "common"):
                record.update(
                    {
                        scope + "_brier": row[scope]["brier"],
                        scope + "_origins": row[scope]["known_supported_origins"],
                        scope + "_units": row[scope]["physical_units"],
                    }
                )
            signal = row["last_value_signal"]
            record.update(last_value_mae=signal["mae"], last_value_origins=signal["known_recorded_origins"],
                          last_value_units=signal["physical_units"], last_value_coverage_fraction=signal["coverage_fraction"])
            writer.writerow(record)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects", required=True)
    parser.add_argument("--engines", default="all")
    parser.add_argument("--input-modes", default="age_context,sensor_only,hybrid")
    parser.add_argument("--seeds", default="42,73")
    parser.add_argument("--development-folds", type=int, choices=(0, 2), default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--research-freeze",
        type=Path,
        default=ROOT / "output/red-entry-v2-20261002/research_data_freeze.json",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    projects, modes = csv_values(args.projects), csv_values(args.input_modes)
    engines = list(ENGINES) if args.engines == "all" else csv_values(args.engines)
    seeds = [int(x) for x in csv_values(args.seeds)]
    if not projects or not engines or not modes or not seeds or len(set(projects)) != len(projects):
        parser.error("Nonempty projects, engines, modes and seeds required; projects unique")
    if (
        set(engines) - set(ENGINES)
        or set(modes) - set(INPUT_MODES)
        or len(set(seeds)) != len(seeds)
    ):
        parser.error("Unsupported engines/modes or duplicate seeds")
    source = read_json(args.research_freeze)
    expected_root = args.research_freeze.resolve().parent / "research-projects"
    configured = os.environ.get("PDM_PROJECTS_ROOT")
    if not configured or Path(configured).expanduser().resolve() != expected_root:
        parser.error(f"Set PDM_PROJECTS_ROOT={expected_root}; original projects are forbidden")
    store = project_store()
    if load_protocol() != source["protocol"]:
        raise ValueError("Protocol differs from research freeze")
    records = {p["project_id"]: p for p in source["projects"]}
    frozen_projects, loaded = [], {}
    for p in projects:
        frozen, data = _project_freeze(p, records[p])
        frozen_projects.append(frozen)
        loaded[p] = data
        if args.development_folds:
            if len(frozen["folds"]) != 2:
                raise ValueError("Exactly two predeclared development folds required")
            for assignment in frozen["folds"]:
                fold_split(data, assignment)
    bounds = {
        **BOUNDS,
        **({"epochs": 1, "maximum_origins_per_physical_unit": 4} if args.smoke else {}),
    }
    proposed = {
        "version": "red_entry_matrix_frozen_v1",
        "projects_root": str(store.root.resolve()),
        "research_freeze_hash": sha256_file(args.research_freeze),
        "projects": frozen_projects,
        "protocol": load_protocol(),
        "bounds": bounds,
        "smoke": args.smoke,
        "engines": engines,
        "input_modes": modes,
        "seeds": seeds,
        "development_folds": args.development_folds,
        "code_hashes": implementation_hashes(),
        "runtime_versions": runtime_versions(),
        "selection": "Train fit, Validation checkpoint/calibration, Test never selects",
        "comparison": "unit-equal known-outcome Brier; native + common origins per horizon; no IPCW",
        "no_synthetic_fallback": True,
        "full_graph_sources": full_graph_freeze(engines),
    }
    output = args.output_dir.resolve()
    frozen = freeze_or_resume(output, proposed, args.resume)
    manifest_path = output / "matrix_manifest.json"
    if args.resume:
        manifest = read_json(manifest_path)
        if manifest.get("manifest_hash") != canonical_json_hash(
            {k: v for k, v in manifest.items() if k != "manifest_hash"}
        ):
            raise ValueError("Matrix manifest integrity mismatch")
        if manifest["contract_hash"] != frozen["contract_hash"]:
            raise ValueError("Manifest freeze binding mismatch")
        if manifest.get("reports"):
            verify_hashes(output, manifest["reports"])
        for snap in manifest["snapshots"]:
            verify_hashes(
                store.snapshot_path(snap["project_id"], snap["snapshot_id"]), snap["files"]
            )
    else:
        snapshots = [
            {
                "project_id": p,
                "snapshot_id": loaded[p]["snapshot_id"],
                "fold": "original",
                "files": file_hashes(loaded[p]["dir"]),
            }
            for p in projects
        ]
        for p in projects:
            for i, assignment in enumerate(
                records[p]["grouped_development_folds"][: args.development_folds]
            ):
                snapshots.append(stage_fold(p, loaded[p], assignment, i + 1))
        manifest = {
            "version": "red_entry_matrix_manifest_v1",
            "contract_hash": frozen["contract_hash"],
            "snapshots": snapshots,
            "tasks": {},
        }
        save_manifest(manifest_path, manifest)
    tasks = [
        dict(
            project_id=s["project_id"],
            snapshot_id=s["snapshot_id"],
            fold=s["fold"],
            engine_id=e,
            params={
                **bounds,
                "input_mode": m,
                "seed": seed,
                "horizons_s": load_protocol()["horizons_s"][
                    loaded[s["project_id"]]["schema"]["source_kind"]
                ],
            },
        )
        for s, e, m, seed in itertools.product(manifest["snapshots"], engines, modes, seeds)
    ]
    print(
        f"Frozen {len(tasks)} tasks; epochs <= {bounds['epochs']}, cap/unit={bounds['maximum_origins_per_physical_unit']}; Full CNS uses all 166700 neurons. Wall time unknown until measured; no quality admission.",
        flush=True,
    )
    if set(manifest["tasks"]) - {task_key(t) for t in tasks}:
        raise ValueError("Manifest contains tasks outside frozen matrix")
    execute_tasks(tasks, output, manifest)
    save_comparison(output, build_comparison(manifest))
    manifest["reports"] = {
        name: sha256_file(output / name) for name in ("comparison.json", "comparison.csv")
    }
    save_manifest(manifest_path, manifest)
    return 1 if any(r["status"] == "failed" for r in manifest["tasks"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
