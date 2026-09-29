"""Read-only presentation helpers for the frozen future-red-entry matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from pdm import ui_copy

MATRIX_SCHEMA = "future_red_matrix_v1"
MODEL_LABELS = {
    "gru": "GRU",
    "lstm": "LSTM",
    "fly": "Fly reservoir",
    "random": "Random reservoir",
    "full_cns": "Full MaleCNS",
    "baseline:always_no_entry": "Baseline: always no entry",
    "baseline:current_red_persistence": "Baseline: current RED persistence",
    "baseline:trend_to_red": "Baseline: trend to RED",
}


def latest_matrix(matrix_root: Path, dataset_id: str | None = None) -> tuple[Path, dict[str, Any]] | None:
    """Return the newest schema-valid matrix for the requested project."""
    candidates: list[tuple[str, Path, dict[str, Any]]] = []
    for manifest_path in matrix_root.glob("*/run_manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema_version") != MATRIX_SCHEMA
            or manifest.get("run_id") != manifest_path.parent.name
            or not (manifest_path.parent / "metrics.csv").is_file()
            or not (manifest.get("run_config") or {}).get("target_artifacts")
        ):
            continue
        targets = (manifest.get("run_config") or {}).get("target_artifacts") or {}
        if dataset_id is not None and dataset_id not in targets:
            continue
        candidates.append((str(manifest.get("created_at") or ""), manifest_path.parent, manifest))
    if not candidates:
        return None
    _, directory, manifest = max(candidates, key=lambda item: (item[0], item[1].name))
    return directory, manifest


def load_test_metrics(directory: Path, dataset_id: str) -> list[dict[str, Any]]:
    """Load unique test metric rows for one dataset from the aggregate CSV."""
    try:
        with (directory / "metrics.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error):
        return []
    selected = [row for row in rows if row.get("dataset_id") == dataset_id and row.get("split") == "test"]
    unique = {row.get("architecture", ""): row for row in selected}
    return [unique[key] for key in sorted(unique)]


def matching_full_cns_run(
    matrix_root: Path, matrix_manifest: dict[str, Any]
) -> tuple[Path, dict[str, Any]] | None:
    """Find the separate Full CNS run only when its target and artifacts match."""
    expected = ((matrix_manifest.get("run_config") or {}).get("target_artifacts") or {}).get("bearings") or {}
    expected_id = expected.get("artifact_id")
    expected_hash = expected.get("targets_sha256")
    if not expected_id or not expected_hash:
        return None
    run_dir = matrix_root.parent / "bearings" / "models" / "full_cns_seed42"
    manifest = _read_json(run_dir / "manifest.json")
    contract = manifest.get("model_contract") or {}
    target_manifest = manifest.get("target_manifest") or {}
    provenance = manifest.get("provenance") or {}
    try:
        matching_horizon = float(contract.get("target_horizon_s") or 0) == float(expected.get("horizon_s") or 0)
        matching_target_horizon = float(target_manifest.get("horizon_s") or 0) == float(expected.get("horizon_s") or 0)
    except (TypeError, ValueError):
        matching_horizon = False
        matching_target_horizon = False
    if (
        manifest.get("schema_version") != "future_red_binary_v1"
        or not isinstance((manifest.get("metrics") or {}).get("test"), dict)
        or contract.get("architecture") != "full_malecns_future_red_v1"
        or contract.get("dataset_id") != "bearings"
        or contract.get("target") != "future_red_entry_binary"
        or contract.get("uses_rul_target") is not False
        or contract.get("target_artifact_id") != expected_id
        or target_manifest.get("schema_version") != "future_sensor_red_entry_targets_v1"
        or target_manifest.get("dataset_id") != "bearings"
        or target_manifest.get("artifact_id") != expected_id
        or target_manifest.get("targets_sha256") != expected_hash
        or not matching_target_horizon
        or provenance.get("target_artifact_id") != expected_id
        or provenance.get("target_sha256") != expected_hash
        or not matching_horizon
    ):
        return None
    try:
        from pdm.future_red_full_cns import verify_full_cns_future_red_artifacts
        from pdm.io_util import sha256_file

        verified = verify_full_cns_future_red_artifacts(run_dir)
        result_names = ("predictions.parquet", "per_unit_metrics.csv",
                        "baseline_predictions.parquet", "baseline_metrics.json")
        hashes = verified.get("artifact_hashes") or {}
        if any(
            not hashes.get(name) or sha256_file(run_dir / name) != hashes[name]
            for name in result_names
        ):
            return None
    except Exception:  # Invalid or incomplete artifacts should not enter a comparison.
        return None
    if not isinstance(verified, dict) or verified.get("model_contract") != contract:
        return None
    return run_dir, verified


def full_cns_test_metric_row(manifest: dict[str, Any], source: str) -> dict[str, Any] | None:
    metrics = (manifest.get("metrics") or {}).get("test")
    if not isinstance(metrics, dict):
        return None
    event = metrics.get("event_level") or {}
    row = {"dataset_id": "bearings", "architecture": "full_cns", "split": "test", "source": source}
    for name in (
        "known_rows", "positive_rows", "tp", "fp", "fn", "tn", "precision", "recall", "f1",
        "brier_score", "average_precision",
    ):
        row[name] = metrics.get(name)
    for name in (
        "event_observed_units", "detected_event_units", "missed_event_units", "unknown_censored_units",
        "event_recall", "mean_lead_time_s", "alert_burden_fraction",
    ):
        row[name] = event.get(name)
    return row


def load_target_split_counts(
    matrix_root: Path, dataset_id: str, target_reference: dict[str, Any]
) -> dict[str, Any] | None:
    """Read reason-specific masked counts from the target file bound to a matrix."""
    artifact_id = target_reference.get("artifact_id")
    expected_hash = target_reference.get("targets_sha256")
    if not artifact_id or not expected_hash:
        return None
    target_dir = matrix_root.parent / dataset_id / "targets" / str(artifact_id)
    manifest = _read_json(target_dir / "manifest.json")
    if (
        manifest.get("schema_version") != "future_sensor_red_entry_targets_v1"
        or manifest.get("dataset_id") != dataset_id
        or manifest.get("artifact_id") != artifact_id
        or manifest.get("targets_sha256") != expected_hash
    ):
        return None
    try:
        from pdm.io_util import sha256_file

        target_path = target_dir / str(manifest.get("targets_file") or "")
        if not target_path.is_file() or sha256_file(target_path) != expected_hash:
            return None
    except OSError:
        return None
    split = ((manifest.get("split_counts") or {}).get("test") or {})
    return split if isinstance(split, dict) else None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def render_training(dataset_id: str, matrix_root: Path) -> None:
    """Launch and inspect the selected dataset's future-red model training."""
    import pandas as pd
    import streamlit as st

    from pdm.cli import spawn_worker
    from pdm.data.prepare import processed_ready
    from pdm.worker import read_status, worker_alive

    label = "Bearings" if dataset_id == "bearings" else "Filters"
    st.header(f"Future-red entry training — {label}")
    horizon = "1,800 seconds (30 minutes)" if dataset_id == "bearings" else "20 seconds"
    st.markdown(
        f"This workflow predicts entry into a saved **RED sensor zone** within **{horizon}**. "
        "It does not predict remaining useful life or a confirmed equipment failure."
    )
    st.caption(
        "Labels come from the saved signal-defined zone rules. Bearings use a 1,800-second horizon; "
        "filters use a 20-second horizon. Health zones remains a separate sensor-state replay."
    )

    current = latest_matrix(matrix_root, dataset_id)
    if current is None:
        st.info("No valid future-red matrix is available yet.")
    else:
        directory, manifest = current
        st.caption(f"Current matrix: `{manifest['run_id']}` · created {manifest.get('created_at', 'time unavailable')}")
        status = _read_json(directory / "status.json")
        config = manifest.get("run_config") or {}
        architectures = config.get("architectures") or []
        rows = []
        for architecture in architectures:
            state = status.get(f"{dataset_id}/{architecture}", {})
            rows.append({
                "Model": MODEL_LABELS.get(architecture, architecture),
                "Status": str(state.get("status") or "pending").title(),
                "Source": f"Matrix run `{manifest['run_id']}`",
            })
        full_cns = matching_full_cns_run(matrix_root, manifest) if dataset_id == "bearings" else None
        if full_cns:
            full_dir, full_manifest = full_cns
            rows.append({
                "Model": MODEL_LABELS["full_cns"],
                "Status": "Completed",
                "Source": f"Separate run `{full_dir.name}`",
            })
            target_id = (full_manifest.get("model_contract") or {}).get("target_artifact_id")
            st.caption(
                f"Full MaleCNS is a separate future-red run bound to matrix target `{target_id}`; "
                "its saved readout and connectome artifact hashes were verified."
            )
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        metrics = load_test_metrics(directory, dataset_id)
        for row in metrics:
            row["source"] = f"Matrix run: {manifest['run_id']}"
        if full_cns:
            full_dir, full_manifest = full_cns
            row = full_cns_test_metric_row(full_manifest, f"Separate run: {full_dir.name}")
            if row:
                metrics.append(row)
        if metrics:
            st.markdown("#### Test split snapshot")
            st.dataframe(_metrics_frame(pd, metrics), width="stretch", hide_index=True)

    st.markdown("#### Train models")
    st.caption("Select model families, then start a new run. Validation selects checkpoints; the test split is evaluated after the model is frozen.")
    selected = st.multiselect(
        "Model families", ["GRU", "LSTM", "Fly reservoir", "Random reservoir"],
        default=["GRU"], key=f"future_red_models:{dataset_id}", help=ui_copy.LEGACY_MODEL_FAMILIES_HELP,
    )
    choices = {"GRU": "gru", "LSTM": "lstm", "Fly reservoir": "fly", "Random reservoir": "random"}
    can_train = processed_ready(dataset_id) and not worker_alive() and bool(selected)
    if st.button("Start training", type="primary", disabled=not can_train, key=f"future_red_start:{dataset_id}",
                 help=ui_copy.LEGACY_START_TRAINING_HELP):
        spawn_worker({
            "kind": "future_red_matrix", "dataset_id": dataset_id,
            "architectures": [choices[label] for label in selected],
        })
        st.rerun()
    if not processed_ready(dataset_id):
        st.info("Import and inspect source data before training.")
    status = read_status()
    if status.get("kind") == "future_red_matrix" and status.get("dataset_id") == dataset_id:
        state = status.get("status")
        if state == "training" and worker_alive():
            st.info("Training is running. Model progress appears in the run table above.")
            @st.fragment(run_every=3)
            def live_matrix_progress() -> None:
                manifests = sorted(matrix_root.glob("*/run_manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True)
                for manifest_path in manifests:
                    manifest = _read_json(manifest_path)
                    if dataset_id not in ((manifest.get("run_config") or {}).get("datasets") or []):
                        continue
                    states = _read_json(manifest_path.parent / "status.json")
                    rows = [
                        {"Model": MODEL_LABELS.get(key.split("/", 1)[1], key),
                         "Status": str(value.get("status") or "pending").title()}
                        for key, value in states.items() if key.startswith(dataset_id + "/")
                    ]
                    if rows:
                        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
                    break

            live_matrix_progress()
        elif state == "failed":
            st.error(status.get("message") or "Training failed")
        elif state == "completed":
            st.success("Training finished. Open Results to review held-out test metrics.")
    with st.expander("Command-line run"):
        st.code(
            f"python scripts/run_future_red_matrix.py --datasets {dataset_id} "
            "--architectures gru,lstm,fly,random",
            language="bash",
        )


def render_report(dataset_id: str, matrix_root: Path) -> None:
    """Render test-set future-red metrics and their label limitations."""
    import pandas as pd
    import streamlit as st

    label = "Bearings" if dataset_id == "bearings" else "Filters"
    st.header(f"Future-red entry model report — {label}")
    current = latest_matrix(matrix_root, dataset_id)
    if current is None:
        st.info("No valid future-red matrix is available yet.")
        return
    directory, manifest = current
    config = manifest.get("run_config") or {}
    targets = config.get("target_artifacts") or {}
    target = targets.get(dataset_id) or {}
    horizon = target.get("horizon_s")
    if dataset_id == "bearings":
        st.markdown(f"Target: first saved RED zone entry within **{horizon or 1800:g} seconds** (30 minutes).")
        st.caption("This is a sensor-zone entry target. It is not remaining useful life or confirmed failure time.")
    else:
        st.markdown(f"Target: first saved RED zone entry within **{horizon or 20:g} seconds**.")
        st.warning(
            "Filter test has no observed RED-entry events in the frozen target split. Event recall and lead time "
            "cannot be estimated there; the row metrics mainly describe known negative windows."
        )
        st.caption(
            "Filter RED labels use provisional laboratory sensor-zone rules based on pressure and operating context. "
            "They are weak labels, not confirmed failures. See Model Report → Health zones for the standalone replay."
        )
        target_counts = load_target_split_counts(matrix_root, dataset_id, target)
        if target_counts is not None:
            masked = int(target_counts.get("masked", 0) or 0)
            censored = int(target_counts.get("masked_censor", 0) or 0)
            quality = int(target_counts.get("masked_quality", 0) or 0)
            gaps = int(target_counts.get("masked_gap", 0) or 0)
            st.caption(
                f"Filter test target rows masked from scoring: {masked:,} total "
                f"({censored:,} censored before the 20-second horizon, {quality:,} quality-masked, {gaps:,} gap-masked)."
            )
    st.caption(f"Matrix `{manifest['run_id']}` · {manifest.get('created_at', 'time unavailable')}")

    metrics = load_test_metrics(directory, dataset_id)
    for row in metrics:
        row["source"] = f"Matrix run: {manifest['run_id']}"
    if dataset_id == "bearings":
        full_cns = matching_full_cns_run(matrix_root, manifest)
        if full_cns:
            full_dir, full_manifest = full_cns
            row = full_cns_test_metric_row(full_manifest, f"Separate run: {full_dir.name}")
            if row:
                metrics.append(row)
                st.caption(
                    f"Full MaleCNS metrics come from separate run `{full_dir.name}`; its frozen target "
                    "matches this matrix and all recorded artifact hashes verified."
                )
    if not metrics:
        st.info("This matrix does not have test metrics for the selected dataset yet.")
        return
    st.dataframe(_metrics_frame(pd, metrics), width="stretch", hide_index=True)

    status = _read_json(directory / "status.json")
    in_progress = [
        MODEL_LABELS.get(architecture, architecture)
        for architecture in config.get("architectures", [])
        if status.get(f"{dataset_id}/{architecture}", {}).get("status") not in {"completed", "failed"}
    ]
    if in_progress:
        st.info("Still running: " + ", ".join(in_progress))
    st.caption(
        "Test metrics are for the held-out split after checkpoint and threshold selection on validation. "
        "They are exploratory, label-bound results; compare precision, event recall, lead time, and alert burden together."
    )


def render_comparison(dataset_id: str, matrix_root: Path) -> None:
    """Compare models evaluated on one frozen future-red target and test split."""
    import pandas as pd
    import streamlit as st

    st.header(f"Compare models — {'Bearings' if dataset_id == 'bearings' else 'Filters'}")
    current = latest_matrix(matrix_root, dataset_id)
    if current is None:
        st.info("Train models to create a comparable future-red test result.")
        return
    directory, manifest = current
    rows = load_test_metrics(directory, dataset_id)
    if dataset_id == "bearings":
        full_cns = matching_full_cns_run(matrix_root, manifest)
        if full_cns:
            full_dir, full_manifest = full_cns
            full_row = full_cns_test_metric_row(full_manifest, f"Separate run: {full_dir.name}")
            if full_row:
                rows.append(full_row)
    if not rows:
        st.info("This matrix has no held-out test metrics for this project yet.")
        return
    st.caption(
        f"Matrix {manifest['run_id']} · same project, frozen target, and held-out test split. "
        "The separately trained Full MaleCNS is included only when its target and artifacts match."
    )
    options = [str(row["architecture"]) for row in rows]
    selected = st.multiselect(
        "Models to compare", options, default=options,
        format_func=lambda key: MODEL_LABELS.get(key, key),
        key=f"future_red_compare:{dataset_id}:{manifest['run_id']}", help=ui_copy.LEGACY_COMPARE_MODELS_HELP,
    )
    chosen = [row for row in rows if row["architecture"] in selected]
    if not chosen:
        st.info("Select at least one model to inspect its test metrics.")
        return
    st.dataframe(_metrics_frame(pd, chosen), width="stretch", hide_index=True)
    if dataset_id == "filters":
        st.warning(
            "The filter test split has no observed RED-entry events. Event recall and warning lead time "
            "cannot be ranked from this holdout."
        )
    else:
        st.caption(
            "Compare precision, missed events, warning lead time, and alert burden together. "
            "A high row-level score alone does not prove a useful maintenance warning."
        )


def _metrics_frame(pd: Any, rows: list[dict[str, Any]]) -> Any:
    frame = pd.DataFrame(rows)
    frame["Model"] = frame["architecture"].map(lambda key: MODEL_LABELS.get(key, key))
    columns = {
        "Model": "Model",
        "source": "Source",
        "known_rows": "Known test windows",
        "positive_rows": "RED-entry windows",
        "precision": "Precision",
        "recall": "Recall",
        "f1": "F1",
        "average_precision": "Average precision",
        "brier_score": "Brier score",
        "event_observed_units": "Units with event",
        "detected_event_units": "Events detected",
        "missed_event_units": "Events missed",
        "unknown_censored_units": "Unknown/censored units",
        "event_recall": "Event recall",
        "mean_lead_time_s": "Mean lead (s)",
        "alert_burden_fraction": "Alert burden",
    }
    view = frame[[column for column in columns if column in frame.columns]].rename(columns=columns)
    for column in (
        "Known test windows", "RED-entry windows", "Units with event", "Events detected", "Events missed",
        "Unknown/censored units",
    ):
        if column in view:
            view[column] = pd.to_numeric(view[column], errors="coerce")
    for column in ("Precision", "Recall", "F1", "Event recall", "Alert burden"):
        if column in view:
            view[column] = pd.to_numeric(view[column], errors="coerce").round(3)
    for column in ("Average precision", "Brier score"):
        if column in view:
            view[column] = pd.to_numeric(view[column], errors="coerce").round(3)
    if "Mean lead (s)" in view:
        view["Mean lead (s)"] = pd.to_numeric(view["Mean lead (s)"], errors="coerce").round(0)
    return view
