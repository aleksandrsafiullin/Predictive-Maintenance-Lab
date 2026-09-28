from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from pdm.future_red_full_cns import ADAPTER_VERSION
from pdm.future_red_models import REPLAY_CONTRACT_VERSION
from pdm.future_red_ui import (
    _metrics_frame,
    full_cns_test_metric_row,
    latest_matrix,
    load_target_split_counts,
    load_test_metrics,
    matching_full_cns_run,
)


def _write_matrix(root, run_id: str, created_at: str, *, valid: bool = True):
    directory = root / run_id
    directory.mkdir(parents=True)
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "future_red_matrix_v1" if valid else "old_schema",
                "run_id": run_id,
                "created_at": created_at,
                "run_config": {"target_artifacts": {"filters": {"horizon_s": 20}}},
            }
        ),
        encoding="utf-8",
    )
    (directory / "metrics.csv").write_text(
        "dataset_id,architecture,split,known_rows,positive_rows,event_observed_units,event_recall\n"
        "filters,gru,test,100,0,0,\n"
        "filters,baseline:always_no_entry,test,100,0,0,\n"
        "filters,gru,validation,100,4,2,0.5\n",
        encoding="utf-8",
    )
    return directory


def test_latest_matrix_selects_newest_schema_valid_run_and_unique_test_rows(tmp_path):
    root = tmp_path / "matrix"
    older = _write_matrix(root, "older", "2026-09-27T12:00:00Z")
    latest = _write_matrix(root, "latest", "2026-09-28T12:00:00Z")
    _write_matrix(root, "invalid", "2026-09-29T12:00:00Z", valid=False)

    result = latest_matrix(root)

    assert result is not None
    assert result[0] == latest
    rows = load_test_metrics(latest, "filters")
    assert {row["architecture"] for row in rows} == {"gru", "baseline:always_no_entry"}
    assert all(row["split"] == "test" for row in rows)
    assert older != latest


def test_latest_matrix_ignores_invalid_manifest_without_metrics(tmp_path):
    root = tmp_path / "matrix"
    directory = root / "incomplete"
    directory.mkdir(parents=True)
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "future_red_matrix_v1",
                "run_id": "incomplete",
                "run_config": {"target_artifacts": {"bearings": {}}},
            }
        ),
        encoding="utf-8",
    )

    assert latest_matrix(root) is None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_matching_full_cns(root: Path, *, artifact: bytes = b"connectome", target_hash: str = "targets-hash"):
    matrix_root = root / "_future_red" / "matrix"
    matrix_manifest = {
        "schema_version": "future_red_matrix_v1",
        "run_id": "matrix_match",
        "created_at": "2026-09-28T12:00:00Z",
        "run_config": {
            "architectures": ["gru", "lstm", "fly", "random"],
            "target_artifacts": {
                "bearings": {
                    "artifact_id": "future_red_entry_v1_match",
                    "targets_sha256": target_hash,
                    "horizon_s": 1800.0,
                },
                "filters": {
                    "artifact_id": "filter-target-v1",
                    "targets_sha256": "filter-target-hash",
                    "horizon_s": 20.0,
                }
            }
        }
    }
    matrix_dir = matrix_root / "matrix_match"
    matrix_dir.mkdir(parents=True)
    (matrix_dir / "run_manifest.json").write_text(json.dumps(matrix_manifest), encoding="utf-8")
    (matrix_dir / "metrics.csv").write_text(
        "dataset_id,architecture,split,known_rows,positive_rows,precision,recall,f1,brier_score,"
        "average_precision,event_observed_units,detected_event_units,missed_event_units,"
        "unknown_censored_units,event_recall,mean_lead_time_s,alert_burden_fraction\n"
        "bearings,gru,test,225,67,0.5,0.6,0.55,0.2,0.7,3,2,1,0,0.667,900,0.2\n"
        "filters,gru,test,100,0,,0,0,0.01,,0,0,0,0,,,\n",
        encoding="utf-8",
    )
    (matrix_dir / "status.json").write_text(
        json.dumps({f"bearings/{name}": {"status": "completed"} for name in matrix_manifest["run_config"]["architectures"]}),
        encoding="utf-8",
    )
    run_dir = root / "_future_red" / "bearings" / "models" / "full_cns_seed42"
    (run_dir / "full_cns").mkdir(parents=True)
    graph_path = run_dir / "full_cns" / "graph.json"
    graph_path.write_bytes(artifact)
    contract = {
        "version": REPLAY_CONTRACT_VERSION,
        "architecture": ADAPTER_VERSION,
        "dataset_id": "bearings",
        "target": "future_red_entry_binary",
        "uses_rul_target": False,
        "target_artifact_id": "future_red_entry_v1_match",
        "target_horizon_s": 1800.0,
        "full_cns_artifact_hashes": {"graph.json": _sha256(graph_path)},
    }
    readout_path = run_dir / "readout.pt"
    torch.save({"model_contract": contract}, readout_path)
    result_names = ("predictions.parquet", "per_unit_metrics.csv",
                    "baseline_predictions.parquet", "baseline_metrics.json")
    for name in result_names:
        (run_dir / name).write_bytes(b"saved result")
    manifest = {
        "schema_version": "future_red_binary_v1",
        "model_contract": contract,
        "target_manifest": {
            "schema_version": "future_sensor_red_entry_targets_v1",
            "dataset_id": "bearings",
            "artifact_id": "future_red_entry_v1_match",
            "targets_sha256": target_hash,
            "horizon_s": 1800.0,
        },
        "provenance": {"target_artifact_id": "future_red_entry_v1_match", "target_sha256": target_hash},
        "artifact_hashes": {
            "readout.pt": _sha256(readout_path),
            "full_cns/graph.json": _sha256(graph_path),
            **{name: _sha256(run_dir / name) for name in result_names},
        },
        "metrics": {
            "test": {
                "known_rows": 23,
                "positive_rows": 4,
                "precision": 0.5,
                "recall": 0.75,
                "f1": 0.6,
                "brier_score": 0.23,
                "average_precision": 0.8,
                "event_level": {
                    "event_observed_units": 2,
                    "detected_event_units": 1,
                    "missed_event_units": 1,
                    "unknown_censored_units": 3,
                    "event_recall": 0.5,
                    "mean_lead_time_s": 900.0,
                    "alert_burden_fraction": 0.1,
                },
            }
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return matrix_root, matrix_manifest, run_dir, manifest


def test_matching_full_cns_is_added_only_after_target_and_artifact_validation(tmp_path):
    matrix_root, matrix_manifest, run_dir, manifest = _write_matching_full_cns(tmp_path)

    result = matching_full_cns_run(matrix_root, matrix_manifest)

    assert result is not None
    assert result[0] == run_dir
    row = full_cns_test_metric_row(manifest, f"Separate run: {run_dir.name}")
    assert row is not None
    assert row["architecture"] == "full_cns"
    assert row["brier_score"] == 0.23
    assert row["unknown_censored_units"] == 3
    assert row["source"] == "Separate run: full_cns_seed42"
    frame = _metrics_frame(pd, [row])
    assert frame.loc[0, "Brier score"] == 0.23
    assert frame.loc[0, "Unknown/censored units"] == 3

    (run_dir / "readout.pt").write_bytes(b"corrupted checkpoint")
    assert matching_full_cns_run(matrix_root, matrix_manifest) is None


def test_matching_full_cns_rejects_missing_or_altered_result_files(tmp_path):
    matrix_root, matrix_manifest, run_dir, manifest = _write_matching_full_cns(tmp_path)
    (run_dir / "predictions.parquet").write_bytes(b"altered predictions")
    assert matching_full_cns_run(matrix_root, matrix_manifest) is None
    (run_dir / "predictions.parquet").write_bytes(b"saved result")
    del manifest["artifact_hashes"]["baseline_metrics.json"]
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert matching_full_cns_run(matrix_root, matrix_manifest) is None


def test_matching_full_cns_rejects_target_hash_and_legacy_rul_mismatches(tmp_path):
    matrix_root, matrix_manifest, run_dir, _ = _write_matching_full_cns(tmp_path)
    wrong_target = json.loads(json.dumps(matrix_manifest))
    wrong_target["run_config"]["target_artifacts"]["bearings"]["targets_sha256"] = "different-hash"
    assert matching_full_cns_run(matrix_root, wrong_target) is None

    run_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run_manifest["schema_version"] = "old_rul_training"
    run_manifest["model_contract"]["uses_rul_target"] = True
    (run_dir / "manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")
    assert matching_full_cns_run(matrix_root, matrix_manifest) is None


def test_target_split_counts_require_matching_manifest_and_file_hash(tmp_path):
    matrix_root = tmp_path / "_future_red" / "matrix"
    target_dir = tmp_path / "_future_red" / "filters" / "targets" / "filter-target-v1"
    target_dir.mkdir(parents=True)
    target_path = target_dir / "targets.csv"
    target_path.write_text("target,data\n", encoding="utf-8")
    target_hash = _sha256(target_path)
    target = {"artifact_id": "filter-target-v1", "targets_sha256": target_hash}
    target_manifest = {
        "schema_version": "future_sensor_red_entry_targets_v1",
        "dataset_id": "filters",
        "artifact_id": "filter-target-v1",
        "targets_sha256": target_hash,
        "targets_file": "targets.csv",
        "split_counts": {"test": {"masked": 9931, "masked_censor": 9918, "masked_quality": 13, "masked_gap": 0}},
    }
    (target_dir / "manifest.json").write_text(json.dumps(target_manifest), encoding="utf-8")

    counts = load_target_split_counts(matrix_root, "filters", target)
    assert counts == target_manifest["split_counts"]["test"]
    target_path.write_text("changed\n", encoding="utf-8")
    assert load_target_split_counts(matrix_root, "filters", target) is None


def test_training_and_reports_show_verified_full_cns_and_filter_masking(tmp_path):
    from streamlit.testing.v1 import AppTest

    matrix_root, _, _, _ = _write_matching_full_cns(tmp_path)
    filter_dir = tmp_path / "_future_red" / "filters" / "targets" / "filter-target-v1"
    filter_dir.mkdir(parents=True)
    target_path = filter_dir / "targets.csv"
    target_path.write_text("target,data\n", encoding="utf-8")
    target_manifest = {
        "schema_version": "future_sensor_red_entry_targets_v1",
        "dataset_id": "filters",
        "artifact_id": "filter-target-v1",
        "targets_sha256": _sha256(target_path),
        "targets_file": "targets.csv",
        "split_counts": {"test": {"masked": 9931, "masked_censor": 9918, "masked_quality": 13, "masked_gap": 0}},
    }
    (filter_dir / "manifest.json").write_text(json.dumps(target_manifest), encoding="utf-8")
    matrix_manifest_path = matrix_root / "matrix_match" / "run_manifest.json"
    matrix_manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
    matrix_manifest["run_config"]["target_artifacts"]["filters"]["targets_sha256"] = target_manifest[
        "targets_sha256"
    ]
    matrix_manifest_path.write_text(json.dumps(matrix_manifest), encoding="utf-8")
    script = tmp_path / "future_red_app_test.py"
    script.write_text(
        "from pathlib import Path\n"
        "from pdm.future_red_ui import render_report, render_training\n"
        f"matrix = Path({str(matrix_root)!r})\n"
        "render_training('bearings', matrix)\n"
        "render_report('bearings', matrix)\n"
        "render_report('filters', matrix)\n",
        encoding="utf-8",
    )

    app = AppTest.from_file(str(script), default_timeout=15).run()

    assert not app.exception
    frames = [element.value for element in app.dataframe]
    model_rows = frames[0]
    assert "Full MaleCNS" in set(model_rows["Model"])
    assert model_rows.loc[model_rows["Model"] == "Full MaleCNS", "Source"].iloc[0] == "Separate run `full_cns_seed42`"
    training_metrics = frames[1]
    assert "Brier score" in training_metrics.columns
    assert "Unknown/censored units" in training_metrics.columns
    assert "Full MaleCNS" in set(training_metrics["Model"])
    assert "Separate run: full_cns_seed42" in set(training_metrics["Source"])
    captions = "\n".join(str(element.value) for element in app.caption)
    assert "9,931 total (9,918 censored before the 20-second horizon, 13 quality-masked, 0 gap-masked)" in captions
