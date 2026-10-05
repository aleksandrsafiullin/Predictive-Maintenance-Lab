"""Runner contracts use tiny real fixtures; no graph/model training here."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdm.io_util import atomic_write_json

spec = importlib.util.spec_from_file_location(
    "red_entry_matrix_runner",
    Path(__file__).resolve().parents[1] / "scripts/run_red_entry_v2_matrix.py",
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_freeze_rejects_changed_and_tampered_resume(tmp_path):
    contract = {"bounds": runner.BOUNDS, "smoke": False, "folds": [["a"], ["b"]]}
    frozen = runner.freeze_or_resume(tmp_path, contract, False)
    assert runner.freeze_or_resume(tmp_path, contract, True) == frozen
    with pytest.raises(ValueError, match="Changed or tampered"):
        runner.freeze_or_resume(tmp_path, {**contract, "smoke": True}, True)
    frozen["contract"]["folds"][0].append("c")
    atomic_write_json(tmp_path / "frozen_contract.json", frozen)
    with pytest.raises(ValueError, match="Changed or tampered"):
        runner.freeze_or_resume(tmp_path, contract, True)


@pytest.mark.parametrize("dependency", [
    "src/pdm/project_zones.py", "src/pdm/models/recurrent.py", "src/pdm/splits.py",
    "src/pdm/data/indirect_dependency.py", "pyproject.toml", "requirements-lock.txt",
])
def test_implementation_freeze_rejects_dependency_drift(tmp_path, dependency):
    root = tmp_path / "source"
    sources = {
        dependency, "src/pdm/project_zones.py", "src/pdm/models/recurrent.py", "src/pdm/splits.py",
        "scripts/run_red_entry_v2_matrix.py", "configs/red_entry_protocol.yaml",
        "pyproject.toml", "requirements-lock.txt",
    }
    for name in sources:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("original\n")
    original = runner.implementation_hashes(root)
    assert list(original) == sorted(sources)
    output = tmp_path / "freeze"
    runner.freeze_or_resume(output, {"code_hashes": original}, False)
    assert runner.freeze_or_resume(output, {"code_hashes": runner.implementation_hashes(root)}, True)
    (root / dependency).write_text("changed\n")
    with pytest.raises(ValueError, match="Changed or tampered"):
        runner.freeze_or_resume(output, {"code_hashes": runner.implementation_hashes(root)}, True)


def test_runtime_versions_are_stable_and_resume_rejects_drift(tmp_path):
    versions = runner.runtime_versions()
    assert versions == runner.runtime_versions()
    assert versions["python"] and versions["packages"]
    assert versions["packages"] == sorted(versions["packages"])
    runner.freeze_or_resume(tmp_path, {"runtime_versions": versions}, False)
    changed = {**versions, "packages": versions["packages"] + [["new-runtime-package", "1"]]}
    with pytest.raises(ValueError, match="Changed or tampered"):
        runner.freeze_or_resume(tmp_path, {"runtime_versions": changed}, True)


def test_completed_is_verified_skipped_and_failures_distinct(monkeypatch, tmp_path):
    class Store:
        def run_path(self, p, r):
            return tmp_path / p / r

    monkeypatch.setattr(runner, "project_store", Store)
    calls = []

    def train(p, s, e, cfg):
        calls.append(e)
        if e == "full_cns":
            raise FileNotFoundError("official graph unavailable")
        if e == "lstm":
            raise ValueError("bad fit")
        directory = tmp_path / p / "saved"
        directory.mkdir(parents=True)
        (directory / "weights.pt").write_bytes(b"real fixture bytes")
        return {
            "run_id": "saved",
            "artifacts": runner.file_hashes(directory),
            "selection": {"test_used": False},
            "support_by_horizon": [True],
        }

    tasks = [
        dict(
            project_id="p",
            snapshot_id="s",
            engine_id=e,
            params={"seed": 42, "input_mode": "hybrid"},
        )
        for e in ("gru", "full_cns", "lstm")
    ]
    manifest = {"tasks": {}}
    runner.execute_tasks(tasks, tmp_path, manifest, train)
    assert [r["status"] for r in manifest["tasks"].values()] == [
        "completed",
        "unavailable",
        "failed",
    ]
    runner.execute_tasks(tasks[:1], tmp_path, manifest, train)
    assert calls.count("gru") == 1
    (tmp_path / "p/saved/weights.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="integrity"):
        runner.execute_tasks(tasks[:1], tmp_path, manifest, train)


def test_nan_unknown_and_common_coverage():
    frame = pd.DataFrame(
        {
            "unit_id": ["a", "a", "b"],
            "physical_unit_id": ["a", "a", "b"],
            "timestamp_s": [0.0, 1.0, 0.0],
            "event_observed": [False, False, True],
            "followup_duration_s": [3.0, 0.5, 2.0],
        }
    )
    probabilities = np.array([[0.2, np.nan], [0.5, 0.9], [0.7, 0.8]])
    cells = runner.supported_cells(frame, probabilities, [1.0, 2.0], [True, True])
    assert len(cells[0]) == 2  # censored short followup excluded
    assert len(cells[1]) == 1  # NaN tail excluded, observed event retained
    complete = runner.supported_cells(
        frame, np.nan_to_num(probabilities, nan=0.4), [1.0, 2.0], [True, True]
    )
    common = runner.summarize_common({"baseline": cells[1], "model": complete[1]})
    assert (
        common["baseline"]["known_supported_origins"]
        == common["model"]["known_supported_origins"]
        == 1
    )
    assert common["baseline"]["brier"] == pytest.approx(0.04)
    assert len(complete[1]) == 2
    assert (
        runner.unit_equal_score({("a", 0): ("a", 0.0), ("a", 1): ("a", 0.0), ("b", 0): ("b", 1.0)})[
            "brier"
        ]
        == 0.5
    )


def test_fold_cannot_cross_physical_equipment_or_expand_development():
    data = {
        "units": pd.DataFrame(
            {"unit_id": ["a", "a2", "b", "c"], "physical_unit_id": ["p", "p", "q", "r"]}
        ),
        "split": {"train": ["a", "a2"], "validation": ["b"], "test": ["c"], "holdout": []},
    }
    with pytest.raises(ValueError, match="Physical equipment"):
        runner.fold_split(data, {"train": ["a"], "validation": ["a2", "b"]})
    with pytest.raises(ValueError, match="exactly"):
        runner.fold_split(data, {"train": ["a", "a2"], "validation": ["b", "c"]})
    split = runner.fold_split(data, {"train": ["b"], "validation": ["a", "a2"]})
    assert split["test"] == ["c"]


@pytest.mark.parametrize("train,validation", [
    (["a", "b"], ["c", "d", "e"]),
    (["c", "d", "e"], ["a", "b"]),
])
def test_staged_uneven_fold_recomputes_all_derived_counts(tmp_path, monkeypatch, train, validation):
    from copy import deepcopy

    from pdm.projects import ProjectStore

    store = ProjectStore(tmp_path / "projects")
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(store.root))
    project = store.create("Fold fixture", "generic_sensor_csv")["project_id"]
    monkeypatch.setattr(runner, "project_store", lambda: store)
    source = tmp_path / "source"
    source.mkdir()
    units = pd.DataFrame({
        "unit_id": list("abcdefg"), "physical_unit_id": list("abcdefg"),
        "event_observed": [True, False, True, True, False, True, False],
        "rejected_signal_rows": range(7), "gap_boundaries": [1] * 7,
    })
    features = pd.DataFrame({"unit_id": [uid for i, uid in enumerate("abcdefg") for _ in range(i + 1)]})
    units.to_parquet(source / "units.parquet", index=False)
    features.to_parquet(source / "features.parquet", index=False)
    atomic_write_json(source / "feature_schema.json", {})
    data = {
        "dir": source, "snapshot_id": "parent", "units": units, "features": features,
        "fingerprint": {},
        "split": {
            "train": list("abcd"), "validation": ["e"], "test": ["f"], "holdout": ["g"],
            "unassigned": ["stale"], "n_train": 4, "n_validation": 1, "n_test": 99,
            "n_holdout": 99, "n_unassigned": 99, "n_train_events": 99,
            "n_validation_events": 99, "n_test_events": 99,
            "realized_counts": {part: 99 for part in ("train", "validation", "test", "holdout", "unassigned")},
        },
        "report": {"split_counts": {"train": 99}, "by_split": {"train": {"rows": 99}}},
    }
    original = deepcopy(data["split"])
    assignment = {"train": train, "validation": validation}
    record = runner.stage_fold(project, data, assignment, 1)
    directory = store.snapshot_path(project, record["snapshot_id"])
    split = runner.read_json(directory / "split.json")
    report = runner.read_json(directory / "data_report.json")
    expected_members = {**assignment, "test": ["f"], "holdout": ["g"], "unassigned": []}
    expected_counts = {part: len(ids) for part, ids in expected_members.items()}
    for part, ids in expected_members.items():
        assert split[part] == ids
        assert split[f"n_{part}"] == expected_counts[part]
    assert split["realized_counts"] == expected_counts
    assert report["split_counts"] == expected_counts
    for part in ("train", "validation", "test"):
        selected = units[units.unit_id.isin(expected_members[part])]
        assert split[f"n_{part}_events"] == int(selected.event_observed.sum())
        assert report["by_split"][part] == {
            "units": len(selected),
            "rows": int(features.unit_id.isin(expected_members[part]).sum()),
            "rejected_signal_rows": int(selected.rejected_signal_rows.sum()),
            "gap_boundaries": int(selected.gap_boundaries.sum()),
        }
    assert data["split"] == original
    changed_units = units.copy()
    changed_units["event_observed"] = ~changed_units.event_observed
    changed = runner.fold_split({**data, "units": changed_units}, assignment)
    assert {part: changed[part] for part in expected_members} == expected_members
    assert store.get(project).get("active_snapshot_id") is None


def test_pairwise_common_retains_overlap_with_unavailable_third_method():
    methods = {"left": {("a", 0): ("a", 0.1), ("b", 0): ("b", 0.2)},
               "right": {("a", 0): ("a", 0.3)}, "unsupported": {}}
    assert runner.summarize_common(methods)["left"]["known_supported_origins"] == 0
    pairs = runner.summarize_pairwise(methods)
    assert len(pairs) == 3  # no outcome-driven omission of unsupported pairs
    available = next(pair for pair in pairs if pair["task_keys"] == ["left", "right"])
    assert available["common"]["left"]["known_supported_origins"] == 1
    assert available["common"]["left"]["brier"] == .1
    assert available["common"]["right"]["brier"] == .3


def test_last_value_is_causal_exact_recorded_and_unit_equal():
    features = pd.DataFrame({
        "unit_id": ["a"] * 4 + ["b"] * 2,
        "timestamp_s": [0., 1., 2., 3., 0., 1.],
        "signal": [10., 11., 12., 13., 20., 29.], "gap_before": False,
        # Future operation has no role in prediction or numerical scoring.
        "operating_age_s": [0., 999., 888., 777., 0., 999.],
    })
    origins = pd.DataFrame({"unit_id": ["a", "a", "a", "b"],
                            "timestamp_s": [0., 1., 2., 0.]})
    cells = runner.last_value_signal_cells({"features": features}, origins, [1., 1.5])
    assert runner.signal_score(cells[0], 4)["mae"] == 5.  # equal units, not 3:1 origins
    assert runner.signal_score(cells[1], 4)["coverage_fraction"] == 0.
    changed = features.copy()
    changed.loc[1, "signal"] = 100.
    changed["operating_age_s"] = -999.
    future = runner.last_value_signal_cells({"features": changed}, origins.iloc[:1], [1.])[0]
    assert next(iter(future.values()))[1] == 90.  # same origin value 10; future only target
    changed = features.copy()
    changed["operating_age_s"] = -999.
    assert runner.last_value_signal_cells({"features": changed}, origins, [1., 1.5]) == cells


@pytest.mark.parametrize("break_column,break_value", [
    ("gap_before", True), ("quality_ok", False), ("usable", False),
    ("quality_status", "bad"), ("component_replaced", True),
    ("cycle_id", "second"), ("segment_id", "second"),
])
def test_last_value_excludes_gaps_quality_incomplete_and_unknown(break_column, break_value):
    features = pd.DataFrame({"unit_id": ["a"] * 3, "timestamp_s": [0., 1., 2.],
                             "signal": [1., 2., 3.], "gap_before": False})
    default = "first" if break_column in {"cycle_id", "segment_id"} else (
        "ok" if break_column == "quality_status" else break_column in {"quality_ok", "usable"})
    features[break_column] = default
    features.loc[1, break_column] = break_value
    origin = pd.DataFrame({"unit_id": ["a"], "timestamp_s": [0.]})
    assert runner.last_value_signal_cells({"features": features}, origin, [2., 3.]) == [{}, {}]
    origin["risk_status"] = "unknown_event_history"
    features.loc[1, break_column] = default
    assert runner.last_value_signal_cells({"features": features}, origin, [2.]) == [{}]


def test_diagnostic_third_train_duration_is_independent_of_action_requirement():
    data = {"features": pd.DataFrame({
        "unit_id": ["train"] * 4 + ["test"] * 2,
        "timestamp_s": [0., 30., 100., 130., 0., 9000.],
        "gap_before": [False, False, True, False, False, False],
    }), "split": {"train": ["train"], "test": ["test"]}}
    origins = pd.DataFrame({
        "unit_id": ["a", "a", "b", "c"], "episode_id": ["a", "a", "b", "c"],
        "timestamp_s": [0., 10., 0., 0.], "event_time_s": [30., 30., 10., 100.],
        "followup_duration_s": [30., 20., 10., 5.], "event_observed": True,
        "split": "test", "context_known": False,
    })
    protocol = {"operational_requirements": {"minimum_action_lead_s": None}}
    report = runner.diagnostic_reference(data, origins, origins.iloc[1:], protocol)
    assert report["average_training_duration_s"] == 60.
    assert report["target_lead_s"] == 20.
    assert report["minimum_action_lead_s"] is None
    assert report["by_part"]["test"]["all_data"]["reachable_events"] == 1
    assert report["by_part"]["test"]["all_data"]["reachable_pre_event_origins"] == 2
    assert report["by_part"]["test"]["all_data"]["known_reliable_events"] == 2
    protocol["operational_requirements"]["minimum_action_lead_s"] = 999.
    changed = runner.diagnostic_reference(data, origins, origins.iloc[1:], protocol)
    assert changed["target_lead_s"] == report["target_lead_s"]
    assert changed["by_part"] == report["by_part"]


def test_diagnostic_preserves_event_denominator_without_eligible_or_sampled_origins():
    data = {"features": pd.DataFrame({"unit_id": ["train", "train"],
                                       "timestamp_s": [0., 60.], "gap_before": False}),
            "split": {"train": ["train"]}}
    origins = pd.DataFrame({
        "unit_id": ["a"], "episode_id": ["a:0"], "split": ["test"],
        "timestamp_s": [30.], "event_time_s": [30.], "event_observed": [True],
        "followup_duration_s": [0.], "at_risk": [False], "risk_status": ["event_observed"],
    })
    events = pd.DataFrame({
        "unit_id": ["a", "b", "c"], "episode_id": ["a:0", "b:0", "c:0"],
        "split": ["test"] * 3, "first_red_timestamp_s": [30., 0., np.nan],
        "first_event_verified": [True, False, False],
    })
    protocol = {"operational_requirements": {"minimum_action_lead_s": 999.}}
    report = runner.diagnostic_reference(data, origins, origins.iloc[:0], protocol, events)
    assert report["target_lead_s"] == 20.
    assert report["minimum_action_lead_s"] == 999.
    for result in report["by_part"]["test"].values():
        assert result["known_reliable_events"] == 1
        assert result["reachable_events"] == 0
        assert result["unreachable_or_no_eligible_origin_events"] == 1
        assert result["no_eligible_pre_event_origin_events"] == 1
        assert result["known_reliable_pre_event_origins"] == 0
        assert result["unknown_history_first_recorded_red_events"] == 1


def test_diagnostic_sampling_keeps_full_known_event_denominator():
    data = {"features": pd.DataFrame({"unit_id": ["train", "train"],
                                       "timestamp_s": [0., 60.], "gap_before": False}),
            "split": {"train": ["train"]}}
    origins = pd.DataFrame({
        "unit_id": ["a", "b"], "episode_id": ["a:0", "b:0"], "split": ["test"] * 2,
        "timestamp_s": [0., 25.], "event_time_s": [30., 30.], "event_observed": [True] * 2,
        "followup_duration_s": [30., 5.],
    })
    events = pd.DataFrame({
        "unit_id": ["a", "b"], "episode_id": ["a:0", "b:0"], "split": ["test"] * 2,
        "first_red_timestamp_s": [30., 30.], "first_event_verified": [True] * 2,
    })
    protocol = {"operational_requirements": {"minimum_action_lead_s": None}}
    results = runner.diagnostic_reference(data, origins, origins.iloc[1:], protocol, events)["by_part"]["test"]
    assert results["all_data"]["known_reliable_events"] == results["sampled"]["known_reliable_events"] == 2
    assert results["all_data"]["reachable_events"] == 1
    assert results["all_data"]["eligible_but_short_lead_events"] == 1
    assert results["sampled"]["reachable_events"] == 0
    assert results["sampled"]["no_eligible_pre_event_origin_events"] == 1
    assert results["sampled"]["eligible_but_short_lead_events"] == 1


def test_tiny_real_baseline_matrix_and_resume(tmp_path, monkeypatch):
    """Exercise publish→train→hash→compare→resume with real local artifacts."""

    from pdm.data.project_prepare import _stage_snapshot, load_snapshot
    from pdm.projects import ProjectStore
    from pdm.red_entry_protocol import canonical_json_hash, load_protocol

    isolated = tmp_path / "research-projects"
    monkeypatch.setenv("PDM_PROJECTS_ROOT", str(isolated))
    store = ProjectStore(isolated)
    p = store.create("Fixture RED matrix", "generic_sensor_csv")["project_id"]
    frame = pd.DataFrame(
        [
            {
                "unit_id": u,
                "physical_unit_id": u,
                "timestamp_s": float(t),
                "signal": float(t + 1),
                "gap_before": False,
            }
            for u in ("a", "b", "c")
            for t in range(0, 121, 5)
        ]
    )
    schema = {
        "source_kind": "hse_filters",
        "signal_column": "signal",
        "signal_unit": "u",
        "thresholds": {"mode": "absolute", "red": 70.0, "direction": "above"},
        "columns": {
            "signal": {"role": "sensor"},
            "unit_id": {"role": "metadata"},
            "timestamp_s": {"role": "metadata"},
        },
    }

    def write(path):
        frame.to_parquet(path / "features.parquet", index=False)
        pd.DataFrame({"unit_id": ["a", "b", "c"], "physical_unit_id": ["a", "b", "c"]}).to_parquet(
            path / "units.parquet", index=False
        )
        atomic_write_json(path / "feature_schema.json", schema)

    staged, sid, _ = _stage_snapshot(
        store.project_path(p) / "snapshots",
        p,
        write_data=write,
        split={"train": ["a"], "validation": ["b"], "test": ["c"], "holdout": []},
        report={},
        fingerprint={},
    )
    staged.rename(store.snapshot_path(p, sid))
    store.update(p, active_snapshot_id=sid)
    data = load_snapshot(p, sid)
    frozen_source = tmp_path / "research_data_freeze.json"
    atomic_write_json(
        frozen_source,
        {
            "protocol": load_protocol(),
            "projects": [
                {
                    "project_id": p,
                    "snapshot_id": sid,
                    "fingerprint_hash": canonical_json_hash(data["fingerprint"]),
                    "rule": None,
                    "grouped_development_folds": [
                        {"train": ["a"], "validation": ["b"]},
                        {"train": ["b"], "validation": ["a"]},
                    ],
                }
            ],
        },
    )
    output = tmp_path / "smoke"
    argv = [
        "--projects",
        p,
        "--engines",
        "always_no_entry",
        "--input-modes",
        "sensor_only",
        "--seeds",
        "42",
        "--development-folds",
        "2",
        "--research-freeze",
        str(frozen_source),
        "--output-dir",
        str(output),
        "--smoke",
    ]
    assert runner.main(argv) == 0
    manifest = runner.read_json(output / "matrix_manifest.json")
    assert len(manifest["tasks"]) == 3
    assert all(t["status"] == "completed" for t in manifest["tasks"].values())
    assert store.get(p)["active_snapshot_id"] == sid
    assert len(manifest["snapshots"]) == 3
    before = {k: t["run_id"] for k, t in manifest["tasks"].items()}
    assert runner.main([*argv, "--resume"]) == 0
    after = runner.read_json(output / "matrix_manifest.json")
    assert before == {k: t["run_id"] for k, t in after["tasks"].items()}
    comparison = runner.read_json(output / "comparison.json")
    assert len(comparison["rows"]) == 24
    assert comparison["quality_gate"]["can_pass"] is False
    assert comparison["rows"][0]["common"]["known_supported_origins"] > 0
    assert comparison["signal_mae"]["status"] == "evaluated"
    assert comparison["rows"][0]["last_value_signal"]["mae"] == 5.
    assert comparison["diagnostic_reference"][0]["target_lead_s"] == 40.
    (output / "comparison.csv").write_text("tampered")
    with pytest.raises(ValueError, match="integrity"):
        runner.main([*argv, "--resume"])
