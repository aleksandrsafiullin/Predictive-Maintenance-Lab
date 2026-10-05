"""Source-only routing checks: no pdm imports, real data, models or workers."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/run_signal_funnel_matrix.py"
BASELINE = ROOT / "output/bearings-learned-funnel-20261002/review/run_signal_funnel_matrix_before_profile_routing.py.txt"
ENGINES = ("gru", "lstm", "quantile_boosting", "full_cns")


def extracted(*names, **injected):
    tree = ast.parse(SOURCE.read_text())
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    namespace = {"Path": Path, "hashlib": hashlib, "json": json, "argparse": argparse, **injected}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return SimpleNamespace(**namespace)


def invented_profile(learned=True):
    return {"forecast_mode": "learned_joint_trajectories" if learned else "joint_residual_paths",
            "horizons_s": [2, 4, 6] if learned else [2, 6], "path_samples": 256,
            "training_samples": 32 if learned else None,
            "max_windows_per_unit": 256 if learned else 128,
            "patience": 25, "min_delta": 0.00001, "batch_sampling": "row_permutation",
            "cv_folds": 3, "future_accepted_advanced": {"arbitrary": [True, 7, "kept"]}}


@pytest.mark.parametrize("engine", ENGINES)
def test_learned_route_preserves_entire_profile(engine):
    profile = invented_profile()
    calls = []
    def learned(snapshot, selected):
        calls.append((snapshot, selected))
        return profile
    def forbidden(*_args):
        raise AssertionError("fallback must not be called")
    code = extracted("training_profile", learned_bearings_training_profile=learned,
                     funnel_training_profile=forbidden)
    snapshot = {"invented": "bearing"}
    assert code.training_profile(snapshot, engine) is profile
    assert calls == [(snapshot, engine)]
    assert profile["future_accepted_advanced"] == {"arbitrary": [True, 7, "kept"]}


def test_unadmitted_snapshot_falls_back_like_ui():
    profile = invented_profile(False)
    calls = []
    code = extracted("training_profile", learned_bearings_training_profile=lambda *_: None,
                     funnel_training_profile=lambda snapshot, engine: calls.append((snapshot, engine)) or profile)
    snapshot = {"invented": "unsupported signal label"}
    assert code.training_profile(snapshot, "gru") is profile
    assert calls == [(snapshot, "gru")]


@pytest.fixture
def frozen_source():
    projects = {"Bearings": {"project_id": "p1", "snapshot_id": "s1"},
                "Filters": {"project_id": "p2", "snapshot_id": "s2"}}
    loads = []
    def load(project, snapshot):
        loads.append((project, snapshot))
        return {"fingerprint": {"invented_project": project}, "learned": project == "p1"}
    def read(path):
        if path.name == "rebuild_manifest.json":
            return {"projects": projects}
        return json.loads(path.read_text())
    def write(path, value):
        path.write_text(json.dumps(value, sort_keys=True))
    code = extracted("digest", "selected_datasets", "training_profile", "profile_policy", "freeze",
                     ROOT=ROOT, ENGINES=ENGINES, read_json=read, atomic_write_json=write,
                     load_snapshot=load, project_store=lambda: SimpleNamespace(
                         get=lambda project: {"active_snapshot_id": "s1" if project == "p1" else "s2"}),
                     source_hashes=lambda: {"invented": "hash"},
                     learned_bearings_training_profile=lambda snapshot, _: invented_profile() if snapshot["learned"] else None,
                     funnel_training_profile=lambda *_: invented_profile(False))
    return code, loads


@pytest.mark.parametrize("selector,expected", [(None, ["Bearings", "Filters"]),
                                               (["Bearings"], ["Bearings"]),
                                               (["Filters", "Bearings"], ["Bearings", "Filters"])])
def test_all_or_selected_snapshots_only_and_manifest_engine_order(frozen_source, tmp_path, selector, expected):
    code, loads = frozen_source
    frozen = code.freeze(tmp_path / "new", False, selector)
    assert loads == [("p1" if key == "Bearings" else "p2", "s1" if key == "Bearings" else "s2") for key in expected]
    assert [(job["dataset"], job["engine_id"]) for job in frozen["contract"]["jobs"]] == [
        (key, engine) for key in expected for engine in ENGINES]
    assert frozen["contract"]["dataset_selector"] == (None if selector is None else expected)


@pytest.mark.parametrize("selector", [[], [""], ["  "], ["Bearings", "Bearings"], ["bearings"], ["Bearings", "Unknown"]])
def test_bad_selectors_fail_before_snapshot_reads(frozen_source, tmp_path, selector):
    code, loads = frozen_source
    with pytest.raises(ValueError):
        code.freeze(tmp_path / "new", False, selector)
    assert loads == []
    assert not (tmp_path / "new").exists()


def test_mixed_contract_truthfully_records_resolved_policies(frozen_source, tmp_path):
    code, _ = frozen_source
    contract = code.freeze(tmp_path / "new", False)["contract"]
    assert "no Validation or Test tuning" not in json.dumps(contract)
    for job in contract["jobs"]:
        policy, params = job["profile_policy"], job["params"]
        for key in ("forecast_mode", "horizons_s", "path_samples", "training_samples", "max_windows_per_unit"):
            assert policy[key] == params[key]
        if job["dataset"] == "Bearings":
            assert "dense" in policy["horizon_policy"]
            assert "best Validation" in policy["checkpoint_selection_policy"]
            assert "early stopping" in policy["checkpoint_selection_policy"]
            assert "uncalibrated" in policy["uncertainty_policy"]
            assert policy["patience"] == 25
        else:
            assert "sparse" in policy["horizon_policy"]
            assert "fixed final" in policy["checkpoint_selection_policy"]
            assert "OOF" in policy["uncertainty_policy"]
            assert "Validation-only" in policy["uncertainty_policy"]


def test_exact_resume_and_refusals_never_overwrite(frozen_source, tmp_path):
    code, _ = frozen_source
    directory = tmp_path / "new"
    frozen = code.freeze(directory, False, ["Bearings"])
    path = directory / "frozen_contract.json"
    before = path.read_bytes()
    assert code.freeze(directory, True, ["Bearings"]) == frozen
    for resume, selector in [(False, ["Bearings"]), (True, None), (True, ["Filters"])]:
        with pytest.raises(ValueError, match="already frozen or has changed"):
            code.freeze(directory, resume, selector)
        assert path.read_bytes() == before
    old = {"contract": {"jobs": []}, "contract_hash": "old-residual-contract"}
    path.write_text(json.dumps(old))
    old_bytes = path.read_bytes()
    with pytest.raises(ValueError, match="already frozen or has changed"):
        code.freeze(directory, True, ["Bearings"])
    assert path.read_bytes() == old_bytes
    with pytest.raises(ValueError, match="No frozen study"):
        code.freeze(tmp_path / "missing", True)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "existing.txt").write_text("preserve")
    with pytest.raises(ValueError, match="must be empty"):
        code.freeze(occupied, False)
    assert not (occupied / "frozen_contract.json").exists()
    assert (occupied / "existing.txt").read_text() == "preserve"


@pytest.mark.parametrize("prepare", [False, True])
@pytest.mark.parametrize("selector", [None, ["Filters", "Bearings"]])
def test_cli_forwards_repeatable_optional_selector(monkeypatch, tmp_path, prepare, selector):
    calls = []
    def freeze(*args):
        calls.append(args)
        return {"contract": {"jobs": []}, "contract_hash": "invented"}
    def run(*args):
        calls.append(args)
        return 0
    code = extracted("main", freeze=freeze, run=run)
    arguments = ["matrix", "--output", str(tmp_path)]
    for dataset in selector or []:
        arguments.extend(["--dataset", dataset])
    if prepare:
        arguments.append("--prepare-only")
    monkeypatch.setattr("sys.argv", arguments)
    assert code.main() == 0
    assert calls == ([(tmp_path, False, selector)] if prepare else
                     [(tmp_path, False, 21600, selector)])


@pytest.mark.parametrize("binding", ["source_hashes", "load_snapshot", "project_store"])
def test_resume_rejects_changed_source_snapshot_or_active_binding(frozen_source, tmp_path, binding):
    code, _ = frozen_source
    directory = tmp_path / "new"
    code.freeze(directory, False, ["Bearings"])
    path = directory / "frozen_contract.json"
    before = path.read_bytes()
    replacements = {
        "source_hashes": lambda: {"invented": "changed"},
        "load_snapshot": lambda *_: {"fingerprint": {"invented_project": "changed"}, "learned": True},
        "project_store": lambda: SimpleNamespace(get=lambda _: {"active_snapshot_id": "changed"}),
    }
    code.freeze.__globals__[binding] = replacements[binding]
    with pytest.raises(ValueError, match="changed"):
        code.freeze(directory, True, ["Bearings"])
    assert path.read_bytes() == before


def test_run_passes_selector_without_launching_anything(tmp_path):
    calls = []
    class StopBeforeJobs(Exception):
        pass
    def freeze(*args):
        calls.append(args)
        raise StopBeforeJobs
    code = extracted("run", freeze=freeze)
    with pytest.raises(StopBeforeJobs):
        code.run(tmp_path, True, 5, ["Bearings"])
    assert calls == [(tmp_path, True, ["Bearings"])]


def test_worker_verification_timeout_and_hash_guards_unchanged():
    def functions(path):
        return {node.name: node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)}
    old, new = functions(BASELINE), functions(SOURCE)
    engine_tree = ast.parse((ROOT / "src/pdm/signal_training.py").read_text())
    engine_value = next(node.value for node in engine_tree.body if isinstance(node, ast.Assign)
                        and any(isinstance(target, ast.Name) and target.id == "ENGINES" for target in node.targets))
    assert ast.literal_eval(engine_value) == ENGINES
    for name in ("digest", "source_hashes", "verify_run"):
        assert ast.dump(old[name]) == ast.dump(new[name])
    assert ast.dump(ast.Module(body=old["run"].body[1:], type_ignores=[])) == ast.dump(
        ast.Module(body=new["run"].body[1:], type_ignores=[]))
    # The exact frozen comparison and fresh-directory refusal tail also remain intact.
    old_tail = old["freeze"].body[-8:]
    new_tail = new["freeze"].body[-8:]
    assert ast.dump(ast.Module(body=old_tail, type_ignores=[])) == ast.dump(
        ast.Module(body=new_tail, type_ignores=[]))
