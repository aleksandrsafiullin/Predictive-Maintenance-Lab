"""One-factor causal population alignment; legacy fitting stays opt-out."""
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from pdm.probabilistic import workflow as flow
from pdm.probabilistic.contract import (
    canonical_hash,
    config_hash,
    default_config,
    trend_config,
    validate_config,
)
from pdm.probabilistic.corridor import center_objective
from pdm.probabilistic.models import TrainingCancelled, fit_model, load_model, predict, save_model
from pdm.probabilistic.windows import build_windows

PROTOCOL = "balanced_rolling_reference_v1"


def cfg_for(engine, H):
    return trend_config(default_config(engine_id=engine, hidden_size=8, torch_threads=1,
                                      max_epochs=3, patience=3, batch_size=16,
                                      max_origins_per_unit=3, boosting_max_iter=2,
                                      min_samples_leaf=2), H)


def source(specs, split, cfg):
    frame = pd.concat([pd.DataFrame({"unit_id": uid, "physical_unit_id": uid,
                                     "timestamp_s": np.arange(len(values))*cfg["cadence_s"],
                                     cfg["target"]:values}) for uid,values in specs],ignore_index=True)
    frame.attrs.update(split=split,dataset_hash="bounded-fixture-source",snapshot_id="bounded-fixture",
                       release_id="bounded-fixture-release",suite="bounded-fixture",profile="synthetic_single_channel")
    return frame


def fixtures(cfg,specs=None):
    specs = specs or [("a",np.full(100,.2)),("b",np.full(100,.4))]
    train = build_windows(source([("train-"+uid,values) for uid,values in specs],"train",cfg),cfg)
    val_frame = source([("val-"+uid,values) for uid,values in specs],"validation",cfg)
    return train,build_windows(val_frame,cfg,mode="validation"),build_windows(val_frame,cfg,mode="reference")


def balanced_cfg(engine="persistence", H=20):
    return trend_config({**cfg_for(engine, H), "training_population_protocol": PROTOCOL})


def populations(cfg, specs=None):
    train, val, ref_val = fixtures(cfg, specs)
    specs = specs or [("a", np.full(100, .2)), ("b", np.full(100, .4))]
    frame = source([("train-" + uid, values) for uid, values in specs], "train", cfg)
    return train, val, ref_val, build_windows(frame, cfg, mode="reference")


def fitted(cfg, specs=None, **kwargs):
    train, val, ref_val, ref_train = populations(cfg, specs)
    model = fit_model(train, val, cfg, reference_validation_windows=ref_val,
                      reference_train_windows=ref_train, **kwargs)
    return model, (train, val, ref_val, ref_train)


def test_fresh_factory_default_absent_and_explicit_balanced_preserved():
    cfg = trend_config(default_config(), 900)
    assert "training_population_protocol" not in cfg
    assert default_config(**cfg) == cfg
    explicit = {**cfg, "training_population_protocol": PROTOCOL}
    assert trend_config(explicit) == explicit
    assert config_hash(default_config()) == "8ad4029bcd6946e72a296fc3668c766b466ab19005abaeda66f372ce108e2bdc"


@pytest.mark.parametrize("protocol", [None, "balanced-v2", True])
def test_unknown_or_unversioned_population_protocol_rejected(protocol):
    cfg = balanced_cfg()
    cfg["training_population_protocol"] = protocol
    with pytest.raises(ValueError, match="training_population_protocol"):
        validate_config(cfg)


def test_nonbounded_mode_cannot_claim_new_population_protocol():
    cfg = default_config()
    cfg["training_population_protocol"] = PROTOCOL
    with pytest.raises(ValueError, match="training_population_protocol"):
        validate_config(cfg)


def test_new_protocol_requires_separate_causal_train_reference_population():
    cfg = balanced_cfg()
    train, val, ref_val, _ = populations(cfg)
    with pytest.raises(ValueError, match="earliest-reference Train"):
        fit_model(train, val, cfg, reference_validation_windows=ref_val)


@pytest.mark.parametrize("attack", ["split", "mode", "source", "policy", "manifest", "physical_duplicate"])
def test_wrong_train_reference_role_provenance_or_identity_rejected(attack):
    cfg = balanced_cfg()
    train, val, ref_val, ref_train = populations(cfg)
    if attack == "split":
        ref_train["split"] = "validation"
    elif attack == "mode":
        ref_train["mode"] = "train"
    elif attack == "source":
        ref_train["dataset_hash"] = "other-source"
    elif attack == "policy":
        manifest = ref_train["origin_manifest"]
        manifest["origin_policy"] = "future-selected-policy"
        manifest["origin_hash"] = canonical_hash({k:v for k,v in manifest.items() if k != "origin_hash"})
    elif attack == "manifest":
        ref_train["origin_manifest"]["origin_hash"] = "mutated"
    else:
        ref_train["physical_units"][1] = ref_train["physical_units"][0]
    with pytest.raises(ValueError, match="Train|train"):
        fit_model(train, val, cfg, reference_validation_windows=ref_val, reference_train_windows=ref_train)


@pytest.mark.parametrize("engine", ["persistence", "local_trend", "gru", "quantile_boosting"])
def test_all_four_engines_audit_separate_risks_and_roundtrip(engine, tmp_path):
    cfg = balanced_cfg(engine)
    model, (train, _, _, ref_train) = fitted(cfg, [("short", .2 + .002*np.arange(65)), ("long", .4 + .001*np.arange(100))])
    summary = model["training_summary"]
    assert summary["training_components"] == {"rolling_train": .5, "earliest_reference_train": .5}
    assert summary["optimization_weight_sum"] == pytest.approx(1.)
    rolling = center_objective(train["y"], predict(model, train["x"])[..., 1], train["mask"], train["weights"], cfg)
    reference = center_objective(ref_train["y"], predict(model, ref_train["x"])[..., 1], ref_train["mask"], ref_train["weights"], cfg)
    assert summary["best_rolling_train_loss"] == pytest.approx(rolling)
    assert summary["best_reference_train_loss"] == pytest.approx(reference)
    assert summary["best_training_loss"] == pytest.approx(.5*rolling + .5*reference)
    for epoch in summary["epochs"]:
        assert epoch["train_loss"] == pytest.approx(.5*epoch["rolling_train_loss"] + .5*epoch["reference_train_loss"])
        assert epoch["train_loss_scope"] == "end_of_epoch_fixed_model_full_population_risk"
        assert epoch["validation_loss"] == pytest.approx(.5*epoch["rolling_validation_loss"] + .5*epoch["reference_validation_loss"])
    outputs = predict(model, ref_train["x"])
    save_model(model, tmp_path/"model")
    assert load_model(tmp_path/"model")["model_hash"] == model["model_hash"]
    np.testing.assert_array_equal(predict(load_model(tmp_path/"model"), ref_train["x"]), outputs)


def test_preprocessing_remains_original_rolling_train_and_baselines_unchanged():
    specs = [("low", .2 + .002*np.arange(100)), ("high", .5 + .001*np.arange(100))]
    for engine in ("persistence", "local_trend"):
        cfg = balanced_cfg(engine)
        balanced, (_, _, _, ref_train) = fitted(cfg, specs)
        legacy_cfg = {k:v for k,v in cfg.items() if k != "training_population_protocol"}
        train, val, ref_val = fixtures(legacy_cfg, specs)
        legacy = fit_model(train, val, legacy_cfg, reference_validation_windows=ref_val)
        assert balanced["preprocessing"] == legacy["preprocessing"]
        assert balanced["training_summary"]["preprocessing_population"] == "original_rolling_train_only"
        np.testing.assert_array_equal(predict(balanced, ref_train["x"]), predict(legacy, ref_train["x"]))


def test_boosting_sample_weights_keep_separate_half_population_normalization(monkeypatch):
    from sklearn.ensemble import HistGradientBoostingRegressor
    cfg = balanced_cfg("quantile_boosting")
    train, val, ref_val, ref_train = populations(cfg, [("short", np.full(65,.2)), ("long", np.full(100,.4))])
    mask = np.concatenate((train["mask"], ref_train["mask"]))
    weights = np.concatenate((.5*train["weights"], .5*ref_train["weights"]))
    rows, leads = np.where(mask)
    seen = []
    original = HistGradientBoostingRegressor.fit
    def capture(self, x, y, sample_weight=None):
        seen.append(np.asarray(sample_weight).copy())
        return original(self, x, y, sample_weight=sample_weight)
    monkeypatch.setattr(HistGradientBoostingRegressor,"fit",capture)
    fit_model(train,val,cfg,reference_validation_windows=ref_val,reference_train_windows=ref_train)
    assert seen
    np.testing.assert_array_equal(seen[0], weights[rows,leads]*len(rows))


def test_gru_stop_resume_exact_and_changed_train_reference_rejected(tmp_path):
    cfg = balanced_cfg("gru")
    train,val,ref_val,ref_train = populations(cfg, [("rise", .2*np.exp(.003*np.arange(100)))])
    stopped = {"value":False}
    def callback(record):
        if record["epoch"] == 1:
            stopped["value"] = True
    options = {"reference_validation_windows":ref_val,"reference_train_windows":ref_train}
    with pytest.raises(TrainingCancelled):
        fit_model(train,val,cfg,**options,status_cb=callback,should_stop=lambda:stopped["value"],checkpoint_directory=tmp_path)
    resumed = fit_model(train,val,cfg,**options,checkpoint_directory=tmp_path)
    complete = fit_model(train,val,cfg,**options)
    assert resumed["model_hash"] == complete["model_hash"]
    changed = deepcopy(ref_train)
    changed["y"][changed["mask"]] *= 1.02
    with pytest.raises(ValueError,match="Resume checkpoint|Reference Train.*content"):
        fit_model(train,val,cfg,reference_validation_windows=ref_val,reference_train_windows=changed,checkpoint_directory=tmp_path)


def test_reference_anchor_remains_causal_under_changed_future_values_and_masks():
    cfg = balanced_cfg()
    frame = source([("train-only", .2 + .002*np.arange(100))],"train",cfg)
    original = build_windows(frame,cfg,mode="reference")
    frame.loc[frame.timestamp_s >= 60*60,cfg["target"]] = np.nan
    changed = build_windows(frame,cfg,mode="reference")
    assert changed["origin_manifest"] == original["origin_manifest"]
    np.testing.assert_array_equal(changed["x"],original["x"])
    assert not changed["mask"].any()


def test_one_fixed_analytic_learner_still_rises_declines_and_holds_healthy():
    cfg = balanced_cfg("gru")
    cfg.update(max_epochs=70,patience=20,learning_rate=.012,batch_size=32)
    specs = [("rise-a",.2*np.exp(.008*np.arange(100))), ("rise-b",.35*np.exp(.008*np.arange(100))),
             ("fall-a",.4*np.exp(-.008*np.arange(100))), ("fall-b",.65*np.exp(-.008*np.arange(100))),
             ("flat-a",np.full(100,.3)), ("flat-b",np.full(100,.5))]
    model,(_,_,ref_val,_) = fitted(cfg,specs)
    center = predict(model,ref_val["x"])[...,1]
    for uid,x,c in zip(ref_val["units"],ref_val["x"],center):
        if "rise" in uid:
            assert c[-1] > x[-1]*1.08
        elif "fall" in uid:
            assert c[-1] < x[-1]*.92
        else:
            assert abs(c[-1]/x[-1]-1) < .08


def test_workflow_opens_only_train_validation_and_publishes_reference_provenance(tmp_path,monkeypatch):
    cfg = balanced_cfg()
    bound = {"snapshot_id":"balanced-fixture", "dataset_hash":"bounded-fixture-source",
             "release_id":"bounded-fixture-release","suite":"bounded-fixture","profile":"synthetic_single_channel"}
    snapshot = {**bound,"config":default_config(),"evaluation_status":"not_exposed"}
    frames = {split:source([(split+"-unit",.2+.001*np.arange(100))],split,cfg) for split in ("train","validation")}
    for frame in frames.values():
        frame.attrs.update(bound)
    opened=[]
    def load_split(snap,split):
        opened.append(split)
        return frames[split]
    monkeypatch.setattr(flow,"snapshot_for_project",lambda *args:snapshot)
    monkeypatch.setattr("pdm.probabilistic.data.load_split",load_split)
    monkeypatch.setattr("pdm.project_snapshot.load_project_limits",lambda *args:None)
    monkeypatch.setattr(flow,"_path",lambda project,run:tmp_path/run)
    class Store:
        def update(self,*args,**kwargs):
            pass
    monkeypatch.setattr(flow,"project_store",lambda:Store())
    run = flow.train_run("fixture","balanced-fixture","persistence",cfg)
    assert opened == ["train","validation"]
    path = __import__("pathlib").Path(run["directory"])/"train_reference_origins.json"
    manifest = json.loads(path.read_text())
    assert manifest["origin_hash"] == run["training_summary"]["reference_train_origin_hash"]
    assert "train_reference_origins.json" in run["files"]


class AdmissionBoundary(RuntimeError):
    """Stop before a predictor or learner runs."""


def admission_populations(cfg, *, cycles=False):
    frame = source([("train-a", .2+.001*np.arange(110)),
                    ("train-b", .4+.001*np.arange(110)),
                    ("train-empty", np.full(60,.3))],"train",cfg)
    if cycles:
        frame.loc[frame.unit_id.isin(("train-a","train-empty")),"physical_unit_id"] = "equipment-cycle-3"
    validation = source([("validation-a", .25+.001*np.arange(110))],"validation",cfg)
    return (build_windows(frame,cfg), build_windows(validation,cfg,mode="validation"),
            build_windows(validation,cfg,mode="reference"), build_windows(frame,cfg,mode="reference"))


def stop_at_admission(monkeypatch, cfg, populations, reference=None):
    from pdm.probabilistic import models
    def stop(*args,**kwargs):
        raise AdmissionBoundary("admitted before predictor")
    monkeypatch.setattr(models,"baseline_predictions",stop)
    train,val,ref_val,ref_train = populations
    return fit_model(train,val,cfg,reference_validation_windows=ref_val,
                     reference_train_windows=ref_train if reference is None else reference)


def rehash_reference(reference):
    manifest = reference["origin_manifest"]
    manifest["origin_hash"] = canonical_hash({k:v for k,v in manifest.items() if k != "origin_hash"})


@pytest.mark.parametrize("attack", ["missing_supported", "missing_empty", "later_anchor", "history", "target", "mask"])
def test_reference_train_self_rehash_cannot_replace_canonical_admission(monkeypatch,attack):
    cfg = balanced_cfg()
    population = admission_populations(cfg)
    train,_,_,original = population
    reference = deepcopy(original)
    row = int(np.flatnonzero(reference["units"] == "train-a")[0])
    if attack.startswith("missing"):
        uid = "train-b" if attack == "missing_supported" else "train-empty"
        keep = reference["units"] != uid
        for key in ("x","y","mask","units","physical_units","origin_s","weights"):
            reference[key] = reference[key][keep]
        reference["origin_manifest"]["records"] = [r for r in reference["origin_manifest"]["records"] if r["unit_id"] != uid]
        reference["origin_manifest"]["reference_selection"]["selected_physical_units"] -= 1
    elif attack == "later_anchor":
        later = int(np.flatnonzero((train["units"] == "train-a") & (train["origin_s"] > reference["origin_s"][row]))[0])
        for key in ("x","y","mask","units","physical_units","origin_s"):
            reference[key][row] = train[key][later]
        reference["origin_manifest"]["records"][row] = deepcopy(train["origin_manifest"]["records"][later])
    elif attack == "history":
        reference["x"][row,0] *= 1.01
    elif attack == "target":
        reference["y"][row,0] *= 1.01
    else:
        reference["mask"][row,-1] = False
    rehash_reference(reference)
    with pytest.raises(ValueError,match="Reference Train.*(population|anchor|content)"):
        stop_at_admission(monkeypatch,cfg,population,reference)


@pytest.mark.parametrize("cycles", [False,True])
def test_canonical_no_future_and_cycle_admission_reaches_predictor(monkeypatch,cycles):
    cfg = balanced_cfg()
    population = admission_populations(cfg,cycles=cycles)
    train,_,_,reference = population
    evidence = train["admission"].get("reference_train_evidence")
    assert evidence is not None
    identities = [{k:r[k] for k in ("unit_id","physical_unit_id","origin_s")} for r in evidence["records"]]
    assert identities == reference["origin_manifest"]["records"]
    assert evidence["evidence_hash"] == canonical_hash({k:v for k,v in evidence.items() if k != "evidence_hash"})
    if not cycles:
        empty = int(np.flatnonzero(reference["units"] == "train-empty")[0])
        assert not reference["mask"][empty].any()
        assert not reference["weights"][empty].any()
        # Invalid payload outside support cannot influence the content comparison.
        reference["y"][empty] = np.inf
    else:
        chosen = min(("train-a","train-empty"),key=lambda uid:(canonical_hash(["equipment-cycle-3",uid]),uid))
        assert chosen in reference["units"]
    with pytest.raises(AdmissionBoundary,match="before predictor"):
        stop_at_admission(monkeypatch,cfg,population,reference)


def test_reference_train_cannot_substitute_another_equipment_cycle(monkeypatch):
    cfg = balanced_cfg()
    population = admission_populations(cfg,cycles=True)
    train,_,_,original = population
    reference = deepcopy(original)
    row = int(np.flatnonzero(reference["physical_units"] == "equipment-cycle-3")[0])
    if reference["units"][row] != "train-empty":
        pytest.fail("Fixture must choose the no-future cycle independently of tail length")
    other = int(np.flatnonzero(train["units"] == "train-a")[0])
    for key in ("x","y","mask","units","physical_units","origin_s"):
        reference[key][row] = train[key][other]
    reference["origin_manifest"]["records"][row] = deepcopy(train["origin_manifest"]["records"][other])
    rehash_reference(reference)
    with pytest.raises(ValueError,match="Reference Train.*anchor"):
        stop_at_admission(monkeypatch,cfg,population,reference)
