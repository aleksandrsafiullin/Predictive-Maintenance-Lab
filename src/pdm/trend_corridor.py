"""Learn a bounded signal corridor; derive RED timing from its two boundaries.

The relative half-width is ±10–15% of the predicted level. These are decision
bounds, not calibrated confidence intervals. Misses remain errors in both the
training objective and saved evaluation; inference never expands the bounds.
"""
from __future__ import annotations

import copy
import time
import uuid
from functools import lru_cache
from numbers import Integral, Real

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingRegressor
from torch import nn
from torch.nn import functional as F

from pdm.io_util import atomic_write_json, read_json, sha256_file
from pdm.models.recurrent import RecurrentEncoder
from pdm.models.signal_full_cns import build_signal_full_cns

MODE = "bounded_trend_corridor"
WIDTH = .20  # Total width: ±10%.
MAX_WIDTH = .30  # Total width: ±15%.
LEGACY_CONTRACT = {
    "version": 1,
    "target": "future recorded signal on the saved cadence grid",
    "width_definition": "(upper-lower)/abs(predicted_level)",
    "target_relative_width": .10,
    "maximum_relative_width": .15,
    "zero_level_policy": "zero width at exactly zero predicted level",
    "objective": "masked equipment-balanced interval score and excess width penalty",
    "red_entry_source": "first boundary crossings; no separate event head",
    "future_inputs": False,
    "coverage_guarantee": False,
}
CONTRACT = {**LEGACY_CONTRACT, "version": 2,
            "target_relative_width": WIDTH, "maximum_relative_width": MAX_WIDTH}


def corridor_widths(contract):
    """Decode only known contracts; saved v1 runs retain their original bounds."""
    if contract not in (LEGACY_CONTRACT, CONTRACT):
        raise ValueError("Unsupported saved corridor contract")
    return contract["target_relative_width"], contract["maximum_relative_width"]


def corridor_params(engine, params, features):
    from pdm.signal_training import ENGINES

    if engine not in ENGINES:
        raise ValueError("Unsupported corridor engine")
    defaults = dict(forecast_mode=MODE, history_length=8, hidden_size=64,
                    epochs=60, batch_size=64, learning_rate=.001, seed=20261005,
                    max_iter=80, max_windows_per_unit=256, cpu_threads=2,
                    patience=12, nominal_coverage=.9)
    raw = dict(params or {})
    if set(raw) - (set(defaults) | {"horizons_s", "target_tolerance_s"}):
        raise ValueError("Unknown bounded corridor parameters")
    out = {**defaults, **raw}
    if out["forecast_mode"] != MODE:
        raise ValueError("Invalid corridor mode")
    for key, minimum, maximum in (("history_length", 2, 256), ("hidden_size", 4, 1024),
                                  ("epochs", 1, 500), ("batch_size", 1, 4096),
                                  ("max_iter", 2, 1000), ("max_windows_per_unit", 2, 10000),
                                  ("cpu_threads", 1, 16), ("patience", 1, 500),
                                  ("seed", 0, 2**31-1)):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, Integral) or not minimum <= value <= maximum:
            raise ValueError(f"Invalid corridor {key}")
        out[key] = int(value)
    for key, minimum, maximum in (("learning_rate", 0, .1), ("nominal_coverage", 0, 1)):
        value = out[key]
        if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value) or not minimum < value < maximum:
            raise ValueError(f"Invalid corridor {key}")
        out[key] = float(value)
    grid = raw.get("horizons_s")
    if not isinstance(grid, (list, tuple)) or not 1 <= len(grid) <= 4096:
        raise ValueError("Corridor needs 1–4096 dense saved horizons")
    if any(isinstance(v, bool) or not isinstance(v, Real) for v in grid):
        raise ValueError("Invalid corridor horizon")
    hs = np.asarray(grid, float)
    if not np.isfinite(hs).all() or hs[0] <= 0 or not np.allclose(hs, np.arange(1, len(hs)+1)*hs[0], rtol=0, atol=1e-6):
        raise ValueError("Corridor horizons must form a dense positive cadence grid")
    out["horizons_s"] = hs.tolist()
    out["target_tolerance_s"] = max(1e-6, min(hs[0]*.01, .01))
    return out


class CorridorNet(nn.Module):
    def __init__(self, engine, config, external_size=0, contract=CONTRACT):
        super().__init__()
        self.width, self.max_width = corridor_widths(contract)
        hidden = config["hidden_size"]
        self.encoder = (RecurrentEncoder(1, hidden, architecture=engine)
                        if engine in {"gru", "lstm"} else
                        nn.Sequential(nn.Linear(external_size, hidden), nn.Tanh()))
        self.head = nn.Linear(hidden, 2*len(config["horizons_s"]))
        # Start from a narrow last-value corridor rather than an arbitrary jump.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        with torch.no_grad():
            self.head.bias[len(config["horizons_s"]):] = -4

    def forward(self, x, current, scale, nonnegative):
        level_delta, width_logits = self.head(self.encoder(x)).chunk(2, -1)
        if nonnegative:
            anchor = (current/scale).clamp_min(1e-8)
            inverse = anchor + torch.log(-torch.expm1(-anchor))
            level = F.softplus(inverse[:, None]+level_delta)*scale
        else:
            level = current[:, None]+level_delta*scale
        relative_width = self.width+(self.max_width-self.width)*torch.sigmoid(width_logits)
        radius = level.abs()*relative_width/2
        return level-radius, level+radius


def interval_loss(lower, upper, target, mask, weights, scale, coverage, nonnegative, contract=CONTRACT):
    # Replace unknown values before any arithmetic, including logarithms.
    target = torch.where(mask, target, torch.zeros_like(target))
    if nonnegative:
        lo, hi, y = (torch.log(v.clamp_min(1e-8)) for v in (lower, upper, target))
    else:
        lo, hi, y = lower/scale, upper/scale, target/scale
    score = hi-lo+2/(1-coverage)*(F.relu(lo-y)+F.relu(y-hi))
    level = (lower+upper)/2
    relative = (upper-lower)/level.abs().clamp_min(1e-8)
    width, max_width = corridor_widths(contract)
    score += .2*((relative-width)/(max_width-width)).square()
    return (torch.where(mask, score, torch.zeros_like(score))*weights).sum()


def _target_weights(frame):
    """Equal equipment, then equal supported leads and origins within equipment."""
    mask = frame["mask"]
    groups = np.asarray(frame["physical_unit_id"])
    active_groups = [g for g in np.unique(groups) if mask[groups == g].any()]
    weights = np.zeros(mask.shape, np.float32)
    for g in active_groups:
        rows = groups == g
        counts = mask[rows].sum(0)
        weights[rows] = mask[rows]/np.maximum(counts, 1)/max(1, (counts > 0).sum())/len(active_groups)
    return weights


def _check(stop):
    if stop and stop():
        raise InterruptedError("Trend corridor training cancelled")


def _input(frame, scaler):
    x = frame["x"]
    if scaler["output_domain"] == "nonnegative":
        x = np.log1p(x)
    return ((x-scaler["mean"])/scaler["std"]).astype(np.float32)


def _design(frame, scaler, encoder, stop=None):
    x = _input(frame, scaler)
    if encoder is None:
        return x
    if encoder["kind"] == "full_cns":
        return encoder["body"].transform(x, should_stop=stop)
    flat = x.reshape(len(x), -1)
    return np.column_stack([h.predict(flat) for h in encoder["heads"]]).astype(np.float32)


def predict_corridor(bundle, frame, stop=None):
    if not len(frame["x"]):
        shape = frame["y"].shape
        return np.empty(shape), np.empty(shape)
    x = _design(frame, bundle["scaler"], bundle["encoder"], stop)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(x), bundle["config"]["batch_size"]):
            _check(stop)
            end = start+bundle["config"]["batch_size"]
            lo, hi = bundle["model"](torch.from_numpy(x[start:end]),
                                     torch.from_numpy(frame["x"][start:end, -1, 0]),
                                     bundle["scaler"]["signal_scale"],
                                     bundle["scaler"]["output_domain"] == "nonnegative")
            outputs.append((lo.numpy(), hi.numpy()))
    lower, upper = (np.concatenate([o[j] for o in outputs]) for j in (0, 1))
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise FloatingPointError("Nonfinite corridor model output")
    return lower, upper


def evaluate_corridor(bundle, frame, stop=None):
    lo, hi = predict_corridor(bundle, frame, stop)
    mask = frame["mask"]
    weights = _target_weights(frame)
    inside = (frame["y"] >= lo) & (frame["y"] <= hi) & mask
    level = (hi+lo)/2
    relative = np.divide(hi-lo, np.abs(level), out=np.zeros_like(level), where=level != 0)
    groups = np.asarray(frame["physical_unit_id"])
    def macro(values, admitted):
        scores = [float(np.mean(values[(groups == g) & admitted])) for g in np.unique(groups)
                  if ((groups == g) & admitted).any()]
        return float(np.mean(scores)) if scores else None
    horizons = []
    # Report operational prefixes and the full support, rather than thousands of rows.
    hs = np.asarray(bundle["config"]["horizons_s"])
    for h in sorted({min(v, hs[-1]) for v in (600, 1200, 1800, 3600, 7200, hs[-1])}):
        n = int(np.searchsorted(hs, h, side="right"))
        if not n:
            continue
        complete = mask[:, :n].all(1)
        horizons.append(dict(horizon_s=float(hs[n-1]),
                             whole_path_coverage=macro(inside[:, :n].all(1), complete),
                             complete_physical_groups=len(set(groups[complete])),
                             complete_origins=int(complete.sum())))
    return dict(known_targets=int(mask.sum()), physical_group_count=len(set(groups)),
                point_coverage=float((inside*weights).sum()) if mask.any() else None,
                mean_relative_width=float((relative*weights).sum()) if mask.any() else None,
                maximum_relative_width=float(relative.max()) if relative.size else None,
                interval_score=float(interval_loss(torch.from_numpy(lo), torch.from_numpy(hi),
                    torch.from_numpy(frame["y"]), torch.from_numpy(mask), torch.from_numpy(weights),
                    bundle["scaler"]["signal_scale"], bundle["config"]["nominal_coverage"],
                    bundle["scaler"]["output_domain"] == "nonnegative",
                    bundle["corridor_contract"])) if mask.any() else None,
                horizons=horizons, evaluation_status="exploratory_reused_splits",
                quality_accepted=False, coverage_guarantee=False)


def train_corridor_run(project_id, data, engine, params, stop=None, report=None):
    from pdm.projects import project_store
    from pdm.signal_training import _digest, _joint_windows, _physical_map, _weights

    report = report or (lambda _: None)
    train_features = data["features"][data["features"].unit_id.astype(str).isin(map(str, data["split"]["train"]))]
    config = corridor_params(engine, params, train_features)
    physical = _physical_map(data)
    frames = {part: _joint_windows(data["features"], data["split"][part], config, physical)
              for part in ("train", "validation")}
    train, val = frames["train"], frames["validation"]
    if not len(train["x"]) or not len(val["x"]) or not train["mask"].any(0).all():
        raise ValueError("Train/Validation need known targets; every output lead needs Train support")
    torch.set_num_threads(config["cpu_threads"])
    torch.manual_seed(config["seed"])
    nonnegative = data["schema"].get("output_domain") == "nonnegative"
    w = _weights(train["physical_unit_id"])
    raw = np.log1p(train["x"]) if nonnegative else train["x"]
    mean = float(np.average(raw.mean((1, 2)), weights=w))
    std = float(max(np.sqrt(np.average(((raw-mean)**2).mean((1, 2)), weights=w)), 1e-5))
    scaler = dict(mean=mean, std=std, signal_scale=float(max(np.average(np.abs(train["x"]).mean((1, 2)), weights=w), 1e-5)),
                  output_domain="nonnegative" if nonnegative else "real",
                  fit_physical_groups=sorted(set(train["physical_unit_id"])))
    encoder = None
    if engine == "full_cns":
        report(dict(stage="encoding", message="Encoding causal histories with the complete MaleCNS"))
        encoder = dict(kind=engine, body=build_signal_full_cns(config["seed"]))
        encoder["body"].provenance["readout_policy"] = "learned bounded trend corridor; RED derived from boundaries"
    elif engine == "quantile_boosting":
        flat = _input(train, scaler).reshape(len(train["x"]), -1)
        indices = np.unique(np.linspace(0, len(config["horizons_s"])-1, min(12, len(config["horizons_s"])), dtype=int))
        heads = []
        for j in indices:
            admitted = train["mask"][:, j]
            for q in (.1, .5, .9):
                _check(stop)
                tree = HistGradientBoostingRegressor(loss="quantile", quantile=q,
                    max_iter=config["max_iter"], max_leaf_nodes=15, min_samples_leaf=5,
                    early_stopping=False, random_state=config["seed"])
                tree.fit(flat[admitted], train["y"][admitted, j],
                         sample_weight=_weights(np.asarray(train["physical_unit_id"])[admitted].tolist()))
                heads.append(tree)
            report(dict(stage="encoding", message=f"Fitting corridor tree features {j+1}/{len(config['horizons_s'])}"))
        encoder = dict(kind=engine, heads=heads)
    design = {part: _design(f, scaler, encoder, stop) for part, f in frames.items()}
    model = CorridorNet(engine, config, design["train"].shape[-1] if encoder else 0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    targets = {part: torch.from_numpy(f["y"]) for part, f in frames.items()}
    masks = {part: torch.from_numpy(f["mask"]) for part, f in frames.items()}
    weights = {part: torch.from_numpy(_target_weights(f)) for part, f in frames.items()}
    def loss(part, indexes):
        _check(stop)
        f = frames[part]
        lo, hi = model(torch.from_numpy(design[part][indexes]),
                       torch.from_numpy(f["x"][indexes, -1, 0]), scaler["signal_scale"], nonnegative)
        return interval_loss(lo, hi, targets[part][indexes], masks[part][indexes],
                             weights[part][indexes], scaler["signal_scale"], config["nominal_coverage"], nonnegative)
    best, selected, best_state, stale, trace = float("inf"), 0, None, 0, []
    rng = np.random.default_rng(config["seed"])
    for epoch in range(config["epochs"]):
        model.train()
        order = rng.permutation(len(train["x"]))
        training_loss = 0.
        for start in range(0, len(order), config["batch_size"]):
            _check(stop)
            indexes = order[start:start+config["batch_size"]]
            objective = loss("train", indexes)
            training_loss += float(objective.detach())
            optimizer.zero_grad()
            (objective*len(train["x"])/len(indexes)).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
        model.eval()
        with torch.inference_mode():
            score = sum(float(loss("validation", slice(s, s+config["batch_size"])))
                        for s in range(0, len(val["x"]), config["batch_size"]))
        if not np.isfinite(score):
            raise FloatingPointError("Nonfinite corridor selection loss")
        trace.append(dict(epoch=epoch+1, train_interval_score=training_loss, validation_interval_score=score))
        if score < best:
            best, selected, best_state, stale = score, epoch+1, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
        report(dict(stage="training", progress=(epoch+1)/config["epochs"],
                    message=f"Trend corridor · epoch {epoch+1}/{config['epochs']} · Validation score {score:.3f}"))
        if stale >= config["patience"]:
            break
    _check(stop)
    model.load_state_dict(best_state)
    model.eval()
    bundle = dict(model=model, scaler=scaler, encoder=encoder, config=config, corridor_contract=CONTRACT)
    store = project_store()
    run_id = "signal-"+time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())+"-"+uuid.uuid4().hex[:8]
    directory = store.run_path(project_id, run_id)
    directory.mkdir(parents=True, exist_ok=False)
    torch.save(model.state_dict(), directory/"corridor.pt")
    metadata = dict(engine_id=engine, params=config, scaler=scaler,
                    external_size=design["train"].shape[-1] if encoder else 0, corridor_contract=CONTRACT)
    atomic_write_json(directory/"corridor_model.json", metadata)
    atomic_write_json(directory/"objective_trace.json", trace)
    artifact_names = ["corridor.pt", "corridor_model.json", "objective_trace.json"]
    if encoder:
        joblib.dump(encoder, directory/"corridor_encoder.joblib")
        artifact_names.append("corridor_encoder.joblib")
    contract = dict(project_id=project_id, snapshot_id=data["snapshot_id"], engine_id=engine,
                    params=config, scaler=scaler, schema=data["schema"],
                    snapshot_fingerprint_sha256=_digest(data["fingerprint"]), corridor_contract=CONTRACT)
    if engine == "full_cns":
        contract["connectome"] = encoder["body"].provenance
    atomic_write_json(directory/"training_contract.json", contract)
    artifact_names.append("training_contract.json")
    manifest = dict(**contract, run_id=run_id, task="signal_forecast", status="completed",
                    schema_version="project_trend_corridor_v2", artifact="corridor.pt",
                    artifacts={name: sha256_file(directory/name) for name in artifact_names},
                    interval_status="bounded_corridor_uncalibrated", quality_accepted=False,
                    selection=dict(criterion="validation_equipment_equal_interval_score", selected_epoch=selected,
                                   validation_score=best, test_feedback=False),
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    restored = load_corridor_bundle({**manifest, "dir": directory})
    audit = {**val, "x": val["x"][:3], "y": val["y"][:3]}
    if any(not np.array_equal(a, b) for a, b in zip(predict_corridor(bundle, audit), predict_corridor(restored, audit))):
        raise AssertionError("Saved corridor reload differs")
    manifest["reload_verified"] = True
    report(dict(stage="evaluating", message="Scoring width and containment; misses remain errors"))
    # Evaluate all eligible origins after selection, never fit against Test.
    evaluation_config = {**config, "max_windows_per_unit": None}
    manifest["metrics"] = {part: evaluate_corridor(restored, _joint_windows(data["features"],
                          data["split"][part], evaluation_config, physical), stop) for part in ("validation", "test")}
    _check(stop)
    atomic_write_json(directory/"manifest.json", manifest)
    store.update(project_id, selected_run_id=run_id)
    report(dict(stage="completed", progress=1., message="Trend corridor saved and reload verified"))
    return manifest


def load_corridor_bundle(run):
    metadata = read_json(run["dir"]/"corridor_model.json")
    for key in ("engine_id", "params", "scaler", "corridor_contract"):
        if metadata.get(key) != run.get(key):
            raise ValueError(f"Saved corridor {key} mismatch")
    corridor_widths(metadata["corridor_contract"])
    corridor_params(run["engine_id"], run["params"], None)
    encoder = joblib.load(run["dir"]/"corridor_encoder.joblib") if metadata["external_size"] else None
    if encoder and encoder["kind"] != run["engine_id"]:
        raise ValueError("Saved corridor encoder mismatch")
    if encoder and encoder["kind"] == "full_cns" and encoder["body"].provenance != run.get("connectome"):
        raise ValueError("Saved corridor connectome mismatch")
    model = CorridorNet(run["engine_id"], run["params"], metadata["external_size"], metadata["corridor_contract"])
    model.load_state_dict(torch.load(run["dir"]/"corridor.pt", map_location="cpu", weights_only=True))
    model.eval()
    return dict(model=model, encoder=encoder, config=run["params"], scaler=run["scaler"],
                corridor_contract=metadata["corridor_contract"])


@lru_cache(maxsize=8)
def _cached_bundle(project_id, run_id, digest, root):
    from pdm.signal_training import load_signal_run
    return load_corridor_bundle(load_signal_run(project_id, run_id))


def boundary_red_entry(points, current, threshold, issued, *, previously_red=False):
    rule = dict(threshold)
    result = dict(status="unavailable", earliest_s=None, latest_s=None,
                  source="trend_corridor_boundaries", coverage_guarantee=False)
    if rule.get("status") != "available":
        return result
    above = rule["direction"] == "above"
    def beyond(v):
        return v >= rule["red"] if above else v <= rule["red"]
    if beyond(current):
        return {**result, "status": "already_red"}
    if previously_red:
        return {**result, "status": "previously_red"}
    possible, certain = ("upper", "lower") if above else ("lower", "upper")
    first_possible = next((i for i, p in enumerate(points) if beyond(p[possible])), None)
    if first_possible is None:
        return {**result, "status": "none_within_horizon"}
    first_certain = next((p for p in points if beyond(p[certain])), None)
    earliest = issued if first_possible == 0 else points[first_possible-1]["target_time_s"]
    return {**result, "status": "derived" if first_certain else "open",
            "earliest_s": earliest,
            "latest_s": first_certain["target_time_s"] if first_certain else None}


def forecast_corridor_prefix(run, data, prefix, unit_id, stop=None, prediction_horizon_s=None, thresholds=None):
    from pdm.project_zones import is_beyond, resolve_thresholds
    from pdm.projects import project_store
    from pdm.signal_training import _segments

    hs = np.asarray(run["params"]["horizons_s"])
    if prediction_horizon_s is not None:
        if prediction_horizon_s < hs[0] or prediction_horizon_s > hs[-1]:
            raise ValueError("Prediction span must be within the saved corridor horizon")
        hs = hs[hs <= prediction_horizon_s]
    issued = float(prefix.timestamp_s.iloc[-1]) if len(prefix) else None
    rule = resolve_thresholds({**run["schema"], **({"thresholds": thresholds} if thresholds else {})}, prefix)
    result = dict(project_id=run["project_id"], run_id=run["run_id"], snapshot_id=run["snapshot_id"],
                  as_of_s=issued, observed_prefix=prefix[["timestamp_s", "signal"]].to_dict("records"),
                  points=[], thresholds=rule, status="unavailable", reason=None,
                  calibration_status="bounded_corridor_uncalibrated", crossing=dict(status="unavailable", time_s=None),
                  red_entry_corridor=dict(status="unavailable"), corridor_contract=run["corridor_contract"])
    history = run["params"]["history_length"]
    segments = _segments(prefix, unit_id)
    if not segments or len(segments[-1]) < history:
        result["reason"] = f"Need {history} observations since the last gap"
        return result
    signal = segments[-1].signal.to_numpy(np.float32)[-history:]
    frame = dict(x=signal[None, :, None], y=np.zeros((1, len(run["params"]["horizons_s"])), np.float32))
    bundle = _cached_bundle(run["project_id"], run["run_id"], tuple(sorted(run["artifacts"].items())), str(project_store().root))
    lo, hi = predict_corridor(bundle, frame, stop)
    points = [dict(target_time_s=issued+float(h), value=float((lo[0, j]+hi[0, j])/2),
                   lower=float(lo[0, j]), upper=float(hi[0, j]), kind="direct") for j, h in enumerate(hs)]
    previously = rule.get("status") == "available" and bool(is_beyond(prefix.signal.to_numpy(), rule["red"], rule["direction"]).any())
    red = boundary_red_entry(points, float(signal[-1]), rule, issued, previously_red=previously)
    width, max_width = corridor_widths(run["corridor_contract"])
    result.update(points=points, status="available", red_entry_corridor=red,
                  crossing=dict(status=red["status"], time_s=None),
                  funnel=dict(mode=MODE, issued_horizon_s=float(hs[-1]), target_relative_width=width,
                              maximum_relative_width=max_width, coverage_guarantee=False))
    return result
