"""Versioned, observed-only parameter forecasts for GRU, LSTM and full MaleCNS.

All available continuous observations enter the causal filter and descriptors.
The recurrent encoders consume an explicit aggregation of that same prefix.
Readouts change global trend parameters, never independent future values.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
from functools import lru_cache

import numpy as np
import torch
from torch import nn

from pdm.long_forecast_data import MIN_HISTORY, segments

PROTOCOL = "stable_observed_trend_v5"
PROTOCOLS = {
    "stable_observed_trend_v2",
    "stable_observed_trend_v3",
    "stable_observed_trend_v4",
    PROTOCOL,
}
ENGINES = ("gru", "lstm", "full_cns")
LABELS = {"gru": "GRU", "lstm": "LSTM", "full_cns": "MaleCNS"}


def config(horizon, cadence=60.0, **overrides):
    cfg = dict(
        protocol=PROTOCOL,
        horizon=int(horizon),
        cadence_s=float(cadence),
        history_policy="all_observed_continuous",
        minimum_history=MIN_HISTORY,
        history_scales=[60, 120, 240, "all"],
        width=0.45,
        seed=21,
        hidden_size=128,
        epochs=80,
        patience=20,
        batch_size=64,
        learning_rate=0.001,
        threads=2,
        stride=60,
        sequence_bins=60,
        cns_sequence_bins=16,
        parameter_bound=math.log(1.5),
        prior_rate_scale=None,
        prior_curve_scale=None,
        prior_basis_size=8,
        parameter_penalty=0.02,
        temporal_penalty=0.5,
        selection="worst_supported_validation_quarter_containment_then_log_error",
        future_inputs=False,
        coverage_guarantee=False,
    )
    if overrides.keys() - cfg.keys():
        raise ValueError("Unknown stable forecast configuration")
    cfg.update(overrides)
    if (
        cfg["protocol"] not in PROTOCOLS
        or not 1 <= cfg["horizon"] <= 4096
        or cfg["width"] not in (0.30, 0.45)
        or cfg["history_policy"] != "all_observed_continuous"
        or cfg["minimum_history"] != MIN_HISTORY
    ):
        raise ValueError("Unsupported stable forecast contract")
    for key in (
        "seed",
        "hidden_size",
        "epochs",
        "patience",
        "batch_size",
        "threads",
        "stride",
        "sequence_bins",
        "cns_sequence_bins",
    ):
        if (
            isinstance(cfg[key], bool)
            or not isinstance(cfg[key], int)
            or cfg[key] < (0 if key == "seed" else 1)
        ):
            raise ValueError(f"Invalid {key}")
    if not np.isfinite(cfg["cadence_s"]) or cfg["cadence_s"] <= 0:
        raise ValueError("Invalid cadence")
    return cfg


def supported_horizon(train, validation, cadence, fraction=0.5):
    """Independent unit support at every lead, using only Train/Validation."""
    limits, evidence = [], {}
    for role, frame in (("train", train), ("validation", validation)):
        duration = {}
        for _, part in segments(frame, cadence):
            uid = str(part.physical_unit_id.iloc[0])
            duration[uid] = max(duration.get(uid, 0), max(0, len(part) - MIN_HISTORY))
        required = max(2, math.ceil(len(duration) * fraction))
        lengths = sorted(duration.values(), reverse=True)
        if len(lengths) < required or lengths[required - 1] < 1:
            raise ValueError(
                f"{role.title()} needs at least two independent units with future targets"
            )
        limits.append(lengths[required - 1])
        evidence[role] = dict(total_units=len(lengths), required_units=required, lengths=lengths)
    horizon = min(4096, *limits)
    for role in evidence:
        evidence[role]["support_at_last_lead"] = sum(
            n >= horizon for n in evidence[role].pop("lengths")
        )
    return dict(
        horizon=horizon,
        cadence_s=cadence,
        required_fraction=fraction,
        rule="at_least_half_of_independent_units_in_each_role_and_at_least_two",
        **evidence,
        quality_guarantee=False,
    )


def full_windows(frame, horizon, cadence, stride=60):
    histories, targets, masks, units, physical, origins = [], [], [], [], [], []
    for uid, part in segments(frame, cadence):
        values = part.signal.to_numpy(float)
        for i in range(MIN_HISTORY - 1, len(values) - 1, stride):
            histories.append(values[: i + 1])
            y, mask = np.zeros(horizon, np.float32), np.zeros(horizon, bool)
            future = values[i + 1 : i + 1 + horizon]
            y[: len(future)], mask[: len(future)] = future, True
            targets.append(y)
            masks.append(mask)
            units.append(uid)
            physical.append(str(part.physical_unit_id.iloc[0]))
            origins.append(float(part.timestamp_s.iloc[i]))
    if not histories:
        raise ValueError("No admitted full-history windows")
    lengths = np.array([len(x) for x in histories])
    x = np.zeros((len(histories), lengths.max()), np.float32)
    for row, history in zip(x, histories, strict=True):
        row[-len(history) :] = history
    return dict(
        x=x,
        lengths=lengths,
        y=np.array(targets),
        mask=np.array(masks),
        units=np.array(units),
        physical=np.array(physical),
        origins=np.array(origins),
        split=frame.attrs.get("split"),
        horizon=horizon,
        cadence_s=cadence,
    )


def full_at_origin(frame, uid, origin, cadence):
    prefix = frame.loc[frame.unit_id.eq(uid) & frame.timestamp_s.le(origin)]
    parts = list(segments(prefix, cadence))
    if not parts or parts[-1][1].timestamp_s.iloc[-1] != origin:
        raise ValueError("Now must be an admitted measurement")
    values = parts[-1][1].signal.to_numpy(float)
    if len(values) < MIN_HISTORY:
        raise ValueError("60 continuous observations are required to start a forecast")
    return values, len(values)


def _filter(observed):
    # Trailing medians: every filtered value uses only measurements already seen.
    padded = np.pad(observed, (8, 0), mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, 9), axis=1)


def _fit(values, cadence, degree=1):
    t = (np.arange(len(values)) - len(values) + 1) * cadence / 3600
    matrix = np.column_stack([t**k for k in range(degree + 1)])
    # Smooth recency weights avoid switching abruptly between competing windows.
    recency = np.exp(t / 1.5)
    weights = recency.copy()
    for _ in range(4):
        root = np.sqrt(weights)
        coef = np.linalg.lstsq(matrix * root[:, None], values * root, rcond=None)[0]
        residual = values - matrix @ coef
        spread = max(1e-5, np.median(np.abs(residual - np.median(residual))) * 1.4826)
        weights = recency * np.minimum(1, 1.5 * spread / np.maximum(np.abs(residual), 1e-12))
    return coef, float(np.median(np.abs(residual)))


def design(x, lengths, cfg):
    features, sequence, levels, slopes, curves = [], [], [], [], []
    for row, n in zip(x, lengths, strict=True):
        n = int(n)
        observed = np.asarray(row, float)[-n:]
        if (
            n < MIN_HISTORY
            or len(observed) != n
            or not np.isfinite(observed).all()
            or (observed <= 0).any()
        ):
            raise ValueError("Invalid full continuous history")
        filtered = _filter(observed)
        scale = max(1e-8, float(np.median(filtered[-15:])))
        relative = filtered / scale
        noise = max(
            1e-4, float(np.median(np.abs(np.diff(observed) - np.median(np.diff(observed)))) / scale)
        )
        tail = relative[-min(240, n) :]
        linear, le = _fit(tail, cfg["cadence_s"])
        quadratic, qe = _fit(tail, cfg["cadence_s"], 2)
        # Acceleration must improve an observed fit; soft weighting avoids a
        # binary candidate switch when one additional noisy sample arrives.
        improvement = max(0.0, le - qe) / (le + noise + 1e-5)
        reliability = improvement / (improvement + 0.15)
        slope = (1 - reliability) * linear[1] + reliability * quadratic[1]
        span = max(1.0, min(240, n) * cfg["cadence_s"] / 3600)
        curve = max(0.0, quadratic[2]) * reliability if slope > 0 else 0.0
        curve = min(curve, max(0.0, slope) / span)
        # Compensate the known four-sample lag of the trailing median.
        lag = 4 * cfg["cadence_s"] / 3600
        intercept = (1 - reliability) * linear[0] + reliability * quadratic[0]
        level = max(1e-8, scale * (intercept + slope * lag + curve * lag**2))
        feat = [np.log(level), noise, np.log1p(n * cfg["cadence_s"] / 3600), slope, curve]
        for window in (60, 120, 240, n):
            values = np.log(relative[-min(window, n) :])
            a, error = _fit(values, cfg["cadence_s"])
            q, qerror = _fit(values, cfg["cadence_s"], 2)
            feat.extend([a[0], a[1], q[2], error, qerror])
        relative_log = np.log(filtered / level)
        blocks = np.array_split(relative_log, min(cfg["sequence_bins"], n))
        seq = np.array([np.mean(block) for block in blocks], np.float32)
        if len(seq) < cfg["sequence_bins"]:
            seq = np.pad(seq, (cfg["sequence_bins"] - len(seq), 0))
        features.append(feat)
        sequence.append(seq)
        levels.append(level)
        slopes.append(slope)
        curves.append(curve)
    return dict(
        features=np.asarray(features, np.float32),
        sequence=np.asarray(sequence, np.float32),
        level=np.asarray(levels, np.float32),
        slope=np.asarray(slopes, np.float32),
        curve=np.asarray(curves, np.float32),
    )


def weights(data, descriptors):
    from pdm.long_forecast import target_weights

    proxy = np.zeros((len(descriptors["features"]), 6))
    proxy[:, 1] = descriptors["features"][:, 1]
    proxy[:, 5] = descriptors["slope"]
    return target_weights(data, proxy)


def curve_log(parameters, descriptors, cfg):
    """Continuous trend parameters plus an optional Train-learned wear prior."""
    tensor = torch.is_tensor(parameters)
    if tensor:
        values = {
            k: torch.as_tensor(descriptors[k], device=parameters.device)
            for k in ("level", "slope", "curve")
        }
        t = torch.arange(1, cfg["horizon"] + 1, device=parameters.device) * cfg["cadence_s"] / 3600
        exp, tanh, log, maximum = torch.exp, torch.tanh, torch.log, torch.clamp_min
    else:
        values = descriptors
        t = np.arange(1, cfg["horizon"] + 1) * cfg["cadence_s"] / 3600
        exp, tanh, log = np.exp, np.tanh, np.log
        maximum = np.maximum
    multiplier = exp(cfg["parameter_bound"] * tanh(parameters[:, :2]))
    slope = values["slope"][:, None] * multiplier[:, :1]
    acceleration = values["curve"][:, None] * multiplier[:, 1:2]
    if cfg["protocol"] == "stable_observed_trend_v3":
        # Train-derived growth scales allow the learned readout to represent
        # gradual wear even before the observed local slope becomes large.
        # These are global continuous parameters, never per-lead tree outputs.
        slope = slope + maximum(tanh(parameters[:, 2:3]), 0) * (cfg.get("prior_rate_scale") or 0.0)
        acceleration = acceleration + maximum(tanh(parameters[:, 3:4]), 0) * (
            cfg.get("prior_curve_scale") or 0.0
        )
    growth = slope * t + acceleration * t**2
    # A declining observed regime has a positive exponential decay. A growing
    # regime has physical linear/quadratic continuation, avoiding log blow-up.
    if cfg["protocol"] == "stable_observed_trend_v3":
        if tensor:
            relative = torch.where(growth >= 0, 1 + growth, exp(torch.clamp_max(growth, 0)))
        else:
            relative = np.where(growth >= 0, 1 + growth, exp(np.minimum(growth, 0)))
    elif tensor:
        relative = torch.where(slope >= 0, 1 + growth, exp(torch.clamp_max(slope * t, 0)))
    else:
        relative = np.where(slope >= 0, 1 + growth, exp(np.minimum(slope * t, 0)))
    result = log(values["level"][:, None]) + log(maximum(relative, 1e-8))
    if cfg["protocol"] in {"stable_observed_trend_v4", PROTOCOL}:
        basis = _wear_basis(cfg["horizon"], cfg["cadence_s"], cfg["prior_basis_size"])
        if tensor:
            basis = torch.as_tensor(basis, device=parameters.device)
        if cfg["protocol"] == PROTOCOL:
            if tensor:
                fractions = torch.sigmoid(parameters[:, 2:]).square()
            else:
                from scipy.special import expit

                fractions = expit(parameters[:, 2:]) ** 2
        else:
            fractions = maximum(tanh(parameters[:, 2:]), 0)
        rates = fractions * (cfg.get("prior_rate_scale") or 0.0)
        result = result + rates @ basis
    return result


@lru_cache(maxsize=16)
def _wear_basis(horizon, cadence, count):
    from scipy.special import betainc

    u = np.arange(1, horizon + 1) / horizon
    k = np.arange(count)[:, None]
    # Integrals of a partition of unity. Positive coefficients create a smooth
    # nonnegative growth rate; their sum integrates to elapsed time exactly.
    return (horizon * cadence / 3600 / count * betainc(k + 1, count - k, u)).astype(np.float32)


class ParameterRecurrent(nn.Module):
    def __init__(self, engine, hidden, features, parameters=10):
        super().__init__()
        self.engine = engine
        self.encoder = (nn.GRU if engine == "gru" else nn.LSTM)(1, hidden, batch_first=True)
        self.head = nn.Linear(hidden + features, parameters)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sequence, features):
        _, state = self.encoder(sequence.unsqueeze(-1))
        if self.engine == "lstm":
            state = state[0]
        return self.head(torch.cat((state[-1], features), -1))


class ParameterCNS(nn.Module):
    def __init__(self, features, parameters=10):
        super().__init__()
        self.head = nn.Linear(features, parameters)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, sequence, features):
        return self.head(features)


def _cns_sequence(sequence, bins):
    return np.stack(
        [
            sequence[:, chunk].mean(1)
            for chunk in np.array_split(np.arange(sequence.shape[1]), bins)
        ],
        1,
    )[..., None]


def _features(model, descriptors, report=None, stop=None):
    features = descriptors["features"]
    if model["engine"] == "full_cns":
        pooled = model["reservoir"].transform(
            _cns_sequence(descriptors["sequence"], model["config"]["cns_sequence_bins"]),
            should_stop=stop,
            status_cb=report,
        )
        features = np.column_stack((features, pooled))
    return features.astype(np.float32)


def _fit_features(model, descriptors, report, stop):
    if model["engine"] != "full_cns":
        return _features(model, descriptors, report, stop)
    from pdm.paths import project_root

    reservoir = model["reservoir"]
    sequence = _cns_sequence(descriptors["sequence"], model["config"]["cns_sequence_bins"])
    source = json.dumps(reservoir.provenance, sort_keys=True, default=str).encode()
    key = hashlib.sha256(
        source + inspect.getsource(type(reservoir).transform).encode() + sequence.tobytes()
    ).hexdigest()
    directory = project_root() / "data/cache/stable-cns-features"
    path = directory / (key + ".npz")
    if stop():
        raise InterruptedError("Training cancelled")
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            pooled = cached["pooled"]
        if (
            pooled.shape != (len(sequence), reservoir.pool.shape[0] + 1)
            or not np.isfinite(pooled).all()
        ):
            raise ValueError("Invalid source-bound MaleCNS feature cache")
        report(len(sequence), len(sequence))
    else:
        pooled = reservoir.transform(sequence, should_stop=stop, status_cb=report)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(key + f".{os.getpid()}.npz")
        np.savez_compressed(temporary, pooled=pooled)
        temporary.replace(path)
    return np.column_stack((descriptors["features"], pooled)).astype(np.float32)


def _parameters(model, descriptors):
    raw = _features(model, descriptors)
    z = ((raw - model["mean"]) / model["std"]).astype(np.float32)
    chunks = []
    model["body"].eval()
    with torch.inference_mode():
        for start in range(0, len(z), 256):
            chunks.append(
                model["body"](
                    torch.from_numpy(descriptors["sequence"][start : start + 256]),
                    torch.from_numpy(z[start : start + 256]),
                ).numpy()
            )
    return np.concatenate(chunks)


def predict(model, x, lengths, horizon=None):
    cfg = model["config"]
    h = cfg["horizon"] if horizon is None else horizon
    if not isinstance(h, int) or not 1 <= h <= cfg["horizon"]:
        raise ValueError("Requested horizon exceeds supported frozen horizon")
    descriptors = design(x, lengths, cfg)
    log_center = curve_log(_parameters(model, descriptors), descriptors, cfg)
    center = np.exp(np.clip(log_center, -80, 80))[:, :h]
    half = cfg["width"] / 2
    return np.stack((center * (1 - half), center, center * (1 + half)), -1)


def selection(log_center, data, target_weights, cfg):
    target = np.log(np.maximum(data["y"], 1e-12))
    delta = log_center - target
    half = cfg["width"] / 2
    inside = (delta >= -np.log1p(half)) & (delta <= -np.log1p(-half))
    quarters = []
    for chunk in np.array_split(np.arange(cfg["horizon"]), 4):
        w = target_weights[:, chunk]
        if w.sum():
            quarters.append(float((w * inside[:, chunk]).sum() / w.sum()))
    return (-min(quarters), float((target_weights * np.abs(delta)).sum()))


def fit(train, validation, cfg, engine, report=None, stop=None):
    cfg = config(**cfg)
    report, stop = report or (lambda _: None), stop or (lambda: False)
    if engine not in ENGINES or train["split"] != "train" or validation["split"] != "validation":
        raise ValueError("Explicit Train/Validation and supported architecture required")
    if set(train["physical"]) & set(validation["physical"]):
        raise ValueError("Physical equipment overlaps Train/Validation")
    for data in (train, validation):
        known = data["y"][data["mask"]]
        if (
            data["horizon"] != cfg["horizon"]
            or data["cadence_s"] != cfg["cadence_s"]
            or not data["mask"].any(0).all()
            or not np.isfinite(known).all()
            or (known <= 0).any()
        ):
            raise ValueError("Invalid supported targets or horizon/cadence")
    td, vd = (
        design(train["x"], train["lengths"], cfg),
        design(validation["x"], validation["lengths"], cfg),
    )
    tw, vw = weights(train, td), weights(validation, vd)
    parameters = (
        2 + cfg["prior_basis_size"]
        if cfg["protocol"] in {"stable_observed_trend_v4", PROTOCOL}
        else 4
        if cfg["protocol"] == "stable_observed_trend_v3"
        else 2
    )
    if parameters > 2:
        quantile = 0.97 if cfg["protocol"] in {"stable_observed_trend_v4", PROTOCOL} else 0.9
        multiplier = 2 if cfg["protocol"] in {"stable_observed_trend_v4", PROTOCOL} else 1
        cfg["prior_rate_scale"] = max(
            0.03, multiplier * float(np.quantile(np.maximum(td["slope"], 0), quantile))
        )
        cfg["prior_curve_scale"] = max(0.01, float(np.quantile(td["curve"], 0.9)))
    model = dict(protocol=cfg["protocol"], engine=engine, config=cfg)
    if engine == "full_cns":
        from pdm.models.signal_full_cns import build_signal_full_cns

        report(dict(stage="connectome", message="Loading the full official MaleCNS connectome"))
        model["reservoir"] = build_signal_full_cns(cfg["seed"])
    raw = []
    for role, descriptors in (("train", td), ("validation", vd)):

        def progress(done, total, role=role):
            if done == total or done % 128 == 0:
                report(dict(stage="reservoir", role=role, done=done, total=total))

        raw.append(_fit_features(model, descriptors, progress, stop))
    model["mean"] = raw[0].mean(0)
    model["std"] = np.maximum(raw[0].std(0), 0.02)
    z, vz = [torch.from_numpy(((a - model["mean"]) / model["std"]).astype(np.float32)) for a in raw]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["threads"])
    network = (
        ParameterCNS(z.shape[1], parameters)
        if engine == "full_cns"
        else ParameterRecurrent(engine, cfg["hidden_size"], z.shape[1], parameters)
    )
    if cfg["protocol"] == PROTOCOL:
        # A hard nonnegative clamp at zero has zero gradient in this runtime.
        # Start the smooth growth activation near zero while keeping it trainable.
        with torch.no_grad():
            network.head.bias[2:].fill_(-2)
    model["body"] = network
    optimizer = torch.optim.Adam(network.parameters(), lr=cfg["learning_rate"])
    sequence, vsequence = torch.from_numpy(td["sequence"]), torch.from_numpy(vd["sequence"])
    target = torch.from_numpy(np.log(np.maximum(train["y"], 1e-12)))
    w = torch.from_numpy(tw)
    baseline_parameters = np.zeros((len(vz), parameters), np.float32)
    baseline_state = copy.deepcopy(network.state_dict())
    if cfg["protocol"] == PROTOCOL:
        baseline_parameters[:, 2:] = -80
        baseline_state["head.bias"][2:] = -80
    baseline = curve_log(baseline_parameters, vd, cfg)
    best, best_epoch, best_state = (
        selection(baseline, validation, vw, cfg),
        0,
        baseline_state,
    )
    baseline_selection = best
    records = []
    rng = np.random.default_rng(cfg["seed"])
    half = cfg["width"] / 2
    # Sensor descriptors one minute earlier provide a consistency constraint.
    previous = None
    if engine != "full_cns":
        eligible = train["lengths"] > MIN_HISTORY
        previous = design(train["x"][eligible, :-1], train["lengths"][eligible] - 1, cfg)
        pz = torch.from_numpy(
            ((previous["features"] - model["mean"]) / model["std"]).astype(np.float32)
        )
        pseq = torch.from_numpy(previous["sequence"])
        index_map = np.full(len(z), -1)
        index_map[eligible] = np.arange(eligible.sum())

    def validate():
        network.eval()
        predicted = []
        with torch.inference_mode():
            for start in range(0, len(vz), 256):
                predicted.append(
                    network(vsequence[start : start + 256], vz[start : start + 256]).numpy()
                )
        return selection(curve_log(np.concatenate(predicted), vd, cfg), validation, vw, cfg)

    for epoch in range(1, cfg["epochs"] + 1):
        network.train()
        for ids in np.array_split(
            rng.permutation(len(z)), max(1, math.ceil(len(z) / cfg["batch_size"]))
        ):
            if stop():
                raise InterruptedError("Training cancelled")
            optimizer.zero_grad()
            params = network(sequence[ids], z[ids])
            desc = {k: td[k][ids] for k in ("level", "slope", "curve")}
            logs = curve_log(params, desc, cfg)
            delta = logs - target[ids]
            excess = torch.relu(delta + np.log1p(-half)) + torch.relu(-delta - np.log1p(half))
            loss = (w[ids] * (delta.abs() + excess)).sum()
            regularized = params
            if cfg["protocol"] == PROTOCOL:
                regularized = torch.cat((params[:, :2], torch.sigmoid(params[:, 2:]).square()), 1)
            loss += cfg["parameter_penalty"] * regularized.square().mean() * float(w[ids].sum())
            if previous is not None and cfg["horizon"] > 1:
                active = index_map[ids] >= 0
                if active.any():
                    ps = index_map[ids][active]
                    pparams = network(pseq[ps], pz[ps])
                    pd = {k: previous[k][ps] for k in ("level", "slope", "curve")}
                    plogs = curve_log(pparams, pd, cfg)
                    # Current lead 1 and previous lead 2 share an absolute time.
                    revision = (logs[active, :-1] - plogs[:, 1:]).abs()
                    loss += cfg["temporal_penalty"] * (w[ids][active, :-1] * revision).sum()
            loss.backward()
            if not torch.isfinite(loss) or any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in network.parameters()
            ):
                raise FloatingPointError("Nonfinite stable training")
            nn.utils.clip_grad_norm_(network.parameters(), 1)
            optimizer.step()
        score = validate()
        record = dict(
            epoch=epoch, worst_quarter_containment=-score[0], validation_log_error=score[1]
        )
        records.append(record)
        report(record)
        if score < best:
            best, best_epoch, best_state = score, epoch, copy.deepcopy(network.state_dict())
        if epoch - best_epoch >= cfg["patience"]:
            break
    network.load_state_dict(best_state)
    network.eval()
    with torch.inference_mode():
        final_parameters = np.concatenate(
            [
                network(vsequence[start : start + 256], vz[start : start + 256]).numpy()
                for start in range(0, len(vz), 256)
            ]
        )
    final_center = np.exp(curve_log(final_parameters, vd, cfg))
    model["_validation_outputs"] = np.stack(
        (final_center * (1 - half), final_center, final_center * (1 + half)), -1
    )
    model["training"] = dict(
        records=records,
        best_epoch=best_epoch,
        stopped_epoch=epoch,
        baseline_selection=list(baseline_selection),
        selected_score=list(best),
        validation_objective=best[1],
        baseline_validation_objective=baseline_selection[1],
        learned_correction_selected=best_epoch > 0,
        fit_split="train",
        selection_split="validation",
        test_used=False,
        train_windows=len(z),
        validation_windows=len(vz),
        temporal_constraint=previous is not None,
    )
    if engine == "full_cns":
        model["training"]["reservoir"] = dict(
            neurons=len(model["reservoir"].node_order),
            edges=model["reservoir"].operator.nnz,
            sequence_bins=cfg["cns_sequence_bins"],
            provenance=model["reservoir"].provenance,
        )
    return model


def metrics(outputs, data, red=None):
    cfg = config(data["horizon"], data["cadence_s"])
    desc = design(data["x"], data["lengths"], cfg)
    w = weights(data, desc)
    center, lo, hi = outputs[..., 1], outputs[..., 0], outputs[..., 2]
    y, mask = data["y"], data["mask"]
    inside = (y >= lo) & (y <= hi)

    def aggregate(active):
        selected = w * active
        if not selected.sum():
            return dict(status="unknown", known_targets=0)
        selected /= selected.sum()
        return dict(
            status="measured",
            point_coverage=float((selected * inside).sum()),
            center_mae=float((selected * np.abs(center - y)).sum()),
            relative_bias=float((selected * (center - y) / np.maximum(y, 0.001)).sum()),
            known_targets=int(active.sum()),
            supported_units=len(set(data["physical"][active.any(1)])),
        )

    result = aggregate(mask)
    result["aggregation"] = (
        "equal physical equipment, observed regime and supported horizon quarter"
    )
    result["unweighted_point_coverage"] = float(inside[mask].mean())
    result["quarters"] = []
    for j, chunk in enumerate(np.array_split(np.arange(data["horizon"]), 4)):
        active = np.zeros_like(mask)
        active[:, chunk] = mask[:, chunk]
        result["quarters"].append(dict(quarter=j + 1, **aggregate(active)))
    rising = desc["slope"] > np.maximum(0.015, desc["features"][:, 1] * 0.5)
    result["observed_growing"] = aggregate(mask & rising[:, None])
    result["observed_stable_or_falling"] = aggregate(mask & ~rising[:, None])
    result["red"] = (
        aggregate(mask & (y >= red)) if red is not None else dict(status="unknown", known_targets=0)
    )
    complete = mask.all(1)
    result["complete_paths"] = int(complete.sum())
    result["whole_path_coverage"] = (
        float(inside[complete].all(1).mean()) if complete.any() else None
    )
    result["whole_supported_path_coverage"] = float((inside | ~mask).all(1).mean())
    result["paths_with_90_percent_points"] = float(
        (np.sum(inside & mask, 1) / mask.sum(1) >= 0.9).mean()
    )
    result["relative_full_width"] = float(np.median((outputs[..., 2] - outputs[..., 0]) / center))
    result["quality_accepted"] = False
    result["coverage_guarantee"] = False
    return result
