"""Joint future-path losses; event buckets refer to recorded grid points, not continuous time."""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch


def validate_path_band_geometry(path_band_geometry):
    """Validate the explicit training-band geometry selector without coercion."""
    if not isinstance(path_band_geometry, str) or path_band_geometry not in (
        "observed_prefix", "issued_prefix",
    ):
        raise ValueError("path_band_geometry must be 'observed_prefix' or 'issued_prefix'")
    return path_band_geometry


def simultaneous_band(paths: torch.Tensor, coverage: float = 0.9, target_mask=None):
    """Empirical simultaneous band for paths [samples,batch,horizon].

    A single quantile of the maximum standardized excursion sets every horizon's
    width. Pointwise quantiles alone would not provide simultaneous coverage.
    """
    if not 0 < coverage < 1 or paths.ndim != 3 or paths.shape[0] < 2:
        raise ValueError("Expected >=2 joint samples and coverage in (0,1)")
    center = paths.median(dim=0).values
    scale = paths.std(dim=0, unbiased=False).clamp_min(1e-5)
    excursion_values = (paths - center).abs() / scale
    if target_mask is not None:
        excursion_values = excursion_values * target_mask.bool()[None, ...]
    excursion = excursion_values.amax(dim=-1)
    radius = torch.quantile(excursion, coverage, dim=0)
    return (center - radius[:, None] * scale).clamp_min(0), center + radius[:, None] * scale


def path_losses(paths, actual_future, target_mask, *, coverage=0.9, signal_scale=1.0, energy_only=False,
                path_band_geometry="observed_prefix"):
    """Return per-row energy, simultaneous width and worst-path miss in RED-normalized physical units.

    Target masking precedes target arithmetic, including NaN handling. Completely unknown
    rows contribute zero. Integration must balance physical units, rather than
    weighting units with more prefixes more heavily.

    The legacy observed_prefix band uses observation-masked paths and radius.
    issued_prefix constructs geometry from the original full requested prefix;
    observation masks then select width/miss scoring coordinates only. Energy
    distances retain the same observation mask in both modes.
    """
    validate_path_band_geometry(path_band_geometry)
    issued_paths = None
    if path_band_geometry == "issued_prefix":
        if not torch.isfinite(paths).all() or (paths < 0).any():
            raise ValueError("Issued-prefix generated paths must be finite and nonnegative")
        issued_paths = paths
    mask = target_mask.bool()
    valid = mask.any(dim=-1)
    target = torch.where(mask, actual_future, torch.zeros_like(actual_future))
    if (target < 0).any() or not torch.isfinite(target).all():
        raise ValueError("Observed future signal must be finite and nonnegative")
    scale = torch.as_tensor(signal_scale, dtype=paths.dtype, device=paths.device).reshape(-1, 1)
    if not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("Frozen signal scale must be positive and finite")
    paths = torch.where(mask[None], paths, torch.zeros_like(paths))
    lp = paths.clamp_min(0) / scale
    lt = target / scale
    count = mask.sum(-1).clamp_min(1)
    # Energy score uses joint distances, preserving dependency across horizons.
    distance = (((lp - lt) ** 2 * mask).sum(-1) / count).clamp_min(1e-12).sqrt()
    # Independent halves avoid O(S^2) memory while retaining an unbiased pair term.
    half = paths.shape[0] // 2
    if half < 1:
        raise ValueError("At least two samples are required")
    pair = (((lp[:half] - lp[half : 2 * half]) ** 2 * mask).sum(-1) / count).clamp_min(1e-12).sqrt()
    energy = distance.mean(0) - 0.5 * pair.mean(0)
    if energy_only:
        return {"energy": energy * valid}
    if issued_paths is None:
        lower, upper = simultaneous_band(paths, coverage, target_mask=mask)
    else:
        lower, upper = simultaneous_band(issued_paths, coverage)
    lo, hi = lower / scale, upper / scale
    width = ((hi - lo) * mask).sum(-1) / count
    miss = (torch.relu(lo - lt) * mask).amax(-1) + (torch.relu(lt - hi) * mask).amax(-1)
    return {"energy": energy * valid, "width": width * valid, "miss": miss * valid}


def path_score_surrogates(paths, actual_future, target_mask, sampled_log_prob, *, coverage=0.9, signal_scale=1.0, energy_only=False,
                          path_band_geometry="observed_prefix"):
    """Unbiased, within-row categorical gradients for joint path objectives.

    Energy is a mean of sample distances and independent paired distances, so
    each distance uses only its own categorical likelihood score. For a band
    statistic every sample matters: four disjoint groups use the band's cost
    with that whole group removed as an independent control variate. Baselines
    are detached and never borrow another equipment row's target or cost.
    """
    validate_path_band_geometry(path_band_geometry)
    if path_band_geometry == "issued_prefix" and (
        not torch.isfinite(paths).all() or (paths < 0).any()
    ):
        raise ValueError("Issued-prefix generated paths must be finite and nonnegative")
    mask = target_mask.bool()
    valid = mask.any(-1)
    count = mask.sum(-1).clamp_min(1)
    scale = torch.as_tensor(signal_scale, dtype=paths.dtype, device=paths.device).reshape(-1, 1)
    with torch.no_grad():
        values = torch.where(mask[None], paths.detach(), torch.zeros_like(paths)) / scale
        target = torch.where(mask, actual_future, torch.zeros_like(actual_future)) / scale
        distance = (((values - target) ** 2).sum(-1) / count).clamp_min(1e-12).sqrt()
        sample_count = len(paths)
        baseline = (distance.sum(0, keepdim=True) - distance) / (sample_count - 1)
        half = sample_count // 2
        pair = (((values[:half] - values[half:2 * half]) ** 2).sum(-1) / count).clamp_min(1e-12).sqrt()
        pair_baseline = (pair.sum(0, keepdim=True) - pair) / (half - 1) if half > 1 else torch.zeros_like(pair)
    energy = ((distance - baseline) * sampled_log_prob).mean(0) - 0.5 * (
        (pair - pair_baseline) * (sampled_log_prob[:half] + sampled_log_prob[half:2 * half])
    ).mean(0)
    if energy_only:
        return {"energy": energy * valid}
    with torch.no_grad():
        full = path_losses(paths.detach(), actual_future, mask, coverage=coverage, signal_scale=signal_scale,
                           path_band_geometry=path_band_geometry)
    result = {"energy": energy * valid, "width": torch.zeros_like(energy), "miss": torch.zeros_like(energy)}
    if sample_count < 4:
        for name in ("width", "miss"):
            result[name] = full[name] * sampled_log_prob.sum(0) * valid
        return result
    groups = torch.tensor_split(torch.arange(sample_count, device=paths.device), min(4, sample_count // 2))
    for group in groups:
        keep = torch.ones(sample_count, dtype=torch.bool, device=paths.device)
        keep[group] = False
        with torch.no_grad():
            reduced = path_losses(paths.detach()[keep], actual_future, mask, coverage=coverage, signal_scale=signal_scale,
                                  path_band_geometry=path_band_geometry)
        score = sampled_log_prob[group].sum(0)
        for name in ("width", "miss"):
            result[name] = result[name] + (full[name] - reduced[name]) * score * valid
    return result


def first_entry_loss(
    event_logits, *, event_allowed_mask=None, event_observed_mask=None, no_entry_prefix=None,
    prefix_length=None,
):
    """Exact mixture likelihood for entry brackets and continuously observed survival.

    H+1 classes: first entry at future grid index 0..H-1, or no entry through H.
    Allowed masks admit every bucket compatible with a recorded crossing bracket.
    no_entry_prefix is a count of CONTIGUOUS observed below-RED grid points; an
    incomplete suffix can censor that prefix. A gap must never be bridged into a
    negative label. Zero means unknown, H means known no-entry through the horizon.
    An optional frozen prefix coarsens late entry and full survival into survival
    through that prefix; it does not manufacture follow-up beyond censoring.
    """
    allowed, known = compatible_entry_mask(
        event_logits,
        event_allowed_mask=event_allowed_mask,
        event_observed_mask=event_observed_mask,
        no_entry_prefix=no_entry_prefix,
    )
    log_prob = torch.log_softmax(event_logits, dim=-1)
    if prefix_length is not None:
        horizon = event_logits.shape[-1] - 1
        if (isinstance(prefix_length, bool) or not isinstance(prefix_length, Integral)
                or not 0 < prefix_length <= horizon):
            raise ValueError("Event objective prefix must be an integer within the saved horizon")
        if prefix_length < horizon:
            # An entry later than this declared prefix and full-grid survival
            # both mean no entry THROUGH the prefix. Censoring before it still
            # admits all compatible early buckets as well as that survival.
            log_prob = torch.cat((log_prob[:, :prefix_length],
                                  torch.logsumexp(log_prob[:, prefix_length:], -1, keepdim=True)), -1)
            allowed = torch.cat((allowed[:, :prefix_length],
                                 allowed[:, prefix_length:].any(-1, keepdim=True)), -1)
    return -torch.logsumexp(log_prob.masked_fill(~allowed, -torch.inf), dim=-1) * known


def compatible_entry_mask(
    event_logits, *, event_allowed_mask=None, event_observed_mask=None, no_entry_prefix=None
):
    """Return classes compatible with observed event evidence, never bridging gaps."""
    batch, classes = event_logits.shape
    horizon = classes - 1
    allowed = torch.zeros_like(event_logits, dtype=torch.bool)
    known = torch.zeros(batch, dtype=torch.bool, device=event_logits.device)
    if event_allowed_mask is not None:
        if event_allowed_mask.shape != event_logits.shape or event_observed_mask is None:
            raise ValueError("Entry masks require [B,H+1] allowed and [B] observed masks")
        known = event_observed_mask.bool().clone()
        allowed = event_allowed_mask.bool() & known[:, None]
        if (known & ~allowed.any(-1)).any():
            raise ValueError("Observed events require at least one admitted bucket")
    if no_entry_prefix is not None:
        prefix = no_entry_prefix.to(device=event_logits.device)
        if ((prefix < 0) | (prefix > horizon) | (prefix != prefix.long())).any():
            raise ValueError("Survival prefix must be an integer in [0,H]")
        survival = (prefix > 0) & ~known
        allowed |= (
            torch.arange(classes, device=prefix.device)[None, :] >= prefix[:, None]
        ) & survival[:, None]
        known = known | survival
    # Unknown rows admit all buckets to avoid logsumexp(-inf) and NaN gradients.
    allowed = allowed | ~known[:, None]
    return allowed, known


def observed_phase_masks(allowed, known, target_mask, already_red):
    """Independent observed targets for phases whose identity is provable.

    A bracket has unknown phase inside its admitted range. A censored survival
    prefix proves only pre-entry points. An observed suffix after a known entry
    can supervise post-entry even when intervening target measurements are gaps.
    """
    horizon = allowed.shape[-1] - 1
    grid = torch.arange(horizon, device=allowed.device)[None, :]
    classes = torch.arange(horizon + 1, device=allowed.device)[None, :]
    earliest = classes.expand_as(allowed).masked_fill(~allowed, horizon + 1).amin(-1)
    latest = classes.expand_as(allowed).masked_fill(~allowed, -1).amax(-1)
    active = known & ~already_red
    observed = target_mask.bool()
    return {
        "pre": observed & active[:, None] & (grid < earliest[:, None]),
        "entry": observed & active[:, None] & (earliest == latest)[:, None] & (grid == earliest[:, None]),
        "post": observed & ((active[:, None] & (grid > latest[:, None])) | already_red[:, None]),
    }


def validate_mc_red_decision_inputs(event_logits, *, prefix_length, already_red,
                                    event_allowed_mask=None, event_observed_mask=None,
                                    no_entry_prefix=None, coverage=.9):
    """Validate marginal logits and fixed evidence without sampling or RNG."""
    if (not isinstance(event_logits, torch.Tensor) or event_logits.ndim != 2
        or event_logits.dtype not in (torch.float32, torch.float64) or event_logits.shape[0] < 1
        or event_logits.shape[1] < 2 or not torch.isfinite(event_logits).all()):
        raise ValueError("event_logits must be finite float32/float64 [B,H+1], B>=1,H>=1")
    batch, classes = event_logits.shape
    horizon = classes - 1
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, Integral) or not 1 <= prefix_length <= horizon:
        raise ValueError("prefix_length must be integer in [1,H]")
    if isinstance(coverage, bool) or not isinstance(coverage, Real) or not math.isfinite(coverage) or not 0 < coverage < 1:
        raise ValueError("coverage must be finite real in (0,1)")
    red = _red_decision_bool(already_red, (batch,), "already_red", event_logits.device)
    if (event_allowed_mask is None) != (event_observed_mask is None):
        raise ValueError("allowed and observed masks must be supplied together")
    allowed_in = observed_in = None
    if event_allowed_mask is not None:
        allowed_in = _red_decision_bool(event_allowed_mask, (batch, classes), "event_allowed_mask", event_logits.device)
        observed_in = _red_decision_bool(event_observed_mask, (batch,), "event_observed_mask", event_logits.device)
    prefix = None
    if no_entry_prefix is not None:
        if (not isinstance(no_entry_prefix, torch.Tensor) or tuple(no_entry_prefix.shape) != (batch,)
            or no_entry_prefix.dtype == torch.bool or no_entry_prefix.is_complex()
            or not torch.isfinite(no_entry_prefix).all()
            or not ((no_entry_prefix >= 0) & (no_entry_prefix <= horizon)
                    & (no_entry_prefix == no_entry_prefix.long())).all()):
            raise ValueError("no_entry_prefix must be finite integer-valued [B] in [0,H]")
        prefix = no_entry_prefix.to(device=event_logits.device, dtype=torch.long)
    compatible_entry_mask(event_logits, event_allowed_mask=allowed_in,
                          event_observed_mask=observed_in, no_entry_prefix=prefix)
    return red, allowed_in, observed_in, prefix


def _red_decision_bool(value, shape, name, device):
    if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or value.dtype != torch.bool:
        raise ValueError(f"{name} must be boolean tensor with shape {shape}")
    return value.to(device=device)


def _red_decision_empirical_cost(entries, allowed, known, red, length, coverage):
    """Discrete costs depend on supplied entries only; no pathwise derivatives."""
    edges = entries.clamp(min=0, max=length).to(torch.float64)
    left = torch.where(edges == length, length + 1, edges)
    right = edges + 1  # atom right also L+1
    alpha = (1 - coverage) / 2
    lower = torch.quantile(left, alpha, dim=0, interpolation="lower")
    upper = torch.quantile(right, 1 - alpha, dim=0, interpolation="higher")
    a = torch.arange(length + 1, device=entries.device, dtype=torch.float64)
    b = a + 1
    a[-1] = length + 1
    each = torch.relu(lower[:, None] - a) + torch.relu(b - upper[:, None])
    miss = each.masked_fill(~allowed, torch.inf).amin(-1)
    width = torch.where(~red, torch.relu(upper - lower), 0)
    miss = torch.where(known & ~red, miss, 0)
    return dict(width=width, compatible_miss=miss,
                empirical_lower=torch.where(red, 0, lower),
                empirical_upper=torch.where(red, 0, upper))


def mc_red_decision_regularizer(event_logits, entries, *, prefix_length, already_red,
                                event_allowed_mask=None, event_observed_mask=None,
                                no_entry_prefix=None, coverage=.9):
    """Per-row finite-S costs with zero-forward-value categorical score gradients.

    Samples are assumed IID from the current full event marginal, independently
    across draws conditional on each row. This assumption is a caller contract,
    not verifiable from entries. No K-component score is required for E-only costs.
    Leave-group baselines are independent of their group's draws and use the same
    row's evidence, prefix, geometry, and coverage. S=2/3 uses no baseline.

    Geometry and marginal log-softmax deliberately use float64, matching the
    reviewed prototype. Exact finite-S unbiasedness assumes one common law;
    dtype rounding in the existing float32 or coupled sampler can make its
    numerical law differ slightly. This is a decision prior, not a calibrated
    population score or the runtime corridor decoder. L+1 is score-only.
    """
    red, allowed_in, observed_in, prefix = validate_mc_red_decision_inputs(
        event_logits, prefix_length=prefix_length, already_red=already_red,
        event_allowed_mask=event_allowed_mask, event_observed_mask=event_observed_mask,
        no_entry_prefix=no_entry_prefix, coverage=coverage,
    )
    batch, classes = event_logits.shape
    horizon = classes - 1
    if (not isinstance(entries, torch.Tensor) or entries.ndim != 2 or entries.shape[1] != batch
        or entries.shape[0] < 2 or entries.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)):
        raise ValueError("entries must be integer [S,B], S>=2")
    entries = entries.to(device=event_logits.device, dtype=torch.long)
    if ((entries < -1) | (entries > horizon) | ((entries == -1) & ~red[None, :])).any():
        raise ValueError("entries must be classes 0..H; -1 allowed only for already RED")
    allowed, known = compatible_entry_mask(event_logits, event_allowed_mask=allowed_in,
                                           event_observed_mask=observed_in, no_entry_prefix=prefix)
    if prefix_length < horizon:
        allowed = torch.cat((allowed[:, :prefix_length], allowed[:, prefix_length:].any(-1, keepdim=True)), -1)
    # Full marginal score, even when geometry and evidence collapse the tail.
    log_q = event_logits.to(torch.float64).log_softmax(-1)
    scores = log_q.gather(1, entries.clamp_min(0).T).T
    scores = torch.where(red[None, :], 0, scores)
    if not torch.isfinite(scores).all():
        raise ValueError("sampled full-event log scores must be finite")
    full = _red_decision_empirical_cost(entries, allowed, known, red, prefix_length, coverage)
    surrogates = {key: torch.zeros_like(full[key]) for key in ("width", "compatible_miss")}
    sample_count = entries.shape[0]
    groups = []
    if sample_count < 4:
        for key in surrogates:
            surrogates[key] = full[key].detach() * scores.sum(0)
    else:
        groups = torch.tensor_split(torch.arange(sample_count, device=entries.device), min(4, sample_count // 2))
        for group in groups:
            keep = torch.ones(sample_count, device=entries.device, dtype=torch.bool)
            keep[group] = False
            reduced = _red_decision_empirical_cost(entries[keep], allowed, known, red, prefix_length, coverage)
            for key in surrogates:
                surrogates[key] = surrogates[key] + (full[key] - reduced[key]).detach() * scores[group].sum(0)
    if any(not torch.isfinite(value).all() for value in surrogates.values()):
        raise ValueError("score surrogate arithmetic must remain finite")
    result = dict(full)
    for key, surrogate in surrogates.items():
        result[key + "_term"] = full[key] + (surrogate - surrogate.detach())
    q = log_q.exp()
    if prefix_length < horizon:
        q = torch.cat((q[:, :prefix_length], q[:, prefix_length:].sum(-1, keepdim=True)), -1)
    result.update(known=known, applicable=~red, compatible_mask=allowed, coarsened_probability=q,
                  sampled_full_event_log_score=scores,
                  audit=dict(sample_count=sample_count, groups=[g.tolist() for g in groups],
                             coordinate="saved_grid_steps", tau_score_only=prefix_length + 1,
                             exact_runtime_decoder=False, finite_S_expected_cost_gradient=True,
                             proper_calibrated_or_unbiased_population_risk=False,
                             width_prior_on_unknown=True, rng_consumed=False))
    return result


def _finite_bound_boolean(value, shape, name, device):
    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.bool
        or tuple(value.shape) != shape
        or value.device != device
    ):
        raise ValueError(f"{name} must be Boolean tensor {shape} on logits device")
    return value


def finite_bound_deficiency(
    event_logits,
    *,
    prefix_length,
    already_red,
    event_allowed_mask=None,
    event_observed_mask=None,
    no_entry_prefix=None,
    coverage=0.9,
):
    """Positive-only finite-upper availability prior, not a proper score.

    D=relu(logtail-logfinite-logit(alpha/2)), alpha=(1-coverage)/2>2e-8.
    The alpha/2 interior margin differs from runtime's alpha-1e-8 threshold.
    Exact zero implies finite availability in real arithmetic; machine decoding
    requires separate checks. No calibration, width, timing or lead guarantee.
    Observed evidence has priority over censoring; unknown admits all classes.
    Valid uninterrupted recorded-grid evidence is an upstream contract that
    this helper cannot infer from masks. Prefix evidence tail is OR, probability
    tail is SUM over every original class >=L. All tail-compatible and already
    RED rows are excluded with exact zero cost and logit derivatives.
    Finite float32/64 CPU/CUDA inputs use float64 arithmetic; CUDA is unverified
    and unsupported devices reject explicitly. Detached shared and per-group
    maxima preserve normalized gradients for promoted repeated float32 extrema.
    Actual derived nonfinite values reject. No clamps, floors, samples or RNG.
    """
    z = event_logits
    if isinstance(z, torch.Tensor) and z.device.type not in ("cpu", "cuda"):
        raise ValueError("event_logits requires CPU/CUDA float64 support")
    if (
        not isinstance(z, torch.Tensor)
        or z.ndim != 2
        or z.dtype not in (torch.float32, torch.float64)
        or z.shape[0] < 1
        or z.shape[1] < 2
        or not bool(torch.isfinite(z).all())
    ):
        raise ValueError("event_logits must be finite float32/64 [B,H+1], B>=1,H>=1")
    B, C = z.shape
    H = C - 1
    if (
        isinstance(prefix_length, bool)
        or not isinstance(prefix_length, Integral)
        or not 1 <= prefix_length <= H
    ):
        raise ValueError("prefix_length must be integer in [1,H]")
    if (
        isinstance(coverage, bool)
        or not isinstance(coverage, Real)
        or not math.isfinite(coverage)
        or not 0 < coverage < 1
        or (1 - coverage) / 2 <= 2e-8
    ):
        raise ValueError("coverage requires finite (0,1) and alpha>2e-8")
    L = int(prefix_length)
    red = _finite_bound_boolean(already_red, (B,), "already_red", z.device)
    if (event_allowed_mask is None) != (event_observed_mask is None):
        raise ValueError("allowed and observed masks must be supplied together")
    allowed = torch.zeros((B, C), dtype=torch.bool, device=z.device)
    known = torch.zeros(B, dtype=torch.bool, device=z.device)
    if event_allowed_mask is not None:
        given = _finite_bound_boolean(event_allowed_mask, (B, C), "event_allowed_mask", z.device)
        known = _finite_bound_boolean(
            event_observed_mask, (B,), "event_observed_mask", z.device
        ).clone()
        allowed = given & known[:, None]
        if bool((known & ~allowed.any(-1)).any()):
            raise ValueError("known observed evidence must admit a class")
    if no_entry_prefix is not None:
        c = no_entry_prefix
        if (
            not isinstance(c, torch.Tensor)
            or tuple(c.shape) != (B,)
            or c.device != z.device
            or c.dtype == torch.bool
            or c.is_complex()
            or not bool(torch.isfinite(c).all())
            or bool(((c < 0) | (c > H) | (c != c.long())).any())
        ):
            raise ValueError("no_entry_prefix must be finite integer [B] in [0,H] on logits device")
        survival = (c > 0) & ~known
        allowed = allowed | (
            (torch.arange(C, device=z.device)[None, :] >= c[:, None]) & survival[:, None]
        )
        known = known | survival
    allowed = allowed | ~known[:, None]
    coarse = torch.cat((allowed[:, :L], allowed[:, L:].any(-1, keepdim=True)), -1)
    eligible = known & ~red & ~coarse[:, -1]
    x = z.double()
    centered = x - x.detach().amax(-1, keepdim=True)
    if not bool(torch.isfinite(centered).all()):
        raise ValueError("shared centering overflow")

    def group_logsumexp(group):
        group_max = group.detach().amax(-1, keepdim=True)
        residual = group - group_max
        if not bool(torch.isfinite(residual).all()):
            raise ValueError("group centering overflow")
        return group_max.squeeze(-1) + torch.logsumexp(residual, -1)

    lf = group_logsumexp(centered[:, :L])
    lt = group_logsumexp(centered[:, L:])
    odds = lt - lf
    alpha = (1 - float(coverage)) / 2
    margin = alpha / 2
    k = math.log(margin) - math.log1p(-margin)
    raw = torch.relu(odds - k)
    cost = torch.where(eligible, raw, torch.zeros_like(raw))
    if not all(bool(torch.isfinite(v).all()) for v in (lf, lt, odds, raw, cost)):
        raise ValueError("derived reduction/logodds/cost overflow")
    return {
        "cost": cost,
        "eligible": eligible,
        "known": known,
        "compatible_full": allowed,
        "compatible_prefix": coarse,
        "logfinite": lf,
        "logtail": lt,
        "logodds": odds,
        "alpha": alpha,
        "confidence_margin": margin,
        "logit_margin": k,
        "internal_dtype": "float64",
        "evidence_contract": "valid continuous admitted recorded-grid prefix upstream",
    }


def known_cdf_timing_loss(event_logits, *, prefix_length, already_red, event_allowed_mask=None,
              event_observed_mask=None, no_entry_prefix=None, scale_steps=30.0):
    """Known-CDF timing mismatch on the full unconditional event marginal.

    Compatible recorded-grid evidence alone defines known CDF indicators. The
    sum of known squared errors uses one fixed scale, without outcome or known
    count normalization. Unknown/already-RED rows have exact zero cost and
    logit derivatives. Partial-evidence masking is not automatically a proper
    or calibrated population score. Continuous censor evidence is an upstream
    contract. CPU float32/64 inputs are promoted before detached max centering;
    CUDA float64 is allowed but unverified. No sampling, floors or clamps.
    """
    z = event_logits
    if isinstance(z, torch.Tensor) and z.device.type not in ('cpu', 'cuda'):
        raise ValueError('requires CPU/CUDA float64 support; other devices unsupported')
    if not isinstance(z, torch.Tensor) or z.dtype not in (torch.float32, torch.float64) or z.ndim != 2 or min(z.shape) < 1 or z.shape[1] < 2 or not bool(torch.isfinite(z).all()):
        raise ValueError('finite float32/64 [B,H+1], B>=1,H>=1 required')
    B, C = z.shape
    H = C - 1
    if isinstance(prefix_length, bool) or not isinstance(prefix_length, Integral) or not 1 <= prefix_length <= H:
        raise ValueError('prefix_length integer in [1,H] required')
    if isinstance(scale_steps, bool) or not isinstance(scale_steps, Real) or not math.isfinite(scale_steps) or scale_steps <= 0:
        raise ValueError('fixed finite positive scale_steps required')
    red = _finite_bound_boolean(already_red, (B,), 'already_red', z.device)
    if (event_allowed_mask is None) != (event_observed_mask is None):
        raise ValueError('allowed/observed must be supplied together')
    A = torch.zeros_like(z, dtype=torch.bool)
    known = torch.zeros(B, device=z.device, dtype=torch.bool)
    if event_allowed_mask is not None:
        given = _finite_bound_boolean(event_allowed_mask, (B,C), 'event_allowed_mask', z.device)
        known = _finite_bound_boolean(event_observed_mask, (B,), 'event_observed_mask', z.device).clone()
        A = given & known[:,None]
        if bool((known & ~A.any(-1)).any()):
            raise ValueError('known observed evidence must admit a class')
    if no_entry_prefix is not None:
        c = no_entry_prefix
        if not isinstance(c, torch.Tensor) or tuple(c.shape) != (B,) or c.device != z.device or c.dtype == torch.bool or c.is_complex() or not bool(torch.isfinite(c).all()) or bool(((c<0)|(c>H)|(c!=c.long())).any()):
            raise ValueError('no_entry_prefix finite integer [B] in [0,H] required')
        censor = (c>0) & ~known
        A = A | ((torch.arange(C,device=z.device)[None,:]>=c[:,None]) & censor[:,None])
        known = known | censor
    A = A | ~known[:,None]
    # Membership quantifiers work even for disjoint compatible sets (holes).
    left = A.to(torch.int64).cumsum(-1)[:,:prefix_length] > 0
    right = A.sum(-1,keepdim=True) > A.to(torch.int64).cumsum(-1)[:,:prefix_length]
    mask = (~left | ~right) & known[:,None] & ~red[:,None]
    labels = (~right & mask).double()  # inactive target sanitized before subtraction
    x = z.double()
    centered = x - x.detach().amax(-1,keepdim=True)
    if not bool(torch.isfinite(centered).all()):
        raise ValueError('shared detached centering overflow')
    q = torch.softmax(centered, -1)
    F = q.cumsum(-1)[:,:prefix_length]
    safe_F = torch.where(mask, F, torch.zeros_like(F))
    cost = (safe_F-labels).square().sum(-1)/float(scale_steps)
    if not bool(torch.isfinite(cost).all()):
        raise ValueError('derived cost overflow')
    return dict(cost=cost, q=q, F=F, labels=labels, mask=mask, A=A, known=known)

