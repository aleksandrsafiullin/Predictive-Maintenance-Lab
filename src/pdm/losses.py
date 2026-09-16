from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS = 1e-8


def weibull_log_sf(r: torch.Tensor, lam: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r, lam, k = r.double(), lam.double(), k.double()
    return -torch.exp(k * (torch.log(r.clamp_min(1e-300)) - torch.log(lam.clamp_min(1e-300))))


def weibull_log_pdf(r: torch.Tensor, lam: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r, lam, k = r.double().clamp_min(1e-300), lam.double().clamp_min(1e-300), k.double().clamp_min(1e-300)
    return torch.log(k) - torch.log(lam) + (k - 1.0) * (torch.log(r) - torch.log(lam)) + weibull_log_sf(r, lam, k)


def weibull_nll(
    duration_s: torch.Tensor,
    event: torch.Tensor,
    lam: torch.Tensor,
    k: torch.Tensor,
    time_scale_s: float,
) -> torch.Tensor:
    """Negative log-likelihood for remaining life with right-censoring.

    r = duration_s / time_scale_s
    log_S(r) = -(r / lambda) ** k
    log_f(r) = log(k) - log(lambda) + (k - 1) * (log(r) - log(lambda)) - (r / lambda) ** k
    loss = -event * log_f(r) - (1 - event) * log_S(r)
    """
    scale = max(float(time_scale_s), EPS)
    r = duration_s.double() / scale
    log_f = weibull_log_pdf(r, lam, k)
    log_s = weibull_log_sf(r, lam, k)
    event = event.to(dtype=log_f.dtype)
    nll = torch.where(event > 0.5, -log_f, -log_s)
    return nll


def weibull_nll_seconds(duration_s, event, scale_s, shape):
    """Shared density convention: all public metrics use internal seconds."""
    return weibull_nll(duration_s, event, scale_s, shape, 1.0)


def legacy_weibull_nll(duration_s, event, lam, k, time_scale_s):
    """Original numerical recipe, exclusively for legacy runs and their resume."""
    r = duration_s / max(float(time_scale_s), EPS)
    ratio = (r / lam.clamp_min(EPS)).clamp_min(EPS)
    log_s = -torch.exp(k * torch.log(ratio)).clamp(max=1e6)
    r, lam, k = r.clamp_min(EPS), lam.clamp_min(EPS), k.clamp_min(EPS)
    power = -torch.exp((k * torch.log((r / lam).clamp_min(EPS))).clamp(max=20.0))
    log_f = torch.log(k) - torch.log(lam) + (k - 1.0) * (torch.log(r) - torch.log(lam)) + power
    event = event.to(dtype=log_f.dtype)
    return -event * log_f - (1.0 - event) * log_s


def weibull_median_rul(lam: torch.Tensor, k: torch.Tensor, time_scale_s: float) -> torch.Tensor:
    scale = max(float(time_scale_s), EPS)
    return scale * lam * (math.log(2.0) ** (1.0 / k.clamp_min(EPS)))


def smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="mean")
