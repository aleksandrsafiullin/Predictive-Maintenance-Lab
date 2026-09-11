from __future__ import annotations

import math

import torch
import torch.nn.functional as F

EPS = 1e-8


def weibull_log_sf(r: torch.Tensor, lam: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    ratio = (r / lam.clamp_min(EPS)).clamp_min(EPS)
    power = torch.exp(k * torch.log(ratio)).clamp(max=1e6)
    return -power


def weibull_log_pdf(r: torch.Tensor, lam: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    r = r.clamp_min(EPS)
    lam = lam.clamp_min(EPS)
    k = k.clamp_min(EPS)
    log_S_term = -torch.exp((k * torch.log((r / lam).clamp_min(EPS))).clamp(max=20.0))
    return torch.log(k) - torch.log(lam) + (k - 1.0) * (torch.log(r) - torch.log(lam)) + log_S_term


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
    r = duration_s / scale
    log_f = weibull_log_pdf(r, lam, k)
    log_s = weibull_log_sf(r, lam, k)
    event = event.to(dtype=log_f.dtype)
    nll = -event * log_f - (1.0 - event) * log_s
    return nll


def weibull_median_rul(lam: torch.Tensor, k: torch.Tensor, time_scale_s: float) -> torch.Tensor:
    scale = max(float(time_scale_s), EPS)
    return scale * lam * (math.log(2.0) ** (1.0 / k.clamp_min(EPS)))


def smooth_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred, target, reduction="mean")
