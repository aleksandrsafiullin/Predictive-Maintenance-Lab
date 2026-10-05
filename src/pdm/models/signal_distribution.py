"""Causal joint-path mixture with an exact first recorded-grid RED-entry latent.

Conditional Gaussian latent paths have diagonal plus low-rank covariance. Smooth
side constraints transform them into nonnegative signal paths whose first RED
entry exactly matches their sampled categorical bucket. Recrossing after entry
is unrestricted. This predicts recorded-grid entry, not an unobserved crossing
between measurements. Current signal is a causal residual anchor for means;
the caller prepends the exact current point for display.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch import nn
from torch.nn import functional as F

from pdm.trajectory_objectives import (
    compatible_entry_mask,
    finite_bound_deficiency,
    first_entry_loss,
    known_cdf_timing_loss,
    mc_red_decision_regularizer,
    observed_phase_masks,
    path_losses,
    path_score_surrogates,
    simultaneous_band,
    validate_mc_red_decision_inputs,
    validate_path_band_geometry,
)


class SignalDistribution(nn.Module):
    def __init__(
        self,
        architecture: str,
        input_size: int,
        n_horizons: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        rank: int = 6,
        external_feature_size: int | None = None,
        dropout: float = 0.1,
        event_location: str = "softplus",
        phase_covariance: str = "shared",
        event_distribution: str = "lognormal_cure",
        path_distribution: str = "single",
        recurrent_context_mode: str = "fixed",
        min_history_length: int | None = None,
        max_history_length: int | None = None,
    ):
        super().__init__()
        if n_horizons < 1 or rank < 1:
            raise ValueError("Positive horizon and rank required")
        self.n_horizons, self.rank = n_horizons, rank
        self.external_feature_size = external_feature_size
        if not isinstance(recurrent_context_mode, str) or recurrent_context_mode not in {"fixed", "variable_causal"}:
            raise ValueError("Unsupported recurrent context mode")
        self.recurrent_context_mode = recurrent_context_mode
        self.min_history_length = min_history_length
        self.max_history_length = max_history_length
        if recurrent_context_mode == "variable_causal":
            if (external_feature_size is not None or not isinstance(architecture, str)
                    or architecture.lower() not in {"gru", "lstm"}):
                raise ValueError("Variable causal context requires a recurrent GRU/LSTM encoder")
            if (any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1
                    for n in (min_history_length, max_history_length))
                    or max_history_length < min_history_length):
                raise ValueError("Variable causal context requires declared positive min/max lengths")
        if event_location not in {"softplus", "log"}:
            raise ValueError("Unsupported event location parameterization")
        self.event_location = event_location
        if not isinstance(event_distribution, str) or event_distribution not in {
            "lognormal_cure", "finite_horizon_mixture"
        }:
            raise ValueError("Unsupported event distribution")
        self.event_distribution = event_distribution
        if not isinstance(path_distribution, str) or path_distribution not in {"single", "coupled_timing_mixture"}:
            raise ValueError("Unsupported path distribution")
        if path_distribution == "coupled_timing_mixture" and event_distribution != "finite_horizon_mixture":
            raise ValueError("Coupled paths require finite_horizon_mixture timing")
        self.path_distribution = path_distribution
        self.path_components = 3 if path_distribution == "coupled_timing_mixture" else 1
        self.effective_event_location = (
            "bounded_horizon_logistic" if event_distribution == "finite_horizon_mixture" else event_location
        )
        if not isinstance(phase_covariance, str) or phase_covariance not in {"shared", "separate"}:
            raise ValueError("Unsupported phase covariance parameterization")
        self.phase_covariance = phase_covariance
        self.path_parameters_per_horizon = 4 + rank if phase_covariance == "shared" else 6 + 3 * rank
        if external_feature_size is not None:
            self.encoder = nn.Sequential(
                nn.Linear(external_feature_size, hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
            )
        else:
            kind = {"gru": nn.GRU, "lstm": nn.LSTM}.get(architecture.lower())
            if kind is None:
                raise ValueError("Architecture must be GRU or LSTM unless using external features")
            self.encoder = kind(
                input_size,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )
        # Ten outputs in either family: three mixture weights, locations,
        # scales, and a survival logit. Legacy locations are independent of grid
        # length; the optional finite family conditions each component on H.
        # Locations are medians measured in grid STEPS, not operating age.
        self.event_head = nn.Linear(hidden_size, 10)
        nn.init.normal_(self.event_head.weight, std=0.01)
        nn.init.zeros_(self.event_head.bias)
        with torch.no_grad():
            # Broad generic initial remaining times for a minute grid. These
            # defaults use no validation/test lifetime or future measurements.
            initial_time = torch.tensor([30.0, 60.0, 120.0])
            if event_distribution == "finite_horizon_mixture":
                safe_time = 0.05 + (n_horizons - 0.05) * torch.tensor([0.25, 0.5, 0.75])
                initial_time = torch.minimum(initial_time, safe_time)
                self.event_head.bias[3:6] = torch.logit((initial_time - 0.05) / (n_horizons - 0.05))
            else:
                self.event_head.bias[3:6] = initial_time.log() if event_location == "log" else initial_time
            scales = torch.tensor([0.5, 0.65, 0.9]) - 0.08
            self.event_head.bias[6:9] = torch.log(torch.expm1(scales))
            self.event_head.bias[9] = torch.log(torch.tensor(0.2 / 0.8))
        self.path_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.path_components * n_horizons * self.path_parameters_per_horizon),
        )
        # Start near causal persistence with modest uncertainty; training remains
        # free to predict arbitrarily large physical spikes through softplus.
        nn.init.normal_(self.path_head[-1].weight, std=0.01)
        nn.init.zeros_(self.path_head[-1].bias)
        with torch.no_grad():
            bias = self.path_head[-1].bias.reshape(self.path_components * n_horizons, self.path_parameters_per_horizon)
            if phase_covariance == "shared":
                bias[:, 3] = -2.25
            else:
                bias[:, 3:6] = -2.25

    def entry_logits(self, encoded):
        """Exact dense bucket masses F(j+1)-F(j), plus horizon survival.

        The continuous latent remaining time is rounded up to the first future
        grid bucket. Tail subtraction uses survival probabilities where the CDF
        approaches one, avoiding cancellation. Float64 is used only for this
        small distribution calculation; paths and the encoder retain their dtype.
        """
        raw = self.event_head(encoded)
        if self.event_distribution == "finite_horizon_mixture" and not torch.isfinite(raw).all():
            raise ValueError("Finite-horizon event parameters must be finite")
        params = raw.double()
        weights = params[:, :3].softmax(-1)
        if self.event_distribution == "finite_horizon_mixture":
            log_medians = (0.05 + (self.n_horizons - 0.05) * params[:, 3:6].sigmoid()).log()
        else:
            log_medians = params[:, 3:6] if self.event_location == "log" else (F.softplus(params[:, 3:6]) + 0.05).log()
        scales = F.softplus(params[:, 6:9]) + 0.08
        cure = params[:, 9:].sigmoid()
        upper = torch.arange(1, self.n_horizons + 1, device=raw.device, dtype=torch.float64)
        z_upper = (upper.log()[None, :, None] - log_medians[:, None]) / scales[:, None]
        z_lower = torch.cat((torch.full_like(z_upper[:, :1], -torch.inf), z_upper[:, :-1]), 1)
        cdf_upper = 0.5 * torch.erfc(-z_upper / 2**0.5)
        cdf_lower = 0.5 * torch.erfc(-z_lower / 2**0.5)
        sf_lower = 0.5 * torch.erfc(z_lower / 2**0.5)
        sf_upper = 0.5 * torch.erfc(z_upper / 2**0.5)
        bucket = torch.where(z_lower > 0, sf_lower - sf_upper, cdf_upper - cdf_lower)
        if self.event_distribution == "finite_horizon_mixture":
            # Each component is conditioned separately on entry through H.
            # Bounded medians guarantee F_component(H) >= 0.5. The final
            # sigmoid is only no recorded entry through H, never permanent cure
            # or a prediction of an extrapolated time after the saved horizon.
            conditional_bucket = bucket / cdf_upper[:, -1:, :]
            finite_mass = (1 - cure) * (conditional_bucket * weights[:, None]).sum(-1)
            no_entry = cure
        else:
            finite_mass = (1 - cure) * (bucket * weights[:, None]).sum(-1)
            no_entry = cure + (1 - cure) * (sf_upper[:, -1] * weights).sum(-1, keepdim=True)
        mass = torch.cat((finite_mass, no_entry), -1)
        # Numerical underflow only: retain finite logits for every compatible
        # class, including survival/unknown evidence. This floor is negligible.
        mass = mass.clamp_min(1e-30)
        return (mass.log() - mass.sum(-1, keepdim=True).log()).to(raw.dtype)

    def joint_entry_logits(self, encoded, event_logits):
        """Actual issued E marginal times stable component responsibilities.

        Flooring belongs to the issued marginal only. Underflowed component
        bucket responsibilities use a float64 tiny floor, preserving a finite
        numerical law without independently flooring joint cells.
        """
        params = self.event_head(encoded).double()
        log_weights = params[:, :3].log_softmax(-1)
        medians = (0.05 + (self.n_horizons - 0.05) * params[:, 3:6].sigmoid()).log()
        scales = F.softplus(params[:, 6:9]) + 0.08
        upper = torch.arange(1, self.n_horizons + 1, device=params.device, dtype=torch.float64)
        z = (upper.log()[None, :, None] - medians[:, None]) / scales[:, None]
        lower = torch.cat((torch.full_like(z[:, :1], -torch.inf), z[:, :-1]), 1)
        cdf = 0.5 * torch.erfc(-z / 2**0.5)
        buckets = torch.where(lower > 0,
                              0.5 * (torch.erfc(lower / 2**0.5) - torch.erfc(z / 2**0.5)),
                              cdf - 0.5 * torch.erfc(-lower / 2**0.5))
        q = buckets / cdf[:, -1:, :]
        finite_responsibilities = (q.clamp_min(torch.finfo(torch.float64).tiny).log()
                                   + log_weights[:, None]).log_softmax(-1)
        responsibilities = torch.cat((finite_responsibilities.transpose(1, 2),
                                      log_weights[:, :, None]), -1)
        joint = event_logits.double().log_softmax(-1)[:, None] + responsibilities
        return joint.to(event_logits.dtype), log_weights.to(event_logits.dtype)

    def forward(self, x, current_signal, external_features=None, *, history_lengths=None, history_mask=None):
        if self.external_feature_size is not None:
            if external_features is None:
                raise ValueError("Causal external_features are required")
            encoded = self.encoder(external_features)
        else:
            if getattr(self, "recurrent_context_mode", "fixed") == "variable_causal":
                packed = self._pack_causal_history(x, history_lengths, history_mask)
                _, hidden = self.encoder(packed)
            else:
                _, hidden = self.encoder(x)
            encoded = (hidden[0] if isinstance(hidden, tuple) else hidden)[-1]
        current = current_signal.reshape(-1, 1)
        if not torch.isfinite(current).all() or (current < 0).any():
            raise ValueError("Current signal must be finite and nonnegative")
        raw = self.path_head(encoded)
        if getattr(self, "path_distribution", "single") == "coupled_timing_mixture":
            raw = raw.reshape(-1, 3, self.n_horizons, self.path_parameters_per_horizon)
        else:
            raw = raw.reshape(-1, self.n_horizons, self.path_parameters_per_horizon)
        moments = {
            "event_logits": self.entry_logits(encoded),
            "mean": raw[..., 0],
            "entry_mean": raw[..., 1],
            "post_mean": raw[..., 2],
            "current_signal": current[:, 0],
        }
        if self.phase_covariance == "shared":
            moments["diagonal_scale"] = F.softplus(raw[..., 3]) + 1e-3
            moments["factors"] = raw[..., 4:] / self.rank**0.5
        else:
            for index, phase in enumerate(("pre", "entry", "post")):
                moments[phase + "_diagonal_scale"] = F.softplus(raw[..., 3 + index]) + 1e-3
                start = 6 + index * self.rank
                moments[phase + "_factors"] = raw[..., start:start + self.rank] / self.rank**0.5
        if getattr(self, "path_distribution", "single") == "coupled_timing_mixture":
            moments["joint_event_logits"], moments["component_logits"] = self.joint_entry_logits(encoded, moments["event_logits"])
        return moments

    def _pack_causal_history(self, x, lengths, mask):
        """Validate real chronological prefixes; padding never updates recurrence."""
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[1] != self.max_history_length:
            raise ValueError("Variable input time dimension must match the saved maximum history")
        if (not isinstance(lengths, torch.Tensor) or lengths.ndim != 1
                or lengths.shape[0] != x.shape[0]
                or lengths.dtype not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}):
            raise ValueError("Variable causal context requires integer history_lengths [B]")
        if ((lengths < self.min_history_length) | (lengths > self.max_history_length)).any():
            raise ValueError("Real history lengths must lie within the saved min/max")
        if (not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool
                or mask.shape != x.shape[:2]):
            raise ValueError("Variable causal context requires boolean history_mask [B,max_history]")
        expected = torch.arange(x.shape[1], device=mask.device)[None] < lengths.to(mask.device)[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("History mask must match contiguous chronological real prefixes")
        real_mask = mask.to(x.device)
        if not torch.isfinite(x[real_mask]).all():
            raise ValueError("Real history features must be finite")
        # Mask before transport as well as packing: arbitrary NaN/Inf padding
        # cannot affect values, recurrent updates, or padding gradients.
        real_x = torch.where(real_mask[..., None], x, torch.zeros_like(x))
        return nn.utils.rnn.pack_padded_sequence(
            real_x, lengths.detach().to(device="cpu", dtype=torch.int64),
            batch_first=True, enforce_sorted=False,
        )

    def sample(self, moments, red_threshold, n_samples=64, generator=None, *, return_entry=False,
               return_latent=False, entry_allowed_mask=None, _entries=None):
        if getattr(self, "path_distribution", "single") == "coupled_timing_mixture" and _entries is None:
            return self._sample_coupled(moments, red_threshold, n_samples, generator,
                                        return_entry=return_entry, return_latent=return_latent,
                                        entry_allowed_mask=entry_allowed_mask)
        probs = moments["event_logits"].softmax(-1)
        entries = (torch.multinomial(probs, n_samples, replacement=True, generator=generator).T
                   if _entries is None else _entries)
        def expanded(name):
            return moments[name][None] if _entries is None else moments[name]

        mean = expanded("mean").expand(n_samples, -1, -1)
        if self.phase_covariance == "shared":
            scale, factors = expanded("diagonal_scale"), expanded("factors")
            eps = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        else:
            # Independent diagonal innovations per phase; one shared low-rank
            # latent below retains joint dependence across phases and leads.
            eps = torch.randn((*mean.shape, 3), dtype=mean.dtype, device=mean.device, generator=generator)
        low_rank = torch.randn(
            (*mean.shape[:2], self.rank), dtype=mean.dtype, device=mean.device, generator=generator
        )
        if self.phase_covariance == "shared":
            noise = scale * eps + (factors * low_rank[..., None, :]).sum(-1)
            pre_noise = entry_noise = post_noise = noise
        else:
            phase_noise = []
            for index, phase in enumerate(("pre", "entry", "post")):
                scale = expanded(phase + "_diagonal_scale")
                factors = expanded(phase + "_factors")
                phase_noise.append(scale * eps[..., index] + (factors * low_rank[..., None, :]).sum(-1))
            pre_noise, entry_noise, post_noise = phase_noise
        threshold = torch.as_tensor(red_threshold, dtype=mean.dtype, device=mean.device)
        if threshold.ndim:
            threshold = threshold.reshape(1, -1, 1)
        if not torch.isfinite(threshold).all() or (threshold <= 0).any():
            raise ValueError("RED threshold must be positive and finite")
        current = moments["current_signal"][None, :, None]
        time = torch.arange(self.n_horizons, device=mean.device)[None, None, :]
        before = time < entries[..., None]
        at = time == entries[..., None]
        # Covariance is indexed by absolute forecast lead in both modes.
        # Numerically open bounds preserve exact categorical first-entry support.
        tiny = torch.finfo(mean.dtype).eps
        fraction = (current / threshold).clamp(min=tiny, max=1 - tiny)
        below = threshold * torch.sigmoid(torch.logit(fraction) + mean + pre_noise).clamp(max=1 - tiny)

        def inverse_softplus(value):
            return value + torch.log(-torch.expm1(-value))

        entry_base = inverse_softplus(threshold * 0.01)
        entering = threshold + F.softplus(entry_base + expanded("entry_mean") + entry_noise)
        entering = entering.maximum(torch.nextafter(threshold, torch.full_like(threshold, torch.inf)))
        post_base = inverse_softplus(torch.maximum(current, threshold))
        # A post-entry growth shape is indexed by elapsed time SINCE entry,
        # so delaying entry shifts growth instead of starting on a mature curve.
        elapsed = (time - entries[..., None] - 1).clamp(0, self.n_horizons - 1)
        post_residual = expanded("post_mean").expand(n_samples, -1, -1).gather(-1, elapsed)
        # The growth mean shifts with entry; forecast uncertainty remains a
        # function of lead from the current origin, not elapsed time since entry.
        post = F.softplus(post_base + post_residual + post_noise)
        paths = torch.where(before, below, torch.where(at, entering, post))
        # Already-RED prefixes have unrestricted positive trajectories anchored
        # at their current value, with no applicable first-future-entry label.
        already_red = current >= threshold
        unrestricted = F.softplus(inverse_softplus(current.clamp_min(tiny)) + expanded("post_mean") + post_noise)
        paths = torch.where(already_red, unrestricted, paths)
        entries = torch.where(already_red[..., 0], -torch.ones_like(entries), entries)
        return (paths, entries) if return_entry else paths

    def coupled_latent_logits(self, moments, red_threshold, entry_allowed_mask=None):
        """Normalized [B,K,H+1] joint law; conditional normalizer stays live.

        Already-RED rows draw K from w, with an internal H placeholder for E.
        The returned path annotation is -1 and its score is log(w_K).
        """
        logits = moments["joint_event_logits"]
        if entry_allowed_mask is not None:
            logits = logits.masked_fill(~entry_allowed_mask[:, None], -torch.inf)
        threshold = torch.as_tensor(red_threshold, device=logits.device).reshape(-1)
        already_red = moments["current_signal"] >= threshold
        red_logits = torch.full_like(logits, -torch.inf)
        red_logits[..., -1] = moments["component_logits"]
        logits = torch.where(already_red[:, None, None], red_logits, logits)
        return logits - torch.logsumexp(logits.flatten(1), -1)[:, None, None]

    def _sample_coupled(self, moments, red_threshold, n_samples, generator, *,
                        return_entry=False, return_latent=False, entry_allowed_mask=None):
        logits = self.coupled_latent_logits(moments, red_threshold, entry_allowed_mask)
        flat = logits.flatten(1)
        cell = torch.multinomial(flat.softmax(-1), n_samples, replacement=True, generator=generator).T
        entries = cell % (self.n_horizons + 1)
        components = cell // (self.n_horizons + 1)
        batch = torch.arange(flat.shape[0], device=flat.device)[None]
        selected = {"event_logits": moments["event_logits"], "current_signal": moments["current_signal"]}
        for name in ("mean", "entry_mean", "post_mean", "diagonal_scale", "factors",
                     "pre_diagonal_scale", "entry_diagonal_scale", "post_diagonal_scale",
                     "pre_factors", "entry_factors", "post_factors"):
            if name in moments:
                selected[name] = moments[name][batch, components]
        paths, entries = self.sample(selected, red_threshold, n_samples, generator,
                                     return_entry=True, _entries=entries)
        score = flat[batch, cell]
        if return_latent:
            return paths, entries, components, score
        return (paths, entries) if return_entry else paths

    @staticmethod
    def simultaneous_band(paths, coverage=0.9):
        return simultaneous_band(paths, coverage)

    def objective(
        self,
        moments,
        actual_future,
        target_mask,
        red_threshold,
        *,
        event_allowed_mask=None,
        event_observed_mask=None,
        no_entry_prefix=None,
        n_samples=64,
        generator=None,
        coverage=0.9,
        width_weight=0.05,
        miss_weight=2.0,
        event_weight=1.0,
        energy_weight=1.0,
        phase_weight=0.5,
        path_objective_prefix_lengths=None,
        event_objective_prefix_lengths=None,
        post_factor_smoothness_weight=0.0,
        path_band_geometry="observed_prefix",
        red_corridor_width_weight=0.0,
        red_corridor_miss_weight=0.0,
        red_corridor_scale_steps=30.0,
        red_finite_bound_weight=0.0,
        event_cdf_weight=0.0,
        event_cdf_scale_steps=30.0,
    ):
        validate_path_band_geometry(path_band_geometry)
        if (isinstance(event_cdf_weight, bool)
                or not isinstance(event_cdf_weight, Real)
                or not math.isfinite(event_cdf_weight) or event_cdf_weight < 0):
            raise ValueError("event_cdf_weight must be finite, nonnegative and nonboolean")
        if (isinstance(event_cdf_scale_steps, bool)
                or not isinstance(event_cdf_scale_steps, Real)
                or not math.isfinite(event_cdf_scale_steps) or event_cdf_scale_steps <= 0):
            raise ValueError("event_cdf_scale_steps must be finite, positive and nonboolean")
        if (isinstance(red_finite_bound_weight, bool)
                or not isinstance(red_finite_bound_weight, Real)
                or not math.isfinite(red_finite_bound_weight) or red_finite_bound_weight < 0):
            raise ValueError("red_finite_bound_weight must be finite, nonnegative and nonboolean")
        for name, value in (("red_corridor_width_weight", red_corridor_width_weight),
                            ("red_corridor_miss_weight", red_corridor_miss_weight)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite, nonnegative and nonboolean")
        if (isinstance(red_corridor_scale_steps, bool)
                or not isinstance(red_corridor_scale_steps, Real)
                or not math.isfinite(red_corridor_scale_steps) or red_corridor_scale_steps <= 0):
            raise ValueError("red_corridor_scale_steps must be finite, positive and nonboolean")
        red_decision_active = red_corridor_width_weight > 0 or red_corridor_miss_weight > 0
        if red_decision_active and (
                isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 2):
            raise ValueError("Positive RED corridor weights require integer n_samples >= 2")
        if (isinstance(post_factor_smoothness_weight, bool)
                or not isinstance(post_factor_smoothness_weight, Real)
                or not math.isfinite(post_factor_smoothness_weight)
                or post_factor_smoothness_weight < 0):
            raise ValueError("Post factor smoothness weight must be finite, nonnegative and nonboolean")
        if post_factor_smoothness_weight > 0 and (
                self.phase_covariance != "separate"
                or getattr(self, "path_distribution", "single") != "single"):
            raise ValueError("Positive post factor smoothness requires separate covariance and single paths")
        expected_shape = (moments["current_signal"].shape[0], self.n_horizons)
        if actual_future.shape != expected_shape or target_mask.shape != expected_shape:
            raise ValueError("Path targets and masks must match the full saved batch/horizon grid")
        # Prefixes are fixed by the saved config, never by observed targets.
        prefix_lengths = path_objective_prefix_lengths
        if prefix_lengths is None:
            prefix_lengths = (self.n_horizons,)
        elif (
            not isinstance(prefix_lengths, (list, tuple))
            or not prefix_lengths
            or any(isinstance(n, bool) or not isinstance(n, Integral) for n in prefix_lengths)
            or not 0 < prefix_lengths[0]
            or prefix_lengths[-1] > self.n_horizons
            or any(a >= b for a, b in zip(prefix_lengths, prefix_lengths[1:]))
        ):
            raise ValueError("Path objective prefixes must be unique increasing integer lengths within the saved horizon")
        if post_factor_smoothness_weight > 0:
            factors = moments.get("post_factors")
            if (not isinstance(factors, torch.Tensor)
                    or factors.shape != (*expected_shape, self.rank)
                    or not torch.isfinite(factors).all()):
                raise ValueError("Post factors must be finite and match [batch, saved horizon, rank]")
            smoothness_threshold = torch.as_tensor(
                red_threshold, dtype=factors.dtype, device=factors.device,
            ).reshape(-1)
            if (smoothness_threshold.numel() not in {1, expected_shape[0]}
                    or not torch.isfinite(smoothness_threshold).all()
                    or (smoothness_threshold <= 0).any()):
                raise ValueError("RED threshold must be positive, finite and scalar or per-row")
            # Absolute forecast leads, rank-summed adjacent variance, and equal
            # prefix weights. The one-lead zero remains connected to factors.
            smoothness = sum(
                (factors[:, 1:length] - factors[:, :length - 1]).square().sum(-1).mean(-1)
                if length > 1 else factors[:, :1].sum((1, 2)) * 0
                for length in prefix_lengths
            ) / len(prefix_lengths) / smoothness_threshold.square()
        if red_decision_active:
            red_event_prefixes = event_objective_prefix_lengths
            if red_event_prefixes is None:
                red_event_prefixes = (self.n_horizons,)
            if (not isinstance(red_event_prefixes, (list, tuple)) or not red_event_prefixes
                    or any(isinstance(n, bool) or not isinstance(n, Integral) for n in red_event_prefixes)
                    or not 0 < red_event_prefixes[0] or red_event_prefixes[-1] > self.n_horizons
                    or any(a >= b for a, b in zip(red_event_prefixes, red_event_prefixes[1:]))):
                raise ValueError("Event objective prefixes must be unique increasing integer lengths within the saved horizon")
            red_threshold_tensor = torch.as_tensor(
                red_threshold, device=moments["current_signal"].device,
            ).reshape(-1)
            if (red_threshold_tensor.numel() not in {1, expected_shape[0]}
                    or not torch.isfinite(red_threshold_tensor).all()
                    or (red_threshold_tensor <= 0).any()):
                raise ValueError("RED threshold must be positive, finite and scalar or per-row")
            red_already = moments["current_signal"] >= red_threshold_tensor
            for length in red_event_prefixes:
                validate_mc_red_decision_inputs(
                    moments["event_logits"], prefix_length=length, already_red=red_already,
                    event_allowed_mask=event_allowed_mask, event_observed_mask=event_observed_mask,
                    no_entry_prefix=no_entry_prefix, coverage=coverage,
                )
        if red_finite_bound_weight > 0:
            finite_prefixes = event_objective_prefix_lengths
            if finite_prefixes is None:
                finite_prefixes = (self.n_horizons,)
            if (not isinstance(finite_prefixes, (list, tuple)) or not finite_prefixes
                    or any(isinstance(n, bool) or not isinstance(n, Integral) for n in finite_prefixes)
                    or not 0 < finite_prefixes[0] or finite_prefixes[-1] > self.n_horizons
                    or any(a >= b for a, b in zip(finite_prefixes, finite_prefixes[1:]))):
                raise ValueError("Event objective prefixes must be unique increasing integer lengths within the saved horizon")
            current = moments["current_signal"]
            if current.ndim != 1 or not torch.isfinite(current).all():
                raise ValueError("Current signal must be finite [batch] for finite-bound eligibility")
            finite_threshold = torch.as_tensor(red_threshold, device=current.device).reshape(-1)
            if (finite_threshold.numel() not in {1, expected_shape[0]}
                    or not torch.isfinite(finite_threshold).all() or (finite_threshold <= 0).any()):
                raise ValueError("RED threshold must be positive, finite and scalar or per-row")
            if (not isinstance(moments["event_logits"], torch.Tensor)
                    or moments["event_logits"].shape != (expected_shape[0], self.n_horizons + 1)):
                raise ValueError("Finite-bound logits must match the full saved batch/horizon grid")
            # Deterministic full unconditional E marginal, before any sampler.
            # Coupled K scores and evidence-conditioned phase draws do not enter.
            finite_cost = sum(finite_bound_deficiency(
                moments["event_logits"], prefix_length=length,
                already_red=current >= finite_threshold,
                event_allowed_mask=event_allowed_mask, event_observed_mask=event_observed_mask,
                no_entry_prefix=no_entry_prefix, coverage=coverage,
            )["cost"] for length in finite_prefixes) / len(finite_prefixes)
        if event_cdf_weight > 0:
            cdf_prefixes = event_objective_prefix_lengths
            if cdf_prefixes is None:
                cdf_prefixes = (self.n_horizons,)
            if (not isinstance(cdf_prefixes, (list, tuple)) or not cdf_prefixes
                    or any(isinstance(n, bool) or not isinstance(n, Integral) for n in cdf_prefixes)
                    or not 0 < cdf_prefixes[0] or cdf_prefixes[-1] > self.n_horizons
                    or any(a >= b for a, b in zip(cdf_prefixes, cdf_prefixes[1:]))):
                raise ValueError("Event objective prefixes must be unique increasing integer lengths within the saved horizon")
            current = moments["current_signal"]
            if current.ndim != 1 or not torch.isfinite(current).all():
                raise ValueError("Current signal must be finite [batch] for known-CDF eligibility")
            cdf_threshold = torch.as_tensor(red_threshold, device=current.device).reshape(-1)
            if (cdf_threshold.numel() not in {1, expected_shape[0]}
                    or not torch.isfinite(cdf_threshold).all() or (cdf_threshold <= 0).any()):
                raise ValueError("RED threshold must be positive, finite and scalar or per-row")
            if (not isinstance(moments["event_logits"], torch.Tensor)
                    or moments["event_logits"].shape != (expected_shape[0], self.n_horizons + 1)):
                raise ValueError("Known-CDF logits must match the full saved batch/horizon grid")
            # Full unconditional E marginal and fixed equal prefix mean before
            # any draw. Evidence selects known CDF bits, never probability mass.
            cdf_cost = sum(known_cdf_timing_loss(
                moments["event_logits"], prefix_length=length,
                already_red=current >= cdf_threshold,
                event_allowed_mask=event_allowed_mask, event_observed_mask=event_observed_mask,
                no_entry_prefix=no_entry_prefix, scale_steps=event_cdf_scale_steps,
            )["cost"] for length in cdf_prefixes) / len(cdf_prefixes)
        # Draw on the full event/path grid once, retaining the same joint paths
        # and categorical scores for every prefix of this unconditional sample.
        coupled = getattr(self, "path_distribution", "single") == "coupled_timing_mixture"
        if coupled:
            paths, entries, _, joint_score = self.sample(
                moments, red_threshold, n_samples, generator, return_latent=True
            )
        else:
            paths, entries = self.sample(
                moments, red_threshold, n_samples, generator, return_entry=True
            )
        # Exact discrete draws require score-function gradients. Within-row
        # independent baselines prevent high-amplitude post-RED rows adding
        # categorical gradient noise to unrelated pre-RED equipment origins.
        log_prob = moments["event_logits"].log_softmax(-1)
        sampled_log_prob = log_prob.gather(1, entries.T.clamp_min(0)).T
        sampled_log_prob = sampled_log_prob * (entries >= 0)
        if coupled:
            sampled_log_prob = joint_score
        prefix_terms = []
        for length in prefix_lengths:
            prefix_paths = paths[..., :length]
            target, mask = actual_future[..., :length], target_mask[..., :length]
            values = path_losses(
                prefix_paths, target, mask, coverage=coverage, signal_scale=red_threshold,
                path_band_geometry=path_band_geometry,
            )
            surrogates = path_score_surrogates(
                prefix_paths, target, mask, sampled_log_prob,
                coverage=coverage, signal_scale=red_threshold,
                path_band_geometry=path_band_geometry,
            )
            prefix_terms.append({
                name: value + surrogates[name] - surrogates[name].detach()
                for name, value in values.items()
            })
        terms = {
            name: sum(prefix[name] for prefix in prefix_terms) / len(prefix_terms)
            for name in prefix_terms[0]
        }
        event_prefixes = event_objective_prefix_lengths
        if event_prefixes is None:
            event_prefixes = (self.n_horizons,)
        elif (
            not isinstance(event_prefixes, (list, tuple)) or not event_prefixes
            or any(isinstance(n, bool) or not isinstance(n, Integral) for n in event_prefixes)
            or not 0 < event_prefixes[0] or event_prefixes[-1] > self.n_horizons
            or any(a >= b for a, b in zip(event_prefixes, event_prefixes[1:]))
        ):
            raise ValueError("Event objective prefixes must be unique increasing integer lengths within the saved horizon")
        terms["event"] = sum(first_entry_loss(
            moments["event_logits"], event_allowed_mask=event_allowed_mask,
            event_observed_mask=event_observed_mask, no_entry_prefix=no_entry_prefix,
            prefix_length=length,
        ) for length in event_prefixes) / len(event_prefixes)
        threshold = torch.as_tensor(red_threshold, device=actual_future.device).reshape(-1)
        terms["event"] = terms["event"] * (moments["current_signal"] < threshold)
        # Supervised phase energy prevents rare entry/post locations receiving
        # gradients only when an unconditional draw happens to select that phase.
        # This auxiliary conditional score is separate from the proper joint
        # energy diagnostic above, which keeps the original mixture unchanged.
        allowed, known = compatible_entry_mask(
            moments["event_logits"], event_allowed_mask=event_allowed_mask,
            event_observed_mask=event_observed_mask, no_entry_prefix=no_entry_prefix,
        )
        conditional = dict(moments)
        conditional["event_logits"] = moments["event_logits"].masked_fill(~allowed, -torch.inf)
        if coupled:
            conditional_paths, _, _, conditional_score = self.sample(
                moments, red_threshold, n_samples, generator,
                entry_allowed_mask=allowed, return_latent=True,
            )
        else:
            conditional_paths = self.sample(conditional, red_threshold, n_samples, generator)
        phase_masks = observed_phase_masks(
            allowed, known, target_mask, moments["current_signal"] >= threshold,
        )
        active_phases = torch.zeros_like(terms["energy"])
        phase_sum = torch.zeros_like(terms["energy"])
        for phase, observed_mask in phase_masks.items():
            value = path_losses(
                conditional_paths, actual_future, observed_mask,
                coverage=coverage, signal_scale=red_threshold, energy_only=True,
            )["energy"]
            if coupled:
                surrogate = path_score_surrogates(
                    conditional_paths, actual_future, observed_mask, conditional_score,
                    signal_scale=red_threshold, energy_only=True,
                )["energy"]
                value = value + surrogate - surrogate.detach()
            terms["phase_" + phase] = value
            phase_sum = phase_sum + value
            active_phases = active_phases + observed_mask.any(-1)
        terms["phase_energy"] = phase_sum / active_phases.clamp_min(1)
        terms["total"] = (
            energy_weight * terms["energy"]
            + width_weight * terms["width"]
            + miss_weight * terms["miss"]
            + event_weight * terms["event"]
            + phase_weight * terms["phase_energy"]
        )
        if red_decision_active:
            # Only the original unconditional E draws and full marginal q(E).
            # Coupled K scores and evidence-conditioned phase draws are excluded.
            red_terms = [mc_red_decision_regularizer(
                moments["event_logits"], entries, prefix_length=length, already_red=red_already,
                event_allowed_mask=event_allowed_mask, event_observed_mask=event_observed_mask,
                no_entry_prefix=no_entry_prefix, coverage=coverage,
            ) for length in red_event_prefixes]
            terms["red_corridor_width"] = sum(t["width_term"] for t in red_terms) / len(red_terms) / red_corridor_scale_steps
            terms["red_corridor_miss"] = sum(t["compatible_miss_term"] for t in red_terms) / len(red_terms) / red_corridor_scale_steps
            terms["total"] = (terms["total"]
                              + red_corridor_width_weight * terms["red_corridor_width"]
                              + red_corridor_miss_weight * terms["red_corridor_miss"])
        if post_factor_smoothness_weight > 0:
            terms["post_factor_smoothness"] = smoothness
            terms["total"] = terms["total"] + post_factor_smoothness_weight * smoothness
        if red_finite_bound_weight > 0:
            terms["red_finite_bound"] = finite_cost
            terms["total"] = terms["total"] + red_finite_bound_weight * finite_cost
        if event_cdf_weight > 0:
            terms["event_cdf"] = cdf_cost
            terms["total"] = terms["total"] + event_cdf_weight * cdf_cost
        return terms
