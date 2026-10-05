"""Causal, physically weighted probabilistic heads for RED survival bins."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

CANDIDATES = ({'n_estimators': 32, 'max_depth': 1, 'learning_rate': .05},
              {'n_estimators': 32, 'max_depth': 2, 'learning_rate': .05})


def _cancel(should_stop):
    if should_stop and should_stop():
        raise InterruptedError('RED-entry adapter training cancelled')


def valid_windows(batch):
    x = np.asarray(batch['x'], dtype=np.float32)
    lengths = np.asarray(batch['lengths'])
    if x.ndim != 3 or lengths.shape != (len(x),) or not np.issubdtype(lengths.dtype, np.integer):
        raise ValueError('Expected windows [N,H,F] and integer lengths [N]')
    if np.any(lengths < 1) or np.any(lengths > x.shape[1]):
        raise ValueError('Every hazard origin needs a nonempty real context window')
    for row, length in zip(x, lengths, strict=True):
        if not np.isfinite(row[:length]).all():
            raise ValueError('Real context values must be finite')
    return x, lengths


def causal_design(batch):
    """Current context and summaries use real causal rows only; ignore padding."""
    x, lengths = valid_windows(batch)
    design = np.empty((len(x), x.shape[2] * 5), dtype=np.float32)
    for i, length in enumerate(lengths):
        window = x[i, :length]
        design[i] = np.concatenate((window[-1], window.mean(axis=0), window.std(axis=0),
                                    window.min(axis=0), window.max(axis=0)))
    return design


def unit_weights(batch):
    mask = np.asarray(batch['mask'], dtype=bool)
    ids = np.asarray(batch['physical_unit_ids']).astype(str)
    if mask.ndim != 2 or len(ids) != len(mask):
        raise ValueError('Survival mask and physical units must align')
    valid = mask.any(axis=1)
    weights = np.zeros(len(mask), dtype=float)
    for uid in np.unique(ids[valid]):
        indices = valid & (ids == uid)
        weights[indices] = 1 / indices.sum()
    return weights


@dataclass
class ConstantHead:
    probability: float

    def predict_proba(self, x):
        p = np.full(len(x), self.probability)
        return np.column_stack((1-p, p))


@dataclass
class HazardHeads:
    heads: list = field(default_factory=list)
    support_by_bin: list = field(default_factory=list)
    head_status: list = field(default_factory=list)
    design_mean: np.ndarray | None = None
    design_std: np.ndarray | None = None
    provenance: dict = field(default_factory=dict)
    red_entry_adapter_module: str = 'pdm.models.red_entry_boosting'

    def predict_design(self, design):
        if design.ndim != 2 or design.shape[1] != len(self.design_mean):
            raise ValueError('Hazard readout feature width changed')
        z = (design-self.design_mean)/self.design_std
        return np.column_stack([h.predict_proba(z)[:, 1] for h in self.heads])


def fit_heads(model, design, train, make_classifier, *, should_stop=None):
    y, mask = np.asarray(train['y']), np.asarray(train['mask'], dtype=bool)
    if y.shape != mask.shape or len(y) != len(design):
        raise ValueError('Targets and masks must align with causal designs')
    if not np.isin(y[mask], (0, 1)).all():
        raise ValueError('Observed hazards must have binary targets')
    weights = unit_weights(train)
    if not weights.any():
        raise ValueError('Train has no observed survival bins')
    model.design_mean = np.average(design, axis=0, weights=weights)
    variance = np.average((design-model.design_mean)**2, axis=0, weights=weights)
    model.design_std = np.maximum(np.sqrt(variance), 1e-6)
    z = (design-model.design_mean)/model.design_std
    for k in range(y.shape[1]):
        _cancel(should_stop)
        rows = mask[:, k]
        model.support_by_bin.append(bool(rows.any()))
        classes = np.unique(y[rows, k])
        if len(classes) < 2:
            p = float(classes[0]) if len(classes) else .5
            model.heads.append(ConstantHead(p))
            model.head_status.append('single_class_constant' if len(classes) else 'unsupported_no_train_bins')
        else:
            head = make_classifier()
            head.fit(z[rows], y[rows, k], sample_weight=weights[rows])
            model.heads.append(head)
            model.head_status.append('fitted_two_class')
    return model


def validation_score(model, design, validation):
    from pdm.red_entry_training import unit_equal_nll
    mask = np.asarray(validation['mask'], dtype=bool) & np.asarray(model.support_by_bin)[None, :]
    if not mask.any():
        raise ValueError('Validation has no training-supported observed bins')
    targets = np.asarray(validation['y'])
    if not np.isin(targets[mask], (0, 1)).all():
        raise ValueError('Observed Validation hazards must have binary targets')
    score = unit_equal_nll(model.predict_design(design),
                           {**validation, 'mask': mask, 'y': np.where(mask, targets, 0)})
    if not np.isfinite(score):
        raise ValueError('Nonfinite Validation NLL')
    return score


def fit_model(train, validation, config, *, should_stop=None, status_cb=None):
    _cancel(should_stop)
    design, val_design = causal_design(train), causal_design(validation)
    best, best_score, scores = None, float('inf'), []
    for i, candidate in enumerate(CANDIDATES):
        model = fit_heads(HazardHeads(), design, train,
                          lambda: GradientBoostingClassifier(**candidate, random_state=config['seed']),
                          should_stop=should_stop)
        score = validation_score(model, val_design, validation)
        scores.append({'candidate': i, 'params': dict(candidate), 'value': score})
        if score < best_score:
            best, best_score, selected = model, score, i
        if status_cb:
            status_cb({'adapter': 'hazard_boosting', 'candidate': i, 'validation_unit_equal_nll': score})
    best.provenance = {'adapter_version': 'red_entry_boosting_v1',
                       'feature_policy': 'last_real_context_mean_std_min_max_v1',
                       'standardization': 'train_only_physical_unit_equal',
                       'weighting': 'one_over_observed_origins_per_physical_unit',
                       'head_status': best.head_status, 'support_by_bin': best.support_by_bin,
                       'candidate_policy': 'fixed_two_depth_candidates_v1', 'seed': config['seed']}
    return best, {'part': 'validation', 'metric': 'unit_equal_masked_survival_nll',
                  'value': best_score, 'selected_candidate': selected, 'candidates': scores,
                  'support_by_bin': best.support_by_bin, 'test_used': False}


def predict_hazard(model, batch):
    return model.predict_design(causal_design(batch))
