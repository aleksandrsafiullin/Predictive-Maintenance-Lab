"""Official complete MaleCNS states with an engineering RED hazard readout.

Every classified neuron and original directed edge participates in dynamics.
Projection, leaky tanh and anatomical pooling are versioned engineering choices,
not biological sensory coding. A persisted sparse operator freezes inference.
"""
from __future__ import annotations

import copy
import hashlib

import numpy as np
from sklearn.linear_model import LogisticRegression

from pdm.models.full_cns import build_full_cns, scipy_csr
from pdm.models.red_entry_boosting import (
    HazardHeads,
    _cancel,
    fit_heads,
    valid_windows,
    validation_score,
)
from pdm.models.signal_full_cns import source_unavailable_reason

EXPECTED_NEURONS = 166700
READOUT_CANDIDATES = (.1, 1.)


def _array_hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(memoryview(a).cast('B'))
    return digest.hexdigest()


class RedEntryFullCNS(HazardHeads):
    def __init__(self, body):
        super().__init__(red_entry_adapter_module='pdm.models.red_entry_full_cns')
        self.operator = scipy_csr(body.W_res).copy()
        self.input_weights = body.W_in.detach().cpu().numpy().copy()
        self.bias = body.b_res.detach().cpu().numpy().copy()
        self.pool = body.pool_operator.copy()
        self.node_order = list(body.node_order)
        self.leak = float(body.alpha)
        if len(self.node_order) != EXPECTED_NEURONS:
            raise ValueError('RED Full MaleCNS requires the complete 166700-neuron official graph')
        if body.provenance.get('is_synthetic') or body.provenance.get('node_sampling'):
            raise ValueError('RED Full MaleCNS refuses synthetic or sampled topology')
        self.provenance = {
            **body.provenance, 'adapter_version': 'red_entry_full_cns_v1',
            'input_projection_version': 'seeded_uniform_multichannel_v1',
            'input_policy': 'all ordered model channels; engineering projection',
            'dynamics_version': 'all_neurons_leaky_tanh_v1',
            'pooling_version': 'mean_superclass_class_somaSide_v1',
            'state_policy': 'zero_reset_per_true_window; only lengths real rows advance',
            'readout_version': 'per_bin_logistic_pooled_plus_last_context_v1',
            'standardization': 'train_only_physical_unit_equal',
            'weighting': 'one_over_observed_origins_per_physical_unit',
            'neurotransmitter_policy': 'unsigned source synapse counts',
            'operator_hash': _array_hash(self.operator.indptr, self.operator.indices, self.operator.data),
            'projection_hash': _array_hash(self.input_weights, self.bias),
            'dynamics_hash': _array_hash(np.asarray([self.leak], dtype=np.float64)),
            'pooling_hash': _array_hash(self.pool.indptr, self.pool.indices, self.pool.data),
            'node_order_hash': hashlib.sha256('\n'.join(self.node_order).encode()).hexdigest(),
            'storage_policy': 'full frozen sparse operator in model.joblib; no source reload',
            'operator_storage_bytes': int(self.operator.data.nbytes+self.operator.indices.nbytes+self.operator.indptr.nbytes),
        }

    def transform(self, batch, *, should_stop=None, status_cb=None):
        x, lengths = valid_windows(batch)
        if x.shape[-1] != self.input_weights.shape[1]:
            raise ValueError('Full CNS input channel count changed')
        design = np.empty((len(x), self.pool.shape[0]+x.shape[-1]), np.float32)
        # Eight independent windows limit temporary state to neurons × eight.
        for start in range(0, len(x), 8):
            values, sizes = x[start:start+8], lengths[start:start+8]
            state = np.zeros((len(self.node_order), len(values)), np.float32)
            for step in range(int(sizes.max())):
                _cancel(should_stop)
                active = np.flatnonzero(sizes > step)
                previous = state[:, active]
                drive = self.operator @ previous + self.input_weights @ values[active, step].T
                drive += self.bias[:, None]
                state[:, active] = (1-self.leak)*previous+self.leak*np.tanh(drive)
            design[start:start+len(values), :self.pool.shape[0]] = (self.pool @ state).T
            design[start:start+len(values), self.pool.shape[0]:] = values[np.arange(len(values)), sizes-1]
            if status_cb:
                status_cb({'adapter': 'full_cns', 'completed_windows': start+len(values), 'total_windows': len(x)})
        if not np.isfinite(design).all():
            raise FloatingPointError('Nonfinite full CNS state/design')
        return design


def fit_model(train, validation, config, *, should_stop=None, status_cb=None):
    _cancel(should_stop)
    reason = source_unavailable_reason()
    if reason:
        raise FileNotFoundError(reason)
    if status_cb:
        status_cb({'adapter': 'full_cns', 'phase': 'loading_complete_official_graph'})
    body = build_full_cns(input_size=train['x'].shape[-1], time_scale_s=1., seed=config['seed'])
    _cancel(should_stop)
    model = RedEntryFullCNS(body)
    del body
    design = model.transform(train, should_stop=should_stop, status_cb=status_cb)
    val_design = model.transform(validation, should_stop=should_stop, status_cb=status_cb)
    best, best_score, scores = None, float('inf'), []
    # Candidate copies share all immutable large graph arrays; only small heads differ.
    for i, strength in enumerate(READOUT_CANDIDATES):
        candidate = copy.copy(model)
        candidate.heads, candidate.support_by_bin, candidate.head_status = [], [], []
        fit_heads(candidate, design, train,
                  lambda: LogisticRegression(C=strength, solver='lbfgs', max_iter=500, random_state=config['seed']),
                  should_stop=should_stop)
        score = validation_score(candidate, val_design, validation)
        scores.append({'candidate': i, 'C': strength, 'value': score})
        if score < best_score:
            best, best_score, selected = candidate, score, i
        if status_cb:
            status_cb({'adapter': 'full_cns', 'candidate': i, 'validation_unit_equal_nll': score})
    best.provenance = {**best.provenance, 'head_status': best.head_status,
                       'support_by_bin': best.support_by_bin, 'candidate_policy': 'fixed_logistic_C_v1'}
    return best, {'part': 'validation', 'metric': 'unit_equal_masked_survival_nll',
                  'value': best_score, 'selected_candidate': selected, 'candidates': scores,
                  'support_by_bin': best.support_by_bin, 'test_used': False}


def predict_hazard(model, batch, *, should_stop=None, status_cb=None):
    return model.predict_design(model.transform(batch, should_stop=should_stop, status_cb=status_cb))
