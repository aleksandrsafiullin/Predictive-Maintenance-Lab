"""Bounded adapter contracts; graph fixture is explicitly not real-brain QA."""
from types import SimpleNamespace

import joblib
import numpy as np
import pytest
import torch
from scipy import sparse

from pdm.models import red_entry_boosting as boost
from pdm.models import red_entry_full_cns as cns


def batch():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(8, 4, 2)).astype(np.float32)
    return {'x': x, 'lengths': np.array([1, 2, 3, 4, 1, 2, 3, 4]),
            'y': np.array([[0, 1, 0], [1, 0, 0]]*4, dtype=np.float32),
            'mask': np.array([[True, True, False]]*8),
            'physical_unit_ids': np.array(['a']*6+['b']*2)}


def tiny_body():
    # Directed, non-symmetric official-shaped fixture. No production fallback.
    matrix = sparse.csr_matrix(np.array([[0, .2, 0], [0, 0, .3], [.1, 0, 0]], np.float32))
    return SimpleNamespace(W_res=torch.tensor(matrix.toarray()).to_sparse_csr(),
                           W_in=torch.tensor([[.1, -.2], [.2, .1], [-.1, .2]]),
                           b_res=torch.zeros(3), alpha=.2,
                           pool_operator=sparse.csr_matrix([[.5, .5, 0], [0, 0, 1]], dtype=np.float32),
                           node_order=['10', '20', '30'],
                           provenance={'is_synthetic': False, 'node_sampling': False,
                                       'graph_hash': 'fixture_only', 'n_nodes': 3})


@pytest.fixture
def mock_official_shape(monkeypatch):
    monkeypatch.setattr(cns, 'EXPECTED_NEURONS', 3)
    monkeypatch.setattr(cns, 'source_unavailable_reason', lambda: None)
    monkeypatch.setattr(cns, 'build_full_cns', lambda **kw: tiny_body())


def test_causal_design_and_physical_weights():
    b = batch()
    design = boost.causal_design(b)
    changed = {**b, 'x': b['x'].copy()}
    for i, length in enumerate(b['lengths']):
        changed['x'][i, length:] = np.nan
    np.testing.assert_array_equal(design, boost.causal_design(changed))
    np.testing.assert_array_equal(design[:, :2], b['x'][np.arange(8), b['lengths']-1])
    weights = boost.unit_weights(b)
    assert weights[:6].sum() == pytest.approx(1)
    assert weights[6:].sum() == pytest.approx(1)


@pytest.mark.parametrize('module', [boost, cns])
def test_hazard_fit_unknown_mask_and_reload(module, mock_official_shape, tmp_path):
    b = batch()
    status = []
    model, selection = module.fit_model(b, b, {'seed': 42}, status_cb=status.append)
    pred = module.predict_hazard(model, b)
    assert pred.shape == (8, 3) and np.isfinite(pred).all()
    assert ((pred >= 0) & (pred <= 1)).all()
    assert model.support_by_bin == [True, True, False]
    assert model.head_status[2] == 'unsupported_no_train_bins'
    assert selection['part'] == 'validation' and selection['test_used'] is False
    assert len(selection['candidates']) == 2 and status
    path = tmp_path/'model.joblib'
    joblib.dump(model, path)
    np.testing.assert_array_equal(pred, module.predict_hazard(joblib.load(path), b))
    changed = {**b, 'y': b['y'].copy()}
    changed['y'][:, 2] = 987  # Unknown targets are neither negatives nor fitted labels.
    other, _ = module.fit_model(changed, b, {'seed': 42})
    np.testing.assert_array_equal(pred, module.predict_hazard(other, b))


def test_full_state_short_history_padding_and_window_reset(mock_official_shape):
    model = cns.RedEntryFullCNS(tiny_body())
    b = batch()
    expected = model.transform(b)
    changed = {**b, 'x': b['x'].copy()}
    for i, length in enumerate(b['lengths']):
        changed['x'][i, length:] = 1e8
    np.testing.assert_array_equal(expected, model.transform(changed))
    for i in range(len(b['x'])):
        one = {**b, 'x': b['x'][i:i+1, :b['lengths'][i]], 'lengths': b['lengths'][i:i+1]}
        np.testing.assert_allclose(expected[i], model.transform(one)[0], atol=1e-7)
    assert model.operator.nnz == 3
    assert model.provenance['operator_hash'] and model.provenance['projection_hash']
    assert model.provenance['pooling_hash'] and model.provenance['node_order_hash']


@pytest.mark.parametrize('module', [boost, cns])
def test_cancel_and_no_observed_validation(module, mock_official_shape):
    b = batch()
    with pytest.raises(InterruptedError):
        module.fit_model(b, b, {'seed': 42}, should_stop=lambda: True)
    unknown = {**b, 'mask': np.zeros_like(b['mask'])}
    with pytest.raises(ValueError, match='Validation'):
        module.fit_model(b, unknown, {'seed': 42})


def test_constant_head_and_strict_complete_graph(mock_official_shape, monkeypatch):
    b = batch()
    b['y'][:] = 0
    model, _ = boost.fit_model(b, b, {'seed': 42})
    assert model.head_status[:2] == ['single_class_constant']*2
    np.testing.assert_array_equal(boost.predict_hazard(model, b)[:, :2], 0)
    monkeypatch.setattr(cns, 'EXPECTED_NEURONS', 166700)
    with pytest.raises(ValueError, match='complete 166700'):
        cns.RedEntryFullCNS(tiny_body())


def test_full_cns_postfit_dispatch_cancellation_and_progress(mock_official_shape):
    from pdm.red_entry_training import predict_hazard
    b = batch()
    model, _ = cns.fit_model(b, b, {'seed': 42})
    checks = []
    def stop_after_dispatch():
        checks.append(True)
        return len(checks) >= 3
    with pytest.raises(InterruptedError):
        predict_hazard(model, b, should_stop=stop_after_dispatch)
    assert len(checks) == 3  # dispatcher and actual graph-step checks
    progress = []
    np.testing.assert_array_equal(predict_hazard(model, b, status_cb=progress.append),
                                  cns.predict_hazard(model, b))
    assert progress
