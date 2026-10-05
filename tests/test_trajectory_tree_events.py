"""Synthetic engineering checks for Train-only tree event feature artifacts."""
import copy

import numpy as np
import pandas as pd
import pytest

from pdm import learned_trajectory as learned


def _frame():
    x = np.arange(48, dtype=np.float32).reshape(12, 2, 2) / 50
    observed = np.arange(12) % 2 == 0
    allowed = np.zeros((12, 4), bool)
    allowed[observed, 1] = True
    allowed[~observed, 3] = True
    return dict(x=x, raw_x=x[:, :, :1], y=np.tile([.5, 1.2, 1.4], (12, 1)).astype(np.float32),
                mask=np.ones((12, 3), bool), current=np.full(12, .3, np.float32),
                red_threshold=np.ones(12, np.float32), event_allowed=allowed,
                event_observed=observed, no_entry_prefix=np.where(observed, 0, 3),
                physical_unit_id=['a'] * 8 + ['b'] * 4, feature_names=['f1', 'f2'])


def _config():
    features = pd.DataFrame(dict(unit_id=['a'] * 4, timestamp_s=[0., 60., 120., 180.],
                                 gap_before=[False] * 4))
    return learned.learned_params('quantile_boosting', dict(
        horizons_s=[60., 120., 180.], history_length=2, epochs=1, hidden_size=16,
        num_layers=1, rank=2, batch_size=16, tree_horizons=1, max_iter=2,
        path_samples=16, training_samples=4, cpu_threads=1), features)


def test_censored_cutoff_targets_and_observed_later_entry_negative():
    frame = _frame()
    frame['current'][0] = 1.1  # Already RED, even though inconsistent event flag is set.
    frame['no_entry_prefix'][1] = 1
    frame['no_entry_prefix'][3] = 0
    labels, known = learned._tree_event_targets(frame, 1)
    assert not known[0] and not known[1] and not known[3]
    assert known[2] and labels[2] == 1
    assert known[5] and labels[5] == 0
    earlier_labels, earlier_known = learned._tree_event_targets(frame, 0)
    assert earlier_known[2] and earlier_labels[2] == 0  # Observed entry after cutoff.
    assert earlier_known[1] and earlier_labels[1] == 0
    assert not earlier_known[3]


def test_event_cutoffs_nearest_clamped_deduplicated_and_full():
    grid = np.arange(1, 101) * 90.
    indices = learned._tree_event_horizons(grid)
    assert indices == [6, 12, 19, 39, 79, 99]
    assert learned._tree_event_horizons([60., 120., 180.]) == [2]


def test_fit_weights_balance_known_physical_groups_and_inputs_exclude_ids(monkeypatch):
    frame = _frame()
    frame['no_entry_prefix'][1] = 0  # This row has no admitted evidence.
    flat = frame['x'].reshape(12, -1)
    calls = []
    class Spy:
        def __init__(self, **kwargs):
            assert kwargs['early_stopping'] is False
        def fit(self, x, labels, sample_weight):
            calls.append((x.copy(), labels.copy(), sample_weight.copy()))
            return self
    monkeypatch.setattr(learned, 'HistGradientBoostingClassifier', Spy)
    heads = learned._fit_tree_event_heads(frame, flat, _config())
    x, labels, weights = calls[0]
    np.testing.assert_array_equal(x, flat[np.arange(12) != 1])
    assert x.shape[1] == 4  # Only numeric causal history.
    assert weights[:7].sum() == pytest.approx(5.5)
    assert weights[7:].sum() == pytest.approx(5.5)
    assert weights.mean() == pytest.approx(1.)
    assert heads[0]['fit_physical_groups'] == ['a', 'b']
    assert heads[0]['known_rows'] == 11
    assert set(labels) == {0, 1}


def test_quantile_weights_rebalance_each_observed_horizon_with_uneven_missingness(monkeypatch):
    train, config = _frame(), _config()
    config['tree_horizons'] = 3
    train['mask'][[0, 2, 4, 6, 9, 11], 2] = False
    train['no_entry_prefix'][[9, 11]] = 2
    original_fit = learned.HistGradientBoostingRegressor.fit
    calls = []
    def record_fit(self, x, y, **kwargs):
        calls.append((x.copy(), kwargs['sample_weight'].copy()))
        return original_fit(self, x, y, **kwargs)
    monkeypatch.setattr(learned.HistGradientBoostingRegressor, 'fit', record_fit)
    bundle = learned.fit_learned_model('quantile_boosting', train, _frame(), config)
    flat = learned._normalized(train, bundle['scaler']).reshape(12, -1)
    assert len(calls) == 9
    for h in range(3):
        valid = train['mask'][:, h]
        groups = np.asarray(train['physical_unit_id'])[valid]
        for x, weights in calls[h * 3:(h + 1) * 3]:
            np.testing.assert_array_equal(x, flat[valid])
            assert weights.mean() == pytest.approx(1.)
            assert weights[groups == 'a'].sum() == pytest.approx(valid.sum() / 2)
            assert weights[groups == 'b'].sum() == pytest.approx(valid.sum() / 2)
    assert bundle['encoder']['tree_weight_policy'] == learned.TREE_WEIGHT_POLICY


def test_constant_class_and_unknown_evidence_are_explicit_artifacts(monkeypatch):
    frame = _frame()
    frame['event_observed'][:] = False
    frame['no_entry_prefix'][:] = 3
    monkeypatch.setattr(learned, 'HistGradientBoostingClassifier',
                        lambda **kw: pytest.fail('Single-class classifier must not be constructed'))
    heads = learned._fit_tree_event_heads(frame, frame['x'].reshape(12, -1), _config())
    assert heads[0]['kind'] == 'constant' and heads[0]['probability'] == 0.
    frame['no_entry_prefix'][:] = 0
    unknown = learned._fit_tree_event_heads(frame, frame['x'].reshape(12, -1), _config())[0]
    assert unknown['known_rows'] == 0 and unknown['probability'] == .5
    assert unknown['reason'] == 'no_known_train_evidence'


def test_fingerprint_binds_event_evidence_and_weighting_groups():
    frame = _frame()
    initial = learned._tree_fit_fingerprint(frame)
    for key in ('event_observed', 'event_allowed', 'no_entry_prefix', 'current', 'red_threshold'):
        changed = copy.deepcopy(frame)
        changed[key].flat[0] = not changed[key].flat[0] if changed[key].dtype == bool else 999
        assert learned._tree_fit_fingerprint(changed) != initial
    changed = copy.deepcopy(frame)
    changed['physical_unit_id'][0] = 'other'
    assert learned._tree_fit_fingerprint(changed) != initial


def test_actual_tiny_boost_fit_reload_train_only_and_legacy_bundle(tmp_path, monkeypatch):
    train, validation, config = _frame(), _frame(), _config()
    validation['physical_unit_id'] = ['validation_only'] * 12
    validation['x'] += 10
    original_fit = learned.HistGradientBoostingClassifier.fit
    fitted = []
    def record_fit(self, x, y, **kwargs):
        fitted.append(x.copy())
        return original_fit(self, x, y, **kwargs)
    monkeypatch.setattr(learned.HistGradientBoostingClassifier, 'fit', record_fit)
    bundle = learned.fit_learned_model('quantile_boosting', train, validation, config)
    assert len(fitted) == 1
    assert bundle['encoder']['event_heads'][0]['fit_physical_groups'] == ['a', 'b']
    assert bundle['encoder']['risk_head_policy']['validation_or_test_tree_fit'] is False
    expected_flat = learned._normalized(train, bundle['scaler']).reshape(12, -1)
    np.testing.assert_array_equal(fitted[0], expected_flat)
    encoder = bundle['encoder']
    external = learned._external(encoder, validation, bundle['scaler'])
    assert np.isfinite(external).all() and ((external[:, -1] >= 0) & (external[:, -1] <= 1)).all()
    artifacts = learned.save_learned_bundle(bundle, tmp_path / 'new')
    run = dict(dir=tmp_path / 'new', artifacts=artifacts, params=config, engine_id='quantile_boosting')
    loaded = learned.load_learned_bundle(run)
    before, after = learned.predict_learned(bundle, validation), learned.predict_learned(loaded, validation)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])
    artifact = tmp_path / 'new' / 'encoder.joblib'
    original = artifact.read_bytes()
    artifact.write_bytes(original + b'changed event artifact')
    with pytest.raises(ValueError, match='hash mismatch: encoder.joblib'):
        learned.load_learned_bundle(run)
    artifact.write_bytes(original)
    # An actual saved pre-event-head encoder has its original feature width and reloads.
    legacy = copy.deepcopy(bundle)
    del legacy['encoder']['event_heads']
    old_external = learned._external(legacy['encoder'], validation, bundle['scaler'])
    np.testing.assert_array_equal(old_external, external[:, :-1])
    legacy['model'] = learned._make_model('quantile_boosting', config, 2, old_external.shape[1])
    legacy['external_scaler']['mean'] = legacy['external_scaler']['mean'][:-1]
    legacy['external_scaler']['std'] = legacy['external_scaler']['std'][:-1]
    old_artifacts = learned.save_learned_bundle(legacy, tmp_path / 'old')
    old_loaded = learned.load_learned_bundle(dict(run, dir=tmp_path / 'old', artifacts=old_artifacts))
    learned.predict_learned(old_loaded, validation)
