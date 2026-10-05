import numpy as np
import pytest

from pdm.signal_funnel import (
    apply_calibration,
    build_residual_bank,
    calibrate_band,
    empirical_band,
    evaluate_paths,
    load_sampler,
    red_corridor,
    sample_paths,
    save_sampler,
)


def bank():
    return build_residual_bank([[1, 2], [2, 4], [-1, -2]], np.ones((3, 2), bool), ['a', 'a', 'b'], [1, 2])


def test_complete_correlated_vectors_anchor_and_balanced_sampling():
    b = build_residual_bank([[1, 2], [2, 4], [-1, -2], [99, 0]],
                            [[1, 1], [1, 1], [1, 1], [1, 0]], ['a', 'a', 'b', 'c'], [1, 2])
    p = sample_paths([10, 20], 3.123456789, b, 10000, 42)
    assert np.all(p[:, 0] == 3.123456789)
    assert np.allclose((p[:, 2]-20), 2*(p[:, 1]-10))
    assert .48 < np.mean(p[:, 1] < 10) < .52
    assert b['excluded_incomplete_count'] == 1
    assert np.array_equal(p, sample_paths([10, 20], 3.123456789, b, 10000, 42))


def test_missing_vectors_do_not_manufacture_support():
    b = build_residual_bank([[1, 2], [99, 0]], [[1, 1], [1, 0]], ['a', 'b'], [1, 2])
    assert b['status'] == 'insufficient_support'
    with pytest.raises(ValueError, match='two physical'):
        sample_paths([0, 0], 0, b)


def test_reload_deterministic_and_tamper(tmp_path):
    path = tmp_path / 'sampler.npz'
    digest = save_sampler(path, bank())
    restored = load_sampler(path, digest)
    assert np.array_equal(sample_paths([0, 0], 1, bank()), sample_paths([0, 0], 1, restored))
    path.write_bytes(path.read_bytes()+b'tamper')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_sampler(path, digest)


def test_three_calibration_groups_insufficient_and_group_max():
    y = np.zeros((30, 2))
    cal = calibrate_band(y-1, y+1, y, np.ones_like(y, bool), np.repeat(['a', 'b', 'c'], 10))
    assert cal['status'] == 'insufficient_calibration'
    assert cal['physical_group_count'] == 3
    assert cal['finite_sample_rank'] == 4
    assert cal['expansion'] is None
    assert np.array_equal(apply_calibration([-2, -1], [-2, 1], cal)[0], [-2, -1])
    y = np.arange(10).reshape(10, 1)
    cal = calibrate_band(np.zeros_like(y), np.zeros_like(y), y, np.ones_like(y, bool), list('abcdefghij'))
    assert cal['status'] == 'calibrated'
    assert cal['expansion'] == 9
    low, high = apply_calibration([3, 0], [3, 0], cal)
    assert low[0] == high[0] == 3
    assert low[1] == -9 and high[1] == 9


def test_unknown_calibration_group_excluded():
    y = np.zeros((4, 2))
    cal = calibrate_band(y, y, y, [[1, 1], [1, 0], [1, 1], [1, 1]], ['a', 'a', 'b', 'c'])
    assert cal['physical_group_count'] == 2
    assert cal['unknown_group_count'] == 1


def test_crossings_no_entry_conditional_brackets_inclusive_below():
    paths = np.array([[1, 2, 4], [1, 4, 1], [1, 1, 1], [1, 1, 1.]])
    corridor = red_corridor(paths, [10, 20], {'status': 'available', 'direction': 'above', 'red': 4}, issued=100)
    assert corridor['probability_within_horizon'] == .5
    assert corridor['no_entry_probability'] == .5
    assert corridor['crossing_brackets_s'] == [[110., 120.], [100., 110.]]
    assert corridor['conditional_bounds_s'] == [100., 120.]
    assert corridor['unconditional_bounds_s'] is None
    assert corridor['unconditional_status'] == 'open_upper_bound'
    below = red_corridor([[5, 4, 3]], [10, 20], {'status': 'available', 'direction': 'below', 'red': 4})
    assert below['crossing_brackets_s'] == [[0., 10.]]
    assert red_corridor([[4, 3, 2]], [1, 2], {'status': 'available', 'direction': 'above', 'red': 4})['status'] == 'already_red'
    assert red_corridor(paths, [1, 2], {'status': 'unavailable'})['status'] == 'unknown_threshold'


def test_interval_width_miss_and_unknown_energy_population():
    paths = np.array([[[0., 0.], [2., 2.]], [[0., 0.], [2., 2.]]])
    result = evaluate_paths(paths, [[3, 3], [1, 99]], [[1, 1], [1, 0]], ['a', 'b'], coverage=.5)
    assert result['interval_score_by_horizon'][0] == pytest.approx(4.)
    assert result['whole_path_coverage'] == 0
    assert result['unknown_path_count'] == 1
    assert result['unknown_target_count_by_horizon'] == [0, 1]
    assert result['joint_energy_score'] == pytest.approx(np.sqrt(2))


def test_joint_inference_future_suffix_invariant_direct_only_and_anchor(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from pdm import signal_inference as inference
    data = {'features': pd.DataFrame({'unit_id': ['test'] * 6, 'timestamp_s': np.arange(6.),
                                     'signal': [1., 2., 3.123456789, 4., 5., 6.], 'gap_before': [False] * 6}),
            'split': {'train': ['a', 'b'], 'test': ['test']}}
    sampler = tmp_path / 'bank.npz'
    digest = save_sampler(sampler, bank())
    calibration = tmp_path / 'cal.json'
    calibration.write_text(json.dumps({'status': 'insufficient_calibration', 'coverage': .9, 'physical_group_count': 3}))
    run = {'snapshot_id': 's', 'params': {'history_length': 3, 'horizons_s': [1., 2.], 'forecast_mode': 'joint_residual_paths', 'seed': 42},
           'schema': {'thresholds': {'mode': 'absolute', 'red': 6., 'direction': 'above'}}, 'engine_id': 'gru',
           'scaler': {}, 'artifact': 'model', 'artifacts': {'model': 'digest', 'bank.npz': digest, 'cal.json': 'hashed'},
           'dir': tmp_path, 'funnel': {'mode': 'joint_residual_v1', 'artifact': 'bank.npz', 'calibration': 'cal.json', 'seed': 42, 'path_samples': 300}}
    monkeypatch.setattr(inference, 'load_signal_run', lambda *args: run)
    monkeypatch.setattr(inference, 'load_snapshot', lambda *args: data)
    monkeypatch.setattr(inference, '_load_model', lambda *args: object())
    calls = []
    def predict(m, e, f, s):
        calls.append(f['x'].copy())
        return np.array([[4., 5.]]), None, None
    monkeypatch.setattr(inference, '_predict', predict)
    first = inference.forecast_prefix('p', 'r', 'test', 2., rollout_steps=99)
    assert first['rollout']['status'] == 'direct_only'
    assert first['funnel']['sampled_path_count'] == 300
    assert first['funnel']['returned_path_count'] == 256
    assert len(first['sampled_paths']) == 256
    expected = sample_paths([4., 5.], 3.123456789, bank(), 300, 42)
    expected_lower, _ = empirical_band(expected)
    assert first['funnel']['lower'] == expected_lower.tolist()
    assert len(first['points']) == 2 and len(calls) == 1
    assert np.all(np.array(first['sampled_paths'])[:, 0] == 3.123456789)
    assert first['funnel']['lower'][0] == first['funnel']['upper'][0] == 3.123456789
    assert first['funnel']['calibration_status'] == 'insufficient_calibration'
    supported = {'status': 'calibrated', 'coverage': .9, 'physical_group_count': 10,
                 'expansion': 10., 'guarantee_scope': 'predeclared_earliest_origin_per_physical_group_only'}
    calibration.write_text(json.dumps(supported))
    earliest = inference.forecast_prefix('p', 'r', 'test', 2.)
    assert earliest['calibration_status'] == 'calibrated'
    assert earliest['funnel']['origin_within_calibration_scope']
    assert earliest['funnel']['lower'][1] == expected_lower[1] - 10
    later = inference.forecast_prefix('p', 'r', 'test', 3.)
    assert later['calibration_status'] == 'outside_calibration_origin_scope'
    assert later['funnel']['band_kind'] == 'raw_empirical_pointwise_band'
    assert later['funnel']['lower'][1] == expected_lower[1]
    assert not later['funnel']['origin_within_calibration_scope']
    data['features']['physical_unit_id'] = 'equipment'
    fragment = data['features'].iloc[[0]].copy()
    fragment['unit_id'] = 'other_fragment'
    data['features'] = pd.concat([data['features'], fragment], ignore_index=True)
    ambiguous = inference.forecast_prefix('p', 'r', 'test', 2.)
    assert ambiguous['calibration_status'] == 'outside_calibration_origin_scope'
    assert ambiguous['funnel']['lower'][1] == expected_lower[1]
    data['features'] = data['features'][data['features'].unit_id == 'test'].drop(columns='physical_unit_id')
    calibration.write_text(json.dumps({'status': 'insufficient_calibration', 'coverage': .9, 'physical_group_count': 3}))
    data['features'].loc[3:, 'signal'] = 1e9
    data['features'].loc[3:, 'timestamp_s'] += 1e6
    data['features'].loc[3:, 'gap_before'] = True
    assert inference.forecast_prefix('p', 'r', 'test', 2., rollout_steps=99) == first
    sampler.write_bytes(sampler.read_bytes()+b'tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        inference.forecast_prefix('p', 'r', 'test', 2.)
