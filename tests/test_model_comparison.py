import copy
import inspect
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import torch

from pdm.benchmark import add_survival_scores, compare_evaluations
from pdm.config import load_dataset_config
from pdm.connectome.sources import load_synthetic_fixture
from pdm.forecasting import predict_failure_interval
from pdm.models import FlyConnectomeReservoir, PDMNet, RandomReservoir
from pdm.models.reservoir import LeakyESN
from pdm.predict import Predictor, prepare_history_window
from pdm.preprocessing import Preprocessor, fit_preprocessor
from pdm.splits import bearings_split
from pdm.visualization.simulation import (
    continuous_forecast_history,
    continuous_trace,
    equipment_forecast_frame,
    window_forecast_history,
)


def evaluation(rid, values, *, dataset='bearings', split='validation'):
    frame = pd.DataFrame({'unit_id': ['u1', 'u1', 'u2', 'u2'], 'timestamp_s': [10., 20., 10., 20.],
                          'predicted_rul_s': values, 'actual_rul_s': [20., 10., 20., 10.],
                          'valid_history_reason': ['', '', '', '']})
    return {'run': {'architecture': 'gru', 'smoke': False, 'n_nodes': None},
            'config': {'run_id': rid, 'eval_id': 'evaluation-' + rid, 'dataset_id': dataset,
                       'dataset_version': 'v', 'features_hash': 'f', 'units_hash': 'u', 'split_hash': 's',
                       'quality_policy_hash': 'q', 'metrics_version': 'v2',
                       'evaluate_mask': {'split': split, 'unit_ids': ['u1', 'u2']}},
            'metrics': {}, 'predictions': frame}


def test_common_cohort_ranking_and_missing_prediction():
    a = evaluation('a', [21., 11., 21., 11.])
    b = evaluation('b', [25., 15., 25., 15.])
    result = compare_evaluations([a, b])
    table = result['table'].set_index('run_id')
    assert table.loc['a', 'rank'] == 1
    assert table.loc['a', 'near_30m_mae_s'] == 1
    b['predictions'].loc[0, 'predicted_rul_s'] = np.nan
    table = compare_evaluations([a, b])['table'].set_index('run_id')
    assert table.loc['b', 'prediction_coverage'] == .75
    assert not table.loc['b', 'rank_eligible']
    assert table.loc['a', 'common_points'] == 4


def test_missing_export_row_cannot_hide_a_difficult_point():
    a, b = evaluation('a', [21., 11., 21., 11.]), evaluation('b', [25., 15., 25., 15.])
    b['predictions'] = b['predictions'].iloc[1:]
    table = compare_evaluations([a, b])['table'].set_index('run_id')
    assert table.loc['a', 'common_points'] == 4
    assert table.loc['b', 'prediction_coverage'] == .75
    assert not table.loc['b', 'rank_eligible']


def test_official_test_labels_are_not_observed_prefix_failures():
    ev = evaluation('a', [21., 11., 21., 11.], dataset='filters', split='test')
    ev['predictions']['outcome_observed'] = 0
    ev['metrics']['alerts'] = {'n_units_with_event': 2}
    row = compare_evaluations([ev])['table'].iloc[0]
    assert row.observed_events == 0
    assert row.primary_score == 1


def test_recurrent_identity_ignores_unused_reservoir_defaults(tmp_path):
    import json

    from pdm.experiments import _enrich_run_identity
    (tmp_path / 'experiment_snapshot.json').write_text(json.dumps({'model': {
        'architecture': 'gru', 'hidden_size': 8, 'recurrent_layers': 2,
        'reservoir': {'n_nodes': 1000, 'graph_mode': 'synthetic_fixture'}}}))
    row = _enrich_run_identity({}, tmp_path)
    assert row['n_nodes'] == 16
    assert row['graph_mode'] is None
    assert row['is_synthetic'] is False


@pytest.mark.parametrize('field', ['dataset_version', 'quality_policy_hash', 'metrics_version', 'split_hash'])
def test_incompatible_identities_cannot_be_ranked(field):
    a, b = evaluation('a', [20]*4), evaluation('b', [20]*4)
    b['config'][field] = 'different'
    with pytest.raises(ValueError, match='Incompatible'):
        compare_evaluations([a, b])


def test_ground_truth_and_duplicate_checks():
    a, b = evaluation('a', [20]*4), evaluation('b', [20]*4)
    b['predictions'].loc[0, 'actual_rul_s'] = 999
    with pytest.raises(ValueError, match='Actual RUL differs'):
        compare_evaluations([a, b])
    b = copy.deepcopy(a)
    b['predictions'] = pd.concat([b['predictions'], b['predictions'].iloc[[0]]])
    with pytest.raises(ValueError, match='duplicate'):
        compare_evaluations([b])


def test_censored_likelihood_and_scale_are_comparable():
    pred = pd.DataFrame({'unit_id': ['censored', 'event'], 'timestamp_s': [10., 10.],
                         'weibull_scale_s': [10., 10.], 'weibull_shape': [1., 1.]})
    units = pd.DataFrame({'unit_id': ['censored', 'event'], 'event_observed': [0, 1],
                          'observation_end_s': [20., 20.], 'event_time_s': [np.nan, 20.]})
    scores = add_survival_scores(pred, units)
    assert scores.survival_nll.iloc[0] == pytest.approx(1.)
    assert scores.survival_nll.iloc[1] == pytest.approx(1. + np.log(10.))


def test_full_matrix_has_nine_main_models_and_paired_validation_study():
    from pdm.batch import evaluation_splits, matrix_tasks

    tasks = matrix_tasks(['bearings', 'filters'])
    assert len(tasks) == 14
    main = [t for t in tasks if t['role'] == 'main']
    assert len(main) == 9
    assert all(evaluation_splits(t) == ('validation', 'test') for t in main)
    paired = [t for t in tasks if t['dataset_id'] == 'filters' and t['architecture'] == 'gru']
    assert {(t['seed'], t['events_only']) for t in paired} == {(seed, events) for seed in (42,43,44) for events in (False,True)}
    assert all(evaluation_splits(t) == ('validation',) for t in paired if t['role'] == 'ablation')


@pytest.mark.parametrize('architecture', ['gru', 'lstm'])
@pytest.mark.parametrize('head', ['rul', 'weibull'])
def test_actual_recurrent_trace_matches_prediction_and_cell_memory(tiny_bearing_tables, architecture, head):
    f, u = tiny_bearing_tables
    cfg = load_dataset_config('bearings')
    prep, _ = fit_preprocessor('bearings', f, u, bearings_split(u), cfg)
    model = PDMNet(len(prep.feature_names), hidden_size=8, num_layers=2, architecture=architecture,
                   head=head, time_scale_s=prep.time_scale_s)
    predictor = Predictor(model, prep, 5)
    prefix = f[f.unit_id == 'Bearing1_1'].iloc[:12]
    plain = predictor.predict_from_history(prefix)
    trace = predictor.predict_from_history(prefix, with_trace=True)
    assert trace['predicted_rul_s'] == plain['predicted_rul_s']
    assert trace['states'].shape == (5, 16)
    window = prepare_history_window(prefix, prep, 5)
    with torch.no_grad():
        _, hidden = model.encoder.rnn(torch.from_numpy(np.array(window['inputs'], order='C', copy=True)).unsqueeze(0))
    if architecture == 'lstm':
        np.testing.assert_allclose(trace['cell_states'][-1], hidden[1][:, 0].reshape(-1).numpy(), atol=1e-6)
        hidden = hidden[0]
    np.testing.assert_allclose(trace['states'][-1], hidden[:, 0].reshape(-1).numpy(), atol=1e-6)
    if head == 'weibull':
        assert plain['lower_rul_s'] <= plain['predicted_rul_s'] <= plain['upper_rul_s']


def _window_reset_parity_model(architecture, prep):
    width = len(prep.feature_names)
    if architecture in ('gru', 'lstm'):
        return PDMNet(width, hidden_size=8, architecture=architecture, time_scale_s=prep.time_scale_s)
    graph = load_synthetic_fixture().graph
    reservoir = FlyConnectomeReservoir if architecture == 'fly_connectome_reservoir' else RandomReservoir
    return reservoir(graph, input_size=width, state_mode='window_reset', time_scale_s=prep.time_scale_s)


@pytest.mark.parametrize('architecture', ['gru', 'lstm', 'fly_connectome_reservoir', 'random_reservoir'])
def test_seek_and_replay_export_use_identical_forecasts(tiny_bearing_tables, architecture):
    from pdm.replay import replay_unit
    from pdm.visualization.simulation import window_forecast_history
    f, u = tiny_bearing_tables
    cfg = load_dataset_config('bearings')
    prep, _ = fit_preprocessor('bearings', f, u, bearings_split(u), cfg)
    model = _window_reset_parity_model(architecture, prep)
    assert model.state_mode == 'window_reset'
    prefix = f[f.unit_id == 'Bearing1_1'].iloc[:16]
    points = window_forecast_history(prefix, model, prep, 5)
    cache = {r['timestamp_s']: r for r in points.to_dict('records')}
    rewind = window_forecast_history(prefix.iloc[:9], model, prep, 5, cached=cache)
    replay = window_forecast_history(
        prefix, model, prep, 5, cached={r['timestamp_s']: r for r in rewind.to_dict('records')},
    )
    overlap = points['timestamp_s'].isin(rewind['timestamp_s'])
    pd.testing.assert_frame_equal(points.loc[overlap].reset_index(drop=True), rewind.reset_index(drop=True))
    pd.testing.assert_frame_equal(points, replay)
    export = replay_unit(prefix, Predictor(model, prep, 5), dataset_id='bearings', unit_id='Bearing1_1', run_id='test',
                         history_length=5, warning_horizon_s=60, truth_units=u)['predictions']
    np.testing.assert_allclose(points['predicted_rul_s'], export['predicted_rul_s'], equal_nan=True, rtol=0, atol=0)
    held = list(cache)[3:8]
    partial = {stamp: cache[stamp] for stamp in held}
    partial_rewind = window_forecast_history(prefix.iloc[:9], model, prep, 5, cached=partial)
    cached_rows = points['timestamp_s'].isin(held)
    got = partial_rewind.loc[partial_rewind['timestamp_s'].isin(held)]
    pd.testing.assert_frame_equal(points.loc[cached_rows].reset_index(drop=True), got.reset_index(drop=True))


def _reject_overlay_and_visited(fn):
    names = [name.lower() for name in inspect.signature(fn).parameters]
    assert 'show_gt' not in names
    assert 'official_rul_at_prefix_end_s' not in names
    assert 'predictions' not in names
    assert not any('visit' in name for name in names)


def _continuous_prefix(n=16):
    frame = pd.DataFrame({
        'unit_id': 'bearing', 'timestamp_s': np.arange(n) * 60.0,
        'operating_age_s': np.arange(n) * 60.0, 'rpm': 1800, 'load_kn': 4,
    })
    for axis in ('horizontal', 'vertical'):
        for stat in ('rms', 'std', 'abs_peak', 'peak_to_peak', 'crest_factor', 'kurtosis',
                     'band_0', 'band_1', 'band_2', 'band_3'):
            frame[f'{axis}_{stat}'] = np.linspace(0.1, 2.0, n)
    frame['gap_before'] = False
    return frame


def _continuous_model():
    model = LeakyESN(
        torch.tensor([[.3], [-.2], [.1]]), torch.eye(3) * .25, torch.zeros(3),
        state_mode='continuous', time_scale_s=600,
    )
    model.node_order = ['a', 'b', 'c']
    model.rul_transform = 'log1p'
    model.readout.load_ridge_vector(np.array([.2, -.3, .4, -.1, 4.0]))
    prep = Preprocessor(
        ['horizontal_rms'], [], [0], [1], 600,
        fill_values={'horizontal_rms': 0}, dataset_id='bearings',
    )
    return model, prep


def test_equipment_forecast_frame_choice(monkeypatch, tiny_bearing_tables):
    _reject_overlay_and_visited(equipment_forecast_frame)
    _reject_overlay_and_visited(continuous_forecast_history)
    signature = inspect.signature(equipment_forecast_frame)
    assert list(signature.parameters) == [
        'prefix', 'model', 'prep', 'history_length', 'profile', 'trace', 'window_cache', 'segment_cache',
    ]
    for name in ('profile', 'trace', 'window_cache', 'segment_cache'):
        assert signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY

    history_length = 3
    model, prep = _continuous_model()
    prefix = _continuous_prefix()
    prefix.loc[10, 'gap_before'] = True
    assert len(prefix) == 16
    visited = {
        float(prefix['timestamp_s'].iloc[0]): {
            'timestamp_s': float(prefix['timestamp_s'].iloc[0]), 'predicted_rul_s': 9.0,
        },
    }
    segment_cache = []
    with monkeypatch.context() as guarded:
        guarded.setattr(
            'pdm.visualization.simulation.window_forecast_history',
            Mock(side_effect=AssertionError('window forecast used')),
        )
        spy = Mock(wraps=continuous_forecast_history)
        guarded.setattr('pdm.visualization.simulation.continuous_forecast_history', spy)
        early = prefix.iloc[:2]
        early_trace = continuous_trace(early, model, prep, history_length)
        equipment_forecast_frame(
            early, model, prep, history_length, profile=None, trace=early_trace,
            window_cache=visited, segment_cache=segment_cache,
        )
        trace = continuous_trace(prefix, model, prep, history_length)
        prior = copy.deepcopy(segment_cache)
        result = equipment_forecast_frame(
            prefix, model, prep, history_length, profile=None, trace=trace,
            window_cache=visited, segment_cache=segment_cache,
        )
        assert [len(call.args[0]) for call in spy.call_args_list] == [2, 16]
        assert spy.call_args.kwargs['trace'] is trace
        assert spy.call_args.kwargs['cached_segments'] is segment_cache
        expected = continuous_forecast_history(
            prefix, model, prep, history_length, trace=trace, cached_segments=prior,
        )
        pd.testing.assert_frame_equal(result, expected)
        assert len(result) == len(prefix) == 16
        assert len(result) != 1
        assert trace['status'] == 'predicted'
        assert float(result.iloc[-1]['predicted_rul_s']) == float(trace['predicted_rul_s'])
        assert segment_cache
        short = prefix.iloc[:9]
        short_trace = continuous_trace(short, model, prep, history_length)
        rewound = equipment_forecast_frame(
            short, model, prep, history_length, profile=None, trace=short_trace,
            window_cache=visited, segment_cache=segment_cache,
        )
        assert len(rewound) == len(short)
        short_end = float(short['timestamp_s'].iloc[-1])
        assert np.all(rewound['timestamp_s'].to_numpy(dtype=float) <= short_end)
        warm = prefix.copy()
        warm['gap_before'] = False
        warm_gap = 14
        warm.loc[warm_gap, 'gap_before'] = True
        warm_trace = continuous_trace(warm, model, prep, history_length)
        warm_result = equipment_forecast_frame(
            warm, model, prep, history_length, profile=None, trace=warm_trace,
            window_cache=visited, segment_cache=segment_cache,
        )
    scalar = warm_trace['predicted_rul_s']
    last = warm_result.iloc[-1]['predicted_rul_s']
    previous = warm_result['predicted_rul_s'].iloc[warm_gap - 1]
    assert warm_trace['status'] != 'predicted'
    assert scalar is None or pd.isna(scalar)
    assert pd.isna(last)
    assert np.isfinite(previous)
    assert not np.isfinite(last)

    features, units = tiny_bearing_tables
    cfg = load_dataset_config('bearings')
    fitted, _ = fit_preprocessor('bearings', features, units, bearings_split(units), cfg)
    measured = features[features.unit_id == 'Bearing1_1'].iloc[:16].reset_index(drop=True)
    graph = load_synthetic_fixture().graph
    decoy = {
        'timestamps_s': np.zeros(1), 'raw_rul_s': np.zeros(1),
        'predicted_rul_s': None, 'status': 'Collecting history',
    }
    for cls in (FlyConnectomeReservoir, RandomReservoir):
        reservoir = cls(
            graph, input_size=len(fitted.feature_names), state_mode='window_reset',
            time_scale_s=fitted.time_scale_s,
        )
        assert reservoir.state_mode == 'window_reset'
        window_cache = {}
        with monkeypatch.context() as guarded:
            guarded.setattr(
                'pdm.visualization.simulation.continuous_forecast_history',
                Mock(side_effect=AssertionError('continuous forecast used')),
            )
            window_spy = Mock(wraps=window_forecast_history)
            guarded.setattr('pdm.visualization.simulation.window_forecast_history', window_spy)
            got = equipment_forecast_frame(
                measured, reservoir, fitted, 5, profile=None, trace=decoy, window_cache=window_cache,
            )
        assert window_spy.call_count == 1
        assert window_spy.call_args.args[0] is measured
        assert window_spy.call_args.kwargs['cached'] is window_cache
        pd.testing.assert_frame_equal(
            got, window_forecast_history(measured, reservoir, fitted, 5, cached=window_cache),
        )

    profile = {
        'version': 1, 'warmup_measurements': history_length, 'smoothing_tau_s': 300,
        'residual_q_low': -1.0, 'residual_q_high': 2.0,
    }
    seen = {}

    def _spy_interval(timestamps_s, raw_rul_s, profile_arg, *, gap_before=None):
        seen['timestamps'] = timestamps_s
        seen['raw'] = raw_rul_s
        seen['profile'] = profile_arg
        seen['gap_before'] = gap_before
        return predict_failure_interval(timestamps_s, raw_rul_s, profile_arg, gap_before=gap_before)

    with monkeypatch.context() as guarded:
        guarded.setattr(
            'pdm.visualization.simulation.window_forecast_history',
            Mock(side_effect=AssertionError('window forecast used')),
        )
        guarded.setattr(
            'pdm.visualization.simulation.continuous_forecast_history',
            Mock(side_effect=AssertionError('continuous forecast used')),
        )
        guarded.setattr('pdm.forecasting.predict_failure_interval', _spy_interval)
        profiled = equipment_forecast_frame(
            prefix, model, prep, history_length, profile=profile, trace=trace, segment_cache=[],
        )
    assert seen['timestamps'] is trace['timestamps_s']
    assert seen['raw'] is trace['raw_rul_s']
    assert seen['profile'] is profile
    assert seen['gap_before'] is None
    expected_profile = predict_failure_interval(trace['timestamps_s'], trace['raw_rul_s'], profile)
    pd.testing.assert_frame_equal(profiled, expected_profile)
    assert list(profiled.columns) == list(expected_profile.columns)
    assert len(profiled) == len(np.asarray(trace['timestamps_s']))
    assert len(profiled) < len(prefix)
    assert 'horizontal_rms' not in profiled.columns
    assert 'gap_before' not in profiled.columns
